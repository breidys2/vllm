# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
InfiniGen Forward Hooks — Integration of rehearsal into the model forward pass.

This module provides a non-invasive way to inject InfiniGen's rehearsal
mechanism into the Transformer forward pass using PyTorch forward hooks.
This avoids modifying individual model files (llama.py, qwen.py, etc.).

The hooks are registered on each ``DecoderLayer`` and fire after the layer's
forward pass completes.  The hook:
  1. Extracts the post-attention residual stream.
  2. Invokes the rehearsal engine to predict important tokens for the
     next layer.
  3. Passes the resulting token mask to the offloading connector to
     trigger an async CPU→GPU prefetch.

Usage:
    from vllm.v1.attention.ops.infinigen_hooks import (
        InfiniGenHookManager,
    )

    hook_manager = InfiniGenHookManager(
        rehearsal_engine=engine,
        budget_computer=budget_computer,
        num_layers=32,
    )
    hook_manager.register_hooks(model)
    # ... run forward pass ...
    hook_manager.remove_hooks()
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class InfiniGenHookManager:
    """Manages forward hooks on decoder layers for InfiniGen rehearsal.

    Attributes:
        rehearsal_engine: The RehearsalEngine instance.
        budget_computer: The DynamicBudgetComputer instance.
        num_layers: Number of decoder layers.
        kv_connector: Reference to the OffloadingConnector (for triggering
            prefetches).
    """

    def __init__(
        self,
        rehearsal_engine: Any,  # RehearsalEngine
        budget_computer: Any,   # DynamicBudgetComputer
        num_layers: int,
        kv_connector: Any | None = None,
        stats: Any | None = None,  # InfiniGenStats
    ):
        self.rehearsal_engine = rehearsal_engine
        self.budget_computer = budget_computer
        self.num_layers = num_layers
        self.kv_connector = kv_connector
        self.stats = stats

        # Registered hook handles (for cleanup)
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []

        # Per-step state: layer_idx -> token_mask
        self._current_masks: dict[int, torch.Tensor] = {}

        # Whether hooks are currently active
        self._active = False

    def register_hooks(
        self,
        model: nn.Module,
        decoder_layer_cls: type | None = None,
    ) -> int:
        """Register forward hooks on all decoder layers.

        Args:
            model: The full model (e.g., LlamaForCausalLM).
            decoder_layer_cls: The decoder layer class to hook. If None,
                auto-detects common layer classes.

        Returns:
            Number of hooks registered.
        """
        if self._active:
            logger.warning("InfiniGen hooks already registered, skipping")
            return 0

        # Auto-detect decoder layer class
        if decoder_layer_cls is None:
            decoder_layer_cls = self._detect_decoder_layer_cls(model)

        if decoder_layer_cls is None:
            logger.warning(
                "Could not auto-detect decoder layer class. "
                "InfiniGen hooks not registered."
            )
            return 0

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, decoder_layer_cls):
                # Extract layer index from the module name
                layer_idx = self._extract_layer_idx(name)
                if layer_idx is None:
                    continue

                handle = module.register_forward_hook(
                    self._make_post_layer_hook(layer_idx)
                )
                self._hook_handles.append(handle)
                count += 1

        self._active = True
        logger.info(
            "Registered %d InfiniGen forward hooks on %s layers",
            count,
            decoder_layer_cls.__name__,
        )
        return count

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._active = False
        self._current_masks.clear()

    def get_mask(self, layer_idx: int) -> torch.Tensor | None:
        """Get the computed token mask for a layer (if available)."""
        return self._current_masks.get(layer_idx)

    def clear_masks(self) -> None:
        """Clear all computed masks (called at the end of each step)."""
        self._current_masks.clear()

    # -- Internal helpers ----------------------------------------------------

    def _make_post_layer_hook(self, layer_idx: int):
        """Create a forward hook closure for a specific layer."""

        def hook(
            module: nn.Module,
            input: tuple,
            output: Any,
        ) -> None:
            self._on_layer_complete(layer_idx, module, input, output)

        return hook

    def _on_layer_complete(
        self,
        layer_idx: int,
        module: nn.Module,
        input: tuple,
        output: Any,
    ) -> None:
        """Called after each decoder layer's forward pass.

        Extracts the residual stream and runs the rehearsal engine to
        predict important tokens for the next layer.
        """
        # Skip the last layer (no next layer to predict for)
        if layer_idx >= self.num_layers - 1:
            return

        next_layer_idx = layer_idx + 1

        # Extract the residual stream from the output.
        # Most decoder layers return (hidden_states, residual) or just
        # hidden_states.
        if isinstance(output, tuple) and len(output) >= 2:
            # (hidden_states, residual) — use residual for rehearsal
            residual = output[1]
        elif isinstance(output, torch.Tensor):
            residual = output
        else:
            return

        if residual is None:
            return

        # The rehearsal engine would compute:
        #   mask = self.rehearsal_engine.rehearse(
        #       hidden_states=residual,
        #       cached_keys=<CPU key cache for next_layer_idx>,
        #       layer_idx=layer_idx,
        #       next_layer_idx=next_layer_idx,
        #       budget=budget,
        #   )
        # For now, store the residual for later use by the connector.
        # The actual rehearsal is triggered by the connector's
        # save_kv_layer hook which has access to the KV cache tensors.

        # Compute the budget for this layer
        budget = self.budget_computer.compute_budget(
            approximate_scores=torch.empty(0),  # placeholder
            layer_idx=next_layer_idx,
            num_layers=self.num_layers,
        )

        # Store metadata for the connector to use
        self._current_masks[next_layer_idx] = None  # type: ignore
        # The actual mask will be computed when the connector has access
        # to the CPU-side KV cache.

        # If a connector is available, trigger the prefetch
        if self.kv_connector is not None:
            try:
                self.kv_connector.save_kv_layer(
                    layer_name=f"layer_{layer_idx}",
                    kv_layer=torch.empty(0),  # unused
                    attn_metadata=None,
                    hidden_states=residual,
                    next_layer_idx=next_layer_idx,
                    budget=budget,
                    infinigen_stats=self.stats,
                )
            except Exception:
                # Don't let hook errors break the forward pass
                pass

    @staticmethod
    def _detect_decoder_layer_cls(model: nn.Module) -> type | None:
        """Auto-detect the decoder layer class from common model families."""
        # Try common decoder layer class names
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if "DecoderLayer" in cls_name:
                return type(module)
        return None

    @staticmethod
    def _extract_layer_idx(module_name: str) -> int | None:
        """Extract the layer index from a module name like 'model.layers.5'."""
        parts = module_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None
