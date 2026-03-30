# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quest Forward Hooks — Pipelined page-level prefetch with exact Q projection.

After layer L's forward pass completes, the hook:
  1. Extracts the post-attention residual stream h_L.
  2. Computes LayerNorm_{L+1}(h_L) using layer L+1's actual RMSNorm weights.
  3. Projects through layer L+1's actual Q weight matrix → exact Q_{L+1}.
  4. Scores per-page metadata via QuestPageSelector.select_pages().
  5. Triggers an async CPU→GPU prefetch of the selected pages.

Unlike InfiniGen's rehearsal which uses partial weights (approximate),
Quest computes the **exact** Q projection for the next layer.  The cost
is one RMSNorm + one matmul per layer, which is negligible compared to
attention over thousands of tokens.

Usage:
    hook_manager = QuestHookManager(
        page_selector=selector,
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
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


class QuestHookManager:
    """Manages forward hooks on decoder layers for Quest page prefetch.

    Attributes:
        page_selector: The QuestPageSelector instance.
        budget_computer: The DynamicBudgetComputer instance (from
            ``infinigen_budget.py``, reused).
        num_layers: Number of decoder layers.
        kv_connector: Reference to the OffloadingConnector.
        stats: Optional PrefetchStats for timing (reused).
    """

    def __init__(
        self,
        page_selector: Any,  # QuestPageSelector
        budget_computer: Any,  # DynamicBudgetComputer
        num_layers: int,
        kv_connector: Any | None = None,
        stats: Any | None = None,  # PrefetchStats
        cpu_scoring: bool = True,
    ):
        self.page_selector = page_selector
        self.budget_computer = budget_computer
        self.num_layers = num_layers
        self.kv_connector = kv_connector
        self.stats = stats
        self.cpu_scoring = cpu_scoring
        self.single_layer_scoring = False  # set after __init__ if needed

        # Registered hook handles (for cleanup)
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []

        # Cached layer weights for exact Q computation
        # layer_idx -> (layernorm_weight, eps)
        self._layernorm_weights: dict[int, tuple[torch.Tensor, float]] = {}
        # layer_idx -> Q weight matrix [q_size, hidden_size]
        self._q_proj_weights: dict[int, torch.Tensor] = {}

        # Per-step state: layer_idx -> selected page indices
        self._current_selections: dict[int, torch.Tensor] = {}

        self._active = False
        self._num_heads: int | None = None
        self._head_dim: int | None = None

    def register_hooks(
        self,
        model: nn.Module,
        decoder_layer_cls: type | None = None,
    ) -> int:
        """Register forward hooks on all decoder layers.

        Also extracts and caches LayerNorm and Q projection weights
        for exact Q computation.

        Args:
            model: The full model (e.g. LlamaForCausalLM).
            decoder_layer_cls: The decoder layer class to hook.

        Returns:
            Number of hooks registered.
        """
        if self._active:
            logger.warning("Quest hooks already registered, skipping")
            return 0

        if decoder_layer_cls is None:
            decoder_layer_cls = self._detect_decoder_layer_cls(model)

        if decoder_layer_cls is None:
            logger.warning(
                "Could not auto-detect decoder layer class. "
                "Quest hooks not registered."
            )
            return 0

        # Extract weights before registering hooks
        self._extract_layer_weights(model)

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, decoder_layer_cls):
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
            "Registered %d Quest forward hooks on %s layers "
            "(Q weights cached for %d layers)",
            count,
            decoder_layer_cls.__name__,
            len(self._q_proj_weights),
        )
        return count

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._active = False
        self._current_selections.clear()

    def get_selection(self, layer_idx: int) -> torch.Tensor | None:
        """Get the computed page selection for a layer (if available)."""
        return self._current_selections.get(layer_idx)

    def clear_selections(self) -> None:
        """Clear all computed selections (called at end of each step)."""
        self._current_selections.clear()

    # -- Weight extraction ---------------------------------------------------

    def _extract_layer_weights(self, model: nn.Module) -> None:
        """Walk the model to extract LayerNorm and Q projection weights.

        For each layer i, caches:
        - self._layernorm_weights[i] = (weight, eps)
        - self._q_proj_weights[i] = W_Q  [q_size, hidden_size]
        """
        layers_module = None
        for name, mod in model.named_modules():
            # Find the layers container (e.g., model.model.layers)
            if name.endswith(".layers") or name == "layers":
                layers_module = mod
                break

        if layers_module is None:
            logger.warning(
                "Could not find 'layers' module in model. "
                "Quest exact Q computation will be disabled."
            )
            return

        for layer_idx, layer in enumerate(layers_module):
            # Extract input_layernorm
            ln = getattr(layer, "input_layernorm", None)
            if ln is not None:
                weight = getattr(ln, "weight", None)
                eps = getattr(ln, "variance_epsilon", None)
                if eps is None:
                    eps = getattr(ln, "eps", 1e-6)
                if weight is not None:
                    self._layernorm_weights[layer_idx] = (
                        weight.data,
                        float(eps),
                    )

            # Extract Q projection from fused QKV
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue

            qkv_proj = getattr(self_attn, "qkv_proj", None)
            if qkv_proj is None:
                continue

            qkv_weight = getattr(qkv_proj, "weight", None)
            if qkv_weight is None:
                continue

            # Determine Q size from attention config
            num_heads = getattr(self_attn, "num_heads", None)
            head_dim = getattr(self_attn, "head_dim", None)

            if num_heads is None or head_dim is None:
                continue

            if self._num_heads is None:
                self._num_heads = num_heads
                self._head_dim = head_dim

            q_size = num_heads * head_dim
            # Q portion is the first q_size rows of the fused QKV weight
            self._q_proj_weights[layer_idx] = (
                qkv_weight.data[:q_size, :]
            )

        logger.info(
            "Extracted Quest Q weights for %d/%d layers "
            "(num_heads=%s, head_dim=%s)",
            len(self._q_proj_weights),
            self.num_layers,
            self._num_heads,
            self._head_dim,
        )

    # -- Core hook logic -----------------------------------------------------

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

        Computes exact Q_{L+1} from the residual stream and selects
        pages for the next layer.
        """
        if layer_idx >= self.num_layers - 1:
            return

        next_layer_idx = layer_idx + 1

        # Can't compute Q without weights
        if next_layer_idx not in self._q_proj_weights:
            return
        if next_layer_idx not in self._layernorm_weights:
            return

        # Extract residual from output
        if isinstance(output, tuple) and len(output) >= 2:
            residual = output[1]
        elif isinstance(output, torch.Tensor):
            residual = output
        else:
            return

        if residual is None:
            return

        # Compute budget for this layer
        budget = self.budget_computer.compute_budget(
            approximate_scores=torch.empty(0),
            layer_idx=next_layer_idx,
            num_layers=self.num_layers,
        )

        if budget >= 1.0:
            # Baseline pipelined path: transfer ALL pages for the next
            # layer without computing Q or scoring.  This gives the
            # baseline full compute/IO overlap (layer-pipelined) without
            # paying for the Q projection or CPU scoring overhead.
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_all_pages=True,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
            return

        if self.single_layer_scoring and layer_idx > 0:
            # Single-layer scoring baseline: layer 0 computed Q and scored
            # pages; all subsequent layers reuse that selection.  The connector
            # looks up its cached _quest_last_selection and immediately starts
            # the transfer — no Q computation or scoring in the critical path.
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_reuse_selection=True,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
            return

        # Compute exact Q for the next layer (timed for stats).
        if self.stats is not None:
            self.stats.begin_q_compute(next_layer_idx)

        query = self._compute_exact_q(residual, next_layer_idx)

        if self.stats is not None:
            self.stats.end_q_compute(next_layer_idx)

        if query is None:
            return

        if self.cpu_scoring:
            # CPU-side scoring: send the query vector to the connector.
            # The connector will copy Q to CPU, score against CPU-resident
            # page metadata, select pages, and issue a selective transfer.
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_query=query,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
        else:
            # GPU-side scoring: score and select pages on GPU, then send
            # the selected page indices to the connector for transfer.
            selected = self.page_selector.select_pages(
                query=query,
                layer_idx=next_layer_idx,
                budget=budget,
                stats=self.stats,
            )

            self._current_selections[next_layer_idx] = selected

            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        hidden_states=residual,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_selected_pages=selected,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass

    @torch.no_grad()
    def _compute_exact_q(
        self,
        residual: torch.Tensor,
        next_layer_idx: int,
    ) -> torch.Tensor | None:
        """Compute exact Q_{L+1} from the residual stream.

        Steps:
        1. Apply RMSNorm using layer L+1's layernorm weights.
        2. Project through layer L+1's Q weight matrix.
        3. Reshape to [B, num_heads, head_dim].

        Args:
            residual: Post-layer residual, shape [B, hidden_size] or
                [num_tokens, hidden_size].
            next_layer_idx: Index of the next layer.

        Returns:
            Q tensor of shape [B, num_heads, head_dim], or None on error.
        """
        ln_weight, eps = self._layernorm_weights[next_layer_idx]
        q_weight = self._q_proj_weights[next_layer_idx]

        device = residual.device
        ln_w = ln_weight.to(device)
        q_w = q_weight.to(device)

        # RMSNorm: x * (1/rms) * weight
        # rms = sqrt(mean(x^2) + eps)
        variance = residual.to(torch.float32).pow(2).mean(-1, keepdim=True)
        h_normed = residual * torch.rsqrt(variance + eps)
        h_normed = (h_normed * ln_w).to(residual.dtype)

        # Q projection: [B, hidden_size] @ [q_size, hidden_size]^T
        # -> [B, q_size]
        q = h_normed @ q_w.t()

        # Reshape to [B, num_heads, head_dim]
        if self._num_heads is not None and self._head_dim is not None:
            q = q.view(-1, self._num_heads, self._head_dim)

        return q

    # -- Static helpers ------------------------------------------------------

    @staticmethod
    def _detect_decoder_layer_cls(model: nn.Module) -> type | None:
        """Auto-detect the decoder layer class."""
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if "DecoderLayer" in cls_name:
                return type(module)
        return None

    @staticmethod
    def _extract_layer_idx(module_name: str) -> int | None:
        """Extract layer index from module name like 'model.layers.5'."""
        parts = module_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None
