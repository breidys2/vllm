# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-decoder-layer CUDA event timing hooks for custom profiling.

Produces JSONL output readable without Nsight Systems.  Each record is one
decoder layer from one forward pass:

    {"iteration": 0, "seq_len": 4096, "num_computed_tokens": 4096,
     "num_cached_tokens": 0, "kv_cache_bytes": 0, "layer_idx": 3,
     "layer_name": "model.layers.3", "elapsed_ms": 4.23, "rank": 0}

seq_len is the full context length (computed + cached).  When prefix
caching is active, num_computed_tokens < seq_len.  kv_cache_bytes is the
byte size of the KV cache fetched from the prefix cache for this layer
(num_cached_tokens * kv_bytes_per_token_per_layer).

Usage in gpu_model_runner.py:

    # Registration (once, after model load)
    from vllm.utils.cuda_event_hooks import LayerwiseCUDAEventHooks
    self.cuda_event_hooks = LayerwiseCUDAEventHooks(output_prefix)
    self.cuda_event_hooks.register_hooks(self.model)

    # Flush (after every model forward)
    self.cuda_event_hooks.flush(num_tokens=input_ids.shape[0],
                                num_cached_tokens=num_cached)
"""

import json
import re

import torch
import torch.nn as nn


def _count_tensor_bytes(obj) -> int:
    """Recursively sum bytes of all floating-point tensors in a nested structure.

    Handles Tensor, list, tuple, and dict.  Any other type (e.g. custom
    attention-metadata objects) contributes 0 bytes.
    """
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size() if obj.is_floating_point() else 0
    if isinstance(obj, (list, tuple)):
        return sum(_count_tensor_bytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_count_tensor_bytes(v) for v in obj.values())
    return 0


class LayerwiseCUDAEventHooks:
    """Records per-decoder-layer GPU wall time using CUDA events.

    Only hooks on the top-level decoder layer modules (e.g.
    ``model.layers.0``), not their sub-modules, so the measured interval
    covers the complete layer including attention, MLP, and norms.

    Events are accumulated during the forward pass without any CPU/GPU
    synchronisation.  Call ``flush()`` once after the forward pass to
    synchronise a single time, compute elapsed times, and append to the
    output JSONL file.
    """

    # Matches names like "Qwen3MoeForCausalLM.model.layers.4" but not
    # "...layers.4.self_attn" — the layer index must be the final component.
    _LAYER_NAME_RE = re.compile(r"\.layers\.(\d+)$")

    def __init__(self, output_prefix: str, rank: int = 0) -> None:
        """
        Args:
            output_prefix: Path prefix for the output file.  The actual file
                will be ``{output_prefix}_rank{rank}.jsonl``.
            rank: TP/DP rank of this worker, embedded in the filename and
                every output record.
        """
        self._output_path = f"{output_prefix}_rank{rank}.jsonl"
        self._rank = rank

        # Truncate / create the file fresh so old runs don't bleed in.
        open(self._output_path, "w").close()

        # Map from nn.Module → (layer_name, layer_idx) for registered modules.
        self._module_info: dict[nn.Module, tuple[str, int]] = {}

        # Map from nn.Module → total parameter bytes (computed once at registration).
        self._module_param_bytes: dict[nn.Module, int] = {}

        # Map from nn.Module → KV cache bytes per token for this layer.
        # Computed once at registration from the Attention sub-module.
        self._module_kv_bytes_per_token: dict[nn.Module, int] = {}

        # LIFO stack of (layer_name, layer_idx, start_event, param_bytes,
        #                 input_bytes, kv_bytes_per_token)
        # accumulated during the current forward pass.  A stack rather than a flat list
        # handles re-entrant / nested calls safely.
        self._start_stack: list[
            tuple[str, int, torch.cuda.Event, int, int, int]
        ] = []

        # Completed (layer_name, layer_idx, start_event, end_event,
        #             param_bytes, input_bytes, output_bytes, kv_bytes_per_token)
        # tuples waiting to be flushed.
        self._pending: list[
            tuple[str, int, torch.cuda.Event, torch.cuda.Event, int, int, int, int]
        ] = []

        self._iteration: int = 0

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _pre_hook(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        layer_name, layer_idx = self._module_info[module]
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        param_bytes = self._module_param_bytes[module]
        input_bytes = _count_tensor_bytes(args) + _count_tensor_bytes(kwargs)
        kv_bpt = self._module_kv_bytes_per_token[module]
        self._start_stack.append(
            (layer_name, layer_idx, start, param_bytes, input_bytes, kv_bpt))

    def _post_hook(
        self,
        module: nn.Module,
        args: tuple,
        output: object,
    ) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        if self._start_stack:
            layer_name, layer_idx, start, param_bytes, input_bytes, kv_bpt = (
                self._start_stack.pop())
            output_bytes = _count_tensor_bytes(output)
            self._pending.append(
                (layer_name, layer_idx, start, end,
                 param_bytes, input_bytes, output_bytes, kv_bpt))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @staticmethod
    def _kv_bytes_per_token(decoder_layer: nn.Module) -> int:
        """Extract KV cache bytes-per-token from a decoder layer.

        Searches the layer's sub-modules for a vLLM ``Attention`` instance
        which exposes ``num_kv_heads``, ``head_size``, and
        ``kv_cache_torch_dtype``.  Returns
        ``num_kv_heads * head_size * 2 (K+V) * element_size``, or 0 if
        the attention module cannot be found.
        """
        for sub in decoder_layer.modules():
            if (hasattr(sub, "num_kv_heads")
                    and hasattr(sub, "head_size")
                    and hasattr(sub, "kv_cache_torch_dtype")):
                elem = torch.tensor([], dtype=sub.kv_cache_torch_dtype).element_size()
                return sub.num_kv_heads * sub.head_size * 2 * elem
        return 0

    def register_hooks(self, model: nn.Module) -> None:
        """Traverse the model and register hooks on decoder layer modules.

        Decoder layers are identified by the ``_LAYER_NAME_RE`` pattern,
        which matches the top-level layer container (e.g. ``model.layers.3``)
        but not its children (e.g. ``model.layers.3.self_attn``).
        """
        for name, module in model.named_modules():
            m = self._LAYER_NAME_RE.search(name)
            if m is None:
                continue
            layer_idx = int(m.group(1))
            self._module_info[module] = (name, layer_idx)
            self._module_param_bytes[module] = sum(
                p.numel() * p.element_size() for p in module.parameters()
            )
            self._module_kv_bytes_per_token[module] = self._kv_bytes_per_token(module)
            module.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            module.register_forward_hook(self._post_hook)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(
        self,
        num_tokens: int | None = None,
        num_cached_tokens: int | None = None,
    ) -> list[dict]:
        """Synchronise pending events and write timing records to JSONL.

        Call this once after every model forward pass.  A single
        ``event.synchronize()`` call is issued on the last end-event so the
        GPU runs unimpeded during the forward pass.

        Args:
            num_tokens: Number of tokens actually computed in this forward
                pass (i.e. ``input_ids.shape[0]``).
            num_cached_tokens: Number of tokens whose KV was served from the
                prefix cache (0 when prefix caching is off).

        Returns:
            The list of dicts written during this call, in layer order.
        """
        if not self._pending:
            return []

        # Single synchronisation point — wait only until the last end marker.
        self._pending[-1][3].synchronize()

        cached = num_cached_tokens or 0
        records: list[dict] = []
        for (layer_name, layer_idx, start, end,
             param_bytes, input_bytes, output_bytes, kv_bpt) in self._pending:
            record: dict = {
                "iteration": self._iteration,
                "rank": self._rank,
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "elapsed_ms": start.elapsed_time(end),
                "param_bytes": param_bytes,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
            }
            if num_tokens is not None:
                record["seq_len"] = num_tokens + cached
                record["num_computed_tokens"] = num_tokens
                record["num_cached_tokens"] = cached
                record["kv_cache_bytes"] = cached * kv_bpt
            records.append(record)

        # Sort by layer index so records are in forward-pass order even if
        # hooks fired out of order (unlikely but defensive).
        records.sort(key=lambda r: r["layer_idx"])

        with open(self._output_path, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        self._pending.clear()
        self._iteration += 1
        return records