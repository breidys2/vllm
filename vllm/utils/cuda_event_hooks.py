# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-decoder-layer CUDA event timing hooks for custom profiling.

Produces JSONL output readable without Nsight Systems.  Each record is one
decoder layer from one forward pass:

    {"iteration": 0, "seq_len": 4096, "num_computed_tokens": 4096,
     "num_cached_tokens": 0, "kv_cache_bytes": 0, "layer_idx": 3,
     "layer_name": "model.layers.3", "elapsed_ms": 4.23,
     "gap_before_ms": 0.12, "rank": 0}

seq_len is the full context length (computed + cached).  When prefix
caching is active, num_computed_tokens < seq_len.  kv_cache_bytes is the
byte size of the KV cache fetched from the prefix cache for this layer
(num_cached_tokens * kv_bytes_per_token_per_layer).

gap_before_ms is the GPU idle time between the previous layer's end and
this layer's start.  For the first decoder layer, it measures the gap from
the forward-pass start (embedding + position encoding) when
mark_forward_start() is used, otherwise 0.  Large gap values indicate
pipeline bubbles — the GPU was stalled waiting for data (e.g. KV cache
arriving from CPU/SSD/network).

An iteration_summary record is emitted after each forward pass:

    {"iteration": 0, "rank": 0, "type": "iteration_summary",
     "total_forward_ms": 42.5, "total_layer_compute_ms": 38.2,
     "total_gap_ms": 2.1, "pre_model_ms": 1.5, "post_model_ms": 0.7,
     "bubble_fraction": 0.049, "num_layers": 32}

Usage in gpu_model_runner.py:

    # Registration (once, after model load)
    from vllm.utils.cuda_event_hooks import LayerwiseCUDAEventHooks
    self.cuda_event_hooks = LayerwiseCUDAEventHooks(output_prefix)
    self.cuda_event_hooks.register_hooks(self.model)

    # Before model forward
    self.cuda_event_hooks.mark_forward_start()

    # After model forward
    self.cuda_event_hooks.mark_forward_end()
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

        # Optional forward-pass boundary events.  Set via mark_forward_start()
        # and mark_forward_end() in the model runner.  Used to measure the
        # pre-first-layer and post-last-layer overhead (embedding, LM head).
        self._forward_start_event: torch.cuda.Event | None = None
        self._forward_end_event: torch.cuda.Event | None = None

    # ------------------------------------------------------------------
    # Forward-pass boundary markers
    # ------------------------------------------------------------------

    def mark_forward_start(self) -> None:
        """Record a CUDA event just before the model forward pass.

        Call this right before ``model(...)`` so that the gap between
        this event and the first decoder layer's start event captures the
        embedding / position-encoding overhead.
        """
        self._forward_start_event = torch.cuda.Event(enable_timing=True)
        self._forward_start_event.record()

    def mark_forward_end(self) -> None:
        """Record a CUDA event just after the model forward pass.

        Call this right after ``model(...)`` returns so that the gap between
        the last decoder layer's end event and this event captures the
        LM-head / final-norm overhead.
        """
        self._forward_end_event = torch.cuda.Event(enable_timing=True)
        self._forward_end_event.record()

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

        # Single synchronisation point — wait for the last end marker, plus
        # the forward-end event if present.
        last_sync = self._forward_end_event or self._pending[-1][3]
        last_sync.synchronize()

        # Sort pending by layer index so records are in forward-pass order
        # even if hooks fired out of order (unlikely but defensive).
        self._pending.sort(key=lambda t: t[1])  # sort by layer_idx

        cached = num_cached_tokens or 0
        records: list[dict] = []
        total_layer_compute_ms = 0.0
        total_gap_ms = 0.0

        for i, (layer_name, layer_idx, start, end,
                param_bytes, input_bytes, output_bytes, kv_bpt) in enumerate(
                    self._pending):
            elapsed = start.elapsed_time(end)
            total_layer_compute_ms += elapsed

            # Compute gap before this layer: time the GPU was idle between
            # the previous layer's end (or forward start) and this layer's
            # start.
            if i == 0 and self._forward_start_event is not None:
                gap_before = self._forward_start_event.elapsed_time(start)
            elif i > 0:
                prev_end = self._pending[i - 1][3]
                gap_before = prev_end.elapsed_time(start)
            else:
                gap_before = 0.0
            total_gap_ms += gap_before

            record: dict = {
                "iteration": self._iteration,
                "rank": self._rank,
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "elapsed_ms": elapsed,
                "gap_before_ms": gap_before,
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

        # Compute overall forward-pass timing if boundary events exist.
        pre_model_ms = 0.0
        post_model_ms = 0.0
        total_forward_ms = 0.0
        has_forward_boundaries = (self._forward_start_event is not None
                                  and self._forward_end_event is not None)
        if has_forward_boundaries:
            total_forward_ms = self._forward_start_event.elapsed_time(
                self._forward_end_event)
            # pre_model_ms is already the first layer's gap_before_ms
            pre_model_ms = records[0]["gap_before_ms"] if records else 0.0
            # post_model_ms: last layer end → forward end
            if self._pending:
                last_end = self._pending[-1][3]
                post_model_ms = last_end.elapsed_time(self._forward_end_event)

        # Emit iteration summary record.
        summary: dict = {
            "iteration": self._iteration,
            "rank": self._rank,
            "type": "iteration_summary",
            "num_layers": len(records),
            "total_layer_compute_ms": total_layer_compute_ms,
            "total_gap_ms": total_gap_ms,
        }
        if has_forward_boundaries:
            summary["total_forward_ms"] = total_forward_ms
            summary["pre_model_ms"] = pre_model_ms
            summary["post_model_ms"] = post_model_ms
            summary["bubble_fraction"] = (
                total_gap_ms / total_forward_ms
                if total_forward_ms > 0 else 0.0
            )
        if num_tokens is not None:
            summary["seq_len"] = num_tokens + cached
            summary["num_computed_tokens"] = num_tokens
            summary["num_cached_tokens"] = cached
        records.append(summary)

        with open(self._output_path, "a") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        self._pending.clear()
        self._forward_start_event = None
        self._forward_end_event = None
        self._iteration += 1
        return records

    # ------------------------------------------------------------------
    # KV transfer logging
    # ------------------------------------------------------------------

    def log_kv_transfers(self, kv_connector_stats) -> list[dict]:
        """Write KV transfer timing records from connector stats to JSONL.

        Called after each engine step with the ``kv_connector_stats`` from
        ``KVConnectorOutput``.  Each individual transfer operation becomes
        a ``kv_transfer`` record:

            {"type": "kv_transfer", "iteration": 5, "rank": 0,
             "direction": "CPU_to_GPU", "transfer_bytes": 125829120,
             "transfer_time_s": 0.0042, "bandwidth_gb_s": 28.6}

        Args:
            kv_connector_stats: An ``OffloadingConnectorStats`` (or any
                ``KVConnectorStats`` whose ``.data`` is a dict mapping
                direction strings to lists of ``{"op_size": int,
                "op_time": float}`` dicts).  ``None`` is silently ignored.

        Returns:
            The list of dicts written during this call.
        """
        # Temporary debug logging
        import os
        def _dbg(msg):
            with open("/tmp/offload_debug.log", "a") as _f:
                _f.write(f"[pid={os.getpid()}] log_kv_transfers: {msg}\n")

        _dbg(f"called, stats={kv_connector_stats}, type={type(kv_connector_stats)}")

        if kv_connector_stats is None:
            _dbg("stats is None, returning early")
            return []

        data = getattr(kv_connector_stats, "data", None)
        _dbg(f"data={data}, type={type(data)}, bool={bool(data) if data is not None else 'N/A'}")
        if not data:
            _dbg("data is falsy, returning early")
            return []

        records: list[dict] = []
        for direction, ops_list in data.items():
            _dbg(f"  direction={direction}, num_ops={len(ops_list)}")
            for op in ops_list:
                # OffloadingOperationMetrics is a dataclass but serialised
                # as a dict when it crosses process boundaries.
                if isinstance(op, dict):
                    op_size = op.get("op_size", 0)
                    op_time = op.get("op_time", 0.0)
                else:
                    op_size = getattr(op, "op_size", 0)
                    op_time = getattr(op, "op_time", 0.0)

                bw = (op_size / op_time / 1e9) if op_time > 0 else 0.0
                records.append({
                    "type": "kv_transfer",
                    "iteration": self._iteration - 1,  # just-flushed iter
                    "rank": self._rank,
                    "direction": direction,
                    "transfer_bytes": op_size,
                    "transfer_time_s": op_time,
                    "bandwidth_gb_s": round(bw, 3),
                })

        _dbg(f"total records to write: {len(records)}, directions: {[r['direction'] for r in records]}")
        if records:
            with open(self._output_path, "a") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        return records