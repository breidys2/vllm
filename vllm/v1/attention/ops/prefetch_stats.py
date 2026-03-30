# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prefetch Runtime Statistics — per-layer, per-step instrumentation.

Provides a centralized, togglable stats accumulator for layer-pipelined
KV cache prefetch (Quest, InfiniGen, etc.).  When ``enabled=False`` (default),
every method returns immediately with a single branch check (~10 ns).

Four categories of metrics:
  (a) KV cache fetched per layer — token count + bytes
  (b) Latency breakdown — fetch vs forward vs rehearsal (pipeline stalls)
  (c) Rehearsal precomputation latency in isolation
  (d) Benchmark accuracy (collected externally, not by this module)

Usage (instrumentation sites)::

    stats.begin_rehearsal(layer_idx)
    mask = engine.rehearse(...)
    stats.end_rehearsal(layer_idx, tokens_cached, tokens_selected)

    stats.begin_fetch(layer_idx, stream)
    # ... async CPU→GPU transfer ...
    stats.end_fetch(layer_idx, stream, num_bytes)

    stats.begin_wait(layer_idx)
    torch.cuda.current_stream().wait_event(event)
    stats.end_wait(layer_idx)

    stats.begin_forward(layer_idx)
    output = attention_forward(...)
    stats.end_forward(layer_idx)

Usage (collection)::

    # In benchmark scripts:
    steps = stats.drain()
    for step in steps:
        for ls in step.layer_stats:
            print(f"Layer {ls.layer_idx}: rehearsal={ls.rehearsal_ms:.3f}ms")

    # In OffloadingConnectorStats pipeline:
    data = stats.to_connector_stats_dict()
    connector_stats.data.update(data)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PrefetchLayerStats:
    """Metrics for a single layer in a single decode/prefill step."""

    layer_idx: int = 0
    tokens_cached: int = 0
    tokens_selected: int = 0
    pages_selected: int = 0
    pages_total: int = 0
    kv_bytes_fetched: int = 0

    # Latencies in milliseconds
    q_compute_ms: float = 0.0
    rehearsal_ms: float = 0.0
    fetch_ms: float = 0.0
    forward_ms: float = 0.0
    stall_ms: float = 0.0

    @property
    def selection_ratio(self) -> float:
        if self.tokens_cached == 0:
            return 0.0
        return self.tokens_selected / self.tokens_cached

    @property
    def kv_gb_fetched(self) -> float:
        return self.kv_bytes_fetched / 1e9

    @property
    def scoring_plus_fetch_ms(self) -> float:
        """Total time on the critical path: scoring + transfer."""
        return self.rehearsal_ms + self.fetch_ms

    @property
    def overlap_ratio(self) -> float:
        """Fraction of scoring+fetch hidden behind compute.

        1.0 = fully overlapped (stall=0), 0.0 = fully exposed.
        """
        total = self.scoring_plus_fetch_ms
        if total <= 0:
            return 1.0
        return max(0.0, 1.0 - self.stall_ms / total)

    def to_dict(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "tokens_cached": self.tokens_cached,
            "tokens_selected": self.tokens_selected,
            "pages_selected": self.pages_selected,
            "pages_total": self.pages_total,
            "selection_ratio": round(self.selection_ratio, 4),
            "kv_bytes_fetched": self.kv_bytes_fetched,
            "kv_gb_fetched": round(self.kv_gb_fetched, 6),
            "q_compute_ms": round(self.q_compute_ms, 3),
            "rehearsal_ms": round(self.rehearsal_ms, 3),
            "fetch_ms": round(self.fetch_ms, 3),
            "forward_ms": round(self.forward_ms, 3),
            "stall_ms": round(self.stall_ms, 3),
            "overlap_ratio": round(self.overlap_ratio, 4),
        }


@dataclass
class PrefetchStepStats:
    """Aggregated stats for one decode step across all layers."""

    step_id: int = 0
    layer_stats: list[PrefetchLayerStats] = field(default_factory=list)

    @property
    def total_rehearsal_ms(self) -> float:
        return sum(ls.rehearsal_ms for ls in self.layer_stats)

    @property
    def total_fetch_ms(self) -> float:
        return sum(ls.fetch_ms for ls in self.layer_stats)

    @property
    def total_forward_ms(self) -> float:
        return sum(ls.forward_ms for ls in self.layer_stats)

    @property
    def total_stall_ms(self) -> float:
        return sum(ls.stall_ms for ls in self.layer_stats)

    @property
    def total_kv_bytes(self) -> int:
        return sum(ls.kv_bytes_fetched for ls in self.layer_stats)

    @property
    def total_tokens_selected(self) -> int:
        return sum(ls.tokens_selected for ls in self.layer_stats)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "total_rehearsal_ms": round(self.total_rehearsal_ms, 3),
            "total_fetch_ms": round(self.total_fetch_ms, 3),
            "total_forward_ms": round(self.total_forward_ms, 3),
            "total_stall_ms": round(self.total_stall_ms, 3),
            "total_kv_bytes": self.total_kv_bytes,
            "total_kv_gb": round(self.total_kv_bytes / 1e9, 6),
            "total_tokens_selected": self.total_tokens_selected,
            "num_layers": len(self.layer_stats),
            "layer_stats": [ls.to_dict() for ls in self.layer_stats],
        }


class PrefetchStats:
    """Thread-safe accumulator for InfiniGen per-layer metrics.

    When ``enabled=False``, all methods are no-ops (zero overhead).
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._current_step: PrefetchStepStats | None = None
        self._completed_steps: list[PrefetchStepStats] = []
        self._step_counter = 0

        # Per-layer CPU timestamps (perf_counter)
        self._q_compute_start: dict[int, float] = {}
        self._rehearsal_start: dict[int, float] = {}
        self._wait_start: dict[int, float] = {}
        self._forward_start: dict[int, float] = {}

        # Per-layer CUDA events for fetch timing
        self._fetch_start_events: dict[int, object] = {}
        self._fetch_end_events: dict[int, object] = {}

    # -- Step lifecycle -------------------------------------------------------

    def begin_step(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            # Finalize any in-flight step
            if self._current_step is not None:
                self._completed_steps.append(self._current_step)
            self._step_counter += 1
            self._current_step = PrefetchStepStats(step_id=self._step_counter)
            self._q_compute_start.clear()
            self._rehearsal_start.clear()
            self._wait_start.clear()
            self._forward_start.clear()
            self._fetch_start_events.clear()
            self._fetch_end_events.clear()

    def end_step(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._current_step is not None:
                # Resolve any pending CUDA event timings
                self._resolve_fetch_timings()
                self._completed_steps.append(self._current_step)
                self._current_step = None

    # -- Q computation timing -------------------------------------------------

    def begin_q_compute(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        self._q_compute_start[layer_idx] = time.perf_counter()

    def end_q_compute(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        start = self._q_compute_start.pop(layer_idx, None)
        elapsed_ms = (time.perf_counter() - start) * 1000 if start else 0.0
        with self._lock:
            ls = self._get_or_create_layer(layer_idx)
            ls.q_compute_ms = elapsed_ms

    # -- Rehearsal/scoring timing ---------------------------------------------

    def begin_rehearsal(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        self._rehearsal_start[layer_idx] = time.perf_counter()

    def end_rehearsal(
        self,
        layer_idx: int,
        tokens_cached: int = 0,
        tokens_selected: int = 0,
    ) -> None:
        if not self.enabled:
            return
        start = self._rehearsal_start.pop(layer_idx, None)
        elapsed_ms = (time.perf_counter() - start) * 1000 if start else 0.0

        with self._lock:
            ls = self._get_or_create_layer(layer_idx)
            ls.rehearsal_ms = elapsed_ms
            ls.tokens_cached = tokens_cached
            ls.tokens_selected = tokens_selected

    # -- Fetch timing (metric a, b) ------------------------------------------

    def begin_fetch(self, layer_idx: int, stream: object = None) -> None:
        """Record fetch start. Uses CUDA events if available, else CPU time."""
        if not self.enabled:
            return
        try:
            import torch
            if stream is not None and torch.cuda.is_available():
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream)  # type: ignore[arg-type]
                self._fetch_start_events[layer_idx] = event
                return
        except ImportError:
            pass
        # Fallback: CPU timing
        self._fetch_start_events[layer_idx] = time.perf_counter()

    def end_fetch(
        self,
        layer_idx: int,
        stream: object = None,
        num_bytes: int = 0,
    ) -> None:
        """Record fetch end and byte count."""
        if not self.enabled:
            return
        try:
            import torch
            if stream is not None and torch.cuda.is_available():
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream)  # type: ignore[arg-type]
                self._fetch_end_events[layer_idx] = event
                with self._lock:
                    ls = self._get_or_create_layer(layer_idx)
                    ls.kv_bytes_fetched = num_bytes
                return
        except ImportError:
            pass
        # Fallback: CPU timing
        start = self._fetch_start_events.pop(layer_idx, None)
        elapsed_ms = 0.0
        if isinstance(start, float):
            elapsed_ms = (time.perf_counter() - start) * 1000
        with self._lock:
            ls = self._get_or_create_layer(layer_idx)
            ls.fetch_ms = elapsed_ms
            ls.kv_bytes_fetched = num_bytes

    # -- Wait/stall timing (metric b) ----------------------------------------

    def begin_wait(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        self._wait_start[layer_idx] = time.perf_counter()

    def end_wait(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        start = self._wait_start.pop(layer_idx, None)
        elapsed_ms = (time.perf_counter() - start) * 1000 if start else 0.0
        with self._lock:
            ls = self._get_or_create_layer(layer_idx)
            ls.stall_ms = elapsed_ms

    # -- Forward pass timing (metric b) --------------------------------------

    def begin_forward(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        self._forward_start[layer_idx] = time.perf_counter()

    def end_forward(self, layer_idx: int) -> None:
        if not self.enabled:
            return
        start = self._forward_start.pop(layer_idx, None)
        elapsed_ms = (time.perf_counter() - start) * 1000 if start else 0.0
        with self._lock:
            ls = self._get_or_create_layer(layer_idx)
            ls.forward_ms = elapsed_ms

    # -- KV fetch size recording (metric a) ----------------------------------

    def record_kv_fetch(
        self,
        layer_idx: int,
        tokens_cached: int,
        tokens_selected: int,
        kv_bytes: int,
    ) -> None:
        """Record KV cache fetch size for a layer (can be called standalone)."""
        if not self.enabled:
            return
        with self._lock:
            ls = self._get_or_create_layer(layer_idx)
            ls.tokens_cached = tokens_cached
            ls.tokens_selected = tokens_selected
            ls.kv_bytes_fetched = kv_bytes

    # -- Collection -----------------------------------------------------------

    def drain(self) -> list[PrefetchStepStats]:
        """Return all completed steps and reset.

        Used by benchmark scripts to collect per-step per-layer data.
        """
        with self._lock:
            # Finalize current step if any
            if self._current_step is not None:
                self._resolve_fetch_timings()
                self._completed_steps.append(self._current_step)
                self._current_step = None

            steps = self._completed_steps
            self._completed_steps = []
            return steps

    def to_connector_stats_dict(self) -> dict[str, list[dict]]:
        """Serialize accumulated stats for OffloadingConnectorStats.data.

        Returns a dict with key ``"infinigen_layer_stats"`` mapping to a
        list of per-layer stat dicts.
        """
        steps = self.drain()
        layer_dicts: list[dict] = []
        for step in steps:
            for ls in step.layer_stats:
                d = ls.to_dict()
                d["step_id"] = step.step_id
                layer_dicts.append(d)
        if not layer_dicts:
            return {}
        return {"prefetch_layer_stats": layer_dicts}

    def reset(self) -> None:
        """Discard all accumulated data."""
        with self._lock:
            self._current_step = None
            self._completed_steps.clear()
            self._q_compute_start.clear()
            self._rehearsal_start.clear()
            self._wait_start.clear()
            self._forward_start.clear()
            self._fetch_start_events.clear()
            self._fetch_end_events.clear()

    # -- Internal helpers -----------------------------------------------------

    def _get_or_create_layer(self, layer_idx: int) -> PrefetchLayerStats:
        """Get or create the layer stats entry in the current step.

        Caller must hold ``self._lock``.
        """
        if self._current_step is None:
            self._step_counter += 1
            self._current_step = PrefetchStepStats(
                step_id=self._step_counter
            )

        for ls in self._current_step.layer_stats:
            if ls.layer_idx == layer_idx:
                return ls

        ls = PrefetchLayerStats(layer_idx=layer_idx)
        self._current_step.layer_stats.append(ls)
        return ls

    def _resolve_fetch_timings(self) -> None:
        """Resolve CUDA event pairs into elapsed milliseconds.

        Called under lock during ``end_step()`` or ``drain()``.
        """
        for layer_idx in list(self._fetch_start_events.keys()):
            start_evt = self._fetch_start_events.get(layer_idx)
            end_evt = self._fetch_end_events.get(layer_idx)

            if start_evt is None or end_evt is None:
                continue

            # CUDA events
            try:
                import torch
                if (
                    isinstance(start_evt, torch.cuda.Event)
                    and isinstance(end_evt, torch.cuda.Event)
                ):
                    # Synchronize to ensure events are recorded
                    end_evt.synchronize()
                    elapsed_ms = start_evt.elapsed_time(end_evt)
                    ls = self._get_or_create_layer(layer_idx)
                    ls.fetch_ms = elapsed_ms
            except (ImportError, RuntimeError):
                pass

        self._fetch_start_events.clear()
        self._fetch_end_events.clear()
