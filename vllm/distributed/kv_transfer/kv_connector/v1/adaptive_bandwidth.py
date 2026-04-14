"""Adaptive bandwidth allocator for ICMS selective prefill.

Computes per-request KV budget based on available network bandwidth and
concurrent request load.  The budget is recomputed at each stride boundary.

See docs/adaptive_bandwidth_design.md for the full algorithm.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RequestDemand:
    """Bandwidth demand for one active request."""
    request_id: str
    demand_bps: float  # bytes per second needed for full KV transfer
    new_tokens: int
    cache_tokens: int


class ComputeSlackTable:
    """Lookup table for per-layer compute time (from offline profiling).

    Interpolates between profiled (new_tokens, cache_tokens) pairs.
    """

    def __init__(self, table_path: str | Path | None = None):
        self._entries: dict[tuple[int, int], float] = {}  # (new, cache) → ms
        self._new_tokens_sorted: list[int] = []
        self._cache_tokens_sorted: list[int] = []
        if table_path:
            self.load(table_path)

    def load(self, path: str | Path):
        with open(path) as f:
            data = json.load(f)
        for entry in data.get("table", {}).values():
            key = (entry["new_tokens"], entry["cache_tokens"])
            self._entries[key] = entry["per_layer_ms"]
        new_set = sorted(set(k[0] for k in self._entries))
        cache_set = sorted(set(k[1] for k in self._entries))
        self._new_tokens_sorted = new_set
        self._cache_tokens_sorted = cache_set
        logger.info("Loaded compute slack table: %d entries from %s",
                    len(self._entries), path)

    def lookup(self, new_tokens: int, cache_tokens: int) -> float:
        """Return per-layer compute time in ms, interpolating if needed."""
        if not self._entries:
            return 15.0  # reasonable default (ms)

        # Find nearest profiled values.
        nt = self._nearest(self._new_tokens_sorted, new_tokens)
        ct = self._nearest(self._cache_tokens_sorted, cache_tokens)
        return self._entries.get((nt, ct), 15.0)

    @staticmethod
    def _nearest(sorted_list: list[int], value: int) -> int:
        if not sorted_list:
            return value
        best = sorted_list[0]
        for v in sorted_list:
            if abs(v - value) < abs(best - value):
                best = v
        return best


class AdaptiveBandwidthAllocator:
    """Compute-side bandwidth allocator.

    Tracks concurrent requests' bandwidth demands and computes per-request
    budget based on proportional bandwidth allocation.
    """

    def __init__(
        self,
        link_bandwidth_bps: float,
        kv_page_bytes: int,
        slack_table: ComputeSlackTable | None = None,
    ):
        self._link_bw = link_bandwidth_bps
        self._kv_page_bytes = kv_page_bytes
        self._slack_table = slack_table or ComputeSlackTable()
        self._active: dict[str, RequestDemand] = {}
        self._lock = threading.Lock()

    def register_request(
        self, request_id: str, new_tokens: int, cache_tokens: int,
        total_cache_pages: int,
    ) -> float:
        """Register a new request and return its initial budget (0.0-1.0).

        Args:
            request_id: unique request identifier
            new_tokens: tokens being computed (continuation)
            cache_tokens: tokens stored externally (context from ICMS)
            total_cache_pages: total context pages stored in ICMS

        Returns:
            budget: fraction of cache pages to fetch (0.0-1.0)
        """
        # Compute per-layer slack from profiling table.
        per_layer_ms = self._slack_table.lookup(new_tokens, cache_tokens)

        # Full KV demand: bytes needed per layer at 100% budget.
        full_kv_bytes = total_cache_pages * self._kv_page_bytes

        # Demand: bytes/sec needed to transfer full KV within the compute slack.
        per_layer_sec = per_layer_ms / 1000.0
        if per_layer_sec > 0:
            demand_bps = full_kv_bytes / per_layer_sec
        else:
            demand_bps = float("inf")

        demand = RequestDemand(
            request_id=request_id,
            demand_bps=demand_bps,
            new_tokens=new_tokens,
            cache_tokens=cache_tokens,
        )

        with self._lock:
            self._active[request_id] = demand
            budget = self._compute_budget(demand)

        logger.debug(
            "register_request %s: new=%d cache=%d pages=%d "
            "slack=%.2fms demand=%.1f MB/s budget=%.3f",
            request_id, new_tokens, cache_tokens, total_cache_pages,
            per_layer_ms, demand_bps / 1e6, budget,
        )
        return budget

    def get_budget(self, request_id: str) -> float:
        """Recompute budget for an existing request (called per stride group).

        Supply may have changed if requests arrived/departed since registration.
        """
        with self._lock:
            demand = self._active.get(request_id)
            if demand is None:
                return 1.0
            return self._compute_budget(demand)

    def unregister_request(self, request_id: str):
        """Remove a finished request from the demand tracker."""
        with self._lock:
            self._active.pop(request_id, None)

    def _compute_budget(self, demand: RequestDemand) -> float:
        """Proportional bandwidth allocation → budget.

        Must be called with self._lock held.
        """
        total_demand = sum(d.demand_bps for d in self._active.values())
        if total_demand <= 0:
            return 1.0

        # Proportional share of the link bandwidth.
        my_share = self._link_bw * (demand.demand_bps / total_demand)
        # Budget = what fraction of the full KV we can fetch in time.
        budget = min(1.0, my_share / demand.demand_bps)
        return budget

    @property
    def num_active(self) -> int:
        with self._lock:
            return len(self._active)
