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
    """Lookup table for end-to-end forward-pass time (from offline profiling).

    Returns total_forward_ms for a (new_tokens, cache_tokens) pair. Paired
    with total-KV-bytes (pages × kv_page_bytes × num_layers) in the
    allocator — end-to-end slack × total KV avoids the noise of dividing
    through by num_layers on both sides.
    """

    # Fallback when no table is loaded or a lookup misses. ~720 ms was the
    # old HF-profile default; the new vLLM profile puts most points near
    # 35-50 ms, so we bias low to avoid overestimating slack on miss.
    _DEFAULT_MS = 50.0

    def __init__(self, table_path: str | Path | None = None):
        self._entries: dict[tuple[int, int], float] = {}  # (new, cache) → total_forward_ms
        self._new_tokens_sorted: list[int] = []
        self._cache_tokens_sorted: list[int] = []
        if table_path:
            self.load(table_path)

    def load(self, path: str | Path):
        with open(path) as f:
            data = json.load(f)
        for entry in data.get("table", {}).values():
            key = (entry["new_tokens"], entry["cache_tokens"])
            # Prefer end-to-end time; fall back to per_layer × num_layers
            # for backward-compat with the pre-2026-04-23 profiles.
            if "total_forward_ms" in entry:
                self._entries[key] = entry["total_forward_ms"]
            elif "per_layer_ms" in entry and "num_layers" in entry:
                self._entries[key] = entry["per_layer_ms"] * entry["num_layers"]
            else:
                continue
        new_set = sorted(set(k[0] for k in self._entries))
        cache_set = sorted(set(k[1] for k in self._entries))
        self._new_tokens_sorted = new_set
        self._cache_tokens_sorted = cache_set
        logger.info("Loaded compute slack table: %d entries from %s",
                    len(self._entries), path)

    def lookup(self, new_tokens: int, cache_tokens: int) -> float:
        """Return end-to-end forward time in ms (nearest-neighbor on both axes)."""
        if not self._entries:
            return self._DEFAULT_MS

        nt = self._nearest(self._new_tokens_sorted, new_tokens)
        ct = self._nearest(self._cache_tokens_sorted, cache_tokens)
        return self._entries.get((nt, ct), self._DEFAULT_MS)

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
        num_layers: int = 1,
    ):
        self._link_bw = link_bandwidth_bps
        self._kv_page_bytes = kv_page_bytes
        self._slack_table = slack_table or ComputeSlackTable()
        # num_layers scales the KV byte count to total cache transfer (all
        # layers × pages × kv_page_bytes). Paired with end-to-end slack
        # from the slack table so the ratio is well-defined. Default 1 is
        # backward-compat for callers that haven't wired model geometry.
        self._num_layers = max(1, int(num_layers))
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
        # End-to-end forward time from the slack table (ms). Paired below
        # with total-KV (all layers) — the ratio is the sustained rate
        # needed to land every cached page during the prefill.
        total_forward_ms = self._slack_table.lookup(new_tokens, cache_tokens)

        # Full KV demand: total bytes across all layers at 100% budget.
        full_kv_bytes = (
            total_cache_pages * self._kv_page_bytes * self._num_layers
        )

        total_forward_sec = total_forward_ms / 1000.0
        if total_forward_sec > 0:
            demand_bps = full_kv_bytes / total_forward_sec
        else:
            demand_bps = float("inf")

        demand = RequestDemand(
            request_id=request_id,
            demand_bps=demand_bps,
            new_tokens=new_tokens,
            cache_tokens=cache_tokens,
        )

        with self._lock:
            n_active_before = len(self._active)
            total_demand_before = sum(d.demand_bps for d in self._active.values())
            stale_rids_sample = list(self._active.keys())[:5]
            self._active[request_id] = demand
            budget = self._compute_budget(demand)
            total_demand_after = sum(d.demand_bps for d in self._active.values())
            my_share = self._link_bw * (demand.demand_bps / total_demand_after) \
                if total_demand_after > 0 else self._link_bw

        logger.info(
            "register_request %s: new=%d cache=%d pages=%d "
            "kv_page_B=%d num_layers=%d full_kv_MB=%.1f "
            "slack=%.2fms demand=%.1f MB/s "
            "n_active_before=%d total_demand_before=%.1f MB/s "
            "stale=%s "
            "link_bw_MBs=%.1f my_share_MBs=%.1f budget=%.3f",
            request_id, new_tokens, cache_tokens, total_cache_pages,
            self._kv_page_bytes, self._num_layers, full_kv_bytes / 1e6,
            total_forward_ms, demand_bps / 1e6,
            n_active_before, total_demand_before / 1e6,
            stale_rids_sample,
            self._link_bw / 1e6, my_share / 1e6, budget,
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
            existed = request_id in self._active
            self._active.pop(request_id, None)
            n_active_after = len(self._active)
        logger.info(
            "unregister_request %s: existed=%s n_active_after=%d",
            request_id, existed, n_active_after,
        )

    def _compute_budget(self, demand: RequestDemand) -> float:
        """Proportional bandwidth allocation → budget (compute side only).

        The storage side manages its own contention internally — it throttles
        KV writes based on its concurrent request load.  The compute side
        only manages its own demand/supply.

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
