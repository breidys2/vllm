"""Adaptive bandwidth allocator for ICMS selective prefill.

Computes per-request KV budget based on available network bandwidth and
concurrent request load.  The budget is recomputed at each stride boundary.

See docs/adaptive_bandwidth_design.md for the full algorithm.
"""

from __future__ import annotations

import json
import logging
import os
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
    # Most recent storage-side effective supply for this rid, in bytes/sec.
    # 0.0 means "not yet seen / adaptive disabled storage-side" — the
    # allocator falls back to compute_supply alone. Updated by
    # apply_storage_supply() from each Score/FetchAll reply.
    last_storage_supply_bps: float = 0.0


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

    # Clamp the computed budget to [_MIN_BUDGET, 1.0] to keep accuracy
    # bounded at very long context (where bandwidth would otherwise drop
    # k below useful page counts) and to skip scoring/summary-fetch
    # overhead when we'd be fetching nearly everything anyway.
    _MIN_BUDGET = 0.1
    # Above this raw budget we round up to 1.0 and skip scoring entirely
    # (caller routes to the kFetchAll fast path on budget==1.0).
    _CEILING_SNAP = 0.95

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
        # Budget floor, env-overridable (default _MIN_BUDGET=0.1). Lets the
        # bench sweep a different accuracy floor (e.g. 0.2) without a code
        # fork. Read once at construction.
        self._min_budget = float(
            os.environ.get("ICMS_ADAPTIVE_MIN_BUDGET", self._MIN_BUDGET))
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

    def compute_budget(self, **kwargs) -> float:
        """Quest-hook adapter (2026-06-03). The Quest hook calls
        `budget_computer.compute_budget(approximate_scores=…, layer_idx=…,
        num_layers=…)` once per (layer, batch). The hook doesn't pass a
        rid because it fires at batch granularity. We return the MIN
        budget across all active rids so no rid is allocated more than
        its proportional share. Returns 1.0 when no rid is registered
        (idle / before any matched-prefix request entered).

        Pre-2026-06-03 this method was missing, causing AttributeError
        if the hook tried to call it — but in practice the hook never
        reached the call because the V2-style allocator pickup chain
        (`connector_worker.spec.get_adaptive_allocator`) didn't match
        ICMS, so `budget_computer` stayed as DynamicBudget(1.0). After
        ICMS-direct allocator pickup landed, this adapter is required.
        kwargs are ignored — kept for API compat with QuestHookManager.
        """
        with self._lock:
            if not self._active:
                return 1.0
            return min(
                self._compute_budget(d) for d in self._active.values())

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

    def _compute_share(self, demand: RequestDemand) -> float:
        """Compute-side proportional share for this request, in B/s.

        Same math as `BandwidthRegistry::effective_supply_bps` on the
        server: link_bw * (mine / total). Must be called with the lock
        held. Returns link_bw when the registry has only this one entry.
        """
        total_demand = sum(d.demand_bps for d in self._active.values())
        if total_demand <= 0:
            return float(self._link_bw)
        return self._link_bw * (demand.demand_bps / total_demand)

    def _compute_budget(self, demand: RequestDemand) -> float:
        """Effective budget = min(compute_share, storage_supply) / demand,
        clamped to [MIN_BUDGET, 1.0] with the 0.95 ceiling-snap.

        `storage_supply` is the most recent value the BF2 returned in
        ScoreReplyPayload.effective_supply_bps for this rid (0 if no
        reply yet, or storage-side allocation is off — in which case
        compute_share alone gates).

        Must be called with self._lock held.
        """
        if demand.demand_bps <= 0:
            return 1.0

        my_share = self._compute_share(demand)
        # Min against storage side. last_storage_supply_bps==0 means
        # "no signal" → compute side alone.
        if demand.last_storage_supply_bps > 0:
            effective = min(my_share, demand.last_storage_supply_bps)
        else:
            effective = my_share
        budget = min(1.0, effective / demand.demand_bps)

        # Clamp + ceiling-snap. Floor at _MIN_BUDGET preserves attention
        # quality at very high ctx (where the natural budget drops below a
        # useful page count). Ceiling snap at >=_CEILING_SNAP rounds up to
        # 1.0 — caller's budget==1.0 dispatch routes to kFetchAll, which
        # skips the per-stride score RPC and per-page top-k selection that
        # would otherwise add fixed overhead for negligible bandwidth
        # savings (≤5% pages skipped).
        if budget >= self._CEILING_SNAP:
            return 1.0
        if budget < self._min_budget:
            return self._min_budget
        return budget

    # ── Storage-side hookup ─────────────────────────────────────────────

    def demand_bps_for(self, request_id: str) -> int:
        """Demand in bytes/sec for a registered request, rounded to int
        (the wire protocol uses uint64). 0 if not registered."""
        with self._lock:
            d = self._active.get(request_id)
            return int(d.demand_bps) if d is not None else 0

    def compute_supply_bps_for(self, request_id: str) -> int:
        """Host-side proportional share of link bw for a registered
        request, rounded to int. 0 if not registered.

        This is the value the connector stuffs into ScorePayload's
        compute_supply_bps so the server can take min() against its own
        proportional share.
        """
        with self._lock:
            d = self._active.get(request_id)
            if d is None:
                return 0
            return int(self._compute_share(d))

    def apply_storage_supply(self, request_id: str,
                              effective_supply_bps: int) -> None:
        """Stash the storage-side effective_supply from a Score/FetchAll
        reply so the next `get_budget(rid)` can take min(compute,
        storage). 0 means "storage-side allocation off" — clear any
        prior stash so we revert to compute-only."""
        with self._lock:
            d = self._active.get(request_id)
            if d is None:
                return
            d.last_storage_supply_bps = float(effective_supply_bps or 0)

    @property
    def num_active(self) -> int:
        with self._lock:
            return len(self._active)
