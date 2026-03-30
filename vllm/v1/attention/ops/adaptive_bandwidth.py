"""Adaptive Bandwidth Allocation for KV cache prefetching.

Overview
--------
When fetching a cached KV prefix from CPU (or remote storage) during prefill,
the amount of data that can be transferred without stalling the compute stream
depends on how much time each decoder layer takes to run — the "compute window".

This module computes a per-request KV fetch budget (fraction of pages to load)
based on:

    demand  = kv_bytes_per_layer / compute_time_per_layer   [bytes/sec]
    budget  = min(1.0, effective_bandwidth / total_demand)

where effective_bandwidth = min(compute_side_bw, storage_side_bw).  With a
single request and one-sided CPU storage, this simplifies to:

    budget = min(1.0, bandwidth_bps * compute_time_per_layer / kv_bytes_per_layer)

Intuitively: load as many pages as the NVMe/PCIe link can deliver during the
time it takes to run one decoder layer.

Compute time estimation
-----------------------
A per-layer compute time look-up table is loaded from an offline profile JSON
(generated once per model + GPU by experiments/profile_compute.py).  The
profile fits the following linear model to measured wall-clock times:

    t(n, L) = a*n + b*n*L + c

where:
    n  = num_new_tokens  (query length during prefill)
    L  = context_length  (cached KV prefix length)
    a  = coefficient for FFN / token-independent work  [sec/token]
    b  = coefficient for attention                     [sec/(token*token)]
    c  = constant layer overhead                       [sec]

If no profile is found the class falls back to a pure analytical estimate
using the model's hyperparameters and an assumed GPU FLOP rate.  The fallback
is clearly warned so you know to run the offline profiler.

Remote storage extensibility
-----------------------------
The allocator stores separate compute-side and storage-side bandwidth values.
For local CPU offloading both are the same NVMe/PCIe bandwidth, so
effective_bw = min(bw, bw) = bw.

For remote storage: the storage node computes its own concurrent-request load
and sends a storage_supply value back with the KV data.  A single call to
override_storage_bandwidth() tightens the budget without any other changes.

Usage
-----
    allocator = AdaptiveBandwidthAllocator.from_profile(
        profile_path="experiments/compute_profiles/qwen3_H100.json",
        compute_bandwidth_gbps=32.0,   # peak PCIe 4.0 x16 in one direction
    )

    # Called by the connector when a load request arrives.
    allocator.register_request(
        req_id="abc",
        kv_bytes_per_layer=16_777_216,   # computed from prefix + spec params
        num_new_tokens=128,
        context_length=32768,
    )

    # Called by QuestHookManager instead of ConstantBudget.compute_budget().
    budget = allocator.compute_budget()   # e.g. 0.23

    # Called when the request finishes.
    allocator.unregister_request("abc")
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile file helpers
# ---------------------------------------------------------------------------

def load_profile(profile_path: str | Path) -> dict[str, Any] | None:
    """Load an offline compute profile from a JSON file.

    Returns the parsed dict, or None with a warning if the file does not
    exist or is malformed.  The caller decides what to do on None.
    """
    path = Path(profile_path)
    if not path.exists():
        logger.warning(
            "Adaptive bandwidth: compute profile not found at %s. "
            "Run experiments/profile_compute.py to generate it. "
            "Falling back to analytical estimate.",
            path,
        )
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded compute profile from %s", path)
        return data
    except Exception as e:
        logger.warning(
            "Adaptive bandwidth: failed to parse profile %s: %s. "
            "Falling back to analytical estimate.",
            path, e,
        )
        return None


def save_profile(
    path: str | Path,
    model_slug: str,
    num_layers: int,
    fit_coefficients: dict[str, float],
    measurements: list[dict],
    extra_info: dict | None = None,
) -> None:
    """Save an offline compute profile to JSON.

    Called by experiments/profile_compute.py after fitting the linear model.

    Args:
        path:              Output file path.
        model_slug:        Short model name for identification (e.g. 'qwen3').
        num_layers:        Number of decoder layers in the model.
        fit_coefficients:  Dict with keys 'a', 'b', 'c' (see module docstring).
        measurements:      Raw measurement rows, each a dict with at least
                           {'new_tokens', 'context_length', 'per_layer_ms_mean'}.
        extra_info:        Optional metadata (GPU name, date, etc.).
    """
    payload: dict[str, Any] = {
        "model": model_slug,
        "num_layers": num_layers,
        # The fitted linear model: t(n,L) = a*n + b*n*L + c  [seconds]
        "fit_coefficients": fit_coefficients,
        # Raw measurements for reference and re-fitting if desired.
        "measurements": measurements,
    }
    if extra_info:
        payload["info"] = extra_info

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved compute profile to %s", path)


def fit_linear_model(
    measurements: list[dict],
) -> dict[str, float]:
    """Fit t(n, L) = a*n + b*n*L + c to measured per-layer times.

    This is a simple ordinary-least-squares fit using numpy.  The three
    basis functions capture the dominant cost terms for a transformer layer:
        - a*n       : FFN cost (scales with query tokens, not KV length)
        - b*n*L     : Attention cost (scales with query × KV token count)
        - c         : Constant overhead (layer norm, residual add, scheduling)

    For MoE models the FFN term is sparse, so 'a' will be smaller than for
    dense models.  The fit handles this automatically from the measurements.

    Args:
        measurements: List of dicts, each with:
            - 'new_tokens'         : int, query length
            - 'context_length'     : int, KV prefix length
            - 'per_layer_ms_mean'  : float, mean measured per-layer time (ms)

    Returns:
        Dict with keys 'a', 'b', 'c' in seconds (not ms).
    """
    if not measurements:
        raise ValueError("Cannot fit model: no measurements provided.")

    # Build the feature matrix and target vector.
    rows = []
    y = []
    for m in measurements:
        n = float(m["new_tokens"])
        L = float(m["context_length"])
        t = float(m["per_layer_ms_mean"]) / 1000.0  # convert ms → seconds
        rows.append([n, n * L, 1.0])                # features: [n, n*L, 1]
        y.append(t)

    X = np.array(rows)
    y_arr = np.array(y)

    # Least-squares fit.  rcond=None suppresses a FutureWarning in numpy.
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y_arr, rcond=None)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    # Log fit quality.
    y_hat = X @ coeffs
    ss_res = float(np.sum((y_arr - y_hat) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    logger.info(
        "Linear model fit: a=%.3e, b=%.3e, c=%.3e  (R²=%.4f, n_pts=%d)",
        a, b, c, r2, len(measurements),
    )

    return {"a": a, "b": b, "c": c}


# ---------------------------------------------------------------------------
# Compute time estimator
# ---------------------------------------------------------------------------

class LayerComputeProfiler:
    """Estimates per-layer GPU compute time for a given (new_tokens, context_length).

    Primary mode: use fit coefficients loaded from an offline profile JSON.
        t(n, L) = a*n + b*n*L + c   [seconds]

    Fallback mode: analytical estimate from model hyperparameters + assumed flops.
        Used only when no profile is available.  Always warns so you know.

    To generate a profile:
        python experiments/profile_compute.py --model qwen3 \\
            --output experiments/compute_profiles/

    Args:
        fit_coefficients:   Dict with 'a', 'b', 'c' in seconds (from offline fit).
                            If None, falls back to analytical estimate.
        num_layers:         Number of decoder layers (used for analytical fallback).
        num_heads:          Number of query attention heads (analytical fallback).
        num_kv_heads:       Number of KV heads (analytical fallback).
        head_dim:           Per-head dimension (analytical fallback).
        hidden_size:        Model hidden size (analytical fallback FFN term).
        gpu_flops_per_sec:  Assumed peak GPU FLOP rate for analytical fallback.
                            Default: 989e12 (H100 SXM bf16 peak).
    """

    def __init__(
        self,
        fit_coefficients: dict[str, float] | None,
        num_layers: int,
        # Analytical fallback parameters — only used if fit_coefficients is None.
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 4096,
        gpu_flops_per_sec: float = 989e12,  # H100 SXM bf16
    ):
        self._fit = fit_coefficients
        self._num_layers = num_layers
        # Analytical fallback params (kept for documentation even if not used).
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._hidden_size = hidden_size
        self._gpu_flops = gpu_flops_per_sec

        if fit_coefficients is None:
            logger.warning(
                "LayerComputeProfiler: no offline profile loaded; "
                "using analytical estimate.  Accuracy may be poor, "
                "especially for MoE models.  Run profile_compute.py."
            )

    @classmethod
    def from_profile(
        cls,
        profile_path: str | Path,
        num_layers: int,
        **kwargs,
    ) -> "LayerComputeProfiler":
        """Construct from an offline profile JSON file.

        If the file does not exist or is malformed, falls back to the
        analytical estimator and logs a warning.
        """
        data = load_profile(profile_path)
        fit = data.get("fit_coefficients") if data else None
        return cls(fit_coefficients=fit, num_layers=num_layers, **kwargs)

    def mean_compute_time(self, num_new_tokens: int, context_length: int) -> float:
        """Return estimated per-layer compute time in seconds.

        Uses the fitted linear model if available, otherwise falls back to
        an analytical estimate.

        Args:
            num_new_tokens:  Number of new (query) tokens being prefilled.
            context_length:  Number of cached KV prefix tokens.

        Returns:
            Estimated compute time per layer in seconds.
        """
        if self._fit is not None:
            return self._fitted(num_new_tokens, context_length)
        return self._analytical(num_new_tokens, context_length)

    def _fitted(self, n: int, L: int) -> float:
        """Evaluate the fitted linear model: t = a*n + b*n*L + c."""
        a = self._fit["a"]
        b = self._fit["b"]
        c = self._fit["c"]
        # Clamp to a small positive value to avoid division-by-zero downstream.
        return max(1e-6, a * n + b * n * L + c)

    def _analytical(self, n: int, L: int) -> float:
        """Rough analytical estimate when no profile is available.

        Models two dominant cost terms per decoder layer:
          1. Attention: 4 * n * (n + L) * num_heads * head_dim  FLOPs
             (Q@K^T and softmax@V for n query tokens attending to n+L KV tokens)
          2. FFN (dense): 8 * n * hidden_size^2 / 4  FLOPs  (SwiGLU, 3 matmuls)
             Note: for MoE this overestimates badly.  Use offline profile instead.

        Both are divided by gpu_flops_per_sec to get seconds.  The estimate
        ignores pipelining, memory-bound regimes at small batch sizes, and MoE
        sparsity — all of which matter in practice.
        """
        attn_flops = 4 * n * (n + L) * self._num_heads * self._head_dim
        # SwiGLU FFN: gate + up projections + down projection ~ 3 matmuls of
        # shape [n, hidden] x [hidden, ffn_dim/2].  For dense models with
        # ffn_dim ≈ 4*hidden this is ≈ 8*n*hidden^2.
        ffn_flops = 8 * n * self._hidden_size * self._hidden_size
        total_flops = attn_flops + ffn_flops
        return max(1e-6, total_flops / self._gpu_flops)


# ---------------------------------------------------------------------------
# Thread-safe demand tracker and budget computer
# ---------------------------------------------------------------------------

class AdaptiveBandwidthAllocator:
    """Computes KV fetch budgets based on available bandwidth vs demand.

    This is the core runtime component.  It tracks active requests, computes
    how much bandwidth each one needs, and returns a budget (fraction of KV
    pages to fetch) that keeps the total transfer within the available link
    capacity.

    The math (for N concurrent requests):
        demand_i      = kv_bytes_per_layer_i / compute_time_per_layer_i
        total_demand  = Σ demand_i
        supply_i      = total_bw * (demand_i / total_demand)   [proportional]
        budget_i      = supply_i / demand_i = total_bw / total_demand

    Note that budget is the SAME for all requests — everyone scales down
    proportionally when bandwidth is scarce.  For a single request:
        budget = min(1.0, bandwidth_bps / demand)

    This object's compute_budget() method has the same signature as the
    ConstantBudget and DynamicBudgetComputer classes, so it can be used as a
    drop-in replacement for the budget_computer argument of QuestHookManager.

    Thread safety:
        _demands is protected by _lock.  compute_budget() acquires the lock
        briefly to sum demands (fast dict traversal, not a blocking operation).

    Remote storage extensibility:
        For local CPU offloading: compute_bandwidth == storage_bandwidth.
        For remote storage: the storage node computes its available bandwidth
        and calls override_storage_bandwidth() with the returned value.  The
        budget formula min(compute_bw, storage_bw) / demand automatically
        tightens when either side is constrained.  No other changes needed.

    Args:
        profiler:               LayerComputeProfiler for per-layer timing.
        compute_bandwidth_bps:  Available PCIe/NVLink bandwidth in bytes/sec.
                                For PCIe 4.0 x16: ~32 GB/s unidirectional.
                                For NVMe (sequential): ~7 GB/s typical.
        storage_bandwidth_bps:  Storage-side bandwidth limit in bytes/sec.
                                None → same as compute_bandwidth_bps.
        min_budget:             Floor on the returned budget.  Prevents
                                fetching nothing even when bandwidth is very
                                scarce.  Default: 0.01 (1%).
    """

    def __init__(
        self,
        profiler: LayerComputeProfiler,
        compute_bandwidth_bps: float,
        storage_bandwidth_bps: float | None = None,
        min_budget: float = 0.01,
    ):
        self.profiler = profiler
        self._compute_bw = compute_bandwidth_bps
        # Default storage bandwidth to compute bandwidth (same node).
        # Override via override_storage_bandwidth() for remote storage.
        self._storage_bw = (
            storage_bandwidth_bps
            if storage_bandwidth_bps is not None
            else compute_bandwidth_bps
        )
        self._min_budget = min_budget

        # req_id -> demand in bytes/sec.  Protected by _lock.
        self._demands: dict[str, float] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_profile(
        cls,
        profile_path: str | Path,
        compute_bandwidth_gbps: float,
        storage_bandwidth_gbps: float | None = None,
        min_budget: float = 0.01,
        # Analytical fallback params (forwarded to LayerComputeProfiler).
        num_layers: int = 32,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        hidden_size: int = 4096,
    ) -> "AdaptiveBandwidthAllocator":
        """Convenience constructor: load profile, build profiler, return allocator.

        Typical usage in QuestOffloadingSpec.get_adaptive_allocator():
            allocator = AdaptiveBandwidthAllocator.from_profile(
                profile_path=extra_config["compute_profile_path"],
                compute_bandwidth_gbps=extra_config.get("compute_bandwidth_gbps", 32.0),
                num_layers=hf_config.num_hidden_layers,
                ...
            )
        """
        profiler = LayerComputeProfiler.from_profile(
            profile_path=profile_path,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
        )
        return cls(
            profiler=profiler,
            compute_bandwidth_bps=compute_bandwidth_gbps * 1e9,
            storage_bandwidth_bps=(
                storage_bandwidth_gbps * 1e9
                if storage_bandwidth_gbps is not None
                else None
            ),
            min_budget=min_budget,
        )

    # ------------------------------------------------------------------
    # Request lifecycle — called by OffloadingConnectorWorker
    # ------------------------------------------------------------------

    def register_request(
        self,
        req_id: str,
        kv_bytes_per_layer: int,
        num_new_tokens: int,
        context_length: int,
    ) -> None:
        """Register a new request and record its bandwidth demand.

        Call this when a load transfer starts (start_kv_transfers).

        The demand is kv_bytes_per_layer / compute_time_per_layer, i.e.,
        the bandwidth rate (bytes/sec) that would be needed to transfer
        exactly one layer's worth of KV data during one compute step.

        Args:
            req_id:              Unique request identifier.
            kv_bytes_per_layer:  Total KV bytes for this request's prefix,
                                 for a single layer.  Compute as:
                                 num_prefix_tokens * num_kv_heads * head_dim * 2 * dtype_bytes
                                 (factor of 2 for K and V).
            num_new_tokens:      Number of new query tokens being prefilled.
            context_length:      Number of cached KV prefix tokens.
        """
        compute_time_s = self.profiler.mean_compute_time(num_new_tokens, context_length)
        # demand: how fast we need to read KV data to fill the compute window
        demand = kv_bytes_per_layer / compute_time_s
        with self._lock:
            self._demands[req_id] = demand
        logger.debug(
            "Registered req %s: kv_bytes/layer=%d, compute=%.2fms, demand=%.2f GB/s",
            req_id, kv_bytes_per_layer, compute_time_s * 1e3, demand / 1e9,
        )

    def unregister_request(self, req_id: str) -> None:
        """Remove a finished request from demand tracking.

        Call this when the request completes (get_finished).
        Safe to call on an already-unregistered req_id (no-op).
        """
        with self._lock:
            self._demands.pop(req_id, None)

    # ------------------------------------------------------------------
    # Budget computation — called by QuestHookManager (drop-in for ConstantBudget)
    # ------------------------------------------------------------------

    def compute_budget(
        self,
        approximate_scores: "torch.Tensor | None" = None,
        layer_idx: int | None = None,
        num_layers: int | None = None,
        **kwargs,
    ) -> float:
        """Return the KV fetch budget for the current set of active requests.

        This method has the same signature as ConstantBudget.compute_budget()
        and DynamicBudgetComputer.compute_budget(), so it can be passed directly
        as the budget_computer argument to QuestHookManager.  Unused arguments
        (approximate_scores, layer_idx, num_layers) are accepted for
        compatibility but ignored.

        Returns:
            A float in [min_budget, 1.0] representing the fraction of KV
            pages to fetch.  1.0 means transfer everything (bandwidth is
            sufficient); lower values mean selective fetching.
        """
        with self._lock:
            total_demand = sum(self._demands.values())

        if total_demand <= 0.0:
            # No active requests with measured demand — full transfer.
            return 1.0

        # Effective bandwidth = min of compute-side and storage-side limits.
        # For local CPU: both are the same, so this is just the NVMe/PCIe BW.
        # For remote storage: storage_bw may be overridden to a lower value.
        effective_bw = min(self._compute_bw, self._storage_bw)

        # All requests share the same budget fraction when bandwidth is scarce
        # (proportional allocation simplifies to total_bw / total_demand).
        raw = effective_bw / total_demand
        budget = max(self._min_budget, min(1.0, raw))
        logger.debug(
            "Budget: effective_bw=%.1f GB/s, total_demand=%.1f GB/s → %.3f",
            effective_bw / 1e9, total_demand / 1e9, budget,
        )
        return budget

    # ------------------------------------------------------------------
    # Remote storage hook
    # ------------------------------------------------------------------

    def override_storage_bandwidth(self, storage_bandwidth_bps: float) -> None:
        """Tighten the budget based on storage-side bandwidth constraints.

        For remote storage nodes: the storage node computes its own available
        bandwidth (total_bw / concurrent_requests) and includes it in the
        transfer response.  The response handler calls this method to update
        the storage-side limit.

        The next call to compute_budget() will use the tighter value
        automatically.  No other changes are needed.

        Args:
            storage_bandwidth_bps:  Storage node's available bandwidth in bytes/sec.
        """
        # No lock needed: float write is atomic on CPython and the value is
        # only read in compute_budget() which already tolerates stale values.
        self._storage_bw = storage_bandwidth_bps
        logger.debug(
            "Storage bandwidth updated to %.2f GB/s", storage_bandwidth_bps / 1e9
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_kv_bytes_per_layer(
    num_prefix_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,  # bf16 / fp16
) -> int:
    """Compute total KV bytes for one layer given a prefix of length L.

    Both K and V are counted (factor of 2).  This is the amount of data
    that must be transferred from CPU/storage to GPU for one layer.

    Args:
        num_prefix_tokens:  Number of cached prefix tokens (= context_length).
        num_kv_heads:       Number of key-value attention heads.
        head_dim:           Per-head feature dimension.
        dtype_bytes:        Bytes per element (2 for bf16/fp16, 4 for fp32).

    Returns:
        Total bytes as an integer.
    """
    # K: [num_prefix_tokens, num_kv_heads, head_dim]
    # V: [num_prefix_tokens, num_kv_heads, head_dim]
    return num_prefix_tokens * num_kv_heads * head_dim * 2 * dtype_bytes
