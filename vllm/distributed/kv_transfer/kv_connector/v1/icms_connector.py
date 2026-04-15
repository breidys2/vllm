# SPDX-License-Identifier: Apache-2.0
"""ICMS storage service KV connector for vLLM v1.

Standalone connector (does NOT inherit from OffloadingConnector) that routes
Quest scoring and KV storage through a unix-socket icms_client talking to a
separate `icms_server` process. See:

  - docs/icms_storage_service_design.md
  - docs/icms_connector_integration_walkthrough.md  (C1–C10 locked designs)
  - docs/icms_v1_relaxations.md

Design decisions (from the walkthrough):
  C1  — block_hashes read once at update_state_after_alloc; per-step
        num_computed_tokens plumbed via IcmsConnectorMetadata.
  C2  — WriteGroup hook at _quest_update_metadata_after_store equivalent;
        per-(req, group) buffer flushed on completion.
  C3  — batch=1 only; per-request dispatch for batch>1 later.
  C4  — KV stored as opaque bytes (K||V concat per page); summaries separate.
  C5  — TP=1 only; NotImplementedError for TP>1.
  C6  — Standalone; zero modifications to offloading_connector.py.
  C7  — quest_benchmark.py --use-icms spawns server.
  C8  — Single pre-allocated sink, N=4 slots.
  C9  — Three-state pipelining: cache-hit → immediate fetch;
        cache-miss + preloaded → SIMD only; cold fallback.
  C10 — Dynamic module path registration; no factory.py edit.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

logger = init_logger(__name__)

# ─── icms_client import ──────────────────────────────────────────────────
def _ensure_icms_client_on_path():
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        cand = ancestor / "storage_service" / "python"
        if cand.is_dir():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return
    raise ImportError("icms_client not found")

_ensure_icms_client_on_path()

from icms_client import IcmsClient                          # noqa: E402
from icms_client.geometry import (                           # noqa: E402
    GROUP_PAGES, PAGE_TOKENS, ModelGeometry, find_model,
)
from icms_client.sink import Sink, allocate_sink             # noqa: E402

# ─── constants ───────────────────────────────────────────────────────────
_SINK_SLOTS = 4           # C8: number of pre-allocated sink slots
_GROUP_BLOCKS = GROUP_PAGES  # blocks per group (= pages per group = 32)

# Module-level queue: worker → scheduler stored-chain notifications.
# The worker appends (chain, num_groups) after WriteGroup completes;
# the scheduler drains it in get_num_new_matched_tokens / build_meta.
# Safe because both run in the same EngineCore process.
_stored_chain_queue: list[tuple[list[int], int]] = []


# ═══════════════════════════════════════════════════════════════════════════
#  Metadata: scheduler → worker per-step payload (C1)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _PerRequestStep:
    """Per-request data packed by the scheduler each step."""
    request_id: str
    num_computed_tokens_start: int   # at beginning of this step
    num_computed_tokens_end: int     # after this step completes


@dataclass
class IcmsConnectorMetadata(KVConnectorMetadata):
    """Carried from scheduler to worker each step via bind_connector_metadata."""
    requests: list[_PerRequestStep] = field(default_factory=list)
    # First-time chain deliveries: req_id → group_hashes_chain.
    # Only populated the first time a request appears (C1).
    new_chains: dict[str, list[int]] = field(default_factory=dict)
    # Per-request in-order CPU block IDs (for block_id → intra_request_idx mapping).
    block_id_maps: dict[str, list[int]] = field(default_factory=dict)
    # Worker→scheduler: chains that were stored in ICMS.
    # List of (chain, num_groups) tuples, drained by the scheduler.
    stored_chain_notifications: list[tuple[list[int], int]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-(request, group) accumulator buffer (C2)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _GroupBuffer:
    """Accumulates per-page KV + summary bytes until a full group is ready."""
    summary_blob: bytearray   # [num_layers × GROUP_PAGES × summary_page_bytes]
    kv_blob: bytearray        # [num_layers × GROUP_PAGES × kv_page_bytes]
    # Track which (layer, page) slots have been filled.
    filled: set[tuple[int, int]]   # {(layer_idx, page_in_group), ...}
    num_layers: int
    pages_in_group: int       # target fill (GROUP_PAGES for full, less for partial)

    def is_complete(self) -> bool:
        return len(self.filled) >= self.num_layers * self.pages_in_group


# ═══════════════════════════════════════════════════════════════════════════
#  Timing / debug statistics
# ═══════════════════════════════════════════════════════════════════════════

class IcmsTimingStats:
    """Accumulated timing and event stats for the icms connector.

    Populated by the worker during forward passes. Can be queried by the
    benchmark script via ``IcmsConnector.get_timing_stats()``.

    Stats levels:
      0 = off (no collection, zero overhead)
      1 = aggregates only (counters + totals, negligible overhead)
      2 = per-call detail (per-layer latency lists, page_ids — adds
          perf_counter calls on every Score/WriteGroup/extract)
    """

    def __init__(self, level: int = 1, log_selections: bool = False):
        self.level = level
        self.log_selections = log_selections

        # ── Level 1: aggregate counters ──────────────────────────────
        self.requests_seen: int = 0
        self.total_groups_written: int = 0
        self.total_pages_recorded: int = 0
        self.total_score_calls: int = 0
        self.total_score_cache_hits: int = 0
        self.total_writegroup_calls: int = 0
        self.total_extract_us: float = 0.0
        self.total_flush_us: float = 0.0
        self.total_score_us: float = 0.0
        self.peak_buffer_bytes: int = 0

        # ── Level 2: per-call detail lists ───────────────────────────
        self.extract_and_record_us: list[float] = []
        self.flush_group_us: list[float] = []
        self.score_roundtrip_us: list[float] = []
        self.score_cache_hit: list[bool] = []
        self.score_page_ids: list[list[int]] = []
        self.score_layer_idx: list[int] = []

        # ── Per-layer TTFT breakdown (level 2) ────────────────────────
        # Per-layer timing for the selective attention path.
        # Each entry: {layer, score_us, fetch_us, modify_us, total_us, k_pages}
        self.layer_breakdown: list[dict] = []

        # ── Selection log (for A/B comparison) ───────────────────────
        self.selections: list[dict] = []
        self._step_counter: int = 0

    def record_extract(self, us: float, n_pages: int):
        if self.level == 0:
            return
        self.total_extract_us += us
        self.total_pages_recorded += n_pages
        if self.level >= 2:
            self.extract_and_record_us.append(us)

    def record_flush(self, us: float):
        if self.level == 0:
            return
        self.total_flush_us += us
        self.total_writegroup_calls += 1
        if self.level >= 2:
            self.flush_group_us.append(us)

    def record_score(self, us: float, cache_hit: bool, page_ids: list[int],
                      layer_idx: int, scores: list[float] | None = None):
        if self.level == 0 and not self.log_selections:
            return
        if self.level >= 1:
            self.total_score_us += us
            self.total_score_calls += 1
            if cache_hit:
                self.total_score_cache_hits += 1
        if self.level >= 2:
            self.score_roundtrip_us.append(us)
            self.score_cache_hit.append(cache_hit)
            self.score_page_ids.append(page_ids)
            self.score_layer_idx.append(layer_idx)
        if self.log_selections and page_ids:
            self.selections.append({
                "step": self._step_counter,
                "layer": layer_idx,
                "page_ids": page_ids,
                "scores": scores or [],
                "cache_hit": cache_hit,
                "k": len(page_ids),
            })

    def record_layer_breakdown(self, layer: int, score_us: float,
                               fetch_us: float, modify_us: float,
                               total_us: float, k_pages: int):
        if self.level < 2:
            return
        self.layer_breakdown.append({
            "layer": layer,
            "score_us": round(score_us, 1),
            "fetch_us": round(fetch_us, 1),
            "modify_us": round(modify_us, 1),
            "total_us": round(total_us, 1),
            "k_pages": k_pages,
        })

    def advance_step(self):
        """Call at the start of each forward pass to increment the step counter."""
        self._step_counter += 1

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        if self.level == 0:
            return {"level": 0}

        def _percentiles(vals):
            if not vals:
                return {}
            s = sorted(vals)
            n = len(s)
            return {
                "count": n,
                "p50_us": s[n // 2],
                "p95_us": s[int(n * 0.95)] if n >= 20 else s[-1],
                "p99_us": s[int(n * 0.99)] if n >= 100 else s[-1],
                "mean_us": sum(s) / n,
                "max_us": s[-1],
                "total_us": sum(s),
            }

        result: dict = {
            "level": self.level,
            "requests_seen": self.requests_seen,
            "total_groups_written": self.total_groups_written,
            "total_pages_recorded": self.total_pages_recorded,
            "total_score_calls": self.total_score_calls,
            "total_score_cache_hits": self.total_score_cache_hits,
            "total_writegroup_calls": self.total_writegroup_calls,
            "total_extract_us": self.total_extract_us,
            "total_flush_us": self.total_flush_us,
            "total_score_us": self.total_score_us,
            "score_cache_hit_rate": (
                self.total_score_cache_hits / self.total_score_calls
                if self.total_score_calls > 0 else 0.0
            ),
            "peak_buffer_bytes": self.peak_buffer_bytes,
        }
        if self.level >= 2:
            result["extract_and_record"] = _percentiles(self.extract_and_record_us)
            result["flush_group"] = _percentiles(self.flush_group_us)
            result["score_roundtrip"] = _percentiles(self.score_roundtrip_us)
            result["layer_breakdown"] = self.layer_breakdown
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  Per-request worker-side state
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _RequestState:
    request_id: str
    chain: list[int] = field(default_factory=list)       # group hashes (C1)
    block_ids: list[int] = field(default_factory=list)    # in-order CPU block IDs
    num_groups_written: int = 0                            # groups fully flushed to icms
    active_group_buffers: dict[int, _GroupBuffer] = field(default_factory=dict)  # group_idx → buf


# ═══════════════════════════════════════════════════════════════════════════
#  Sink slot allocator (C8)
# ═══════════════════════════════════════════════════════════════════════════

class _SinkSlotPool:
    """Pre-allocated fixed-size sink partitioned into N slots."""

    def __init__(self, sink: Sink, slot_bytes: int, n_slots: int):
        self.sink = sink
        self.slot_bytes = slot_bytes
        self.n_slots = n_slots
        self._free: list[int] = list(range(n_slots))
        self._lock = threading.Lock()

    def acquire(self) -> int:
        """Get a free slot index. Blocks if none available."""
        while True:
            with self._lock:
                if self._free:
                    return self._free.pop()
            # Spin briefly — with N=4 and at most 2 in flight this rarely triggers.
            threading.Event().wait(0.0001)

    def release(self, slot: int):
        with self._lock:
            self._free.append(slot)

    def offset_for_slot(self, slot: int) -> int:
        return slot * self.slot_bytes


# ═══════════════════════════════════════════════════════════════════════════
#  Facade
# ═══════════════════════════════════════════════════════════════════════════

class IcmsConnector(KVConnectorBase_V1):
    """Standalone vLLM v1 KV connector for the ICMS storage service.

    Does NOT inherit from OffloadingConnector (C6). Implements all abstract
    methods of KVConnectorBase_V1. Dispatches to _Scheduler / _Worker based
    on role.
    """

    def __init__(
        self,
        vllm_config,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        # C5: TP=1 gate.
        tp = getattr(getattr(vllm_config, "parallel_config", None),
                      "tensor_parallel_size", 1)
        if tp > 1:
            raise NotImplementedError(
                f"IcmsConnector requires TP=1 in v1 (got TP={tp}). "
                "See docs/icms_connector_integration_walkthrough.md C5."
            )

        extra = (vllm_config.kv_transfer_config.kv_connector_extra_config or {})
        if isinstance(extra, str):
            import json
            extra = json.loads(extra)
        self._socket_path: str = extra.get("icms_socket_path", "/tmp/icms.sock")
        self._model_name: str  = extra.get("icms_model_name", "")
        self._k: int           = int(extra.get("icms_k", 16))
        self._budget: float    = float(extra.get("icms_budget", 0.2))
        self._stats_level: int = int(extra.get("icms_stats_level", 1))
        self._log_selections: bool = bool(extra.get("icms_log_selections", False))
        self._score_stride: int = int(extra.get("icms_score_stride", 6))
        self._adaptive_bandwidth: bool = bool(extra.get("adaptive_bandwidth", False))
        self._link_bandwidth_bps: float = float(extra.get(
            "link_bandwidth_bps", 25e9 / 8))  # default 25 Gbps (BF2)
        self._compute_slack_table: str = extra.get("compute_slack_table", "")

        self._sched: _Scheduler | None = None
        self._worker: _Worker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self._sched = _Scheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            # Extract num_q_heads from model config for FA3 scheduling.
            num_q_heads = getattr(
                getattr(vllm_config, "model_config", None),
                "get_num_attention_heads", lambda _: 32)(
                getattr(vllm_config, "parallel_config", None))
            self._worker = _Worker(
                socket_path=self._socket_path,
                model_name=self._model_name,
                k=self._k,
                score_stride=self._score_stride,
                num_q_heads=num_q_heads,
                stats_level=self._stats_level,
                log_selections=self._log_selections,
                adaptive_bandwidth=self._adaptive_bandwidth,
                link_bandwidth_bps=self._link_bandwidth_bps,
                compute_slack_table=self._compute_slack_table,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  Worker-side abstract methods
    # ══════════════════════════════════════════════════════════════════════

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Stash GPU KV cache tensors for Path B selective fetch."""
        if self._worker is not None:
            self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        """Stash attn_metadata from the forward context and drain metadata."""
        if self._worker is None:
            return
        # Stash the per-layer attn_metadata dict so wait_for_layer_load
        # can override block_table + seq_lens for selective attention.
        if hasattr(forward_context, "attn_metadata"):
            self._worker.set_attn_metadata(forward_context.attn_metadata)
        meta = self._connector_metadata
        if meta is None or not isinstance(meta, IcmsConnectorMetadata):
            return
        self._worker.on_step_start(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Path B: fetch selected KV from icms into GPU + override block table.

        This runs BEFORE each layer's attention. For layers where the
        connector has Score results (from the previous layer's Quest hook),
        it fetches the selected pages' KV from the icms sink into GPU
        block slots and overrides the block_table and seq_lens so
        FlashAttention only sees the k selected pages.
        """
        if self._worker is None:
            return
        self._worker.wait_for_layer(layer_name)

    def save_kv_layer(  # noqa: C901
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs,
    ) -> None:
        """Called per-layer from the Quest hook with quest_query / quest_selected_pages.

        Routes to the icms write + score path:
        - If quest_query is present: fire async Score on icms (C9 State 2/3).
        - If quest_selected_pages is present: record GPU-side selection (not
          used in the icms path, but logged for A/B comparison).
        - If quest_all_pages=True: budget >= 1.0, transfer all pages.
        """
        if self._worker is None:
            return

        # Path B: restore original block_table + seq_lens after this layer's
        # attention ran with the overridden values. Must happen before the
        # Quest hook runs (which will set up overrides for the NEXT layer).
        if self._worker is not None:
            self._worker.restore_attn_metadata(layer_name)

        quest_query = kwargs.get("quest_query")
        next_layer_idx = kwargs.get("next_layer_idx")
        budget = kwargs.get("budget", self._budget)
        quest_stats = kwargs.get("quest_stats")
        quest_all_pages = kwargs.get("quest_all_pages", False)
        quest_reuse_selection = kwargs.get("quest_reuse_selection", False)

        # Standard save_kv_layer: extract per-block K summaries and populate
        # the icms group buffers (C2 write path). Only when kv_layer has
        # actual data (not from Quest hooks which pass empty tensors).
        is_quest_call = (quest_query is not None
                         or quest_all_pages
                         or quest_reuse_selection)
        if (not is_quest_call
                and self._worker is not None
                and kv_layer is not None
                and kv_layer.numel() > 0
                and attn_metadata is not None):
            self._worker.extract_and_record(layer_name, kv_layer, attn_metadata)

        if quest_all_pages:
            # Budget >= 1.0: transfer all pages. For icms this means "no
            # sparse selection needed" — skip scoring entirely.
            self._worker.on_layer_all_pages(next_layer_idx, budget, quest_stats)
            return

        if quest_reuse_selection:
            # Single-layer-scoring reuse: the connector reuses previous
            # selection. For icms this is a cache-hit Score (C9 State 1).
            self._worker.on_layer_reuse(next_layer_idx, budget, quest_stats)
            return

        if quest_query is not None and next_layer_idx is not None:
            # With the V2 model runner, the standard save_kv_layer (with
            # kv_layer data) doesn't fire — only Quest hooks do. So we
            # extract K/V from the stashed GPU KV caches here instead.
            if (self._worker is not None
                    and not self._worker._prefill_done
                    and self._worker._gpu_kv_caches
                    and self._worker._attn_metadata is not None):
                # Map quest hook's layer_name ("layer_N") to attention
                # layer name ("model.layers.N.self_attn.attn") for the
                # CURRENT layer (not next_layer_idx — we extract K/V
                # for the layer that just completed, using its index
                # from the layer_name).
                parts = layer_name.split("_")
                if len(parts) == 2 and parts[1].isdigit():
                    cur_layer = int(parts[1])
                    attn_name = f"model.layers.{cur_layer}.self_attn.attn"
                    kv = self._worker._gpu_kv_caches.get(attn_name)
                    am = self._worker._attn_metadata.get(attn_name) if isinstance(self._worker._attn_metadata, dict) else None
                    if kv is not None and kv.ndim == 5 and am is not None:
                        self._worker.extract_and_record(attn_name, kv, am)

            if not self._worker._prefill_done:
                self._worker.on_layer_score(
                    next_layer_idx, quest_query, budget, quest_stats,
                    connector_meta=self._connector_metadata,
                )

    def wait_for_save(self) -> None:
        if self._worker is not None:
            self._worker.wait_for_pending_writes()

    def get_finished(
        self, finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        # Clean up worker-side request state for finished requests.
        if self._worker is not None and finished_req_ids:
            for rid in finished_req_ids:
                self._worker.on_request_finished(rid)
        return None, None

    def shutdown(self):
        if self._worker is not None:
            self._worker.shutdown()

    # ══════════════════════════════════════════════════════════════════════
    #  Scheduler-side abstract methods
    # ══════════════════════════════════════════════════════════════════════

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self._sched is not None:
            return self._sched.get_num_new_matched_tokens(
                request, num_computed_tokens)
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int,
    ) -> None:
        """C1: read Request.block_hashes on first alloc, derive group chain."""
        if self._sched is not None:
            self._sched.on_alloc(request, blocks)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self._sched is not None:
            return self._sched.build_meta(scheduler_output)
        return IcmsConnectorMetadata()

    def request_finished(
        self, request: Request, block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._sched is not None:
            self._sched.on_finished(request.request_id)
        if self._worker is not None:
            self._worker.on_request_finished(request.request_id)
        return False, None

    # ══════════════════════════════════════════════════════════════════════
    #  Direct helper API (used by smoke tests, compatible with the old
    #  skeleton — kept for backward compat with existing tests).
    # ══════════════════════════════════════════════════════════════════════

    def write_group_for_request(
        self, request_id: str, chain: list[int],
        summary_blob: bytes, kv_blob: bytes,
    ):
        if self._worker is None:
            raise RuntimeError("worker-side only")
        return self._worker.direct_write_group(request_id, chain, summary_blob, kv_blob)

    def score_for_request(
        self, request_id: str, chain: list[int],
        layer: int, query: torch.Tensor, k: int | None = None,
    ):
        if self._worker is None:
            raise RuntimeError("worker-side only")
        return self._worker.direct_score(request_id, chain, layer, query, k or self._k)

    def get_timing_stats(self) -> dict | None:
        """Return accumulated timing/debug stats from the worker.

        Call from the benchmark script BEFORE shutdown to capture stats.
        Returns a JSON-serializable dict, or None if no worker.
        """
        if self._worker is not None:
            return self._worker.stats.to_dict()
        return None

    def get_icms_server_stats(self) -> dict | None:
        """Query the icms_server's Stats reply and return as a dict.

        Call from the benchmark script BEFORE server teardown.
        """
        if self._worker is None or self._worker._client is None:
            return None
        try:
            s = self._worker._client.stats()
            return {
                "trie_num_nodes": s.trie_num_nodes,
                "trie_num_inserts": s.trie_num_inserts,
                "trie_total_groups": s.trie_total_groups,
                "trie_max_depth": s.trie_max_depth,
                "trie_mean_path_len": s.trie_mean_path_len,
                "trie_leaf_count": s.trie_leaf_count,
                "alloc_capacity_bytes": s.alloc_capacity_bytes,
                "alloc_free_bytes": s.alloc_free_bytes,
                "alloc_used_bytes": s.alloc_used_bytes,
                "alloc_num_chunks": s.alloc_num_chunks,
                "alloc_total_extents": s.alloc_total_extents,
                "score_cache_size": s.score_cache_size,
                "score_cache_hits": s.score_cache_hits,
                "score_cache_misses": s.score_cache_misses,
            }
        except Exception as e:
            logger.warning("Failed to query icms server stats: %s", e)
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  Scheduler
# ═══════════════════════════════════════════════════════════════════════════

class _Scheduler:
    """Scheduler-side state. Reads block_hashes from Request, packs metadata."""

    def __init__(self, vllm_config):
        self._vllm_config = vllm_config
        # Per-request: group hashes chain (C1). Populated on first alloc.
        self._chains: dict[str, list[int]] = {}
        # Per-request: in-order CPU block IDs.
        self._block_ids: dict[str, list[int]] = {}
        # Requests whose chains haven't been sent to the worker yet.
        self._pending_chain_sends: set[str] = set()

        # ── Global prefix index (Option 1: in-process hash dict) ──
        # Maps chain prefix (as tuple of group hashes) → num_groups stored.
        # Updated via metadata from the worker after WriteGroup completes.
        # Used by get_num_new_matched_tokens to report externally-stored
        # prefix length to the scheduler, so vLLM skips recomputing them.
        self._stored_chains: list[tuple[tuple[int, ...], int]] = []

    def on_alloc(self, request: Request, blocks: KVCacheBlocks):
        rid = request.request_id
        if rid in self._chains:
            return  # already seen

        # Compute chain from token IDs directly using SHA-256.
        # vLLM's block_hashes include engine_id and differ across processes,
        # so we bypass them and hash tokens directly for cross-process
        # prefix matching.
        import hashlib
        token_ids = getattr(request, "all_token_ids", None)
        if token_ids is None or len(token_ids) == 0:
            token_ids = getattr(request, "prompt_token_ids", [])

        group_tokens = _GROUP_BLOCKS * PAGE_TOKENS  # 512 tokens per group
        chain: list[int] = []
        if token_ids and len(token_ids) >= PAGE_TOKENS:
            for g in range(64):
                end = min((g + 1) * group_tokens, len(token_ids))
                start = g * group_tokens
                if end <= start:
                    break
                # Chained: hash all tokens up to this group boundary.
                h = hashlib.sha256(
                    repr(tuple(token_ids[:end])).encode()
                ).digest()[:8]
                chain.append(int.from_bytes(h, "little"))
        if not chain:
            h = hashlib.sha256(repr(tuple(token_ids or [])).encode()).digest()[:8]
            chain.append(int.from_bytes(h, "little"))

        logger.debug("on_alloc: rid=%s tokens=%d groups=%d",
                      rid, len(token_ids or []), len(chain))
        self._chains[rid] = chain
        self._pending_chain_sends.add(rid)

    def build_meta(self, scheduler_output: SchedulerOutput) -> IcmsConnectorMetadata:
        # Also drain worker notifications here (belt-and-suspenders with
        # the drain in get_num_new_matched_tokens) to catch pushes that
        # arrived between scheduling and metadata building.
        global _stored_chain_queue
        if _stored_chain_queue:
            for chain, n_groups in _stored_chain_queue:
                self.record_stored_chain(chain, n_groups)
            _stored_chain_queue.clear()

        meta = IcmsConnectorMetadata()
        # Per-step scheduled token counts (req_id → num_tokens this step).
        sched_tokens = getattr(scheduler_output, "scheduled_num_tokens", {})

        # New requests: list[NewRequestData] with .req_id, .num_computed_tokens.
        for sr in getattr(scheduler_output, "scheduled_new_reqs", []):
            rid = sr.req_id
            n_computed = getattr(sr, "num_computed_tokens", 0)
            n_step = sched_tokens.get(rid, 0)
            meta.requests.append(_PerRequestStep(
                request_id=rid,
                num_computed_tokens_start=n_computed,
                num_computed_tokens_end=n_computed + n_step,
            ))

        # Cached requests: CachedRequestData with parallel lists
        # (.req_ids, .num_computed_tokens, etc.).
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is not None and hasattr(cached, "req_ids"):
            for i, rid in enumerate(cached.req_ids):
                n_computed = cached.num_computed_tokens[i] if i < len(cached.num_computed_tokens) else 0
                n_step = sched_tokens.get(rid, 0)
                meta.requests.append(_PerRequestStep(
                    request_id=rid,
                    num_computed_tokens_start=n_computed,
                    num_computed_tokens_end=n_computed + n_step,
                ))

        # Deliver chains for requests we haven't sent yet.
        for rid in list(self._pending_chain_sends):
            if rid in self._chains:
                meta.new_chains[rid] = self._chains[rid]
            self._pending_chain_sends.discard(rid)
        return meta

    # ── Global prefix index operations ──────────────────────────────

    def record_stored_chain(self, chain: list[int], num_groups: int):
        """Called when the worker reports that groups were written to ICMS."""
        key = tuple(chain[:num_groups])
        # Update existing or append.
        for i, (sc, ng) in enumerate(self._stored_chains):
            if sc == key:
                self._stored_chains[i] = (key, max(ng, num_groups))
                return
        self._stored_chains.append((key, num_groups))

    def _lookup_stored_prefix(self, chain: list[int]) -> int:
        """Find longest stored chain prefix. Returns matched group count."""
        best = 0
        for stored_chain, n_groups in self._stored_chains:
            match_len = 0
            for a, b in zip(chain, stored_chain):
                if a == b:
                    match_len += 1
                else:
                    break
            if match_len > 0:
                best = max(best, min(n_groups, match_len))
        return best

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Report how many tokens beyond local cache are stored in ICMS.

        The scheduler calls this once per new request. We compute the
        chain from token IDs and check our in-process prefix index.
        Returns (num_external_tokens, is_async).
        """
        # Drain worker → scheduler notifications first.
        global _stored_chain_queue
        if _stored_chain_queue:
            for chain, n_groups in _stored_chain_queue:
                self.record_stored_chain(chain, n_groups)
            _stored_chain_queue.clear()

        rid = request.request_id
        # Ensure chain is computed.
        if rid not in self._chains:
            # Compute chain eagerly (before on_alloc which needs blocks).
            import hashlib
            token_ids = getattr(request, "all_token_ids", None)
            if token_ids is None or len(token_ids) == 0:
                token_ids = getattr(request, "prompt_token_ids", [])
            group_tokens = _GROUP_BLOCKS * PAGE_TOKENS
            chain: list[int] = []
            if token_ids and len(token_ids) >= PAGE_TOKENS:
                for g in range(64):
                    end = min((g + 1) * group_tokens, len(token_ids))
                    start = g * group_tokens
                    if end <= start:
                        break
                    h = hashlib.sha256(
                        repr(tuple(token_ids[:end])).encode()
                    ).digest()[:8]
                    chain.append(int.from_bytes(h, "little"))
            if not chain:
                return 0, False
            self._chains[rid] = chain
            self._pending_chain_sends.add(rid)

        chain = self._chains[rid]
        matched_groups = self._lookup_stored_prefix(chain)
        logger.debug(
            "prefix_lookup: rid=%s chain_len=%d stored=%d matched=%d",
            rid, len(chain), len(self._stored_chains), matched_groups,
        )
        if matched_groups == 0:
            return 0, False

        # Convert groups to tokens. Each group = GROUP_PAGES * PAGE_TOKENS.
        matched_tokens = matched_groups * _GROUP_BLOCKS * PAGE_TOKENS
        # Subtract what's already locally computed.
        ext_tokens = max(0, matched_tokens - num_computed_tokens)
        if ext_tokens == 0:
            return 0, False

        # Don't claim more external tokens than the prompt has.  Leave at
        # least PAGE_TOKENS new tokens for the model to actually compute
        # (otherwise prompt_logprobs would be empty for the continuation).
        total_prompt = len(token_ids) if token_ids else 0
        max_ext = max(0, total_prompt - num_computed_tokens - PAGE_TOKENS)
        ext_tokens = min(ext_tokens, max_ext)
        if ext_tokens <= 0:
            return 0, False

        logger.debug(
            "get_num_new_matched_tokens: rid=%s matched_groups=%d "
            "matched_tokens=%d local=%d ext=%d",
            rid, matched_groups, matched_tokens, num_computed_tokens, ext_tokens,
        )
        # Synchronous loading (False): the worker fills blocks during
        # the forward pass via wait_for_layer_load.
        return ext_tokens, False

    def on_finished(self, request_id: str):
        self._chains.pop(request_id, None)
        self._block_ids.pop(request_id, None)
        self._pending_chain_sends.discard(request_id)


# ═══════════════════════════════════════════════════════════════════════════
#  Worker
# ═══════════════════════════════════════════════════════════════════════════

class _Worker:
    """Worker-side state. Owns IcmsClient, sink pool, per-request buffers."""

    def __init__(self, *, socket_path: str, model_name: str, k: int,
                 score_stride: int = 6, num_q_heads: int = 32,
                 stats_level: int = 1, log_selections: bool = False,
                 adaptive_bandwidth: bool = False,
                 link_bandwidth_bps: float = 25e9 / 8,
                 compute_slack_table: str = ""):
        self._socket_path = socket_path
        self._model_name = model_name
        self._k = k
        self._num_q_heads = num_q_heads
        # Score stride: fresh Quest scoring every N layers.
        # stride=1 → per-layer scoring, stride=6 → strided, stride=48 → single
        self._score_stride = max(1, score_stride)

        # Adaptive bandwidth allocator (optional).
        self._adaptive_allocator = None
        if adaptive_bandwidth:
            from vllm.distributed.kv_transfer.kv_connector.v1.adaptive_bandwidth import (
                AdaptiveBandwidthAllocator, ComputeSlackTable,
            )
            slack_table = ComputeSlackTable(
                compute_slack_table if compute_slack_table else None)
            # kv_page_bytes will be set after _connect() when geometry is known.
            # For now use a placeholder; it's updated in _connect().
            self._adaptive_allocator = AdaptiveBandwidthAllocator(
                link_bandwidth_bps=link_bandwidth_bps,
                kv_page_bytes=1,  # updated in _connect()
                slack_table=slack_table,
            )

        self._client: IcmsClient | None = None
        self._geom: ModelGeometry | None = None
        self._sink_pool: _SinkSlotPool | None = None

        # Per-request state (C1 worker-side cache).
        self._requests: dict[str, _RequestState] = {}

        # Persistent chain cache: survives request eviction so Quest hooks
        # can find the chain even after request_finished runs.
        self._last_chain_for_rid: dict[str, list[int]] = {}

        # Cross-request context tracking: maps chain prefix (as tuple) to
        # the number of groups written for that prefix.  This lets Phase 2
        # (read) know how many context pages were stored by Phase 1 (write),
        # even though they're different vLLM requests.
        self._stored_chain_groups: list[tuple[list[int], int]] = []

        # Timing / debug stats.
        self.stats = IcmsTimingStats(level=stats_level, log_selections=log_selections)

        # Pending async score results (C9).
        # Key: layer_name → dict[request_id → (reply, req_idx_in_batch)]
        self._pending_scores: dict[str, dict[str, tuple]] = {}
        self._pending_reuse: dict[str, dict[str, tuple]] = {}
        self._score_lock = threading.Lock()

        # Path B: GPU KV cache tensors + per-step attn_metadata for
        # selective fetch + block table override.
        self._gpu_kv_caches: dict[str, torch.Tensor] = {}
        self._attn_metadata: dict | None = None  # layer_name → AttentionMetadata
        # Fetch buffer (allocated in register_kv_caches).
        self._fetch_key_cache: torch.Tensor | None = None
        self._fetch_value_cache: torch.Tensor | None = None
        self._fetch_block_table: torch.Tensor | None = None
        self._fetch_block_size: int = 16

        # Phase tracking: selective prefill → dense decode.
        # After the first wait_for_save (end of prefill), _prefill_done=True
        # and all subsequent steps use dense attention (no fetch buffer).
        self._prefill_done: bool = False


        # Pending notifications for the scheduler (stored chain info).
        # Drained into IcmsConnectorMetadata by the facade.
        self._pending_stored_notifications: list[tuple[list[int], int]] = []

        self._connect()

    def _connect(self):
        self._client = IcmsClient(self._socket_path)
        self._client.connect()
        ack = self._client.hello(self._model_name)
        self._geom = find_model(self._model_name) or ModelGeometry(
            name=self._model_name,
            num_layers=ack.num_layers,
            num_kv_heads=ack.num_kv_heads,
            head_dim=ack.head_dim,
            elem_bytes=ack.elem_bytes,
        )
        # Update allocator with actual kv_page_bytes from model geometry.
        if self._adaptive_allocator is not None:
            self._adaptive_allocator._kv_page_bytes = self._geom.kv_page_bytes

        # Sink sizing: stride layers × k pages × kv_page_bytes.
        # One stride group's worth of KV fits in the sink at a time.
        per_layer_bytes = self._k * self._geom.kv_page_bytes
        total_sink = self._score_stride * per_layer_bytes
        sink = self._client.register_sink(total_sink)
        self._sink_pool = _SinkSlotPool(sink, per_layer_bytes, _SINK_SLOTS)
        self._sink_per_layer_bytes = per_layer_bytes
        logger.info(
            "IcmsConnector worker: connected to %s, model=%s, "
            "sink=%d layers × %d B/layer = %d B total, score_stride=%d",
            self._socket_path, self._model_name,
            self._score_stride, per_layer_bytes, total_sink,
            self._score_stride,
        )

    def _get_stored_context_groups(self, chain: list[int]) -> int:
        """Find the longest stored chain prefix matching the given chain.

        Returns the number of groups written for that prefix (0 if none).
        This allows a new request to know how many context pages were
        stored by a prior request with the same prefix.
        """
        best = 0
        for stored_chain, n_groups in self._stored_chain_groups:
            match_len = 0
            for a, b in zip(chain, stored_chain):
                if a == b:
                    match_len += 1
                else:
                    break
            if match_len > 0:
                best = max(best, min(n_groups, match_len))
        return best

    def _record_stored_groups(self, chain: list[int], n_groups: int):
        """Record that n_groups were written for this chain prefix."""
        for i, (sc, _) in enumerate(self._stored_chain_groups):
            if sc == chain:
                self._stored_chain_groups[i] = (chain, max(_, n_groups))
                return
        self._stored_chain_groups.append((list(chain), n_groups))

    def shutdown(self):
        if self._client is None:
            return
        # Dump connector-side timing stats to a file before teardown.
        if self.stats.level > 0 or self.stats.log_selections:
            try:
                import json
                pid = os.getpid()
                stats_path = f"/tmp/icms_connector_stats_{pid}.json"
                with open(stats_path, "w") as f:
                    json.dump(self.stats.to_dict(), f, indent=2)
                logger.info("Connector stats dumped to %s", stats_path)
                if self.stats.log_selections and self.stats.selections:
                    sel_path = f"/tmp/icms_selections_{pid}.jsonl"
                    with open(sel_path, "w") as f:
                        for entry in self.stats.selections:
                            f.write(json.dumps(entry) + "\n")
                    logger.info("Page selections dumped to %s (%d entries)",
                                 sel_path, len(self.stats.selections))
            except Exception as e:
                logger.warning("Failed to dump connector stats: %s", e)
        # Evict all active requests.
        for rid, rs in list(self._requests.items()):
            if rs.chain:
                try:
                    self._client.evict(rs.chain)
                except Exception:
                    pass
        self._requests.clear()
        try:
            if self._sink_pool is not None:
                self._client.unregister_sink(self._sink_pool.sink)
                self._sink_pool.sink.close()
        finally:
            self._client.close()
            self._client = None

    # ─── metadata drain (C1) ─────────────────────────────────────────────

    def on_step_start(self, meta: IcmsConnectorMetadata):
        """Called from start_load_kv. Drains scheduler metadata into caches."""
        self.stats.advance_step()
        # Reset prefill_done when a new request arrives (new chain delivered).
        if meta.new_chains:
            self._prefill_done = False
        for rid, chain in meta.new_chains.items():
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.chain = chain
            self._last_chain_for_rid[rid] = chain
        for rid, bids in meta.block_id_maps.items():
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.block_ids = bids
        # C9: fire speculative summary preloads for new requests.
        for rid, chain in meta.new_chains.items():
            if chain:
                self._fire_preload(rid, chain)

    def _fire_preload(self, request_id: str, chain: list[int]):
        """C9 speculative preload: Score with key_dim=0 → preload summaries."""
        try:
            icms_rid = self._icms_request_id(request_id, 0)
            q_empty = np.zeros(0, dtype=np.float32)
            # Score with key_dim=0 signals preload-only.
            self._client.score(
                request_id=icms_rid, chain=chain, layer=0,
                query=q_empty, k=self._k, sink=self._sink_pool.sink,
            )
        except Exception as e:
            logger.debug("preload failed for %s: %s", request_id, e)

    # ─── layer hooks (from save_kv_layer kwargs) ─────────────────────────

    def on_layer_all_pages(self, next_layer_idx, budget, stats):
        """Budget >= 1.0: no sparse selection. Transfer all pages."""
        # For icms this means: don't score, just ensure all pages are written.
        # The actual transfer of all pages is handled by the standard offload
        # path — icms is only involved when scoring is needed.
        pass

    def on_layer_reuse(self, next_layer_idx, budget, stats):
        """Reuse layer: KV already in sink from the stride group's Score call.

        Reads the pre-filled KV from the sink at the pre-computed offsets
        and stores as a pending score result. The wait_for_layer path
        picks it up and populates the fetch buffer.
        """
        import copy
        attn_layer_name = f"model.layers.{next_layer_idx}.self_attn.attn"
        with self._score_lock:
            reuse_data = self._pending_reuse.pop(attn_layer_name, None)
        if reuse_data is None:
            return
        # reuse_data is dict[rid → (reply, reuse_offsets)]
        with self._score_lock:
            pending = self._pending_scores.setdefault(attn_layer_name, {})
            for rid, (reply, reuse_offsets) in reuse_data.items():
                reuse_reply = copy.copy(reply)
                reuse_reply.sink_offsets = reuse_offsets
                # Preserve req_idx from the original score result.
                orig = self._pending_scores.get(
                    f"model.layers.{next_layer_idx - 1}.self_attn.attn", {})
                req_idx = orig.get(rid, (None, 0))[1] if orig else 0
                pending[rid] = (reuse_reply, req_idx)

    def on_layer_score(self, next_layer_idx, quest_query, budget, stats,
                        connector_meta=None):
        """CPU-scoring path: Q arrived, fire Score against icms (C9 State 2/3)."""
        try:
            self._on_layer_score_impl(
                next_layer_idx, quest_query, budget, stats, connector_meta)
        except Exception:
            logger.exception("on_layer_score FAILED for layer %d", next_layer_idx)

    def _on_layer_score_impl(self, next_layer_idx, quest_query, budget, stats,
                              connector_meta=None):
        # Build the request list in BATCH ORDER from connector metadata.
        # The metadata.requests list matches the scheduler's batch ordering,
        # so req_idx corresponds to block_table row index.
        requests_to_score = []
        if (connector_meta is not None
                and isinstance(connector_meta, IcmsConnectorMetadata)
                and connector_meta.requests):
            for req_idx, step_req in enumerate(connector_meta.requests):
                rid = step_req.request_id
                rs = self._requests.get(rid)
                if rs is None:
                    chain = connector_meta.new_chains.get(rid, [])
                    if not chain:
                        chain = self._last_chain_for_rid.get(rid, [])
                    if chain:
                        rs = _RequestState(request_id=rid, chain=chain)
                        self._requests[rid] = rs
                if rs is not None and rs.chain:
                    requests_to_score.append((req_idx, rid, rs))

        # Fallback: use _requests dict order (batch=1 case).
        if not requests_to_score:
            for req_idx, (rid, rs) in enumerate(self._requests.items()):
                if rs.chain:
                    requests_to_score.append((req_idx, rid, rs))

        if not requests_to_score:
            return

        # Split the Q tensor per request using token counts from metadata.
        # quest_query shape: [total_tokens, num_heads, head_dim]
        per_request_q = {}
        if (quest_query is not None
                and isinstance(quest_query, torch.Tensor)
                and quest_query.ndim >= 2
                and len(requests_to_score) > 1
                and connector_meta is not None
                and connector_meta.requests):
            # Compute per-request token counts from metadata.
            token_starts = []
            offset = 0
            for step_req in connector_meta.requests:
                n_tokens = max(0, step_req.num_computed_tokens_end
                               - step_req.num_computed_tokens_start)
                token_starts.append((offset, offset + n_tokens, step_req.request_id))
                offset += n_tokens
            for start, end, rid in token_starts:
                if end <= quest_query.shape[0]:
                    per_request_q[rid] = quest_query[start:end]

        # Score each request with its own Q slice.
        for req_idx, rid, rs in requests_to_score:
            q_for_request = per_request_q.get(rid, quest_query)
            self._score_one_request(
                rid, rs, req_idx, next_layer_idx, q_for_request,
                budget, stats, connector_meta)

    def _score_one_request(self, rid, rs, req_idx, next_layer_idx,
                           quest_query, budget, stats, connector_meta):
        """Score a single request against ICMS and store the result."""

        # Score against icms trie. Works when the trie has data from a
        # prior request or prior prefill pass. If the trie is empty
        # (first request), Score returns empty winners and no fetch
        # buffer is populated — attention falls back to full GPU KV.

        # NOTE: The server's score cache is currently bypassed (adaptive
        # bandwidth allocation requires fresh scoring per stride group since
        # the budget may change).  The stride-group-based request_id is kept
        # for when the cache is re-enabled; it ensures fresh scores at stride
        # boundaries while allowing reuse within a group.
        stride_group = next_layer_idx // self._score_stride
        icms_rid = self._icms_request_id(rid, stride_group)
        # Use stored context groups (from prior requests with same prefix)
        # if the current request hasn't written anything yet.
        stored_groups = self._get_stored_context_groups(rs.chain)
        effective_groups = max(rs.num_groups_written, stored_groups)
        total_pages = effective_groups * _GROUP_BLOCKS

        # Determine effective budget: adaptive allocator or static.
        effective_budget = budget
        if self._adaptive_allocator is not None and total_pages > 0:
            # First layer of the request: register. Subsequent: recompute.
            if not hasattr(rs, '_adaptive_registered') or not rs._adaptive_registered:
                # Estimate new_tokens and cache_tokens from request state.
                cache_tokens = total_pages * PAGE_TOKENS
                # new_tokens is not directly available here; approximate from
                # the metadata's num_computed_tokens range.
                new_tokens = max(16, cache_tokens // 10)  # rough estimate
                if connector_meta and connector_meta.requests:
                    step_req = connector_meta.requests[0]
                    new_tokens = max(16, step_req.num_computed_tokens_end
                                    - step_req.num_computed_tokens_start)
                effective_budget = self._adaptive_allocator.register_request(
                    rid, new_tokens=new_tokens, cache_tokens=cache_tokens,
                    total_cache_pages=total_pages)
                rs._adaptive_registered = True
            else:
                effective_budget = self._adaptive_allocator.get_budget(rid)

        k = max(1, int(total_pages * effective_budget)) if total_pages > 0 else 1
        k = min(k, self._k)

        # Marshal query to fp32 numpy, mean-pooling Q heads per KV-head
        # group for GQA (matching QuestPageSelector.compute_page_scores).
        geom = self._geom
        if isinstance(quest_query, torch.Tensor):
            q = quest_query.detach().to(dtype=torch.float32, device="cpu")
            if q.ndim == 3:
                q = q.squeeze(0) if q.shape[0] == 1 else q.mean(dim=0)
            # q is now [num_heads, head_dim]. Mean-pool for GQA:
            # [num_heads, head_dim] → [num_kv_heads, heads_per_group, head_dim]
            # → mean → [num_kv_heads, head_dim] → [key_dim]
            if q.ndim == 2 and geom is not None:
                num_heads = q.shape[0]
                num_kv_heads = geom.num_kv_heads
                head_dim = geom.head_dim
                if num_heads > num_kv_heads and num_heads % num_kv_heads == 0:
                    heads_per_group = num_heads // num_kv_heads
                    q = (q.reshape(num_kv_heads, heads_per_group, head_dim)
                          .mean(dim=1))  # [num_kv_heads, head_dim]
            q = q.reshape(-1)
            q_np = q.contiguous().numpy()
        else:
            q_np = np.asarray(quest_query, dtype=np.float32).ravel()

        # Fire Score synchronously for v1. Use the shared sink (slot 0)
        # without the slot pool — in v1 we don't fetch KV back to GPU, so
        # we don't need per-request sink isolation. The score result is
        # just for measuring page selection accuracy.
        t_score_start = time.perf_counter()
        # Compute reuse range for this stride group.
        num_layers = self._geom.num_layers if self._geom else 48
        reuse_through = min(
            next_layer_idx + self._score_stride - 1, num_layers - 1)

        # Clear sink ready flags before writing a new stride group.
        self._sink_pool.sink.clear_ready_flags()

        logger.debug("Score: layer=%d reuse_through=%d chain_len=%d k=%d",
                      next_layer_idx, reuse_through, len(rs.chain), k)

        try:
            reply = self._client.score(
                request_id=icms_rid,
                chain=rs.chain,
                layer=next_layer_idx,
                query=q_np,
                k=k,
                sink=self._sink_pool.sink,
                reuse_through_layer=reuse_through,
            )
            t_score_end = time.perf_counter()
            # Stash storage-side concurrent request count for adaptive budget.
            rs._last_storage_concurrent = getattr(
                reply, 'concurrent_requests', 0)
            self.stats.record_score(
                (t_score_end - t_score_start) * 1e6,
                reply.cache_hit,
                list(reply.page_ids),
                next_layer_idx,
                scores=list(reply.scores),
            )

            # Store the score result keyed by (layer, request_id).
            attn_layer_name = f"model.layers.{next_layer_idx}.self_attn.attn"
            with self._score_lock:
                self._pending_scores.setdefault(attn_layer_name, {})[rid] = (
                    reply, req_idx)

            # Store references for reuse layers.
            per_layer_bytes = self._sink_per_layer_bytes
            for delta in range(1, reuse_through - next_layer_idx + 1):
                reuse_layer = next_layer_idx + delta
                reuse_attn = f"model.layers.{reuse_layer}.self_attn.attn"
                reuse_offsets = [off + delta * per_layer_bytes
                                 for off in reply.sink_offsets]
                with self._score_lock:
                    self._pending_reuse.setdefault(reuse_attn, {})[rid] = (
                        reply, reuse_offsets)
        except Exception as e:
            t_score_end = time.perf_counter()
            self.stats.record_score(
                (t_score_end - t_score_start) * 1e6,
                False, [], next_layer_idx,
            )
            logger.debug("Score failed layer %d: %s", next_layer_idx, e)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Stash GPU KV cache tensors for Path B reordered block table."""
        self._gpu_kv_caches = dict(kv_caches)
        # No separate fetch buffer needed — Path B uses a reordered block
        # table pointing into the main cache.  This preserves continuation
        # self-attention (see _populate_fetch_buffer).
        logger.info("Path B: registered %d KV cache layers (reordered block table mode)",
                     len(kv_caches))

    def set_attn_metadata(self, attn_metadata):
        """Stash per-step attn_metadata dict from ForwardContext."""
        self._attn_metadata = attn_metadata

    def wait_for_layer(self, layer_name: str):
        """Path B: fetch selected KV from icms → GPU + modify block table rows.

        For each pending Score result at this layer, fetches the winning
        pages' KV from the icms sink into GPU block slots and modifies the
        corresponding block_table row + seq_lens entry so FlashAttention
        only sees the k selected pages for that request.

        Supports batch > 1: each request's block_table row is modified
        independently. Original values are saved for restoration in
        restore_attn_metadata.
        """
        with self._score_lock:
            per_request = self._pending_scores.pop(layer_name, None)

        if not per_request:
            return

        if (self._prefill_done
                or not self._gpu_kv_caches
                or self._attn_metadata is None):
            return

        t_layer_start = time.perf_counter()
        total_k_pages = 0
        for rid, (reply, req_idx) in per_request.items():
            if reply is None or not reply.page_ids:
                continue
            try:
                self._apply_selective_attention(
                    layer_name, reply, req_idx, rid)
                total_k_pages += len(reply.page_ids)
            except Exception as e:
                logger.error("Path B: apply_selective FAILED for %s req=%s: %s",
                             layer_name, rid, e, exc_info=True)

        # Record per-layer breakdown for TTFT analysis.
        t_layer_end = time.perf_counter()
        layer_idx = self._extract_layer_idx(layer_name)
        if layer_idx is not None:
            self.stats.record_layer_breakdown(
                layer=layer_idx, score_us=0.0,
                fetch_us=(t_layer_end - t_layer_start) * 1e6,
                modify_us=0.0,
                total_us=(t_layer_end - t_layer_start) * 1e6,
                k_pages=total_k_pages,
            )

    def _apply_selective_attention(self, layer_name: str, reply,
                                   req_idx: int, rid: str = ""):
        """Modify block_table row for one request: fill selected KV + trim.

        Fetches selected pages' KV from the ICMS sink into GPU blocks,
        then rewrites block_table[req_idx] to contain only the selected
        context blocks + continuation blocks. Saves the original row for
        restoration.

        Returns (req_idx, orig_bt_row, orig_seq_len) for restoration,
        or None if nothing was modified.
        """
        kv = self._gpu_kv_caches.get(layer_name)
        if kv is None or kv.ndim != 5:
            return None
        main_key = kv[0]
        main_value = kv[1]

        am = self._attn_metadata.get(layer_name) if isinstance(self._attn_metadata, dict) else None
        if am is None or not hasattr(am, "block_table"):
            return None
        bt = am.block_table  # [num_reqs, max_blocks]

        if req_idx >= bt.shape[0]:
            return None

        seq_len = int(am.seq_lens[req_idx])

        # Look up request state by request_id.
        rs = self._requests.get(rid)
        if rs is None:
            # Fallback: try last known chain.
            chain = self._last_chain_for_rid.get(rid, [])
            if chain:
                rs = _RequestState(request_id=rid, chain=chain)
            else:
                return None

        stored_groups = self._get_stored_context_groups(rs.chain)
        effective_groups = max(rs.num_groups_written, stored_groups)
        context_pages = effective_groups * _GROUP_BLOCKS
        context_tokens = context_pages * PAGE_TOKENS
        total_blocks = (seq_len + PAGE_TOKENS - 1) // PAGE_TOKENS

        # ── Fetch selected pages' KV from ICMS sink → GPU blocks ──
        selected = sorted(
            pid for pid in reply.page_ids[:min(len(reply.page_ids), self._k)]
            if pid < context_pages
        )
        k = len(selected)
        if k == 0:
            return None

        pid_to_sink_off = {}
        for i, pid in enumerate(reply.page_ids):
            if i < len(reply.sink_offsets):
                pid_to_sink_off[pid] = reply.sink_offsets[i]

        geom = self._geom
        sink_view = self._sink_pool.sink.view()
        kv_page_bytes = geom.kv_page_bytes
        half_bytes = kv_page_bytes // 2
        model_dtype = main_key.dtype
        device = main_key.device
        page_shape = (PAGE_TOKENS, geom.num_kv_heads, geom.head_dim)

        filled_blocks = []
        for pid in selected:
            sink_off = pid_to_sink_off.get(pid)
            if sink_off is None or pid >= bt.shape[1]:
                continue
            phys_block = int(bt[req_idx, pid])
            if phys_block >= main_key.shape[0]:
                continue
            raw = bytes(sink_view[sink_off:sink_off + kv_page_bytes])
            k_np = np.frombuffer(raw[:half_bytes], dtype=np.float16).reshape(page_shape)
            v_np = np.frombuffer(raw[half_bytes:], dtype=np.float16).reshape(page_shape)
            k_t = torch.from_numpy(k_np.copy()).to(dtype=model_dtype, device=device)
            v_t = torch.from_numpy(v_np.copy()).to(dtype=model_dtype, device=device)
            main_key[phys_block].copy_(k_t)
            main_value[phys_block].copy_(v_t)
            filled_blocks.append(phys_block)

        # ── Build trimmed block table ──
        # [filled context blocks | continuation blocks]
        new_entries = list(filled_blocks)
        for blk_idx in range(context_pages, total_blocks):
            if blk_idx < bt.shape[1]:
                new_entries.append(int(bt[req_idx, blk_idx]))
        if not new_entries:
            return None

        continuation_tokens = max(0, seq_len - context_tokens)
        new_seq_len = len(filled_blocks) * PAGE_TOKENS + continuation_tokens

        bt_device = bt.device
        new_bt = torch.tensor([new_entries], dtype=torch.int32, device=bt_device)
        new_sl = torch.tensor([new_seq_len], dtype=torch.int32, device=bt_device)

        logger.debug(
            "Path B: layer=%s req_idx=%d fetched %d/%d pages, "
            "seq_len %d→%d",
            layer_name, req_idx, len(filled_blocks), context_pages,
            seq_len, new_seq_len,
        )

        # Set fetch state for FlashAttention to read.
        from vllm.v1.attention import icms_fetch_state
        icms_fetch_state.set_active(icms_fetch_state.IcmsFetchState(
            key_cache=main_key,
            value_cache=main_value,
            block_table=new_bt,
            seq_lens=new_sl,
            max_seq_len=new_seq_len,
        ))
        return True  # signal that fetch state was set

    @staticmethod
    def _extract_layer_idx(layer_name: str) -> int | None:
        """Extract layer index from 'model.layers.N.self_attn.attn'."""
        parts = layer_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def restore_attn_metadata(self, layer_name: str):
        """Clear the ICMS fetch state after attention ran with it."""
        from vllm.v1.attention import icms_fetch_state
        icms_fetch_state.clear()

    def wait_for_pending_writes(self):
        """Block until all buffered WriteGroups are flushed.

        Also marks prefill as done — subsequent steps use dense decode.
        """
        if logger.isEnabledFor(10):
            n_reqs = len(self._requests)
            n_bufs = sum(len(rs.active_group_buffers) for rs in self._requests.values())
            logger.debug("wait_for_pending_writes: %d reqs, %d buffers", n_reqs, n_bufs)
        for rid, rs in self._requests.items():
            for gidx in list(rs.active_group_buffers.keys()):
                self._flush_group(rid, gidx, partial=True)
            # Record how many groups were written for this chain, so future
            # requests with the same prefix can find the stored context.
            if rs.chain and rs.num_groups_written > 0:
                self._record_stored_groups(rs.chain, rs.num_groups_written)
                # Queue notification for the scheduler's global prefix index.
                _stored_chain_queue.append(
                    (list(rs.chain), rs.num_groups_written))
                logger.info(
                    "Recorded %d stored groups for chain len=%d",
                    rs.num_groups_written, len(rs.chain),
                )
        if not self._prefill_done:
            self._prefill_done = True
            logger.info("Prefill done. Switching to dense decode (no fetch buffer).")

    # ─── KV extraction from GPU paged buffer (C2 write path) ────────────

    def extract_and_record(self, layer_name: str, kv_layer: torch.Tensor,
                            attn_metadata) -> None:
        """Extract per-block K/V from the GPU KV cache and populate icms buffers.

        Called from save_kv_layer for each standard (non-Quest) per-layer call.

        kv_layer shape: [2 (K+V), num_blocks, block_size, num_kv_heads, head_dim]
        attn_metadata: FlashAttentionMetadata with block_table, seq_lens, etc.
        """
        t0 = time.perf_counter()
        logger.debug("extract_and_record: layer=%s kv_shape=%s", layer_name,
                      tuple(kv_layer.shape) if hasattr(kv_layer, 'shape') else '?')
        if not self._requests or self._geom is None:
            return

        # Parse layer index from layer_name like "model.layers.5.self_attn.attn"
        layer_idx = self._parse_layer_idx(layer_name)
        if layer_idx is None or layer_idx >= self._geom.num_layers:
            return

        # kv_layer: [2, num_blocks, block_size, num_kv_heads, head_dim]
        if kv_layer.ndim != 5 or kv_layer.shape[0] != 2:
            return
        k_cache = kv_layer[0]  # [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache = kv_layer[1]

        # block_table: [num_reqs, max_num_blocks_per_req] — per-request block IDs
        block_table = getattr(attn_metadata, "block_table", None)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if block_table is None or seq_lens is None:
            return

        # Block size from the KV cache tensor.
        block_size = k_cache.shape[1]

        # With batch=1 (C3), process only the first request.
        # block_table may be a tensor on GPU; move to CPU for indexing.
        if isinstance(block_table, torch.Tensor):
            bt_cpu = block_table.cpu()
        else:
            bt_cpu = block_table
        if isinstance(seq_lens, torch.Tensor):
            sl_cpu = seq_lens.cpu()
        else:
            sl_cpu = seq_lens

        # Find the first active request.
        rid = None
        rs = None
        for r, s in self._requests.items():
            rid = r
            rs = s
            break
        if rs is None or not rs.chain:
            return

        # Number of blocks for this request.
        if hasattr(sl_cpu, '__len__') and len(sl_cpu) > 0:
            seq_len = int(sl_cpu[0]) if isinstance(sl_cpu[0], (int, float, torch.Tensor)) else 0
        else:
            seq_len = 0
        if seq_len <= 0:
            return
        num_blocks = (seq_len + block_size - 1) // block_size

        # Extract the request's block IDs from the block table.
        if bt_cpu.ndim >= 2:
            req_block_ids = bt_cpu[0, :num_blocks].tolist()
        elif bt_cpu.ndim == 1:
            req_block_ids = bt_cpu[:num_blocks].tolist()
        else:
            return

        # Only process blocks we haven't recorded yet for this layer.
        recorded_key = (rid, layer_idx)
        if not hasattr(rs, '_recorded_blocks'):
            rs._recorded_blocks = {}
        already_recorded = rs._recorded_blocks.get(recorded_key, 0)
        if already_recorded >= len(req_block_ids):
            return  # nothing new

        new_ids = req_block_ids[already_recorded:]
        valid_ids = [bid for bid in new_ids if 0 <= bid < k_cache.shape[0]]
        if not valid_ids:
            rs._recorded_blocks[recorded_key] = len(req_block_ids)
            return

        # Batched GPU→CPU copy: index all new blocks at once.
        idx_tensor = torch.tensor(valid_ids, dtype=torch.long, device=k_cache.device)
        k_batch = k_cache[idx_tensor].cpu()  # [N, block_size, num_kv_heads, head_dim]
        v_batch = v_cache[idx_tensor].cpu()

        for i, intra_idx in enumerate(range(already_recorded, already_recorded + len(valid_ids))):
            self.record_page(rid, intra_idx, layer_idx, k_batch[i], v_batch[i])

        rs._recorded_blocks[recorded_key] = len(req_block_ids)
        self.stats.record_extract(
            (time.perf_counter() - t0) * 1e6, len(valid_ids))

    @staticmethod
    def _parse_layer_idx(layer_name: str) -> int | None:
        """Extract layer index from 'model.layers.N.self_attn.attn'."""
        parts = layer_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    # ─── group buffer management (C2) ────────────────────────────────────

    def record_page(
        self,
        request_id: str,
        intra_request_block_idx: int,
        layer_idx: int,
        key_block: torch.Tensor,    # [block_size, num_kv_heads, head_dim]
        value_block: torch.Tensor,  # same shape
    ):
        """Record one page's K/V + summary into the per-(req, group) buffer.

        Called from the metadata-update-after-store equivalent in the
        connector's save path.
        """
        rs = self._requests.get(request_id)
        if rs is None:
            return
        geom = self._geom

        group_idx, page_in_group = divmod(intra_request_block_idx, _GROUP_BLOCKS)
        buf = rs.active_group_buffers.get(group_idx)
        if buf is None:
            buf = _GroupBuffer(
                summary_blob=bytearray(geom.num_layers * geom.summary_group_bytes),
                kv_blob=bytearray(geom.num_layers * geom.kv_group_bytes),
                filled=set(),
                num_layers=geom.num_layers,
                pages_in_group=_GROUP_BLOCKS,
            )
            rs.active_group_buffers[group_idx] = buf

        # Summary: compute per-page channel-wise min/max from K (C2/C4).
        keys = key_block.to(dtype=torch.float32, device="cpu")
        if keys.ndim == 3:
            keys = keys.reshape(keys.shape[0], -1)  # [block_size, key_dim]
        kmin = keys.min(dim=0).values.to(torch.float16)
        kmax = keys.max(dim=0).values.to(torch.float16)
        summary_bytes = kmin.numpy().tobytes() + kmax.numpy().tobytes()

        # KV: K || V byte concatenation (C4).
        k_bytes = key_block.to(torch.float16).contiguous().numpy().tobytes()
        v_bytes = value_block.to(torch.float16).contiguous().numpy().tobytes()
        kv_bytes = k_bytes + v_bytes

        # Write into buffer at the correct offset.
        spb = geom.summary_page_bytes
        kpb = geom.kv_page_bytes
        s_off = layer_idx * geom.summary_group_bytes + page_in_group * spb
        k_off = layer_idx * geom.kv_group_bytes + page_in_group * kpb
        buf.summary_blob[s_off:s_off + len(summary_bytes)] = summary_bytes
        buf.kv_blob[k_off:k_off + len(kv_bytes)] = kv_bytes
        buf.filled.add((layer_idx, page_in_group))

        # Flush if complete.
        if buf.is_complete():
            self._flush_group(request_id, group_idx, partial=False)

    def _flush_group(self, request_id: str, group_idx: int, partial: bool = False):
        """Issue WriteGroup for a completed (or partial) group buffer."""
        rs = self._requests.get(request_id)
        if rs is None:
            return
        buf = rs.active_group_buffers.pop(group_idx, None)
        if buf is None:
            return
        chain_prefix = rs.chain[:group_idx + 1]
        if not chain_prefix:
            return
        pages = _GROUP_BLOCKS if not partial else len({p for (_, p) in buf.filled})
        logger.debug("flush_group: group=%d pages=%d", group_idx, pages)
        t0 = time.perf_counter()
        try:
            self._client.write_group(
                chain_prefix, bytes(buf.summary_blob), bytes(buf.kv_blob),
                pages_in_group=pages,
            )
            self.stats.record_flush((time.perf_counter() - t0) * 1e6)
            self.stats.total_groups_written += 1
            if group_idx >= rs.num_groups_written:
                rs.num_groups_written = group_idx + 1
        except Exception as e:
            self.stats.record_flush((time.perf_counter() - t0) * 1e6)
            logger.warning("WriteGroup failed for req=%s group=%d: %s",
                           request_id, group_idx, e)

    # ─── request lifecycle ───────────────────────────────────────────────

    def on_request_finished(self, request_id: str):
        rs = self._requests.pop(request_id, None)
        if rs is None:
            return
        # Reset prefill_done for the next request.
        self._prefill_done = False
        # Flush any partial group buffers.
        for gidx in list(rs.active_group_buffers.keys()):
            self._flush_group(request_id, gidx, partial=True)
        # KV data is NOT evicted — it persists for prefix reuse by
        # subsequent requests. Eviction is managed by the server's LRU
        # when capacity is full.

        # Unregister from adaptive bandwidth allocator.
        if self._adaptive_allocator is not None:
            self._adaptive_allocator.unregister_request(request_id)

    # ─── direct helper API (backward compat with smoke tests) ────────────

    def direct_write_group(self, request_id: str, chain: list[int],
                            summary_blob: bytes, kv_blob: bytes):
        rs = self._requests.setdefault(
            request_id, _RequestState(request_id=request_id))
        rs.chain = list(chain)
        return self._client.write_group(chain, summary_blob, kv_blob,
                                         pages_in_group=_GROUP_BLOCKS)

    def direct_score(self, request_id: str, chain: list[int],
                      layer: int, query: torch.Tensor, k: int):
        if isinstance(query, torch.Tensor):
            q_np = query.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
        else:
            q_np = np.asarray(query, dtype=np.float32)
        if q_np.ndim > 1:
            q_np = q_np.reshape(-1)
        # Use 0 as num_computed_tokens so all layers of the same direct
        # call sequence share a single icms request_id (cross-layer reuse).
        icms_rid = self._icms_request_id(request_id, 0)
        return self._client.score(
            request_id=icms_rid, chain=chain, layer=layer,
            query=q_np, k=k, sink=self._sink_pool.sink,
        )

    # ─── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _icms_request_id(request_id: str, num_computed_tokens: int) -> int:
        """Derive a u64 icms request_id from vLLM request_id + chunk position.

        Per Q6/D5: per-prefill-chunk namespacing so the score cache is fresh
        per chunk. For decode (num_computed_tokens doesn't change the hash
        for the same prefill chunk), subsequent layers reuse the same cache.
        """
        h = hash((request_id, num_computed_tokens)) & 0xFFFFFFFFFFFFFFFF
        return h
