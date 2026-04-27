# SPDX-License-Identifier: Apache-2.0
"""ICMS storage service KV connector for vLLM v1.

Standalone connector (does NOT inherit from OffloadingConnector) that routes
Quest scoring and KV storage through icms_client talking to a separate
`icms_server` process. Supports both Unix socket (local) and RDMA (remote)
transport. See:

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
import queue as _queue
import sys
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
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
    GROUP_PAGES, PAGE_TOKENS, KvLayout, ModelGeometry, find_model,
    parse_scored_layers,
)
import dataclasses as _dataclasses                            # noqa: E402
from icms_client.sink import Sink, allocate_sink             # noqa: E402

# RDMA client (optional — pyverbs must be installed).
try:
    from icms_client.rdma_client import RdmaIcmsClient       # noqa: E402
    from icms_client.rdma_transport import RdmaTransportConfig  # noqa: E402
    _HAVE_RDMA = True
except ImportError:
    _HAVE_RDMA = False

# C++ ArrivalPoller (slack diagnostic). Optional — the binding may
# pre-date the poller addition. When absent, the slack path falls back
# to the inline-probe (lower-bound) scheme.
try:
    from icms_client._icms_client import ArrivalPoller as _ArrivalPoller  # noqa: E402
    _HAVE_ARRIVAL_POLLER = True
except ImportError:
    _ArrivalPoller = None  # type: ignore[assignment]
    _HAVE_ARRIVAL_POLLER = False

# ─── constants ───────────────────────────────────────────────────────────
# TODO(batching): _SINK_SLOTS=4 caps in-flight Score/FetchAll RPCs at 4
# concurrent operations. Fine for batch=1 perf-sweep workloads, but blocks
# under heavy batching when more than 4 requests in a step want a fresh
# Score/Fetch (acquire() will spin-wait on slot release). Should be
# parameterised (env or kv_connector_extra_config) before serving traffic
# with max_num_seqs > 4. (Audit 2026-04-26.)
_SINK_SLOTS = 4           # C8: number of pre-allocated sink slots
_GROUP_BLOCKS = GROUP_PAGES  # blocks per group (= pages per group = 32)

# TP sharding: each rank prefixes its chain with a rank-tagged sentinel so
# server-side trie entries never collide across ranks. At TP=1 this adds
# one no-op level to the trie; wire and scoring behavior are unchanged.
# See docs/icms_connector_tp_support.md.
_RANK_TAG_MAGIC = 0xE1C5A4EE00000000

def _rank_tag(tp_rank: int) -> int:
    return _RANK_TAG_MAGIC | int(tp_rank)

def _rank_tagged_chain(tp_rank: int, chain):
    if chain is None:
        return chain
    return [_rank_tag(tp_rank), *chain]


def _tp_broadcast_score_reply(reply, tp_rank: int, tp_size: int):
    """Option W: rank 0 has the server's ScoreReply; broadcast it to every
    rank so each rank populates _pending_scores identically.

    Uses a scalar-header + variable-length payload protocol over NCCL
    broadcast of CUDA tensors. The reply's page_ids / scores /
    sink_offsets are small (k × 16 bytes typical) so two broadcasts
    (header u64×8 + payload) is cheap.
    """
    import torch.distributed as dist  # noqa: E402
    from vllm.distributed.parallel_state import get_tp_group
    tp_group = get_tp_group()
    dev_group = tp_group.device_group
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Header: [status, trie_walk_ns, summary_read_ns, score_ns,
    #          sink_write_ns, cache_hit, concurrent_requests,
    #          server_ingest_to_ready_ns, num_pages]  — 9 i64 slots.
    if tp_rank == 0 and reply is not None:
        n_pages = len(reply.page_ids)
        hdr = torch.tensor([
            int(reply.status),
            int(reply.trie_walk_ns),
            int(reply.summary_read_ns),
            int(reply.score_ns),
            int(reply.sink_write_ns),
            int(bool(reply.cache_hit)),
            int(reply.concurrent_requests),
            int(reply.server_ingest_to_ready_ns),
            int(n_pages),
        ], dtype=torch.int64, device=dev)
    else:
        hdr = torch.zeros(9, dtype=torch.int64, device=dev)
    dist.broadcast(hdr, src=tp_group.first_rank, group=dev_group)

    n_pages = int(hdr[8].item())
    if n_pages == 0:
        from icms_client.protocol import ScoreReply
        return ScoreReply(
            status=int(hdr[0]), trie_walk_ns=int(hdr[1]),
            summary_read_ns=int(hdr[2]), score_ns=int(hdr[3]),
            sink_write_ns=int(hdr[4]), cache_hit=bool(hdr[5]),
            concurrent_requests=int(hdr[6]),
            server_ingest_to_ready_ns=int(hdr[7]),
            page_ids=[], scores=[], sink_offsets=[],
        )

    # Payload: page_ids (i64) + sink_offsets (i64) + scores (f32).
    if tp_rank == 0:
        pids = torch.tensor(list(reply.page_ids), dtype=torch.int64, device=dev)
        offs = torch.tensor(list(reply.sink_offsets), dtype=torch.int64, device=dev)
        scs  = torch.tensor(list(reply.scores), dtype=torch.float32, device=dev)
    else:
        pids = torch.zeros(n_pages, dtype=torch.int64, device=dev)
        offs = torch.zeros(n_pages, dtype=torch.int64, device=dev)
        scs  = torch.zeros(n_pages, dtype=torch.float32, device=dev)
    dist.broadcast(pids, src=tp_group.first_rank, group=dev_group)
    dist.broadcast(offs, src=tp_group.first_rank, group=dev_group)
    dist.broadcast(scs,  src=tp_group.first_rank, group=dev_group)

    from icms_client.protocol import ScoreReply
    return ScoreReply(
        status=int(hdr[0]), trie_walk_ns=int(hdr[1]),
        summary_read_ns=int(hdr[2]), score_ns=int(hdr[3]),
        sink_write_ns=int(hdr[4]), cache_hit=bool(hdr[5]),
        concurrent_requests=int(hdr[6]),
        server_ingest_to_ready_ns=int(hdr[7]),
        page_ids=pids.cpu().tolist(),
        scores=scs.cpu().tolist(),
        sink_offsets=offs.cpu().tolist(),
    )

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
    # BUG-N2: rids for which the scheduler determined no new complete
    # group will be formed by this step (stored prefix already covers
    # the prompt's complete-group count). Worker uses this to short-
    # circuit `extract_and_record`'s GPU→CPU copy + (at TP>1)
    # AllGather × num_layers. Computed scheduler-side from the
    # authoritative `_stored_chains` so both worker ranks see the
    # same flag — symmetric across TP, no AllGather risk.
    skip_extract_rids: set[str] = field(default_factory=set)


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
    stored_groups: int = 0  # groups already in ICMS under this chain prefix (dedup-aware skip)


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
#  Deferred write pipeline
# ═══════════════════════════════════════════════════════════════════════════

class _WritePipeline:
    """Background worker that runs extract + flush off the TTFT critical path.

    wait_for_pending_writes used to do the GPU→CPU copies, numpy
    conversion, summary min/max, bytearray fills, and WriteGroup RPCs
    all synchronously — which land inside vLLM's wait_for_save, which
    is on the TTFT critical path. This class offloads all of that to a
    single worker thread. The main thread enqueues a task and returns
    immediately; the drain is done in on_request_finished (off TTFT).

    Single worker thread (not a pool) because:
      * ICMS client is NOT thread-safe (one lock to serialize with main-
        thread Score/FetchAll RPCs is sufficient; no additional worker-
        vs-worker racing).
      * Work ordering matters (later WriteGroups depend on earlier ones
        for the same request).
    """

    def __init__(self, name: str = "icms-writes"):
        self._q: "_queue.Queue" = _queue.Queue()
        self._pending = 0
        self._cv = threading.Condition()
        self._stop = False
        self._t = threading.Thread(
            target=self._loop, name=name, daemon=True)
        self._t.start()

    def submit(self, fn, tag: str = ""):
        with self._cv:
            self._pending += 1
        self._q.put((fn, tag))

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:  # poison
                return
            fn, tag = item
            try:
                fn()
            except Exception:
                logger.exception("WritePipeline[%s]: task failed", tag)
            finally:
                with self._cv:
                    self._pending -= 1
                    if self._pending == 0:
                        self._cv.notify_all()

    def pending(self) -> int:
        with self._cv:
            return self._pending

    def drain(self, timeout: float | None = None) -> bool:
        """Block until pending == 0. Returns True on drain complete,
        False on timeout."""
        with self._cv:
            if timeout is None:
                while self._pending > 0:
                    self._cv.wait()
                return True
            deadline = time.monotonic() + timeout
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            return True

    def shutdown(self, timeout: float = 5.0):
        self._stop = True
        self._q.put(None)
        self._t.join(timeout=timeout)


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

        # TP>1 support: per-rank chain namespacing (commit 2) + server-side
        # Score/FetchAll fan-out (commit 4). Each rank's connector opens its
        # own connection with its own tp_rank/tp_size (sent at Hello), stores
        # chunks under its own rank-tagged chain, and registers its own sink
        # on f"cuda:{tp_rank}". Rank 0 all_gathers Q across the TP group and
        # issues one Score RPC; other ranks skip Score and just poll their
        # own flag sink. See docs/icms_connector_tp_support.md.
        tp = getattr(getattr(vllm_config, "parallel_config", None),
                      "tensor_parallel_size", 1)
        if tp > 1:
            logger.info(
                "IcmsConnector: TP=%d detected; using per-rank chain "
                "namespacing + server-side fan-out (tp_size>1 path).", tp)

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

        # RDMA transport (replaces Unix socket + POSIX shmem).
        self._use_rdma: bool = bool(extra.get("icms_rdma", False))
        self._rdma_server_host: str = extra.get("icms_server_host", "sprc01")
        self._rdma_port: int = int(extra.get("icms_rdma_port", 18515))
        self._rdma_ib_dev: str = extra.get("icms_ib_dev", "mlx5_0")
        # GPUDirect RDMA: server writes KV directly into GPU HBM.
        self._gpu_direct: bool = bool(extra.get("icms_gpu_direct", False))
        self._gpu_device: str = extra.get("icms_gpu_device", "cuda:0")

        # ICMS is designed as an external KV backing store: turn-end drop
        # is the intended default, and cross-turn prefix hits are served
        # by get_num_new_matched_tokens + Path B. If vLLM's in-process
        # prefix caching is also on, it races ICMS for the same prefix
        # match and wins by default (HBM hit before scheduler asks the
        # connector), which means the ICMS fetch path never fires.
        cache_cfg = getattr(vllm_config, "cache_config", None)
        if cache_cfg is not None and getattr(cache_cfg, "enable_prefix_caching", False):
            logger.warning(
                "IcmsConnector: vLLM prefix caching is enabled. ICMS expects "
                "drop-between-turns as the default; prefix caching will hide "
                "cross-turn ICMS fetches. Pass enable_prefix_caching=False to "
                "the LLM constructor when using ICMS.")

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
                use_rdma=self._use_rdma,
                rdma_server_host=self._rdma_server_host,
                rdma_port=self._rdma_port,
                rdma_ib_dev=self._rdma_ib_dev,
                gpu_direct=self._gpu_direct,
                gpu_device=self._gpu_device,
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

        # KV extraction is deferred OFF the forward-pass critical path:
        # save_kv_layer no longer calls extract_and_record inline. The
        # actual GPU→CPU copy + summary compute + buffer fill runs in a
        # batch at end-of-forward-pass in wait_for_pending_writes.
        is_quest_call = (quest_query is not None
                         or quest_all_pages
                         or quest_reuse_selection)

        if quest_all_pages:
            # Budget >= 1.0: fetch every stored page via kFetchAll — server
            # skips scoring and streams every page under the chain prefix
            # into the sink.
            self._worker.on_layer_all_pages(
                next_layer_idx, budget, quest_stats,
                connector_meta=self._connector_metadata,
            )
            return

        if quest_reuse_selection:
            # Single-layer-scoring reuse: the connector reuses previous
            # selection. For icms this is a cache-hit Score (C9 State 1).
            self._worker.on_layer_reuse(next_layer_idx, budget, quest_stats)
            return

        if quest_query is not None and next_layer_idx is not None:
            # Extraction is deferred to wait_for_pending_writes; save_kv
            # only needs to drive scoring here.
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
        # Cross-process bridge for the prefix index. The worker writes the
        # chain to its process-local `_stored_chain_queue` when iter-1 saves
        # to ICMS, but at TP>1 with multiproc_executor the scheduler runs in
        # a different Python process and never sees that queue. Without
        # this, get_num_new_matched_tokens returns 0 at TP>1 → vLLM never
        # elides the prefill → TP=2 ICMS C runs the full 16k forward
        # (~890 ms) instead of the intended 16-token short-circuit
        # (~130 ms). The scheduler already has rid→chain from
        # get_num_new_matched_tokens, and n_groups is deterministic from
        # the prompt length, so we record it here directly without IPC.
        if self._sched is not None:
            self._sched.on_finished_record_stored(request)
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
            # Cover the entire prompt. Earlier hardcoded `range(64)` cap
            # silently truncated chains to the first 32k tokens, capping
            # cross-turn elision at 32k (qwen3 ≥64k C TTFT regressed
            # badly because vLLM had to forward the un-chained tail).
            #
            # Incremental hashing: each group's hash chains from the
            # previous group's digest + new group's tokens. Keeps the
            # whole chain at O(n) total work instead of O(n²) — at 128k
            # the old per-group "hash all tokens up to here" pattern
            # would do ~150 MB of repr+SHA per request.
            num_groups = (len(token_ids) + group_tokens - 1) // group_tokens
            prev_digest = b""
            for g in range(num_groups):
                end = min((g + 1) * group_tokens, len(token_ids))
                start = g * group_tokens
                if end <= start:
                    break
                h = hashlib.sha256()
                h.update(prev_digest)
                h.update(repr(tuple(token_ids[start:end])).encode())
                d = h.digest()
                chain.append(int.from_bytes(d[:8], "little"))
                prev_digest = d
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

        # BUG-N2: per-rid skip-extract decision. We compute this on the
        # scheduler so both TP worker ranks observe the same flag (no
        # rank-local race that would lead to AllGather asymmetry). For
        # each scheduled request, skip extract iff the stored prefix
        # already covers every COMPLETE group the prompt will have
        # after this step. Requests whose new tokens form a partial
        # trailing group still skip (group never flushes anyway).
        #
        # BUG-N13 fix (2026-04-26): the first scheduler tick for a new
        # request sometimes arrives BEFORE vLLM populates
        # scheduled_num_tokens, so n_step = 0 and end = 0. With end=0
        # and stored_groups=0 (fresh chain), the predicate `0 >= 0`
        # was vacuously true and we'd add the rid to skip_set. The
        # subsequent forward (which used THIS metadata) then skipped
        # extract entirely → iter 0 never stored → scheduler bridge
        # then claimed phantom storage → iter 1+ silently elided
        # against missing KV. Require complete_groups_after > 0 so
        # the metadata-only first tick doesn't poison the decision.
        _GROUP_TOKENS = PAGE_TOKENS * _GROUP_BLOCKS  # 16 * 32 = 512
        for prs in meta.requests:
            chain = self._chains.get(prs.request_id, [])
            if not chain:
                continue
            complete_groups_after = (
                prs.num_computed_tokens_end // _GROUP_TOKENS)
            if complete_groups_after == 0:
                # First-tick metadata or tiny prompt — extract path
                # has nothing to flush either way. Don't poison the
                # skip decision when we don't actually have visibility
                # into the upcoming forward yet.
                continue
            stored_groups = self._lookup_stored_prefix(chain)
            if stored_groups >= complete_groups_after:
                meta.skip_extract_rids.add(prs.request_id)
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
                # Mirrors the chain-construction in on_alloc — full prompt
                # coverage with incremental hashing. Both call sites must
                # produce identical hashes to match across requests.
                num_groups = (len(token_ids) + group_tokens - 1) // group_tokens
                prev_digest = b""
                for g in range(num_groups):
                    end = min((g + 1) * group_tokens, len(token_ids))
                    start = g * group_tokens
                    if end <= start:
                        break
                    h = hashlib.sha256()
                    h.update(prev_digest)
                    h.update(repr(tuple(token_ids[start:end])).encode())
                    d = h.digest()
                    chain.append(int.from_bytes(d[:8], "little"))
                    prev_digest = d
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

    def on_finished_record_stored(self, request: Request) -> None:
        """Record this request's chain into the scheduler-side prefix index
        so subsequent iter-2 lookups via get_num_new_matched_tokens find it.

        Required at TP>1 with multiproc_executor where the worker's save
        notifications never reach the scheduler process (the
        ``_stored_chain_queue`` global is per-process). The chain was
        already computed in get_num_new_matched_tokens for this rid;
        n_groups is deterministic from prompt length.

        Idempotent: record_stored_chain merges duplicates by chain key.
        """
        rid = request.request_id
        chain = self._chains.get(rid)
        if not chain:
            return
        token_ids = getattr(request, "all_token_ids", None)
        if token_ids is None or len(token_ids) == 0:
            token_ids = getattr(request, "prompt_token_ids", [])
        if not token_ids:
            return
        # n_groups = number of COMPLETE groups written. The worker only
        # saves complete groups, so floor-divide is correct (matches the
        # match_best=32 behavior we see at TP=1 for prompt_len=16448:
        # 16448 // 512 = 32).
        group_tokens = _GROUP_BLOCKS * PAGE_TOKENS  # 32 * 16 = 512
        n_groups = len(token_ids) // group_tokens
        if n_groups <= 0:
            return
        self.record_stored_chain(chain, n_groups)
        logger.debug(
            "on_finished_record_stored: rid=%s chain_len=%d n_groups=%d "
            "prompt_len=%d",
            rid, len(chain), n_groups, len(token_ids))

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
                 compute_slack_table: str = "",
                 use_rdma: bool = False,
                 rdma_server_host: str = "sprc01",
                 rdma_port: int = 18515,
                 rdma_ib_dev: str = "mlx5_0",
                 gpu_direct: bool = False,
                 gpu_device: str = "cuda:0"):
        self._socket_path = socket_path
        self._use_rdma = use_rdma
        self._rdma_server_host = rdma_server_host
        self._rdma_port = rdma_port
        self._rdma_ib_dev = rdma_ib_dev
        self._gpu_direct = gpu_direct
        self._gpu_device = gpu_device
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

        # ICMS_DIAG_SLACK: optional C++ poll thread that records each
        # ready-flag's first-non-zero timestamp without holding the GIL.
        # Allocated lazily on the first slack-enabled step.
        self._slack_poller = None  # type: ignore[var-annotated]

        # Tensor-parallel identity. At TP=1 this is {0, 1} and the rank-tag
        # prefix is a no-op level in the trie. At TP>1 each rank prefixes
        # every outbound chain with its own rank tag (_rank_tagged_chain).
        self._tp_rank = 0
        self._tp_size = 1
        self._deployment_id = 0
        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            self._tp_rank = int(get_tensor_model_parallel_rank())
            self._tp_size = int(get_tensor_model_parallel_world_size())
        except Exception:
            # vLLM parallel state not initialized (e.g. in unit tests or
            # driver-side smokes). Fall back to TP=1.
            pass

        # Option W: every rank of the same vLLM deployment must share a
        # deployment_id so the storage server can bind them into a
        # tp_group and fan out sink writes across all ranks. Rank 0
        # generates and NCCL-broadcasts; other ranks receive.
        if self._tp_size > 1:
            try:
                import torch.distributed as dist
                from vllm.distributed.parallel_state import get_tp_group
                tp_group = get_tp_group()
                dev_group = tp_group.device_group
                if self._tp_rank == 0:
                    import secrets as _secrets
                    # 63 bits so the value fits in signed int64 used for
                    # the NCCL broadcast tensor; still plenty of entropy
                    # for uniqueness across concurrent deployments.
                    self._deployment_id = _secrets.randbits(63)
                id_tensor = torch.tensor([int(self._deployment_id)],
                                          dtype=torch.int64,
                                          device=torch.device(
                                              f"cuda:{torch.cuda.current_device()}"))
                dist.broadcast(id_tensor, src=tp_group.first_rank,
                                group=dev_group)
                self._deployment_id = int(id_tensor.item()) & 0x7FFFFFFFFFFFFFFF
            except Exception as e:
                logger.warning("IcmsConnector: deployment_id broadcast failed "
                                "(%s); falling back to rank-independent mode.",
                                e)
                self._deployment_id = 0

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

        # Skip KV extraction when reading from ICMS (Phase 2).
        # The context is already stored — no need to re-extract on the
        # critical TTFT path.  Set in on_step_start when stored chains
        # exist for the incoming request.
        self._skip_extract: bool = False


        # Pending notifications for the scheduler (stored chain info).
        # Drained into IcmsConnectorMetadata by the facade.
        self._pending_stored_notifications: list[tuple[list[int], int]] = []

        # Deferred write pipeline: extract (GPU→CPU, summary compute) +
        # flush (WriteGroup RPCs) moved off the vLLM wait_for_save critical
        # path. Worker thread runs tasks after wait_for_save returns. Drain
        # happens in on_request_finished (after first-token emission, off
        # TTFT). Reuses self._score_lock for client-call serialization so
        # main-thread Score/FetchAll don't race with background WriteGroup.
        self._write_pipeline = _WritePipeline()

        # TTFT-breakdown instrumentation. Per-request phase timestamps +
        # per-hook accumulators collected across the prefill critical path
        # (on_step_start → wait_for_layer × 48 → wait_for_pending_writes).
        # Enabled when ICMS_TTFT_BREAKDOWN=1.
        self._ttft_enabled = os.environ.get("ICMS_TTFT_BREAKDOWN", "1") != "0"
        self._ttft: dict[str, dict] = {}

        self._connect()

    def _connect(self):
        if self._use_rdma:
            if not _HAVE_RDMA:
                raise ImportError(
                    "icms_rdma=True but pyverbs is not installed. "
                    "Install python3-pyverbs (apt) or pyverbs (pip)."
                )
            cfg = RdmaTransportConfig(
                server_host=self._rdma_server_host,
                tcp_port=self._rdma_port,
                ib_dev_name=self._rdma_ib_dev,
            )
            self._client = RdmaIcmsClient(cfg)
            transport_desc = f"rdma://{self._rdma_server_host}:{self._rdma_port}"
        else:
            self._client = IcmsClient(self._socket_path)
            transport_desc = self._socket_path

        # Stagger worker connects so multiple TP ranks don't all hit BF2's
        # accept queue in the same millisecond — RDMA Hello races have
        # been observed at TP>1 (TP1 timed out while TP0's hello was
        # still in-flight, even though BF2 was healthy).
        import time as _t
        if self._tp_size > 1 and self._tp_rank > 0:
            _t.sleep(0.5 * self._tp_rank)

        # BF2 occasionally dies between the main-process restart and when
        # individual workers try to connect. Retry with auto-restart on
        # connect/hello failure. Catch is broad on purpose: ConnectionError,
        # OSError, IcmsError (Hello failed), TimeoutError. Disabled via
        # ICMS_BF2_AUTORESTART=0.
        _max_attempts = 4
        for _attempt in range(_max_attempts):
            try:
                self._client.connect()
                ack = self._client.hello(self._model_name,
                                         tp_rank=self._tp_rank,
                                         tp_size=self._tp_size,
                                         deployment_id=self._deployment_id)
                break  # success
            except Exception as _e:  # broad: any connect/hello failure
                if _attempt == _max_attempts - 1:
                    raise
                logger.warning(
                    "[icms-connect] attempt %d/%d failed: %s; trying "
                    "BF2 auto-restart and retrying (tp_rank=%d)",
                    _attempt + 1, _max_attempts, _e, self._tp_rank)
                try:
                    from lib.bf2_runner import ensure_bf2_running  # type: ignore
                    ensure_bf2_running(probe_host=self._rdma_server_host,
                                        probe_port=self._rdma_port,
                                        timeout=30.0)
                except Exception:
                    pass  # auto-restart helper unavailable; just retry
                # Re-create the client object — the previous one may have
                # left dangling RDMA state after a failed connect.
                if self._use_rdma:
                    self._client = RdmaIcmsClient(cfg)
                else:
                    self._client = IcmsClient(self._socket_path)
                # Backoff with jitter on tp_rank so retries don't re-collide.
                _t.sleep(2.0 + 0.5 * _attempt + 0.3 * self._tp_rank)
        self._geom = find_model(self._model_name) or ModelGeometry(
            name=self._model_name,
            num_layers=ack.num_layers,
            num_kv_heads=ack.num_kv_heads,
            head_dim=ack.head_dim,
            elem_bytes=ack.elem_bytes,
        )
        # Apply scored-layers mask from env. Clamp to the model's actual
        # num_layers so a global env spec like "0,6,12,18,24,30,36,42"
        # works across heterogeneous models — bits beyond num_layers get
        # cleared (matches the server-side clamp at handlers.cc::handle_hello).
        # Without this clamp, llama-3 (32 layers) packs 8 summary slots
        # but the server expects 6 → "WriteGroup: bytes underflow" → kError
        # on the very first WriteGroup. Mistral (40 layers) hit the same
        # asymmetry. (2026-04-26 llama3 dive.)
        _scored_spec = os.environ.get("ICMS_SCORED_LAYERS", "")
        _scored_mask = parse_scored_layers(_scored_spec)
        if _scored_mask != 0 and self._geom.num_layers < 64:
            _valid = ((1 << self._geom.num_layers) - 1)
            _scored_mask &= _valid
        if _scored_mask != 0:
            self._geom = _dataclasses.replace(self._geom,
                                               scored_layers_mask=_scored_mask)
            logger.info("[icms] scored_layers mask=0x%x popcount=%d "
                         "(num_layers=%d)",
                         _scored_mask, self._geom.num_scored_layers,
                         self._geom.num_layers)
        # KV on-disk layout from env. Must match server --kv-layout.
        _kv_layout_str = os.environ.get("ICMS_KV_LAYOUT", "layer-major").lower()
        if _kv_layout_str in ("page-major", "page_major", "page"):
            self._geom = _dataclasses.replace(self._geom,
                                               kv_layout=KvLayout.PAGE_MAJOR)
            logger.info("[icms] kv_layout=page-major")
        else:
            logger.info("[icms] kv_layout=layer-major")
        # Update allocator with actual kv_page_bytes and num_layers from
        # model geometry. The allocator pairs end-to-end slack with total
        # KV bytes (all layers × pages × kv_page_bytes), so num_layers is
        # required for a well-defined demand number.
        if self._adaptive_allocator is not None:
            self._adaptive_allocator._kv_page_bytes = self._geom.kv_page_bytes
            self._adaptive_allocator._num_layers = int(self._geom.num_layers)

        # Sink sizing: needs to hold one server Phase-2 dump at a time.
        # - Score path (stride-gated): score_stride layers × k pages.
        # - FetchAll path (B, budget=1.0): num_layers × k pages (one call
        #   covers every reuse layer, not just score_stride).
        # Size for the worst case so B doesn't overflow → server's sink-
        # bounds check rejects writes → GPU-direct index_select trips a
        # CUDA OOB assertion.
        per_layer_bytes = self._k * self._geom.kv_page_bytes
        sink_layers = max(self._score_stride, int(self._geom.num_layers))
        total_sink = sink_layers * per_layer_bytes
        # Allocate a per-layer ready-flag sink (one u32 per model layer)
        # alongside the main KV sink. The server flips slots via small
        # RDMA writes and the connector polls them to overlap compute
        # with remaining layer transfers.
        flag_slots = int(self._geom.num_layers)
        # At TP>1, each rank registers its sink on its own GPU. Use the
        # current CUDA device (vLLM has already set it for this worker) to
        # avoid cross-GPU addressing. The configured _gpu_device string
        # (cuda:0 by default) only applies at TP=1.
        gpu_dev = self._gpu_device
        if self._tp_size > 1:
            try:
                cur = torch.cuda.current_device()
                gpu_dev = f"cuda:{int(cur)}"
            except Exception:
                gpu_dev = f"cuda:{self._tp_rank}"
        if self._gpu_direct and self._use_rdma:
            sink = self._client.register_gpu_sink(
                total_sink, gpu_dev, flag_slots=flag_slots)
            sink_desc = f"gpu_direct({gpu_dev})"
        else:
            sink = self._client.register_sink(total_sink)
            sink_desc = "host"
        self._sink_pool = _SinkSlotPool(sink, per_layer_bytes, _SINK_SLOTS)
        self._sink_per_layer_bytes = per_layer_bytes
        logger.info(
            "IcmsConnector worker: connected to %s, model=%s, "
            "sink=%s %d layers × %d B/layer = %d B total, score_stride=%d",
            transport_desc, self._model_name, sink_desc,
            self._score_stride, per_layer_bytes, total_sink,
            self._score_stride,
        )

    def _rank_chain(self, chain):
        """Apply this worker's rank tag to an outbound chain (TP sharding)."""
        return _rank_tagged_chain(self._tp_rank, chain)

    def _get_stored_context_groups(self, chain: list[int]) -> int:
        """Find the longest stored chain prefix matching the given chain.

        Returns the number of groups written for that prefix (0 if none).
        This allows a new request to know how many context pages were
        stored by a prior request with the same prefix.
        """
        best = 0
        best_src_len = 0
        best_src_n = 0
        for stored_chain, n_groups in self._stored_chain_groups:
            match_len = 0
            for a, b in zip(chain, stored_chain):
                if a == b:
                    match_len += 1
                else:
                    break
            if match_len > 0 and match_len >= best:
                candidate = min(n_groups, match_len)
                if candidate > best:
                    best = candidate
                    best_src_len = len(stored_chain)
                    best_src_n = n_groups
        if not hasattr(self, "_diag_get_stored_count"):
            self._diag_get_stored_count = 0
        if self._diag_get_stored_count < 20:
            logger.debug(
                "[diag-match] chain_len=%d stored_entries=%d match_best=%d "
                "(from entry len=%d n_groups=%d)",
                len(chain), len(self._stored_chain_groups),
                best, best_src_len, best_src_n,
            )
            self._diag_get_stored_count += 1
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
        # Drain + stop the write pipeline before we tear down the
        # client (pending WriteGroup RPCs would fail otherwise).
        try:
            self._write_pipeline.drain(timeout=10.0)
            self._write_pipeline.shutdown(timeout=5.0)
        except Exception:
            logger.exception("shutdown: write-pipeline drain/stop failed")
        # Evict all active requests.
        for rid, rs in list(self._requests.items()):
            if rs.chain:
                try:
                    self._client.evict(self._rank_chain(rs.chain))
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

    # ─── TTFT-breakdown helpers ──────────────────────────────────────────

    def _ttft_reset(self, rid: str, t_start: float):
        if not self._ttft_enabled:
            return
        self._ttft[rid] = {
            "t_step_start": t_start,
            "t_first_wait_layer": None,
            "t_last_fetch_done": None,
            "num_waits": 0,
            "wait_spin_us_total": 0.0,
            "wait_apply_us_total": 0.0,
            "num_scores": 0,
            "score_rpc_us_total": 0.0,
            "num_fetch_alls": 0,
            "fetch_all_rpc_us_total": 0.0,
        }

    def _ttft_add(self, rid: str, **deltas):
        if not self._ttft_enabled:
            return
        entry = self._ttft.get(rid)
        if entry is None:
            return
        for k, v in deltas.items():
            # Fields prefixed with "t_" are timestamps: overwrite.
            # Everything else is a numeric accumulator.
            if k.startswith("t_"):
                entry[k] = v
            elif k in entry and isinstance(entry[k], (int, float)):
                entry[k] = entry[k] + v
            else:
                entry[k] = v

    def _ttft_stamp_once(self, rid: str, field: str, t: float):
        if not self._ttft_enabled:
            return
        entry = self._ttft.get(rid)
        if entry is None:
            return
        if entry.get(field) is None:
            entry[field] = t

    def _ttft_emit(self, t_save_enter: float):
        if not self._ttft_enabled or not self._ttft:
            return
        for rid, e in list(self._ttft.items()):
            t_step = e.get("t_step_start")
            if t_step is None:
                continue
            t_first = e.get("t_first_wait_layer") or t_save_enter
            t_fetch = e.get("t_last_fetch_done") or t_first
            step_to_first_ms = (t_first - t_step) * 1e3
            first_to_fetch_ms = (t_fetch - t_first) * 1e3
            fetch_to_save_ms = (t_save_enter - t_fetch) * 1e3
            total_ms = (t_save_enter - t_step) * 1e3
            logger.debug(
                "[ttft-breakdown] rid=%s total=%.1fms "
                "step_to_first_layer=%.1fms first_to_fetch_done=%.1fms "
                "fetch_to_wait_for_save=%.1fms | "
                "waits=%d wait_spin=%.1fms wait_apply=%.1fms | "
                "scores=%d score_rpc=%.1fms | "
                "fetch_alls=%d fetch_all_rpc=%.1fms",
                rid, total_ms,
                step_to_first_ms, first_to_fetch_ms, fetch_to_save_ms,
                e["num_waits"],
                e["wait_spin_us_total"] / 1e3,
                e["wait_apply_us_total"] / 1e3,
                e["num_scores"],
                e["score_rpc_us_total"] / 1e3,
                e["num_fetch_alls"],
                e["fetch_all_rpc_us_total"] / 1e3,
            )
            # Clear after emit so next step doesn't double-log.
            self._ttft.pop(rid, None)

    def _slack_emit_and_reset(self) -> None:
        """ICMS_DIAG_SLACK: format the per-layer bracketed slack table.

        Three checkpoints per layer L (timestamps relative to step start):
          t_post[L]  = on_layer_score(L) entry  ≈ compute end of L-1
          t_pre[L]   = on_layer_score(L) exit   ≈ compute start of L
          t_call[L]  = wait_for_layer(L) entry  ≈ start of L's attention
        Plus three flag observations (ready at each checkpoint) and the
        spin-observed t_after_spin[L] for the not-ready-at-call case.

        Categorisation per layer:
          IDLE    : flag True at post-hook → arrived ≤ t_post; idle ≥ (t_call - t_post)
          NEAR    : False at post, True at pre → arrived in (post, pre);
                    idle ∈ [0, t_call - t_pre]; small slack
          BRINK   : False at pre, True at call → arrived in (pre, call);
                    no idle, but no stall either
          STALL   : False at call → captured by spin; stall = (t_after_spin - t_call)
        """
        t_step = getattr(self, "_slack_t_step", None)
        t_post = getattr(self, "_slack_t_post_hook", None)
        if t_step is None or t_post is None:
            return
        t_pre = self._slack_t_pre_hook
        t_call = self._slack_t_called
        t_aspin = self._slack_t_after_spin
        f_post = self._slack_flag_at_post
        f_pre = self._slack_flag_at_pre
        f_call = self._slack_flag_at_call
        n = len(t_post)

        # Stop the C++ poller if it ran this step. arrivals_ns is in the
        # same clock as time.perf_counter() (CLOCK_MONOTONIC on Linux),
        # so we can subtract Python timestamps directly.
        arrivals_ns: list[int] | None = None
        if self._slack_poller is not None:
            try:
                arrivals_ns = list(self._slack_poller.stop())
            except Exception:
                logger.exception("ArrivalPoller.stop failed; using inline probes")
                arrivals_ns = None

        idle_ms = 0.0
        stall_ms = 0.0
        n_idle = n_near = n_brink = n_stall = 0
        rows = []
        for L in range(n):
            tp = t_post[L]; tr = t_pre[L]; tc = t_call[L]; tas = t_aspin[L]
            if tp is None or tc is None:
                rows.append((L, "NA", 0.0))
                continue
            t_arr_s: float | None = None
            if (arrivals_ns is not None
                    and 0 <= L < len(arrivals_ns)
                    and arrivals_ns[L] > 0):
                t_arr_s = arrivals_ns[L] / 1e9
            if t_arr_s is not None:
                # Exact arrival timestamp from the C++ poller.
                if t_arr_s <= tc:
                    # Idle = compute waited from arrival to t_call.
                    delta = (tc - t_arr_s) * 1e3
                    idle_ms += delta
                    if t_arr_s <= tp:
                        n_idle += 1
                        rows.append((L, "IDLE", delta))
                    elif tr is not None and t_arr_s <= tr:
                        n_near += 1
                        rows.append((L, "NEAR", delta))
                    else:
                        n_brink += 1
                        rows.append((L, "BRINK", delta))
                else:
                    # Stall = compute waited from t_call to arrival.
                    delta = (t_arr_s - tc) * 1e3
                    stall_ms += delta
                    n_stall += 1
                    rows.append((L, "STALL", -delta))
            else:
                # Fallback: flag-observation lower bound.
                fp = f_post[L]; fr = f_pre[L]; fc = f_call[L]
                if fp is True:
                    lb = (tc - tp) * 1e3
                    idle_ms += lb
                    n_idle += 1
                    rows.append((L, "IDLE", lb))
                elif fr is True:
                    ub = (tc - tr) * 1e3 if tr is not None else 0.0
                    n_near += 1
                    rows.append((L, "NEAR", ub))
                elif fc is True:
                    n_brink += 1
                    rows.append((L, "BRINK", 0.0))
                else:
                    stall = ((tas - tc) * 1e3) if tas is not None else 0.0
                    stall_ms += stall
                    n_stall += 1
                    rows.append((L, "STALL", -stall))

        per_layer = " ".join(f"{L}:{cat}:{v:.2f}" for (L, cat, v) in rows)
        src = "cpp" if arrivals_ns is not None else "inline"
        logger.info(
            "[diag-slack] src=%s n=%d idle=%d near=%d brink=%d stall=%d "
            "idle_ms=%.2f stall_ms=%.2f per_layer=[%s]",
            src, n, n_idle, n_near, n_brink, n_stall,
            idle_ms, stall_ms, per_layer,
        )
        # Reset state for next step.
        for L in range(n):
            t_post[L] = None; t_pre[L] = None; t_call[L] = None
            t_aspin[L] = None
            f_post[L] = None; f_pre[L] = None; f_call[L] = None
        self._slack_t_step = None

    # ─── metadata drain (C1) ─────────────────────────────────────────────

    def on_step_start(self, meta: IcmsConnectorMetadata):
        """Called from start_load_kv. Drains scheduler metadata into caches."""
        t_step = time.perf_counter()
        self.stats.advance_step()
        # Reset prefill_done when a new request arrives (new chain delivered).
        if meta.new_chains:
            self._prefill_done = False
            for rid in meta.new_chains.keys():
                self._ttft_reset(rid, t_step)

        # ICMS_DIAG_SLACK=1: inline-probe slack tracking. Earlier polling
        # thread approach GIL-thrashed wait_for_layer's spin loop at TP=2,
        # wedging on shm_broadcast 60s. Inline probes at three checkpoints
        # per layer (on_layer_score entry/exit + wait_for_layer entry)
        # bracket the flag-flip time without any thread:
        #   case 1: flag True at on_layer_score entry  → idle ≥ (called - post_hook)
        #   case 2: flag True at on_layer_score exit   → small idle / small stall
        #   case 3: flag True at wait_for_layer entry  → small stall
        #   case 4: flag False, captured by spin       → stall = (after_spin - called)
        if (os.environ.get("ICMS_DIAG_SLACK") == "1"
                and self._sink_pool is not None):
            sink = self._sink_pool.sink
            flag_count = int(getattr(sink, "flag_count", 0) or 0)
            if flag_count > 0:
                # Clear ALL flags so stale True values from the previous
                # step don't contaminate this step's first-observed-true
                # readings. Without this, layers that aren't covered by
                # an early stride would still hold last step's flip.
                sink.clear_ready_flags()
                self._slack_t_step = t_step
                self._slack_t_post_hook: list = [None] * flag_count
                self._slack_flag_at_post: list = [None] * flag_count
                self._slack_t_pre_hook: list = [None] * flag_count
                self._slack_flag_at_pre: list = [None] * flag_count
                self._slack_t_called: list = [None] * flag_count
                self._slack_flag_at_call: list = [None] * flag_count
                self._slack_t_after_spin: list = [None] * flag_count
                # Start the C++ poll thread (GIL released) — records exact
                # arrival timestamps for each flag without contending with
                # the main worker thread. Falls back to inline probes when
                # the binding doesn't expose the poller.
                if (_HAVE_ARRIVAL_POLLER
                        and getattr(sink, "flag_buffer", None) is not None):
                    if self._slack_poller is None:
                        self._slack_poller = _ArrivalPoller()
                    try:
                        self._slack_poller.start(sink.flag_buffer)
                    except Exception:
                        logger.exception(
                            "ArrivalPoller.start failed; falling back to "
                            "inline-probe slack data this step")
                        self._slack_poller = None

        # Dedup-aware extraction: skip blocks that already live in a
        # group covered by a previously-stored chain prefix. Saves the
        # GPU→CPU copy + summary cost for the stored portion, while
        # still extracting the novel suffix.
        #
        # BUG-N2: scheduler tells us per-rid whether the upcoming step
        # will form any new complete group. We skip extract (entire
        # GPU→CPU copy + AllGather × num_layers) iff EVERY active rid
        # in this step is in skip_extract_rids — conservative: a single
        # non-skipping rid in the batch makes the whole step extract
        # normally. Decision is metadata-driven (broadcast identically
        # to both TP ranks) so no AllGather asymmetry.
        active_rids = set(meta.new_chains) | set(self._requests)
        self._skip_extract = bool(active_rids) and active_rids.issubset(
            meta.skip_extract_rids)
        # BUG-N13 Phase 1 diag: per-step skip_extract decision trace.
        # Toggle with ICMS_DIAG_N13=1.
        if os.environ.get("ICMS_DIAG_N13", "0") == "1":
            req_summaries = []
            for prs in meta.requests:
                req_summaries.append(
                    f"rid={prs.request_id} "
                    f"computed=[{prs.num_computed_tokens_start},"
                    f"{prs.num_computed_tokens_end}] "
                    f"complete_grp_after={prs.num_computed_tokens_end // (PAGE_TOKENS * _GROUP_BLOCKS)}"
                )
            logger.info(
                "[diag-step] rank=%d new_chains=%d active_rids=%d "
                "skip_extract=%s skip_set=%s requests=[%s]",
                self._tp_rank,
                len(meta.new_chains),
                len(active_rids),
                self._skip_extract,
                sorted(meta.skip_extract_rids),
                "; ".join(req_summaries))
        for rid, chain in meta.new_chains.items():
            n_stored = self._get_stored_context_groups(chain)
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.stored_groups = n_stored
            if os.environ.get("ICMS_DIAG_N13", "0") == "1":
                logger.info(
                    "[diag-step] rank=%d new_chain rid=%s chain_len=%d "
                    "stored_groups=%d", self._tp_rank, rid, len(chain),
                    n_stored)
        for rid, chain in meta.new_chains.items():
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.chain = chain
            self._last_chain_for_rid[rid] = chain
        for rid, bids in meta.block_id_maps.items():
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.block_ids = bids
        # Client-side preload DISABLED by default — the server now kicks off
        # the same async preload internally on first Score (see
        # kick_off_summary_preload in handlers.cc), which avoids the
        # client-side cold-frame penalty entirely. Set
        # ICMS_DISABLE_PRELOAD=0 to re-enable the client-fired path.
        if int(os.environ.get("ICMS_DISABLE_PRELOAD", "1")) == 0:
            for rid, chain in meta.new_chains.items():
                if chain:
                    self._fire_preload(rid, chain)

    def _fire_preload(self, request_id: str, chain: list[int]):
        """Fire a one-way summary preload on the scheduler thread.

        Sends a preload frame and returns immediately — no client-side
        wait. The server populates its DRAM cache for this request_id
        while the forward pass starts on the GPU. The first real Score
        on the same connection is FIFO-ordered behind the preload on
        the reactor, so it's guaranteed to hit the cache.
        """
        try:
            icms_rid = self._icms_request_id(request_id, 0)
            self._client.preload(request_id=icms_rid,
                                 chain=self._rank_chain(chain))
        except Exception as e:
            logger.debug("preload failed for %s: %s", request_id, e)

    # ─── layer hooks (from save_kv_layer kwargs) ─────────────────────────

    def on_layer_all_pages(self, next_layer_idx, budget, stats,
                            connector_meta=None):
        """Budget >= 1.0: fetch every stored page via kFetchAll (no scoring)."""
        # ICMS_DIAG_SLACK probe #1: post-hook of L-1. Mirrors what
        # on_layer_score / on_layer_reuse do — required so B's slack
        # output is observable. Without this, the kFetchAll dispatch
        # path leaves t_post_hook[L]=None for every layer and the
        # diag-slack consumer drops to NA, producing all-zero state
        # counters and idle_ms_mean=stall_ms_mean=0.
        self._slack_probe_post_hook(next_layer_idx)
        try:
            self._on_layer_all_pages_impl(
                next_layer_idx, budget, stats, connector_meta)
            # Probe #2: pre-hook of L, just before L's forward starts.
            self._slack_probe_pre_hook(next_layer_idx)
        except Exception:
            logger.exception("on_layer_all_pages FAILED for layer %d",
                             next_layer_idx)

    def _on_layer_all_pages_impl(self, next_layer_idx, budget, stats,
                                  connector_meta=None):
        # Build request list in batch order (mirrors _on_layer_score_impl).
        requests_to_fetch = []
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
                    requests_to_fetch.append((req_idx, rid, rs))
        if not requests_to_fetch:
            for req_idx, (rid, rs) in enumerate(self._requests.items()):
                if rs.chain:
                    requests_to_fetch.append((req_idx, rid, rs))
        if not requests_to_fetch:
            return

        for req_idx, rid, rs in requests_to_fetch:
            self._fetch_all_one_request(
                rid, rs, req_idx, next_layer_idx, budget, stats)

    def _fetch_all_one_request(self, rid, rs, req_idx, next_layer_idx,
                                budget, stats):
        """FetchAll path: one RPC per request that covers ALL layers.

        Budget=1.0 means the caller wants every page for every layer. No
        scoring, no adaptive budget, no per-stride-group budget change —
        so there's nothing to recompute at layer 6/12/18/… Issuing a
        single FetchAll(reuse_through_layer=num_layers-1) lets the
        server stream the full 48-layer KV in one job and eliminates
        the 7 additional sync round-trips per forward pass.

        Subsequent scoring-boundary calls for the same request promote
        pre-populated _pending_reuse[layer] entries into _pending_scores
        (same pattern as on_layer_reuse).
        """
        num_layers = self._geom.num_layers if self._geom else 48
        attn_layer_name = f"model.layers.{next_layer_idx}.self_attn.attn"

        # Fast path: earlier scoring boundary already issued the single
        # full-request FetchAll. Just promote pre-populated reuse entry
        # into pending_scores for this layer.
        if getattr(rs, "_fetch_all_complete", False):
            import copy
            with self._score_lock:
                reuse_entry = self._pending_reuse.get(
                    attn_layer_name, {}).pop(rid, None)
            if reuse_entry is None:
                logger.debug(
                    "fetch_all promote: no _pending_reuse for rid=%s layer=%d",
                    rid, next_layer_idx)
                return
            reply, reuse_offsets = reuse_entry
            promoted = copy.copy(reply)
            promoted.sink_offsets = reuse_offsets
            with self._score_lock:
                self._pending_scores.setdefault(attn_layer_name, {})[rid] = (
                    promoted, req_idx)
            return

        # Slow path: first scoring boundary for this request — issue
        # the single full-request FetchAll covering every layer.
        icms_rid = self._icms_request_id(rid, 0)
        # BUG-N7: read the cached per-request value populated by
        # on_step_start instead of rescanning _stored_chain_groups
        # (O(N stored × chain_len)) on every layer in the hot path.
        stored_groups = rs.stored_groups
        effective_groups = max(rs.num_groups_written, stored_groups)
        total_pages = effective_groups * _GROUP_BLOCKS

        if not getattr(rs, "_budget_logged", False):
            # Once-per-request decision marker the perf-sweep scrapes.
            logger.info(
                "icms_budget rid=%s layer=%d src=fetch_all budget=%.3f k=%d "
                "total_pages=%d",
                rid, next_layer_idx, budget, total_pages, total_pages,
            )
            rs._budget_logged = True

        reuse_through = num_layers - 1

        self._sink_pool.sink.clear_ready_flags()

        t_start = time.perf_counter()
        try:
            reply = self._client.fetch_all(
                request_id=icms_rid,
                chain=self._rank_chain(rs.chain),
                layer=next_layer_idx,
                sink=self._sink_pool.sink,
                reuse_through_layer=reuse_through,
                # Reply-early: server ships the FetchAll reply as soon as
                # page_ids are known; Phase-2 KV writes + per-layer flag
                # flips run in background so the GPU forward pass overlaps
                # with the transfer. Note: reply's sink_write_ns /
                # server_ingest_to_ready_ns are reported as 0 in this
                # mode — the transfer wall-time shows up as the sum of
                # per-layer wait_spin on the client side.
                use_flags=True,
            )
            t_end = time.perf_counter()
            rs._last_storage_concurrent = getattr(
                reply, 'concurrent_requests', 0)
            rpc_ms = (t_end - t_start) * 1e3
            slow_tag = " SLOW" if rpc_ms > 5.0 else ""
            logger.debug(
                "[rpc] src=fetch_all mode=sync rid=%s layers=%d..%d k=%d "
                "rpc=%.2fms score=%.2fms summary_read=%.2fms "
                "sink_write=%.2fms server=%.2fms hit=%d concurrent=%d "
                "data=%.1fMB%s",
                rid, next_layer_idx, reuse_through, len(reply.page_ids),
                rpc_ms,
                reply.score_ns / 1e6,
                reply.summary_read_ns / 1e6,
                reply.sink_write_ns / 1e6,
                reply.server_ingest_to_ready_ns / 1e6,
                int(getattr(reply, 'cache_hit', 0)),
                int(getattr(reply, 'concurrent_requests', 0)),
                len(reply.page_ids) * self._geom.kv_page_bytes
                * (reuse_through - next_layer_idx + 1) / (1 << 20),
                slow_tag,
            )
            self.stats.record_score(
                (t_end - t_start) * 1e6,
                reply.cache_hit,
                list(reply.page_ids),
                next_layer_idx,
                scores=list(reply.scores),
            )
            self._ttft_add(
                rid,
                num_fetch_alls=1,
                fetch_all_rpc_us_total=(t_end - t_start) * 1e6,
                t_last_fetch_done=t_end,
            )
            # Reply-early (use_flags=True): server flips per-layer
            # flags via RDMA as Phase-2 writes complete — don't force
            # them ready locally or wait_for_layer will skip the flag
            # spin and read KV before it lands.
            sink = self._sink_pool.sink
            _ = sink
            with self._score_lock:
                self._pending_scores.setdefault(attn_layer_name, {})[rid] = (
                    reply, req_idx)
            # Reply-specific per-layer stride: server packs sink as
            # k*kv_page_bytes per layer (NOT self._k — that's just the
            # cap used to size the sink). Using self._k here would send
            # reuse offsets past the server's actual layer-delta stride.
            actual_k = len(reply.page_ids)
            per_layer_bytes = actual_k * self._geom.kv_page_bytes
            for delta in range(1, reuse_through - next_layer_idx + 1):
                reuse_layer = next_layer_idx + delta
                reuse_attn = f"model.layers.{reuse_layer}.self_attn.attn"
                reuse_offsets = [off + delta * per_layer_bytes
                                 for off in reply.sink_offsets]
                with self._score_lock:
                    self._pending_reuse.setdefault(reuse_attn, {})[rid] = (
                        reply, reuse_offsets)
            rs._fetch_all_complete = True
        except Exception as e:
            t_end = time.perf_counter()
            self.stats.record_score(
                (t_end - t_start) * 1e6, False, [], next_layer_idx,
            )
            logger.debug("fetch_all failed layer %d: %s", next_layer_idx, e)

    def on_layer_reuse(self, next_layer_idx, budget, stats):
        """Reuse layer: KV already in sink from the stride group's Score call.

        Reads the pre-filled KV from the sink at the pre-computed offsets
        and stores as a pending score result. The wait_for_layer path
        picks it up and populates the fetch buffer.
        """
        # ICMS_DIAG_SLACK probe #1: post-hook of L-1. Non-stride layers
        # route here directly via save_kv_layer(quest_reuse_selection=
        # True), so without this probe slack data only captured stride
        # boundaries.
        self._slack_probe_post_hook(next_layer_idx)
        import copy
        attn_layer_name = f"model.layers.{next_layer_idx}.self_attn.attn"
        with self._score_lock:
            reuse_data = self._pending_reuse.pop(attn_layer_name, None)
        if reuse_data is None:
            self._slack_probe_pre_hook(next_layer_idx)
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
        self._slack_probe_pre_hook(next_layer_idx)

    def on_layer_score(self, next_layer_idx, quest_query, budget, stats,
                        connector_meta=None):
        """CPU-scoring path: Q arrived, fire Score against icms (C9 State 2/3).

        Stride-gated: we only fire a fresh Score at stride boundaries
        (next_layer_idx ∈ {0, stride, 2*stride, ...}). In between, the
        prior Score's Phase-2 KV writes already cover this layer — we
        just route to ``on_layer_reuse`` which promotes the pre-stashed
        reuse entry into _pending_scores so wait_for_layer can consume it.

        Layer 0 is delivered by quest_hooks' forward-pre-hook (Q_0 is
        not reachable from a post-hook since there's no layer −1). The
        pattern covers layers 0..num_layers−1 uniformly.
        """
        # ICMS_DIAG_LAYER_ARRIVAL=1: log on_layer_score entry/exit timing
        # relative to step start so we can see where the per-stride
        # ~1050ms gap goes (Score RPC vs other work).
        _diag = os.environ.get("ICMS_DIAG_LAYER_ARRIVAL") == "1"
        if _diag:
            t_entry = time.perf_counter()
            is_boundary = (next_layer_idx % self._score_stride) == 0
            t_step = None
            if (connector_meta is not None
                    and isinstance(connector_meta, IcmsConnectorMetadata)
                    and connector_meta.requests):
                rid0 = connector_meta.requests[0].request_id
                e = self._ttft.get(rid0)
                if e is not None:
                    t_step = e.get("t_step_start")
            entry_rel_ms = ((t_entry - t_step) * 1e3) if t_step else -1.0
        # ICMS_DIAG_SLACK probe #1: post-hook of L-1, just after L-1
        # forward ended. Records wall time and whether layer L's ready
        # flag is already up.
        self._slack_probe_post_hook(next_layer_idx)
        try:
            if (next_layer_idx % self._score_stride) != 0:
                self.on_layer_reuse(next_layer_idx, budget, stats)
                if _diag:
                    t_exit = time.perf_counter()
                    exit_rel_ms = ((t_exit - t_step) * 1e3) if t_step else -1.0
                    logger.info(
                        "[diag-ols] layer=%d boundary=0 entry=%.2fms exit=%.2fms "
                        "duration=%.2fms",
                        next_layer_idx, entry_rel_ms, exit_rel_ms,
                        (t_exit - t_entry) * 1e3,
                    )
                self._slack_probe_pre_hook(next_layer_idx)
                return
            self._on_layer_score_impl(
                next_layer_idx, quest_query, budget, stats, connector_meta)
            if _diag:
                t_exit = time.perf_counter()
                exit_rel_ms = ((t_exit - t_step) * 1e3) if t_step else -1.0
                logger.info(
                    "[diag-ols] layer=%d boundary=1 entry=%.2fms exit=%.2fms "
                    "duration=%.2fms",
                    next_layer_idx, entry_rel_ms, exit_rel_ms,
                    (t_exit - t_entry) * 1e3,
                )
            self._slack_probe_pre_hook(next_layer_idx)
        except Exception:
            logger.exception("on_layer_score FAILED for layer %d", next_layer_idx)

    def _slack_probe_post_hook(self, layer_idx: int) -> None:
        """ICMS_DIAG_SLACK probe #1: just after L-1's forward ended,
        i.e. at the per-layer dispatch entry for layer L. Records the
        wall time and whether L's ready flag is already up.

        Used by on_layer_score, on_layer_reuse, and on_layer_all_pages
        — must fire from all three so aggregate_slack can compute
        per-layer slack regardless of which dispatch path served the
        layer. Without it, B's slack is all-NA because t_post_hook[L]
        stays None for every layer.
        """
        arr = getattr(self, "_slack_t_post_hook", None)
        if (arr is not None
                and 0 <= layer_idx < len(arr)
                and self._sink_pool is not None):
            arr[layer_idx] = time.perf_counter()
            self._slack_flag_at_post[layer_idx] = bool(
                self._sink_pool.sink.is_layer_ready(layer_idx))

    def _slack_probe_pre_hook(self, layer_idx: int) -> None:
        """ICMS_DIAG_SLACK probe #2: just before layer L's forward starts."""
        arr = getattr(self, "_slack_t_pre_hook", None)
        if (arr is not None
                and 0 <= layer_idx < len(arr)
                and self._sink_pool is not None):
            arr[layer_idx] = time.perf_counter()
            self._slack_flag_at_pre[layer_idx] = bool(
                self._sink_pool.sink.is_layer_ready(layer_idx))

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

        # All layers of a prefill share one icms request_id so the
        # server-side per-request DRAM summary cache (populated by
        # preload) is reused across every Score. The stride-group
        # namespacing that previously lived here was for the bypassed
        # score cache only.
        icms_rid = self._icms_request_id(rid, 0)
        # Use stored context groups (from prior requests with same prefix)
        # if the current request hasn't written anything yet.
        # BUG-N7: cached on rs by on_step_start; avoids per-layer rescan.
        stored_groups = rs.stored_groups
        effective_groups = max(rs.num_groups_written, stored_groups)
        total_pages = effective_groups * _GROUP_BLOCKS

        # First-turn / empty-chain short-circuit. No stored pages means
        # nothing to score and no sink to fill — skip everything,
        # including the TP>1 NCCL AllGather(Q) + broadcast(reply) path.
        # This is symmetric across ranks (total_pages is derived from
        # scheduler-propagated state), so all ranks return together.
        if total_pages == 0:
            return

        # Determine effective budget: adaptive allocator or static.
        effective_budget = budget
        budget_source = "static"
        if self._adaptive_allocator is not None and total_pages > 0:
            # First layer of the request: register. Subsequent: recompute.
            if not hasattr(rs, '_adaptive_registered') or not rs._adaptive_registered:
                # Estimate new_tokens and cache_tokens from request state.
                cache_tokens = total_pages * PAGE_TOKENS
                # new_tokens for the slack-table lookup. quest_query is the
                # Q tensor for layer 0 of THIS step; shape[0] is the count
                # of new tokens being processed (i.e., pf in the perf-sweep
                # terminology). This is the reliable signal — the
                # scheduler-metadata fallback below was broken in V1: at
                # register-time, step_req.num_computed_tokens_end
                # - num_computed_tokens_start consistently returned 0,
                # capping new_tokens at the floor (16) for every prefill
                # regardless of pf. With new=16 ALL slack-table lookups hit
                # the small-pf row (~29 ms forward), producing a fixed
                # k≈123 cap and starving the allocator of actual headroom
                # at large pf. Verified + fixed 2026-04-27.
                new_tokens = max(16, cache_tokens // 10)  # last-resort estimate
                if (quest_query is not None
                        and hasattr(quest_query, 'shape')
                        and quest_query.ndim >= 1
                        and int(quest_query.shape[0]) >= 16):
                    new_tokens = int(quest_query.shape[0])
                elif connector_meta and connector_meta.requests:
                    step_req = connector_meta.requests[0]
                    new_tokens = max(16, step_req.num_computed_tokens_end
                                    - step_req.num_computed_tokens_start)
                effective_budget = self._adaptive_allocator.register_request(
                    rid, new_tokens=new_tokens, cache_tokens=cache_tokens,
                    total_cache_pages=total_pages)
                rs._adaptive_registered = True
                budget_source = "adaptive-new"
            else:
                effective_budget = self._adaptive_allocator.get_budget(rid)
                budget_source = "adaptive-cached"

        k = max(1, int(total_pages * effective_budget)) if total_pages > 0 else 1
        k = min(k, self._k)
        # Log the chosen budget on the first scoring call of each request
        # so perf scripts can recover it without waiting for the
        # end-of-run stats dump (which is only written on connector
        # shutdown). Scoring fires every score_stride layers starting at
        # layer score_stride, so the first trigger is next_layer_idx ==
        # score_stride. We also always log an "adaptive-new" decision
        # since that's the interesting per-request event.
        if budget_source == "adaptive-new" or not getattr(rs, "_budget_logged", False):
            # Once-per-request decision marker the perf-sweep scrapes.
            logger.info(
                "icms_budget rid=%s layer=%d src=%s budget=%.3f k=%d "
                "total_pages=%d",
                rid, next_layer_idx, budget_source, effective_budget,
                k, total_pages,
            )
            rs._budget_logged = True

        # Marshal query to fp32 numpy, mean-pooling Q heads per KV-head
        # group for GQA (matching QuestPageSelector.compute_page_scores).
        geom = self._geom
        if isinstance(quest_query, torch.Tensor):
            # ── DIAGNOSTIC: log forward batch size at the first scored layer.
            # quest_query.shape[0] is the actual number of forward tokens
            # vLLM ran through this layer. If vLLM honored the connector's
            # ext_tokens=16384, this would be 64 (just the new prompt tail);
            # if vLLM ran the full prefill anyway, it would be 16448.
            # Fires once per request via the _budget_logged flag pattern.
            if not getattr(rs, "_q_shape_logged", False):
                logger.debug(
                    "[diag-fwd-shape] rid=%s layer=%s q.shape=%s tp_rank=%d",
                    rid, getattr(rs, "current_layer", "?"),
                    tuple(quest_query.shape), self._tp_rank)
                rs._q_shape_logged = True
            # Mean-pool on GPU FIRST, then move the small result to CPU.
            # Pre-2026-04-26: did `.to(cpu, fp32)` on the full Q tensor first,
            # then mean-pooled. At 128k context with qwen3 TP=2 that's
            # [131072, 16, 128] bf16 = 537 MB GPU→CPU per stride boundary,
            # consuming ~1050ms per stride × 7 strides = ~7 sec on the
            # critical path. The GPU mean-pool runs in microseconds and the
            # subsequent CPU transfer is on a [num_heads, head_dim]-sized
            # tensor (~2 KB), so the H2D vanishes.
            q = quest_query.detach()
            if q.ndim == 3:
                q = q.squeeze(0) if q.shape[0] == 1 else q.mean(dim=0)
            # q is now [num_heads, head_dim]. Mean-pool for GQA against the
            # rank-local KV-head count: at TP=1 that's geom.num_kv_heads;
            # at TP>1 each rank holds only num_kv_heads/tp_size kv-heads
            # and the same fraction of Q-heads. The downstream AllGather
            # concatenates per-rank slices into the full Q the server's
            # scoring kernel expects.
            if q.ndim == 2 and geom is not None:
                num_heads = q.shape[0]
                head_dim = geom.head_dim
                full_kv_heads = geom.num_kv_heads
                local_kv_heads = (full_kv_heads // self._tp_size
                                   if self._tp_size > 1 else full_kv_heads)
                if (local_kv_heads >= 1 and num_heads >= local_kv_heads
                        and num_heads % local_kv_heads == 0):
                    heads_per_group = num_heads // local_kv_heads
                    q = (q.reshape(local_kv_heads, heads_per_group, head_dim)
                          .mean(dim=1))  # [local_kv_heads, head_dim]
            q = q.reshape(-1).to(dtype=torch.float32, device="cpu")
            q_np = q.contiguous().numpy()
        else:
            q_np = np.asarray(quest_query, dtype=np.float32).ravel()

        # Option W: each rank produces its num_kv_heads_local × head_dim
        # Q slice; AllGather across TP and concat to get the full Q the
        # server would see at TP=1, so scoring is mathematically
        # identical to TP=1 (bit-identical page selection).
        nccl_q_us = 0.0
        if self._tp_size > 1:
            try:
                import torch.distributed as dist  # noqa: E402
                from vllm.distributed.parallel_state import get_tp_group
                tp_group = get_tp_group()
                dev_group = tp_group.device_group
                qt = torch.from_numpy(q_np).to(
                    device=f"cuda:{torch.cuda.current_device()}")
                gq = [torch.empty_like(qt) for _ in range(self._tp_size)]
                _trace = os.environ.get("ICMS_NCCL_TRACE", "0") == "1"
                if _trace:
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                dist.all_gather(gq, qt.contiguous(), group=dev_group)
                if _trace:
                    torch.cuda.synchronize()
                nccl_q_us = (time.perf_counter() - _t0) * 1e6
                q_full = torch.cat(gq, dim=0)
                q_np = q_full.detach().cpu().numpy().astype(np.float32, copy=False)
                if _trace:
                    logger.info("[nccl-trace] phase=score_q_allgather "
                                 "rank=%d layer=%d q_bytes=%d us=%.1f",
                                 self._tp_rank, next_layer_idx,
                                 int(qt.numel() * qt.element_size()),
                                 nccl_q_us)
            except Exception as e:
                logger.warning("Score: TP AllGather(Q) failed (%s); using "
                                "rank-local Q which will give wrong pages.", e)

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

        reply = None
        t_score_end = t_score_start
        try:
            # Option W: only rank 0 issues the Score RPC. Server's
            # drain-time fan-out replicates the sink bytes to every
            # peer rank's sink. Rank 0 then broadcasts the reply tuple
            # so every rank can populate _pending_scores identically.
            if self._tp_size > 1 and self._tp_rank != 0:
                reply = None
            else:
                reply = self._client.score(
                    request_id=icms_rid,
                    chain=self._rank_chain(rs.chain),
                    layer=next_layer_idx,
                    query=q_np,
                    k=k,
                    sink=self._sink_pool.sink,
                    reuse_through_layer=reuse_through,
                    # Smoke note: keep use_flags=True everywhere for now.
                    # Fan-out in drain_completions runs after Phase 2
                    # regardless of when the reply was shipped — rank 1
                    # gets the broadcasted reply and proceeds, and in
                    # practice the sink bytes will land before it's
                    # consumed (sweep-verified at TP=1). TP=2 correctness
                    # re-check will be done before removing this note.
                    use_flags=True,
                )
            t_score_end = time.perf_counter()
        except Exception as e:
            t_score_end = time.perf_counter()
            logger.exception("Score RPC failed rank=%d layer=%d: %s",
                              self._tp_rank, next_layer_idx, e)
            reply = None

        # Broadcast reply to all ranks (outside the try so a Score RPC
        # failure on rank 0 still reaches the collective on all ranks).
        nccl_bcast_us = 0.0
        if self._tp_size > 1:
            try:
                _trace = os.environ.get("ICMS_NCCL_TRACE", "0") == "1"
                if _trace:
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                reply = _tp_broadcast_score_reply(
                    reply, self._tp_rank, self._tp_size)
                if _trace:
                    torch.cuda.synchronize()
                nccl_bcast_us = (time.perf_counter() - _t0) * 1e6
                if _trace:
                    logger.info("[nccl-trace] phase=score_reply_broadcast "
                                 "rank=%d layer=%d us=%.1f",
                                 self._tp_rank, next_layer_idx, nccl_bcast_us)
            except Exception as e:
                logger.warning("Score: reply broadcast failed: %s", e)
                reply = None
        if reply is None:
            # Nothing to do; don't touch _pending_scores / _pending_reuse.
            self.stats.record_score(
                (t_score_end - t_score_start) * 1e6,
                False, [], next_layer_idx,
            )
            return
        try:
            # Reply-early (use_flags=True): server flips per-layer
            # flags over RDMA as Phase 2 writes complete — do NOT
            # force them ready locally or wait_for_layer will skip
            # the flag spin and read KV before it lands.
            # (Legacy sync mode kept the force-ready here.)
            sink = self._sink_pool.sink
            _ = sink
            # Stash storage-side concurrent request count for adaptive budget.
            rs._last_storage_concurrent = getattr(
                reply, 'concurrent_requests', 0)
            rpc_ms = (t_score_end - t_score_start) * 1e3
            # Target Score reply-early latency at 16k context is ~2 ms;
            # flag anything noticeably above so we can grep spurious slow
            # calls out of sweep logs.
            slow_tag = " SLOW" if rpc_ms > 3.0 else ""
            logger.debug(
                "[rpc] src=score mode=reply-early rid=%s layers=%d..%d k=%d "
                "rpc=%.2fms score=%.2fms summary_read=%.2fms "
                "sink_write=%.2fms server=%.2fms hit=%d concurrent=%d%s",
                rid, next_layer_idx, reuse_through, len(reply.page_ids),
                rpc_ms,
                reply.score_ns / 1e6,
                reply.summary_read_ns / 1e6,
                reply.sink_write_ns / 1e6,
                reply.server_ingest_to_ready_ns / 1e6,
                int(getattr(reply, 'cache_hit', 0)),
                int(getattr(reply, 'concurrent_requests', 0)),
                slow_tag,
            )
            self.stats.record_score(
                (t_score_end - t_score_start) * 1e6,
                reply.cache_hit,
                list(reply.page_ids),
                next_layer_idx,
                scores=list(reply.scores),
            )
            self._ttft_add(
                rid,
                num_scores=1,
                score_rpc_us_total=(t_score_end - t_score_start) * 1e6,
                t_last_fetch_done=t_score_end,
            )
            if logger.isEnabledFor(10) and reply.page_ids:
                logger.debug(
                    "score layer=%d rid=%s chain_len=%d k=%d top_pages=%s",
                    next_layer_idx, rid, len(rs.chain), k,
                    list(reply.page_ids)[:8],
                )

            # Store the score result keyed by (layer, request_id).
            attn_layer_name = f"model.layers.{next_layer_idx}.self_attn.attn"
            with self._score_lock:
                self._pending_scores.setdefault(attn_layer_name, {})[rid] = (
                    reply, req_idx)

            # Store references for reuse layers. Server's per-layer
            # stride is actual_k*kv_page_bytes (NOT self._k — that's
            # the sink cap). Using self._k here would overshoot the
            # server's layer-delta offsets by a large factor.
            actual_k = len(reply.page_ids)
            per_layer_bytes = actual_k * self._geom.kv_page_bytes
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
        for rid in per_request.keys():
            self._ttft_stamp_once(rid, "t_first_wait_layer", t_layer_start)

        # Spin-wait for the layer's KV to land (reply-early overlap mode).
        # If the sink doesn't expose ready flags, is_layer_ready returns
        # True and this is a no-op.
        abs_layer = self._extract_layer_idx(layer_name)
        # ICMS_DIAG_SLACK probe #3: record wait_for_layer entry time per
        # layer (= compute start of layer L, approximately).
        _slack_called = getattr(self, "_slack_t_called", None)
        if (abs_layer is not None
                and _slack_called is not None
                and 0 <= abs_layer < len(_slack_called)):
            _slack_called[abs_layer] = t_layer_start
        ready_at_call = True
        if (abs_layer is not None
                and self._sink_pool is not None
                and getattr(self._sink_pool.sink, "flag_count", 0) > 0):
            sink = self._sink_pool.sink
            ready_at_call = sink.is_layer_ready(abs_layer)
            if (_slack_called is not None
                    and 0 <= abs_layer < len(_slack_called)):
                self._slack_flag_at_call[abs_layer] = bool(ready_at_call)
            if not ready_at_call:
                # Poll with exponential backoff starting at 0 (pure spin)
                # — writes land within ~100µs so spin is right. Cap total
                # wait at 5s to avoid hangs if something goes wrong.
                deadline = t_layer_start + 5.0
                while not sink.is_layer_ready(abs_layer):
                    if time.perf_counter() > deadline:
                        logger.warning(
                            "wait_for_layer: flag timeout for layer=%d", abs_layer)
                        break
        t_after_spin = time.perf_counter()
        # ICMS_DIAG_SLACK probe #4: when did the spin observe the flip?
        # (Equal to t_layer_start when ready_at_call=True.)
        if (abs_layer is not None
                and getattr(self, "_slack_t_after_spin", None) is not None
                and 0 <= abs_layer < len(self._slack_t_after_spin)):
            self._slack_t_after_spin[abs_layer] = t_after_spin
        spin_us = (t_after_spin - t_layer_start) * 1e6
        # ICMS_DIAG_LAYER_ARRIVAL=1: log per-layer arrival timing relative
        # to step start. Lets us pin down whether per-stride RDMA writes are
        # blocking compute (compute waits) or arriving early (compute slow).
        if os.environ.get("ICMS_DIAG_LAYER_ARRIVAL") == "1":
            for rid in per_request.keys():
                e = self._ttft.get(rid)
                if e is None:
                    continue
                t_step = e.get("t_step_start") or t_layer_start
                logger.info(
                    "[diag-arr] rid=%s layer=%d called_at=%.2fms "
                    "ready_at_call=%d ready_at=%.2fms spin=%.0fus",
                    rid, abs_layer if abs_layer is not None else -1,
                    (t_layer_start - t_step) * 1e3,
                    int(ready_at_call),
                    (t_after_spin - t_step) * 1e3,
                    spin_us,
                )

        total_k_pages = 0
        for rid, (reply, req_idx) in per_request.items():
            if reply is None or not reply.page_ids:
                continue
            try:
                t_apply_start = time.perf_counter()
                self._apply_selective_attention(
                    layer_name, reply, req_idx, rid)
                apply_us = (time.perf_counter() - t_apply_start) * 1e6
                self._ttft_add(
                    rid,
                    num_waits=1,
                    wait_spin_us_total=spin_us,
                    wait_apply_us_total=apply_us,
                )
                total_k_pages += len(reply.page_ids)
            except Exception as e:
                # CRITICAL: do NOT silently swallow. Without re-raising,
                # the layer falls through to vLLM's default attention
                # against the FULL prefix-cached HBM blocks → ICMS budget
                # silently ignored, every budget produces baseline-
                # equivalent numbers. That's a worst-case silent
                # corruption mode for accuracy benchmarks. See the
                # 2026-04-26 audit (docs/tp2_audit_2026-04-26.md
                # Finding 2) for the discovery context.
                #
                # Mark the request's failed-layers set so future
                # hardening can treat the request as poisoned, then
                # re-raise so the engine surfaces the failure instead
                # of emitting meaningless numbers.
                rs = self._requests.get(rid)
                if rs is not None:
                    rs._apply_failed_layers = (
                        getattr(rs, "_apply_failed_layers", set())
                        | {layer_name})
                logger.error(
                    "Path B: apply_selective FAILED for %s req=%s: %s "
                    "— re-raising (silent fall-through to default "
                    "attention would give unmonitored budget=1.0)",
                    layer_name, rid, e, exc_info=True)
                if os.environ.get("ICMS_APPLY_SOFT_FAIL", "0") == "1":
                    # Legacy behavior, gated and explicit. Use only if
                    # production traffic absolutely cannot tolerate a
                    # failed-request surface here.
                    return
                raise

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
        # Per-layer per-rank trace, gated by ICMS_LAYER_TRACE=1. Emits one
        # line per layer per rank with phase timings + a request-relative
        # timestamp so a TP=2 timeline can be diff'd between ranks.
        if os.environ.get("ICMS_LAYER_TRACE", "0") == "1":
            apply_total_us = 0.0
            entry = next((self._ttft.get(rid) for rid in per_request.keys()
                          if rid in self._ttft), None)
            t_rel_ms = 0.0
            if entry is not None and entry.get("t_step_start") is not None:
                t_rel_ms = (t_layer_start - entry["t_step_start"]) * 1e3
            for rid, (reply, _req_idx) in per_request.items():
                if reply is None or not reply.page_ids:
                    continue
                logger.info(
                    "[layer-trace] rank=%d rid=%s layer=%s t_rel=%.2fms "
                    "spin_us=%.1f apply_total_us=%.1f end_t_rel=%.2fms "
                    "k_pages=%d",
                    self._tp_rank, rid, str(layer_idx), t_rel_ms,
                    spin_us, (t_layer_end - t_after_spin) * 1e6,
                    (t_layer_end - (entry["t_step_start"] if entry else t_layer_start)) * 1e3,
                    len(reply.page_ids))

    def _apply_selective_attention(self, layer_name: str, reply,
                                   req_idx: int, rid: str = ""):
        # Per-line wall-time instrumentation, gated by ICMS_LINE_TIMING=1.
        # Captures the Python-side wall time of every step including
        # H2D tensor creations, dict/list ops, and tensor casts. Used
        # to localize the ~3.6 ms/layer Python overhead at TP=2 that
        # remains after seq_len caching. NOT for production.
        import os as _os_lt
        _line_dbg = _os_lt.environ.get("ICMS_LINE_TIMING", "0") == "1"
        _LT = []  # list of (label, t_perf_counter)
        def _lt(label):
            if _line_dbg:
                _LT.append((label, time.perf_counter()))
        _lt("entry")
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

        # PERF: int(am.seq_lens[req_idx]) is a CPU↔GPU host sync (single-
        # element tensor read). Per layer × 48 layers, at TP=2 this
        # serializes against in-flight NCCL collectives and adds ~4 ms
        # per layer ≈ ~190 ms of "wait_apply" — the dominant cost we
        # couldn't explain via slicing / apply-stream / bound check.
        # Cache per-request: seq_lens is invariant within a single
        # forward pass, so we read it once on the first layer's apply
        # and reuse on subsequent layers. ICMS_NO_SEQLEN_CACHE=1
        # disables the cache for A/B testing.
        rs = self._requests.get(rid)
        import os as _os_sl
        _no_seqlen_cache = _os_sl.environ.get("ICMS_NO_SEQLEN_CACHE", "0") == "1"
        if (rs is not None
                and not _no_seqlen_cache
                and getattr(rs, "_apply_cached_seq_len", None) is not None
                and getattr(rs, "_apply_cached_attn_md", None) is am):
            seq_len = rs._apply_cached_seq_len
        else:
            seq_len = int(am.seq_lens[req_idx])
            if rs is not None and not _no_seqlen_cache:
                # Cache keyed on the attn_metadata identity; vLLM creates
                # a new attn_metadata per forward, so this naturally
                # invalidates between forwards.
                rs._apply_cached_seq_len = seq_len
                rs._apply_cached_attn_md = am
        _lt("after_seq_len")


        if rs is None:
            # Fallback: try last known chain.
            chain = self._last_chain_for_rid.get(rid, [])
            if chain:
                rs = _RequestState(request_id=rid, chain=chain)
            else:
                return None

        # BUG-N7: cached on rs by on_step_start; avoids per-layer rescan.
        stored_groups = rs.stored_groups
        effective_groups = max(rs.num_groups_written, stored_groups)
        context_pages = effective_groups * _GROUP_BLOCKS
        context_tokens = context_pages * PAGE_TOKENS
        total_blocks = (seq_len + PAGE_TOKENS - 1) // PAGE_TOKENS

        # ── Fetch selected pages' KV from ICMS sink → GPU blocks ──
        # BUG-N9: the truncation to self._k is a real safety rail (sink
        # buffer is sized to self._k pages; reading past it would corrupt).
        # In normal operation len(reply.page_ids) <= self._k. Make the
        # truncation visible so a server/client cap mismatch (or
        # reply-early concurrency returning more winners than requested)
        # doesn't silently drop pages.
        n_pages = len(reply.page_ids)
        if n_pages > self._k:
            logger.warning(
                "[apply] reply has %d pages but sink capacity self._k=%d; "
                "truncating. Indicates a server/client cap mismatch — "
                "check budget vs sink size.",
                n_pages, self._k)
        selected = sorted(
            pid for pid in reply.page_ids[:min(n_pages, self._k)]
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
        sink = self._sink_pool.sink
        # ICMS_PER_RANK_SLICE=1 (server-side Option Y): the server has
        # already gathered THIS rank's nkv_local heads and packed them
        # at the start of each page slot in the sink. The remote-page
        # layout becomes (PAGE_TOKENS, nkv_local, head_dim) per K and V,
        # halving wire bandwidth + GPU memcpy. Skip the read-time slice
        # below in that mode.
        import os as _os
        per_rank_slice = (
            self._tp_size > 1
            and _os.environ.get("ICMS_PER_RANK_SLICE", "0") == "1")
        nkv_local_runtime = (geom.num_kv_heads // self._tp_size
                              if self._tp_size > 1 else geom.num_kv_heads)
        if per_rank_slice:
            # Per-page bytes effectively used = full bytes / tp_size.
            # We keep kv_page_bytes (used to step into the sink) at the
            # FULL value because the sink slot is still allocated full-
            # size; only the first half holds valid data.  half_bytes
            # also halves so the K/V split lands at the correct boundary.
            kv_page_bytes_eff = geom.kv_page_bytes // self._tp_size
            kv_page_bytes = geom.kv_page_bytes  # sink stride unchanged
            half_bytes = kv_page_bytes_eff // 2
            page_shape = (PAGE_TOKENS, nkv_local_runtime, geom.head_dim)
        else:
            kv_page_bytes = geom.kv_page_bytes
            half_bytes = kv_page_bytes // 2
            page_shape = (PAGE_TOKENS, geom.num_kv_heads, geom.head_dim)
        model_dtype = main_key.dtype
        device = main_key.device
        gpu_direct = getattr(sink, 'is_gpu_direct', False)

        # CPU-side filter: drop pids that lack a sink offset or overflow
        # the block-table row. This loop touches Python lists only, no GPU.
        valid_pids: list[int] = []
        valid_sink_offs: list[int] = []
        bt_row_max = int(bt.shape[1])
        for pid in selected:
            off = pid_to_sink_off.get(pid)
            if off is None or pid >= bt_row_max:
                continue
            valid_pids.append(pid)
            valid_sink_offs.append(off)
        if not valid_pids:
            return None
        _lt("after_python_filter")

        if gpu_direct:
            # ── Batched GPU-direct path: one gather + one dtype convert + ──
            # ── one scatter per layer, instead of 4 kernels + 1 host-sync ──
            # ── per page (old path ran ~48k kernels + ~12k syncs per req). ──
            sink_base = sink.gpu_view(0, sink.size)          # [sink_bytes] u8
            sink_pages = sink_base.view(-1, kv_page_bytes)   # [N, page_bytes]

            # One host→device copy for the index tensors. The naive
            # `torch.tensor(list, device=cuda)` path takes ~3.5 ms/layer
            # at TP=2 (vs ~30 µs for subsequent H2Ds in the same call) —
            # PyTorch builds an unpinned CPU tensor under the hood, and
            # the driver's pageable→pinned staging stalls the stream.
            # Build a pinned CPU tensor explicitly and do a non-blocking
            # async copy: amortizes to ~30 µs/layer.
            page_idx_py = [off // kv_page_bytes
                            for off in valid_sink_offs]
            _lt("before_h2d_tensors")
            _page_idx_cpu = torch.tensor(
                page_idx_py, dtype=torch.int64, pin_memory=True)
            page_idx_dev = _page_idx_cpu.to(device, non_blocking=True)
            _lt("after_page_idx_dev")
            _valid_pids_cpu = torch.tensor(
                valid_pids, dtype=torch.int64, pin_memory=True)
            valid_pids_dev = _valid_pids_cpu.to(device, non_blocking=True)
            _lt("after_valid_pids_dev")

            # phys_blocks via on-device gather (no sync).
            phys_blocks_dev = bt[req_idx].index_select(
                0, valid_pids_dev).to(torch.int64)
            _lt("after_phys_blocks")

            # Bound check: phys_block < main_key.shape[0]. ICMS_SKIP_BOUNDS=1
            # disables this — the .item() forced a CPU↔GPU sync per layer
            # (48× per request). At TP=2 it appears to serialize against
            # in-flight NCCL collectives, dominating wait_apply (200 ms vs
            # ~17 ms at TP=1, despite TP=2 doing half the data). We've
            # never hit the out-of-bounds branch in practice.
            import os as _os_bc
            if _os_bc.environ.get("ICMS_SKIP_BOUNDS", "0") != "1":
                max_blocks_hbm = main_key.shape[0]
                if bool((phys_blocks_dev >= max_blocks_hbm).any().item()):
                    bounds_mask = phys_blocks_dev < max_blocks_hbm
                    phys_blocks_dev = phys_blocks_dev[bounds_mask]
                    page_idx_dev = page_idx_dev[bounds_mask]
                    if phys_blocks_dev.numel() == 0:
                        return None

            # ─── Dedicated apply stream (ICMS_APPLY_STREAM=1) ─────────
            # Apply work has only ~5 ms of GPU compute total (across 48
            # layers, measured with explicit syncs) but Python sees
            # ~200 ms because the kernels queue behind in-flight TP
            # AllReduces on the default CUDA stream at TP=2. Running
            # them on a separate stream lets them overlap with NCCL.
            #
            # Correctness: bracket the apply with two events —
            #   - in_event: apply stream waits for default stream's
            #     prior writes (block_table mods, sink data, etc.).
            #   - out_event: default stream waits for apply's writes
            #     before the NEXT layer's attention reads main_key/_value.
            import os as _os_at
            _apply_dbg = _os_at.environ.get("ICMS_APPLY_TIMING", "0") == "1"
            _use_apply_stream = (
                _os_at.environ.get("ICMS_APPLY_STREAM", "0") == "1")
            if _use_apply_stream and not hasattr(self, "_apply_stream"):
                self._apply_stream = torch.cuda.Stream(device=device)
            apply_stream = (self._apply_stream
                             if _use_apply_stream else None)
            default_stream = torch.cuda.current_stream(device)
            if apply_stream is not None:
                in_event = torch.cuda.Event()
                in_event.record(default_stream)
                apply_stream.wait_event(in_event)
            stream_ctx = (torch.cuda.stream(apply_stream)
                          if apply_stream is not None
                          else nullcontext())

            def _t():
                if _apply_dbg:
                    torch.cuda.synchronize(device)
                return time.perf_counter()
            _t0 = _t()

            with stream_ctx:
                # One gather kernel: [k, page_bytes] u8.
                pages_u8 = sink_pages.index_select(0, page_idx_dev)
                _t1 = _t()
                if per_rank_slice:
                    # Sink slot is full-size but only the first
                    # kv_page_bytes_eff bytes hold valid data (this
                    # rank's slice). K/V split is at half of effective.
                    pages_u8 = pages_u8[:, :kv_page_bytes_eff].contiguous()
                # PERF: skip the legacy fp16->bf16 round-trip; server
                # bytes are already in model_dtype.
                k_bytes = pages_u8[:, :half_bytes].contiguous()
                v_bytes = pages_u8[:, half_bytes:].contiguous()
                _t2 = _t()
                k_pages = k_bytes.view(model_dtype).reshape(-1, *page_shape)
                v_pages = v_bytes.view(model_dtype).reshape(-1, *page_shape)
                _t3 = _t()

                # Option W broadcast path: when server didn't slice,
                # each rank still extracts its head range here.
                if self._tp_size > 1 and not per_rank_slice:
                    nkv_local = geom.num_kv_heads // self._tp_size
                    start = self._tp_rank * nkv_local
                    k_pages = k_pages[:, :, start:start + nkv_local, :].contiguous()
                    v_pages = v_pages[:, :, start:start + nkv_local, :].contiguous()

                # Scatter: two kernels total for this layer.
                main_key.index_copy_(0, phys_blocks_dev, k_pages)
                main_value.index_copy_(0, phys_blocks_dev, v_pages)
                _t4 = _t()
            _lt("after_scatter")

            if apply_stream is not None:
                # Make default stream's next ops (the next layer's
                # attention) wait for apply to finish.
                out_event = torch.cuda.Event()
                out_event.record(apply_stream)
                default_stream.wait_event(out_event)

            if _apply_dbg:
                # One-shot per-layer log; aggregate by hand from log lines.
                logger.warning(
                    "[apply-timing] layer=%s gather=%.0fus split=%.0fus "
                    "view=%.0fus scatter=%.0fus k=%d nkv=%d stream=%s",
                    layer_name,
                    (_t1 - _t0) * 1e6, (_t2 - _t1) * 1e6,
                    (_t3 - _t2) * 1e6, (_t4 - _t3) * 1e6,
                    int(phys_blocks_dev.numel()), nkv_local_runtime,
                    "apply" if apply_stream else "default")

            filled_blocks_count = int(phys_blocks_dev.numel())
            _lt("after_filled_count")

            # Build trimmed block table entirely on device — avoids a
            # per-continuation-block int(bt[...]) sync.
            cont_end = min(total_blocks, bt.shape[1])
            if cont_end > context_pages:
                # Build cont_idx on CPU (pinned) and async-copy to GPU.
                # `torch.arange(..., device=cuda)` issues a kernel that
                # serializes against in-flight model work, repeating the
                # pageable-staging stall pattern.
                _cont_idx_cpu = torch.arange(
                    context_pages, cont_end,
                    dtype=torch.int64, pin_memory=True)
                cont_idx = _cont_idx_cpu.to(device, non_blocking=True)
                cont_blocks = bt[req_idx].index_select(
                    0, cont_idx).to(torch.int32)
                new_bt_row = torch.cat(
                    [phys_blocks_dev.to(torch.int32), cont_blocks])
            else:
                new_bt_row = phys_blocks_dev.to(torch.int32)
            if new_bt_row.numel() == 0:
                return None
            new_bt = new_bt_row.unsqueeze(0)
            _lt("after_bt_build")
        else:
            # Fallback per-page loop for host-sink path. Mirrors the
            # GPU-direct branch's per-rank slicing at lines 2970-2974.
            # Pre-fix-0b (2026-04-26 audit Finding 1) this missed the
            # slice and crashed at .copy_() when tp_size>1 with a
            # full-rank wire shape.
            nkv_local = geom.num_kv_heads
            head_start = 0
            if self._tp_size > 1 and not per_rank_slice:
                nkv_local = geom.num_kv_heads // self._tp_size
                head_start = self._tp_rank * nkv_local
            filled_blocks = []
            for pid, sink_off in zip(valid_pids, valid_sink_offs):
                phys_block = int(bt[req_idx, pid])
                if phys_block >= main_key.shape[0]:
                    continue
                sink_view = sink.view()
                raw = bytes(sink_view[sink_off:sink_off + kv_page_bytes])
                # page_shape upstream is (PAGE_TOKENS, geom.num_kv_heads,
                # head_dim) — the FULL-rank shape on the wire when
                # per_rank_slice=False.
                k_np = np.frombuffer(raw[:half_bytes], dtype=np.float16).reshape(page_shape)
                v_np = np.frombuffer(raw[half_bytes:], dtype=np.float16).reshape(page_shape)
                k_t = torch.from_numpy(k_np.copy()).to(dtype=model_dtype, device=device)
                v_t = torch.from_numpy(v_np.copy()).to(dtype=model_dtype, device=device)
                if self._tp_size > 1 and not per_rank_slice:
                    # Slice to this rank's KV-head range BEFORE the copy.
                    k_t = k_t[:, head_start:head_start + nkv_local, :]
                    v_t = v_t[:, head_start:head_start + nkv_local, :]
                # Now k_t / v_t shape == main_key[phys_block].shape per-rank.
                # Assert is intentional — fix-0a re-raises the exception
                # so this fails loudly under future regressions.
                assert k_t.shape == main_key[phys_block].shape, (
                    f"host-sink K shape {tuple(k_t.shape)} != main_key slice "
                    f"{tuple(main_key[phys_block].shape)} at "
                    f"tp_rank={self._tp_rank}/{self._tp_size}, "
                    f"per_rank_slice={per_rank_slice}")
                main_key[phys_block].copy_(k_t)
                main_value[phys_block].copy_(v_t)
                filled_blocks.append(phys_block)
            if not filled_blocks:
                return None
            new_entries = list(filled_blocks)
            for blk_idx in range(context_pages, total_blocks):
                if blk_idx < bt.shape[1]:
                    new_entries.append(int(bt[req_idx, blk_idx]))
            if not new_entries:
                return None
            new_bt = torch.tensor(
                [new_entries], dtype=torch.int32, device=device)
            filled_blocks_count = len(filled_blocks)

        continuation_tokens = max(0, seq_len - context_tokens)
        new_seq_len = filled_blocks_count * PAGE_TOKENS + continuation_tokens
        # Pin + async-copy: same pattern as the page_idx_dev fix above.
        # Without pinning, this last H2D inherits the implicit pageable-
        # staging sync that `cuda.synchronize`-equivalent stalls the
        # apply path's wall time at the LAST tensor() call.
        _new_sl_cpu = torch.tensor(
            [new_seq_len], dtype=torch.int32, pin_memory=True)
        new_sl = _new_sl_cpu.to(device, non_blocking=True)

        logger.debug(
            "Path B: layer=%s req_idx=%d fetched %d/%d pages, "
            "seq_len %d→%d",
            layer_name, req_idx, filled_blocks_count, context_pages,
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
        _lt("after_set_active")
        if _line_dbg and len(_LT) >= 2:
            parts = []
            for i in range(1, len(_LT)):
                label, t = _LT[i]
                d_us = (t - _LT[i-1][1]) * 1e6
                parts.append(f"{label}={d_us:.0f}us")
            total_us = (_LT[-1][1] - _LT[0][1]) * 1e6
            logger.warning(
                "[apply-line] layer=%s total=%.0fus %s",
                layer_name, total_us, " ".join(parts))
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
        """Kick off deferred write work; DO NOT block on it.

        vLLM calls this at the end of a forward pass via wait_for_save;
        that call sits on the TTFT critical path. We must return fast:
        snapshot what the extract + flush work needs, submit a single
        closure to the background pipeline, and return.

        The drain (i.e., synchronous wait for the pipeline to finish
        this request's writes) happens in on_request_finished, which
        runs AFTER first-token emission and is off the TTFT critical
        path.

        Legacy inline behavior is still invoked when pipeline is absent
        (shouldn't happen in normal operation).
        """
        t_save_enter = time.perf_counter()
        self._ttft_emit(t_save_enter)
        self._slack_emit_and_reset()
        if not (self._gpu_kv_caches and self._attn_metadata is not None
                and self._requests and not self._skip_extract):
            # Nothing to extract — still flip the prefill_done flag.
            if not self._prefill_done:
                self._prefill_done = True
                logger.info("Prefill done. Switching to dense decode.")
            return

        # Snapshot references that the background task needs. The KV
        # cache tensors and request state are expected to remain valid
        # until on_request_finished drains the pipeline.
        snap_caches = dict(self._gpu_kv_caches)
        snap_meta   = self._attn_metadata
        snap_rids   = list(self._requests.keys())

        def _task():
            self._do_deferred_extract_and_flush(
                snap_caches, snap_meta, snap_rids)

        self._write_pipeline.submit(_task, tag="wait_for_save")

        # Optional measurement mode: block until this prefill's writes
        # complete before returning. Default behavior leaves writes async
        # in the background, which is faster but means iter N's writes
        # can race iter N+1's fetches on the same RDMA link, polluting
        # slack measurements (per-layer fetch arrival times shift later
        # under contention). With ICMS_BLOCK_WRITES=1, all writes are on
        # the current prefill's TTFT critical path — the next prefill
        # begins with an idle link.
        if os.environ.get("ICMS_BLOCK_WRITES") == "1":
            try:
                drained = self._write_pipeline.drain(timeout=60.0)
                if not drained:
                    logger.warning(
                        "ICMS_BLOCK_WRITES drain timed out (60s)")
            except Exception:
                logger.exception("ICMS_BLOCK_WRITES drain failed")

        if not self._prefill_done:
            self._prefill_done = True
            logger.info("Prefill done. Switching to dense decode.")

    def _do_deferred_extract_and_flush(
        self,
        caches: dict,
        meta,
        rids: list[str],
    ):
        """Runs on the write-pipeline worker thread.

        Does the heavy lifting (GPU→CPU copies inside extract_and_record,
        summary compute, bytearray fills, and WriteGroup RPCs) without
        holding up the vLLM forward-pass return. Touches shared client
        state under self._score_lock to serialize with main-thread
        Score/FetchAll RPCs."""
        # BUG-N13 Phase 1 diag: entry trace. Captures pipeline-thread state
        # at the moment we attempt extract+flush. Toggle ICMS_DIAG_N13=1.
        if os.environ.get("ICMS_DIAG_N13", "0") == "1":
            meta_kind = "dict" if isinstance(meta, dict) else type(meta).__name__
            rid_summaries = []
            for rid in rids:
                rs_dbg = self._requests.get(rid)
                if rs_dbg is None:
                    rid_summaries.append(f"{rid}=missing")
                else:
                    rid_summaries.append(
                        f"{rid}=chain_len={len(rs_dbg.chain)} "
                        f"stored={rs_dbg.stored_groups} "
                        f"written={rs_dbg.num_groups_written} "
                        f"buffers={len(rs_dbg.active_group_buffers)}")
            logger.info(
                "[diag-extract-entry] rank=%d caches=%d meta=%s rids=[%s] "
                "skip_extract=%s",
                self._tp_rank, len(caches), meta_kind,
                "; ".join(rid_summaries), self._skip_extract)
        # ─── 1. Batch extraction
        for layer_name, kv in caches.items():
            if kv is None or kv.ndim != 5:
                continue
            if isinstance(meta, dict):
                am = meta.get(layer_name)
            else:
                am = meta
            if am is None:
                continue
            try:
                self.extract_and_record(layer_name, kv, am)
            except Exception:
                logger.exception(
                    "deferred-extract: extract_and_record failed "
                    "for layer=%s", layer_name)

        # ─── 2. Flush complete group buffers via WriteGroup RPCs.
        # BUG-N5: no _score_lock here. The icms client serializes
        # QP access internally (per-conn QP mutex); _score_lock only
        # guards _pending_scores / _pending_reuse dict mutations.
        # Holding it across a 170 ms write_group RPC needlessly
        # blocked main-thread wait_for_layer / on_layer_score dict
        # ops. (The original comment on this block claimed the lock
        # serialized the QP — that's stale: score()/fetch_all()
        # already run lock-free at the connector level.)
        for rid in rids:
            rs = self._requests.get(rid)
            if rs is None:
                continue
            for gidx in sorted(rs.active_group_buffers.keys()):
                buf = rs.active_group_buffers.get(gidx)
                if buf is None:
                    continue
                if buf.is_complete():
                    self._flush_group(rid, gidx, partial=False)

        # ─── 3. Push stored-prefix entries so subsequent requests can
        #       skip extraction for the matched prefix.
        for rid in rids:
            rs = self._requests.get(rid)
            if rs is None:
                continue
            if rs.chain and rs.num_groups_written > 0:
                self._record_stored_groups(rs.chain, rs.num_groups_written)
                _stored_chain_queue.append(
                    (list(rs.chain), rs.num_groups_written))

    # ─── KV extraction from GPU paged buffer (C2 write path) ────────────

    def extract_and_record(self, layer_name: str, kv_layer: torch.Tensor,
                            attn_metadata) -> None:
        """Extract per-block K/V from the GPU KV cache and populate icms buffers.

        Called from save_kv_layer for each standard (non-Quest) per-layer call.

        kv_layer shape: [2 (K+V), num_blocks, block_size, num_kv_heads, head_dim]
        attn_metadata: FlashAttentionMetadata with block_table, seq_lens, etc.
        """
        # Diagnostic short-circuit: ICMS_SKIP_EXTRACT=1 makes this a no-op.
        # Used to verify whether extract_and_record's deferred GPU work
        # (AllGather K/V at TP>1, GPU->CPU memcpy) is silently serializing
        # with the model's forward via shared CUDA stream. If TTFT drops
        # noticeably with this set, those ops are on the critical path
        # despite running in a "background" thread.
        import os as _os
        # BUG-N13 Phase 1 diag: trace every entry + every early-return
        # path in extract_and_record. Only log on layer_idx==0 to avoid
        # 48× spam per request.
        _diag_n13 = _os.environ.get("ICMS_DIAG_N13", "0") == "1"
        if _os.environ.get("ICMS_SKIP_EXTRACT", "0") == "1":
            if _diag_n13:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=env_skip",
                            self._tp_rank, layer_name)
            return
        # BUG-N2: scheduler-decided per-step skip flag. Set by
        # on_step_start when every active request's stored prefix
        # already covers its complete-group count. Symmetric across TP
        # ranks because it's derived from broadcast metadata.
        if self._skip_extract:
            if _diag_n13:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=N2_skip_extract",
                            self._tp_rank, layer_name)
            return
        t0 = time.perf_counter()
        logger.debug("extract_and_record: layer=%s kv_shape=%s", layer_name,
                      tuple(kv_layer.shape) if hasattr(kv_layer, 'shape') else '?')
        if not self._requests or self._geom is None:
            if _diag_n13:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=no_requests/geom "
                            "requests=%d geom=%s",
                            self._tp_rank, layer_name,
                            len(self._requests), self._geom is not None)
            return

        # Parse layer index from layer_name like "model.layers.5.self_attn.attn"
        layer_idx = self._parse_layer_idx(layer_name)
        if layer_idx is None or layer_idx >= self._geom.num_layers:
            if _diag_n13:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=bad_layer_idx",
                            self._tp_rank, layer_name)
            return

        # kv_layer: [2, num_blocks, block_size, num_kv_heads, head_dim]
        if kv_layer.ndim != 5 or kv_layer.shape[0] != 2:
            if _diag_n13 and layer_idx == 0:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=bad_kv_shape "
                            "ndim=%d shape0=%d", self._tp_rank, layer_name,
                            kv_layer.ndim,
                            kv_layer.shape[0] if kv_layer.ndim > 0 else -1)
            return
        k_cache = kv_layer[0]  # [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache = kv_layer[1]

        # block_table: [num_reqs, max_num_blocks_per_req] — per-request block IDs
        block_table = getattr(attn_metadata, "block_table", None)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if block_table is None or seq_lens is None:
            if _diag_n13 and layer_idx == 0:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=missing_attn_md "
                            "bt=%s sl=%s", self._tp_rank, layer_name,
                            block_table is not None, seq_lens is not None)
            return
        if _diag_n13 and layer_idx == 0:
            sl0 = (int(sl_cpu[0]) if (hasattr(seq_lens, '__len__') and len(seq_lens) > 0)
                   else None) if False else None
            # Don't crash on weird metadata — just log shape
            logger.info("[diag-extract] rank=%d layer=%s ENTRY ok kv_shape=%s "
                        "bt_shape=%s",
                        self._tp_rank, layer_name,
                        tuple(kv_layer.shape),
                        tuple(block_table.shape) if hasattr(block_table, 'shape') else '?')

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

        # BUG-N1 fix: under prefix elision (TP>1 + ICMS), vLLM only
        # allocates the trailing slice of the block table, but seq_lens
        # still reports the full prompt length. Trailing slots in
        # bt_cpu[0, k:] hold stale IDs from another request's
        # allocation; indexing k_cache by those produces a CUDA OOB
        # that surfaces as a device-side assert at the next .cpu().
        # Clamp num_blocks to the actually-allocated row width so we
        # never read past the live region. Both ranks at TP>1 observe
        # bt_cpu shape symmetrically, so this clamp is collective-safe.
        # The existing dedup early-return at `effective_start >=
        # len(req_block_ids)` below then short-circuits when the
        # already-stored groups cover the (clamped) allocation — but
        # crucially that early-return only fires after the AllGather,
        # so an asymmetric skip *before* AllGather would deadlock.
        if bt_cpu.ndim >= 2:
            num_blocks = min(num_blocks, int(bt_cpu.shape[1]))
        elif bt_cpu.ndim == 1:
            num_blocks = min(num_blocks, int(bt_cpu.shape[0]))
        else:
            return

        # Extract the request's block IDs from the block table.
        if bt_cpu.ndim >= 2:
            req_block_ids = bt_cpu[0, :num_blocks].tolist()
        else:
            req_block_ids = bt_cpu[:num_blocks].tolist()

        # Only process blocks we haven't recorded yet for this layer.
        recorded_key = (rid, layer_idx)
        if not hasattr(rs, '_recorded_blocks'):
            rs._recorded_blocks = {}
        already_recorded = rs._recorded_blocks.get(recorded_key, 0)

        # Dedup-aware skip: blocks in groups already stored under the
        # request's chain prefix don't need re-extraction. The server
        # would dedup the write anyway, but the GPU→CPU copy + summary
        # compute are wasted client-side work on the critical path.
        stored_blocks = rs.stored_groups * _GROUP_BLOCKS
        effective_start = max(already_recorded, stored_blocks)
        if effective_start > already_recorded:
            # Advance num_groups_written so later _flush_group calls and
            # the stored-prefix push correctly reflect "stored + new".
            if rs.stored_groups > rs.num_groups_written:
                rs.num_groups_written = rs.stored_groups

        if effective_start >= len(req_block_ids):
            rs._recorded_blocks[recorded_key] = len(req_block_ids)
            return  # everything we'd process is already stored

        new_ids = req_block_ids[effective_start:]
        valid_ids = [bid for bid in new_ids if 0 <= bid < k_cache.shape[0]]
        if not valid_ids:
            rs._recorded_blocks[recorded_key] = len(req_block_ids)
            return

        # Batched GPU→CPU copy: index all new blocks at once.
        idx_tensor = torch.tensor(valid_ids, dtype=torch.long, device=k_cache.device)
        k_batch_gpu = k_cache[idx_tensor]
        v_batch_gpu = v_cache[idx_tensor]

        # Option W write-path: at TP>1 each rank's kv_cache holds only
        # its num_kv_heads_local slice. NCCL-AllGather across the TP
        # group along the num_kv_heads dim (dim=2 for
        # [N, block_size, num_kv_heads_local, head_dim]) so rank 0
        # can emit a full-head blob via write_group. Both ranks must
        # reach the collective: extract_and_record's early-returns
        # (self._requests empty / seq_len<=0 / nothing-to-record) are
        # derived from scheduler-propagated state and so fire
        # symmetrically at TP=2. If we ever see a hang here it means
        # that assumption is wrong — fall back by wrapping with a
        # pre-sync AllReduce(have-data) flag.
        if self._tp_size > 1:
            try:
                import torch.distributed as dist  # noqa: E402
                from vllm.distributed.parallel_state import get_tp_group
                tp_group = get_tp_group()
                dev_group = tp_group.device_group
                gk = [torch.empty_like(k_batch_gpu)
                      for _ in range(self._tp_size)]
                gv = [torch.empty_like(v_batch_gpu)
                      for _ in range(self._tp_size)]
                _trace = os.environ.get("ICMS_NCCL_TRACE", "0") == "1"
                if _trace:
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                dist.all_gather(gk, k_batch_gpu.contiguous(),
                                 group=dev_group)
                dist.all_gather(gv, v_batch_gpu.contiguous(),
                                 group=dev_group)
                if _trace:
                    torch.cuda.synchronize()
                _us = (time.perf_counter() - _t0) * 1e6
                k_batch_gpu = torch.cat(gk, dim=2)
                v_batch_gpu = torch.cat(gv, dim=2)
                if _trace:
                    bytes_per_rank = int(
                        k_batch_gpu.numel() * k_batch_gpu.element_size()
                        + v_batch_gpu.numel() * v_batch_gpu.element_size())
                    logger.info("[nccl-trace] phase=extract_kv_allgather "
                                 "rank=%d layer=%d kv_bytes_per_rank=%d "
                                 "us=%.1f",
                                 self._tp_rank, layer_idx,
                                 bytes_per_rank, _us)
            except Exception as e:
                logger.warning("extract_and_record: TP AllGather failed "
                                "(%s); half-head write will result in "
                                "garbage on the server.", e)

        k_batch = k_batch_gpu.cpu()
        v_batch = v_batch_gpu.cpu()

        for i, intra_idx in enumerate(range(effective_start, effective_start + len(valid_ids))):
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
            # Summary region is packed by scored_rank; size accordingly.
            buf = _GroupBuffer(
                summary_blob=bytearray(
                    geom.num_scored_layers * geom.summary_group_bytes),
                kv_blob=bytearray(geom.num_layers * geom.kv_group_bytes),
                filled=set(),
                num_layers=geom.num_layers,
                pages_in_group=_GROUP_BLOCKS,
            )
            rs.active_group_buffers[group_idx] = buf

        # KV: K || V byte concatenation (C4). Always stored (every layer).
        k_bytes = key_block.to(torch.float16).contiguous().numpy().tobytes()
        v_bytes = value_block.to(torch.float16).contiguous().numpy().tobytes()
        kv_bytes = k_bytes + v_bytes

        spb = geom.summary_page_bytes
        kpb = geom.kv_page_bytes
        # Byte offset within buf.kv_blob for (layer_idx, page_in_group).
        # Must mirror the server's kv_offset() — both sides agree via
        # the kv_layout config. Offsets here are relative to the start
        # of the KV region (buf.kv_blob does not include summaries).
        if geom.kv_layout == KvLayout.LAYER_MAJOR:
            k_off = layer_idx * geom.kv_group_bytes + page_in_group * kpb
        else:  # PAGE_MAJOR
            k_off = page_in_group * geom.num_layers * kpb + layer_idx * kpb
        buf.kv_blob[k_off:k_off + len(kv_bytes)] = kv_bytes

        # Summary: only compute + store for scored layers. Non-scored layers
        # have no on-disk summary slot.
        if geom.is_scored(layer_idx):
            keys = key_block.to(dtype=torch.float32, device="cpu")
            if keys.ndim == 3:
                keys = keys.reshape(keys.shape[0], -1)  # [block_size, key_dim]
            kmin = keys.min(dim=0).values.to(torch.float16)
            kmax = keys.max(dim=0).values.to(torch.float16)
            summary_bytes = kmin.numpy().tobytes() + kmax.numpy().tobytes()
            rank = geom.scored_rank(layer_idx)
            s_off = rank * geom.summary_group_bytes + page_in_group * spb
            buf.summary_blob[s_off:s_off + len(summary_bytes)] = summary_bytes

        buf.filled.add((layer_idx, page_in_group))
        # NOTE: no inline flush on completion. Flushing is deferred to
        # wait_for_pending_writes (called at the end of each forward
        # pass by vLLM) so that write_group RPCs and the extraction
        # work don't stall per-layer save_kv_layer calls during the
        # prefill forward pass.

    def _flush_group(self, request_id: str, group_idx: int, partial: bool = False):
        """Issue WriteGroup for a completed (or partial) group buffer."""
        # BUG-N13 Phase 1 diag: catch the early-return paths.
        _diag_n13 = os.environ.get("ICMS_DIAG_N13", "0") == "1"
        rs = self._requests.get(request_id)
        if rs is None:
            if _diag_n13:
                logger.info("[diag-flush-attempted] rank=%d rid=%s gidx=%d "
                            "EARLY_RETURN=rs_missing", self._tp_rank,
                            request_id, group_idx)
            return
        buf = rs.active_group_buffers.pop(group_idx, None)
        if buf is None:
            if _diag_n13:
                logger.info("[diag-flush-attempted] rank=%d rid=%s gidx=%d "
                            "EARLY_RETURN=buf_missing buffers=%s",
                            self._tp_rank, request_id, group_idx,
                            sorted(rs.active_group_buffers.keys()))
            return
        chain_prefix = rs.chain[:group_idx + 1]
        if not chain_prefix:
            if _diag_n13:
                logger.info("[diag-flush-attempted] rank=%d rid=%s gidx=%d "
                            "EARLY_RETURN=empty_chain_prefix chain_len=%d",
                            self._tp_rank, request_id, group_idx,
                            len(rs.chain))
            return
        if _diag_n13:
            logger.info("[diag-flush-attempted] rank=%d rid=%s gidx=%d "
                        "partial=%s buf.filled=%d/%d chain_prefix_len=%d",
                        self._tp_rank, request_id, group_idx, partial,
                        len(buf.filled),
                        _GROUP_BLOCKS * (self._geom.num_layers if self._geom else 48),
                        len(chain_prefix))
        pages = _GROUP_BLOCKS if not partial else len({p for (_, p) in buf.filled})
        phase = "decode" if self._prefill_done else "prefill"
        logger.debug("[diag-flush] group=%d pages=%d partial=%s phase=%s "
                     "chain_len=%d num_groups_written(pre)=%d",
                     group_idx, pages, partial, phase,
                     len(chain_prefix), rs.num_groups_written)
        t0 = time.perf_counter()
        try:
            # Option W: only rank 0 sends to the server. All ranks still
            # update local bookkeeping (_record_stored_groups etc.) below.
            if self._tp_size > 1 and self._tp_rank != 0:
                pass  # rank>0 skips the wire write_group
            else:
                self._client.write_group(
                    self._rank_chain(chain_prefix),
                    bytes(buf.summary_blob), bytes(buf.kv_blob),
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
        # Drain the deferred-write pipeline before touching this
        # request's state. The pipeline's extract + flush tasks for
        # this request may still be running from the last wait_for_save;
        # they need rs.active_group_buffers etc. to still exist.
        # The drain ALSO serves as a NCCL-ordering barrier at TP>1:
        # `extract_and_record` inside the pipeline thread does an
        # AllGather on the TP group; if the next request's main-thread
        # forward starts (with its own NCCL all-reduces) before the
        # pipeline's AllGather completes, both threads hammer the same
        # NCCL group concurrently and deadlock. Reverted BUG-N6 attempt
        # caused exactly that hang at A-2k iter-2 (TP=1 stuck after
        # layer 5; engine sample_tokens timeout). The drain runs after
        # vLLM has already returned first-token (off TTFT critical
        # path of the *current* request) and *before* the next
        # request's forward begins, which is the right ordering for
        # NCCL serialization.
        # Drain timeout sized for the worst case mistral budget=0.5 @ 30k
        # ctx (≈70 GiB of WriteGroup traffic at ~10 GiB/s ⇒ ~7 s on the
        # link, plus the AllGather barrier). Past 30 s the timeout fired
        # routinely, returned early, and the next request's NCCL collided
        # with the still-running pipeline AllGather → shm_broadcast hang
        # → sample_tokens RPC timeout. 90 s is safely above the observed
        # 17-25 s drain times during the 2026-04-27 mistral quality run
        # plus headroom for the larger budgets.
        t0 = time.perf_counter()
        ok = self._write_pipeline.drain(timeout=90.0)
        drain_us = (time.perf_counter() - t0) * 1e6
        if not ok:
            logger.warning(
                "on_request_finished: write-pipeline drain TIMED OUT "
                "(>90s, %d tasks pending); request-finish proceeding — "
                "some writes may be lost.",
                self._write_pipeline.pending())
        elif drain_us > 1000:
            logger.info(
                "on_request_finished: write-pipeline drain took %.1f ms",
                drain_us / 1000.0)

        rs = self._requests.get(request_id)
        if rs is None:
            return
        # Flush any partial group buffers first (must happen BEFORE we
        # pop rs, since _flush_group re-reads self._requests[rid]).
        # BUG-N5: no _score_lock here — the icms client serializes
        # QP access internally; the lock would only block dict
        # mutations in wait_for_layer / on_layer_score for no gain.
        for gidx in list(rs.active_group_buffers.keys()):
            self._flush_group(request_id, gidx, partial=True)
        # Push the final group-count to the scheduler's prefix index so
        # a subsequent request with the same prefix can skip prefill.
        # wait_for_pending_writes is NOT called after on_request_finished,
        # so the last partial group's contribution to num_groups_written
        # would otherwise be lost.
        if rs.chain and rs.num_groups_written > 0:
            self._record_stored_groups(rs.chain, rs.num_groups_written)
            _stored_chain_queue.append(
                (list(rs.chain), rs.num_groups_written))
            logger.debug("[diag-finish] rid=%s chain_len=%d num_groups_written=%d "
                        "pushed to stored-prefix index",
                        request_id, len(rs.chain), rs.num_groups_written)
        else:
            logger.debug("[diag-finish] rid=%s chain_len=%d num_groups_written=%d "
                        "(NOT pushed — chain empty or no groups)",
                        request_id, len(rs.chain) if rs.chain else 0,
                        rs.num_groups_written)
        # BUG-N8: sweep _pending_scores and _pending_reuse for this rid.
        # Both maps are {layer_name: {rid: tuple}}, drained per-layer-name.
        # If a layer never fires (request finished mid-stride, layer
        # skipped, etc.), the inner {rid: ...} entries persist forever.
        # Bounded slow leak; clean up here to prevent unbounded growth
        # under heavy churn.
        for inner in self._pending_reuse.values():
            inner.pop(request_id, None)
        for inner in self._pending_scores.values():
            inner.pop(request_id, None)
        self._pending_reuse  = {k: v for k, v in self._pending_reuse.items()  if v}
        self._pending_scores = {k: v for k, v in self._pending_scores.items() if v}
        # Now pop the request state and reset prefill_done.
        self._requests.pop(request_id, None)
        self._prefill_done = False
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
        # Option W: only rank 0 RPCs; other ranks no-op but keep local state.
        if self._tp_size > 1 and self._tp_rank != 0:
            return None
        return self._client.write_group(self._rank_chain(chain),
                                         summary_blob, kv_blob,
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
            request_id=icms_rid, chain=self._rank_chain(chain), layer=layer,
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
