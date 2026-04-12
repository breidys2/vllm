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

import struct
import sys
import threading
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
from vllm.v1.core.kv_cache_utils import KVCacheBlocks
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

        self._sched: _Scheduler | None = None
        self._worker: _Worker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self._sched = _Scheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self._worker = _Worker(
                socket_path=self._socket_path,
                model_name=self._model_name,
                k=self._k,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  Worker-side abstract methods
    # ══════════════════════════════════════════════════════════════════════

    def start_load_kv(self, forward_context: ForwardContext, **kwargs) -> None:
        """Kick off async KV loading for this step.

        In the three-state pipelining model (C9):
        - For cache-hit layers: fire KV fetch immediately (no Q needed).
        - For cache-miss layers: fire speculative summary preload.
        Actual scoring waits for Q in save_kv_layer / wait_for_layer_load.
        """
        if self._worker is None:
            return
        meta = self._connector_metadata
        if meta is None or not isinstance(meta, IcmsConnectorMetadata):
            return
        self._worker.on_step_start(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until KV for this layer is ready in the GPU buffer.

        For cache-hit layers the KV fetch was fired in start_load_kv and
        should be complete by now. For cache-miss layers this is where we
        block on the scoring result.
        """
        if self._worker is None:
            return
        self._worker.wait_for_layer(layer_name)

    def save_kv_layer(
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

        quest_query = kwargs.get("quest_query")
        next_layer_idx = kwargs.get("next_layer_idx")
        budget = kwargs.get("budget", self._budget)
        quest_stats = kwargs.get("quest_stats")
        quest_all_pages = kwargs.get("quest_all_pages", False)
        quest_reuse_selection = kwargs.get("quest_reuse_selection", False)

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
            # CPU-scoring path: Q arrived, fire Score against icms.
            self._worker.on_layer_score(
                next_layer_idx, quest_query, budget, quest_stats,
            )

    def wait_for_save(self) -> None:
        if self._worker is not None:
            self._worker.wait_for_pending_writes()

    def get_finished(
        self, finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
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
        # v1: no external prefix match. vLLM does full prefill.
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

    def on_alloc(self, request: Request, blocks: KVCacheBlocks):
        rid = request.request_id
        if rid in self._chains:
            return  # already seen
        # Derive group-hash chain from block_hashes (C1).
        bh = getattr(request, "block_hashes", None)
        if bh is None or len(bh) == 0:
            self._chains[rid] = []
            return
        # block_hashes is a list of BlockHash objects. Each BlockHash has a
        # .hash_value (int). Group identity = last block hash in each group.
        chain: list[int] = []
        for i in range(_GROUP_BLOCKS - 1, len(bh), _GROUP_BLOCKS):
            h = bh[i]
            # BlockHash may be a namedtuple or dataclass; extract int.
            if isinstance(h, int):
                chain.append(h & 0xFFFFFFFFFFFFFFFF)
            elif hasattr(h, "hash_value"):
                chain.append(h.hash_value & 0xFFFFFFFFFFFFFFFF)
            else:
                chain.append(hash(h) & 0xFFFFFFFFFFFFFFFF)
        # Partial trailing group: use the last available hash.
        if len(bh) % _GROUP_BLOCKS != 0:
            last = bh[-1]
            if isinstance(last, int):
                chain.append(last & 0xFFFFFFFFFFFFFFFF)
            elif hasattr(last, "hash_value"):
                chain.append(last.hash_value & 0xFFFFFFFFFFFFFFFF)
            else:
                chain.append(hash(last) & 0xFFFFFFFFFFFFFFFF)
        self._chains[rid] = chain
        self._pending_chain_sends.add(rid)

        # TODO(C2): extract in-order block IDs from `blocks` for the
        # block_id → intra_request_idx mapping. For v1 the worker uses
        # the direct helper API which bypasses this mapping.

    def build_meta(self, scheduler_output: SchedulerOutput) -> IcmsConnectorMetadata:
        meta = IcmsConnectorMetadata()
        # Pack per-request step info.
        for sr in getattr(scheduler_output, "scheduled_new_reqs", []):
            rid = sr.req_id if hasattr(sr, "req_id") else str(sr)
            meta.requests.append(_PerRequestStep(
                request_id=rid,
                num_computed_tokens_start=0,
                num_computed_tokens_end=getattr(sr, "num_tokens", 0),
            ))
        for sr in getattr(scheduler_output, "scheduled_cached_reqs", []):
            rid = sr.req_id if hasattr(sr, "req_id") else str(sr)
            meta.requests.append(_PerRequestStep(
                request_id=rid,
                num_computed_tokens_start=getattr(sr, "num_computed_tokens", 0),
                num_computed_tokens_end=getattr(sr, "num_computed_tokens", 0)
                                       + getattr(sr, "num_new_tokens", 0),
            ))
        # Deliver chains for requests we haven't sent yet.
        for rid in list(self._pending_chain_sends):
            if rid in self._chains:
                meta.new_chains[rid] = self._chains[rid]
            self._pending_chain_sends.discard(rid)
        return meta

    def on_finished(self, request_id: str):
        self._chains.pop(request_id, None)
        self._block_ids.pop(request_id, None)
        self._pending_chain_sends.discard(request_id)


# ═══════════════════════════════════════════════════════════════════════════
#  Worker
# ═══════════════════════════════════════════════════════════════════════════

class _Worker:
    """Worker-side state. Owns IcmsClient, sink pool, per-request buffers."""

    def __init__(self, *, socket_path: str, model_name: str, k: int):
        self._socket_path = socket_path
        self._model_name = model_name
        self._k = k

        self._client: IcmsClient | None = None
        self._geom: ModelGeometry | None = None
        self._sink_pool: _SinkSlotPool | None = None

        # Per-request state (C1 worker-side cache).
        self._requests: dict[str, _RequestState] = {}

        # Pending async score results (C9).
        self._pending_scores: dict[str, Any] = {}  # layer_name → result
        self._score_lock = threading.Lock()

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
        # C8: pre-allocate sink with N slots.
        slot_bytes = self._k * self._geom.kv_page_bytes
        total_sink = slot_bytes * _SINK_SLOTS
        sink = self._client.register_sink(total_sink)
        self._sink_pool = _SinkSlotPool(sink, slot_bytes, _SINK_SLOTS)
        logger.info(
            "IcmsConnector worker: connected to %s, model=%s, "
            "sink=%d slots × %d B = %d B total",
            self._socket_path, self._model_name,
            _SINK_SLOTS, slot_bytes, total_sink,
        )

    def shutdown(self):
        if self._client is None:
            return
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
        for rid, chain in meta.new_chains.items():
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.chain = chain
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
        """Single-layer-scoring reuse: reuse previous selection (C9 State 1)."""
        # Score cache on the icms server handles this automatically —
        # the Score call for this layer will hit the cache.
        pass

    def on_layer_score(self, next_layer_idx, quest_query, budget, stats):
        """CPU-scoring path: Q arrived, fire Score against icms (C9 State 2/3)."""
        # Find the active request. With batch=1 (C3) there's exactly one.
        if not self._requests:
            return
        rid, rs = next(iter(self._requests.items()))
        if not rs.chain:
            return

        icms_rid = self._icms_request_id(rid, 0)
        k = max(1, int(len(rs.chain) * _GROUP_BLOCKS * budget))
        k = min(k, self._k)

        # Marshal query to fp32 numpy.
        if isinstance(quest_query, torch.Tensor):
            q = quest_query.detach().to(dtype=torch.float32, device="cpu")
            if q.ndim == 3:
                # [B, num_heads, head_dim] → mean-pool if GQA, then flatten
                q = q.mean(dim=0) if q.shape[0] > 1 else q.squeeze(0)
            if q.ndim == 2:
                # [num_heads, head_dim] → flatten
                q = q.reshape(-1)
            q_np = q.contiguous().numpy()
        else:
            q_np = np.asarray(quest_query, dtype=np.float32).ravel()

        # Fire Score synchronously for v1.
        # TODO(C9): make this async with a background thread.
        slot = self._sink_pool.acquire()
        try:
            reply = self._client.score(
                request_id=icms_rid,
                chain=rs.chain,
                layer=next_layer_idx,
                query=q_np,
                k=k,
                sink=self._sink_pool.sink,
            )
            layer_name = f"layer_{next_layer_idx}"
            with self._score_lock:
                self._pending_scores[layer_name] = (reply, slot)
        except Exception as e:
            self._sink_pool.release(slot)
            logger.warning("Score failed for layer %d: %s", next_layer_idx, e)

    def wait_for_layer(self, layer_name: str):
        """Block until Score result for this layer is available."""
        with self._score_lock:
            result = self._pending_scores.pop(layer_name, None)
        if result is not None:
            reply, slot = result
            # TODO: cudaMemcpyAsync from sink slot into GPU block slots.
            # For v1 we just release the slot; the actual GPU copy is
            # deferred to the full integration (R12).
            self._sink_pool.release(slot)

    def wait_for_pending_writes(self):
        """Block until all buffered WriteGroups are flushed."""
        for rid, rs in self._requests.items():
            for gidx in list(rs.active_group_buffers.keys()):
                self._flush_group(rid, gidx, partial=True)

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
        try:
            self._client.write_group(
                chain_prefix, bytes(buf.summary_blob), bytes(buf.kv_blob),
                pages_in_group=pages,
            )
            if group_idx >= rs.num_groups_written:
                rs.num_groups_written = group_idx + 1
        except Exception as e:
            logger.warning("WriteGroup failed for req=%s group=%d: %s",
                           request_id, group_idx, e)

    # ─── request lifecycle ───────────────────────────────────────────────

    def on_request_finished(self, request_id: str):
        rs = self._requests.pop(request_id, None)
        if rs is None:
            return
        # Flush any partial group buffers.
        for gidx in list(rs.active_group_buffers.keys()):
            self._flush_group(request_id, gidx, partial=True)
        # Evict from icms.
        if rs.chain:
            try:
                self._client.evict(rs.chain)
            except Exception as e:
                logger.debug("evict failed for %s: %s", request_id, e)

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
