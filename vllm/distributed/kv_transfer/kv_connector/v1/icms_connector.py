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


import errno
import os
import queue as _queue
import sys
import threading
import time

# Bug 8 debug (ICMS_DUMP_ON_USR1=1): register faulthandler so SIGUSR1
# dumps Python stacks for ALL threads to stderr. Used to investigate
# TP=2 + M3 decode hang where ptrace_scope=1 prevents py-spy/gdb attach.
if os.environ.get("ICMS_DUMP_ON_USR1") == "1":
    import faulthandler as _faulthandler_dbg
    import signal as _signal_dbg
    _faulthandler_dbg.register(_signal_dbg.SIGUSR1, all_threads=True)
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import icms_provenance
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
# ── Tracing + stored-chain-queue helpers moved to icms_connector_trace.py
#    (behavior-preserving split). Re-imported so the public + test import
#    surface is unchanged. _stored_chain_generation is a REBOUND int: read it
#    live via _trace._stored_chain_generation (module attr), never the
#    re-exported copy, so cross-thread generation bumps are observed. ──
from vllm.distributed.kv_transfer.kv_connector.v1 import (  # noqa: E402
    icms_connector_trace as _trace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import (  # noqa: E402
    _instr_timing, _icms_trace, _icms_fulltrace, _icms_chain_fp, _allow_batch,
    _pack_fetch_bitmap, _ICMS_TRACE_ENABLED, _ICMS_TRACE_LOCK, _ICMS_TRACE_FH,
    _ICMS_FULLTRACE_ENABLED, _ICMS_FULLTRACE_LOCK, _ICMS_FULLTRACE_FH,
    _stored_chain_queue, _stored_chain_lock, _stored_chain_cond,
    _stored_chain_generation, _bump_stored_chain_generation,
    _append_stored_chain_queue, _drain_stored_chain_queue,
)



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
from icms_client.client import IcmsError                     # noqa: E402
from icms_client.geometry import (                           # noqa: E402
    GROUP_PAGES, PAGE_TOKENS, KvLayout, ModelGeometry, find_model,
    parse_scored_layers,
)
import dataclasses as _dataclasses                            # noqa: E402
from icms_client.sink import Sink, allocate_sink             # noqa: E402

# ── _Scheduler moved to icms_connector_scheduler.py (behavior-preserving
#    split). Imported AFTER the icms_client bootstrap above because the
#    scheduler module imports icms_client.geometry at its top level.
#    Re-exported so the public + test import surface is unchanged. ──
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_scheduler import (  # noqa: E402
    _Scheduler,
)

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
# Legacy default for `icms_sink_slots`. The actual slot count is now
# read from extra_config["icms_sink_slots"] in _Worker.__init__ and
# defaults to max(1, scheduler_config.max_num_seqs) when
# ICMS_ALLOW_BATCH=1, else 1. Kept here for back-compat in case any
# external caller still references this symbol.
_SINK_SLOTS = 4           # legacy default (replaced by extra_config knob)
_GROUP_BLOCKS = GROUP_PAGES  # blocks per group (= pages per group = 32)


# ─── ICMS_TRACE diagnostic logger (2026-05-09) ───────────────────────────
# Env-gated structured one-line JSON trace for diffing against the C++
# server-side trace. Investigates the multi-rid batched-mode bug where
# the server's view of a rid's chain mysteriously shrinks within one
# prefill at layer 6+. Format MUST match the C++ side exactly so the
# diff harness can parse both. Zero overhead when disabled.




# 2026-05-12 multi-rid slot-1 bug-hunt: a comprehensive per-rid per-layer
# per-event JSONL emitter. Cheap when disabled; gated by a single env to
# keep grep across slot 0..3 timelines straightforward.







# TP sharding: each rank prefixes its chain with a rank-tagged sentinel so
# server-side trie entries never collide across ranks. At TP=1 this adds
# one no-op level to the trie; wire and scoring behavior are unchanged.
# See docs/icms_connector_tp_support.md.



# ── TP/NCCL collectives moved to icms_connector_tp.py (behavior-preserving
#    split). Imported back so the public + test import surface is unchanged. ──
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import (  # noqa: E402
    _ICMS_NCCL_GROUP_CACHE,
    _ICMS_NCCL_GROUP_LOCK,
    _RANK_TAG_MAGIC,
    _rank_tag,
    _rank_tagged_chain,
    _get_icms_nccl_group,
    _tp_broadcast_score_reply,
    _tp_allreduce_max_int,
    _tp_allreduce_min_int,
    _tp_broadcast_bool_list,
    _tp_broadcast_bool,
)


# Module-level queue: worker → scheduler stored-chain notifications.
# The worker appends (chain, num_groups) after WriteGroup completes;
# the scheduler drains it in get_num_new_matched_tokens / build_meta.
# Safe because both run in the same EngineCore process.

# 2026-05-09 cross-iter stored-chain race fix.
#
# Bug context (project_block_writes_root_cause / per_rid_event /
# lru_chain_pin all 2026-05-09): in batched per-batch/per-budget mode
# (max_num_seqs>=2), phase 2 (warm) starts before phase 1 (cold) has
# fully drained ALL its rids' write pipelines. The second cold rid's
# WriteGroup may still be in flight when phase 2's scheduler does the
# stored-prefix lookup → matched_groups=0 → _score_one_request early-
# returns at the `total_pages==0` guard → the warm rid never populates
# `_pending_scores[layer][rid]` → wait_for_layer pops a partial dict
# → apply runs for only one of two warm rids → the other rid silently
# falls back to dense over its prefix-cached blocks → garbage output.
#
# Mechanism: scheduler-side `get_num_new_matched_tokens` waits on a
# **monotonic generation counter** when the lookup misses and the
# chain looks substantial. The counter increments on every
# `_stored_chain_queue.append` (worker pipeline thread). The scheduler
# captures `gen` BEFORE the lookup, then if matched=0 waits for
# `gen` to advance — closing the lost-wakeup hole that bare
# `threading.Event.set/clear` pulse semantics had (a producer that
# pulses BEFORE the consumer reaches `wait()` is missed entirely).
#
# This is the cleanup of the original Event-pulse design that shipped
# earlier today; the empirical validation showed pulses landing 8s
# before waiters, all silently lost. See feedback_event_pulse_lost
# _wakeup_2026-05-09 for the failure mode + Condition rewrite.




# Bug #5 fix (race-audit 2026-05-08): atomic append + atomic snapshot+clear
# helpers for `_stored_chain_queue`. Pre-fix, the queue was mutated lock-free
# by worker pipeline threads (append) and consumed lock-free by the
# scheduler thread (bare `for x in queue:` + `queue.clear()`). The
# v2 lost-wakeup fix used Condition for the wait/notify pair but left the
# append + drain operations themselves unsynchronized, so a producer
# append landing between iteration end and the `clear()` was silently
# wiped. These helpers serialize append/drain under `_stored_chain_cond`
# so notifications can't be lost between iter and clear.




# ═══════════════════════════════════════════════════════════════════════════
#  Metadata: scheduler → worker per-step payload (C1)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#  Metadata + dataclasses moved to icms_connector_types.py (behavior-preserving
#  split). Imported back here so the public + test import surface is unchanged.
# ═══════════════════════════════════════════════════════════════════════════
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import (  # noqa: E402
    _PerRequestStep,
    IcmsConnectorMetadata,
    _GroupBuffer,
    IcmsTimingStats,
    _RequestState,
    _SinkSlotPool,
    _WritePipeline,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Facade
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  Scheduler
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  Worker
# ═══════════════════════════════════════════════════════════════════════════

# ── _Worker split into responsibility mixins (behavior-preserving). Composed
#    below; each mixin imports its deps from neutral modules (no cycle back
#    into icms_connector). Imported AFTER the icms_client bootstrap above. ──
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_worker_base import _WorkerBase  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_worker_scoring import _WorkerScoringMixin  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_worker_fetch import _WorkerFetchApplyMixin  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_worker_write import _WorkerWritePipelineMixin  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_worker_state import _WorkerStateMixin  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_worker_diag import _WorkerDiagMixin  # noqa: E402


class _Worker(
    _WorkerScoringMixin,
    _WorkerFetchApplyMixin,
    _WorkerWritePipelineMixin,
    _WorkerStateMixin,
    _WorkerDiagMixin,
    _WorkerBase,
):
    """Worker-side state. Owns IcmsClient, sink pool, per-request buffers."""








    # ─── TTFT-breakdown helpers ──────────────────────────────────────────






    # ─── runtime invariant flags ────────────────────────────────────────







    # ─── layer hooks (from save_kv_layer kwargs) ─────────────────────────

































    # ─── KV extraction from GPU paged buffer (C2 write path) ────────────




    # ─── group buffer management (C2) ────────────────────────────────────

        # NOTE: no inline flush on completion. Flushing is deferred to
        # wait_for_pending_writes (called at the end of each forward
        # pass by vLLM) so that write_group RPCs and the extraction
        # work don't stall per-layer save_kv_layer calls during the
        # prefill forward pass.




    # ─── request lifecycle ───────────────────────────────────────────────


        # No per-rid Evict here.
        #
        # Eviction is a 3-layer design:
        #   1. Local LRU on the data node — reactive memory-pressure
        #      handler invoked from WriteGroup when the allocator OOMs
        #      (handlers.cc lru_evict_oldest). Always on, refcount-aware
        #      so shared prefixes survive.
        #   2. Explicit kEvict RPC — server-side API for callers with
        #      global visibility (handlers.cc handle_evict).
        #   3. Distributed KV cache manager (future) — the upstream
        #      caller of (2). Tracks chain references across data nodes
        #      and drives explicit evictions.
        #
        # The connector is a worker, not the global manager. It cannot
        # see whether a chain is still referenced elsewhere (e.g., turn-1
        # of a NIAH bench example finishing while turn-2 is about to
        # start with the same haystack prefix). Firing per-rid Evict here
        # broke the dedup-skip optimization in extract_and_record:
        # the local ledger said "N groups stored" but the server-side
        # trie had just been emptied → Score returned ENOENT.
        #
        # Until (3) lands, capacity is handled by (1) alone. crashed-
        # client / clean-shutdown cleanup is handled by on_closed
        # server-side. Bench-level explicit cleanup, if needed, should
        # call client.evict(chain) AFTER all budget iterations for an
        # example complete — not at every per-rid request_finished.

    # ─── direct helper API (backward compat with smoke tests) ────────────



    # ─── helpers ─────────────────────────────────────────────────────────



# ── IcmsConnector facade moved to icms_connector_facade.py (behavior-preserving
#    split). Re-exported here so the public path
#    'vllm...v1.icms_connector.IcmsConnector' (and all test importers) resolve
#    unchanged. Placed at end-of-file so _Worker (referenced function-locally by
#    the facade at instantiation time) is fully defined first. ──
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_facade import (  # noqa: E402
    IcmsConnector,
)






