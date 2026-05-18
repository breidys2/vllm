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
_ICMS_TRACE_ENABLED = os.environ.get("ICMS_TRACE") == "1"
_ICMS_TRACE_LOCK = threading.Lock()
_ICMS_TRACE_FH = None


def _icms_trace(op, rid, layer=-1, chain_fp="", pc_before=-1, pc_after=-1,
                extra=None):
    """Append one JSON line to the ICMS trace file.

    Schema (must match C++ side):
      {"ts_ns":<int>,"side":"connector","op":"<op>","rid":"<rid>",
       "layer":<int_or_-1>,"chain_fp":"<hex>","pc_before":<int>,
       "pc_after":<int>,"tid":<int>,"extra":{...}}

    No-op when ICMS_TRACE != "1". Caller pays only the env-cached bool
    check on the hot path.
    """
    if not _ICMS_TRACE_ENABLED:
        return
    global _ICMS_TRACE_FH
    import json as _json_trace
    rec = {
        "ts_ns": time.monotonic_ns(),
        "side": "connector",
        "op": op,
        "rid": str(rid),
        "layer": int(layer) if layer is not None else -1,
        "chain_fp": chain_fp or "",
        "pc_before": int(pc_before) if pc_before is not None else -1,
        "pc_after": int(pc_after) if pc_after is not None else -1,
        "tid": threading.get_ident(),
        "extra": extra if extra is not None else {},
    }
    try:
        line = _json_trace.dumps(rec)
    except (TypeError, ValueError):
        # Fallback for non-serializable extras: stringify them.
        rec["extra"] = {k: str(v) for k, v in (extra or {}).items()}
        line = _json_trace.dumps(rec)
    with _ICMS_TRACE_LOCK:
        if _ICMS_TRACE_FH is None:
            _path = os.environ.get(
                "ICMS_TRACE_FILE", "/tmp/icms_trace_connector.jsonl")
            _ICMS_TRACE_FH = open(_path, "a", buffering=1)
        _ICMS_TRACE_FH.write(line + "\n")


# 2026-05-12 multi-rid slot-1 bug-hunt: a comprehensive per-rid per-layer
# per-event JSONL emitter. Cheap when disabled; gated by a single env to
# keep grep across slot 0..3 timelines straightforward.
_ICMS_FULLTRACE_ENABLED = os.environ.get("ICMS_DIAG_FULLTRACE") == "1"
_ICMS_FULLTRACE_LOCK = threading.Lock()
_ICMS_FULLTRACE_FH = None


def _icms_fulltrace(op, rid="", layer=-1, **extra):
    """Append a structured JSONL record for ICMS_DIAG_FULLTRACE=1.

    Emits to `${ICMS_DIAG_FULLTRACE_FILE}` (default
    /tmp/icms_fulltrace.jsonl). One line per call. All `extra` kwargs
    are JSON-serialized; non-serializable values are stringified.
    Caller pays only the env-cached bool check on the hot path.
    """
    if not _ICMS_FULLTRACE_ENABLED:
        return
    global _ICMS_FULLTRACE_FH
    import json as _json_ft
    rec = {
        "ts_ns": time.monotonic_ns(),
        "op": op,
        "rid": str(rid),
        "layer": int(layer) if layer is not None else -1,
        "tid": threading.get_ident(),
        "pid": os.getpid(),
        **extra,
    }
    try:
        line = _json_ft.dumps(rec)
    except (TypeError, ValueError):
        safe = {k: (v if isinstance(v, (int, float, str, bool, list, dict,
                                         type(None))) else str(v))
                for k, v in rec.items()}
        try:
            line = _json_ft.dumps(safe)
        except Exception:
            line = _json_ft.dumps({"op": op, "rid": str(rid),
                                    "layer": int(layer or -1),
                                    "_serialize_error": True})
    with _ICMS_FULLTRACE_LOCK:
        if _ICMS_FULLTRACE_FH is None:
            _path = os.environ.get(
                "ICMS_DIAG_FULLTRACE_FILE", "/tmp/icms_fulltrace.jsonl")
            _ICMS_FULLTRACE_FH = open(_path, "a", buffering=1)
        _ICMS_FULLTRACE_FH.write(line + "\n")


def _icms_chain_fp(chain) -> str:
    """Best-effort fingerprint of a chain (list of ints). Returns hex of
    a short blake2b digest of the chain's repr. Empty string on failure
    or empty input.
    """
    if not chain:
        return ""
    try:
        import hashlib as _hl_fp
        # chain entries are 64-bit ints; pack as decimal-comma string.
        # Cheap and stable across runs.
        b = ",".join(str(int(x)) for x in chain).encode("ascii")
        return _hl_fp.blake2b(b, digest_size=8).hexdigest()
    except Exception:
        return ""


def _allow_batch() -> bool:
    """ICMS_ALLOW_BATCH=1 gates the multi-rid (N>=2) batching path.

    When unset (default), the connector behaves exactly as the
    single-request implementation: per-rid `set_active`, first-rid-only
    extract, global `_skip_extract`. When set:
      - wait_for_layer aggregates per-rid trimmed states into one
        multi-row IcmsFetchState per layer (see icms_fetch_state.py).
      - extract_and_record walks every active rid (per-layer single
        AllGather at TP>1).
      - _skip_extract becomes a per-rid set, not a global bool.

    The flag is read fresh on each call so tests can flip it without a
    process restart. See docs/icms_vllm_integration_audit_2026-05-05.md
    for the Phase A landing plan.
    """
    return os.environ.get("ICMS_ALLOW_BATCH") == "1"

# TP sharding: each rank prefixes its chain with a rank-tagged sentinel so
# server-side trie entries never collide across ranks. At TP=1 this adds
# one no-op level to the trie; wire and scoring behavior are unchanged.
# See docs/icms_connector_tp_support.md.
_RANK_TAG_MAGIC = 0xE1C5A4EE00000000

def _pack_fetch_bitmap(fetched: "set[int]", total_pages: int) -> bytes:
    """Pack a set of page IDs into the decode-mode bitmap wire format.

    bit n of byte n/8 (LSB-first) ⇒ page_id n is "already fetched" on
    the client. Sized to ceil(total_pages / 8) bytes. Page IDs outside
    [0, total_pages) are ignored. M3 calls this on a stride-group's
    fetched_pages set to build the wire suffix for decode-mode Score.
    """
    n_bytes = (total_pages + 7) // 8
    if n_bytes == 0 or not fetched:
        return b"\x00" * n_bytes
    bm = bytearray(n_bytes)
    for pid in fetched:
        if 0 <= pid < total_pages:
            bm[pid >> 3] |= 1 << (pid & 0x7)
    return bytes(bm)


def _rank_tag(tp_rank: int) -> int:
    return _RANK_TAG_MAGIC | int(tp_rank)

def _rank_tagged_chain(tp_rank: int, chain):
    if chain is None:
        return chain
    return [_rank_tag(tp_rank), *chain]


# Cached separate NCCL group for ICMS connector collectives.
# When ICMS_USE_SEPARATE_NCCL_GROUP=1, the connector's NCCL ops
# (extract_and_record's K/V AllGather on the pipeline thread + the
# main-thread Score/FetchAll Q AllGather + reply broadcasts) are
# routed through a dedicated NCCL communicator instead of the default
# TP group. This eliminates contention with vLLM's main-thread
# NCCL ops (per-layer all_reduce, attention all_gather), which is the
# root cause of the chunked-prefill TP>1 + fetchall TP>1 hang
# (shm_broadcast TimeoutError).
#
# Same world (ranks) as the TP group, but a separate NCCL communicator
# → ops on different communicators don't serialize through NCCL's
# single-stream queue.
_ICMS_NCCL_GROUP_CACHE: dict = {"group": None, "init_attempted": False}
# 2026-05-10 audit fix #6: serialize the lazy first-call init.
# Without this lock, two threads (e.g. forward + pipeline) can race
# past the `init_attempted` check simultaneously, each calling
# `dist.new_group` — itself a NCCL collective that must be invoked by
# every rank in the same order. If rank-0's forward thread races
# rank-1's pipeline thread, the resulting ProcessGroup objects can
# mismatch across ranks → first collective on the sep-NCCL group
# silently hangs. The check-then-set on `init_attempted` is also not
# atomic in CPython for compound operations.
_ICMS_NCCL_GROUP_LOCK: threading.Lock = threading.Lock()


def _get_icms_nccl_group():
    """Return the separate NCCL group for ICMS collectives, or None
    if disabled / unavailable.

    Lazy: created on first call under `_ICMS_NCCL_GROUP_LOCK` so
    concurrent first callers across threads share a single
    `dist.new_group` invocation. Subsequent calls hit the lock-free
    fast path. None means caller should fall back to the default TP
    device group (legacy behavior).
    """
    if os.environ.get("ICMS_USE_SEPARATE_NCCL_GROUP", "0") != "1":
        return None
    # Fast path: once init has been attempted (and the cache fully
    # populated under the lock below), subsequent reads are lock-free.
    if _ICMS_NCCL_GROUP_CACHE["init_attempted"]:
        return _ICMS_NCCL_GROUP_CACHE["group"]
    with _ICMS_NCCL_GROUP_LOCK:
        # Double-checked: another thread may have finished init while
        # we were waiting on the lock.
        if _ICMS_NCCL_GROUP_CACHE["init_attempted"]:
            return _ICMS_NCCL_GROUP_CACHE["group"]
        try:
            import torch.distributed as dist
            from vllm.distributed.parallel_state import get_tp_group
            tp_group = get_tp_group()
            # Build a fresh NCCL group with the same ranks as the TP
            # group. All TP-rank workers must call this in the same
            # order; the lock above guarantees the first caller within
            # this process runs to completion before any other thread
            # observes init_attempted=True. The production code paths
            # that reach this site are symmetric across ranks.
            new_group = dist.new_group(
                ranks=list(tp_group.ranks), backend="nccl")
            # Pre-warm the NCCL communicator with a tiny no-op
            # collective. `dist.new_group` returns immediately with a
            # ProcessGroup object but defers the real NCCL P2P channel
            # + buffer setup to the first collective op. Forcing setup
            # here means OOM (if any) surfaces at init time rather
            # than during a hot-path Score broadcast where the failure
            # manifests as "Rank N has no transport for send peer M".
            # A 1-element broadcast is enough to trigger the lazy
            # setup.
            try:
                dev = torch.device(
                    f"cuda:{torch.cuda.current_device()}")
                warmup_t = torch.zeros(1, dtype=torch.int64, device=dev)
                dist.broadcast(warmup_t, src=tp_group.first_rank,
                               group=new_group)
                torch.cuda.synchronize()
            except Exception as warm_e:
                logger.warning(
                    "[icms-nccl] separate NCCL group warmup broadcast "
                    "failed (%s) — falling back to TP device group. "
                    "Try lowering --gpu-memory-utilization to leave "
                    "headroom for NCCL P2P channels (~512 MiB on "
                    "H100 NVL).", warm_e)
                # Cache the negative result under the lock so other
                # threads' lock-free fast path sees (group=None,
                # init_attempted=True) atomically. Order matters:
                # populate `group` BEFORE flipping init_attempted.
                _ICMS_NCCL_GROUP_CACHE["group"] = None
                _ICMS_NCCL_GROUP_CACHE["init_attempted"] = True
                return None
            _ICMS_NCCL_GROUP_CACHE["group"] = new_group
            _ICMS_NCCL_GROUP_CACHE["init_attempted"] = True
            logger.info(
                "[icms-nccl] Created separate NCCL group for ICMS "
                "collectives (ranks=%s) — warmup OK",
                list(tp_group.ranks))
            return new_group
        except Exception as e:
            logger.warning(
                "[icms-nccl] Failed to create separate NCCL group, "
                "falling back to TP device group: %s", e)
            _ICMS_NCCL_GROUP_CACHE["group"] = None
            _ICMS_NCCL_GROUP_CACHE["init_attempted"] = True
            return None


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
    dev_group = _get_icms_nccl_group() or tp_group.device_group
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Header: [status, trie_walk_ns, summary_read_ns, score_ns,
    #          sink_write_ns, cache_hit, concurrent_requests,
    #          server_ingest_to_ready_ns, effective_supply_bps,
    #          request_id, num_pages]  — 11 i64 slots.
    # Bug fix (2026-04-30): added effective_supply_bps (slot 8) to match
    # the protocol's two-sided adaptive-bandwidth wire format. Without
    # this, the dataclass reconstruction at the end of this function
    # raised "ScoreReply.__init__() missing effective_supply_bps" on
    # every TP>1 broadcast and Score replies never reached non-rank-0
    # workers — wedging decode at iter 1 with shm_broadcast timeouts.
    # 2026-05-05 (v5): added request_id (slot 9). Without it, the
    # broadcast-side ScoreReply reconstruction misses the new field
    # and raises "missing 1 required positional argument: 'request_id'"
    # — same failure mode as the 2026-04-30 fix above.
    # request_id is a 64-bit unsigned value (hash output of
    # _icms_request_id). torch.int64 has range [-2^63, 2^63), so any
    # u64 with bit 63 set overflows when packed via int(...). Reinterpret
    # through signed int64 here and on the receive side: the bit pattern
    # round-trips, only the int's sign flips for high half-range values.
    def _u64_to_i64(v: int) -> int:
        v &= 0xFFFFFFFFFFFFFFFF
        return v if v < (1 << 63) else v - (1 << 64)
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
            int(reply.effective_supply_bps),
            _u64_to_i64(int(getattr(reply, "request_id", 0))),
            int(n_pages),
        ], dtype=torch.int64, device=dev)
    else:
        hdr = torch.zeros(11, dtype=torch.int64, device=dev)
    dist.broadcast(hdr, src=tp_group.first_rank, group=dev_group)

    def _i64_to_u64(v: int) -> int:
        return v if v >= 0 else v + (1 << 64)

    n_pages = int(hdr[10].item())
    if n_pages == 0:
        from icms_client.protocol import ScoreReply
        return ScoreReply(
            request_id=_i64_to_u64(int(hdr[9])),
            status=int(hdr[0]), trie_walk_ns=int(hdr[1]),
            summary_read_ns=int(hdr[2]), score_ns=int(hdr[3]),
            sink_write_ns=int(hdr[4]), cache_hit=bool(hdr[5]),
            concurrent_requests=int(hdr[6]),
            server_ingest_to_ready_ns=int(hdr[7]),
            effective_supply_bps=int(hdr[8]),
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
        request_id=_i64_to_u64(int(hdr[9])),
        status=int(hdr[0]), trie_walk_ns=int(hdr[1]),
        summary_read_ns=int(hdr[2]), score_ns=int(hdr[3]),
        sink_write_ns=int(hdr[4]), cache_hit=bool(hdr[5]),
        concurrent_requests=int(hdr[6]),
        server_ingest_to_ready_ns=int(hdr[7]),
        effective_supply_bps=int(hdr[8]),
        page_ids=pids.cpu().tolist(),
        scores=scs.cpu().tolist(),
        sink_offsets=offs.cpu().tolist(),
    )

def _tp_allreduce_max_int(value: int, tp_size: int) -> int:
    """All-reduce-MAX of an int64 across TP ranks.

    Used by `_drain_pending_flush_queue` (2026-05-10) to symmetrize
    per-rank pipeline progress before issuing N collective broadcasts.
    Pre-fix, asymmetric local queue lengths caused one rank to call
    `_tp_broadcast_bool` more times than the other → CUDA hang at
    NCCL collective. Post-fix, every rank derives the same `n` here
    and broadcasts exactly that many times (padding shorter local
    queues with sentinels).

    TP=1 → return value unchanged. TP>1 → tiny i64 all-reduce on the
    ICMS TP group. Cost: ~few μs; called once per drain.

    Failure mode: on collective failure, return the local value (best
    effort — drain proceeds with whatever local entries it has, which
    may deadlock; the warning log makes the failure visible).
    """
    if tp_size <= 1:
        return int(value)
    import torch.distributed as dist  # noqa: E402
    from vllm.distributed.parallel_state import get_tp_group
    try:
        tp_group = get_tp_group()
        dev_group = _get_icms_nccl_group() or tp_group.device_group
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        t = torch.tensor([int(value)], dtype=torch.int64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=dev_group)
        return int(t.item())
    except Exception as e:
        logger.warning(
            "[icms-tp] _tp_allreduce_max_int failed: %r — "
            "falling back to local value (drain may deadlock if "
            "queue lengths are asymmetric)", e)
        return int(value)


def _tp_allreduce_min_int(value: int, tp_size: int) -> int:
    """All-reduce-MIN of an int64 across TP ranks.

    Audit #20 fix (2026-05-11): symmetrize per-rank shape-derived
    bounds before they gate per-rank filtering. Canonical use:
    `bt_row_max = int(bt.shape[1])` in the apply slow-path — if
    vLLM ever allocates differently sized block-table rows per rank
    for the same rid, post-filter `valid_pids` diverges across ranks
    and the scatter is asymmetric. MIN-reducing the bound ensures
    every rank drops the same pids.

    TP=1 → return value unchanged. TP>1 → tiny i64 all-reduce on the
    ICMS TP group. Cost: ~few μs; called once per apply slow-path.

    Failure mode: on collective failure, return the local value (best
    effort — apply proceeds with the local bound, which may be
    asymmetric across ranks but is no worse than pre-fix behavior).
    """
    if tp_size <= 1:
        return int(value)
    import torch.distributed as dist  # noqa: E402
    from vllm.distributed.parallel_state import get_tp_group
    try:
        tp_group = get_tp_group()
        dev_group = _get_icms_nccl_group() or tp_group.device_group
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        t = torch.tensor([int(value)], dtype=torch.int64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=dev_group)
        return int(t.item())
    except Exception as e:
        logger.warning(
            "[icms-tp] _tp_allreduce_min_int failed: %r — "
            "falling back to local value (post-filter pids may be "
            "asymmetric across ranks)", e)
        return int(value)


def _tp_broadcast_bool(value: bool, tp_rank: int, tp_size: int) -> bool:
    """Broadcast rank-0's bool to every rank. Used by `_flush_group` to
    propagate WriteGroup success/failure so all ranks bump
    `flushed_local`/`num_groups_written` symmetrically.

    TP=1 → returns the input (no collective). TP>1 → tiny i64 broadcast
    over the same NCCL group as `_tp_broadcast_score_reply`. Cost:
    ~few microseconds; called once per `_flush_group` (≤ K per request).

    Failure mode: if the broadcast itself raises (e.g., NCCL channel
    not warmed yet), we conservatively return False so all ranks treat
    it as a write failure. Better to skip a bump symmetrically than to
    diverge. Pre-2026-05-08 this divergence (rank-0 raise on
    write_group → rank-0 skips bump, rank-1 doesn't) was the TP=2
    multi-rid hang root cause; see project_tp2_writegroup_asymmetry_2026-05-07.
    """
    if tp_size <= 1:
        return value
    import torch.distributed as dist  # noqa: E402
    from vllm.distributed.parallel_state import get_tp_group
    try:
        tp_group = get_tp_group()
        dev_group = _get_icms_nccl_group() or tp_group.device_group
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        t = torch.tensor([1 if (tp_rank == 0 and value) else 0],
                         dtype=torch.int64, device=dev)
        dist.broadcast(t, src=tp_group.first_rank, group=dev_group)
        return bool(int(t.item()))
    except Exception as e:
        logger.warning(
            "[icms-tp] _tp_broadcast_bool failed (rank=%d): %r — "
            "treating as False on every rank to keep ledgers symmetric",
            tp_rank, e)
        return False


# Module-level queue: worker → scheduler stored-chain notifications.
# The worker appends (chain, num_groups) after WriteGroup completes;
# the scheduler drains it in get_num_new_matched_tokens / build_meta.
# Safe because both run in the same EngineCore process.
_stored_chain_queue: list[tuple[list[int], int]] = []

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
_stored_chain_lock: threading.Lock = threading.Lock()
_stored_chain_cond: threading.Condition = threading.Condition(_stored_chain_lock)
_stored_chain_generation: int = 0


def _bump_stored_chain_generation() -> None:
    """Increment the generation counter and wake any waiter. Called by
    the worker pipeline thread immediately after every
    `_stored_chain_queue.append`."""
    global _stored_chain_generation
    with _stored_chain_cond:
        _stored_chain_generation += 1
        _stored_chain_cond.notify_all()


# Bug #5 fix (race-audit 2026-05-08): atomic append + atomic snapshot+clear
# helpers for `_stored_chain_queue`. Pre-fix, the queue was mutated lock-free
# by worker pipeline threads (append) and consumed lock-free by the
# scheduler thread (bare `for x in queue:` + `queue.clear()`). The
# v2 lost-wakeup fix used Condition for the wait/notify pair but left the
# append + drain operations themselves unsynchronized, so a producer
# append landing between iteration end and the `clear()` was silently
# wiped. These helpers serialize append/drain under `_stored_chain_cond`
# so notifications can't be lost between iter and clear.
def _append_stored_chain_queue(chain: list[int], n_groups: int) -> None:
    """Atomically append to the worker→scheduler chain queue, bump the
    generation counter, and wake any waiter. Producers must use this
    helper rather than bare `_stored_chain_queue.append(...)` so a
    concurrent drain can't iterate-then-clear past this notification."""
    global _stored_chain_generation
    with _stored_chain_cond:
        _stored_chain_queue.append((chain, n_groups))
        _stored_chain_generation += 1
        _stored_chain_cond.notify_all()


def _drain_stored_chain_queue() -> list[tuple[list[int], int]]:
    """Atomically snapshot + clear the worker→scheduler chain queue.
    Returns the snapshot for the caller to process OUTSIDE the lock
    (so the lock isn't held across `record_stored_chain` work).
    Replaces the bare `for ... in _stored_chain_queue: ... ;
    _stored_chain_queue.clear()` pattern, which was racy: a producer
    append landing between the iteration's end and the clear was
    silently wiped (Bug #5)."""
    with _stored_chain_cond:
        if not _stored_chain_queue:
            return []
        snapshot = list(_stored_chain_queue)
        _stored_chain_queue.clear()
    return snapshot


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
    # KV-block provenance tracing (ICMS_TRACE_KV_PROVENANCE). Per-rid counts
    # of ICMS-elided tokens and vLLM-prefix-cached tokens, plus the alloc'd
    # block_ids list, ferried scheduler→worker so the worker-side tracker
    # can derive which blocks ICMS is responsible for populating.
    prov_ext_comp_tokens: dict[str, int] = field(default_factory=dict)
    prov_local_cached_tokens: dict[str, int] = field(default_factory=dict)
    prov_block_ids: dict[str, list[int]] = field(default_factory=dict)
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
    # 2026-05-06: scheduler-side stored_groups lookup, ferried via
    # metadata to the worker. The worker's local `_stored_chain_groups`
    # ledger is populated only when the deferred write pipeline's
    # `_record_stored_groups` runs (and via on_request_finished).
    # At turn N+1's first `on_step_start`, that ledger is racy —
    # turn N's pipeline may not have committed yet — so the worker's
    # `_get_stored_context_groups` returns 0 → `_score_one_request`
    # early-returns at `total_pages == 0` → turn N+1's prefill Score
    # NEVER fires with the question's Q. The scheduler's
    # `_stored_chains` is up-to-date because it drains
    # `_stored_chain_queue` at the top of `build_meta`. Pass the
    # authoritative value here so the worker doesn't have to consult
    # its own potentially-stale cache.
    stored_groups_by_rid: dict[str, int] = field(default_factory=dict)


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
    num_groups_written: int = 0                            # groups covered by THIS request's
                                                            # writes OR inherited stored prefix
                                                            # (drives skip-extract elision)
    flushed_local: int = 0                                  # groups THIS request actually
                                                            # flushed via _flush_group success
                                                            # (drives stored-prefix recording —
                                                            # never inflated by elision path)
    active_group_buffers: dict[int, _GroupBuffer] = field(default_factory=dict)  # group_idx → buf
    stored_groups: int = 0  # groups already in ICMS under this chain prefix (dedup-aware skip)
    # 2026-05-10 TP>1 stored_groups symmetrization (extract-side):
    # Pipeline-thread `extract_and_record` reads this instead of
    # `stored_groups` so it can avoid issuing NCCL itself (collides
    # with main-thread per-layer all_reduce). Forward-thread
    # `on_step_start` populates it via _tp_allreduce_max_int(n_stored).
    # Same value as `stored_groups` at TP=1; possibly larger at TP>1.
    _effective_stored_groups: int = 0
    # Per-stride apply cache (set at the scored layer, reused on the
    # following stride-1 reuse layers). The block_table layout, seq_len,
    # phys_blocks, and per-page page_idx baseline are identical across all
    # layers in a stride — only the sink_offset shifts by delta * actual_k
    # pages. Caching skips ~0.5–1 ms of Python pid filter / sort / dict /
    # tensor build at every reuse layer (5 of every 6 layers).
    _apply_cached_layer_start: int = -1
    _apply_cached_phys_blocks_dev: object = None  # torch.Tensor or None
    _apply_cached_page_idx_dev: object = None
    _apply_cached_actual_k: int = 0
    _apply_cached_valid_pids: object = None  # diag-only: list[int] or None
    _apply_cached_new_bt: object = None
    _apply_cached_new_sl: object = None
    _apply_cached_max_seq_len: int = 0
    _apply_cached_filled_count: int = 0
    # 2026-05-11 audit fix #21: cumulative-set size at the moment
    # `_apply_cached_new_bt` was baked. The fast path checks this
    # against `len(rs.fetched_pages[stride_root])` at entry; if the
    # cumulative set grew between the slow path's bake and the fast
    # path's reuse (decode/cross-iter async fetch-all adding pages
    # mid-stride), the cached `new_bt` is stale and the fast path
    # would attend to wrong / missing pages → silent KV gap
    # (anomaly C signature on mistral-small h128k niah_single).
    # -1 sentinel = "never cached" → fast path treats as invalid.
    _apply_cached_cumulative_count: int = -1
    # Per-request cached `cont_idx` device tensor for the trimmed
    # block-table tail. cont_idx depends on (context_pages, cont_end)
    # which are per-request constants during prefill — cache once at the
    # first scored layer, reuse on every subsequent stride.
    _apply_cached_cont_idx_dev: object = None
    _apply_cached_cont_idx_range: tuple = (0, 0)  # (context_pages, cont_end)
    # Decode-mode fetch tracking (M2 of icms_decode_path_plan).
    # Maps the *scored layer index* (e.g. 0, 6, 12, … for stride=6) to
    # the set of page_ids already returned by Score / FetchAll for the
    # stride group rooted at that layer. Updated after every Score
    # reply during prefill (so by the prefill→decode transition the
    # set already reflects everything fetched so far) and during
    # decode iters once M3 wires the decode hooks. Use _pack_fetch_bitmap
    # to encode for the wire suffix.
    fetched_pages: dict = field(default_factory=dict)
    # ICMS_ORIGINAL_QUEST=1 only: per-layer GPU-side K-min/K-max summary
    # stack, populated incrementally as pages are staged during prefill.
    # Shape per layer: dict[layer_idx -> tuple[Tensor, Tensor]] where each
    # Tensor is [P_so_far, num_kv_heads, head_dim] fp16. Replaces the
    # BF2-side summary store for the local Quest scorer. Empty (and
    # untouched) when ICMS_ORIGINAL_QUEST is unset — no impact on the
    # default path. See quest_local_scorer.py.
    quest_gpu_summaries: dict = field(default_factory=dict)
    # ICMS_DIAG_SCORE_DUMP only: per-(scored layer) Q tensor snapshot,
    # captured inside _score_one_request and used by on_request_finished
    # to write a complete (q + kmin/kmax + picked_page_ids) bundle for
    # offline alt-scoring analysis. The per-layer .pt file written at
    # Score time has q + picked but no kmin/kmax (extract_and_record
    # hasn't run yet); the per-rid summaries .pt has kmin/kmax but no q
    # — so we stash q here at Score time and join on rid in the
    # summaries dump. Empty when ICMS_DIAG_SCORE_DUMP is unset.
    last_q_by_layer: dict = field(default_factory=dict)
    # ICMS_DIAG_SCORE_DUMP only: per-(scored layer) Score reply snapshot
    # — what page IDs Score returned and what scores it gave them.
    # Stashed alongside last_q_by_layer so the per-rid summaries dump
    # bundles {q, kmin, kmax, picked, server_scores} per layer.
    last_picked_by_layer: dict = field(default_factory=dict)
    last_scores_by_layer: dict = field(default_factory=dict)
    # M4: once a decode-mode Score reply yields 0 net-new pages for any
    # stride group, the bitmap is effectively saturated — flip the
    # request into dense mode and skip all further Score RPCs / Quest
    # hooks until the request finishes. Adaptive to chain growth: as
    # long as Score keeps returning new pages, we keep scoring.
    dense_mode: bool = False
    # ICMS_DIAG_FULL: counter that increments each forward pass after
    # dense_mode flips. Used to gate verbose post-dense-flip metadata
    # logging — first ~3 iters after the flip are the most likely to
    # carry stale state into the natural-bt decode path.
    _post_dense_iter: int = -1  # -1 = pre-flip, 0,1,2,... = post-flip iter
    # Step 2 per-rid Condition + flush_seq ordering fix
    # (2026-05-09; 2026-05-10 audit #5 follow-up). Bumped by
    # `_drain_pending_flush_queue` after each successful WriteGroup
    # commit; awaited by `_score_one_request` when chain coverage
    # lags or on ENOENT retry.
    #
    # History:
    #   * Pre-fix the consumer used `time.sleep(retry_delay)` —
    #     correct but slow.
    #   * 2026-05-09 swap to `threading.Event.set() + .clear()` — fast
    #     but LOSES WAKEUPS: if the producer pulses BEFORE the
    #     consumer enters wait(), the consumer hangs for the full
    #     retry timeout. Same bug class as the v1 `_stored_chain_event`
    #     before that was switched to Condition+gen.
    #   * 2026-05-10 (audit #5): swap to threading.Condition + the
    #     existing `flush_seq` monotonic counter. Consumer snapshots
    #     `flush_seq` BEFORE waiting; producer increments under the
    #     Condition lock + notify_all. Late pulses are observed via
    #     the counter, not lost. See `feedback_event_pulse_lost_wakeup_2026-05-09.md`
    #     for the standing rule.
    flush_cond: threading.Condition = field(
        default_factory=threading.Condition)
    flush_seq: int = 0
    # FAPS audit Finding 4 fix (2026-05-11): track the number of
    # server-side-committed groups (`flushed_local + stored_groups`)
    # AT the moment FAPS' first FetchAll RPC completed. Used by
    # `_fetch_all_one_request`'s top-of-function check to detect
    # chain growth between chunks of a chunked prefill (e.g.,
    # mistral-small >16K with default `max_num_batched_tokens=16384`).
    # When new groups commit AFTER FAPS dispatched, the cached
    # `_pending_reuse` offsets only cover the chunk-0 page set →
    # subsequent chunks' apply paths see under-coverage → silent KV
    # mismatch. Invalidating `_fetch_all_complete` on growth forces a
    # fresh FAPS dispatch covering the new chain.
    _fetch_all_committed_at_dispatch: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  Sink slot allocator (C8)
# ═══════════════════════════════════════════════════════════════════════════

class _SinkSlotPool:
    """Pre-allocated fixed-size sink partitioned into N slots.

    NOTE (2026-05-05): the per-RPC slot allocation API
    (acquire/release/offset_for_slot) is currently DORMANT — the server
    chose its own offsets within the registered sink in 60913a126 (Apr
    2026), so the connector no longer hands out client-side slots.
    The pool object survives because:
      (a) `n_slots` documents the connector's expectation of how many
          concurrent in-flight RPCs the sink can hold (used at sink
          sizing time to multiply total_sink by n_slots).
      (b) The acquire/release API is preserved (now Semaphore-backed,
          not spin-wait) so a future re-introduction of client-side
          slot allocation doesn't have to re-plumb the call sites.
    The Semaphore replaces the prior spin-wait so saturated callers
    block on a kernel primitive instead of a busy-loop.
    """

    def __init__(self, sink: Sink, slot_bytes: int, n_slots: int):
        self.sink = sink
        self.slot_bytes = slot_bytes
        self.n_slots = n_slots
        self._free: list[int] = list(range(n_slots))
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(n_slots)

    def acquire(self) -> int:
        """Get a free slot index. Blocks if none available."""
        self._sem.acquire()
        with self._lock:
            return self._free.pop()

    def release(self, slot: int):
        with self._lock:
            self._free.append(slot)
        self._sem.release()

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
        # B2 (2026-05-05): per-rid task counter so on_request_finished can
        # drain only the finishing rid's pending tasks instead of the
        # entire pipeline. Each submit increments every rid in `rids`;
        # task completion decrements them. drain_rid(X) waits on X's
        # count alone. At N=1 this collapses to the legacy global drain.
        self._rid_pending: dict[str, int] = {}
        self._cv = threading.Condition()
        self._stop = False
        self._t = threading.Thread(
            target=self._loop, name=name, daemon=True)
        self._t.start()

    def submit(self, fn, tag: str = "", rids: "list[str] | None" = None):
        rid_list = list(rids) if rids else []
        with self._cv:
            self._pending += 1
            for r in rid_list:
                self._rid_pending[r] = self._rid_pending.get(r, 0) + 1
        self._q.put((fn, tag, rid_list))

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:  # poison
                return
            # Back-compat: legacy 2-tuple form (fn, tag).
            if len(item) == 2:
                fn, tag = item
                rid_list = []
            else:
                fn, tag, rid_list = item
            try:
                fn()
            except Exception:
                logger.exception("WritePipeline[%s]: task failed", tag)
            finally:
                with self._cv:
                    self._pending -= 1
                    for r in rid_list:
                        n = self._rid_pending.get(r, 0) - 1
                        if n <= 0:
                            self._rid_pending.pop(r, None)
                        else:
                            self._rid_pending[r] = n
                    if self._pending == 0 or rid_list:
                        # Wake every drainer; each rechecks its own
                        # predicate (per-rid or global).
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

    def drain_rid(self, rid: str,
                   timeout: float | None = None) -> bool:
        """Block until tasks involving `rid` are all complete.

        Tasks that don't touch this rid are NOT awaited — the legacy
        full-pipeline drain forced a finishing rid to wait for in-flight
        writes of unrelated still-active rids. With per-rid tagging
        added at submit() time, this is the right semantic for
        on_request_finished. Returns False on timeout.

        Falls back to global drain when no submit ever tagged this rid
        (e.g., the rid finished without any writes, or the caller
        passed rids=None).
        """
        with self._cv:
            if rid not in self._rid_pending:
                return True
            if timeout is None:
                while self._rid_pending.get(rid, 0) > 0:
                    self._cv.wait()
                return True
            deadline = time.monotonic() + timeout
            while self._rid_pending.get(rid, 0) > 0:
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
        # Cross-budget KV reuse (2026-05-05): per-call budget override
        # poked by `experiments/accuracy_bench_icms.py` via
        # llm.collective_rpc(_set_icms_budget, args=(b,)) so the bench can
        # sweep multiple budgets against ONE LLM instead of rebuilding
        # the model per budget. None = use init-time `icms_budget`.
        self._budget_override: float | None = None
        self._stats_level: int = int(extra.get("icms_stats_level", 1))
        self._log_selections: bool = bool(extra.get("icms_log_selections", False))
        self._score_stride: int = int(extra.get("icms_score_stride", 6))
        # FAPS (FetchAll-post-Score): bench mode that combines Score's
        # K-page selection signal with a follow-up FetchAll for the full
        # N pages so decode runs dense. Read from extra_config (rather
        # than env var) so spawn-method workers see it deterministically;
        # the legacy ICMS_FETCH_ALL_POST_SCORE env var path is still
        # checked as a fallback for shell-based benches.
        self._fetch_all_post_score: bool = bool(
            extra.get("fetch_all_post_score", False))
        # Sparse-prefill + dense-decode: Score's K-page picks ARE applied
        # during prefill (unlike FAPS, which discards them), then a single
        # FetchAll fires at the prefill→decode transition so decode attends
        # over all N pages. Mutually exclusive with fetch_all_post_score.
        self._sparse_prefill_dense_decode: bool = bool(
            extra.get("sparse_prefill_dense_decode", False))
        # B1 (2026-05-05): sink-size multiplier for concurrent in-flight
        # Score / FetchAll RPCs. Today the server allocates sink offsets
        # internally, so the connector's only job is to register a sink
        # large enough for `icms_sink_slots` concurrent dumps. Default is
        # 1 (preserves pre-batching behavior). Under ICMS_ALLOW_BATCH=1
        # the connector defaults to scheduler_config.max_num_seqs so
        # batched fetches don't collide on offsets. Explicit override
        # via extra_config["icms_sink_slots"] takes precedence.
        _scheduler_cfg = getattr(vllm_config, "scheduler_config", None)
        _max_num_seqs = (int(getattr(_scheduler_cfg, "max_num_seqs", 1) or 1)
                         if _scheduler_cfg is not None else 1)
        # Stash for downstream gates (e.g., write-pipeline drain default
        # in wait_for_save — race-audit follow-up 2026-05-08).
        self._max_num_seqs: int = _max_num_seqs
        # 2026-05-09 (Item C audit): widened to also default to
        # max(1, max_num_seqs) when max_num_seqs > 1, even if
        # ICMS_ALLOW_BATCH=1 is unset. Otherwise the sink registers
        # for 1 slot and multi-rid concurrent Score/FetchAll RPCs
        # collide on offsets. Mirrors _Worker._is_multi_rid_mode (kept
        # as a separate inline check here because this site runs in
        # IcmsConnector.__init__ before _Worker exists).
        _is_multi_rid_init = _allow_batch() or _max_num_seqs > 1
        _default_sink_slots = (max(1, _max_num_seqs)
                               if _is_multi_rid_init else 1)
        self._sink_slots: int = max(1, int(
            extra.get("icms_sink_slots", _default_sink_slots)))
        self._adaptive_bandwidth: bool = bool(extra.get("adaptive_bandwidth", False))
        self._link_bandwidth_bps: float = float(extra.get(
            "link_bandwidth_bps", 25e9 / 8))  # default 25 Gbps (BF2)
        self._compute_slack_table: str = extra.get("compute_slack_table", "")

        # RDMA transport (replaces Unix socket + POSIX shmem).
        self._use_rdma: bool = bool(extra.get("icms_rdma", False))
        # FetchAll RPC support flag. Wired across all three client types
        # as of 2026-05-11:
        #   - RdmaIcmsClient.fetch_all (rdma_client.py:311) — RDMA path
        #   - IcmsClient.fetch_all (client.py) — unix-socket path
        #   - ShmemIcmsClient inherits IcmsClient.fetch_all
        # The C++ server's handle_fetch_all (handlers.cc:2306) is
        # sink-agnostic — it dispatches to a worker that handles local
        # shmem, CUDA-IPC, and RDMA sinks at job time. So
        # `_supports_fetch_all` is now True for every transport the
        # connector can be configured with. Kept as a named flag (vs.
        # inlined `True`) so future transports added without fetch_all
        # support can opt out via a one-line change.
        self._supports_fetch_all: bool = True
        self._rdma_server_host: str = extra.get("icms_server_host", "sprc01")
        self._rdma_port: int = int(extra.get("icms_rdma_port", 18515))
        self._rdma_ib_dev: str = extra.get("icms_ib_dev", "mlx5_0")
        # In-process POSIX shmem transport (alt to AF_UNIX). When set,
        # the connector instantiates ShmemIcmsClient instead of IcmsClient
        # and ignores _socket_path. Mutually exclusive with --rdma — the
        # accuracy bench plumbs this via --inprocess-icms.
        self._shmem_name: str = extra.get("icms_shmem_name", "")
        # GPUDirect RDMA: server writes KV directly into GPU HBM.
        self._gpu_direct: bool = bool(extra.get("icms_gpu_direct", False))
        self._gpu_device: str = extra.get("icms_gpu_device", "cuda:0")
        # CUDA-IPC sink for the local mem-backend (Phase 1 of
        # docs/cuda_ipc_local_sink_plan_2026-05-05.md). Mutually exclusive
        # with --rdma. Set by the bench via --ipc-gpu-sink.
        self._local_gpu_direct: bool = bool(
            extra.get("icms_local_gpu_direct", False)) and not self._use_rdma

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
                fetch_all_post_score=self._fetch_all_post_score,
                sparse_prefill_dense_decode=
                    self._sparse_prefill_dense_decode,
                shmem_name=self._shmem_name,
                sink_slots=self._sink_slots,
                local_gpu_direct=self._local_gpu_direct,
            )
            # Plumb max_num_seqs through to the worker for the
            # write-pipeline drain default in wait_for_pending_writes
            # (line ~6485). Pre-2026-05-08 evening this attribute lived
            # only on the outer connector and `_Worker` raised
            # AttributeError under sync scheduling. The race-audit
            # follow-up that introduced the gate didn't propagate the
            # value — fixed here.
            self._worker._max_num_seqs = self._max_num_seqs

    # ══════════════════════════════════════════════════════════════════════
    #  Worker-side abstract methods
    # ══════════════════════════════════════════════════════════════════════

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Stash GPU KV cache tensors for Path B selective fetch."""
        if self._worker is not None:
            self._worker.register_kv_caches(kv_caches)

    def set_input_batch_req_ids(self, req_ids: list[str]) -> None:
        """Receive the authoritative input_batch.req_ids ordering from vLLM.

        2026-05-12 multi-rid row-mapping fix: this is the source of truth
        for FA's per-rid bt / seq_lens / query_start_loc row order. The
        scheduler-side `connector_meta.requests` order (new+cached) does
        NOT match this when multi-rid batched mode is active. Routed to
        the worker; the connector facade only delegates.
        """
        if self._worker is not None:
            self._worker.set_input_batch_req_ids(req_ids)

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
        # 2026-05-16 diag (gated): log each unique layer_name vLLM passes
        # once. Used to confirm whether the connector's hardcoded
        # `model.layers.N.self_attn.attn` write-key matches the actual
        # read-key (e.g. for Mistral3ForConditionalGeneration the LM is
        # wrapped at prefix `language_model`, so reads may arrive as
        # `language_model.model.layers.N.self_attn.attn` and the
        # _pending_scores.pop in wait_for_layer would silently miss).
        if os.environ.get("ICMS_DIAG_LAYERNAME") == "1":
            seen = getattr(self, "_diag_layername_seen", None)
            if seen is None:
                seen = set()
                self._diag_layername_seen = seen
            if layer_name not in seen:
                seen.add(layer_name)
                logger.warning(
                    "[diag-layername] wait_for_layer_load layer_name=%r",
                    layer_name)
        self._worker.wait_for_layer(layer_name)

    def is_dense_for_active_request(self) -> bool:
        """M4: True if every currently-active request has flipped to
        dense_mode (bitmap saturated → no more Score RPCs to issue).

        Quest hooks call this at the top of their per-layer callback
        to short-circuit the Q-compute path entirely. Returns False
        when there are no active requests so we don't accidentally
        suppress the very first Score of a new request.

        O(1) — reads the worker's memoized _cached_all_dense flag,
        which is invalidated/recomputed at the three state-change
        sites (on_step_start with new chains, the two flip sites,
        on_request_finished). Pre-cache this iterated _requests every
        call (×48 layers × N decode iters ≈ 2 ms/iter at high ctx)."""
        if self._worker is None:
            return False
        return bool(getattr(self._worker, "_cached_all_dense", False))

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
        if self._budget_override is not None:
            budget = self._budget_override
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
            # only needs to drive scoring here. M3: in decode, the
            # connector continues to issue Score calls (with the
            # decode-mode bitmap filter) until each stride group's
            # bitmap is full; once full, _score_one_request short-
            # circuits and decode runs sparse against the in-cache set.
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
            # KV provenance capture (env-gated; cheap no-op when disabled).
            # We snapshot per-rid: ext_comp count (the ICMS-elided range),
            # local_cached count (vLLM prefix-cache range), and the
            # allocated block_id list. Plumbed into metadata in build_meta
            # so the worker can derive which blocks ICMS is responsible
            # for populating.
            if icms_provenance.is_enabled():
                try:
                    _n_ext = int(num_external_tokens)
                    _n_total = int(getattr(request, "num_computed_tokens", 0))
                    _n_local = max(0, _n_total - _n_ext)
                    _bids: list[int] = []
                    try:
                        _groups = blocks.get_block_ids(allow_none=True)
                        if _groups:
                            _bids = list(_groups[0])
                    except Exception:
                        _bids = []
                    self._sched._prov_alloc[request.request_id] = (
                        _n_ext, _n_local, _bids)
                except Exception as _e:
                    logger.debug(
                        "icms_provenance capture failed for rid=%s: %s",
                        request.request_id, _e)
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
        #
        # Fix I (2026-04-29): order the worker call FIRST. The worker's
        # `on_request_finished` drains the deferred-extract pipeline
        # (which finishes flushing groups) and then pushes the chain to
        # `_stored_chain_queue`. At TP=1 (same process), the scheduler's
        # subsequent `on_finished_record_stored` drains that queue into
        # `_stored_chains` so the next request's
        # `get_num_new_matched_tokens` sees `matched_groups > 0` instead
        # of falling back to a full cold prefill on the first warm.
        # Without this swap, first-warm-after-cold at large ctx took 96 s
        # (full prefill of 32k tokens) instead of <300 ms (skip prefill).
        if self._worker is not None:
            self._worker.on_request_finished(request.request_id)
        if self._sched is not None:
            self._sched.on_finished_record_stored(request)
            self._sched.on_finished(request.request_id)
            # Drop any undelivered prov alloc snapshot.
            self._sched._prov_alloc.pop(request.request_id, None)
        # KV-provenance: drop tracker state for this rid (no-op if env off).
        icms_provenance.tracker().clear_request(request.request_id)
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

        # KV-block provenance capture (env-gated). Set by
        # update_state_after_alloc; drained into IcmsConnectorMetadata
        # in build_meta. Value: (ext_comp_tokens, local_cached_tokens,
        # block_ids).
        self._prov_alloc: dict[str, tuple[int, int, list[int]]] = {}

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
            #
            # Bug 11 fix (2026-04-29): floor division (was ceil). The
            # partial last group can't be reliably stored on the server
            # (worker's Fix D guards against partial-group writes), so
            # including its hash in the chain made the server resolve N+1
            # groups while the client only counted N — leaving N..N+1
            # group's pages as "out-of-range from client's view, never
            # in bitmap, perpetually returned by Score top-k". Use floor
            # so client and server agree on group count.
            num_groups = len(token_ids) // group_tokens
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
        if _ICMS_FULLTRACE_ENABLED:
            try:
                import hashlib as _hl_oa
                _tok_bytes = repr(tuple(token_ids or [])).encode()
                _icms_fulltrace(
                    "on_alloc", rid=rid,
                    prompt_len=int(len(token_ids or [])),
                    chain_len=int(len(chain)),
                    chain_head=[int(x) for x in chain[:4]],
                    chain_tail=[int(x) for x in chain[-4:]],
                    chain_fp=_icms_chain_fp(chain),
                    token_sha=_hl_oa.sha1(_tok_bytes).hexdigest()[:16],
                    token_first8=list(int(t) for t in (token_ids or [])[:8]),
                    token_last8=list(int(t) for t in (token_ids or [])[-8:]),
                )
            except Exception:
                pass
        self._chains[rid] = chain
        self._pending_chain_sends.add(rid)

    def build_meta(self, scheduler_output: SchedulerOutput) -> IcmsConnectorMetadata:
        # Also drain worker notifications here (belt-and-suspenders with
        # the drain in get_num_new_matched_tokens) to catch pushes that
        # arrived between scheduling and metadata building.
        # Bug #5 fix (race-audit 2026-05-08): use the atomic helper to
        # avoid the iter+clear race (a producer append between the
        # for-loop's end and the .clear() was silently dropped pre-fix).
        for chain, n_groups in _drain_stored_chain_queue():
            self.record_stored_chain(chain, n_groups)

        meta = IcmsConnectorMetadata()
        # Per-step scheduled token counts (req_id → num_tokens this step).
        # 2026-05-07 BUG FIX: vLLM V2's canonical field is
        # `num_scheduled_tokens` (see vllm/v1/core/sched/output.py:194-196),
        # NOT `scheduled_num_tokens`. The typo silently returned `{}`
        # via getattr's default → every n_step was 0 → meta_tokens=0
        # for every rid → per-rid Q-slicing in _on_layer_score_impl
        # produced empty `quest_query[0:0]` slices for every rid →
        # Score saw no Q → garbage page selection → batched-mode
        # accuracy collapse at higher k. The cascading effect is the
        # SAME bug as the comment block at :1349-1358 attributed to
        # "first scheduler tick race" — the comment was wrong; it's
        # not a race, it's a 100% miss because the field name is
        # wrong. Fall back to the typo'd name as a defensive measure
        # in case some older vLLM build does use it.
        sched_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
        if sched_tokens is None:
            sched_tokens = getattr(
                scheduler_output, "scheduled_num_tokens", {})

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
            if stored_groups > 0:
                # Ferry the authoritative scheduler-side value to the
                # worker so it doesn't consult its racy local cache.
                meta.stored_groups_by_rid[prs.request_id] = stored_groups
            if stored_groups >= complete_groups_after:
                meta.skip_extract_rids.add(prs.request_id)

        # Drain KV-provenance alloc snapshots into metadata for the
        # worker. Only emits when the env flag is on; cheap no-op
        # otherwise. We drain on every build_meta so per-rid records
        # land on the worker side ASAP after alloc.
        if icms_provenance.is_enabled() and self._prov_alloc:
            for rid, (n_ext, n_local, bids) in list(self._prov_alloc.items()):
                meta.prov_ext_comp_tokens[rid] = n_ext
                meta.prov_local_cached_tokens[rid] = n_local
                meta.prov_block_ids[rid] = list(bids)
                # Pop after delivery — a re-alloc for the same rid
                # (preemption) will re-stamp.
                self._prov_alloc.pop(rid, None)
        return meta

    # ── Global prefix index operations ──────────────────────────────

    def record_stored_chain(self, chain: list[int], num_groups: int):
        """Called when the worker reports that groups were written to ICMS."""
        key = tuple(chain[:num_groups])
        if _ICMS_FULLTRACE_ENABLED:
            try:
                _icms_fulltrace(
                    "record_stored_chain", rid="",
                    n_groups=int(num_groups),
                    chain_len=int(len(chain)),
                    chain_fp=_icms_chain_fp(list(chain)),
                    chain_head=[int(x) for x in chain[:4]],
                    stored_chains_count_pre=int(len(self._stored_chains)),
                )
            except Exception:
                pass
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
        if _ICMS_FULLTRACE_ENABLED:
            _icms_fulltrace(
                "gnnmt_entry", rid=request.request_id,
                num_computed_tokens=int(num_computed_tokens),
                force_cold=(os.environ.get("ICMS_FORCE_COLD") == "1"),
                opt1_env=os.environ.get(
                    "ICMS_OPTION_1_COLD_FALLBACK", "0"),
            )
        # ICMS_FORCE_COLD=1: pretend nothing is externally cached. vLLM
        # then takes the cold prefill path even on run 2 (no apply, no
        # FetchAll). Sanity check: if run 2 PASSES with this flag, the
        # bug is purely in the warm-path code (apply / matched-tokens
        # contract). If it FAILS, BF2 server state from run 1 is
        # poisoning run 2 (e.g., NVMe namespace pollution).
        if os.environ.get("ICMS_FORCE_COLD") == "1":
            if _ICMS_FULLTRACE_ENABLED:
                _icms_fulltrace(
                    "gnnmt_exit_force_cold", rid=request.request_id,
                    returned=(0, False))
            return 0, False
        # 2026-05-12 CROSS-RID BLOCK ALIASING FALLBACK (Option 1, opt-in):
        # In multi-rid mode, reporting num_external_tokens > 0 causes vLLM's
        # BlockManager to ALIAS the matched-prefix physical blocks across
        # all rids sharing the prefix. ICMS apply's scatter to those shared
        # blocks → cross-rid contamination.
        # Workaround: return 0 in multi-rid → cold prefill per rid → unique
        # blocks. Trade-off: Phase 2 re-prefills the haystack per rid.
        # Opt-in via ICMS_OPTION_1_COLD_FALLBACK=1. The primary fix is
        # Option 4 (ICMS-owned scratch tensor); see register_kv_caches.
        if os.environ.get("ICMS_OPTION_1_COLD_FALLBACK", "0") == "1":
            _scheduler_cfg = getattr(self._vllm_config,
                                     "scheduler_config", None)
            _max_num_seqs = int(getattr(_scheduler_cfg,
                                        "max_num_seqs", 1) or 1)
            _ab = _allow_batch()
            if _max_num_seqs > 1 or _ab:
                logger.info(
                    "[icms-opt1] FIRED: returning (0, False) for rid=%s "
                    "(max_num_seqs=%d allow_batch=%s)",
                    request.request_id, _max_num_seqs, _ab)
                return 0, False
            else:
                logger.info(
                    "[icms-opt1] env set but gate FALSE: "
                    "max_num_seqs=%d allow_batch=%s — letting fall through",
                    _max_num_seqs, _ab)
        # Drain worker → scheduler notifications first.
        # Bug #5 fix (race-audit 2026-05-08): atomic snapshot+clear under
        # `_stored_chain_cond` so a producer append landing during the
        # drain isn't wiped by .clear().
        for chain, n_groups in _drain_stored_chain_queue():
            self.record_stored_chain(chain, n_groups)

        rid = request.request_id
        # Always read token_ids — the cached-rid branch below uses len(token_ids)
        # at line 1495 to clamp ext_tokens. Hoisting outside the if-block fixes
        # an UnboundLocalError that fired on the second call for the same rid
        # (observed at h=128k with prefix-cache hits).
        token_ids = getattr(request, "all_token_ids", None)
        if token_ids is None or len(token_ids) == 0:
            token_ids = getattr(request, "prompt_token_ids", [])
        # Ensure chain is computed.
        if rid not in self._chains:
            # Compute chain eagerly (before on_alloc which needs blocks).
            import hashlib
            group_tokens = _GROUP_BLOCKS * PAGE_TOKENS
            chain: list[int] = []
            if token_ids and len(token_ids) >= PAGE_TOKENS:
                # Mirrors the chain-construction in on_alloc — full prompt
                # coverage with incremental hashing. Both call sites must
                # produce identical hashes to match across requests.
                # Bug 11 fix (2026-04-29): floor (not ceil) so the partial
                # last group is excluded — see on_alloc for rationale.
                num_groups = len(token_ids) // group_tokens
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
        # 2026-05-09 cross-iter stored-chain race fix (v2: Condition +
        # generation counter; v1's Event.set/clear pulse lost wakeups
        # because producers fired ~8s before consumers waited).
        #
        # In multi-rid batched mode (max_num_seqs>=2) under per-batch/
        # per-budget, phase 2 (warm) can start before phase 1 (cold)
        # has fully drained ALL its rids' write pipelines. The second
        # cold rid's WriteGroup may still be in flight when this
        # lookup runs → matched_groups=0 → _score_one_request's
        # `total_pages==0` early-return fires later → silent dense
        # fallback. Empirically (2026-05-09):
        #   CONTROL (BLOCK_WRITES=0, max_num_seqs=2): 0.080
        #   SEQ1    (BLOCK_WRITES=0, max_num_seqs=1): 0.760
        #   FIX     (BLOCK_WRITES=1, max_num_seqs=2): 0.580
        #
        # Mitigation: when the chain is non-empty (worth waiting for)
        # and lookup missed, capture the producer generation, drain
        # the queue, re-lookup. If still missed, Condition.wait until
        # the generation advances (or timeout). Loop with bounded
        # retries × small timeout — worst case ~600ms per warm rid.
        if matched_groups == 0 and len(chain) >= 4:
            try:
                _wait_count = int(
                    os.environ.get("ICMS_STORED_CHAIN_WAIT_COUNT", "3"))
                _wait_timeout_s = float(
                    os.environ.get("ICMS_STORED_CHAIN_WAIT_MS", "200")) / 1000.0
            except ValueError:
                _wait_count, _wait_timeout_s = 3, 0.200
            for _ in range(_wait_count):
                # Drain whatever's already queued (may include appends
                # that landed BEFORE we entered this branch — those
                # would otherwise be lost-wakeups).
                # Bug #5 fix (race-audit 2026-05-08): atomic snapshot+clear.
                for c, n in _drain_stored_chain_queue():
                    self.record_stored_chain(c, n)
                matched_groups = self._lookup_stored_prefix(chain)
                if matched_groups > 0:
                    break
                # Snapshot the generation then wait for it to advance.
                with _stored_chain_cond:
                    gen_at_wait = _stored_chain_generation
                    _stored_chain_cond.wait_for(
                        lambda: _stored_chain_generation > gen_at_wait,
                        timeout=_wait_timeout_s)
        logger.debug(
            "prefix_lookup: rid=%s chain_len=%d stored=%d matched=%d",
            rid, len(chain), len(self._stored_chains), matched_groups,
        )
        # Bug 10 diag (ICMS_DIAG_NMT=1): log get_num_new_matched_tokens
        # decision per request. If we see matched=0 for a warm-prefix
        # request when the previous cold run STORED that chain, the
        # _stored_chains index hasn't been populated yet — vLLM falls
        # back to a full prefill, polluting first-warm timings.
        if os.environ.get("ICMS_DIAG_NMT") == "1":
            logger.info(
                "[icms-nmt] rid=%s chain_len=%d stored_chains=%d "
                "matched_groups=%d num_computed=%d",
                rid, len(chain), len(self._stored_chains),
                matched_groups, num_computed_tokens)
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
        if _ICMS_FULLTRACE_ENABLED:
            _icms_fulltrace(
                "get_num_new_matched_tokens", rid=rid,
                chain_len=int(len(chain)),
                stored_chains_count=int(len(self._stored_chains)),
                matched_groups=int(matched_groups),
                matched_tokens=int(matched_tokens),
                num_computed_tokens=int(num_computed_tokens),
                ext_tokens=int(ext_tokens),
                returned=(int(ext_tokens), False),
                prompt_len=int(total_prompt),
            )
        # Synchronous loading (False): the worker fills blocks during
        # the forward pass via wait_for_layer_load.
        return ext_tokens, False

    def on_finished_record_stored(self, request: Request) -> None:
        """Record this request's chain into the scheduler-side prefix
        index so subsequent iter-2 lookups via get_num_new_matched_tokens
        find it.

        BOUNDED by the worker's actual flushed count to prevent
        overstating. Previous behavior — record `len(token_ids) //
        group_tokens` regardless of what the worker actually wrote —
        caused Score RPCs to ENOENT when WriteGroup partially failed
        (allocator OOM, transient errors): the scheduler claimed N
        groups stored, vLLM elided that range, but the trie only had
        K < N groups → Score for the elided prefix returned
        "no resolvable groups". Cap on the worker's reported count
        keeps the ledger honest with what the trie actually has.

        We drain the worker queue first, then look up the matching
        chain in self._stored_chains. If found, the entry was written
        by the worker via _stored_chain_queue with `flushed_local`
        (which only counts successful WriteGroup calls). If not found,
        nothing was successfully flushed for this chain — record
        nothing, since claiming any count would overstate.

        Idempotent: record_stored_chain merges duplicates by chain key.
        """
        # Drain worker → scheduler notifications so we can read the
        # latest flushed count for this rid's chain.
        # Bug #5 fix (race-audit 2026-05-08): atomic snapshot+clear.
        for chain_q, n_groups_q in _drain_stored_chain_queue():
            self.record_stored_chain(chain_q, n_groups_q)

        rid = request.request_id
        chain = self._chains.get(rid)
        if not chain:
            return

        # Look up the worker-reported flushed count for this chain. The
        # ledger is keyed on `tuple(chain[:n_groups])` — search for any
        # entry whose key is a prefix of this request's chain.
        chain_t = tuple(chain)
        worker_n_groups = 0
        for stored_chain, n_groups in self._stored_chains:
            if len(stored_chain) <= len(chain_t) and \
                    chain_t[:len(stored_chain)] == stored_chain:
                worker_n_groups = max(worker_n_groups, n_groups)

        if worker_n_groups <= 0:
            # No successful flush for this chain — don't record. Better
            # to under-promise (no scheduler-side prefix elision for
            # this chain) than to over-promise and trigger ENOENT.
            logger.debug(
                "on_finished_record_stored: rid=%s chain_len=%d "
                "worker_n_groups=0 — NOT recording (no flushed groups)",
                rid, len(chain))
            return

        # The worker entry already exists from the queue drain above;
        # this call is just defensive (idempotent merge). The bound is
        # the worker's flushed_local — never the prompt-length estimate.
        self.record_stored_chain(chain, worker_n_groups)
        logger.debug(
            "on_finished_record_stored: rid=%s chain_len=%d "
            "worker_n_groups=%d (bounded by flushed_local)",
            rid, len(chain), worker_n_groups)

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
                 gpu_device: str = "cuda:0",
                 fetch_all_post_score: bool = False,
                 sparse_prefill_dense_decode: bool = False,
                 shmem_name: str = "",
                 sink_slots: int = 1,
                 local_gpu_direct: bool = False):
        self._socket_path = socket_path
        self._use_rdma = use_rdma
        self._rdma_server_host = rdma_server_host
        self._rdma_port = rdma_port
        self._rdma_ib_dev = rdma_ib_dev
        self._gpu_direct = gpu_direct
        # CUDA-IPC sink for the local mem-backend (Phase 1 of
        # docs/cuda_ipc_local_sink_plan_2026-05-05.md). Mutually exclusive
        # with --rdma; only takes effect when the connector talks to a
        # spawned mem-backend icms_server. When True, the connector
        # implicitly enables gpu_direct semantics on the apply path.
        self._local_gpu_direct = bool(local_gpu_direct) and not use_rdma
        if self._local_gpu_direct:
            self._gpu_direct = True
        self._gpu_device = gpu_device
        self._shmem_name = shmem_name  # non-empty → use ShmemIcmsClient
        self._model_name = model_name
        self._k = k
        # B1 (2026-05-05): plumbed from IcmsConnector.__init__ via
        # extra_config["icms_sink_slots"]. _connect() multiplies the
        # registered sink size by this so the server's offset allocator
        # can dispatch N concurrent Score/FetchAll RPCs without
        # collisions. Default 1 = legacy single-rid sink size.
        self._sink_slots: int = max(1, int(sink_slots))
        # FAPS gate. Take the connector-init value (kv_connector_extra_config),
        # but allow ICMS_FETCH_ALL_POST_SCORE=1 in env to opt in too — keeps
        # the legacy shell-env benches working without code changes.
        self._fetch_all_post_score: bool = (
            bool(fetch_all_post_score)
            or os.environ.get("ICMS_FETCH_ALL_POST_SCORE") == "1")
        # Sparse-prefill + dense-decode variant. Unlike FAPS, Score's
        # K-page picks ARE applied during prefill; FetchAll fires once
        # at the prefill→decode transition so decode goes dense.
        # Mutually exclusive with fetch_all_post_score above.
        self._sparse_prefill_dense_decode: bool = (
            bool(sparse_prefill_dense_decode)
            or os.environ.get("ICMS_SPARSE_PREFILL_DENSE_DECODE") == "1")
        if (self._fetch_all_post_score
                and self._sparse_prefill_dense_decode):
            raise ValueError(
                "fetch_all_post_score and sparse_prefill_dense_decode "
                "are mutually exclusive; pass at most one.")
        logger.info(
            "[icms-init] _Worker init: fetch_all_post_score=%s "
            "sparse_prefill_dense_decode=%s "
            "(extra_config=%s/%s, env=%r/%r)",
            self._fetch_all_post_score,
            self._sparse_prefill_dense_decode,
            fetch_all_post_score, sparse_prefill_dense_decode,
            os.environ.get("ICMS_FETCH_ALL_POST_SCORE"),
            os.environ.get("ICMS_SPARSE_PREFILL_DENSE_DECODE"))
        # Bug 11 family verification (2026-04-30): bf16 byte round-trip
        # self-test. record_page writes raw bytes via view(uint8); apply
        # reads back via view(model_dtype). If pytorch's reinterpret
        # contract changes (or numpy strips/reorders bytes), this catches
        # it at boot before any actual K/V flows. ~10µs cost, runs once.
        try:
            _x = torch.randn(8, 4, 128).to(torch.bfloat16)
            _b = _x.contiguous().view(torch.uint8).numpy().tobytes()
            _y = torch.frombuffer(bytearray(_b),
                                   dtype=torch.bfloat16).reshape(_x.shape)
            assert torch.equal(_x, _y), "bf16 byte round-trip mismatch"
            logger.info("[icms-init] bf16 byte round-trip self-test PASS")
        except Exception as _e:
            logger.error("[icms-init] bf16 byte round-trip self-test "
                          "FAIL: %s", _e)
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

        # ICMS_QUEST_MODE=per_kv_head: lazily-allocated summaries sink
        # (separate from the main KV sink, used only by the v8 per-head
        # opcodes). Untouched in any other mode.
        self._per_head_summary_sink: "Sink | None" = None
        self._per_head_summary_sink_capacity: int = 0
        # Once-per-process warning flags for the per-head path. Init
        # explicitly (rather than letting `getattr(..., False)` create
        # them lazily) so static-analysis / dir(self) introspection
        # sees the full attribute surface — and so a typo elsewhere in
        # the path can't accidentally create a NEW lazy attribute that
        # silently never warns.
        self._quest_per_head_tp_warned: bool = False
        self._quest_per_head_client_warned: bool = False
        # Once-per-process info log on first per-head Score call so a
        # smoke run can confirm the path actually fired (vs silently
        # falling through to the legacy gate's `return`).
        self._quest_per_head_first_call_logged: bool = False

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

        # 2026-05-09 N2 deferral: pipeline-thread `_flush_group` enqueues
        # WriteGroup ok-bits here instead of inline-broadcasting them on
        # the TP NCCL group. The forward thread drains this queue at the
        # top of `wait_for_pending_writes` (BEFORE the memcpy gate) and
        # runs `_tp_broadcast_bool` there, then applies the symmetric
        # ledger bumps (num_groups_written, flushed_local, flush_cond,
        # _record_stored_groups). Moving the broadcast off the pipeline
        # thread closes the TP>1 NCCL collision (pipeline-thread NCCL
        # racing iter N+1's forward-thread NCCL on the same comm). At
        # TP=1 the broadcast is a no-op so the only behavioral change
        # is a one-iter delay in the bumps — recoverable per the audit.
        # Each entry: (rid, group_idx, ok_local, partial, pages,
        #              chain_prefix, _GroupBuffer-not-needed)
        self._pending_flush_q: "list[tuple]" = []
        self._pending_flush_lock: threading.Lock = threading.Lock()

        # 2026-05-11 Option-1 (batched WriteGroup): when
        # `ICMS_WRITE_BATCH_N=N>1`, the connector buffers up to N
        # consecutive full-group flushes per rid and sends them as a
        # SINGLE WriteGroup RPC carrying K=N tail groups. The server's
        # protocol already supports K>1 (handlers.cc handle_write_group
        # reads num_tail + K×(summary+kv) blobs; client.py:418-420
        # documents the K>1 path). Pre-fix: each haystack group was a
        # separate RPC; at gemma-3 h32k that meant ~3875 RPCs and ~100 s
        # Phase 1 wall time. Post-fix at N=32 a typical Phase 1 sends
        # ~120 RPCs — server work + small per-RPC overhead amortizes.
        # Default N=1 preserves the pre-fix per-group RPC behavior so
        # in-flight cells aren't affected. Per-rid buffer keyed by rid;
        # entries are dicts holding all state needed to reconstruct the
        # post-RPC bookkeeping in `_record_flush_outcome`.
        self._write_batch_buf: "dict[str, list[dict]]" = {}
        # 2026-05-11 write-batching-audit Finding 2: guards
        # `_write_batch_buf` from concurrent mutation. Pipeline thread
        # appends via `_flush_group` (`setdefault(...).append(...)` +
        # `len(buf) >= batch_n` check) while forward thread drains via
        # `_flush_write_batch_now` (`pop(rid, [])`) — without the lock
        # at ICMS_WRITE_BATCH_N>1 we get torn dict/list state. Acquire
        # is short (no I/O / no RPC under the lock).
        self._write_batch_buf_lock = threading.Lock()

        # Memoized "all active rs are in dense_mode" — checked O(1) by the
        # per-layer hooks (wait_for_layer, Quest hook's
        # is_dense_for_active_request) instead of iterating self._requests
        # each call. Iterating per layer (×48) per iter (×N decode tokens)
        # measured at ~2 ms/iter on Qwen3-30B-A3B at 131k ctx — that was
        # the residual gap between sparse decode and pure dense baseline.
        # Cache invalidated at three sites: on_step_start (new request),
        # the two flip sites (line ~2664, ~3046), and on_request_finished.
        self._cached_all_dense: bool = False

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
        # 2026-05-08 race-audit X1 wrap-or-nullcontext gate.
        #
        # History: shipped a real threading.Lock first to fix the
        # native-client tx_/rx_ buffer race (icms_client.h:8-10
        # contract violation). Empirically REGRESSED accuracy on
        # llama3 vt 32K batched b=0.05: control 0.260 → postfix 0.160
        # at n=30. Background investigation 2026-05-08 found the
        # mechanism: Score holds the lock for the full client RTT,
        # blocking the deferred-write pipeline thread → chain-state
        # lag GROWS, not shrinks → Score reads from a chain prefix
        # the server hasn't fully committed → silent wrong-pages.
        #
        # The actual fix for the dominant ordering race lives elsewhere
        # (BLOCK_WRITES default extended to batched mode; per-rid Event
        # Step 2 queued). The buffer-corruption race the lock was meant
        # to fix is real but not dominant; the C++ IcmsClient::CallGuard
        # (icms_client.h) is the cheap defensive net — it throws on
        # concurrent entry with no perf cost.
        #
        # Set ICMS_RPC_MUTEX=1 to re-enable the Python wrap for A/B
        # testing; default is nullcontext (no-op).
        import contextlib as _ctxlib
        if os.environ.get("ICMS_RPC_MUTEX", "0") == "1":
            self._rpc_lock = threading.Lock()
        else:
            self._rpc_lock = _ctxlib.nullcontext()

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
        # Legacy (single-rid path): a global bool — extract is skipped
        # iff EVERY active rid in the step is skip-extract. Conservative
        # but correct.
        # Multi-rid path (ICMS_ALLOW_BATCH=1): a per-rid set replaces
        # the global gate inside extract_and_record. The bool is still
        # populated for the legacy single-rid path.
        self._skip_extract: bool = False
        self._skip_extract_rids: set[str] = set()
        # Stash of the most recent IcmsConnectorMetadata.requests so the
        # multi-rid extract path can map batch row index → rid without
        # threading an extra arg through extract_and_record.
        self._last_step_requests: list = []
        # 2026-05-12 multi-rid row-mapping fix: vLLM's input_batch.req_ids
        # is the authoritative source for FA's per-rid bt/qsl/seq_lens
        # row order. `_last_step_requests` order (= scheduled_new_reqs +
        # scheduled_cached_reqs from build_meta) does NOT match it under
        # multi-rid batched mode. We receive the live ordering via
        # set_input_batch_req_ids() before each forward and use it for
        # rid → row lookups. None means single-rid path / pre-fix smoke.
        self._input_batch_req_ids: list[str] | None = None
        self._rid_to_bt_row: dict[str, int] = {}


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

        # 2026-05-05 async-Score dispatch pool (option 2 of the Score
        # overlap analysis). At TP=1 + ICMS_ALLOW_BATCH=1, the per-layer
        # score-dispatch loop fires N rids' _score_one_request calls
        # concurrently on this pool so the async-capable IcmsClient
        # actually overlaps the underlying RPCs. Sized for the worst
        # observed batch (max_num_seqs ≤ 8 today). At TP>1 the path is
        # never taken (NCCL collective ordering would deadlock).
        import concurrent.futures as _cf
        self._score_dispatch_pool = _cf.ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="icms-score")

        # TTFT-breakdown instrumentation. Per-request phase timestamps +
        # per-hook accumulators collected across the prefill critical path
        # (on_step_start → wait_for_layer × 48 → wait_for_pending_writes).
        # Enabled when ICMS_TTFT_BREAKDOWN=1.
        self._ttft_enabled = os.environ.get("ICMS_TTFT_BREAKDOWN", "1") != "0"
        self._ttft: dict[str, dict] = {}

        # Cache apply-path env vars at init (was per-layer
        # os.environ.get → ~us each × 48 layers × every iter). Static for
        # the lifetime of the connector.
        self._cfg_per_rank_slice = (
            os.environ.get("ICMS_PER_RANK_SLICE", "0") == "1")
        # 2026-05-07 PERF: cache _extract_layer_idx results.
        # Called ~6 sites × 32-40 layers per forward = ~200+ calls;
        # each does a string split + linear scan. Layer names are
        # interned by vLLM's layer map and stable for the connector's
        # lifetime → safe to cache by string key. Few hundred µs/forward
        # win.
        self._layer_idx_cache: "dict[str, int | None]" = {}
        self._cfg_apply_stream = (
            os.environ.get("ICMS_APPLY_STREAM", "0") == "1")
        self._cfg_skip_bounds = (
            os.environ.get("ICMS_SKIP_BOUNDS", "0") == "1")
        self._cfg_line_timing = (
            os.environ.get("ICMS_LINE_TIMING", "0") == "1")
        self._cfg_apply_timing = (
            os.environ.get("ICMS_APPLY_TIMING", "0") == "1")
        self._cfg_no_seqlen_cache = (
            os.environ.get("ICMS_NO_SEQLEN_CACHE", "0") == "1")
        self._cfg_apply_soft_fail = (
            os.environ.get("ICMS_APPLY_SOFT_FAIL", "0") == "1")
        # Cached sink_pages view, populated lazily on first apply call
        # (depends on geom which is only known post-_connect()).
        self._sink_pages_view: object = None

        self._connect()

    def _is_multi_rid_mode(self) -> bool:
        """Multi-rid (batched) mode is active when EITHER the env knob
        ICMS_ALLOW_BATCH=1 is set OR the scheduler is configured with
        max_num_seqs > 1.

        Audit (Item C, 2026-05-09): historically the connector gated
        every batched-mode code path on `_allow_batch()` alone. But
        `max_num_seqs` (set by vLLM's --max-num-seqs / SchedulerConfig)
        is an independent indicator that the engine *will* schedule
        multiple rids per step. If a launcher sets max_num_seqs=2 but
        forgets ICMS_ALLOW_BATCH=1, the legacy "single-rid only" code
        paths fire at multi-rid time:

          - extract_and_record's plan-build only sees the first rid;
            the second rid's KV is silently dropped.
          - _use_batch_path falls back to a single-rid IcmsFetchState
            whose block_table shape doesn't match the actual N-rid
            forward batch → FA3 batch_size_k mismatch crash.
          - Sink slot count defaults to 1 → multi-rid concurrent
            Score/FetchAll RPCs collide on offsets.

        All correctness-critical batched-mode gates now route through
        this helper. Performance-only gates (e.g. the async Score pool
        at TP=1, log lines) keep the narrower `_allow_batch()` env-only
        check.

        Cheap to call: just an env-var lookup + an int compare.
        Cached `_max_num_seqs` is plumbed by IcmsConnector.__init__
        post-_Worker construction (line ~1100). Tests that build a
        bare _Worker without the outer connector won't have it set;
        defaults to 1 in that case.
        """
        if _allow_batch():
            return True
        try:
            n = int(getattr(self, "_max_num_seqs", 1) or 1)
        except (TypeError, ValueError):
            n = 1
        return n > 1

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
        elif self._shmem_name:
            from icms_client.shmem_client import ShmemIcmsClient
            self._client = ShmemIcmsClient(self._shmem_name)
            transport_desc = f"shmem:/dev/shm/{self._shmem_name}"
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
                with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                    ack = self._client.hello(self._model_name,
                                             tp_rank=self._tp_rank,
                                             tp_size=self._tp_size,
                                             deployment_id=self._deployment_id,
                                             sink_slot_count=self._sink_slots)
                break  # success
            except Exception as _e:  # broad: any connect/hello failure
                if _attempt == _max_attempts - 1:
                    raise
                logger.warning(
                    "[icms-connect] attempt %d/%d failed: %s; trying "
                    "BF2 auto-restart and retrying (tp_rank=%d)",
                    _attempt + 1, _max_attempts, _e, self._tp_rank)
                # ensure_bf2_running runs a kill_local_orphans sweep that
                # SIGTERMs every locally-owned EngineCore/icms_server PID
                # not in the bench's own safe-set. With multiple benches
                # sharing a host (different GPUs), one bench's connect
                # retry would kill the OTHER bench's EngineCore — observed
                # 2026-05-08 with mistral-small + qwen3-128K co-located.
                # Only run the BF2 helper when actually using BF2 (RDMA).
                if self._use_rdma:
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
                elif self._shmem_name:
                    from icms_client.shmem_client import ShmemIcmsClient
                    self._client = ShmemIcmsClient(self._shmem_name)
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

        # 2026-05-08 server-audit P0 startup asserts (D3, E4):
        #
        # D3: TP fan-out per-rank slicing in worker_pool.cc:107 requires
        #   kv_page_bytes % tp_size == 0
        # else the server falls through to a broadcast (full-page write
        # to every rank) and the client only reads kv_page_bytes/tp_size
        # bytes per page → silent data drop on the other ranks. Catches
        # any future model whose head_dim * num_kv_heads_per_rank doesn't
        # divide cleanly. Cheap: fires once at hello() return.
        if self._tp_size > 1:
            kvb = int(self._geom.kv_page_bytes)
            if kvb % self._tp_size != 0:
                raise RuntimeError(
                    f"[icms-startup-assert D3] kv_page_bytes={kvb} "
                    f"not divisible by tp_size={self._tp_size}; the "
                    f"server's per-rank slicing falls through to a "
                    f"broadcast and the client silently drops other "
                    f"ranks' KV bytes. See server audit "
                    f"project_icms_server_audit_2026-05-08.")
        # E4: connector's apply path computes
        #   page_idx_dev = [off // kv_page_bytes for off in valid_sink_offs]
        # which requires sink_offset_base for slot N to be a multiple
        # of kv_page_bytes. Server slabs are sink_size / sink_slot_count;
        # if either divisor doesn't land cleanly, the floor div loses
        # information → wrong sink page index → corrupted apply. Both
        # numbers are computed from per_layer_bytes which we control
        # here, so the assert IS expected to hold; if it ever fails
        # we want a loud crash, not silent corruption.
        per_layer_bytes_check = self._k * self._geom.kv_page_bytes
        sink_layers_check = max(self._score_stride,
                                  int(self._geom.num_layers))
        slab_bytes_check = sink_layers_check * per_layer_bytes_check
        if slab_bytes_check % self._geom.kv_page_bytes != 0:
            raise RuntimeError(
                f"[icms-startup-assert E4] sink slab "
                f"({slab_bytes_check}) not divisible by kv_page_bytes "
                f"({self._geom.kv_page_bytes}); off // kv_page_bytes in "
                f"the apply path will lose precision. See server audit.")

        # Sink sizing: needs to hold one server Phase-2 dump at a time.
        # - Score path (stride-gated): score_stride layers × k pages.
        # - FetchAll path (B, budget=1.0): num_layers × k pages (one call
        #   covers every reuse layer, not just score_stride).
        # Size for the worst case so B doesn't overflow → server's sink-
        # bounds check rejects writes → GPU-direct index_select trips a
        # CUDA OOB assertion.
        per_layer_bytes = self._k * self._geom.kv_page_bytes
        sink_layers = max(self._score_stride, int(self._geom.num_layers))
        # B1: scale the sink to hold `_sink_slots` concurrent dumps.
        # Default _sink_slots=1 preserves legacy behavior; under
        # ICMS_ALLOW_BATCH=1 it defaults to max_num_seqs so the server's
        # offset allocator has room for N parallel Score / FetchAll
        # writes without colliding.
        total_sink = sink_layers * per_layer_bytes * self._sink_slots
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
            with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                sink = self._client.register_gpu_sink(
                    total_sink, gpu_dev, flag_slots=flag_slots)
            sink_desc = f"gpu_direct({gpu_dev})"
        elif self._local_gpu_direct:
            # Phase-1 CUDA-IPC sink for the local mem-backend. No flag
            # sink (Phase 1 reply is synchronous; flag-poll only ships
            # with RDMA where ready-flags ride the same QP). The IPC
            # sink reports is_gpu_direct=True so the apply path's
            # gpu_direct branch fires unchanged.
            sink = self._client.register_cuda_ipc_sink(total_sink, gpu_dev)
            sink_desc = f"cuda_ipc({gpu_dev})"
        else:
            with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                sink = self._client.register_sink(total_sink)
            sink_desc = "host"
        self._sink_pool = _SinkSlotPool(sink, per_layer_bytes,
                                          self._sink_slots)
        self._sink_per_layer_bytes = per_layer_bytes
        if os.environ.get("ICMS_DIAG_TP_SINK", "0") == "1":
            try:
                _peek_init = bytes(sink.mm[0:16]).hex()
                logger.info(
                    "[diag-tp-sink-init] rank=%d pid=%d shm_name=%s "
                    "sink_id=%s kv_bytes=%d init_head16=%s",
                    self._tp_rank, os.getpid(),
                    getattr(sink, "name", "?"),
                    getattr(sink, "sink_id", -1),
                    getattr(sink, "_kv_data_size", -1),
                    _peek_init)
            except Exception as _diag_e:
                logger.warning(
                    "[diag-tp-sink-init] peek failed: %s", _diag_e)
        logger.info(
            "IcmsConnector worker: connected to %s, model=%s, "
            "sink=%s %d layers × %d B/layer × %d slots = %d B total, "
            "score_stride=%d, allow_batch=%s",
            transport_desc, self._model_name, sink_desc,
            sink_layers, per_layer_bytes, self._sink_slots, total_sink,
            self._score_stride, _allow_batch(),
        )

    def _rank_chain(self, chain):
        """Pass-through. Trie chains are rank-agnostic at every TP.

        Historical: at TP>1 we used to prepend a per-rank tag to namespace
        chains in the global trie. That predates the Option-W server-side
        fan-out design (only rank 0 talks to the wire; the server fans
        out sink bytes to every rank's sink). With rank-0-only writes
        there is no cross-rank trie conflict to prevent, and the synthetic
        prefix has no associated KV bytes so it broke the v6 per-tail
        byte accounting in WriteGroup. Removing it also lets a chain
        written at TP=2 be readable at TP=1 (and vice versa). The
        _rank_tag / _rank_tagged_chain helpers remain in this module
        for the moment but are unused — slated for cleanup.
        """
        return list(chain) if chain is not None else chain

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
        """Record that n_groups were written for this chain prefix.

        Contract: `n_groups <= len(chain)`. Each chain element is one
        GroupHash (one per group), so the count cannot exceed the chain
        length. The query side (`_get_stored_context_groups`) silently
        caps at `min(n_groups, match_len)`, so any over-record beyond
        `len(chain)` is invisible to readers — making caller bugs hard
        to diagnose. Assert loudly here so any future contract violation
        surfaces at record time, not buried inside a downstream query
        cap. See project_apply_path_helper_tests_2026-05-09.md.
        """
        assert n_groups <= len(chain), (
            f"_record_stored_groups: n_groups={n_groups} > len(chain)="
            f"{len(chain)}; query side will silently cap at match_len. "
            f"Caller is recording more groups than chain elements — "
            f"likely a stale-chain or wrong-counter bug.")
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
                with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                    self._client.unregister_sink(self._sink_pool.sink)
                self._sink_pool.sink.close()
            if self._per_head_summary_sink is not None:
                try:
                    with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                        self._client.unregister_sink(
                            self._per_head_summary_sink)
                    self._per_head_summary_sink.close()
                except Exception:
                    pass
                self._per_head_summary_sink = None
                self._per_head_summary_sink_capacity = 0
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

    # ─── runtime invariant flags ────────────────────────────────────────

    def _assert_pending_scores_no_clobber(self, layer_key, rid, source: str):
        """Runtime invariant flag (2026-05-15): every write to
        `_pending_scores[layer][rid]` MUST happen against an empty
        slot. Today the connector uses `.setdefault(...)[rid] = ...`
        at 5+ call sites, which silently CLOBBERS any pre-existing
        entry → stale (reply, req_idx) discarded, the new one wins,
        the apply path may consume the wrong req_idx.

        Call this BEFORE the `[rid] = ...` assignment at each site:
            self._assert_pending_scores_no_clobber(
                layer_key, rid, source="score-reply-landing")
            self._pending_scores.setdefault(layer_key, {})[rid] = (...)

        Default: log WARNING. Strict (`ICMS_PENDING_CLOBBER_FLAG=strict`):
        raise RuntimeError. The strict mode is useful in CI for
        catching design-level violations early; default-warn keeps
        production paths flowing even if a transient race happens.
        """
        layer_dict = self._pending_scores.get(layer_key)
        if layer_dict is None or rid not in layer_dict:
            return  # clean slot, nothing to flag
        msg = (
            f"[icms-invariant] _pending_scores clobber at "
            f"layer={layer_key} rid={rid} (source={source}). A prior "
            f"(reply, req_idx) is being overwritten — if the new "
            f"req_idx differs from the existing one, apply will use "
            f"the new value AND the prior reply's KV bytes are "
            f"discarded. Audit S-1 (2026-05-15).")
        if os.environ.get(
                "ICMS_PENDING_CLOBBER_FLAG", "warn") == "strict":
            raise RuntimeError(msg)
        logger.warning("%s", msg)

    def _check_faps_reply_coverage(self, rid, rs, reply,
                                    *, source: str):
        """Runtime invariant flag (2026-05-15): a FAPS reply MUST
        cover essentially the full chain. The expected page count is
        `len(rs.chain) * GROUP_PAGES`; allow 10% under (last group
        may be partial in practice). Below threshold = silent
        byte-source mismatch class (the SPDD v3 failure mode
        generalized to non-SPDD FAPS callers).

        The SPDD helper already does this check inline. This is
        the same check, exposed as a method so non-SPDD FAPS
        callers (`_fetch_all_one_request`, per-scoring-boundary
        FAPS, etc.) can adopt the same flag with one line:
            self._check_faps_reply_coverage(rid, rs, reply,
                                             source="layer-FAPS")

        Returns True if reply is "full enough", False if partial.
        Logs WARNING by default; raises under
        `ICMS_FAPS_PARTIAL_REPLY_FLAG=strict`.
        """
        page_ids = getattr(reply, "page_ids", None)
        if page_ids is None:
            return True   # nothing to check; caller decides
        chain = getattr(rs, "chain", None) or []
        expected = len(chain) * GROUP_PAGES
        if expected <= 0:
            return True   # empty chain; no expectation
        threshold = int(expected * 0.90)
        actual = len(page_ids)
        if actual >= threshold:
            return True
        msg = (
            f"[icms-invariant] FAPS partial reply for rid={rid} "
            f"(source={source}): {actual} pages received vs "
            f"{expected} expected (chain_len={len(chain)} × "
            f"GROUP_PAGES={GROUP_PAGES}, threshold=90%). Downstream "
            f"apply will scatter sink-routed bytes at the received "
            f"page IDs only; other positions retain whatever was "
            f"there → byte-source mismatch on uncovered positions.")
        if os.environ.get(
                "ICMS_FAPS_PARTIAL_REPLY_FLAG", "warn") == "strict":
            raise RuntimeError(msg)
        logger.warning("%s", msg)
        return False

    def _check_prefill_rid_order_stable(self, new_requests):
        """Runtime invariant flag (2026-05-15): during chunked prefill,
        vLLM's request ordering in `connector_meta.requests` MUST be
        stable across chunks of the same prefill — if a rid that
        appeared at position N in step K appears at position N' in
        step K+1 (while still in prefill), the apply path's `req_idx`
        will index into the wrong bt-row → silent KV corruption.

        This is the 2026-05-12 multi-rid slot-1 bug class. The fix
        landed (plumb `input_batch.req_ids` through), but no runtime
        CHECK existed that the scheduler ordering is actually stable.
        This is that check.

        Default behavior: log WARNING and continue (the
        `_rid_to_bt_row` plumbing handles correctness independently).
        Set ICMS_RID_ORDER_FLAG=strict to RAISE instead — useful in
        tests and CI to surface drift loudly.

        No-op once `_prefill_done=True` (decode-phase reordering is
        normal and handled by the apply path's per-step req_idx
        resolution).
        """
        if getattr(self, "_prefill_done", False):
            return
        prior = getattr(self, "_last_step_requests", None)
        if not prior:
            return
        prior_pos = {step.request_id: i for i, step in enumerate(prior)}
        drift = []
        for new_pos, step in enumerate(new_requests):
            rid = step.request_id
            old_pos = prior_pos.get(rid)
            if old_pos is not None and old_pos != new_pos:
                drift.append((rid, old_pos, new_pos))
        if not drift:
            return
        msg = (
            "[icms-invariant] rid-order drift across chunked prefill: "
            "%d rid(s) moved between consecutive prefill chunks "
            "(rid, old_pos, new_pos) → %s. This is the 2026-05-12 "
            "multi-rid slot-1 bug class — apply path's req_idx will "
            "index into wrong bt-row unless _rid_to_bt_row plumbing "
            "is intact. If the run is producing correct outputs, the "
            "plumbing is doing its job; this warning still pins the "
            "structural hazard for refactor visibility."
        ) % (len(drift), drift[:5])
        if os.environ.get("ICMS_RID_ORDER_FLAG", "warn") == "strict":
            raise RuntimeError(msg)
        logger.warning("%s", msg)

    def _check_rid_to_bt_row_present(self, active_rids, source: str):
        """Runtime invariant flag (2026-05-15): every rid we look up via
        `self._rid_to_bt_row.get(rid)` at apply / extract time MUST have
        an entry in the dict. A missing entry is the pre-2026-05-12
        multi-rid bug class — the lookup falls back to enumerate order
        which is meta.requests (append) ordering, NOT FA's input_batch
        row ordering. That mismatch silently routed the wrong Q-slice
        or wrong bt-row to each rid → slot-N deterministic accuracy
        collapse.

        Default behavior: log WARNING, name the missing rids, and
        continue (the fallback enumerate path still runs, so the
        invariant flag is OBSERVABILITY for the structural hazard).
        Set ICMS_RID_TO_BT_ROW_FLAG=strict to RAISE instead.

        `active_rids` is whatever iterable of rids the caller is about
        to look up. `source` names the call site for the log message.
        """
        missing = [r for r in active_rids
                   if r not in (self._rid_to_bt_row or {})]
        if not missing:
            return
        msg = (
            "[icms-invariant] _rid_to_bt_row missing %d rid(s) at "
            "%s: %s. Pre-2026-05-12 multi-rid slot-N bug class: "
            "lookup falls back to enumerate(meta.requests) order, "
            "which does NOT match FA input_batch row order under "
            "multi-rid batching. If this run is correct, "
            "set_input_batch_req_ids ran for these rids before the "
            "lookup; otherwise the lookup is routing the wrong rid."
        ) % (len(missing), source, missing[:5])
        if os.environ.get("ICMS_RID_TO_BT_ROW_FLAG",
                          "warn") == "strict":
            raise RuntimeError(msg)
        logger.warning("%s", msg)

    def on_step_start(self, meta: IcmsConnectorMetadata):
        """Called from start_load_kv. Drains scheduler metadata into caches."""
        t_step = time.perf_counter()
        self.stats.advance_step()
        # Reset prefill_done when a new request arrives (new chain delivered).
        if meta.new_chains:
            self._prefill_done = False
            # Invalidate the all-dense cache: a new request always starts
            # with rs.dense_mode=False, so the answer is necessarily False
            # until the new rs flips. Avoids needing a recompute (we know
            # the answer without iterating).
            self._cached_all_dense = False
            for rid in meta.new_chains.keys():
                self._ttft_reset(rid, t_step)
        # ICMS_DIAG_FULL: bump _post_dense_iter at the start of each
        # forward AFTER dense_mode flipped. Capped at 3 — beyond that the
        # post-flip diagnostics turn off automatically (we only need the
        # first few iters to spot stale-state leak).
        if os.environ.get("ICMS_DIAG_FULL") == "1":
            for rs in self._requests.values():
                if rs.dense_mode and rs._post_dense_iter >= 0:
                    rs._post_dense_iter = min(rs._post_dense_iter + 1, 99)
        # ICMS_DIAG_DECODE_ITER=1: bump per-decode-iter wall-time counter
        # on each post-prefill step + log iter wall_ms. Used to measure
        # M3+M4-A overhead per decode iter at large ctx.
        if os.environ.get("ICMS_DIAG_DECODE_ITER") == "1":
            now = time.perf_counter()
            for rs in self._requests.values():
                if not getattr(self, "_prefill_done", False):
                    rs._decode_iter_count = 0
                    rs._decode_iter_t_last = now
                    continue
                if not hasattr(rs, "_decode_iter_count"):
                    rs._decode_iter_count = 0
                    rs._decode_iter_t_last = now
                else:
                    rs._decode_iter_count += 1
                    iter_ms = (now - rs._decode_iter_t_last) * 1e3
                    rs._decode_iter_t_last = now
                    logger.info(
                        "[icms] decode_iter rid=%s iter=%d ms=%.2f "
                        "dense=%s fetched_layers=%d",
                        rs.request_id, rs._decode_iter_count,
                        iter_ms, rs.dense_mode,
                        len(rs.fetched_pages))

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
        # Multi-rid path uses a per-rid set instead of the global gate.
        self._skip_extract_rids = set(meta.skip_extract_rids)
        # 2026-05-15 runtime invariant flag: chunked-prefill rid-order
        # stability. The 2026-05-12 multi-rid slot-1 bug was that vLLM
        # swapped rid order between chunks of the same prefill; the
        # connector's apply path consumed `connector_meta.requests`
        # index as req_idx, so rid_A's apply scattered into rid_B's
        # bt-row. Fix landed (plumb input_batch.req_ids through), but
        # there was no runtime CHECK that the scheduler's ordering is
        # stable. This is that check.
        self._check_prefill_rid_order_stable(meta.requests)
        # Stash for extract_and_record's batch-order walk.
        self._last_step_requests = list(meta.requests)
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
            # Prefer the scheduler-side value when present. The worker's
            # `_stored_chain_groups` ledger races: turn N's pipeline may
            # not have committed by turn N+1's first on_step_start. The
            # scheduler's `_stored_chains` is drained from
            # `_stored_chain_queue` at build_meta entry, so it reflects
            # everything the worker has actually flushed.
            sched_n = (meta.stored_groups_by_rid.get(rid, 0)
                       if hasattr(meta, "stored_groups_by_rid")
                       else 0)
            local_n = self._get_stored_context_groups(chain)
            n_stored = max(sched_n, local_n)
            # 2026-05-10 TP>1 stored_groups asymmetry fix
            # (forward-thread symmetrize): per-rank `local_n` from the
            # `_stored_chain_groups` ledger can diverge widely across
            # ranks (observed rank-0=12 vs rank-1=63 at low budget
            # batched mode). Without symmetrization, the
            # extract/apply/score paths on different ranks compute
            # different `effective_start` / `context_pages`, leading to
            # divergent K bytes scattered into the cache → garbled
            # output starting at the second scored layer. We are on
            # the worker forward thread here (called from
            # bind_connector_metadata under vLLM's main NCCL group),
            # so the all-reduce is safe and bit-symmetric. Both ranks
            # iterate `meta.new_chains` in the same order (dict
            # insertion order from the same scheduler output).
            if self._tp_size > 1:
                n_stored = _tp_allreduce_max_int(n_stored, self._tp_size)
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.stored_groups = n_stored
            # Mirrored for the pipeline-thread extract path which
            # MUST NOT issue NCCL itself (collides with main-thread
            # per-layer all_reduce).
            rs._effective_stored_groups = n_stored
            if os.environ.get("ICMS_DIAG_N13", "0") == "1":
                logger.info(
                    "[diag-step] rank=%d new_chain rid=%s chain_len=%d "
                    "stored_groups=%d (sched=%d local=%d)",
                    self._tp_rank, rid, len(chain),
                    n_stored, sched_n, local_n)
        for rid, chain in meta.new_chains.items():
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.chain = chain
            self._last_chain_for_rid[rid] = chain
        for rid, bids in meta.block_id_maps.items():
            rs = self._requests.setdefault(rid, _RequestState(request_id=rid))
            rs.block_ids = bids

        # KV-block provenance: record alloc snapshots delivered via
        # metadata. Only fires when env flag is on (zero cost otherwise).
        # Records per-rid the ext_comp range so check_bt can verify each
        # layer's attention bt covers ext_comp blocks only through ICMS
        # apply scatter, never through unpopulated free-pool blocks.
        if (icms_provenance.is_enabled()
                and getattr(meta, "prov_block_ids", None)):
            for rid, bids in meta.prov_block_ids.items():
                n_ext = int(meta.prov_ext_comp_tokens.get(rid, 0))
                n_local = int(meta.prov_local_cached_tokens.get(rid, 0))
                icms_provenance.tracker().record_alloc(
                    rid=rid,
                    block_ids=bids,
                    num_external_tokens=n_ext,
                    num_local_cached_tokens=n_local,
                    page_tokens=PAGE_TOKENS,
                )
        # Client-side preload DISABLED by default — the server now kicks off
        # the same async preload internally on first Score (see
        # kick_off_summary_preload in handlers.cc), which avoids the
        # client-side cold-frame penalty entirely. Set
        # ICMS_DISABLE_PRELOAD=0 to re-enable the client-fired path.
        if int(os.environ.get("ICMS_DISABLE_PRELOAD", "1")) == 0:
            for rid, chain in meta.new_chains.items():
                if chain:
                    self._fire_preload(rid, chain)

        # 2026-05-10 follow-up audit trace: emit a per-rid state hash
        # for every active rid so we can diff rank 0 vs rank 1 after
        # on_step_start. Catches any future per-rid field that
        # silently drifts between ranks. Pair fields with the
        # SYMMETRIC_PER_RID_FIELDS list in the test harness.
        if _ICMS_TRACE_ENABLED:
            try:
                import hashlib as _hl_rs
                for rid, rs in self._requests.items():
                    chain = list(getattr(rs, "chain", []) or [])
                    fields = {
                        "stored_groups": int(getattr(rs, "stored_groups", 0)),
                        "_effective_stored_groups": int(getattr(
                            rs, "_effective_stored_groups", 0)),
                        "num_groups_written": int(getattr(
                            rs, "num_groups_written", 0)),
                        "flushed_local": int(getattr(
                            rs, "flushed_local", 0)),
                        "chain_len": len(chain),
                        "block_ids_len": len(list(
                            getattr(rs, "block_ids", []) or [])),
                    }
                    # Hash the chain so divergent chains are caught
                    # without dumping the full list per rid per step.
                    chain_b = ",".join(str(x) for x in chain).encode("ascii")
                    chain_h = _hl_rs.blake2b(chain_b,
                                              digest_size=8).hexdigest()
                    fields["chain_hash"] = chain_h
                    _icms_trace(
                        "rs_state_hash", rid, layer=-1, chain_fp=chain_h,
                        extra={
                            "tp_rank": int(self._tp_rank),
                            "tp_size": int(self._tp_size),
                            **fields,
                        })
            except Exception:
                pass

    def _fire_preload(self, request_id: str, chain: list[int]):
        """Fire a one-way summary preload on the scheduler thread.

        Sends a preload frame and returns immediately — no client-side
        wait. The server populates its DRAM cache for this request_id
        while the forward pass starts on the GPU. The first real Score
        on the same connection is FIFO-ordered behind the preload on
        the reactor, so it's guaranteed to hit the cache.
        """
        # Option W: only rank 0 talks to the wire. Score is also rank-0-only,
        # so a rank-N>0 preload would populate a cache entry that nothing
        # ever reads — wasted work + duplicate frames on the server.
        if self._tp_size > 1 and self._tp_rank != 0:
            return
        try:
            icms_rid = self._icms_request_id(request_id, 0)
            with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
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

        # FAPS audit Finding 4 fix (2026-05-11): chunked-prefill
        # invalidation. When prefill is chunked (e.g., mistral-small
        # with default `max_num_batched_tokens=16384` at ctx > 16K),
        # FAPS' FetchAll fires at chunk-0's layer-0 forward and the
        # reply only covers the groups COMMITTED to the server by
        # then. Later chunks commit more groups (via `_flush_group`
        # → `write_group` RPC → `_drain_pending_flush_queue` bumps
        # `rs.flushed_local`). The cached `_pending_reuse` offsets
        # from chunk-0 do NOT cover those new pages → next chunk's
        # `on_layer_reuse` returns the chunk-0-sized page set →
        # vLLM's attention under-covers the chain → silent KV
        # mismatch labelled as FAPS-success.
        #
        # Detect via the server-side-committed count
        # (`flushed_local + stored_groups`) vs the count stamped at
        # last successful FAPS dispatch. If it grew, invalidate
        # `_fetch_all_complete` and clear the rid's cached
        # `_pending_reuse` / `_pending_scores` entries so the slow
        # path re-fires below covering the new chain. Note: we set
        # the stamp at the BOTTOM of the function under the
        # cross-rank consensus branch from Finding 1, so the
        # invalidation is rank-symmetric (every rank sees the same
        # `flushed_local`/`stored_groups` post the on_step_start
        # allreduce-MAX, so the comparison is rank-deterministic
        # → no asymmetric NCCL collective shape).
        _committed_now = (int(getattr(rs, "flushed_local", 0))
                           + int(getattr(rs, "stored_groups", 0)))
        _committed_at_dispatch = int(getattr(
            rs, "_fetch_all_committed_at_dispatch", 0))
        if (getattr(rs, "_fetch_all_complete", False)
                and _committed_now > _committed_at_dispatch):
            rs._fetch_all_complete = False
            with self._score_lock:
                for _inner in self._pending_reuse.values():
                    _inner.pop(rid, None)
                for _inner in self._pending_scores.values():
                    _inner.pop(rid, None)
            if os.environ.get("ICMS_DIAG_FAPS") == "1":
                logger.info(
                    "[diag-faps] chain grew rid=%s committed %d→%d — "
                    "invalidating cached reuse for re-FetchAll "
                    "(chunked-prefill audit Finding 4)",
                    rid[:8], _committed_at_dispatch, _committed_now)

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
                if os.environ.get("ICMS_DIAG_FAPS") == "1":
                    logger.info(
                        "[diag-faps] fast-path MISS rid=%s layer=%d",
                        rid[:8], next_layer_idx)
                return
            # Tuple grew from 2- to 3-element on 2026-05-05 (added
            # req_idx) to fix the multi-rid stride-reuse path that was
            # defaulting req_idx=0 for the second rid. Accept both
            # shapes; old 2-element entries fall back to the caller's
            # req_idx (correct in this slow-path entry which has it).
            if len(reuse_entry) == 3:
                reply, reuse_offsets, _stored_req_idx = reuse_entry
            else:
                reply, reuse_offsets = reuse_entry
            promoted = copy.copy(reply)
            promoted.sink_offsets = reuse_offsets
            with self._score_lock:
                self._assert_pending_scores_no_clobber(
                    attn_layer_name, rid, source="faps-fast-path-promote")
                self._pending_scores.setdefault(attn_layer_name, {})[rid] = (
                    promoted, req_idx)
            if os.environ.get("ICMS_DIAG_FAPS") == "1":
                logger.info(
                    "[diag-faps] fast-path HIT rid=%s layer=%d k=%d sink_off_n=%d",
                    rid[:8], next_layer_idx, len(reply.page_ids),
                    len(reuse_offsets))
            return

        # Slow path: first scoring boundary for this request — issue
        # the single full-request FetchAll covering every layer.
        icms_rid = self._icms_request_id(rid, 0)
        # BUG-N7: read the cached per-request value populated by
        # on_step_start instead of rescanning _stored_chain_groups
        # (O(N stored × chain_len)) on every layer in the hot path.
        stored_groups = rs.stored_groups
        effective_groups = max(rs.num_groups_written, stored_groups)
        # 2026-05-10 TP>1 num_groups_written symmetry fix (fetch-all):
        # `num_groups_written` is per-rank by design — sentinel padding
        # in `_drain_pending_flush_queue` makes the broadcast symmetric
        # while letting each rank's bumps track local pipeline
        # progress. Pre-fix, this site relied on
        # `max(num_groups_written, stored_groups)` being dominated by
        # the symmetric `stored_groups` to stay safe across ranks. A
        # future code path that pushes `num_groups_written >
        # stored_groups` on one rank only would cause `total_pages` to
        # differ between ranks → rank-0's fetch_all RPC sized smaller
        # than rank-1's local context_pages → `pid >= context_pages`
        # filter on rank-1's apply drops pages → asymmetric attention.
        # Mirrors the all-reduce at _score_one_request:~4002 and
        # _apply_selective_attention:~6033. Both ranks reach this site
        # symmetrically (scheduler-broadcast metadata gates entry).
        if self._tp_size > 1:
            effective_groups = _tp_allreduce_max_int(
                effective_groups, self._tp_size)
        total_pages = effective_groups * _GROUP_BLOCKS

        if not getattr(rs, "_budget_logged", False):
            # Once-per-request decision marker the perf-sweep scrapes.
            logger.info(
                "icms_budget rid=%s layer=%d src=fetch_all budget=%.3f k=%d "
                "total_pages=%d",
                rid, next_layer_idx, budget, total_pages, total_pages,
            )
            rs._budget_logged = True

        # Symmetric early-return on fresh chain (2026-05-07): mirrors
        # _score_one_request's `if total_pages == 0: return`. Without
        # this, rank 0 fires fetch_all RPC on an unresolved chain → server
        # returns ENOENT → IcmsError raised → exception bypassed the
        # broadcast (was inside the outer try block) → rank 1 deadlocked
        # in dist.broadcast → vLLM sample_tokens RPC timed out at 5 min.
        # `total_pages` is now symmetric across ranks via the all-reduce
        # above, so the early-return fires symmetrically.
        if total_pages == 0:
            return

        reuse_through = num_layers - 1

        # ICMS_TRACE_FLAGS=1: snapshot flag state at the call site. The
        # actual clear happens inside rdma_client.py:193/238 (when
        # use_flags=True), not here — see Bug #1 fix below.
        if os.environ.get("ICMS_TRACE_FLAGS") == "1":
            try:
                snap = self._sink_pool.sink.snapshot_flags() if hasattr(
                    self._sink_pool.sink, "snapshot_flags") else None
                _t = time.perf_counter()
                _set_layers = ([i for i, v in enumerate(snap) if v]
                               if snap is not None else None)
                logger.info(
                    "[trace-flags] CLEAR site=fetch_all t=%.6f rid=%s "
                    "set_before=%s",
                    _t, request_id, _set_layers)
            except Exception:
                pass
        # Bug #1 fix (race-audit 2026-05-08): the connector previously
        # called self._sink_pool.sink.clear_ready_flags() right here,
        # then rdma_client.py:238 cleared again inside fetch_all() ⇒
        # double-clear with a tiny window between. At ICMS_ALLOW_BATCH=1
        # another rid's flag-flip can land between the two clears and be
        # silently wiped, causing this rid's wait_for_layer to spin to
        # the 5s timeout. Removed: rdma_client's clear is sufficient
        # (use_flags=True path), and unix-socket / mem-backend transports
        # don't read flags at all (wait_for_layer at icms_connector.py:5147
        # gates on flag_count > 0, so no-op there).

        # Adaptive-bandwidth fields for the wire. Both 0 when adaptive is
        # off, in which case the server skips its registry + min and the
        # reply's effective_supply_bps is 0.
        ab_demand_bps = 0
        ab_compute_supply_bps = 0
        if self._adaptive_allocator is not None:
            ab_demand_bps = self._adaptive_allocator.demand_bps_for(rid)
            ab_compute_supply_bps = (
                self._adaptive_allocator.compute_supply_bps_for(rid))

        t_start = time.perf_counter()
        reply = None
        # Defense-in-depth (2026-05-07): isolate the RPC in its own
        # try/except so a rank-0 RPC exception (e.g., ENOENT on a
        # racing chain) does NOT bypass the broadcast collective below.
        # Mirrors `on_layer_score`'s structure — broadcast is outside
        # the RPC try so every rank participates regardless of rank-0
        # outcome. The legacy structure had the broadcast inside the
        # try, which deadlocked rank 1 on dist.broadcast at TP=2 +
        # fresh chain.
        try:
            # Audit B1 fix (2026-05-06): rank-gate the RPC. Server's
            # drain-time fan-out replicates rank-0's sink to every peer
            # rank, so non-zero ranks don't need the wire round-trip —
            # they just need the reply tuple (page_ids, sink_offsets, …)
            # to populate _pending_scores / _pending_reuse identically.
            # Mirrors Score's gate at on_layer_score (line ~3298).
            if self._tp_size > 1 and self._tp_rank != 0:
                reply = None
            else:
                with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                    reply = self._client.fetch_all(
                        request_id=icms_rid,
                        chain=self._rank_chain(rs.chain),
                        layer=next_layer_idx,
                        sink=self._sink_pool.sink,
                        reuse_through_layer=reuse_through,
                        # Reply-early: server ships the FetchAll reply as soon
                        # as page_ids are known; Phase-2 KV writes + per-layer
                        # flag flips run in background so the GPU forward pass
                        # overlaps with the transfer. Note: reply's
                        # sink_write_ns / server_ingest_to_ready_ns are
                        # reported as 0 in this mode — the transfer wall-time
                        # shows up as the sum of per-layer wait_spin on the
                        # client side.
                        use_flags=(os.environ.get("ICMS_REPLY_EARLY", "1") == "1"),
                        demand_bps=ab_demand_bps,
                        compute_supply_bps=ab_compute_supply_bps,
                    )
        except Exception as e:
            # 2026-05-08: keep at warning so future silent-failure
            # regressions are visible (debug-level masked the qwen3
            # b=1.0 corruption for ~3 weeks). Cheap log; only fires on
            # actual exceptions.
            logger.warning(
                "fetch_all RPC FAILED rank=%d layer=%d type=%s status=%s "
                "msg=%s",
                self._tp_rank, next_layer_idx, type(e).__name__,
                getattr(e, "status", "?"),
                str(e)[:200])
            reply = None
        t_end = time.perf_counter()

        # Broadcast reply across ranks UNCONDITIONALLY — must be outside
        # the RPC try-except so a rank-0 failure still reaches the
        # collective on every rank. Empty reply still broadcasts a header
        # with n_pages=0 and returns symmetrically.
        if self._tp_size > 1:
            try:
                reply = _tp_broadcast_score_reply(
                    reply, self._tp_rank, self._tp_size)
            except Exception:
                logger.exception(
                    "fetch_all broadcast failed rank=%d layer=%d",
                    self._tp_rank, next_layer_idx)
                reply = None
        if os.environ.get("ICMS_DIAG_FA_TRACE") == "1":
            logger.info(
                "[diag-fa-trace] post-broadcast rid=%s layer=%d "
                "total_pages=%d reply_is_none=%d n_page_ids=%d",
                rid, next_layer_idx, total_pages,
                int(reply is None),
                (len(reply.page_ids) if reply is not None else -1))
        # ICMS_STRICT_ASSERTIONS=1: convert silent-state-empty failure
        # patterns into loud crashes. Pre-2026-05-08, fetch_all silently
        # AttributeError'd on non-RDMA → reply=None → empty
        # _pending_scores → corrupted KV → qwen3 b=1.0 = 0.067. This
        # assert would have crashed on the first b=1.0 call instead of
        # silently producing wrong output for ~3 weeks. Cheap; safe to
        # leave on for sweeps + CI; opt-out for paper runs.
        if (os.environ.get("ICMS_STRICT_ASSERTIONS", "0") == "1"
                and total_pages > 0
                and (reply is None or not reply.page_ids)):
            raise RuntimeError(
                f"FetchAll returned empty reply but total_pages={total_pages}; "
                f"client={type(self._client).__name__} rid={rid} "
                f"layer={next_layer_idx} tp_rank={self._tp_rank}. "
                f"This usually means the transport doesn't implement "
                f"fetch_all (use _supports_fetch_all gate) or the RPC "
                f"silently failed (see warning log line above for the "
                f"actual exception)."
            )
        if reply is None or not reply.page_ids:
            # Rank-0 RPC failed, broadcast erred, or server returned
            # zero pages. Record a miss so stats stay consistent and
            # skip post-processing — _pending_scores stays empty for
            # this layer/rid and the downstream wait_for_layer falls
            # through.
            self.stats.record_score(
                (t_end - t_start) * 1e6, False, [], next_layer_idx,
            )
            return
        # Audit Finding 1 fix (2026-05-11): track whether THIS rank's
        # post-broadcast bookkeeping completed without raising. We
        # synchronize across ranks AFTER the try/except so all ranks
        # agree on the resulting `_fetch_all_complete` state. Pre-fix,
        # an exception on one rank between the broadcast (above) and
        # the `_fetch_all_complete=True` write could leave ranks in
        # asymmetric states: the success rank would fast-path next
        # stride (no broadcast) while the failure rank would re-enter
        # the slow path (calling broadcast) → deadlock.
        _fa_local_ok = False
        try:
            rs._last_storage_concurrent = getattr(
                reply, 'concurrent_requests', 0)
            # Adaptive-bandwidth: stash the storage-side effective supply
            # so the next stride's get_budget(rid) takes
            # min(compute, storage). 0 means adaptive off server-side.
            if self._adaptive_allocator is not None:
                self._adaptive_allocator.apply_storage_supply(
                    rid, int(getattr(reply, 'effective_supply_bps', 0) or 0))
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
            # M2: prime decode-mode fetched-pages tracking. FetchAll
            # returns every page in the resolved chain, so this single
            # update marks the stride-group fully populated — useful for
            # M4's "skip Score when bitmap full" optimization later.
            if reply.page_ids:
                rs.fetched_pages.setdefault(
                    next_layer_idx, set()).update(reply.page_ids)
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
                self._assert_pending_scores_no_clobber(
                    attn_layer_name, rid, source="faps-slow-path-landing")
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
                    # 3-tuple now carries req_idx so on_layer_reuse can
                    # restore the correct batch position per-rid
                    # (multi-rid stride-reuse fix, 2026-05-05).
                    self._pending_reuse.setdefault(reuse_attn, {})[rid] = (
                        reply, reuse_offsets, req_idx)
            # Audit Finding 1 fix: defer the `_fetch_all_complete=True`
            # write until after cross-rank consensus below. Setting it
            # here (inside the try) was the pre-fix bug — if a peer
            # rank's try raises after this point, ranks diverge.
            _fa_local_ok = True
            if os.environ.get("ICMS_DIAG_FAPS") == "1":
                logger.info(
                    "[diag-faps] slow-path DONE rid=%s layer=%d k=%d "
                    "total_pages=%d reuse_layers_set=%d sink_off_n=%d",
                    rid[:8], next_layer_idx, len(reply.page_ids),
                    total_pages, reuse_through - next_layer_idx,
                    len(reply.sink_offsets))
            # ICMS_DIAG_PAGE_IDS dump (FetchAll path mirror of the Score
            # path block at ~line 3857). Lets us diff page-ID sets from
            # FetchAll vs Score on the same prompt to localize the
            # qwen3-specific b=1.0 corruption.
            _diag_pids_path = os.environ.get("ICMS_DIAG_PAGE_IDS", "")
            if (_diag_pids_path
                    and not getattr(self, "_diag_page_ids_fa_fired", False)):
                self._diag_page_ids_fa_fired = True
                try:
                    import json as _json
                    _pids = list(int(p) for p in reply.page_ids)
                    _sorted_pids = sorted(_pids)
                    _hash = 0
                    for _p in _sorted_pids:
                        _hash = (_hash * 31 + _p) & 0xFFFFFFFFFFFFFFFF
                    out_path = (_diag_pids_path
                                if _diag_pids_path != "1"
                                else f"/tmp/icms_diag_pids_FA_tp{self._tp_size}_"
                                     f"r{self._tp_rank}_pid{os.getpid()}.json")
                    if _diag_pids_path != "1":
                        # When user passed a path, suffix with _FA so
                        # FetchAll dump doesn't overwrite Score dump.
                        _root, _ext = os.path.splitext(out_path)
                        out_path = f"{_root}_FA{_ext or '.json'}"
                    _scores = [float(s) for s in reply.scores]
                    _pid_score = list(zip(_pids, _scores))
                    with open(out_path, "w") as f:
                        _json.dump({
                            "src": "fetch_all",
                            "tp_rank": self._tp_rank,
                            "tp_size": self._tp_size,
                            "rid": rid,
                            "layer": next_layer_idx,
                            "n_pids": len(_pids),
                            "pids_sorted_hash": f"{_hash:016x}",
                            "pids_sorted": _sorted_pids,
                            "pid_score_unsorted": _pid_score,
                        }, f)
                    logger.info(
                        "[diag-page-ids] src=fetch_all rank=%d tp_size=%d "
                        "rid=%s layer=%d n_pids=%d head[:8]=%s tail[:8]=%s "
                        "sorted_hash=%016x dumped=%s",
                        self._tp_rank, self._tp_size, rid, next_layer_idx,
                        len(_pids), _pids[:8], _pids[-8:], _hash, out_path)
                except Exception as _e:
                    logger.warning("diag-page-ids (fetch_all) failed: %s", _e)
        except Exception as e:
            t_end = time.perf_counter()
            self.stats.record_score(
                (t_end - t_start) * 1e6, False, [], next_layer_idx,
            )
            # Bug #3 fix (race-audit 2026-05-08): roll back any partial
            # writes to _pending_scores / _pending_reuse for this rid
            # before returning. Without this, an exception that lands
            # AFTER the _pending_scores.setdefault(...) at line ~3147 or
            # PARTWAY through the per-reuse-layer loop at ~3155-3165
            # leaves stale entries that on_layer_reuse pops at later
            # layers as if they were valid — silent corruption.
            # _fetch_all_complete is left False (never set in the except
            # path), so the next stride boundary will re-issue fetch_all
            # against a clean slate. Reuses the same per-rid sweep pattern
            # as on_request_finished (BUG-N8 cleanup at icms_connector
            # line ~7763-7774).
            with self._score_lock:
                for _inner in self._pending_reuse.values():
                    _inner.pop(rid, None)
                for _inner in self._pending_scores.values():
                    _inner.pop(rid, None)
            logger.debug("fetch_all failed layer %d: %s", next_layer_idx, e)

        # Audit Finding 1 fix (2026-05-11): cross-rank consensus on
        # whether THIS stride's fetch_all bookkeeping succeeded
        # everywhere. MIN-reduce so any rank's failure flips the
        # consensus to False; only flip `_fetch_all_complete=True`
        # when EVERY rank succeeded. If we locally succeeded but a
        # peer failed, roll back our pending_* writes so next
        # stride all ranks re-enter the slow path symmetrically.
        if self._tp_size > 1:
            try:
                _fa_consensus = bool(
                    _tp_allreduce_min_int(int(_fa_local_ok), self._tp_size))
            except Exception as _bcast_e:
                logger.warning(
                    "_fetch_all_one_request: consensus broadcast failed "
                    "(%s) — falling back to local value; cross-rank "
                    "state may diverge", _bcast_e)
                _fa_consensus = _fa_local_ok
        else:
            _fa_consensus = _fa_local_ok
        if _fa_consensus:
            rs._fetch_all_complete = True
            # FAPS audit Finding 4 fix (2026-05-11): stamp the
            # server-side-committed group count AT this successful
            # FAPS dispatch. The top-of-function chain-grew check
            # compares the current count against this stamp; if it
            # grew, the cached reuse offsets from THIS dispatch don't
            # cover the new groups → invalidate + re-fire. See
            # `_fetch_all_committed_at_dispatch` docstring on
            # `_RequestState` for the chunked-prefill rationale.
            rs._fetch_all_committed_at_dispatch = (
                int(getattr(rs, "flushed_local", 0))
                + int(getattr(rs, "stored_groups", 0)))
        elif _fa_local_ok:
            # Local success but a peer failed — roll back so next
            # stride re-enters slow path on every rank symmetrically.
            with self._score_lock:
                for _inner in self._pending_reuse.values():
                    _inner.pop(rid, None)
                for _inner in self._pending_scores.values():
                    _inner.pop(rid, None)
            logger.debug(
                "fetch_all: rolled back rid=%s layer=%d due to peer "
                "rank failure (consensus=False)", rid, next_layer_idx)

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
        # reuse_data is dict[rid → (reply, reuse_offsets, req_idx)]
        # (3-element tuple as of 2026-05-05; older 2-element entries
        # are still accepted for back-compat with the previous-layer-
        # lookup fallback).
        with self._score_lock:
            pending = self._pending_scores.setdefault(attn_layer_name, {})
            for rid, entry in reuse_data.items():
                if len(entry) == 3:
                    reply, reuse_offsets, req_idx = entry
                else:
                    # Legacy 2-tuple. Fall back to the previous layer's
                    # popped-dict lookup; if that's gone (always at
                    # multi-rid because wait_for_layer pops eagerly),
                    # default to 0. The 2-tuple write sites have all
                    # been migrated above, so this branch should not
                    # fire post-2026-05-05; kept defensively.
                    reply, reuse_offsets = entry
                    orig = self._pending_scores.get(
                        f"model.layers.{next_layer_idx - 1}.self_attn.attn", {})
                    req_idx = orig.get(rid, (None, 0))[1] if orig else 0
                reuse_reply = copy.copy(reply)
                reuse_reply.sink_offsets = reuse_offsets
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
        # 2026-05-09: with a non-zero scored_layers_mask (e.g. dense-only for
        # gemma-3 sliding-window models), the server rejects Score for layers
        # not in the mask ("Score: layer is not in scored_layers set", EINVAL).
        # Skip the Score+reuse dispatch entirely on non-scored layers; the
        # corresponding wait_for_layer call will also short-circuit and clear
        # the active fetch state so attention falls back to natural full-
        # context bt for those layers (e.g. SW layers in gemma-3).
        if not self._geom.is_scored(next_layer_idx):
            self._slack_probe_pre_hook(next_layer_idx)
            return
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

        # Fallback: when connector_meta.requests is empty (rare; e.g.,
        # transient scheduler-output edge during connector startup or
        # an exception path that bypasses build_connector_meta).
        # Audit Finding 16 fix (2026-05-11): mirror the audit-#19
        # fix at `extract_and_record` — prefer `_last_step_requests`
        # (broadcast scheduler metadata, identical order across ranks)
        # over `self._requests.items()` (Python dict insertion order,
        # which can drift if a retry path on one rank inserts an extra
        # rs that the other rank doesn't have). At TP>1, dict-iter
        # divergence would cause different ranks to build different
        # `requests_to_score` lists → asymmetric Score-RPC count →
        # NCCL collective shape mismatch → hang. Falls back to the
        # legacy dict-iter only when `_last_step_requests` is also
        # empty (the genuinely-empty-step case).
        if not requests_to_score:
            _last_step = getattr(self, "_last_step_requests", None)
            if _last_step:
                for req_idx, step_req in enumerate(_last_step):
                    rid = step_req.request_id
                    rs = self._requests.get(rid)
                    if rs is not None and rs.chain:
                        requests_to_score.append((req_idx, rid, rs))
            else:
                for req_idx, (rid, rs) in enumerate(self._requests.items()):
                    if rs.chain:
                        requests_to_score.append((req_idx, rid, rs))

        _diag_se_n = getattr(self, "_diag_score_entry_count", 0)
        if (os.environ.get("ICMS_DIAG_SCORE_ENTRY", "0") == "1"
                and _diag_se_n < 5):
            self._diag_score_entry_count = _diag_se_n + 1
            try:
                _meta_n_reqs = (
                    len(connector_meta.requests)
                    if connector_meta is not None
                       and isinstance(connector_meta, IcmsConnectorMetadata)
                       and connector_meta.requests is not None
                    else -1)
                _meta_new_chains_n = (
                    len(connector_meta.new_chains)
                    if connector_meta is not None
                       and isinstance(connector_meta, IcmsConnectorMetadata)
                       and connector_meta.new_chains is not None
                    else -1)
                _self_reqs_n = len(self._requests)
                _self_reqs_with_chain = sum(
                    1 for rs in self._requests.values() if rs.chain)
                _last_chain_n = len(getattr(self, "_last_chain_for_rid", {}))
                logger.info(
                    "[diag-score-entry] rank=%d tp_size=%d layer=%d "
                    "meta_n_reqs=%d meta_new_chains=%d self_requests=%d "
                    "self_with_chain=%d last_chain_for_rid_n=%d "
                    "requests_to_score=%d",
                    self._tp_rank, self._tp_size, next_layer_idx,
                    _meta_n_reqs, _meta_new_chains_n, _self_reqs_n,
                    _self_reqs_with_chain, _last_chain_n,
                    len(requests_to_score))
            except Exception as _e:
                logger.warning("diag-score-entry failed: %s", _e)

        if not requests_to_score:
            return

        # 2026-05-07 DEFENSIVE ASSERT: validate quest_query length against
        # FA's authoritative cu_q (query_start_loc[-1]). Any mismatch
        # means our per-rid token-count math is fundamentally desync'd
        # from what FA itself sees. The Q-slice typo (scheduled_num_tokens
        # vs num_scheduled_tokens) at line 1309 silently produced
        # mismatched math for 4 weeks of debugging because nothing
        # surfaced this divergence. Fail loud now so future
        # vLLM-API-rename or scheduler-output changes are caught
        # immediately. Off via ICMS_QSLICE_NO_ASSERT=1 if needed.
        if (os.environ.get("ICMS_QSLICE_NO_ASSERT", "0") != "1"
                and quest_query is not None
                and isinstance(quest_query, torch.Tensor)
                and quest_query.ndim >= 1):
            try:
                _layer_name = (
                    f"model.layers.{next_layer_idx}.self_attn.attn")
                _am = (self._attn_metadata.get(_layer_name)
                       if isinstance(self._attn_metadata, dict)
                       else None)
                _qsl = getattr(_am, "query_start_loc", None) \
                    if _am is not None else None
                if _qsl is not None and _qsl.numel() > 0:
                    _expected = int(_qsl[-1].item())
                    if quest_query.shape[0] != _expected:
                        logger.warning(
                            "[icms-qslice-assert] quest_query length "
                            "mismatch: q=%d vs query_start_loc[-1]=%d "
                            "(layer=%d). Q slicing math diverges from "
                            "FA's cu_q — page selection will be wrong.",
                            quest_query.shape[0], _expected,
                            next_layer_idx)
            except Exception as _e:
                # Don't fail the request just because the probe broke.
                logger.warning("[icms-qslice-assert] probe error: %s", _e)

        # Split the Q tensor per request using token counts from metadata.
        # quest_query shape: [total_tokens, num_heads, head_dim]
        # 2026-05-07 BUG FIX: the prior offset-based slicing relied on
        # `num_computed_tokens_end - num_computed_tokens_start` being
        # the per-rid token count in quest_query. Empirically those
        # metadata fields are ZERO at decode AND in many prefill iters
        # in this vLLM version → every per-rid slice was empty
        # (quest_query[0:0]). The fallback `per_request_q.get(rid,
        # quest_query)` did NOT fire because rid IS in the dict (just
        # with an empty tensor). Each rid's Score saw an empty Q →
        # server picked top-K based on no information → page selection
        # was effectively random → batched-mode accuracy collapse.
        #
        # Detection + correct slicing:
        # 1. Decode mode: quest_query.shape[0] == len(meta.requests).
        #    Each rid has 1 token; slice by req_idx.
        # 2. Prefill with valid metadata: meta_tokens equals
        #    quest_query.shape[0]; use offset-based slicing as before.
        # 3. Otherwise: skip the dict; fallback to full quest_query.
        per_request_q = {}
        if (quest_query is not None
                and isinstance(quest_query, torch.Tensor)
                and quest_query.ndim >= 2
                and len(requests_to_score) > 1
                and connector_meta is not None
                and connector_meta.requests):
            _q_n = quest_query.shape[0]
            _meta_n_reqs = len(connector_meta.requests)

            # Phase B (2026-05-07): query_start_loc-based slicing.
            # FA's cu_seqlens_q is the AUTHORITATIVE per-rid token
            # boundary (flash_attn.py:197). Use it when available so
            # we're anchored to the same metadata FA itself uses,
            # robust against future scheduler-output API drift.
            # Falls through to the metadata-token-count math if qsl
            # isn't available or doesn't match.
            _qsl_used = False
            try:
                _layer_name = (
                    f"model.layers.{next_layer_idx}.self_attn.attn")
                _am = (self._attn_metadata.get(_layer_name)
                       if isinstance(self._attn_metadata, dict)
                       else None)
                _qsl = (getattr(_am, "query_start_loc", None)
                        if _am is not None else None)
                if (_qsl is not None and _qsl.numel() == _meta_n_reqs + 1
                        and int(_qsl[-1].item()) == _q_n):
                    # qsl is FA's authoritative cu_q boundary tensor;
                    # its row i corresponds to vLLM's input_batch row i,
                    # NOT to connector_meta.requests[i]. 2026-05-12 fix:
                    # use `self._rid_to_bt_row` (populated from
                    # input_batch.req_ids via set_input_batch_req_ids)
                    # to map each rid to its FA row. Pre-fix this
                    # enumerated connector_meta.requests order, which
                    # is `scheduled_new_reqs + scheduled_cached_reqs`
                    # append order — not FA's row order. The mismatch
                    # silently assigned the wrong Q-slice to each rid
                    # under multi-rid batched mode → wrong K-similarity
                    # in Score → wrong page selection → slot N fails
                    # deterministically. See docs/multi_rid_root_cause_2026-05-12.md.
                    _rid_to_bt_row_qsl = (
                        getattr(self, "_rid_to_bt_row", None) or {})
                    if _rid_to_bt_row_qsl:
                        for step_req in connector_meta.requests:
                            rid = step_req.request_id
                            fa_idx = _rid_to_bt_row_qsl.get(rid)
                            if fa_idx is None or fa_idx + 1 >= _qsl.numel():
                                continue
                            start = int(_qsl[fa_idx].item())
                            end = int(_qsl[fa_idx + 1].item())
                            if 0 <= start < end <= _q_n:
                                per_request_q[rid] = (
                                    quest_query[start:end])
                    else:
                        # Pre-fix fallback: no input_batch ordering
                        # plumbed through. Use legacy enumerate order
                        # — known buggy but preserves behavior for
                        # callers that haven't been migrated yet.
                        for req_idx, step_req in enumerate(connector_meta.requests):
                            start = int(_qsl[req_idx].item())
                            end = int(_qsl[req_idx + 1].item())
                            if 0 <= start < end <= _q_n:
                                per_request_q[step_req.request_id] = (
                                    quest_query[start:end])
                    _qsl_used = True
            except Exception as _qsl_err:
                logger.warning("[icms] qsl slicing failed: %s; "
                               "falling back to meta-tokens path",
                               _qsl_err)
                per_request_q = {}
                _qsl_used = False

            if not _qsl_used:
                # Legacy fallback paths (kept for robustness):
                _meta_total_tokens = sum(
                    max(0, sr.num_computed_tokens_end
                        - sr.num_computed_tokens_start)
                    for sr in connector_meta.requests)
                if _q_n == _meta_n_reqs:
                    # Decode mode: 1 token per rid, batch-ordered.
                    for req_idx, step_req in enumerate(connector_meta.requests):
                        per_request_q[step_req.request_id] = (
                            quest_query[req_idx:req_idx + 1])
                elif (_meta_total_tokens > 0
                        and _meta_total_tokens == _q_n):
                    # Prefill mode with valid token metadata.
                    offset = 0
                    for step_req in connector_meta.requests:
                        n_tokens = max(0, step_req.num_computed_tokens_end
                                       - step_req.num_computed_tokens_start)
                        if n_tokens > 0 and offset + n_tokens <= _q_n:
                            per_request_q[step_req.request_id] = (
                                quest_query[offset:offset + n_tokens])
                        offset += n_tokens
                # else: no slicing strategy fits. Audit #24 fix
                # (2026-05-11): rather than silently leaving
                # per_request_q empty (so `.get(rid, quest_query)`
                # falls back to full multi-rid Q for every rid —
                # the bug class that caused the 2026-05-07 batched
                # scoring collapse before the qsl path landed),
                # raise loudly. This is the BATCHED branch
                # (len(requests_to_score) > 1); the single-rid
                # path doesn't enter this block. Set
                # ICMS_ALLOW_QSLICE_FALLBACK=1 to opt back into
                # the legacy silent fallback if a future batched
                # workload trips this and you want to ship-it-and-
                # investigate-later (the diag-log warning above
                # still fires).
            if (not per_request_q
                    and quest_query is not None
                    and len(requests_to_score) > 1
                    and os.environ.get(
                        "ICMS_ALLOW_QSLICE_FALLBACK", "0") != "1"):
                raise RuntimeError(
                    f"[icms-qslice] no Q-slicing strategy fit in "
                    f"batched mode: _q_n={_q_n} _meta_n_reqs={_meta_n_reqs}"
                    f" _meta_total_tokens={_meta_total_tokens} "
                    f"requests_to_score={len(requests_to_score)} "
                    f"layer={next_layer_idx}. Silent fallback to "
                    f"full multi-rid Q would mis-score every rid "
                    f"(audit #24). Set "
                    f"ICMS_ALLOW_QSLICE_FALLBACK=1 to permit the "
                    f"legacy silent fallback."
                )
            # ICMS_DIAG_QSLICE=1: probe agent suspect #1 — per-rid Q
            # slicing silently falls back to full-batch Q if the slice
            # is dropped, causing rid X to score against ALL rids'
            # tokens. Logs which rids got a slice vs fell back, plus
            # the shape mismatch.
            if os.environ.get("ICMS_DIAG_QSLICE") == "1":
                _rids_in_score = [r for _, r, _ in requests_to_score]
                _has_slice = [r in per_request_q for r in _rids_in_score]
                _meta_total = sum(
                    max(0, sr.num_computed_tokens_end
                        - sr.num_computed_tokens_start)
                    for sr in connector_meta.requests)
                _q_total = quest_query.shape[0]
                _all_have = all(_has_slice)
                _any_drop = not _all_have
                if _any_drop or _meta_total != _q_total:
                    logger.warning(
                        "[diag-qslice] layer=%d rids_to_score=%d "
                        "have_slice=%s all_have=%s meta_tokens=%d "
                        "q_tokens=%d mismatch=%s",
                        next_layer_idx, len(_rids_in_score),
                        _has_slice, _all_have, _meta_total, _q_total,
                        _meta_total != _q_total)

        # Score each request with its own Q slice.
        # Async batching (2026-05-05): at TP=1 and N>=2 batched rids,
        # fire all N _score_one_request calls on a thread pool so their
        # underlying score RPCs actually overlap on the wire. The
        # async-capable IcmsClient (option 2 refactor) demuxes the N
        # replies by request_id, so each thread's `client.score()`
        # blocks only on its OWN reply rather than serializing through
        # the legacy per-RPC lock. Skipped at TP>1 because the
        # _score_one_request body issues NCCL AllGather + broadcast
        # collectives whose ordering must match across ranks — running
        # them concurrently from N threads would risk a deadlock.
        # 2026-05-12 ICMS_DISABLE_SCORE_ASYNC=1: force the serial dispatch
        # path even when conditions for async pool dispatch are met.
        # Used as a discriminating test for H1 (async-demux race) — if
        # the smoke passes with this flag set, the bug is in the async
        # path (race condition or pool-state contamination across rids).
        _disable_async = os.environ.get("ICMS_DISABLE_SCORE_ASYNC") == "1"
        if (not _disable_async
                and _allow_batch() and self._tp_size == 1
                and len(requests_to_score) >= 2
                and hasattr(self._client, "score_async")):
            pool = self._score_dispatch_pool
            futs = []
            for req_idx, rid, rs in requests_to_score:
                q_for_request = per_request_q.get(rid, quest_query)
                if (os.environ.get("ICMS_DIAG_QSLICE") == "1"
                        and per_request_q
                        and rid not in per_request_q):
                    _q_shape = (tuple(quest_query.shape)
                                if hasattr(quest_query, "shape") else None)
                    logger.warning(
                        "[diag-qslice-fallback] layer=%d rid=%s "
                        "fell back to full-batch Q (shape=%s) — "
                        "wrong page selection imminent",
                        next_layer_idx, rid, _q_shape)
                # 2026-05-12: log Q-tensor SHA per rid for slot-1
                # contamination investigation. Two rids with identical
                # Q SHA at the same layer == they're scoring against the
                # same tokens → cross-rid contamination at the score side.
                if (os.environ.get("ICMS_DIAG_Q_SHA") == "1"
                        and hasattr(q_for_request, "shape")
                        and hasattr(q_for_request, "cpu")):
                    try:
                        import hashlib as _hl
                        # bf16 → float32 first; numpy lacks bf16.
                        _qb = (q_for_request.contiguous().to(
                            torch.float32).cpu().numpy().tobytes())
                        _qsha = _hl.sha256(_qb).hexdigest()[:16]
                        _q0 = float(q_for_request.flatten()[0].item())
                        logger.info(
                            "[diag-q-sha] layer=%d rid=%s req_idx=%d "
                            "q_shape=%s q_first=%.6e q_sha=%s",
                            next_layer_idx, rid[:8], req_idx,
                            tuple(q_for_request.shape), _q0, _qsha)
                    except Exception as _eq:
                        logger.warning("diag-q-sha failed: %s", _eq)
                futs.append(pool.submit(
                    self._score_one_request,
                    rid, rs, req_idx, next_layer_idx, q_for_request,
                    budget, stats, connector_meta))
            # Block on every thread; the first exception (if any)
            # propagates to the caller — same semantics as the serial
            # loop's exception path.
            for f in futs:
                f.result()
        else:
            for req_idx, rid, rs in requests_to_score:
                q_for_request = per_request_q.get(rid, quest_query)
                if (os.environ.get("ICMS_DIAG_QSLICE") == "1"
                        and per_request_q
                        and rid not in per_request_q):
                    _q_shape = (tuple(quest_query.shape)
                                if hasattr(quest_query, "shape") else None)
                    logger.warning(
                        "[diag-qslice-fallback] layer=%d rid=%s "
                        "fell back to full-batch Q (shape=%s) — "
                        "wrong page selection imminent",
                        next_layer_idx, rid, _q_shape)
                # 2026-05-12: Q-SHA diag (mirror of the async branch above).
                if (os.environ.get("ICMS_DIAG_Q_SHA") == "1"
                        and hasattr(q_for_request, "shape")
                        and hasattr(q_for_request, "cpu")):
                    try:
                        import hashlib as _hl
                        # bf16 → float32 first; numpy lacks bf16.
                        _qb = (q_for_request.contiguous().to(
                            torch.float32).cpu().numpy().tobytes())
                        _qsha = _hl.sha256(_qb).hexdigest()[:16]
                        _q0 = float(q_for_request.flatten()[0].item())
                        logger.info(
                            "[diag-q-sha] layer=%d rid=%s req_idx=%d "
                            "q_shape=%s q_first=%.6e q_sha=%s",
                            next_layer_idx, rid[:8], req_idx,
                            tuple(q_for_request.shape), _q0, _qsha)
                    except Exception as _eq:
                        logger.warning("diag-q-sha failed: %s", _eq)
                self._score_one_request(
                    rid, rs, req_idx, next_layer_idx, q_for_request,
                    budget, stats, connector_meta)

        # ICMS_FETCH_ALL_POST_SCORE=1: after the per-request Score loop
        # populates pending_scores with the K-page sparse selection,
        # immediately follow up with a FetchAll for each request so the
        # sink + pending_scores end up holding ALL N pages of the chain.
        # This implements "sparse prefill score signal + dense decode":
        # Score still fires (its K-selection is logged as the icms_budget
        # marker; useful signal for offline analysis), but the sink
        # scattered to GPU is the full-N FetchAll result, so decode
        # attends over every page exactly like the no-ICMS dense
        # baseline. Block-table sizing is handled by the existing
        # _fetch_all_one_request reuse-promote path.
        #
        # Cost: one extra RPC per scoring boundary at layer 0 only —
        # subsequent boundaries early-return on rs._fetch_all_complete.
        #
        if self._fetch_all_post_score:
            if os.environ.get("ICMS_DIAG_FAPS") == "1":
                logger.info(
                    "[diag-faps] post-score fire layer=%d n_req=%d rids=%s",
                    next_layer_idx, len(requests_to_score),
                    [r[1][:8] for r in requests_to_score])
            for req_idx, rid, rs in requests_to_score:
                try:
                    self._fetch_all_one_request(
                        rid, rs, req_idx, next_layer_idx, budget, stats)
                except Exception:
                    logger.exception(
                        "ICMS_FETCH_ALL_POST_SCORE: fetch_all failed "
                        "for rid=%s layer=%d", rid, next_layer_idx)

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
        # 2026-05-12 cross-rid slot-1 investigation: log per-(rid, layer)
        # the score-state inputs that feed total_pages. If rid_5 (slot 1)
        # shows a smaller effective_groups than rids 4/6/7 in the same
        # layer, that's the smoking gun for "rid_5's stored_groups
        # truncated by an asymmetric _stored_chains snapshot".
        if os.environ.get("ICMS_DIAG_STORED", "0") == "1":
            logger.info(
                "[diag-stored] rid=%s layer=%d stored_groups=%d "
                "num_groups_written=%d effective_groups=%d "
                "chain_len=%d total_pages_will_be=%d",
                rid[:10], next_layer_idx,
                int(stored_groups), int(rs.num_groups_written),
                int(effective_groups), len(rs.chain) if rs.chain else 0,
                int(effective_groups * _GROUP_BLOCKS))
        # 2026-05-10 TP>1 stored_groups asymmetry fix:
        # `effective_groups` derives from per-rank state (num_groups_written
        # bumped by per-rank pipeline thread; stored_groups read from
        # per-rank `_stored_chain_groups` ledger). Pipeline progress is
        # async per rank, so at the FIRST Score call for a scored rid
        # that reads a chain still being filled by cold-writer rids,
        # ranks can disagree by tens of groups. Pre-fix: rank-0 (often
        # the lagging rank, observed empirically) computed total_pages
        # from a tiny effective_groups (e.g. 5 groups → 160 pages),
        # issued Score with k=16, broadcast that tiny reply to other
        # ranks, model attended to ~17 of intended ~100 pages → garbage
        # output ("1. The variables are: A, B, C, D, E, F" instead of
        # haystack content). Empirically TP=2 batched accuracy = 0.0.
        #
        # Fix: all-reduce-MAX effective_groups across TP ranks. The
        # shared GPU sink (CUDA-IPC'd) actually contains the data the
        # leader-rank wrote, so the lagging rank consuming the larger
        # value is safe — Score reply's sink_offsets cover the full
        # range. Cost: 1 i64 all-reduce per Score (~few μs); fires per
        # layer × per scored iter, dominated by the actual Score RPC.
        # Always all-reduce at TP>1 — must be unconditional for NCCL
        # collective participation symmetry. Skipping when local==0
        # would deadlock when one rank has work and another doesn't.
        if self._tp_size > 1:
            effective_groups = _tp_allreduce_max_int(
                effective_groups, self._tp_size)
        total_pages = effective_groups * _GROUP_BLOCKS

        # 2026-05-07 RCA log: ICMS_DIAG_CHAIN=1 dumps the client's view
        # of chain state at every Score call. Combined with the existing
        # `[diag-reply]` log + the implied server total (max page_id in
        # reply rounded up to GROUP_BLOCKS), we can detect divergence
        # where server thinks the chain has more groups than the client
        # does — symptom of pending writes still in deferred-write
        # pipeline at long context (mistral-nemo 128K).
        if os.environ.get("ICMS_DIAG_CHAIN") == "1":
            logger.info(
                "[diag-chain] PRE rid=%s layer=%d chain_len=%d "
                "stored_groups=%d num_groups_written=%d "
                "effective_groups=%d client_total_pages=%d",
                rid, next_layer_idx, len(rs.chain),
                stored_groups, rs.num_groups_written,
                effective_groups, total_pages,
            )
        # 2026-05-08 race-audit follow-up: per-Score chain-state lag
        # diagnostic. The agent's hypothesis is that pre-fix BLOCK_WRITES=0
        # batched runs hit Score with `flushed_local < num_groups_written`
        # (i.e., the deferred-write pipeline is behind the scheduler's
        # expected committed groups), so the server returns a valid
        # PREFIX of the chain and connector reads silent wrong-pages.
        # With the Step 1 fix (BLOCK_WRITES default-on for batched), every
        # Score should see lag=False. Counts kept in self.stats so we can
        # report a final tally; line-per-Score logs only when lag fires.
        if os.environ.get("ICMS_DIAG_CHAIN_LAG", "0") == "1":
            _flushed = int(getattr(rs, "flushed_local", -1))
            _expected = int(rs.num_groups_written)
            _lag = (_flushed >= 0) and (_flushed < _expected)
            if not hasattr(self, "_diag_chain_lag_count"):
                self._diag_chain_lag_count = 0
                self._diag_chain_lag_total = 0
            self._diag_chain_lag_total += 1
            if _lag:
                self._diag_chain_lag_count += 1
                logger.info(
                    "[diag-chain-lag] LAG rid=%s layer=%d chain_len=%d "
                    "flushed_local=%d num_groups_written=%d "
                    "stored_groups=%d (lag_count=%d/%d=%.1f%%)",
                    rid, next_layer_idx, len(rs.chain),
                    _flushed, _expected, stored_groups,
                    self._diag_chain_lag_count,
                    self._diag_chain_lag_total,
                    100.0 * self._diag_chain_lag_count
                    / max(1, self._diag_chain_lag_total))

        # First-turn / empty-chain short-circuit. No stored pages means
        # nothing to score and no sink to fill — skip everything,
        # including the TP>1 NCCL AllGather(Q) + broadcast(reply) path.
        # This is symmetric across ranks (total_pages is derived from
        # scheduler-propagated state), so all ranks return together.
        if total_pages == 0:
            return

        # M3: decode-mode short-circuit. If we're past prefill and this
        # stride group's bitmap is already full (every page is on the
        # client), there's nothing left to fetch — let decode run
        # sparse against the in-cache set without issuing a Score RPC.
        # _prefill_done is the worker-level gate; with max_num_seqs=1
        # this is also per-request. Multi-request decode would need
        # per-rs gating which is future work.
        is_decode = bool(self._prefill_done)
        # 2026-05-15 SPDD first-principles fix: override budget to 1.0
        # on the FIRST decode-iter scored layer when sparse-prefill-
        # dense-decode is enabled. The existing Score path then fires
        # with k=total_pages, server returns all N pages, fetched_pages
        # saturates, dense_mode flips — exactly as sparse Quest's
        # natural saturation iter does. Subsequent scored layers in
        # this decode iter hit the `is_decode and rs.dense_mode` early-
        # return below, so the override fires exactly once per rid.
        # The boundary fetch helper was removed in favor of this
        # in-control-flow override; sparse Quest's flow handles
        # everything else identically.
        if (is_decode
                and not rs.dense_mode
                and self._sparse_prefill_dense_decode):
            budget = 1.0
        # M4: once any prior decode-mode Score returned 0 net-new pages
        # for this rs, the bitmap is saturated — drop into dense decode
        # for the rest of the request and skip every subsequent Score
        # / Quest-hook on this rs. See update site below for the flip.
        if is_decode and rs.dense_mode:
            # M4 verification (ICMS_DIAG_DENSE=1): increment + log a
            # counter so we can confirm Score is *actually* skipped
            # post-flip rather than just trusting the early-return.
            if os.environ.get("ICMS_DIAG_DENSE") == "1":
                if not hasattr(rs, "_dense_skip_count"):
                    rs._dense_skip_count = 0
                rs._dense_skip_count += 1
                if rs._dense_skip_count <= 5 or rs._dense_skip_count % 50 == 0:
                    logger.info(
                        "[icms] dense_skip rid=%s layer=%d skip_count=%d",
                        rid, next_layer_idx, rs._dense_skip_count)
            return
        already_fetched = rs.fetched_pages.get(next_layer_idx, set())
        # Configurable flip threshold (2026-05-01). Default 1.0 = full
        # saturation (legacy). Set ICMS_DENSE_FLIP_FRAC=B (0<B<=1) to flip
        # once per-layer fetched_pages reaches B*total. Intended for the
        # sparse-decode case: combine with ICMS_DECODE_APPLY=0 so the
        # leftover trimmed bt from prefill apply persists post-flip and
        # decode reads a partial-context KV (≈B fraction). For mode (c)
        # (DECODE_APPLY=1) early flip is INCORRECT — natural-bt attention
        # post-flip reads unpopulated blocks. Stay at default 1.0 there.
        try:
            _flip_frac = float(os.environ.get("ICMS_DENSE_FLIP_FRAC", "1.0"))
        except ValueError:
            _flip_frac = 1.0
        _flip_frac = max(0.0, min(_flip_frac, 1.0))
        _flip_threshold = int(total_pages * _flip_frac)
        if _flip_frac < 1.0 and _flip_threshold < 1:
            _flip_threshold = 1
        # 2026-05-07 BUG FIX (sibling to M4 fix at line 3656): the
        # threshold flip below has the same per-layer-trigger /
        # per-rs-effect asymmetry as the M4 flip we already gated.
        # `already_fetched` is a single scored layer's bitmap, but
        # `rs.dense_mode = True` flips the WHOLE request to natural-bt
        # over all layers. With DECODE_APPLY=1, post-flip layers that
        # didn't reach the threshold still have unpopulated KV slots →
        # corrupted softmax. At default `_flip_frac=1.0` this rarely
        # bites because lockstep bitmap growth across scored layers
        # means they saturate together; at `_flip_frac<1.0` it bites
        # immediately. The doc-string above already warned to "Stay at
        # default 1.0 there" but the code didn't enforce it. Now we
        # gate explicitly: flip is only safe when DECODE_APPLY=0
        # (trim-bt persists post-flip) or when the bitmap is genuinely
        # saturated (every page on host for this layer).
        _flip_safe = (
            os.environ.get("ICMS_DECODE_APPLY", "1") == "0"
            or len(already_fetched) >= total_pages
        )
        if is_decode and len(already_fetched) >= _flip_threshold and _flip_safe:
            # Threshold reached (default = full saturation). Once dense_mode
            # flips, the Quest hook short-circuits at quest_hooks.py:458 and
            # wait_for_layer takes the cheaper early-return path
            # (DECODE_APPLY=0 short-circuit at line 3220-3221, OR all-dense
            # clear at 3246-3264 for DECODE_APPLY=1). With APPLY=0 the
            # leftover trimmed bt persists → genuine sparse decode.
            if not rs.dense_mode:
                rs.dense_mode = True
                rs._post_dense_iter = 0
                # Recompute the cached all-dense flag now that this rs
                # flipped. With max_num_seqs=1 typical case is a single rs
                # → cache becomes True immediately. Cheaper to recompute
                # here (once per flip) than to iterate _requests in every
                # per-layer hook call (×48 layers ×N decode iters).
                self._cached_all_dense = (bool(self._requests) and all(
                    getattr(r, "dense_mode", False)
                    for r in self._requests.values()))
                logger.info(
                    "[icms] dense_mode flip rid=%s layer=%d "
                    "(fetched=%d total=%d threshold=%d frac=%.3f)",
                    rid, next_layer_idx, len(already_fetched),
                    total_pages, _flip_threshold, _flip_frac)
                if os.environ.get("ICMS_DIAG_FULL") == "1":
                    self._diag_full_dense_flip_snapshot(rs, next_layer_idx)
            return

        # ICMS_ORIGINAL_QUEST=1 — isolated branch. Replaces the BF2 Score
        # RPC with a local per-(KV-head) Quest scorer running directly on
        # the GPU-side summaries we retained in record_page(). Default
        # path is byte-identical when env is unset; the rest of this
        # method (adaptive budget, Q AllGather, RPC fire, reply handling)
        # is skipped via the early `return` at the end of this branch.
        if os.environ.get("ICMS_ORIGINAL_QUEST", "0") == "1":
            self._quest_local_score_one_layer(
                rid, rs, next_layer_idx, quest_query, budget,
                total_pages, already_fetched)
            return

        # ICMS_QUEST_MODE=per_kv_head — server-fetched summaries +
        # GPU per-(KV-head) scoring + server-fetched union of selected
        # pages, via v8 wire opcodes 25/27/29. Mirrors the existing
        # ICMS_ORIGINAL_QUEST path (per-layer union → fed to vLLM's
        # PagedAttention as-is) but sources summaries from the storage
        # server instead of a local GPU stash, so it works for chains
        # written by other requests too.
        if os.environ.get("ICMS_QUEST_MODE", "") == "per_kv_head":
            self._quest_per_kv_head_score_one_layer(
                rid, rs, req_idx, next_layer_idx, quest_query, budget,
                total_pages, already_fetched)
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
        # M3: 10%-of-total floor in decode-mode so the bitmap fills in
        # ≤10 iters. The adaptive allocator can give more if there's
        # bandwidth slack (decode iters are short → typically gives
        # plenty), but the floor guarantees a minimum convergence rate.
        if is_decode and total_pages > 0:
            floor_k = (total_pages + 9) // 10  # ceil(0.10 * total_pages)
            k = max(k, floor_k)
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

        # Ceiling-snap redirect: when the adaptive allocator clamped budget
        # up to 1.0 (raw >= 0.95), routing through scoring would just waste
        # an RPC + summary-fetch RTT to ask for the top-(total_pages) pages
        # — which is every page. Redirect to the kFetchAll dispatch so the
        # server skips scoring entirely. Only fires at the first scored
        # layer of each request; subsequent strides see _fetch_all_complete
        # and reuse the cached chain on _pending_reuse.
        # M3: in decode, the bitmap-aware Score path is what we want even
        # at budget=1.0 — kFetchAll ignores the bitmap and would re-fetch
        # everything. Skip the redirect when decoding.
        # 2026-05-08: gate on transport support. The unix-socket and
        # shmem IcmsClient classes don't implement fetch_all (RDMA-only
        # path); previously this redirect silently failed leaving
        # _pending_scores empty → corrupted KV (qwen3 niah_multikey_2
        # b=1.0 = 0.067 vs 1.000 with shortcut off). Skip the redirect
        # for non-RDMA transports so we fall through to the Score path
        # below with k=total_pages, which gives the same "every page"
        # result via scoring.
        # Audit fix #18 (2026-05-10): broadcast the ceiling-snap
        # decision from rank 0 so all ranks dispatch to the same branch.
        # Without this, transient per-rank divergence in
        # `effective_budget` (one rank's _adaptive_allocator unregistered
        # a finished request a step earlier than the other → smaller
        # `total_demand_bps` → larger `_compute_share` → higher budget
        # → ceiling-snaps to 1.0 first) routes rank 0 into
        # `_fetch_all_one_request` while rank 1 stays in the Score path
        # → divergent NCCL collective shapes (Score's broadcast vs
        # FetchAll's broadcast on different participating ranks) → CUDA
        # hang. Cost: one i64 broadcast (~few μs) per Score call;
        # piggybacks on the same sep-NCCL group used by
        # `_tp_broadcast_score_reply` below. _tp_broadcast_bool's TP=1
        # path is a no-op (returns input unchanged), so this is free
        # outside TP>1.
        ceiling_snap = (effective_budget >= 1.0 and total_pages > 0
                         and not is_decode
                         and getattr(self, "_supports_fetch_all", False))
        ceiling_snap = _tp_broadcast_bool(
            ceiling_snap, self._tp_rank, self._tp_size)
        if ceiling_snap:
            try:
                self._fetch_all_one_request(
                    rid, rs, req_idx, next_layer_idx, effective_budget, stats)
                return  # skip scoring; FetchAll handles every page
            except Exception:
                logger.exception(
                    "ceiling-snap redirect to kFetchAll failed for rid=%s "
                    "layer=%d; falling back to scoring path",
                    rid, next_layer_idx)

        # Marshal query to fp32 numpy, mean-pooling Q heads per KV-head
        # group for GQA.
        geom = self._geom
        _diag_q_n = getattr(self, "_diag_q_shape_count", 0)
        if (isinstance(quest_query, torch.Tensor)
                and os.environ.get("ICMS_DIAG_Q_SHAPE", "0") == "1"
                and _diag_q_n < 30):
            self._diag_q_shape_count = _diag_q_n + 1
            try:
                logger.info(
                    "[diag-q-shape] n=%d rank=%d tp_size=%d rid=%s "
                    "layer=%d is_decode=%s prefill_done=%s q.shape=%s",
                    _diag_q_n, self._tp_rank, self._tp_size, rid,
                    next_layer_idx,
                    is_decode, getattr(self, "_prefill_done", "?"),
                    tuple(quest_query.shape))
            except Exception as _e:
                logger.warning("diag-q-shape failed: %s", _e)
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
                # 2026-05-11 audit P1 (initial hypothesis) + REVISION:
                # the original P1 fix assumed `quest_query` contained
                # haystack + question tokens, with q.mean diluting the
                # question's discriminative signal. Code-trace revision
                # showed otherwise for models with prefix_caching=True
                # (qwen3, mistral-small sparse path): Phase-1 prefills
                # the haystack, Phase-2 hits prefix cache on haystack
                # tokens so the scored-layer Quest hook fires only on
                # the QUESTION's new tokens. q.shape[0] = question
                # length (~50 tokens), not full prompt. So q.mean is
                # already the question-only mean.
                # For gemma-3 (prefix_caching=False due to multimodal
                # arch), q.shape[0] = full prompt. Question slicing OR
                # enabling prefix caching would be the right fix there.
                # Default 'mean' restored. `ICMS_Q_AGG=last` available
                # as a knob for diagnostic A/B testing on gemma-3.
                _q_agg = os.environ.get("ICMS_Q_AGG", "mean").lower()
                if q.shape[0] == 1:
                    q = q.squeeze(0)
                elif _q_agg == "last":
                    q = q[-1]
                else:  # default 'mean'
                    q = q.mean(dim=0)
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
                    grouped = q.reshape(
                        local_kv_heads, heads_per_group, head_dim)
                    pool = os.environ.get("ICMS_Q_GQA_POOL", "mean").lower()
                    if pool == "absmax":
                        # Pick the per-channel value with largest |q| across
                        # heads (sign preserved). Reduces dilution when query
                        # heads in a GQA group disagree on direction.
                        absg = grouped.abs()
                        idx = absg.argmax(dim=1, keepdim=True)
                        q = grouped.gather(dim=1, index=idx).squeeze(1)
                    elif pool == "max":
                        q = grouped.amax(dim=1)
                    else:  # default: mean
                        q = grouped.mean(dim=1)
                    # q: [local_kv_heads, head_dim]
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
                # 2026-05-10 audit fix #4: re-raise instead of falling
                # back to rank-local Q.  Pre-fix this swallowed the
                # exception, set q_np to the rank-local slice (wrong
                # shape for the server), then proceeded into the
                # rank-0-only RPC + reply broadcast.  When only ONE
                # rank's AllGather raised (e.g., NCCL transient), the
                # other rank ran normally → divergent Q across ranks →
                # silent corrupted page selection (same shape as the
                # 2026-05-10 batched-mode bug we just root-caused).
                # Surfacing the failure loudly via the connector's
                # existing exception propagation is correct: NCCL
                # collectives are bilateral, so a real transient will
                # also fail (or time out) on the peer rank — both ranks
                # then abort symmetrically rather than diverging.
                logger.error(
                    "Score: TP AllGather(Q) failed rank=%d layer=%d: %s "
                    "— re-raising. Silent rank-local fallback would "
                    "cause divergent Q across ranks and silently wrong "
                    "page selection.",
                    self._tp_rank, next_layer_idx, e, exc_info=True)
                raise

        # ── Q-vector hash trace (TP=1 vs TP=2 ranking diagnostic).
        # Logs hash + first 4 + last 4 values + l2 norm of the full Q
        # vector that gets sent to Score. At TP>1 this fires after
        # AllGather + cat (so rank 0 and 1 see the same Q). Compare
        # against TP=1 for the same prompt+layer to confirm whether the
        # Q vector itself differs (bf16 ULP non-determinism, GQA pool
        # ordering, AllGather concat order).
        if os.environ.get("ICMS_Q_HASH_TRACE") == "1":
            try:
                import hashlib as _hl_q
                _q_bytes = q_np.astype(np.float32, copy=False).tobytes()
                _q_hash = _hl_q.blake2b(_q_bytes, digest_size=8).hexdigest()
                _q_norm = float(np.linalg.norm(q_np))
                _q_head = [float(x) for x in q_np[:4].tolist()]
                _q_tail = [float(x) for x in q_np[-4:].tolist()]
                _icms_trace(
                    op="score_q_hash",
                    rid=rid,
                    layer=int(next_layer_idx),
                    extra={
                        "tp_rank": int(self._tp_rank),
                        "tp_size": int(self._tp_size),
                        "q_len": int(q_np.size),
                        "q_hash": _q_hash,
                        "q_norm": _q_norm,
                        "q_head4": _q_head,
                        "q_tail4": _q_tail,
                    },
                )
            except Exception as _qhe:
                logger.debug("score_q_hash trace failed: %s", _qhe)

        # Fire Score synchronously for v1. Use the shared sink (slot 0)
        # without the slot pool — in v1 we don't fetch KV back to GPU, so
        # we don't need per-request sink isolation. The score result is
        # just for measuring page selection accuracy.
        t_score_start = time.perf_counter()
        # Compute reuse range for this stride group.
        num_layers = self._geom.num_layers if self._geom else 48
        reuse_through = min(
            next_layer_idx + self._score_stride - 1, num_layers - 1)

        # ICMS_TRACE_FLAGS=1: snapshot flag state at the call site. The
        # actual clear happens inside rdma_client.py:193 (when
        # use_flags=True), not here — see Bug #1 fix below.
        if os.environ.get("ICMS_TRACE_FLAGS") == "1":
            try:
                snap = self._sink_pool.sink.snapshot_flags() if hasattr(
                    self._sink_pool.sink, "snapshot_flags") else None
                _t = time.perf_counter()
                _set_layers = ([i for i, v in enumerate(snap) if v]
                               if snap is not None else None)
                logger.info(
                    "[trace-flags] CLEAR site=score-impl t=%.6f rid=%s "
                    "layer=%d set_before=%s",
                    _t, rid, next_layer_idx, _set_layers)
            except Exception:
                pass
        # Bug #1 fix (race-audit 2026-05-08): see _fetch_all_one_request
        # for the full rationale. Connector-side clear removed; rdma_client
        # clears once at RPC entry. Unix-socket / mem-backend doesn't read
        # flags (wait_for_layer at line 5147 gates on flag_count > 0).

        logger.debug("Score: layer=%d reuse_through=%d chain_len=%d k=%d",
                      next_layer_idx, reuse_through, len(rs.chain), k)

        reply = None
        t_score_end = t_score_start
        try:
            # Option W: only rank 0 issues the Score RPC. Server's
            # drain-time fan-out replicates the sink bytes to every
            # peer rank's sink. Rank 0 then broadcasts the reply tuple
            # so every rank can populate _pending_scores identically.
            # M3: pack the decode-mode bitmap from rs.fetched_pages.
            # Empty bytes preserves prefill behavior (server's flag bit
            # stays 0). During decode this masks already-fetched page
            # IDs so they fall out of top-k server-side.
            fetch_bitmap = (
                _pack_fetch_bitmap(already_fetched, total_pages)
                if is_decode and already_fetched else b""
            )
            if self._tp_size > 1 and self._tp_rank != 0:
                reply = None
            else:
                # Adaptive-bandwidth pass-through. Both 0 if disabled.
                ab_demand_bps = 0
                ab_compute_supply_bps = 0
                if self._adaptive_allocator is not None:
                    ab_demand_bps = self._adaptive_allocator.demand_bps_for(rid)
                    ab_compute_supply_bps = (
                        self._adaptive_allocator.compute_supply_bps_for(rid))
                # 2026-05-07 BUG FIX: retry on `Score: no resolvable
                # groups` (status=ENOENT). Symptom at long context
                # (mistral-nemo 128K, ~25% rate): Score fires at scored
                # layers DURING prefill while WriteGroup acks for the
                # chain's first group are still racing through the
                # network/server pipeline → server's trie lookup
                # returns empty → ENOENT. Retrying after a brief sleep
                # gives the racing WriteGroup time to commit. Without
                # retry, the connector silently falls through to dense
                # for that layer → mixed sparse/dense produces
                # corrupted accuracy. Tunable via
                # ICMS_SCORE_RETRY_COUNT (default 3) and
                # ICMS_SCORE_RETRY_DELAY_MS (default 20ms).
                try:
                    _retry_count = int(
                        os.environ.get("ICMS_SCORE_RETRY_COUNT", "3"))
                    _retry_delay_s = float(
                        os.environ.get("ICMS_SCORE_RETRY_DELAY_MS",
                                       "20")) / 1000.0
                except ValueError:
                    _retry_count, _retry_delay_s = 3, 0.020

                _rk_chain = self._rank_chain(rs.chain)
                _score_kwargs = dict(
                    request_id=icms_rid,
                    chain=_rk_chain,
                    layer=next_layer_idx,
                    query=q_np,
                    k=k,
                    sink=self._sink_pool.sink,
                    reuse_through_layer=reuse_through,
                    use_flags=(os.environ.get("ICMS_REPLY_EARLY", "1") == "1"),
                    fetch_bitmap=fetch_bitmap,
                    demand_bps=ab_demand_bps,
                    compute_supply_bps=ab_compute_supply_bps,
                )
                # 2026-05-08 race-audit follow-up: log the chain prefix
                # actually sent to Score. The pids-hash diag showed
                # control returning sequential pages 0..k-1 starting at
                # layer 6 — strong signal that the SERVER's view of the
                # chain shrinks mid-prefill. This logs the client-side
                # chain length sent on the wire, plus a fingerprint of
                # the chain hashes (first/last entry). If the chain bytes
                # are stable across layers but server returns different
                # n_pids, the bug is server-side. If the chain bytes
                # SHRINK across layers, it's a client-side rs.chain
                # mutation race.
                if os.environ.get("ICMS_DIAG_SCORE_CHAIN", "0") == "1":
                    _ch_len = len(_rk_chain)
                    _ch_first = _rk_chain[0] if _rk_chain else None
                    _ch_last = _rk_chain[-1] if _rk_chain else None
                    logger.info(
                        "[diag-score-chain] rid=%s layer=%d k=%d "
                        "chain_len=%d ch_first=%s ch_last=%s "
                        "rs_chain_len=%d flushed_local=%d",
                        rid, next_layer_idx, k, _ch_len,
                        _ch_first, _ch_last, len(rs.chain),
                        int(getattr(rs, "flushed_local", -1)))
                _last_err = None
                reply = None
                # ICMS_TRACE: emit a `score_req` line just before the wire
                # send. Paired with `score_rep` on the reply side; lets a
                # diff harness map connector→server requests by (rid,
                # layer, ts_ns) and verify the chain length sent matches
                # what the server received.
                _icms_trace(
                    "score_req", rid, layer=next_layer_idx,
                    chain_fp=_icms_chain_fp(_rk_chain),
                    extra={
                        "requested_k": int(k),
                        "chain_len_local": len(rs.chain),
                        "num_layers": int(getattr(self._geom, "num_layers", -1))
                                       if self._geom is not None else -1,
                    })
                for _attempt in range(_retry_count + 1):
                    # 2026-05-10 audit #5 follow-up: snapshot flush_seq
                    # BEFORE the RPC so the post-failure wait observes
                    # any pulse that lands during the RPC (lost-wakeup
                    # safe — see flush_cond rationale on _RequestState).
                    _flush_seq_at_attempt = rs.flush_seq
                    try:
                        with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                            reply = self._client.score(**_score_kwargs)
                        break
                    except IcmsError as _e:
                        if _e.status != errno.ENOENT:
                            raise
                        _last_err = _e
                        if _attempt < _retry_count:
                            # Step 2 per-rid Condition + flush_seq
                            # (2026-05-09 → 2026-05-10 audit #5): wait
                            # for the next WriteGroup commit FOR THIS
                            # rid rather than blunt time.sleep. If the
                            # pipeline thread flushed a group AT ANY
                            # POINT after `_flush_seq_at_attempt` was
                            # captured above, the predicate is already
                            # true and wait_for returns immediately
                            # (closes the lost-wakeup hole the Event
                            # version had). If no flush happens, we
                            # fall back to the original sleep behavior
                            # via the timeout. Critical: we must NOT
                            # hold _rpc_lock across the wait (would
                            # block the pipeline's WriteGroup →
                            # deadlock); the `with self._rpc_lock`
                            # block above already released it on
                            # exception. The Condition's own lock is
                            # separate and not held by the producer
                            # outside its notify_all.
                            with rs.flush_cond:
                                rs.flush_cond.wait_for(
                                    lambda: (rs.flush_seq
                                             > _flush_seq_at_attempt),
                                    timeout=_retry_delay_s)
                            continue
                # ICMS_TRACE: emit `score_rep` immediately after the RPC
                # path resolves (success, ENOENT-soft-fail, or other).
                # Captures returned page count + first 16 page IDs +
                # status. Server-side `score_rep` is emitted right after
                # the reply is sent on the wire; the two should
                # bracket the wire RTT.
                if _ICMS_TRACE_ENABLED:
                    _trace_pids = []
                    _trace_status = -1
                    _trace_n = 0
                    _trace_pids_hash = ""
                    _trace_pids_sorted_hash = ""
                    if reply is not None:
                        try:
                            _all_pids = [int(p) for p in
                                         list(reply.page_ids)]
                            _trace_pids = _all_pids[:16]
                            _trace_n = len(_all_pids)
                            _trace_status = int(getattr(reply, "status", 0))
                            # Hash the full set so we can compare TP=1
                            # vs TP=2 ranking + recall at the same prompt.
                            import hashlib as _hl_pi
                            _b = ",".join(str(p)
                                           for p in _all_pids).encode("ascii")
                            _trace_pids_hash = _hl_pi.blake2b(
                                _b, digest_size=8).hexdigest()
                            _bs = ",".join(
                                str(p) for p in sorted(_all_pids)
                            ).encode("ascii")
                            _trace_pids_sorted_hash = _hl_pi.blake2b(
                                _bs, digest_size=8).hexdigest()
                        except Exception:
                            pass
                    _icms_trace(
                        "score_rep", rid, layer=next_layer_idx,
                        chain_fp=_icms_chain_fp(_rk_chain),
                        extra={
                            "returned_page_count": _trace_n,
                            "returned_page_ids": _trace_pids,
                            "returned_page_ids_hash": _trace_pids_hash,
                            "returned_page_ids_sorted_hash":
                                _trace_pids_sorted_hash,
                            "tp_rank": int(self._tp_rank),
                            "tp_size": int(self._tp_size),
                            "status": _trace_status,
                            "soft_fail_enoent": (reply is None
                                                 and _last_err is not None),
                        })
                # All retries exhausted on ENOENT: log loudly + leave
                # reply=None so this Score's contribution is dropped
                # (instead of silently dense). Earlier code raised here;
                # we now soft-fail because ENOENT can be a transient
                # race during prefill that retrying eventually resolves.
                if reply is None and _last_err is not None:
                    if os.environ.get(
                            "ICMS_SCORE_RETRY_FAIL_LOUD", "0") == "1":
                        raise _last_err
                    logger.warning(
                        "Score RPC ENOENT rank=%d layer=%d rid=%s "
                        "chain_len=%d k=%d total_pages=%d — exhausted "
                        "%d retries (×%.0fms), dropping this Score.",
                        self._tp_rank, next_layer_idx, rid,
                        len(rs.chain), k, total_pages,
                        _retry_count, _retry_delay_s * 1000)
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
        # ICMS_STRICT_ASSERTIONS=1: same silent-state-empty guard as
        # _fetch_all_one_request, but with a soft-fail allowlist. Score
        # has a documented soft-fail on ENOENT-after-retries (see
        # ICMS_SCORE_RETRY_FAIL_LOUD comment ~line 3850); that's a
        # transient race during prefill, not a bug. Crash only when
        # reply is None for an UNDOCUMENTED reason — i.e., no _last_err
        # set, which means we hit the outer except (real exception, not
        # ENOENT-soft-fail). Catches missing methods, broadcast bugs,
        # silent transport failures.
        #
        # 2026-05-11 audit #23: broadcast the soft-fail flag so every
        # rank reaches the same decision. Pre-fix: `_last_err` was
        # only assigned inside the rank-0 RPC branch (line ~4591 else),
        # so non-rank-0 ranks always saw `_soft_failed=False`. Under
        # `ICMS_STRICT_ASSERTIONS=1` with a soft-fail ENOENT on rank-0,
        # rank-0 would swallow (soft_failed=True) but rank-1 would
        # raise (soft_failed=False) → asymmetric crash → engine hang.
        # Post-fix: rank-0's soft-fail bit is broadcast over the same
        # NCCL group as the reply, every rank takes the same branch.
        _strict = os.environ.get("ICMS_STRICT_ASSERTIONS", "0") == "1"
        _soft_failed_local = (locals().get("_last_err") is not None)
        if self._tp_size > 1:
            try:
                _soft_failed = _tp_broadcast_bool(
                    _soft_failed_local, self._tp_rank, self._tp_size)
            except Exception as _bcast_e:
                logger.warning(
                    "Score: soft-fail broadcast failed (%s) — falling back "
                    "to local value; strict-mode may raise asymmetrically",
                    _bcast_e)
                _soft_failed = _soft_failed_local
        else:
            _soft_failed = _soft_failed_local
        if (_strict and total_pages > 0 and reply is None
                and not _soft_failed):
            raise RuntimeError(
                f"Score returned empty reply with no soft-fail reason; "
                f"total_pages={total_pages} client="
                f"{type(self._client).__name__} rid={rid} "
                f"layer={next_layer_idx} tp_rank={self._tp_rank} k={k}. "
                f"Check warning logs for the underlying exception."
            )
        # Item B (2026-05-09): ALSO honor a non-zero `reply.status` from
        # the wire. Today the C++ rdma_client native binding strips
        # status before returning to Python (returns None on any non-
        # zero), so this branch only fires for the unix-socket
        # IcmsClient (whose `_parse_score_reply` raises on nonzero) or
        # for a future binding that surfaces status. Treating
        # non-zero status as a hard failure when strict, and as a
        # logged-loud warning when not, ensures the bench can't
        # silently consume corrupted KV from a server-side soft error
        # (e.g. a transient summary-cache miss returning empty top-k).
        # See project_e_status_plumbing_design_2026-05-08.md for the
        # full plumbing scope; this is the connector-side hook.
        if reply is not None and getattr(reply, "status", 0) != 0:
            _status = int(reply.status)
            _fail_loud = os.environ.get(
                "ICMS_SCORE_RETRY_FAIL_LOUD", "0") == "1"
            if _strict or _fail_loud:
                raise RuntimeError(
                    f"Score reply has non-zero status={_status}; "
                    f"client={type(self._client).__name__} rid={rid} "
                    f"layer={next_layer_idx} tp_rank={self._tp_rank} "
                    f"page_ids_len={len(reply.page_ids)}. "
                    f"Server returned a soft error — discarding silently "
                    f"would corrupt KV.")
            logger.warning(
                "Score reply non-zero status=%d rank=%d layer=%d rid=%s "
                "client=%s — dropping reply (set ICMS_STRICT_ASSERTIONS=1 "
                "or ICMS_SCORE_RETRY_FAIL_LOUD=1 to crash instead).",
                _status, self._tp_rank, next_layer_idx, rid,
                type(self._client).__name__)
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
            if (os.environ.get("ICMS_DIAG_TP_SINK", "0") == "1"
                    and reply.sink_offsets):
                try:
                    _off0 = int(reply.sink_offsets[0])
                    _peek = bytes(sink.mm[_off0:_off0 + 16]).hex()
                    _sink_name = getattr(sink, "name", "?")
                    _sink_id = getattr(sink, "sink_id", -1)
                    logger.info(
                        "[diag-tp-sink-read] rank=%d pid=%d rid=%s layer=%d "
                        "n_pids=%d n_offsets=%d off0=%d sink_kv_bytes=%d "
                        "shm_name=%s sink_id=%d head16=%s",
                        self._tp_rank, os.getpid(), rid, next_layer_idx,
                        len(reply.page_ids), len(reply.sink_offsets),
                        _off0, sink._kv_data_size,
                        _sink_name, _sink_id, _peek)
                except Exception as _diag_e:
                    logger.warning(
                        "[diag-tp-sink-read] peek failed: %s", _diag_e)
            # First-Score chain-state diag: log chain length, flushed_local,
            # and any per-rid bookkeeping that affects what Score can see.
            # If chain doesn't cover all pages of the prefill, server can
            # only score a prefix → returned page IDs cap at chain-len.
            if (os.environ.get("ICMS_DIAG_PAGE_IDS", "")
                    and not getattr(self, "_diag_chain_state_fired", False)):
                self._diag_chain_state_fired = True
                try:
                    logger.info(
                        "[diag-chain-state] rank=%d tp_size=%d rid=%s "
                        "layer=%d chain_len=%d flushed_local=%d "
                        "stored_groups=%d num_groups_written=%d "
                        "k_requested=%d total_pages=%d",
                        self._tp_rank, self._tp_size, rid, next_layer_idx,
                        len(rs.chain), int(getattr(rs, "flushed_local", -1)),
                        int(getattr(rs, "stored_groups", -1)),
                        int(getattr(rs, "num_groups_written", -1)),
                        k, total_pages)
                except Exception as _e:
                    logger.warning("diag-chain-state failed: %s", _e)
            # 2026-05-08 race-audit follow-up: per-Score page-ID hash
            # diagnostic (compact, every Score). Lets us compare control
            # vs step1fix runs at the same (rid, layer) — if the hash
            # differs systematically, the bug is upstream of apply
            # (Score state); if hashes match but accuracy differs, the
            # bug is in apply / attention compute. Set ICMS_DIAG_PIDS_HASH=1
            # to emit one-line-per-Score; cheap.
            if os.environ.get("ICMS_DIAG_PIDS_HASH", "0") == "1":
                _pids = sorted(int(p) for p in reply.page_ids)
                _h = 0
                for _p in _pids:
                    _h = (_h * 31 + _p) & 0xFFFFFFFFFFFFFFFF
                logger.info(
                    "[diag-pids-hash] rid=%s layer=%d k=%d n_pids=%d "
                    "head=%s tail=%s sorted_hash=%016x",
                    rid, next_layer_idx, k, len(_pids),
                    _pids[:4], _pids[-4:], _h)
            # First-Score page_ids fingerprint + full dump. Lets us
            # compare which pages Score selected across TP=1 / TP=2.
            # Dumps the FULL set of page_ids to a JSON file so we can
            # check membership of any specific page (e.g. the needle's
            # page) without parsing log lines.
            _diag_pids_path = os.environ.get("ICMS_DIAG_PAGE_IDS", "")
            if (_diag_pids_path
                    and not getattr(self, "_diag_page_ids_fired", False)):
                self._diag_page_ids_fired = True
                try:
                    import json as _json
                    _pids = list(int(p) for p in reply.page_ids)
                    _sorted_pids = sorted(_pids)
                    _hash = 0
                    for _p in _sorted_pids:
                        _hash = (_hash * 31 + _p) & 0xFFFFFFFFFFFFFFFF
                    out_path = (_diag_pids_path
                                if _diag_pids_path != "1"
                                else f"/tmp/icms_diag_pids_tp{self._tp_size}_"
                                     f"r{self._tp_rank}_pid{os.getpid()}.json")
                    # Also dump scores for distribution analysis.
                    _scores = [float(s) for s in reply.scores]
                    _pid_score = list(zip(_pids, _scores))
                    with open(out_path, "w") as f:
                        _json.dump({
                            "tp_rank": self._tp_rank,
                            "tp_size": self._tp_size,
                            "rid": rid,
                            "layer": next_layer_idx,
                            "n_pids": len(_pids),
                            "pids_sorted_hash": f"{_hash:016x}",
                            "pids_sorted": _sorted_pids,
                            "pid_score_unsorted": _pid_score,
                        }, f)
                    logger.info(
                        "[diag-page-ids] rank=%d tp_size=%d rid=%s "
                        "layer=%d n_pids=%d head[:8]=%s tail[:8]=%s "
                        "sorted_hash=%016x dumped=%s",
                        self._tp_rank, self._tp_size, rid, next_layer_idx,
                        len(_pids), _pids[:8], _pids[-8:], _hash, out_path)
                except Exception as _e:
                    logger.warning("diag-page-ids failed: %s", _e)
            # ICMS_DIAG_FULLTRACE: emit per-(rid, layer) score-reply
            # summary. Includes full sorted page_ids list (capped to
            # 512 entries — at h=32k b=0.20 the top-K is ~400 pages,
            # so this captures the full set) so we can diff slot-1's
            # selection against slot-0/2/3.
            if _ICMS_FULLTRACE_ENABLED:
                try:
                    _pids_ft = [int(p) for p in reply.page_ids]
                    _scores_ft = [float(s) for s in reply.scores]
                    _icms_fulltrace(
                        "score", rid=rid, layer=int(next_layer_idx),
                        k_requested=int(k),
                        total_pages=int(total_pages),
                        n_pids=len(_pids_ft),
                        page_ids_unsorted=_pids_ft[:512],
                        page_ids_sorted_head=sorted(_pids_ft)[:8],
                        page_ids_sorted_tail=sorted(_pids_ft)[-8:],
                        scores_head=_scores_ft[:8],
                        scores_tail=_scores_ft[-8:],
                        chain_len=int(len(rs.chain) if rs.chain else 0),
                        stored_groups=int(getattr(rs, "stored_groups", -1)),
                        num_groups_written=int(getattr(
                            rs, "num_groups_written", -1)),
                        flushed_local=int(getattr(rs, "flushed_local", -1)),
                        cache_hit=int(getattr(reply, "cache_hit", 0)),
                        reply_chain_groups=int(getattr(
                            reply, "matched_chain_groups", -1)),
                    )
                except Exception:
                    pass
            # Stash storage-side concurrent request count for adaptive budget.
            rs._last_storage_concurrent = getattr(
                reply, 'concurrent_requests', 0)
            # Adaptive-bandwidth: feed the storage-side effective supply
            # back into the allocator so the next stride's get_budget()
            # takes min(compute, storage). 0 ⇒ adaptive off server-side
            # (no-op stash).
            if self._adaptive_allocator is not None:
                self._adaptive_allocator.apply_storage_supply(
                    rid, int(getattr(reply, 'effective_supply_bps', 0) or 0))
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
            # ICMS_DIAG_SCORE_DUMP=<dir>: dump Q + per-page kmin/kmax
            # summaries + server-picked page IDs + per-page scores so the
            # user can replay alternate scoring algorithms offline without
            # re-running the bench. Runs in the regular sparse path (not
            # ICMS_ORIGINAL_QUEST); summaries come from record_page which
            # now also populates them under this flag (see ~line 9794).
            # Per-process file cap via ICMS_DIAG_SCORE_DUMP_LIMIT (default
            # 256) so a multi-task/multi-budget run can't fill disk.
            # Rank-gated to tp_rank==0 since per-rank KV-head slicing
            # makes cross-rank summaries hard to reconstruct offline.
            _dump_dir = os.environ.get("ICMS_DIAG_SCORE_DUMP", "")
            _picked_only_stash = os.environ.get(
                "ICMS_DIAG_SCORE_DUMP_PICKED_ONLY", "") in (
                "1", "true", "True")
            if _dump_dir and int(self._tp_rank) == 0:
                # Always stash q for this (rid, layer) so the per-rid
                # summaries dump (in on_request_finished) can join q
                # with the kmin/kmax that record_page populates AFTER
                # the forward pass completes. Independent of the
                # per-layer file write below — even when that's
                # skipped (cap hit, exception, etc.), the q stash
                # still gives us complete data via the summaries dump.
                # This is the 2026-05-14 fix for the "rid=1 had layer
                # dumps but other rids didn't" data-collection bug.
                # PICKED_ONLY=1: stash a 1-byte sentinel instead of the
                # full Q tensor (~250 MB for 32k tokens × 32 heads ×
                # 128 dims bf16). The on_request_finished dump still
                # needs `last_q_by_layer` to be non-empty to fire for
                # generate rids, but the tensor content is irrelevant
                # when only membership analysis runs downstream.
                if isinstance(quest_query, torch.Tensor):
                    try:
                        if _picked_only_stash:
                            rs.last_q_by_layer[int(next_layer_idx)] = (
                                torch.zeros(1, dtype=torch.uint8))
                        else:
                            rs.last_q_by_layer[int(next_layer_idx)] = (
                                quest_query.detach().cpu())
                    except Exception as _e_qstash:
                        logger.warning(
                            "[icms_diag_score_dump] q-stash failed "
                            "layer=%d rid=%s: %s",
                            next_layer_idx, rid, _e_qstash)
                # Also stash Score reply contents per layer — picked
                # page IDs and the server's score for each, so the
                # per-rid summaries dump has complete data even when
                # the per-layer .pt write fails for some rids.
                try:
                    rs.last_picked_by_layer[int(next_layer_idx)] = [
                        int(p) for p in reply.page_ids]
                    rs.last_scores_by_layer[int(next_layer_idx)] = (
                        [float(s) for s in reply.scores]
                        if reply.scores is not None else [])
                except Exception as _e_picked:
                    logger.warning(
                        "[icms_diag_score_dump] picked-stash failed "
                        "layer=%d rid=%s: %s",
                        next_layer_idx, rid, _e_picked)
                try:
                    _limit = int(os.environ.get(
                        "ICMS_DIAG_SCORE_DUMP_LIMIT", "256"))
                except ValueError:
                    _limit = 256
                _n_dumped = getattr(self, "_diag_score_dump_n", 0)
                # PICKED_ONLY=1 skips the per-(rid, layer) legacy
                # files entirely — they're ~75 MB each (kmin/kmax for
                # the full haystack) and aren't needed for membership
                # analysis. The per-rid v3 summaries dump (in
                # on_request_finished) carries picked_by_layer which
                # is the only thing membership needs.
                if _picked_only_stash:
                    _n_dumped = _limit  # short-circuit
                if _n_dumped < _limit:
                    try:
                        import os as _os
                        _os.makedirs(_dump_dir, exist_ok=True)
                        _per_layer = rs.quest_gpu_summaries.get(
                            next_layer_idx)
                        if _per_layer:
                            _items = sorted(_per_layer, key=lambda t: t[0])
                            _items = _items[:total_pages]
                            _kmin = torch.stack(
                                [m for _, m, _ in _items], dim=0)
                            _kmax = torch.stack(
                                [m for _, _, m in _items], dim=0)
                            _abs_pids = [int(p) for p, _, _ in _items]
                        else:
                            _kmin = None
                            _kmax = None
                            _abs_pids = []
                        _safe_rid = str(rid).replace("/", "_")
                        _path = _os.path.join(
                            _dump_dir,
                            f"{_safe_rid}_layer{next_layer_idx:02d}.pt")
                        _payload = {
                            "rid": str(rid),
                            "layer": int(next_layer_idx),
                            "q": (quest_query.detach().cpu()
                                  if isinstance(quest_query, torch.Tensor)
                                  else None),
                            "kmin": (_kmin.detach().cpu()
                                     if _kmin is not None else None),
                            "kmax": (_kmax.detach().cpu()
                                     if _kmax is not None else None),
                            "summary_abs_pids": _abs_pids,
                            "total_pages": int(total_pages),
                            "k": int(k),
                            "budget": float(effective_budget),
                            "picked_page_ids": [int(p) for p in
                                                reply.page_ids],
                            "server_scores": [float(s) for s in
                                              reply.scores]
                                if reply.scores is not None else [],
                            "tp_rank": int(self._tp_rank),
                            "tp_size": int(self._tp_size),
                        }
                        torch.save(_payload, _path)
                        self._diag_score_dump_n = _n_dumped + 1
                    except Exception as _e:
                        logger.warning(
                            "[icms_diag_score_dump] save failed at "
                            "layer %d rid=%s: %s",
                            next_layer_idx, rid, _e)
            # M2: prime the decode-mode fetched-pages set for this
            # stride group. After prefill completes this dict will hold
            # exactly the pages on host (per scored layer); M3 reads it
            # to build the bitmap wire suffix on decode-mode Score
            # calls. No behavior change for prefill.
            #
            # M4: detect saturation. In decode mode, if the bitmap-
            # filtered Score returned no net-new pages for this stride
            # group, the server has nothing left to give — flip the
            # whole rs into dense mode so subsequent Score / Quest-hook
            # calls short-circuit. Adaptive to chain growth: as long as
            # any Score keeps returning new pages we stay sparse.
            page_set = rs.fetched_pages.setdefault(next_layer_idx, set())
            prev_size = len(page_set)
            if reply.page_ids:
                page_set.update(reply.page_ids)
            # 2026-05-07 BUG FIX: server-truth chain-state sync.
            # Symptom: at long context (mistral-nemo 128K), the server's
            # trie has +1 more committed group than the connector's
            # `effective_groups` reports, because Phase 1's deferred-
            # write pipeline still has 1+ writes in flight when Phase
            # 2's first Score arrives (request_finished's drain timeout
            # fires before all writes ack). Server scores the larger
            # chain and returns page_ids beyond client's `total_pages`.
            # The client's bitmap (sized to client's total_pages) clamps
            # those pids → server returns them again on the next Score
            # → delta_new=0 dups=k loop, eventually bitmap looks
            # saturated to M4. Fix: when the reply's max page_id implies
            # a longer chain than `effective_groups`, update
            # rs.stored_groups so the next Score sizes its bitmap
            # correctly. This is one-way (only grows stored_groups);
            # benign if server happens to return only in-range pids.
            if reply.page_ids:
                _max_pid = max(reply.page_ids)
                _implied_groups = (_max_pid // _GROUP_BLOCKS) + 1
                if _implied_groups > effective_groups:
                    _old_stored = rs.stored_groups
                    # 2026-05-09: clamp at len(rs.chain). Without this,
                    # a server returning a page_id from a leaked / stale
                    # chain (e.g. previous rid not yet evicted) bumps
                    # stored_groups past chain length. Downstream
                    # num_groups_written follows (line ~7013), then the
                    # extract pipeline can construct a buffer for an
                    # out-of-chain group_idx, eventually flushing it
                    # non-partial → flushed_local > len(chain) → trips the
                    # _record_stored_groups assert (line 2272). Original
                    # crash: gemma-3 sparse 32K rid=29-a4a35d7c. See
                    # docs/gemma3_assert_investigation_2026-05-09.md.
                    _chain_len = len(rs.chain) if rs.chain else _implied_groups
                    rs.stored_groups = max(rs.stored_groups,
                                            min(_implied_groups, _chain_len))
                    if (rs.stored_groups != _old_stored
                            and os.environ.get("ICMS_DIAG_CHAIN") == "1"):
                        logger.info(
                            "[icms-sync] rid=%s layer=%d stored_groups "
                            "%d -> %d (server chain has %d groups, "
                            "client had effective_groups=%d, "
                            "clamped_at_chain_len=%d)",
                            rid, next_layer_idx, _old_stored,
                            rs.stored_groups, _implied_groups,
                            effective_groups, _chain_len,
                        )
                    # Loud one-time WARNING when the server overshoots
                    # past chain_len — surfaces the underlying server
                    # leak even though we clamped the client side.
                    if (_implied_groups > _chain_len
                            and not getattr(rs, "_overshoot_warned", False)):
                        logger.warning(
                            "[icms-sync] rid=%s server returned "
                            "max_pid=%d (implies %d groups) but client "
                            "chain has only %d groups; clamping. "
                            "Likely a stale-chain leak server-side; "
                            "investigate if this fires repeatedly.",
                            rid, _max_pid, _implied_groups, _chain_len)
                        rs._overshoot_warned = True
            # ICMS_DIAG_REPLY: log per-Score-reply size + delta added to
            # fetched_pages. If reply size > delta, server returned pages
            # already in client's bitmap = bitmap filtering is broken.
            # If reply size < k (cap), server's score+filter returned fewer
            # candidates than requested.
            if (os.environ.get("ICMS_DIAG_REPLY") == "1"
                    or os.environ.get("ICMS_DIAG_CHAIN") == "1") and is_decode:
                _reply_n = len(reply.page_ids) if reply.page_ids else 0
                _delta = len(page_set) - prev_size
                _dups = _reply_n - _delta
                _reply_pids_h = (list(reply.page_ids[:5])
                                  if reply.page_ids else [])
                _reply_pids_t = (list(reply.page_ids[-5:])
                                  if reply.page_ids and len(reply.page_ids) > 5
                                  else [])
                # 2026-05-07: derive server's implied chain length from
                # max(reply.page_ids). If max_pid >= total_pages, the
                # server has more groups than the client thinks → root
                # of the bitmap-filter dups bug (client can't represent
                # those pids in the next bitmap; server returns them
                # again next call).
                _max_pid = (max(reply.page_ids) if reply.page_ids else -1)
                _implied_server_groups = (
                    (_max_pid // _GROUP_BLOCKS) + 1 if _max_pid >= 0 else 0)
                _implied_server_total = _implied_server_groups * _GROUP_BLOCKS
                _divergence = (
                    "DIVERGENCE" if _implied_server_total > total_pages
                    else "ok")
                logger.info(
                    "[diag-reply] rid=%s layer=%d k=%d client_total=%d "
                    "max_reply_pid=%d implied_server_total=%d %s "
                    "reply_n=%d prev_fetched=%d post_fetched=%d "
                    "delta_new=%d dups=%d reply_pids[:5]=%s "
                    "reply_pids[-5:]=%s",
                    rid, next_layer_idx, k, total_pages,
                    _max_pid, _implied_server_total, _divergence,
                    _reply_n, prev_size, len(page_set),
                    _delta, _dups, _reply_pids_h, _reply_pids_t)
            # 2026-05-07 BUG FIX: the M4 flip below was firing per-layer
            # but setting rs.dense_mode globally for the whole request.
            # In DECODE_APPLY=1 (default), that meant other layers' still-
            # unpopulated slots got read by post-flip natural-bt attention
            # → softmax sees stale/zero KV → garbage outputs. Symptom:
            # vt_h32000 b=0.05 produces 0.125 vs FAPS dense's 0.99.
            #
            # The flip is safe only when ALL pages are populated (then
            # the threshold flip path above handles it cleanly), or in
            # DECODE_APPLY=0 mode (trim-bt persists post-flip; no read
            # from unpopulated slots).
            #
            # Restoring the M4 flip requires either (a) gating on every
            # scored layer also being saturated, or (b) building a
            # populated-slots block table at flip time. Neither is in
            # place yet. Until then, gate the flip on
            # ICMS_M4_EARLY_FLIP=1 (opt-in) so accidental users don't
            # silently get bad accuracy. The threshold flip at default
            # frac=1.0 still fires when the bitmap is genuinely full.
            _m4_flip_enabled = (
                os.environ.get("ICMS_M4_EARLY_FLIP", "0") == "1"
                or os.environ.get("ICMS_DECODE_APPLY", "1") == "0"
            )
            if (is_decode and not rs.dense_mode and len(page_set) == prev_size
                    and _m4_flip_enabled):
                rs.dense_mode = True
                # Mark the request as just-flipped so the next forward
                # pass logs full metadata (gated by ICMS_DIAG_FULL=1).
                rs._post_dense_iter = 0
                # Recompute the all-dense cache (paired with the
                # saturation-flip site above and on_step_start invalidate).
                self._cached_all_dense = (bool(self._requests) and all(
                    getattr(r, "dense_mode", False)
                    for r in self._requests.values()))
                # Track decode-iter at which the flip fired (1-based).
                # Counter is bumped in `wait_for_pending_writes` (called
                # once per forward = once per decode token after prefill).
                _flip_iter = getattr(rs, "_decode_iter_count", -1)
                logger.info(
                    "[icms] dense_mode flip rid=%s layer=%d fetched=%d "
                    "decode_iter=%d total_pages=%d "
                    "(no net-new pages from bitmap-filtered Score)",
                    rid, next_layer_idx, prev_size,
                    _flip_iter, total_pages,
                )
                if os.environ.get("ICMS_DIAG_FULL") == "1":
                    self._diag_full_dense_flip_snapshot(rs, next_layer_idx)
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
            # Bug fix (2026-04-30): in FAPS mode, the FetchAll slow path
            # (line ~2346) and the Score-then-promote fast path (line
            # ~2230) are both responsible for writing _pending_scores
            # for this layer with the N-page reply. If Score writes its
            # K-page reply here too, there's a TOCTOU window where
            # wait_for_layer can read Score's smaller reply before FAPS
            # overwrites it. The downstream cache (_apply_cached_actual_k)
            # then snapshots K, and fast-path layers 1..(stride-1) read
            # the wrong sink offsets (delta*K instead of delta*N). At
            # B=0.2 (K=51, N=256) the resulting per-layer reads land in
            # incoherent slices of layer 0's region → degenerate output.
            # Skip the Score write under FAPS — let FAPS be sole writer.
            if not self._fetch_all_post_score:
                with self._score_lock:
                    self._assert_pending_scores_no_clobber(
                        attn_layer_name, rid,
                        source="score-reply-landing")
                    self._pending_scores.setdefault(attn_layer_name, {})[rid] = (
                        reply, req_idx)

            # Store references for reuse layers. Server's per-layer
            # stride is actual_k*kv_page_bytes (NOT self._k — that's
            # the sink cap). Using self._k here would overshoot the
            # server's layer-delta offsets by a large factor.
            #
            # Skip in ICMS_FETCH_ALL_POST_SCORE mode: FetchAll's slow
            # path at layer 0 already populated _pending_reuse[1..47]
            # with the N-page reply for ALL reuse layers. Pre-populating
            # here would clobber those entries for layers within this
            # stride window with Score's smaller K-page reply, leaving
            # the reuse layers' apply scattering only K pages while the
            # stride boundary's apply gets the full N — i.e., 30/48
            # layers under-fill main_key. Bug 11 family fix (2026-04-29).
            if not self._fetch_all_post_score:
                actual_k = len(reply.page_ids)
                per_layer_bytes = actual_k * self._geom.kv_page_bytes
                for delta in range(1, reuse_through - next_layer_idx + 1):
                    reuse_layer = next_layer_idx + delta
                    reuse_attn = f"model.layers.{reuse_layer}.self_attn.attn"
                    reuse_offsets = [off + delta * per_layer_bytes
                                     for off in reply.sink_offsets]
                    with self._score_lock:
                        # 3-tuple carries req_idx (multi-rid fix,
                        # 2026-05-05). See companion site at line ~2569.
                        self._pending_reuse.setdefault(reuse_attn, {})[rid] = (
                            reply, reuse_offsets, req_idx)
            elif os.environ.get("ICMS_DIAG_FAPS") == "1":
                logger.info(
                    "[diag-faps] score-path FAPS-gate skip rid=%s layer=%d "
                    "k=%d (would-have-set reuse_layers=%d)",
                    rid[:8], next_layer_idx, len(reply.page_ids),
                    reuse_through - next_layer_idx)
        except Exception as e:
            t_score_end = time.perf_counter()
            self.stats.record_score(
                (t_score_end - t_score_start) * 1e6,
                False, [], next_layer_idx,
            )
            logger.debug("Score failed layer %d: %s", next_layer_idx, e)

    def _quest_local_score_one_layer(self, rid, rs, next_layer_idx,
                                      quest_query, budget,
                                      total_pages, already_fetched):
        """ICMS_ORIGINAL_QUEST=1 path: per-(KV-head) Quest top-K over
        GPU-side summaries, union → fetched_pages. No BF2 Score RPC, no
        adaptive allocator, no Q AllGather (TP=1 only). Isolated entry-
        point so the default path is unaffected.

        Stats: emitted in the same shape as the BF2 path so downstream
        analysis (icms_perf_sweep, accuracy bench) sees a Score event
        with the picked page IDs and a wall-time micro-measurement.
        """
        from vllm.distributed.kv_transfer.kv_connector.v1.quest_local_scorer \
            import quest_score_local_chunked

        # Mirror the default-path's input validation (icms_connector.py:2563)
        # — quest_query may be None or a non-Tensor when the caller doesn't
        # have one for this request slice. Skipping is the safe behavior:
        # downstream apply machinery just sees an empty fetched_pages
        # update and falls back to dense attention over what's already there.
        if not isinstance(quest_query, torch.Tensor) or quest_query.ndim < 2:
            return

        if self._tp_size > 1:
            if not getattr(self, "_quest_tp_warned", False):
                logger.warning(
                    "[icms_quest] ICMS_ORIGINAL_QUEST=1 currently supports "
                    "TP=1 only; got tp_size=%d. Falling through to a no-op "
                    "(no Score, no fetched pages added).", self._tp_size)
                self._quest_tp_warned = True
            return

        per_layer = rs.quest_gpu_summaries.get(next_layer_idx)
        if not per_layer:
            # Summaries not yet materialized for this layer (first prefill
            # pass before record_page fires). Nothing to score.
            return

        # Stack summaries into [P, H_kv, D] tensors, ordered by absolute pid.
        # Note: per_layer is appended to in record_page order; sort to be
        # safe against any out-of-order staging. P is bounded by total_pages.
        items = sorted(per_layer, key=lambda t: t[0])
        # Truncate to total_pages (defensive: stale entries from prior
        # requests under request_id reuse shouldn't appear, but guard).
        items = items[:total_pages]
        if not items:
            # Defensive: total_pages truncation could leave us empty even
            # after the per_layer non-empty check. Skip silently.
            return
        kmin = torch.stack([m for _, m, _ in items], dim=0)  # [P, H_kv, D]
        kmax = torch.stack([m for _, _, m in items], dim=0)
        # Quest top-K page budget. Mirror the existing path's k formula:
        # k = floor(total_pages * budget), at least 1.
        k = max(1, int(total_pages * float(budget)))

        t0 = time.perf_counter()
        try:
            picked = quest_score_local_chunked(
                quest_query, kmin, kmax, k=k,
                num_kv_heads=kmin.shape[1],
                exclude_pages=already_fetched)
        except Exception as e:
            logger.warning("[icms_quest] scorer failed at layer %d: %s",
                           next_layer_idx, e)
            picked = []
        wall_us = (time.perf_counter() - t0) * 1e6

        # ICMS_DIAG_SCORE_DUMP=<dir>: dump Q + per-page min/max summaries
        # + picked page IDs per (rid, scored layer) so the user can replay
        # different scoring algorithms offline without re-running the bench.
        _dump_dir = os.environ.get("ICMS_DIAG_SCORE_DUMP", "")
        if _dump_dir:
            try:
                import os as _os
                _os.makedirs(_dump_dir, exist_ok=True)
                _safe_rid = str(rid).replace("/", "_")
                _path = _os.path.join(
                    _dump_dir, f"{_safe_rid}_layer{next_layer_idx:02d}.pt")
                torch.save({
                    "rid": str(rid),
                    "layer": int(next_layer_idx),
                    "q": quest_query.detach().cpu(),
                    "kmin": kmin.detach().cpu(),
                    "kmax": kmax.detach().cpu(),
                    "total_pages": int(total_pages),
                    "k": int(k),
                    "budget": float(budget),
                    "picked": list(picked),
                    "already_fetched": list(already_fetched) if already_fetched else [],
                }, _path)
            except Exception as _e:
                logger.warning("[icms_diag_score_dump] save failed at "
                               "layer %d rid=%s: %s", next_layer_idx, rid, _e)

        rs.fetched_pages.setdefault(next_layer_idx, set()).update(picked)
        # Match the existing Score-stats shape so downstream tooling
        # doesn't have to special-case the Quest path.
        self.stats.record_score(wall_us, bool(picked), picked, next_layer_idx)
        logger.debug(
            "[icms_quest] layer=%d total_pages=%d k=%d new=%d "
            "fetched_so_far=%d wall_us=%.1f",
            next_layer_idx, total_pages, k, len(picked),
            len(rs.fetched_pages.get(next_layer_idx, set())), wall_us)

    def _ensure_summary_sink(self, total_pages: int) -> "Sink | None":
        """Lazy-allocate (or grow) the per-head summaries sink.

        Lives on the worker, alongside `self._sink_pool.sink`. Sized for
        `total_pages * summary_page_bytes`; doubled on first allocation
        to amortize regrowths as the chain grows. Returns the active
        sink or None if the underlying client doesn't have a working
        register_sink() (RDMA / shmem clients differ — for those the
        per-head pipeline is not supported in v1).

        2026-05-14 sink-id-collision fix: when the main sink is CUDA-IPC
        (``self._local_gpu_direct``), the shmem ``register_sink`` would
        return sink_id=1, which COLLIDES with the CUDA-IPC main sink's
        own sink_id=1 — the two server-side registries have INDEPENDENT
        ``next_id_`` counters (sink_registry.h:77 and
        cuda_ipc_sink_registry.h:124, both start at 1). The server's
        sink lookup chain (handlers_per_head.cc:593-616) tries the
        shmem registry FIRST, so ``fetch_union_per_head(sink_id=1)``
        resolves to the SHMEM SUMMARY sink (small) instead of the
        intended CUDA-IPC main sink (large) — ENOMEM at any union
        whose KV bytes exceed the summary sink's capacity, AND silent
        data corruption when the write fits (pages land in summary
        shmem, but apply path reads from the CUDA-IPC GPU sink).

        Fix: when ``_local_gpu_direct`` is True, register the summary
        sink as a CUDA-IPC sink too. Both sinks then live in the
        ``cuda_ipc_sinks_`` registry with sequential IDs (1 and 2),
        and the shmem registry stays empty — server lookup falls
        through to the cuda_ipc_sinks_ registry cleanly.
        """
        if self._client is None or self._geom is None or total_pages <= 0:
            return None
        # Worst-case sizing: the same sink_id may be presented to
        # FetchUnionPerHead if any sink-id-routing ambiguity persists,
        # AND a future server change could conceivably reuse the
        # summaries sink for a union dump. Size for max(summaries
        # bytes, worst-case union bytes) so the assertion at
        # handlers_per_head.cc:735 cannot fire on either RPC.
        #
        # Summaries:  total_pages * summary_page_bytes
        # Worst-case union (single layer): total_pages * kv_page_bytes
        #   (the union is bounded by total_pages — even if per-head
        #   top-k overlaps zero, the union can't exceed the chain).
        summary_needed = int(total_pages) * int(
            self._geom.summary_page_bytes)
        union_worst   = int(total_pages) * int(self._geom.kv_page_bytes)
        needed = max(summary_needed, union_worst)
        existing = getattr(self, "_per_head_summary_sink", None)
        capacity = getattr(self, "_per_head_summary_sink_capacity", 0)
        if existing is not None and capacity >= needed:
            return existing
        new_size = max(needed * 2, 64 * 1024)
        use_cuda_ipc = bool(getattr(self, "_local_gpu_direct", False))
        try:
            with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                if use_cuda_ipc:
                    # Pin to the same GPU as the main CUDA-IPC sink so the
                    # server's per-thread CUDA context binding stays
                    # consistent across summary and union RPCs.
                    gpu_dev = self._gpu_device
                    if self._tp_size > 1:
                        try:
                            gpu_dev = f"cuda:{int(torch.cuda.current_device())}"
                        except Exception:
                            gpu_dev = f"cuda:{self._tp_rank}"
                    new_sink = self._client.register_cuda_ipc_sink(
                        new_size, gpu_dev)
                else:
                    new_sink = self._client.register_sink(new_size)
        except Exception as e:
            logger.warning(
                "[icms_quest_per_head] could not allocate summaries "
                "sink (size=%d, total_pages=%d, cuda_ipc=%s): %r",
                needed, total_pages, use_cuda_ipc, e)
            return None
        if existing is not None:
            try:
                with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                    self._client.unregister_sink(existing)
                existing.close()
            except Exception:
                pass
        self._per_head_summary_sink = new_sink
        self._per_head_summary_sink_capacity = new_size
        logger.info(
            "[icms_quest_per_head] (re)allocated summary sink: "
            "size=%d bytes (summary_needed=%d, union_worst=%d, "
            "total_pages=%d) cuda_ipc=%s",
            new_size, summary_needed, union_worst, total_pages,
            use_cuda_ipc)
        return new_sink

    def _quest_per_kv_head_score_one_layer(self, rid, rs, req_idx,
                                            next_layer_idx, quest_query,
                                            budget, total_pages,
                                            already_fetched):
        """ICMS_QUEST_MODE=per_kv_head path: server-fetched summaries +
        GPU per-(KV-head) Quest scoring + server-fetched union of
        selected pages, via v8 wire opcodes 25 / 27 / 29.

        Mirrors the ICMS_ORIGINAL_QUEST shape (per-layer union → fed to
        vLLM's PagedAttention via _pending_scores) except summaries come
        from the server instead of a local stash. That makes this path
        usable for chains written by other requests (cross-request
        reuse), while ICMS_ORIGINAL_QUEST is only valid when the same
        process wrote the chain.

        TP=1 only for v1 — same constraint as ICMS_ORIGINAL_QUEST.

        `req_idx` is the caller's batch slot for this rid. Stamped on
        the synthesized `_pending_scores` entry so the apply path
        (which slices per-rid query/block-table tensors by req_idx)
        reads the correct slice. Pre-fix this was hardcoded to 0,
        breaking multi-rid batches.
        """
        if not isinstance(quest_query, torch.Tensor) or quest_query.ndim < 2:
            return
        if self._tp_size > 1:
            if not self._quest_per_head_tp_warned:
                logger.warning(
                    "[icms_quest_per_head] ICMS_QUEST_MODE=per_kv_head "
                    "currently supports TP=1 only; got tp_size=%d. "
                    "Falling through to a no-op.", self._tp_size)
                self._quest_per_head_tp_warned = True
            return
        if total_pages <= 0 or self._client is None or self._geom is None:
            return
        # The new opcodes are not implemented for RDMA or shmem clients
        # in v1 — they live in the unix-socket IcmsClient only. Detect
        # by feature presence and bail with a clear log.
        if not hasattr(self._client, "fetch_summaries_per_head"):
            if not self._quest_per_head_client_warned:
                logger.warning(
                    "[icms_quest_per_head] active client (%s) lacks "
                    "fetch_summaries_per_head; per-head mode requires "
                    "the unix-socket IcmsClient. No-op.",
                    type(self._client).__name__)
                self._quest_per_head_client_warned = True
            return
        sum_sink = self._ensure_summary_sink(total_pages)
        if sum_sink is None:
            return
        if not self._quest_per_head_first_call_logged:
            logger.info(
                "[icms_quest_per_head] FIRST CALL — per-head path active "
                "(rid=%s layer=%d total_pages=%d budget=%.3f)",
                rid[:8], next_layer_idx, total_pages, float(budget))
            self._quest_per_head_first_call_logged = True

        chain = self._rank_chain(rs.chain)
        icms_rid = self._icms_request_id(rid, 0)
        t0 = time.perf_counter()

        # 1. Fetch per-page summaries for THIS layer.
        try:
            with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                sum_reply = self._client.fetch_summaries_per_head(
                    request_id=icms_rid, chain=chain, layer=next_layer_idx,
                    sink=sum_sink, reuse_through_layer=next_layer_idx)
        except IcmsError as e:
            logger.warning(
                "[icms_quest_per_head] FetchSummariesPerHead failed at "
                "layer %d (rid=%s): %r", next_layer_idx, rid[:8], e)
            return

        P = len(sum_reply.page_ids)
        if P == 0:
            return
        # Pin the page_id ordering invariant: server MUST return
        # `page_ids = [0, 1, ..., P-1]` so that scorer-returned indices
        # (row indices into kmin/kmax) coincide with real page IDs sent
        # back via fetch_union_per_head. If the server's reply ever
        # diverges from this (partial chain, page_id remapping, etc.),
        # `picked` would mean different things to the scorer (row idx)
        # vs the server (page id). Fail fast.
        if (sum_reply.page_ids[0] != 0 or sum_reply.page_ids[-1] != P - 1
                or len(sum_reply.page_ids) != P):
            logger.warning(
                "[icms_quest_per_head] sum_reply.page_ids violates the "
                "[0..P-1] contract (P=%d, head=%s, tail=%s); aborting "
                "this layer to avoid corrupted KV reads.",
                P, sum_reply.page_ids[:4], sum_reply.page_ids[-4:])
            return

        # 2. Reshape sink bytes → (kmin, kmax) tensors on the GPU.
        # Per-page blob layout matches record_page() at the connector
        # write side: kmin.tobytes() ++ kmax.tobytes(), each
        # [H_kv, D] fp16.
        H_kv = int(self._geom.num_kv_heads)
        D = int(self._geom.head_dim)
        elem = int(self._geom.elem_bytes)
        spb = int(self._geom.summary_page_bytes)
        if elem != 2:
            logger.warning(
                "[icms_quest_per_head] elem_bytes=%d not supported in v1 "
                "(fp16 only); no-op.", elem)
            return
        # Bulk decode summaries → [P, 2, H_kv, D] fp16 tensor on GPU.
        # The server packs page i at sink offset slot_base + i * spb,
        # so the bytes are contiguous in the sink starting at
        # sum_reply.sink_offsets[0]. One bulk slice instead of P
        # per-page memcpys — the loop was the dominant Python cost
        # at long context (P × spb bytes; ~64 MB/layer at qwen3 128K).
        slot_base = int(sum_reply.sink_offsets[0])
        total_bytes = P * spb
        contiguous = all(
            int(sum_reply.sink_offsets[i]) == slot_base + i * spb
            for i in range(P))
        device = quest_query.device if quest_query.is_cuda else "cuda"
        # 2026-05-14 sink-id-collision fix: when sum_sink is a CUDA-IPC
        # sink (the `_local_gpu_direct` path), the bytes are already on
        # the GPU — slice directly instead of D2H→host_bytes→H2D.
        if getattr(sum_sink, "is_gpu_direct", False):
            gpu_tensor = getattr(sum_sink, "gpu_tensor", None)
            if gpu_tensor is None:
                logger.warning(
                    "[icms_quest_per_head] GPU-IPC summary sink missing "
                    "gpu_tensor; aborting layer %d", next_layer_idx)
                return
            if contiguous:
                flat_u8 = gpu_tensor[
                    slot_base:slot_base + total_bytes]
            else:
                logger.warning(
                    "[icms_quest_per_head] sink_offsets are non-contiguous "
                    "(server-side layout change?); falling back to per-page "
                    "D2D memcpy.")
                flat_u8 = torch.empty(total_bytes, dtype=torch.uint8,
                                      device=gpu_tensor.device)
                for i, off in enumerate(sum_reply.sink_offsets):
                    flat_u8[i * spb:(i + 1) * spb] = \
                        gpu_tensor[off:off + spb]
            # uint8 → fp16 view (zero-copy reinterpret) → reshape.
            flat_gpu = flat_u8.view(torch.float16).reshape(
                P, 2, H_kv, D)
            # Ensure on the query's device (typically same as gpu_tensor's).
            if flat_gpu.device != torch.device(device):
                flat_gpu = flat_gpu.to(device=device, non_blocking=True)
        else:
            view = sum_sink.view()
            # Defensive: confirm contiguity. If a future server-side layout
            # change ever breaks the dense [slot_base, slot_base+P*spb)
            # invariant, fall back to per-page rather than silently
            # scrambling KV. The compare loop is O(P) ints — noise vs the
            # avoided memcpy.
            if contiguous:
                host_bytes = bytes(view[slot_base:slot_base + total_bytes])
            else:
                logger.warning(
                    "[icms_quest_per_head] sink_offsets are non-contiguous "
                    "(server-side layout change?); falling back to per-page "
                    "memcpy.")
                host_buf = bytearray(total_bytes)
                for i, off in enumerate(sum_reply.sink_offsets):
                    host_buf[i * spb:(i + 1) * spb] = bytes(view[off:off + spb])
                host_bytes = bytes(host_buf)
            # `host_bytes` is immutable so torch.frombuffer can wrap it
            # without copying; .to(device) is the single H2D copy.
            flat_cpu = torch.frombuffer(host_bytes, dtype=torch.float16,
                                          count=P * 2 * H_kv * D).reshape(
                P, 2, H_kv, D)
            flat_gpu = flat_cpu.to(device=device, non_blocking=True)
        kmin = flat_gpu[:, 0, :, :].contiguous()
        kmax = flat_gpu[:, 1, :, :].contiguous()

        # 3. Score on GPU (existing per-(KV-head) scorer; union of top-K).
        from vllm.distributed.kv_transfer.kv_connector.v1.quest_local_scorer \
            import quest_score_local_chunked
        k = max(1, int(total_pages * float(budget)))
        try:
            picked = quest_score_local_chunked(
                quest_query, kmin, kmax, k=k,
                num_kv_heads=H_kv,
                exclude_pages=already_fetched)
        except Exception as e:
            logger.warning(
                "[icms_quest_per_head] scorer failed at layer %d: %s",
                next_layer_idx, e)
            return
        if not picked:
            return

        # 4. Fetch the per-head union of selected pages into the main
        # KV sink. Reply offsets feed _pending_scores so the apply path
        # is unchanged from the BF2 Score path.
        try:
            with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                union_reply = self._client.fetch_union_per_head(
                    request_id=icms_rid, chain=chain, layer=next_layer_idx,
                    sink=self._sink_pool.sink, page_ids=picked,
                    reuse_through_layer=next_layer_idx)
        except IcmsError as e:
            logger.warning(
                "[icms_quest_per_head] FetchUnionPerHead failed at "
                "layer %d (rid=%s, k=%d): %r",
                next_layer_idx, rid[:8], len(picked), e)
            return

        # 5. Hand off to the existing apply path. Synthesize a
        # ScoreReply-shaped object with picked + sink_offsets so
        # wait_for_layer / _apply_fetch_impl don't need to special-case
        # per-head replies. Scores are recomputed from the same kernel
        # (max-over-heads of `score[pid, h]`) so perf-sweep histograms
        # don't see degenerate all-zero distributions in per-head mode.
        from vllm.distributed.kv_transfer.kv_connector.v1.quest_local_scorer \
            import quest_score_pages_max
        try:
            score_map = quest_score_pages_max(
                quest_query, kmin, kmax, picked, num_kv_heads=H_kv)
        except Exception as e:
            logger.warning(
                "[icms_quest_per_head] score-map compute failed at "
                "layer %d: %s; defaulting to zeros.", next_layer_idx, e)
            score_map = {pid: 0.0 for pid in picked}
        from icms_client.protocol import ScoreReply
        synth = ScoreReply(
            request_id=icms_rid,
            status=0,
            trie_walk_ns=0,
            summary_read_ns=int(sum_reply.summary_read_ns),
            score_ns=int((time.perf_counter() - t0) * 1e9),
            sink_write_ns=int(union_reply.sink_write_ns),
            cache_hit=bool(sum_reply.cache_hit),
            concurrent_requests=int(union_reply.concurrent_requests),
            server_ingest_to_ready_ns=int(
                union_reply.server_ingest_to_ready_ns),
            effective_supply_bps=int(union_reply.effective_supply_bps),
            page_ids=list(picked),
            scores=[score_map.get(pid, 0.0) for pid in picked],
            sink_offsets=list(union_reply.sink_offsets),
        )
        attn_layer_name = f"model.layers.{next_layer_idx}.self_attn.attn"
        with self._score_lock:
            self._assert_pending_scores_no_clobber(
                attn_layer_name, rid, source="per-kv-head-score-landing")
            self._pending_scores.setdefault(attn_layer_name, {})[rid] = (
                synth, req_idx)

        rs.fetched_pages.setdefault(next_layer_idx, set()).update(picked)
        wall_us = (time.perf_counter() - t0) * 1e6
        self.stats.record_score(wall_us, bool(picked), picked, next_layer_idx)
        logger.debug(
            "[icms_quest_per_head] layer=%d total_pages=%d k=%d new=%d "
            "fetched_so_far=%d wall_us=%.1f",
            next_layer_idx, total_pages, k, len(picked),
            len(rs.fetched_pages.get(next_layer_idx, set())), wall_us)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Stash GPU KV cache tensors for Path B reordered block table."""
        self._gpu_kv_caches = dict(kv_caches)
        # No separate fetch buffer needed — Path B uses a reordered block
        # table pointing into the main cache.  This preserves continuation
        # self-attention (see _populate_fetch_buffer).
        logger.info("Path B: registered %d KV cache layers (reordered block table mode)",
                     len(kv_caches))

        # 2026-05-12 OPTION 4 — ICMS-OWNED SCRATCH TENSOR:
        # Cross-rid block aliasing bug (project_multi_rid_slot1_residual_gap_2026-05-12.md):
        # in multi-rid mode, vLLM aliases prefix-cached physical blocks
        # across rids; ICMS scatter into bt[req_idx][valid_pids] (where
        # valid_pids fall in the matched-prefix range) lands on SHARED
        # blocks → cross-rid contamination. Workaround: allocate an
        # ICMS-owned scratch tensor (separate from main_key), redirect
        # the scatter there, and tell FA to read from the scratch via
        # IcmsFetchState.key_cache / .value_cache override
        # (forks/vllm/vllm/v1/attention/backends/flash_attn.py:719+).
        #
        # Sizing: one slot per (rid, selected_or_cont_block). The slot
        # count per rid is icms_k (selected pages cap) + MAX_CONT (8 is
        # enough for question + early decode). Allocated per layer to
        # match the layer's KV cache shape.
        #
        # Gated by ICMS_OPTION_4_SCRATCH=1 (default OFF for safety while
        # under test). When OFF, behavior is unchanged from current main
        # code path.
        self._icms_scratch_k: dict[str, torch.Tensor] = {}
        self._icms_scratch_v: dict[str, torch.Tensor] = {}
        if os.environ.get("ICMS_OPTION_4_SCRATCH", "0") == "1":
            max_n_seqs = max(1, int(getattr(self, "_max_num_seqs", 1) or 1))
            k_cap = int(getattr(self, "_k", 2000) or 2000)
            MAX_CONT = 16  # safe upper bound for question + early decode
            slots_per_rid = k_cap + MAX_CONT
            total_slots = max_n_seqs * slots_per_rid
            mem_total = 0
            for layer_name, kv in kv_caches.items():
                # kv shape: typically [num_blocks, 2, page_size, num_kv_heads, head_dim]
                # OR [2, num_blocks, ...] depending on backend. We need
                # a scratch with the SAME trailing dims and dtype.
                if kv.ndim != 5:
                    logger.warning(
                        "[icms-scratch] layer=%s unexpected kv.ndim=%d; "
                        "scratch not allocated. Multi-rid fix unavailable "
                        "for this layer.", layer_name, kv.ndim)
                    continue
                if kv.shape[0] == 2:
                    # Layout [2, num_blocks, page_size, num_kv_heads, head_dim]
                    _per_layer_shape = kv.shape[1:]  # drop the leading "2"
                elif kv.shape[1] == 2:
                    # Layout [num_blocks, 2, page_size, num_kv_heads, head_dim]
                    _per_layer_shape = (kv.shape[0],) + kv.shape[2:]
                else:
                    logger.warning(
                        "[icms-scratch] layer=%s kv.shape=%s — neither dim "
                        "is 2. Scratch not allocated for this layer.",
                        layer_name, tuple(kv.shape))
                    continue
                # _per_layer_shape is now [num_blocks, page_size, num_kv_heads, head_dim]
                # Replace num_blocks with total_slots for scratch.
                scratch_shape = (total_slots,) + tuple(_per_layer_shape[1:])
                k_scratch = torch.zeros(
                    *scratch_shape, dtype=kv.dtype, device=kv.device)
                v_scratch = torch.zeros(
                    *scratch_shape, dtype=kv.dtype, device=kv.device)
                self._icms_scratch_k[layer_name] = k_scratch
                self._icms_scratch_v[layer_name] = v_scratch
                mem_total += k_scratch.numel() * k_scratch.element_size() * 2
            self._icms_scratch_slots_per_rid = slots_per_rid
            logger.info(
                "[icms-scratch] Option-4 enabled: per-layer scratch allocated "
                "for %d layers, max_num_seqs=%d, slots_per_rid=%d, "
                "total scratch mem=%.1f MiB",
                len(self._icms_scratch_k), max_n_seqs,
                slots_per_rid, mem_total / (1024 * 1024))
        else:
            self._icms_scratch_slots_per_rid = 0

    def set_attn_metadata(self, attn_metadata):
        """Stash per-step attn_metadata dict from ForwardContext."""
        self._attn_metadata = attn_metadata

    def set_input_batch_req_ids(self, req_ids: list[str]) -> None:
        """Capture vLLM's authoritative input_batch.req_ids ordering.

        2026-05-12 multi-rid row-mapping fix: every connector site that
        derived a batch row index from `enumerate(self._last_step_requests)`
        was using the wrong order (meta.requests = new+cached append
        order, not vLLM input_batch order). With this hook the connector
        knows the live order and can map rid → FA bt row correctly.
        """
        self._input_batch_req_ids = list(req_ids)
        self._rid_to_bt_row = {rid: i for i, rid in enumerate(req_ids)}

    def _provenance_check_natural_bt(
        self, layer_name: str, abs_layer: int, path: str = "?",
    ) -> None:
        """Check natural attn_metadata.block_table against ICMS-populated
        block_ids for each active rid. Flags ext_comp blocks in the
        layer's bt that ICMS did not populate (i.e., attention will
        read uninitialized / stale free-pool KV).

        Called from wait_for_layer's non-scored short-circuit. Gated on
        ICMS_TRACE_KV_PROVENANCE; safe no-op otherwise.
        """
        try:
            am = (self._attn_metadata.get(layer_name)
                  if isinstance(self._attn_metadata, dict) else None)
            if am is None or not hasattr(am, "block_table"):
                return
            bt = am.block_table
            if bt is None:
                return
            # bt: [num_reqs, max_blocks_per_req] tensor (GPU). Move to
            # CPU once for indexing — only fires when env flag is on.
            if hasattr(bt, "cpu"):
                bt_cpu = bt.cpu()
            else:
                bt_cpu = bt
            tracker = icms_provenance.tracker()
            for req_idx, (rid, rs) in enumerate(self._requests.items()):
                if req_idx >= bt_cpu.shape[0]:
                    break
                # Cap to seq_lens to ignore padding tail.
                _row = bt_cpu[req_idx].tolist()
                tracker.check_bt(
                    rid=rid, layer_idx=abs_layer,
                    bt_block_ids=_row, path=path,
                )
        except Exception as _e:
            # Tracing must never crash the forward path. Log + swallow.
            logger.debug("provenance check failed: %s", _e)

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
        # Fast path: once every active rs is dense_mode AND DECODE_APPLY=0,
        # no Score RPCs are pending and decode is meant to be sparse over
        # the leftover trimmed bt from the last prefill stride. Skip the
        # lock + dict-pop + diag block (was ~50 µs × 48 layers × N decode
        # iters post-flip on Qwen3-30B-A3B at 131k = ~2 ms/iter — the
        # residual gap between sparse decode and pure dense baseline).
        # Restricted to DECODE_APPLY=0 because for DECODE_APPLY=1 (mode c)
        # the existing path at line 3246-3264 clears icms_fetch_state on
        # the all-dense check — that clear is required so attention falls
        # back to natural-bt full-context dense (post-saturation, all
        # pages are populated from incremental apply). Skipping the clear
        # in mode (c) would leave a stale trimmed bt and corrupt
        # attention for layers that didn't set their own state.
        if (self._cached_all_dense
                and os.environ.get("ICMS_DECODE_APPLY") == "0"
                and self._prefill_done):
            return
        # 2026-05-09: with a non-zero scored_layers_mask, non-scored layers
        # never had Score fired (gated in on_layer_score) so _pending_scores
        # has nothing for them. But the prior scored layer's set_active()
        # leaves a trimmed bt active — vLLM would use that for this layer's
        # attention, corrupting it. Clear active state and short-circuit so
        # attention falls back to natural full-context bt (e.g. SW layers in
        # gemma-3 attend over the full prefill, with their own SW mask).
        _abs_layer_for_mask = self._extract_layer_idx(layer_name)
        if (_abs_layer_for_mask is not None
                and not self._geom.is_scored(_abs_layer_for_mask)):
            from vllm.v1.attention import icms_fetch_state
            icms_fetch_state.clear()
            # KV-provenance: this is the non-scored short-circuit path.
            # Attention will read the NATURAL bt from attn_metadata,
            # which includes ext_comp blocks that ICMS may not have
            # populated for this layer. Check per active rid and log
            # any unpopulated ext_comp blocks in their bt row.
            if icms_provenance.is_enabled():
                self._provenance_check_natural_bt(
                    layer_name, _abs_layer_for_mask, path="nonscored")
            return
        # 2026-05-16 fix: every write to `_pending_scores` keys by the
        # CONSTRUCTED `f"model.layers.{idx}.self_attn.attn"` (5+ call
        # sites: faps-fast-path-promote ~3793, faps-slow-path-landing
        # ~4058, score-reply-landing ~6140, per-kv-head ~6622, etc.).
        # The read here used the RAW vLLM-passed `layer_name`, which
        # for `Mistral3ForConditionalGeneration` is prefixed with
        # `language_model.` (the LM is wrapped at that prefix; see
        # mistral3.py:489 `init_vllm_registered_model(prefix=...)`).
        # The pop silently missed for every layer → Score replies
        # accumulated in `_pending_scores` (the source of the per-layer
        # `_pending_scores clobber` warnings on mistral-small but not
        # qwen3/gemma-3 which register at the top level) → KV-overlay
        # apply never ran → attention read uncovered KV → token-loop
        # gibberish on every sparse mistral-small run (even at b=1.0
        # where coverage was complete — the smoking gun in
        # [[mistral-sparse-path-bug-2026-05-15]]).
        canonical_key = (
            f"model.layers.{_abs_layer_for_mask}.self_attn.attn"
            if _abs_layer_for_mask is not None else layer_name)
        with self._score_lock:
            per_request = self._pending_scores.pop(canonical_key, None)

        if os.environ.get("ICMS_DIAG_FAPS") == "1":
            _abs = self._extract_layer_idx(layer_name)
            if per_request:
                for _rid_dbg, (_reply_dbg, _) in per_request.items():
                    logger.info(
                        "[diag-faps] wfl POP rid=%s layer=%s k=%d sink_off_n=%d",
                        _rid_dbg[:8], _abs, len(_reply_dbg.page_ids),
                        len(_reply_dbg.sink_offsets))
            else:
                logger.info(
                    "[diag-faps] wfl POP-empty layer=%s", _abs)

        # Bug 11 instrumentation (2026-04-29): log every wait_for_layer
        # call's pop result + decode/prefill phase so we can see whether
        # _pending_scores has entries during decode iters at budget < 1.0.
        # Gated by ICMS_DIAG_WFL=1 to avoid log spam.
        if os.environ.get("ICMS_DIAG_WFL") == "1":
            _phase_dbg = "decode" if self._prefill_done else "prefill"
            _per_req_n = len(per_request) if per_request else 0
            _abs_layer_dbg = self._extract_layer_idx(layer_name)
            try:
                from vllm.v1.attention import icms_fetch_state as _ifs
                _active_set = _ifs.get_active() is not None
            except Exception:
                _active_set = "err"
            # Per-rs dense_mode + post-flip iter context.
            _dense_summary = ",".join(
                f"{r[:8]}:dense={rs.dense_mode}/pdi={rs._post_dense_iter}"
                for r, rs in self._requests.items())
            logger.info(
                "[diag-wfl] phase=%s layer=%s abs=%s per_req=%d "
                "active_set_pre=%s rs=%s",
                _phase_dbg, layer_name, _abs_layer_dbg, _per_req_n,
                _active_set, _dense_summary)
        # ICMS_DIAG_FULL: at layer 0 (one per forward), dump natural attn
        # metadata for any post-dense-flip request — pinpoints stale bt /
        # slot_mapping / seq_lens drift across the prefill→decode
        # transition or the dense-mode flip.
        if (os.environ.get("ICMS_DIAG_FULL") == "1"
                and self._extract_layer_idx(layer_name) == 0):
            self._diag_full_iter_metadata(
                layer_name, self._attn_metadata, where="wfl_entry")

        # ICMS_DIAG_SPDD_COMPREHENSIVE=1 (2026-05-15): per-layer wfl-entry
        # snapshot for SPDD diagnosis. Logs whether a pending reply exists
        # for each active rid at this layer, dense_mode, _post_dense_iter,
        # and per_request size. Critical signal: in SPDD-with-FetchAll,
        # _pending_scores[0] should have a populated reply at decode iter 1
        # layer 0; without FetchAll, it should be empty (normal Score path
        # repopulates it later). Diff between the two runs at this hook
        # exposes layer-by-layer behavior changes.
        if os.environ.get("ICMS_DIAG_SPDD_COMPREHENSIVE") == "1":
            try:
                _abs_l = self._extract_layer_idx(layer_name)
                _phase = "decode" if self._prefill_done else "prefill"
                # Snapshot per-rid context. _requests may be empty during
                # transients; tolerate.
                _per_req_keys = (list(per_request.keys()) if per_request
                                  else [])
                _per_req_n = len(_per_req_keys)
                _reuse_pres = (layer_name in
                                getattr(self, "_pending_reuse", {}))
                _rs_summary = []
                for _r, _rs in list(self._requests.items()):
                    _ri = self._rid_to_bt_row.get(_r, -1)
                    _has_pending = _r in (per_request or {})
                    _rs_summary.append(
                        f"{_r[:8]}:ri={_ri}/dense={_rs.dense_mode}/"
                        f"pdi={_rs._post_dense_iter}/"
                        f"chain={len(_rs.chain)}/"
                        f"fetched={getattr(_rs,'fetched_pages',-1)}/"
                        f"pending={_has_pending}")
                logger.info(
                    "[diag-spdd-cs-wfl] phase=%s layer=%s abs=%s "
                    "per_req_n=%d per_req_keys=%s reuse_present=%s "
                    "rs_summary=%s",
                    _phase, layer_name, _abs_l, _per_req_n,
                    [k[:8] for k in _per_req_keys[:4]],
                    _reuse_pres, _rs_summary[:4])
            except Exception as _diag_e:
                logger.warning(
                    "[diag-spdd-cs-wfl] log failed: %r", _diag_e)

        if not per_request:
            return

        # Bug 11 (2026-04-29): the prefill_done short-circuit was making the
        # M3+M4-A decode-time apply path at line ~3500 dead code: chain
        # pages incrementally fetched via decode-iter Score replies never
        # landed in main_key, so at budget < 1.0 decode read garbage from
        # un-applied chain blocks. Drop prefill_done from the gate; the
        # other two guards still protect pre-init / shutdown.
        #
        # ICMS_DECODE_APPLY=0 — restore the legacy prefill_done short-
        # circuit. With this set, decode-time Score replies are NOT
        # scattered to main_key; decode attends only over the K pages
        # selected at prefill (no bitmap growth, no M4-A dense flip).
        # This gives the pure-sparse mode (no incremental fetch).
        if (os.environ.get("ICMS_DECODE_APPLY") == "0"
                and self._prefill_done):
            return
        if (not self._gpu_kv_caches or self._attn_metadata is None):
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
        # M4-A (Fix F, 2026-04-29): once any active request has flipped
        # to dense_mode, no Score/FetchAll fires this iter, so per-layer
        # flags will never be set — skip the spin to avoid 5s timeouts
        # × num_layers per decode iter. Inline check (we're on _Worker;
        # is_dense_for_active_request lives on IcmsConnector).
        if self._requests:
            _all_dense = all(
                getattr(rs, "dense_mode", False)
                for rs in self._requests.values())
            if _all_dense:
                # Clear icms_fetch_state to prevent the prior iter's
                # set_active(trimmed bt) from leaking into this iter's
                # attention. Once dense_mode flips, save_kv_layer (the
                # usual restore_attn_metadata trigger) is short-circuited
                # by the Quest hook, so without an explicit clear here
                # the stale _active sticks around for the rest of decode.
                from vllm.v1.attention import icms_fetch_state
                icms_fetch_state.clear()
                if (os.environ.get("ICMS_DIAG_FULL") == "1"
                        and self._extract_layer_idx(layer_name) == 0):
                    self._diag_full_iter_metadata(
                        layer_name, self._attn_metadata,
                        where="wfl_all_dense_return")
                return
        # ICMS_REPLY_EARLY=0 disables the flag-spin entirely so the
        # connector only proceeds after the sync Score/FetchAll reply
        # has returned (= Phase-2 fully done). Used to test whether the
        # warm-prefix corruption is a reply-early ordering issue.
        if (os.environ.get("ICMS_REPLY_EARLY", "1") != "0"
                and abs_layer is not None
                and self._sink_pool is not None
                and getattr(self._sink_pool.sink, "flag_count", 0) > 0):
            sink = self._sink_pool.sink
            ready_at_call = sink.is_layer_ready(abs_layer)
            if (_slack_called is not None
                    and 0 <= abs_layer < len(_slack_called)):
                self._slack_flag_at_call[abs_layer] = bool(ready_at_call)
            # ICMS_TRACE_FLAGS=1: log every poll. With ICMS_TRACE_FLAGS_RID,
            # we can correlate against this rid's owns clear/score events
            # to see if a different rid clobbered our flag.
            _trace_flags = os.environ.get("ICMS_TRACE_FLAGS") == "1"
            if _trace_flags:
                _t = time.perf_counter()
                logger.info(
                    "[trace-flags] POLL t=%.6f rid=%s layer=%d ready=%s",
                    _t, rid, abs_layer, bool(ready_at_call))
            if not ready_at_call:
                # Poll with exponential backoff starting at 0 (pure spin)
                # — writes land within ~100µs so spin is right. Cap total
                # wait at 5s to avoid hangs if something goes wrong.
                deadline = t_layer_start + 5.0
                _poll_count = 0
                while not sink.is_layer_ready(abs_layer):
                    _poll_count += 1
                    if time.perf_counter() > deadline:
                        if _trace_flags:
                            logger.warning(
                                "[trace-flags] TIMEOUT rid=%s layer=%d "
                                "polls=%d wait_s=%.3f",
                                rid, abs_layer, _poll_count,
                                time.perf_counter() - t_layer_start)
                        logger.warning(
                            "wait_for_layer: flag timeout for layer=%d", abs_layer)
                        break
                if _trace_flags and _poll_count > 0:
                    _t2 = time.perf_counter()
                    logger.info(
                        "[trace-flags] READY rid=%s layer=%d polls=%d "
                        "wait_us=%.1f",
                        rid, abs_layer, _poll_count,
                        (_t2 - t_layer_start) * 1e6)
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

        # Multi-rid batching path (ICMS_ALLOW_BATCH=1): collect per-rid
        # IcmsFetchStates without firing set_active inside the loop,
        # then aggregate into one multi-row state and fire set_active
        # ONCE per layer. The single-rid path (legacy) still calls
        # set_active inside _apply_selective_attention as before.
        #
        # 2026-05-06 fix: was `len(per_request) >= 2` which fell through
        # to the single-rid path when only 1 rid had its Score reply
        # ready at this layer (others' replies arrived a later layer due
        # to timing). The single-rid IcmsFetchState has block_table
        # shape [1, k+c] but the actual forward batch has N rids → FA3
        # then crashes with `batch_size must be equal to batch_size_k`
        # because cu_seqlens_q has N entries but seqused_k has 1.
        # The aggregate path uses natural_bt as base (shape
        # [N, max_blocks]) and overrides only the 1 row whose rid had
        # Score, which is the correct shape regardless of how many
        # captures exist.
        # 2026-05-09 (Item C audit): widened from `_allow_batch()` to
        # `_is_multi_rid_mode()` so a launcher that sets
        # max_num_seqs > 1 without ICMS_ALLOW_BATCH=1 still takes the
        # aggregate fetchstate path. The legacy single-rid path's
        # block_table shape doesn't match an N-rid forward batch and
        # crashes FA3 with batch_size_k mismatch. Was the false-positive
        # in today's debug session.
        _use_batch_path = self._is_multi_rid_mode() and len(per_request) >= 1
        _captures: list | None = [] if _use_batch_path else None

        # 2026-05-12 FIX (cross-rid contamination): req_idx cached in
        # _pending_scores at Score-fire time can be STALE by the time
        # _apply_selective_attention runs. Under chunked prefill with
        # max_num_seqs>1, the same rid can occupy different batch
        # positions in successive forward passes (e.g., rid=4 at
        # req_idx=0 in chunk N, req_idx=3 in chunk N+1). If Apply uses
        # the stale value, `combined_bt[req_idx, :k+c] = trim_row` at
        # line ~6351 writes to the WRONG row → that row's owner reads
        # rid_A's selected KV pages → cross-rid contamination (rid_B
        # generates rid_A's needle). Discovered in the qwen3 batched
        # smoke: 4 examples score 0.75; 8 examples score 0.25; single-rid
        # baseline scores 1.00 for the same inputs.
        #
        # Fix: build a live rid → req_idx map from `_last_step_requests`
        # (set at the start of every forward step from meta.requests)
        # and use that instead of the cached value. If a rid in
        # per_request is no longer in the current batch (e.g., it
        # finished), skip its apply — applying it would target some
        # other rid's row.
        _live_rid_to_req_idx: dict[str, int] = {}
        if self._is_multi_rid_mode():
            # 2026-05-12 multi-rid row-mapping fix: prefer the
            # authoritative `self._rid_to_bt_row` populated from vLLM's
            # `input_batch.req_ids`. Pre-fix this rebuild used the SAME
            # `_last_step_requests` enumeration as the buggy source, so
            # it didn't actually fix the cross-rid contamination it was
            # written to fix. `getattr` keeps access safe for mocks.
            _rid_to_bt_row_safe = getattr(self, "_rid_to_bt_row", None) or {}
            if _rid_to_bt_row_safe:
                _live_rid_to_req_idx = dict(_rid_to_bt_row_safe)
            elif self._last_step_requests:
                for _live_idx, _prs in enumerate(self._last_step_requests):
                    _live_rid_to_req_idx[_prs.request_id] = _live_idx

        total_k_pages = 0
        for rid, (reply, _cached_req_idx) in per_request.items():
            if reply is None or not reply.page_ids:
                continue
            # Re-derive the live req_idx for this rid. If the rid is no
            # longer in the current step's batch, skip apply (the cached
            # req_idx would point into another rid's row).
            if self._is_multi_rid_mode():
                if rid not in _live_rid_to_req_idx:
                    if os.environ.get(
                            "ICMS_DIAG_REQIDX_DRIFT", "1") != "0":
                        logger.warning(
                            "[reqidx-drift] layer=%s rid=%s NOT in current "
                            "batch (cached_req_idx=%d, current batch n=%d) "
                            "— SKIPPING apply to avoid writing to another "
                            "rid's block_table row",
                            layer_name, rid[:8], _cached_req_idx,
                            len(_live_rid_to_req_idx))
                    continue
                req_idx = _live_rid_to_req_idx[rid]
                if (req_idx != _cached_req_idx
                        and os.environ.get(
                            "ICMS_DIAG_REQIDX_DRIFT", "1") != "0"):
                    logger.warning(
                        "[reqidx-drift] layer=%s rid=%s cached_req_idx=%d "
                        "live_req_idx=%d — applying at live (cached would "
                        "have caused cross-rid contamination)",
                        layer_name, rid[:8], _cached_req_idx, req_idx)
            else:
                # Single-rid path: cached value is fine (only 1 rid
                # per forward pass).
                req_idx = _cached_req_idx
            try:
                t_apply_start = time.perf_counter()
                self._apply_selective_attention(
                    layer_name, reply, req_idx, rid,
                    _capture=_captures)
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
                if self._cfg_apply_soft_fail:
                    # Legacy behavior, gated and explicit. Use only if
                    # production traffic absolutely cannot tolerate a
                    # failed-request surface here.
                    return
                raise

        # Aggregate per-rid captures into one multi-row IcmsFetchState
        # and fire set_active once. Skipped when no rids produced a
        # capture (all replies empty / all rids errored soft-fail).
        if _use_batch_path and _captures:
            self._aggregate_and_set_fetch_state(layer_name, _captures)

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

    def _aggregate_and_set_fetch_state(self, layer_name: str,
                                       captures: list) -> None:
        """Stack per-rid IcmsFetchStates into one multi-row state.

        captures: list of (req_idx, IcmsFetchState) tuples produced by
        _apply_selective_attention(_capture=[...]) under the multi-rid
        batching path (ICMS_ALLOW_BATCH=1, N>=2).

        Strategy:
          1. Start from the natural attn_metadata.block_table
             (shape [num_reqs, max_blocks_natural]) and seq_lens
             (shape [num_reqs]). Clone so we don't mutate vLLM's tensors.
          2. For each captured rid, overwrite the first len(trim_row)
             positions of combined_bt[req_idx] with the trim row. The
             tail of that row keeps natural values, but FA stops at
             seq_lens[req_idx] so they're never read.
          3. Override seq_lens[req_idx] with the trim seq_len.
          4. max_seq_len = natural max (trim seq_lens are <= natural,
             so the natural max is a safe upper bound for FA's tile
             scheduler).
        Single-rid path (no aggregation) is untouched.
        """
        from vllm.v1.attention import icms_fetch_state
        am = (self._attn_metadata.get(layer_name)
              if isinstance(self._attn_metadata, dict) else None)
        if am is None or not hasattr(am, "block_table"):
            # Fallback: no natural metadata → set the first capture only
            # (degrades to legacy single-rid behavior for this layer).
            icms_fetch_state.set_active(captures[0][1])
            return
        natural_bt = am.block_table
        natural_sl = am.seq_lens
        # Clone so we don't mutate the engine's tensors. Both are GPU
        # int32 typically; .clone() preserves dtype/device.
        combined_bt = natural_bt.clone()
        combined_sl = natural_sl.clone()
        # ICMS_DIAG_AGG=1: log per-call shapes + per-rid req_idx so we
        # can pinpoint the "batch_size must be equal to batch_size_k"
        # crash that surfaces after a few batches at TP=2 (2026-05-05).
        _diag_agg = os.environ.get("ICMS_DIAG_AGG", "0") == "1"
        # Per-row override.
        skipped: list = []
        for req_idx, state in captures:
            trim_row = state.block_table  # [1, k+c] tensor
            if trim_row.dim() == 2:
                trim_row = trim_row[0]
            k_plus_c = int(trim_row.shape[0])
            # Audit #7 fix (2026-05-11): the seq_lens write must move
            # together with the block_table write. Pre-fix, a skipped bt
            # row still got `combined_sl[req_idx] = trim_sl_val` →
            # FA's tile scheduler read up to the (smaller) trim_sl_val
            # against the (untrimmed) natural block_table for that row.
            # Latent at paper config (_use_batch_path is False at
            # max_num_seqs=1); kept correct for batched / TP>1 callers.
            _bt_in_range = (req_idx < combined_bt.shape[0]
                            and k_plus_c <= combined_bt.shape[1])
            if _bt_in_range:
                combined_bt[req_idx, :k_plus_c] = trim_row
            else:
                skipped.append((req_idx, k_plus_c))
            trim_sl = state.seq_lens
            if trim_sl.dim() >= 1:
                trim_sl_val = trim_sl[0]
            else:
                trim_sl_val = trim_sl
            if _bt_in_range and req_idx < combined_sl.shape[0]:
                combined_sl[req_idx] = trim_sl_val
        # 2026-05-07: tried `int(combined_sl.max().item())` here based on
        # an audit hypothesis that natural max_seq_len caused FA to
        # read past the trimmed region. Empirically REGRESSED accuracy
        # at every budget on llama3 batched (0.60→0.30 at b=0.05,
        # 0.125→0.0 at b=0.20). Per-example: first 4-5 decode tokens
        # decoded CORRECTLY (matching haystack variable codes) then
        # the continuation went incoherent — suggests stride-boundary
        # interaction we don't yet understand. Reverted to natural max.
        # The trimmed-seq-len hypothesis may still be right but the
        # naive .max() fix isn't the answer.
        # Diag knobs:
        #   ICMS_USE_TRIM_MAXSEQ=1   → use combined_sl.max() (the regressed fix)
        #   ICMS_FORCE_MAXSEQ_SYNC=1 → keep natural max but trigger
        #                              the .item() CUDA sync (probe D)
        _diag_maxseq = os.environ.get("ICMS_USE_TRIM_MAXSEQ") == "1"
        _diag_force_sync = os.environ.get("ICMS_FORCE_MAXSEQ_SYNC") == "1"
        if _diag_maxseq:
            max_seq_len = int(combined_sl.max().item())
        elif _diag_force_sync:
            # Probe D: trigger the sync side-effect without using the
            # smaller value. Discriminates "regression was the sync"
            # from "regression was the smaller value."
            _ = int(combined_sl.max().item())  # sync-only
            max_seq_len = int(getattr(am, "max_seq_len", 0))
        else:
            max_seq_len = int(getattr(am, "max_seq_len", 0))
        # KV pointers are per-layer constants; all captures share them.
        head_state = captures[0][1]
        if _diag_agg:
            cap_summary = [(int(req_idx),
                            tuple(state.block_table.shape),
                            tuple(state.seq_lens.shape))
                           for req_idx, state in captures]
            logger.info(
                "[diag-agg] tp=%d layer=%s nat_bt=%s nat_sl=%s "
                "combined_bt=%s combined_sl=%s max_seq_len=%d "
                "captures=%s skipped=%s",
                self._tp_rank, layer_name,
                tuple(natural_bt.shape), tuple(natural_sl.shape),
                tuple(combined_bt.shape), tuple(combined_sl.shape),
                max_seq_len, cap_summary, skipped)
        icms_fetch_state.set_active(icms_fetch_state.IcmsFetchState(
            key_cache=head_state.key_cache,
            value_cache=head_state.value_cache,
            block_table=combined_bt,
            seq_lens=combined_sl,
            max_seq_len=max_seq_len,
        ))

    def _get_sink_pages(self, sink, kv_page_bytes: int):
        """Cached `sink_pages` view of the entire sink as [N, page_bytes].

        The sink's GPU buffer and shape are fixed for the connector's
        lifetime, but constructing the gpu_view + view per layer fires
        ~us of Python + CUDA driver overhead on every apply (≥48 layers
        per iter). Cache the result on first call.
        """
        if self._sink_pages_view is not None:
            return self._sink_pages_view
        sink_base = sink.gpu_view(0, sink.size)
        sink_pages = sink_base.view(-1, kv_page_bytes)
        self._sink_pages_view = sink_pages
        return sink_pages

    def _apply_selective_attention(self, layer_name: str, reply,
                                   req_idx: int, rid: str = "",
                                   _capture: list | None = None):
        # ICMS_SKIP_APPLY=1: skip the apply entirely (no scatter, no
        # bt override). Used to isolate whether the apply is the cause
        # of warm-prefix corruption — if run 2 still fails with apply
        # disabled, the bug is elsewhere (e.g., Quest hooks or save_kv).
        if os.environ.get("ICMS_SKIP_APPLY") == "1":
            return None
        # _capture: when not None, multi-rid batching path. Append the
        # would-be-set IcmsFetchState here as (req_idx, key_cache,
        # value_cache, new_bt, new_sl, new_seq_len) and skip the actual
        # set_active() call so the caller can aggregate per-rid states
        # into one multi-row state per layer. KV scatter side effects
        # (main_key.index_copy_) still happen — only set_active is
        # deferred.
        # Per-line wall-time instrumentation, gated by ICMS_LINE_TIMING=1.
        # Captures the Python-side wall time of every step including
        # H2D tensor creations, dict/list ops, and tensor casts. Used
        # to localize the ~3.6 ms/layer Python overhead at TP=2 that
        # remains after seq_len caching. NOT for production.
        _line_dbg = self._cfg_line_timing
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
        # See extract_and_record for layout discussion (gemma-3 / TRITON_ATTN
        # uses [num_blocks, 2, ...] vs standard [2, num_blocks, ...]).
        if kv.shape[0] == 2:
            main_key = kv[0]
            main_value = kv[1]
        elif kv.shape[1] == 2:
            main_key = kv[:, 0]
            main_value = kv[:, 1]
        else:
            return None

        am = self._attn_metadata.get(layer_name) if isinstance(self._attn_metadata, dict) else None
        if am is None or not hasattr(am, "block_table"):
            return None
        bt = am.block_table  # [num_reqs, max_blocks]

        if req_idx >= bt.shape[0]:
            return None

        # ICMS_TRACE: capture ORIGINAL block_table row + req_idx at apply
        # entry, BEFORE the function mutates bt[req_idx]. This is what we
        # need to detect cross-rid contamination of the logical→physical
        # page mapping in batched mode.
        if _ICMS_TRACE_ENABLED:
            try:
                _bt_row = bt[req_idx]
                _bt_first16 = [int(x) for x in _bt_row[:16].tolist()]
                _bt_total_nonzero = int((_bt_row != 0).sum().item())
                _icms_trace(
                    "apply_bt", rid, layer=self._extract_layer_idx(layer_name) or -1,
                    chain_fp="",
                    extra={
                        "req_idx": int(req_idx),
                        "bt_shape": list(bt.shape),
                        "bt_row_first16": _bt_first16,
                        "bt_row_nonzero_count": _bt_total_nonzero,
                        "layer_name": layer_name,
                    })
            except Exception:
                pass

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
        _no_seqlen_cache = self._cfg_no_seqlen_cache
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
        # 2026-05-10 TP>1 stored_groups asymmetry fix (apply-side):
        # Mirror the same all-reduce-MAX from `_score_one_request`.
        # Without it, apply's `pid >= context_pages` filter (and the
        # downstream `if pid < context_pages` slicing) uses a per-rank
        # `context_pages` that can be tens of groups smaller than the
        # other rank's. Score returns 100 pids covering 0..2015, but
        # the lagging rank's apply only accepts pids < context_pages
        # (e.g., 480) → 1 of 100 pids survives → model attends to 1
        # page of haystack → garbage.
        # Both ranks reach _apply_selective_attention symmetrically
        # per-layer (called from set_active during attention), so the
        # all-reduce is NCCL-safe.
        if self._tp_size > 1:
            effective_groups = _tp_allreduce_max_int(
                effective_groups, self._tp_size)
        context_pages = effective_groups * _GROUP_BLOCKS
        context_tokens = context_pages * PAGE_TOKENS
        total_blocks = (seq_len + PAGE_TOKENS - 1) // PAGE_TOKENS

        # ─── FAST PATH: reuse the scored layer's cached apply on
        # subsequent reuse layers within the same stride. The block_table,
        # seq_len, valid pids, and phys_blocks are stride-invariant; only
        # the per-layer sink offsets shift by `delta * cached_actual_k`
        # pages. This skips the Python pid sort + filter + dict +
        # pinned-tensor build at every reuse layer (5 of every 6 layers).
        # The actual GPU memcpy (gather + slice + scatter) still runs.
        layer_idx_for_cache = self._extract_layer_idx(layer_name)
        cached_start = rs._apply_cached_layer_start
        if (layer_idx_for_cache is not None
                and cached_start >= 0
                and rs._apply_cached_phys_blocks_dev is not None
                and rs._apply_cached_new_bt is not None):
            delta = layer_idx_for_cache - cached_start
            # 2026-05-11 audit fix #21: invalidate the cached new_bt
            # if the cumulative-page set grew between the slow path's
            # bake (at the stride-root layer) and now. Without this,
            # cross-iter async FetchAll replies adding pages mid-stride
            # leave the cached block_table missing pages that ARE in
            # main_key → attention reads silent KV gap → garbled
            # tokens (anomaly C signature: mistral-small h128k
            # niah_single_{1,2,3} fails identically across budgets).
            _stride_root_chk = (
                (cached_start // self._score_stride) * self._score_stride)
            _live_cum = len(rs.fetched_pages.get(_stride_root_chk, set()))
            _cached_cum = rs._apply_cached_cumulative_count
            _fast_path_valid = (_live_cum == _cached_cum)
            if 0 < delta < self._score_stride and _fast_path_valid:
                cached_k_pages = rs._apply_cached_actual_k
                page_idx_dev = (rs._apply_cached_page_idx_dev
                                + (delta * cached_k_pages))
                phys_blocks_dev = rs._apply_cached_phys_blocks_dev

                geom = self._geom
                sink = self._sink_pool.sink
                per_rank_slice = (self._tp_size > 1 and self._cfg_per_rank_slice)
                nkv_local_runtime = (
                    geom.num_kv_heads // self._tp_size
                    if self._tp_size > 1 else geom.num_kv_heads)
                if per_rank_slice:
                    kv_page_bytes_eff = geom.kv_page_bytes // self._tp_size
                    kv_page_bytes = geom.kv_page_bytes
                    half_bytes = kv_page_bytes_eff // 2
                    page_shape = (PAGE_TOKENS, nkv_local_runtime, geom.head_dim)
                else:
                    kv_page_bytes = geom.kv_page_bytes
                    half_bytes = kv_page_bytes // 2
                    page_shape = (PAGE_TOKENS, geom.num_kv_heads, geom.head_dim)
                model_dtype = main_key.dtype
                device = main_key.device

                if self._cfg_apply_stream and not hasattr(self, "_apply_stream"):
                    self._apply_stream = torch.cuda.Stream(device=device)
                apply_stream = (self._apply_stream
                                if self._cfg_apply_stream else None)
                default_stream = torch.cuda.current_stream(device)
                if apply_stream is not None:
                    in_event = torch.cuda.Event()
                    in_event.record(default_stream)
                    apply_stream.wait_event(in_event)
                stream_ctx = (torch.cuda.stream(apply_stream)
                              if apply_stream is not None
                              else nullcontext())

                with stream_ctx:
                    sink_pages = self._get_sink_pages(sink, kv_page_bytes)
                    pages_u8 = sink_pages.index_select(0, page_idx_dev)
                    if per_rank_slice:
                        pages_u8 = pages_u8[:, :kv_page_bytes_eff].contiguous()
                    k_bytes = pages_u8[:, :half_bytes].contiguous()
                    v_bytes = pages_u8[:, half_bytes:].contiguous()
                    # Bug 11 family fix (2026-04-30): bytes stored in
                    # model_dtype now. Mirror of the slow-path change.
                    k_pages = (k_bytes.view(model_dtype)
                                .reshape(-1, *page_shape))
                    v_pages = (v_bytes.view(model_dtype)
                                .reshape(-1, *page_shape))
                    if self._tp_size > 1 and not per_rank_slice:
                        nkv_local = geom.num_kv_heads // self._tp_size
                        start = self._tp_rank * nkv_local
                        if (os.environ.get("ICMS_DIAG_TP_APPLY", "0") == "1"
                                and not getattr(self, "_diag_tp_apply_fast_fired", False)):
                            try:
                                _pre = bytes(k_pages[0, 0, 0].view(torch.uint8).cpu().numpy().tobytes()[:16]).hex()
                                _sl = k_pages[:, :, start:start + nkv_local, :].contiguous()
                                _post = bytes(_sl[0, 0, 0].view(torch.uint8).cpu().numpy().tobytes()[:16]).hex()
                                logger.info(
                                    "[diag-tp-apply-fast] rank=%d nkv_total=%d "
                                    "nkv_local=%d start=%d k_pages_shape=%s "
                                    "pre_slice_h0_b16=%s post_slice_h%d_b16=%s",
                                    self._tp_rank, geom.num_kv_heads,
                                    nkv_local, start, list(k_pages.shape),
                                    _pre, start, _post)
                                self._diag_tp_apply_fast_fired = True
                            except Exception as _e:
                                logger.warning("diag-tp-apply-fast failed: %s", _e)
                        k_pages = k_pages[:, :, start:start + nkv_local, :].contiguous()
                        v_pages = v_pages[:, :, start:start + nkv_local, :].contiguous()
                    # ICMS_DIAG_KV_DIFF=1: compare prefix-cache KV vs the
                    # ICMS-fetched KV at the SAME phys_block before
                    # index_copy_ overwrites. If hashes differ, ICMS-stored
                    # KV diverges from what vLLM prefilled — explains the
                    # b=1.0 < b=0.50 inversion (full overwrite at b=1.0
                    # corrupts more pages than top-K overwrite at b=0.50).
                    if (os.environ.get("ICMS_DIAG_KV_DIFF") == "1"
                            and phys_blocks_dev.numel() > 0):
                        try:
                            import hashlib as _h_kvd
                            _kvd_layer = self._extract_layer_idx(layer_name)
                            if _kvd_layer in (0, 6, 12, 18, 24, 30, 36, 42):
                                _phys0 = int(phys_blocks_dev[0].item())
                                _pre_k = main_key[_phys0].cpu().numpy().tobytes()
                                _new_k = k_pages[0].cpu().numpy().tobytes()
                                _pre_v = main_value[_phys0].cpu().numpy().tobytes()
                                _new_v = v_pages[0].cpu().numpy().tobytes()
                                _pre_kh = _h_kvd.sha1(_pre_k).hexdigest()[:16]
                                _new_kh = _h_kvd.sha1(_new_k).hexdigest()[:16]
                                _pre_vh = _h_kvd.sha1(_pre_v).hexdigest()[:16]
                                _new_vh = _h_kvd.sha1(_new_v).hexdigest()[:16]
                                _diff_k = _pre_kh != _new_kh
                                _diff_v = _pre_vh != _new_vh
                                # First-bytes hex sample to see content
                                _pre_k_head = _pre_k[:16].hex()
                                _new_k_head = _new_k[:16].hex()
                                logger.info(
                                    "[diag-kv-diff] rid=%s layer=%d phys=%d "
                                    "k_pre=%s k_new=%s diff_k=%s "
                                    "v_pre=%s v_new=%s diff_v=%s "
                                    "k_pre_head=%s k_new_head=%s",
                                    rid[:8], _kvd_layer, _phys0,
                                    _pre_kh, _new_kh, _diff_k,
                                    _pre_vh, _new_vh, _diff_v,
                                    _pre_k_head, _new_k_head)
                        except Exception as _e_kvd:
                            logger.warning("diag-kv-diff failed: %s", _e_kvd)
                    main_key.index_copy_(0, phys_blocks_dev, k_pages)
                    main_value.index_copy_(0, phys_blocks_dev, v_pages)
                    # KV provenance: record blocks ICMS just populated for
                    # this layer. Cheap no-op when env flag is off.
                    if icms_provenance.is_enabled():
                        try:
                            _pb_list = phys_blocks_dev.tolist()
                            icms_provenance.tracker().record_icms_populated(
                                rid=rid,
                                layer_idx=layer_idx_for_cache,
                                block_ids=_pb_list,
                            )
                        except Exception:
                            pass

                # Multi-layer canary read in fast path (2026-04-30):
                # detect mis-pack of layers 1..47 in the FAPS sink. Slow
                # path only fires at layer 0, so without this block we'd
                # never see hashes for higher layers.
                _fast_layer_idx = self._extract_layer_idx(layer_name)
                if (os.environ.get("ICMS_DIAG_CANARY") == "1"
                        and _fast_layer_idx in
                        (6, 12, 18, 24, 30, 36, 42)):
                    import hashlib as _hl_fast
                    chain_head = (rs.chain[:1]
                                   if rs is not None and rs.chain else [])
                    for probe_pid in (0, 17, 100):
                        try:
                            canary_idx = (rs._apply_cached_valid_pids.index(probe_pid)
                                          if hasattr(rs, "_apply_cached_valid_pids")
                                          and rs._apply_cached_valid_pids is not None
                                          else None)
                        except (ValueError, AttributeError):
                            canary_idx = None
                        if canary_idx is None or canary_idx >= k_bytes.shape[0]:
                            continue
                        k_raw = k_bytes[canary_idx].cpu().numpy().tobytes()
                        v_raw = v_bytes[canary_idx].cpu().numpy().tobytes()
                        kh = _hl_fast.sha1(k_raw).hexdigest()[:16]
                        vh = _hl_fast.sha1(v_raw).hexdigest()[:16]
                        khead = k_raw[:32].hex()
                        nz = int((k_pages[canary_idx] != 0).sum())
                        nt = int(k_pages[canary_idx].numel())
                        logger.info(
                            "[diag-canary-read] rid=%s chain_head=%s "
                            "layer=%d pid=%d canary_idx=%d "
                            "nonzero_k=%d/%d k_sha=%s v_sha=%s "
                            "k_head=%s",
                            rid, chain_head, _fast_layer_idx,
                            probe_pid, canary_idx, nz, nt, kh, vh, khead)

                if apply_stream is not None:
                    out_event = torch.cuda.Event()
                    out_event.record(apply_stream)
                    default_stream.wait_event(out_event)

                if os.environ.get("ICMS_SKIP_BT_OVERRIDE") != "1":
                    from vllm.v1.attention import icms_fetch_state
                    _state_fast = icms_fetch_state.IcmsFetchState(
                        key_cache=main_key,
                        value_cache=main_value,
                        block_table=rs._apply_cached_new_bt,
                        seq_lens=rs._apply_cached_new_sl,
                        max_seq_len=rs._apply_cached_max_seq_len,
                    )
                    if _capture is not None:
                        _capture.append((req_idx, _state_fast))
                    else:
                        icms_fetch_state.set_active(_state_fast)
                return True

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
        # ICMS_DIAG_APPLY=1: log layer-0 reply distribution to detect
        # duplicate pids (which dedup last-wins, possibly overwriting
        # real offsets with phantom-group zeros), wrong-layer payloads,
        # and to reconcile reply_n_pages vs expected K-page count.
        # ICMS_DIAG_KSCALE=1: also count dups across ALL layers (not
        # just layer 0) — Agent suspect #3 says dup probability rises
        # at higher layers + higher k.
        _diag_apply = (os.environ.get("ICMS_DIAG_APPLY") == "1"
                       and layer_idx_for_cache == 0)
        _diag_kscale = os.environ.get("ICMS_DIAG_KSCALE") == "1"
        _dup_count = 0
        _seen_pids: set = set()
        _track_dups = _diag_apply or _diag_kscale
        for i, pid in enumerate(reply.page_ids):
            if i < len(reply.sink_offsets):
                if _track_dups and pid in _seen_pids:
                    _dup_count += 1
                pid_to_sink_off[pid] = reply.sink_offsets[i]
                if _track_dups:
                    _seen_pids.add(pid)
        if _diag_kscale and _dup_count > 0:
            logger.warning(
                "[diag-kscale-replydup] rid=%s layer=%d n=%d uniq=%d "
                "dup=%d — server returned duplicate pids; last-wins "
                "dedup may overwrite real offsets with phantom-group "
                "zeros at higher layers / higher k",
                rid, layer_idx_for_cache,
                len(reply.page_ids), len(_seen_pids), _dup_count)
        if _diag_apply:
            _pid_list = list(reply.page_ids)
            _off_list = list(reply.sink_offsets)
            _n = len(_pid_list)
            kpb_log = self._geom.kv_page_bytes if self._geom else 0
            _pid_min = min(_pid_list) if _pid_list else -1
            _pid_max = max(_pid_list) if _pid_list else -1
            _off_min = min(_off_list) if _off_list else -1
            _off_max = max(_off_list) if _off_list else -1
            # Show pid+off pairs at head/tail.
            _pairs_h = list(zip(_pid_list[:8], _off_list[:8]))
            _pairs_t = list(zip(_pid_list[-8:], _off_list[-8:])) \
                if _n > 8 else []
            # Slot bucketing: how many pids per (offset // kv_page_bytes
            # // GROUP_BLOCKS) — i.e., per "group slot" in the sink.
            slot_counts: dict[int, int] = {}
            if kpb_log > 0 and _GROUP_BLOCKS > 0:
                for off in _off_list:
                    slot = (off // kpb_log) // _GROUP_BLOCKS
                    slot_counts[slot] = slot_counts.get(slot, 0) + 1
            logger.info(
                "[diag-apply-reply] rid=%s layer=0 n=%d uniq_pids=%d "
                "dup_pids=%d pid_range=[%d..%d] off_range=[%d..%d] "
                "kv_page_bytes=%d slot_counts=%s "
                "pairs_head=%s pairs_tail=%s",
                rid, _n, len(_seen_pids), _dup_count,
                _pid_min, _pid_max, _off_min, _off_max,
                kpb_log, sorted(slot_counts.items()),
                _pairs_h, _pairs_t)

        geom = self._geom
        sink = self._sink_pool.sink

        # ICMS_DIAG_APPLY=1: read first 32 bytes from sink at the START
        # of each 32-page slot (slots 0, 1, 2 — the three groups the
        # server is returning). Pair with [diag-canary-write] on run 1
        # to identify which slot holds which group's K data. If slot 0
        # matches pid=0's write canary, slot 0 has g1's real data and
        # apply is correct. If slot 0 matches a different pid's canary
        # (e.g., g2's first page), the server enumerated trie nodes in
        # the wrong order.
        if _diag_apply:
            try:
                import hashlib as _hl_diag
                kpb_log2 = self._geom.kv_page_bytes if self._geom else 0
                bytes_per_slot = _GROUP_BLOCKS * kpb_log2
                # Try to read sink bytes via gpu-direct or host buffer.
                _sink_obj = self._sink_pool.sink
                # Use the ABI: read_bytes(offset, length) if available;
                # fall back to torch tensor view of the sink if not.
                _read_fn = getattr(_sink_obj, "read_bytes", None)
                slot_sigs = []
                for slot_i in range(3):
                    base = slot_i * bytes_per_slot
                    if _read_fn is not None:
                        head32 = bytes(_read_fn(base, 32))
                    else:
                        # Fallback: torch tensor of the sink as uint8.
                        sp_local = self._get_sink_pages(_sink_obj, kpb_log2)
                        head32 = bytes(sp_local[slot_i * _GROUP_BLOCKS, :32]
                                       .cpu().numpy())
                    sha = _hl_diag.sha1(head32).hexdigest()[:16]
                    slot_sigs.append(
                        f"slot{slot_i}@off={base}:k_sha={sha}:"
                        f"head32={head32.hex()}")
                logger.info("[diag-apply-sink-slots] rid=%s layer=0 %s",
                            rid, " | ".join(slot_sigs))
            except Exception as _e:
                logger.warning("[diag-apply-sink-slots] failed: %r", _e)

        # ICMS_PER_RANK_SLICE=1 (server-side Option Y): the server has
        # already gathered THIS rank's nkv_local heads and packed them
        # at the start of each page slot in the sink. The remote-page
        # layout becomes (PAGE_TOKENS, nkv_local, head_dim) per K and V,
        # halving wire bandwidth + GPU memcpy. Skip the read-time slice
        # below in that mode.
        per_rank_slice = (self._tp_size > 1 and self._cfg_per_rank_slice)
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
        # Audit #20 fix (2026-05-11): MIN-reduce bt_row_max across TP
        # ranks so every rank uses the same bound when filtering pids.
        # If vLLM ever allocates differently sized bt rows per rank for
        # the same rid, post-filter valid_pids would otherwise diverge
        # → asymmetric scatter / NCCL shape mismatch downstream. At
        # paper config (max_num_seqs=1, single rid) shapes are
        # symmetric in practice so this is a no-op; the call enforces
        # the contract explicitly for TP>1 future-proofing.
        # TP=1 → identity (helper short-circuits at tp_size<=1).
        bt_row_max = _tp_allreduce_min_int(bt_row_max, self._tp_size)
        for pid in selected:
            off = pid_to_sink_off.get(pid)
            if off is None or pid >= bt_row_max:
                continue
            valid_pids.append(pid)
            valid_sink_offs.append(off)
        if not valid_pids:
            return None
        # ICMS_DIAG_KSCALE=1: probe for k-scaling apply-path bugs
        # (Agent suspect #3 — duplicate dest indices in valid_pids
        # would cause undefined index_copy_ behavior, scrambling K
        # data placement). Fires across all layers, not just layer 0.
        if os.environ.get("ICMS_DIAG_KSCALE") == "1":
            _vp_len = len(valid_pids)
            _vp_uniq = len(set(valid_pids))
            if _vp_len != _vp_uniq:
                logger.warning(
                    "[diag-kscale-dup] rid=%s layer=%d k=%d valid_pids=%d "
                    "uniq=%d dup_count=%d — index_copy_ dest is "
                    "non-unique → scrambled main_key writes",
                    rid, layer_idx_for_cache, k,
                    _vp_len, _vp_uniq, _vp_len - _vp_uniq)
        _lt("after_python_filter")

        # M3+M4-A: during decode, attention's block_table must reference
        # ALL pages fetched so far for this stride group, not just the
        # current Score reply's slice. rs.fetched_pages[stride_root]
        # tracks the cumulative set (populated by every Score reply
        # including this one — see _score_one_request line ~2730).
        # Sink-scatter still uses only valid_pids (current reply) since
        # older pages already landed in main_key on prior iters.
        # Prefill: cumulative == current (one-shot scoring per chain), so
        # no behavior change.
        cumulative_pids: list[int] = valid_pids
        _diag_cum = (os.environ.get("ICMS_DIAG_CUM") == "1"
                     and layer_idx_for_cache == 0)
        if (self._prefill_done
                and rs is not None
                and layer_idx_for_cache is not None
                and self._score_stride > 0):
            stride_root = (layer_idx_for_cache // self._score_stride) \
                          * self._score_stride
            prior_set = rs.fetched_pages.get(stride_root, set())
            if _diag_cum:
                logger.info(
                    "[diag-cum] rid=%s layer=0 stride_root=%d valid=%d "
                    "prior=%d merged_in_range=%s",
                    rid, stride_root, len(valid_pids), len(prior_set),
                    "n/a" if not prior_set else len([
                        p for p in (prior_set | set(valid_pids))
                        if p < context_pages and p < bt_row_max]))
            if prior_set:
                merged = prior_set | set(valid_pids)
                cumulative_pids = sorted(
                    p for p in merged
                    if p < context_pages and p < bt_row_max)
        elif _diag_cum:
            logger.info(
                "[diag-cum] rid=%s layer=0 PREFILL_DONE_FALSE valid=%d",
                rid, len(valid_pids))

        if gpu_direct:
            # ── Batched GPU-direct path: one gather + one dtype convert + ──
            # ── one scatter per layer, instead of 4 kernels + 1 host-sync ──
            # ── per page (old path ran ~48k kernels + ~12k syncs per req). ──
            sink_pages = self._get_sink_pages(sink, kv_page_bytes)

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

            # ICMS_DIAG_APPLY=1: dump scatter destinations at layer 0 to
            # see whether apply targets the same physical blocks vLLM
            # uses for attention. Compare vs diag-attn output (bt[0][:8])
            # in flash_attn.py. Layer-0 only to keep log cheap.
            if (os.environ.get("ICMS_DIAG_APPLY") == "1"
                    and layer_idx_for_cache == 0):
                bt_row = bt[req_idx]
                _vp_h = valid_pids[:8]
                _vp_t = valid_pids[-8:] if len(valid_pids) > 8 else []
                _pb_h = phys_blocks_dev[:8].tolist()
                _pb_t = (phys_blocks_dev[-8:].tolist()
                         if phys_blocks_dev.numel() > 8 else [])
                _bt_h = bt_row[:8].tolist()
                _bt_pad = (bt_row[len(valid_pids):
                                  len(valid_pids)+8].tolist()
                           if len(valid_pids) < bt_row.numel() else [])
                logger.info(
                    "[diag-apply] rid=%s layer=0 req_idx=%d "
                    "n_valid=%d bt_row_len=%d ctx_pages=%d eff_grp=%d "
                    "stored_grp=%d num_grp_written=%d seq_len=%d "
                    "reply_n_pages=%d "
                    "valid_pids[:8]=%s valid_pids[-8:]=%s "
                    "phys_blocks[:8]=%s phys_blocks[-8:]=%s "
                    "bt[req_idx][:8]=%s bt[req_idx][n_valid:n_valid+8]=%s "
                    "selected_count=%d sink_off_count=%d",
                    rid, req_idx, len(valid_pids), int(bt_row.numel()),
                    context_pages, effective_groups,
                    int(rs.stored_groups), int(rs.num_groups_written),
                    int(seq_len), int(len(reply.page_ids)),
                    _vp_h, _vp_t, _pb_h, _pb_t, _bt_h, _bt_pad,
                    len(selected), len(pid_to_sink_off))

            # M3+M4-A: build a parallel phys_blocks tensor for the
            # cumulative pid set (used for new_bt only). Falls through
            # to phys_blocks_dev when cumulative == current to avoid
            # the extra H2D + index_select on the prefill path.
            if cumulative_pids is valid_pids or cumulative_pids == valid_pids:
                phys_blocks_for_bt_dev = phys_blocks_dev
            else:
                _cum_pids_cpu = torch.tensor(
                    cumulative_pids, dtype=torch.int64, pin_memory=True)
                cum_pids_dev = _cum_pids_cpu.to(device, non_blocking=True)
                phys_blocks_for_bt_dev = bt[req_idx].index_select(
                    0, cum_pids_dev).to(torch.int64)
            _lt("after_phys_blocks_for_bt")

            # Bound check: phys_block < main_key.shape[0]. The .item() is a
            # CPU↔GPU sync; never observed to fire in practice. Gated by
            # ICMS_SKIP_BOUNDS=1 to disable. Empirically (2026-04-27 audit)
            # leaving it on costs <2 ms per iter — keep enabled for safety.
            if not self._cfg_skip_bounds:
                max_blocks_hbm = main_key.shape[0]
                # Same-tensor sentinel: in the cumulative == valid case
                # (the prefill path) phys_blocks_for_bt_dev IS the SAME
                # tensor object as phys_blocks_dev.  After masking
                # phys_blocks_dev below, the variable is reassigned to a
                # NEW tensor — phys_blocks_for_bt_dev still points at the
                # OLD unfiltered one, so OOB blocks would land in
                # new_bt_row → FA reads garbage.  Detect identity now so
                # we can resync after the mask.
                shared_with_bt = phys_blocks_for_bt_dev is phys_blocks_dev
                if bool((phys_blocks_dev >= max_blocks_hbm).any().item()):
                    bounds_mask = phys_blocks_dev < max_blocks_hbm
                    phys_blocks_dev = phys_blocks_dev[bounds_mask]
                    page_idx_dev = page_idx_dev[bounds_mask]
                    if phys_blocks_dev.numel() == 0:
                        return None
                    # 2026-05-10 audit fix #22: keep phys_blocks_for_bt_dev
                    # in sync.  Without this, the existing bound check
                    # silently leaked OOB phys blocks into new_bt_row →
                    # FA's attention block_table referenced past-end GPU
                    # pages (uninitialized HBM read).
                    if shared_with_bt:
                        phys_blocks_for_bt_dev = phys_blocks_dev
                # 2026-05-10 audit fix #22 (cumulative != valid path):
                # M3+M4-A decode-mode builds phys_blocks_for_bt_dev from
                # cumulative_pids (a superset of valid_pids).  The bound
                # check on phys_blocks_dev does NOT cover it; OOB blocks
                # in the cumulative set would silently land in new_bt_row
                # — FA reads garbage at those phys blocks.
                if not shared_with_bt:
                    if bool((phys_blocks_for_bt_dev >= max_blocks_hbm)
                            .any().item()):
                        bt_bounds_mask = (
                            phys_blocks_for_bt_dev < max_blocks_hbm)
                        phys_blocks_for_bt_dev = (
                            phys_blocks_for_bt_dev[bt_bounds_mask])
                        if phys_blocks_for_bt_dev.numel() == 0:
                            return None

            # ─── Dedicated apply stream (ICMS_APPLY_STREAM=1) ─────────
            # Optional alternate-stream dispatch for the apply gather +
            # scatter, intended to overlap NCCL on the default stream.
            # 2026-04-27 audit: provides no measurable win after seq_len
            # cache + per-stride apply cache landed; left in place but
            # off by default. Brackets with in/out events for correctness:
            #   - in_event: apply stream waits for default stream's
            #     prior writes (block_table mods, sink data, etc.).
            #   - out_event: default stream waits for apply's writes
            #     before the NEXT layer's attention reads main_key/_value.
            _apply_dbg = self._cfg_apply_timing
            if self._cfg_apply_stream and not hasattr(self, "_apply_stream"):
                self._apply_stream = torch.cuda.Stream(device=device)
            apply_stream = (self._apply_stream
                             if self._cfg_apply_stream else None)
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
                # TP=2 sink probe (2026-04-30): dump first 32 bytes of
                # pages_u8[0] per-rank. Lets us discriminate "rank-1 sink
                # is empty" (PRS=0 Option-W bug) vs "ranks see different
                # but non-zero bytes" (PRS=1 server-slicing bug). Layer 0
                # only.
                if (os.environ.get("ICMS_DIAG_TP_SINK") == "1"
                        and self._tp_size > 1
                        and layer_name.endswith(".0.self_attn.attn")):
                    try:
                        page0_full = pages_u8[0].contiguous().cpu().numpy()
                        head_b = bytes(page0_full[:32])
                        nonzero = sum(1 for b in head_b if b != 0)
                        # FNV-1a over the full 32K page slot (matches
                        # server-canary-pre-strided-write algorithm so
                        # we can sanity-diff what the server posted vs.
                        # what arrived at the sink).
                        h_full = 0xcbf29ce484222325
                        for _b in page0_full.tobytes():
                            h_full = ((h_full ^ _b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
                        # Also hash just the rank-local valid window
                        # ([0..kv_page_bytes_eff)) — what the apply
                        # actually consumes under PRS=1.
                        if per_rank_slice:
                            valid = page0_full[:kv_page_bytes_eff]
                        else:
                            valid = page0_full
                        h_valid = 0xcbf29ce484222325
                        for _b in valid.tobytes():
                            h_valid = ((h_valid ^ _b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
                        # Tail probe: is bytes [16K..32K) of the slot
                        # zeros (correct under PRS=1 with valid-window-
                        # only writes), or is it populated (suggests
                        # server is echoing full 32K)?
                        if per_rank_slice:
                            tail = page0_full[kv_page_bytes_eff:]
                            tail_nz = int((tail != 0).sum())
                        else:
                            tail_nz = -1
                        logger.info(
                            "[diag-tp-sink] rank=%d per_rank_slice=%s "
                            "page0[:32]=%s nonzero=%d/32 "
                            "kv_page_bytes_eff=%s "
                            "h_full=%016x h_valid=%016x tail_nonzero=%d",
                            self._tp_rank,
                            per_rank_slice,
                            head_b.hex(),
                            nonzero,
                            kv_page_bytes_eff if per_rank_slice else "n/a",
                            h_full, h_valid, tail_nz)
                    except Exception as _e:
                        logger.warning("[diag-tp-sink] dump failed: %r", _e)
                # 2026-05-10 TP=2 KV-content corruption diag: hash the
                # raw sink bytes for the FIRST page at L=0 BEFORE any
                # per-rank slicing. Pair with extract_kv_hash on the
                # producer + apply_kv_hash post-scatter to localize
                # whether corruption is in extract→sink or sink→apply.
                if (_ICMS_TRACE_ENABLED
                        and self._extract_layer_idx(layer_name) == 0
                        and pages_u8.shape[0] > 0):
                    try:
                        import hashlib as _hl
                        _sink_bytes = pages_u8[0].cpu().numpy().tobytes()
                        _icms_trace(
                            "sink_kv_hash", rid, layer=0, chain_fp="",
                            extra={
                                "sink_byte_count": int(len(_sink_bytes)),
                                "sink_first_8_hex": _sink_bytes[:8].hex(),
                                "sink_first_128_hash": _hl.sha1(
                                    _sink_bytes[:128]).hexdigest()[:16],
                                "sink_full_hash": _hl.sha1(
                                    _sink_bytes).hexdigest()[:16],
                                "sink_zero_count_first_128": int(sum(
                                    1 for b in _sink_bytes[:128] if b == 0)),
                                "kv_page_bytes_full": int(kv_page_bytes),
                                "kv_page_bytes_eff": int(kv_page_bytes_eff)
                                    if per_rank_slice else -1,
                                "per_rank_slice": bool(per_rank_slice),
                                "tp_rank": int(self._tp_rank),
                                "tp_size": int(self._tp_size),
                            })
                    except Exception:
                        pass
                if per_rank_slice:
                    # Sink slot is full-size but only the first
                    # kv_page_bytes_eff bytes hold valid data (this
                    # rank's slice). K/V split is at half of effective.
                    pages_u8 = pages_u8[:, :kv_page_bytes_eff].contiguous()
                k_bytes = pages_u8[:, :half_bytes].contiguous()
                v_bytes = pages_u8[:, half_bytes:].contiguous()
                _t2 = _t()
                # Bug 11 family fix (2026-04-30): record_page now stores
                # raw model_dtype bytes (was fp16-cast). View directly
                # as model_dtype with no precision-losing cast. The
                # legacy fp16 round-trip lost ~3 mantissa bits per value
                # at the bf16→fp16→bf16 boundary; that drift accumulated
                # over ~745M values per request at 32k chain and derailed
                # multi-key NIAH retrieval. Both bf16 and fp16 are 2
                # bytes/element so the byte count is identical — only the
                # byte-pattern interpretation changes. Write-side change:
                # record_page line ~4691; mirror in fast path: line ~3448
                # and host-sink fallback: line ~3957.
                k_pages = (k_bytes.view(model_dtype)
                            .reshape(-1, *page_shape))
                v_pages = (v_bytes.view(model_dtype)
                            .reshape(-1, *page_shape))
                _t3 = _t()

                # ICMS_DIAG_CANARY=1: read-side fingerprint for layer 0
                # pages 0, 17, 100, 500 (whichever exist in valid_pids).
                # Pair with [diag-canary-write] in record_page.
                # Multi-layer canary read (2026-04-30): extend to scored
                # layers 6,12,18,24,30,36,42 so we get write↔read hash
                # comparisons across the full FAPS reuse range. This is
                # how we detect server-side per-layer mis-packing in
                # the sink (which would show "layer 0 matches but layer
                # 6 doesn't").
                _layer_idx_for_canary = self._extract_layer_idx(layer_name)
                if (os.environ.get("ICMS_DIAG_CANARY") == "1"
                        and _layer_idx_for_canary in
                        (0, 6, 12, 18, 24, 30, 36, 42)):
                    # Layer-0 only: sink-slot bytes + offset listing (layout-
                    # specific to the first FAPS slow-path slice).
                    if _layer_idx_for_canary == 0:
                        import hashlib as _hl
                        for slot in (0, 1, 2, 17, 100, 500):
                            if slot >= sink_pages.shape[0]:
                                continue
                            slot_bytes = bytes(
                                sink_pages[slot, :32].cpu().numpy())
                            logger.info(
                                "[diag-sink-slot] layer=0 sink_slot=%d "
                                "first32_hex=%s",
                                slot, slot_bytes.hex())
                        logger.info(
                            "[diag-sink-offs] layer=0 "
                            "valid_sink_offs[:5]=%s page_idx_dev[:5]=%s",
                            valid_sink_offs[:5],
                            page_idx_dev[:5].cpu().tolist())
                    import hashlib
                    chain_head = rs.chain[:1] if rs is not None and rs.chain else []
                    # Layer 0 keeps full probe set; higher layers sample
                    # a few pids to cap log volume across 7 extra layers.
                    _probe_set = (
                        (0, 1, 2, 3, 4, 8, 16, 17, 24, 31, 32, 33, 100, 500)
                        if _layer_idx_for_canary == 0
                        else (0, 17, 100))
                    for probe_pid in _probe_set:
                        if probe_pid not in valid_pids:
                            continue
                        canary_idx = valid_pids.index(probe_pid)
                        # Bug 11 verification (2026-04-30): hash the RAW
                        # wire bytes (uint8 view, pre-model_dtype
                        # interpretation), so write/read sha1 are
                        # directly comparable regardless of how the
                        # bytes are reinterpreted on either side.
                        k_raw = k_bytes[canary_idx].cpu().numpy().tobytes()
                        v_raw = v_bytes[canary_idx].cpu().numpy().tobytes()
                        kh = hashlib.sha1(k_raw).hexdigest()[:16]
                        vh = hashlib.sha1(v_raw).hexdigest()[:16]
                        khead = k_raw[:32].hex()
                        nonzero = int((k_pages[canary_idx] != 0).sum())
                        n_total = int(k_pages[canary_idx].numel())
                        logger.info("[diag-canary-read] rid=%s chain_head=%s "
                                     "layer=%s pid=%d canary_idx=%d "
                                     "nonzero_k=%d/%d k_sha=%s v_sha=%s "
                                     "k_head=%s",
                                     rid, chain_head,
                                     _layer_idx_for_canary,
                                     probe_pid, canary_idx,
                                     nonzero, n_total,
                                     kh, vh, khead)

                # Option W broadcast path: when server didn't slice,
                # each rank still extracts its head range here.
                if self._tp_size > 1 and not per_rank_slice:
                    nkv_local = geom.num_kv_heads // self._tp_size
                    start = self._tp_rank * nkv_local
                    if (os.environ.get("ICMS_DIAG_TP_APPLY", "0") == "1"
                            and not getattr(self, "_diag_tp_apply_fired", False)):
                        try:
                            _pre = bytes(k_pages[0, 0, 0].view(torch.uint8).cpu().numpy().tobytes()[:16]).hex()
                            _sl = k_pages[:, :, start:start + nkv_local, :].contiguous()
                            _post = bytes(_sl[0, 0, 0].view(torch.uint8).cpu().numpy().tobytes()[:16]).hex()
                            logger.info(
                                "[diag-tp-apply-slow] rank=%d nkv_total=%d "
                                "nkv_local=%d start=%d k_pages_shape=%s "
                                "pre_slice_h0_b16=%s post_slice_h%d_b16=%s",
                                self._tp_rank, geom.num_kv_heads,
                                nkv_local, start, list(k_pages.shape),
                                _pre, start, _post)
                            self._diag_tp_apply_fired = True
                        except Exception as _e:
                            logger.warning("diag-tp-apply-slow failed: %s", _e)
                    k_pages = k_pages[:, :, start:start + nkv_local, :].contiguous()
                    v_pages = v_pages[:, :, start:start + nkv_local, :].contiguous()

                # ICMS_DIAG_KV_DIFF=1: see same diag in fast path.
                # Mirror — slow path is the canonical path for first-
                # apply-of-request and many bench configs. Logs pre/
                # post hashes + first-bytes hex sample to detect ICMS-
                # stored vs prefix-cached KV divergence.
                if (os.environ.get("ICMS_DIAG_KV_DIFF") == "1"
                        and phys_blocks_dev.numel() > 0):
                    try:
                        import hashlib as _h_kvd
                        _kvd_layer = self._extract_layer_idx(layer_name)
                        if _kvd_layer in (0, 6, 12, 18, 24, 30, 36, 42):
                            # 2026-05-12 extension: also log the LAST
                            # phys_block_dev (the descending-range block
                            # for shorter-chain rids like rid_5 in the
                            # multi-rid contamination investigation).
                            # phys_blocks_dev[0] = first selected page's
                            # block; phys_blocks_dev[-1] = last selected
                            # page's block (e.g., block 659 for rid_5
                            # at valid_pid=1976 in batched mode).
                            _n_phys = int(phys_blocks_dev.numel())
                            _probe_idxs = [0]
                            if _n_phys > 1:
                                _probe_idxs.append(_n_phys - 1)
                            for _pidx in _probe_idxs:
                                _phys = int(phys_blocks_dev[_pidx].item())
                                # bf16 → uint8 view → numpy.
                                _pre_k = main_key[_phys].view(torch.uint8).cpu().numpy().tobytes()
                                _new_k = k_pages[_pidx].view(torch.uint8).cpu().numpy().tobytes()
                                _pre_v = main_value[_phys].view(torch.uint8).cpu().numpy().tobytes()
                                _new_v = v_pages[_pidx].view(torch.uint8).cpu().numpy().tobytes()
                                _pre_kh = _h_kvd.sha1(_pre_k).hexdigest()[:16]
                                _new_kh = _h_kvd.sha1(_new_k).hexdigest()[:16]
                                _pre_vh = _h_kvd.sha1(_pre_v).hexdigest()[:16]
                                _new_vh = _h_kvd.sha1(_new_v).hexdigest()[:16]
                                _diff_k = _pre_kh != _new_kh
                                _diff_v = _pre_vh != _new_vh
                                _pre_k_head = _pre_k[:16].hex()
                                _new_k_head = _new_k[:16].hex()
                                _zero_pre_k = (_pre_k == b'\x00' * len(_pre_k))
                                logger.info(
                                    "[diag-kv-diff] path=slow rid=%s layer=%d "
                                    "probe=%s/%d phys=%d k_pre=%s k_new=%s "
                                    "diff_k=%s v_pre=%s v_new=%s diff_v=%s "
                                    "pre_zero=%s k_pre_head=%s k_new_head=%s",
                                    rid[:8], _kvd_layer,
                                    "first" if _pidx == 0 else "last",
                                    _n_phys, _phys,
                                    _pre_kh, _new_kh, _diff_k,
                                    _pre_vh, _new_vh, _diff_v,
                                    _zero_pre_k,
                                    _pre_k_head, _new_k_head)
                    except Exception as _e_kvd:
                        logger.warning("diag-kv-diff slow failed: %s", _e_kvd)
                # Scatter: two kernels total for this layer.
                main_key.index_copy_(0, phys_blocks_dev, k_pages)
                main_value.index_copy_(0, phys_blocks_dev, v_pages)
                # ICMS_DIAG_FULLTRACE: per-rid per-layer apply summary
                # AFTER scatter. Logs the full list of dst phys_blocks
                # so cross-rid contamination can be detected by simple
                # set-intersection across rids' logs. For layer 0 we
                # additionally read back the K bytes at every dst phys
                # block and SHA them so we can verify the scatter
                # actually landed correctly (catches an overwrite by a
                # concurrent scatter from another rid).
                if _ICMS_FULLTRACE_ENABLED:
                    try:
                        import hashlib as _hl_ftap
                        _layer_idx_ftap = self._extract_layer_idx(layer_name)
                        _vp_list = list(valid_pids)
                        _pb_list = phys_blocks_dev.tolist()
                        _extra_ftap: dict = {
                            "req_idx": int(req_idx),
                            "n_valid_pids": int(len(_vp_list)),
                            "valid_pids_head": _vp_list[:16],
                            "valid_pids_tail": _vp_list[-16:],
                            "phys_blocks_head": [int(x) for x in _pb_list[:16]],
                            "phys_blocks_tail": [int(x) for x in _pb_list[-16:]],
                            "phys_blocks_full": [int(x) for x in _pb_list[:512]],
                            "phys_blocks_uniq": int(len(set(int(x) for x in _pb_list))),
                            "context_pages": int(context_pages),
                            "effective_groups": int(effective_groups),
                            "stored_groups": int(getattr(rs, "stored_groups", -1)),
                            "num_groups_written": int(getattr(
                                rs, "num_groups_written", -1)),
                            "seq_len": int(seq_len),
                            "bt_row_max": int(bt_row_max),
                            "tp_rank": int(self._tp_rank),
                        }
                        if (_layer_idx_ftap == 0
                                and phys_blocks_dev.numel() > 0):
                            # Sample first/mid/last dst phys blocks and SHA
                            # the post-scatter K + V bytes at each.
                            n_pb = int(phys_blocks_dev.numel())
                            _idxs = [0]
                            if n_pb > 1:
                                _idxs.append(n_pb // 2)
                            if n_pb > 2:
                                _idxs.append(n_pb - 1)
                            samples_ftap: list = []
                            for _i in _idxs:
                                _pb = int(phys_blocks_dev[_i].item())
                                _kfull = main_key[_pb].contiguous().view(
                                    torch.uint8).cpu().numpy().tobytes()
                                _vfull = main_value[_pb].contiguous().view(
                                    torch.uint8).cpu().numpy().tobytes()
                                samples_ftap.append({
                                    "scatter_idx": int(_i),
                                    "src_pid": int(_vp_list[_i])
                                        if _i < len(_vp_list) else -1,
                                    "dst_phys_block": _pb,
                                    "post_scatter_k_sha":
                                        _hl_ftap.sha1(_kfull).hexdigest()[:16],
                                    "post_scatter_v_sha":
                                        _hl_ftap.sha1(_vfull).hexdigest()[:16],
                                    "post_scatter_k_head8":
                                        _kfull[:8].hex(),
                                })
                            _extra_ftap["post_scatter_samples"] = samples_ftap
                        _icms_fulltrace(
                            "apply", rid=rid,
                            layer=int(_layer_idx_ftap or -1),
                            **_extra_ftap)
                    except Exception:
                        pass
                # KV provenance: record blocks ICMS just populated.
                if icms_provenance.is_enabled():
                    try:
                        _layer_idx_prov = self._extract_layer_idx(layer_name)
                        _pb_list = phys_blocks_dev.tolist()
                        icms_provenance.tracker().record_icms_populated(
                            rid=rid,
                            layer_idx=_layer_idx_prov,
                            block_ids=_pb_list,
                        )
                    except Exception:
                        pass
                _t4 = _t()
            _lt("after_scatter")

            if apply_stream is not None:
                # Make default stream's next ops (the next layer's
                # attention) wait for apply to finish.
                out_event = torch.cuda.Event()
                out_event.record(apply_stream)
                default_stream.wait_event(out_event)

            # 2026-05-10 TP=2 KV-content corruption diag: hash the K
            # bytes of the FIRST scattered page at L=0 only. Pair with
            # extract_kv_hash on the producer side to detect whether
            # the K bytes that were stored to ICMS match what came
            # back at apply.
            if (_ICMS_TRACE_ENABLED
                    and self._extract_layer_idx(layer_name) == 0
                    and phys_blocks_dev.numel() > 0):
                try:
                    import hashlib as _hl
                    _phys0 = int(phys_blocks_dev[0].item())
                    _kbg = main_key[_phys0].view(torch.uint8).cpu().numpy().tobytes()[:128]
                    _vbg = main_value[_phys0].view(torch.uint8).cpu().numpy().tobytes()[:128]
                    _icms_trace(
                        "apply_kv_hash", rid, layer=0, chain_fp="",
                        extra={
                            "physical_page": _phys0,
                            "n_phys_blocks": int(phys_blocks_dev.numel()),
                            "k_bytes_hash": _hl.sha1(_kbg).hexdigest()[:16],
                            "v_bytes_hash": _hl.sha1(_vbg).hexdigest()[:16],
                            "k_first_8_bytes_hex": _kbg[:8].hex(),
                            "k_shape": list(main_key[_phys0].shape),
                            "tp_rank": int(self._tp_rank),
                            "tp_size": int(self._tp_size),
                        })
                    # 2026-05-10 follow-up audit: full-page POST-scatter
                    # hash of main_key/main_value at sampled physical
                    # blocks. Closes the audit gap where we only checked
                    # K bytes pre-scatter (k_pages) — this verifies
                    # main_key actually holds what we wrote and wasn't
                    # overwritten by a concurrent scatter from another
                    # rid. Sampled at first/middle/last phys block.
                    n_pb = int(phys_blocks_dev.numel())
                    _idxs = [0]
                    if n_pb > 1:
                        _idxs.append(n_pb // 2)
                    if n_pb > 2:
                        _idxs.append(n_pb - 1)
                    samples = []
                    for _i in _idxs:
                        _pb = int(phys_blocks_dev[_i].item())
                        _kfull = main_key[_pb].view(
                            torch.uint8).cpu().numpy().tobytes()
                        _vfull = main_value[_pb].view(
                            torch.uint8).cpu().numpy().tobytes()
                        samples.append({
                            "phys_block": _pb,
                            "phys_idx": _i,
                            "k_full_hash":
                                _hl.sha1(_kfull).hexdigest()[:16],
                            "v_full_hash":
                                _hl.sha1(_vfull).hexdigest()[:16],
                        })
                    _icms_trace(
                        "apply_post_main_kv_hash", rid, layer=0,
                        chain_fp="",
                        extra={
                            "tp_rank": int(self._tp_rank),
                            "tp_size": int(self._tp_size),
                            "n_phys_blocks": n_pb,
                            "samples": samples,
                        })
                except Exception:
                    pass

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

            # M3+M4-A: filled_blocks_count drives new_seq_len which
            # bounds attention's K/V reads. Must match the cumulative
            # phys_blocks count actually exposed via new_bt — otherwise
            # attention either over-reads (garbage) or under-reads
            # (drops valid pages).
            filled_blocks_count = int(phys_blocks_for_bt_dev.numel())
            _lt("after_filled_count")

            # Build trimmed block table entirely on device — avoids a
            # per-continuation-block int(bt[...]) sync.
            cont_end = min(total_blocks, bt.shape[1])
            if cont_end > context_pages:
                # cont_idx depends only on (context_pages, cont_end),
                # both per-request constants during prefill. Cache the
                # device-side tensor on rs after the first scored
                # layer; subsequent strides reuse it directly. Saves
                # the pinned-tensor build + async-H2D per stride.
                cached_range = rs._apply_cached_cont_idx_range
                if (rs._apply_cached_cont_idx_dev is not None
                        and cached_range == (context_pages, cont_end)):
                    cont_idx = rs._apply_cached_cont_idx_dev
                else:
                    _cont_idx_cpu = torch.arange(
                        context_pages, cont_end,
                        dtype=torch.int64, pin_memory=True)
                    cont_idx = _cont_idx_cpu.to(device, non_blocking=True)
                    rs._apply_cached_cont_idx_dev = cont_idx
                    rs._apply_cached_cont_idx_range = (context_pages, cont_end)
                cont_blocks = bt[req_idx].index_select(
                    0, cont_idx).to(torch.int32)
                # M3+M4-A: cumulative phys_blocks for attention's
                # block_table. During prefill / when cumulative ==
                # current, phys_blocks_for_bt_dev is the same tensor
                # as phys_blocks_dev — zero-cost.
                new_bt_row = torch.cat(
                    [phys_blocks_for_bt_dev.to(torch.int32), cont_blocks])
            else:
                new_bt_row = phys_blocks_for_bt_dev.to(torch.int32)
            if new_bt_row.numel() == 0:
                return None
            new_bt = new_bt_row.unsqueeze(0)
            _lt("after_bt_build")

            # ICMS_DIAG_SPDD_COMPREHENSIVE=1 (2026-05-15): log every apply's
            # full routing state — valid_pids → phys_blocks (scatter dest),
            # bt[req_idx][valid_pids] (what FA WOULD read at those pids),
            # and new_bt (what FA actually reads after set_active). Paired
            # SPDD-with vs SPDD-without-FetchAll runs differ exactly in
            # whether _pending_scores[0]+_pending_reuse[1..47] are populated,
            # so diffing these lines per-layer reveals what FetchAll changes
            # in the apply pipeline.
            if os.environ.get("ICMS_DIAG_SPDD_COMPREHENSIVE") == "1":
                try:
                    _is_decode = bool(getattr(rs, "_post_dense_iter", -1) >= 0
                                       or self._prefill_done)
                    _layer_idx = layer_idx_for_cache
                    _vp_n = len(valid_pids)
                    _pb_cpu = phys_blocks_dev.cpu().tolist()
                    _vp_head = valid_pids[:8]
                    _vp_tail = valid_pids[-8:] if _vp_n > 8 else []
                    _pb_head = _pb_cpu[:8]
                    _pb_tail = _pb_cpu[-8:] if len(_pb_cpu) > 8 else []
                    _bt_row_full = bt[req_idx].cpu().tolist()
                    _bt_at_vp = [_bt_row_full[p] if p < len(_bt_row_full)
                                  else -1 for p in valid_pids[:8]]
                    _bt_at_vp_tail = ([_bt_row_full[p] if p < len(_bt_row_full)
                                        else -1 for p in valid_pids[-8:]]
                                       if _vp_n > 8 else [])
                    _new_bt_cpu = new_bt.cpu().tolist()[0]
                    _new_bt_head = _new_bt_cpu[:8]
                    _new_bt_tail = (_new_bt_cpu[-8:]
                                     if len(_new_bt_cpu) > 8 else [])
                    # Mismatch detector: pb[i] should == bt[req_idx][vp[i]].
                    _mismatches = []
                    for _i, _p in enumerate(valid_pids[:16]):
                        _expect = (_bt_row_full[_p]
                                    if _p < len(_bt_row_full) else -1)
                        _actual = (_pb_cpu[_i] if _i < len(_pb_cpu)
                                    else -1)
                        if _expect != _actual:
                            _mismatches.append((_i, _p, _expect, _actual))
                    logger.info(
                        "[diag-spdd-cs-apply] layer=%d rid=%s req_idx=%d "
                        "is_decode=%s dense=%s pdi=%d "
                        "vp_n=%d pb_n=%d new_bt_n=%d "
                        "vp_head=%s vp_tail=%s "
                        "pb_head=%s pb_tail=%s "
                        "bt_at_vp_head=%s bt_at_vp_tail=%s "
                        "new_bt_head=%s new_bt_tail=%s "
                        "routing_mismatches[:8]=%s",
                        _layer_idx, rid[:8], req_idx, _is_decode,
                        rs.dense_mode, rs._post_dense_iter,
                        _vp_n, len(_pb_cpu), len(_new_bt_cpu),
                        _vp_head, _vp_tail,
                        _pb_head, _pb_tail,
                        _bt_at_vp, _bt_at_vp_tail,
                        _new_bt_head, _new_bt_tail,
                        _mismatches[:8])
                except Exception as _diag_e:
                    logger.warning(
                        "[diag-spdd-cs-apply] log failed: %r", _diag_e)
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
                # Bug 11 family fix (2026-04-30): bytes are stored in
                # model_dtype (was fp16). NumPy lacks bf16 so we go
                # through torch.frombuffer to view the raw bytes as
                # model_dtype. Mirror of slow/fast GPU-direct path
                # changes — see record_page line ~4691.
                k_buf = torch.frombuffer(bytearray(raw[:half_bytes]),
                                          dtype=model_dtype).reshape(page_shape)
                v_buf = torch.frombuffer(bytearray(raw[half_bytes:]),
                                          dtype=model_dtype).reshape(page_shape)
                k_t = k_buf.clone().to(device=device)
                v_t = v_buf.clone().to(device=device)
                if self._tp_size > 1 and not per_rank_slice:
                    if (os.environ.get("ICMS_DIAG_TP_APPLY", "0") == "1"
                            and not getattr(self, "_diag_tp_apply_host_fired", False)):
                        # set flag UNCONDITIONALLY to ensure single-shot
                        self._diag_tp_apply_host_fired = True
                        try:
                            # k_t shape: (PAGE_TOKENS, num_kv_heads_full, head_dim)
                            _h0_sum = k_t[0, 0, :].float().sum().item()
                            _h_local_sum = k_t[0, head_start, :].float().sum().item()
                            _post = k_t[:, head_start:head_start + nkv_local, :]
                            _post_h0_sum = _post[0, 0, :].float().sum().item()
                            # Block-table comparison: log this rid's bt row
                            # head + the phys_block we're about to scatter to
                            try:
                                _bt_row_head = bt[req_idx, :8].cpu().tolist()
                                _bt_row_tail = (bt[req_idx, max(0, bt.shape[1]-8):].cpu().tolist()
                                                if bt.shape[1] > 8 else [])
                                _bt_row_len = int(bt.shape[1])
                                _phys_block = int(bt[req_idx, pid])
                                _seq_len_dbg = int(seq_len) if seq_len is not None else -1
                            except Exception:
                                _bt_row_head = _bt_row_tail = []
                                _bt_row_len = _phys_block = _seq_len_dbg = -1
                            logger.info(
                                "[diag-tp-apply-host] rank=%d nkv_total=%d "
                                "nkv_local=%d head_start=%d k_t_shape=%s "
                                "k_t[0,0,:].sum=%.4f "
                                "k_t[0,head_start=%d,:].sum=%.4f "
                                "post[0,0,:].sum=%.4f post_shape=%s "
                                "first_pid=%d phys_block=%d bt_row_len=%d "
                                "seq_len=%d bt[head]=%s bt[tail]=%s",
                                self._tp_rank, geom.num_kv_heads,
                                nkv_local, head_start, list(k_t.shape),
                                _h0_sum, head_start, _h_local_sum,
                                _post_h0_sum, list(_post.shape),
                                pid, _phys_block, _bt_row_len, _seq_len_dbg,
                                _bt_row_head, _bt_row_tail)
                        except Exception as _e:
                            logger.warning("diag-tp-apply-host failed: %s", _e)
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
        # Clamp to the actual prompt length. Without this, the last
        # cached group's partial-page padding (e.g., 33 stored groups
        # of 32 pages each = 1056 pages = 16896 tokens, but the prompt
        # is only 16466 tokens) inflates new_seq_len and Q for the
        # next decode token gets rotary-embedded for the WRONG
        # position (16896 instead of 16466). Cold path passes the
        # actual seq_len to FlashAttention; warm path must too.
        new_seq_len = min(
            filled_blocks_count * PAGE_TOKENS + continuation_tokens,
            seq_len,
        )
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

        # ICMS_DIAG_FULL: dump the slow-path inputs that get baked into
        # set_active. Layer 0 only to keep log compact. Captures (a) the
        # main_key/main_value pointers + shapes, (b) the trimmed bt
        # contents, (c) the seq_lens/max_seq_len, (d) the natural bt
        # contents at the same row for cross-check.
        if (os.environ.get("ICMS_DIAG_FULL") == "1"
                and layer_idx_for_cache == 0):
            try:
                _bt_first = new_bt[0][:8].cpu().tolist() if new_bt.numel() > 0 else []
                _bt_last = new_bt[0][-8:].cpu().tolist() if new_bt.numel() > 8 else []
                _bt_shape = tuple(new_bt.shape)
                _natural_first = bt[req_idx][:8].cpu().tolist()
                _natural_last_row = (bt[req_idx][cont_end-8:cont_end].cpu().tolist()
                                      if cont_end > 0 else [])
                _phys_h = phys_blocks_dev[:4].tolist() if phys_blocks_dev.numel() > 0 else []
                _phys_t = phys_blocks_dev[-4:].tolist() if phys_blocks_dev.numel() > 4 else []
                logger.info(
                    "[diag-full-apply] layer=0 rid=%s pdf=%d "
                    "main_key_ptr=%s main_key_shape=%s main_value_ptr=%s "
                    "n_valid=%d n_cumulative=%d new_bt_shape=%s "
                    "new_bt[0][:8]=%s new_bt[0][-8:]=%s "
                    "natural_bt[0][:8]=%s natural_bt[%d-8:%d]=%s "
                    "phys_blocks_dev[:4]=%s phys_blocks_dev[-4:]=%s "
                    "filled_blocks_count=%d new_seq_len=%d cont_end=%d "
                    "context_pages=%d total_blocks=%d seq_len=%d "
                    "prefill_done=%s dense_mode=%s",
                    rid, getattr(rs, "_post_dense_iter", -1),
                    hex(main_key.data_ptr()), tuple(main_key.shape),
                    hex(main_value.data_ptr()),
                    len(valid_pids), len(cumulative_pids), _bt_shape,
                    _bt_first, _bt_last,
                    _natural_first, cont_end, cont_end, _natural_last_row,
                    _phys_h, _phys_t,
                    filled_blocks_count, new_seq_len, cont_end,
                    context_pages, total_blocks, seq_len,
                    self._prefill_done,
                    getattr(rs, "dense_mode", False))
            except Exception as _e:
                logger.warning("[diag-full-apply] failed: %r", _e)

        # Set fetch state for FlashAttention to read.
        # ICMS_SKIP_BT_OVERRIDE=1 disables the bt/seq_lens override so
        # attention reads main_key via vLLM's natural block_table. Used
        # to isolate apply-scatter correctness from bt-override
        # correctness during debugging.
        if os.environ.get("ICMS_SKIP_BT_OVERRIDE") != "1":
            from vllm.v1.attention import icms_fetch_state
            _state_slow = icms_fetch_state.IcmsFetchState(
                key_cache=main_key,
                value_cache=main_value,
                block_table=new_bt,
                seq_lens=new_sl,
                max_seq_len=new_seq_len,
            )
            if _capture is not None:
                _capture.append((req_idx, _state_slow))
            else:
                icms_fetch_state.set_active(_state_slow)
        _lt("after_set_active")

        # Populate the per-stride apply cache for the next reuse layers.
        # All quantities below are stride-invariant: phys_blocks and new_bt
        # depend only on the selected page-id set; new_sl/max_seq_len
        # depend only on filled_blocks_count + seq_len. The sink offset
        # shift between layers is `delta * actual_k` pages — applied as a
        # single device-side add in the fast path. Only safe to cache from
        # the gpu_direct branch; the host-sink fallback exits earlier.
        if (rs is not None
                and gpu_direct
                and layer_idx_for_cache is not None
                and self._score_stride > 1):
            rs._apply_cached_layer_start = layer_idx_for_cache
            # phys_blocks_dev = current reply's pids → used by the fast
            # path to scatter THIS iter's sink data into main_key on
            # subsequent layers within the stride. Stays current-only.
            rs._apply_cached_phys_blocks_dev = phys_blocks_dev
            rs._apply_cached_page_idx_dev = page_idx_dev
            # Audit fix #9 (2026-05-11): cache the FILTERED count
            # (post-`valid_pids` filter), not the raw reply count. The
            # slow path at L0 scattered into sink at the filtered stride
            # (valid_pids drops dup + OOB pids). The fast path on
            # L1..L5 of the stride reads from sink at offset
            # `delta * cached_actual_k * kv_page_bytes`. If we cached
            # the RAW count here, the fast-path stride disagreed with
            # the slow-path stride whenever any pid was filtered out
            # → reuse layers read garbage KV bytes from the wrong sink
            # slot. Symptom: model fabricates plausible-format outputs
            # (e.g., 7-digit numbers for NIAH-multikey) at higher
            # budgets where filtering is more likely. See audit doc
            # detailed entry #9 + the diagnostic
            # `[diag-kscale-stride]` warning below.
            rs._apply_cached_actual_k = len(valid_pids)
            # Diagnostic only (was the audit-#9 probe): log when the
            # raw vs filtered count diverges so we can confirm any
            # future regression that re-introduces the mismatch.
            # Always logs (no env gate) since the situation is rare
            # but worth a one-line warning if it ever fires.
            _raw_n = len(reply.page_ids)
            if _raw_n != len(valid_pids):
                logger.warning(
                    "[icms-cache-actualk] rid=%s layer=%d filtered "
                    "raw=%d → valid=%d (diff=%d). Fast-path stride "
                    "now uses filtered count (audit #9 fix). If you "
                    "see this firing every Score, investigate why "
                    "the server is returning duplicates / OOB pids.",
                    rid, layer_idx_for_cache, _raw_n, len(valid_pids),
                    _raw_n - len(valid_pids))
            # Diag-only: cache the valid_pids list so the fast path's
            # multi-layer canary can map probe_pid → canary_idx without
            # rebuilding it. Has no effect on hot path correctness.
            rs._apply_cached_valid_pids = list(valid_pids)
            # new_bt is built from cumulative pids (M3+M4-A) and is
            # reused by the fast path for attention's block_table.
            rs._apply_cached_new_bt = new_bt
            # 2026-05-11 audit fix #21: pin the cumulative-set size used
            # to build new_bt so the fast path can detect mid-stride
            # growth and invalidate. The fast path entry check at
            # ~line 6165 compares this against the live
            # `rs.fetched_pages[stride_root]` count; mismatch → fall
            # through to slow path which rebuilds new_bt.
            _stride_root_cache = (
                (layer_idx_for_cache // self._score_stride)
                * self._score_stride)
            rs._apply_cached_cumulative_count = len(
                rs.fetched_pages.get(_stride_root_cache, set()))
            rs._apply_cached_new_sl = new_sl
            rs._apply_cached_max_seq_len = new_seq_len
            rs._apply_cached_filled_count = filled_blocks_count
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
        # ICMS_TRACE: emit `apply` per (rid, layer). KV bytes are GPU
        # tensors here (kv = self._gpu_kv_caches[layer_name]) so we
        # cannot cheaply hash the bytes — record the shape instead so
        # the diff harness can sanity-check geometry.
        if _ICMS_TRACE_ENABLED:
            try:
                _trace_pids_app = [int(p) for p in valid_pids[:16]]
                _trace_layer_idx = (layer_idx_for_cache
                                    if layer_idx_for_cache is not None
                                    else -1)
                _trace_kv_shape = (list(kv.shape)
                                   if hasattr(kv, "shape") else [])
                # 2026-05-10 TP=2 first-iter sparse-apply diag:
                # track each filter stage's count so we can pinpoint
                # which filter dropped the pids. At TP=2 L=0 first
                # iter, n_valid_pids=17 from a reply of ~100 pids.
                _trace_n_reply_pids = len(reply.page_ids) if hasattr(
                    reply, "page_ids") else -1
                _trace_n_reply_offs = len(reply.sink_offsets) if hasattr(
                    reply, "sink_offsets") else -1
                _trace_n_unique_pids = len(set(reply.page_ids)) if hasattr(
                    reply, "page_ids") else -1
                _trace_n_dict = len(pid_to_sink_off)
                _trace_n_selected = len(selected)
                _icms_trace(
                    "apply", rid, layer=_trace_layer_idx,
                    chain_fp=_icms_chain_fp(getattr(rs, "chain", None)),
                    extra={
                        "page_ids_applied": _trace_pids_app,
                        "n_valid_pids": len(valid_pids),
                        "n_reply_pids": _trace_n_reply_pids,
                        "n_reply_offs": _trace_n_reply_offs,
                        "n_unique_pids": _trace_n_unique_pids,
                        "n_pid_to_sink_off_dict": _trace_n_dict,
                        "n_selected": _trace_n_selected,
                        "context_pages": int(context_pages),
                        "bt_row_max": int(bt_row_max),
                        "tp_size": int(self._tp_size),
                        "tp_rank": int(self._tp_rank),
                        "layer_name": layer_name,
                        "kv_shape": _trace_kv_shape,
                    })
            except Exception:
                # Never let tracing break the apply path.
                pass
        return True  # signal that fetch state was set

    def _extract_layer_idx(self, layer_name: str) -> int | None:
        """Extract layer index from 'model.layers.N.self_attn.attn'.

        Cached by string key (`_layer_idx_cache`, populated lazily).
        Layer names are interned by vLLM's layer map and stable for
        the connector's lifetime, so the cache is correct + cheap.
        Saves a string split + linear scan on every call (~200+ per
        forward across all call sites).
        """
        cache = self._layer_idx_cache
        if layer_name in cache:
            return cache[layer_name]
        parts = layer_name.split(".")
        result: "int | None" = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    result = int(parts[i + 1])
                    break
                except ValueError:
                    pass
        cache[layer_name] = result
        return result

    def restore_attn_metadata(self, layer_name: str):
        """Clear the ICMS fetch state after attention ran with it."""
        from vllm.v1.attention import icms_fetch_state
        icms_fetch_state.clear()

    def _drain_pending_flush_queue(self):
        """Forward-thread drain of pipeline-thread WriteGroup ok-bits.

        Runs `_tp_broadcast_bool` on each enqueued (rid, group_idx,
        ok_local, partial, pages, chain_prefix) tuple, then applies the
        symmetric ledger bumps + `_record_stored_groups` push that the
        old inline path in `_flush_group` did. Called from
        `wait_for_pending_writes` BEFORE the memcpy gate so that all
        pipeline-thread NCCL is now on this (forward) thread → no
        same-rank concurrent NCCL with iter N+1's model.forward.

        Order is preserved per-rid because the pipeline thread is a
        single worker that flushes groups in monotonic gidx order; this
        drain processes them in FIFO order.

        2026-05-10 TP>1 asymmetric-queue fix:
        Pipeline progress is async per-rank, so at any given drain
        point one rank's `_pending_flush_q` may be shorter than the
        other's. Pre-fix, each rank called `_tp_broadcast_bool` once
        per local entry → asymmetric NCCL collective counts → CUDA
        hang. Post-fix, all-reduce-MAX the local count first; every
        rank then issues exactly `n` broadcasts, padding the local
        queue with sentinel entries (rid="", group_idx=-1) when
        local < n. The sentinel broadcasts use ok_local=True so the
        broadcast no-ops on rank-0; bumps are skipped via the
        sentinel check inside the loop. Pipeline catches up in
        future drains where the missed flush appears as a real entry.
        """
        # 1. Snapshot local queue (under lock).
        with self._pending_flush_lock:
            entries = self._pending_flush_q
            self._pending_flush_q = []
        # 2026-05-10 audit #11: sort by (rid, group_idx) so per-rank
        # ordering is canonical at the drain. Pre-fix, the queue
        # reflected pipeline-thread completion order which could
        # legitimately differ across ranks (rank-0 enqueues [A:0, B:0]
        # while rank-1 enqueues [B:0, A:0]) — broadcasting only
        # `ok_local` at lockstep i then meant rank-0's `ok_A` was
        # applied to rank-1's B (wrong-rid bump). Sort ensures both
        # ranks see the same entry at the same slot when they share
        # entries; combined with the sentinel-padding below this also
        # keeps lockstep when one rank has an entry the other doesn't
        # (the missing-side gets a sentinel at the would-be slot for
        # entries it doesn't have, and the present-side's bump lands
        # only on the rank that has the entry).
        #
        # Known residual limitation: if both ranks have non-empty
        # queues with strictly different entries (e.g., rank-0 only
        # has A, rank-1 only has B), sort+pad lines them up by sort
        # key, not by rid — broadcasts can still cross. The pipeline
        # producer is single-threaded per rank and driven by the same
        # ICMS server completion order, so production divergence here
        # is transient. A future tuple-broadcast (rid_hash, gidx, ok)
        # would close this fully; flagged in the audit doc.
        entries = sorted(entries, key=lambda e: (str(e[0]), int(e[1])))
        local_n = len(entries)
        # 3. Symmetrize across TP ranks. At TP=1 this is a no-op and
        #    falls through to the legacy fast-path (early-return on 0).
        n = _tp_allreduce_max_int(local_n, self._tp_size)
        if n == 0:
            return
        # 4. Pad local queue with sentinels if shorter than the global
        #    max. Sentinel `rid=""` + `group_idx=-1` is recognized
        #    inside the loop and skips the per-rid bump (no rs to find).
        if local_n < n:
            sentinel = ("", -1, True, False, 0, ())
            entries = list(entries) + [sentinel] * (n - local_n)
        # Broadcasts + bumps run lock-free — bumps mutate per-rid state
        # only, and rs is single-threaded (forward thread or
        # pipeline-thread-during-pipeline-task) so no contention here.
        for rid, group_idx, ok_local, partial, pages, chain_prefix in entries:
            ok = _tp_broadcast_bool(
                ok_local, self._tp_rank, self._tp_size)
            # Sentinel entry from the symmetrize-padding above: keep
            # the broadcast (collective participation is the whole
            # reason it's there) but skip the bump path.
            if rid == "" and group_idx == -1:
                continue
            if ok:
                # Mirror the inline path's success-trace + bumps. Gating
                # on `ok` (broadcast result, not `ok_local`) preserves
                # Design B symmetry — all ranks bump or all skip.
                if _ICMS_TRACE_ENABLED:
                    try:
                        _wg_pids = [int(group_idx) * _GROUP_BLOCKS + i
                                    for i in range(_GROUP_BLOCKS)][:16]
                        _icms_trace(
                            "writegroup_commit", rid, layer=-1,
                            chain_fp=_icms_chain_fp(list(chain_prefix)),
                            extra={
                                "group_idx": int(group_idx),
                                "page_ids_in_group": _wg_pids,
                                "success": True,
                                "partial": bool(partial),
                                "pages": int(pages),
                                "deferred_broadcast": True,
                            })
                    except Exception:
                        pass
                self.stats.total_groups_written += 1
                rs = self._requests.get(rid)
                if rs is not None:
                    if group_idx >= rs.num_groups_written:
                        rs.num_groups_written = group_idx + 1
                    if group_idx >= rs.flushed_local and not partial:
                        _chain_len = (len(rs.chain) if rs.chain
                                      else group_idx + 1)
                        rs.flushed_local = min(
                            group_idx + 1, _chain_len)
                        # 2026-05-10 audit #5: bump flush_seq under the
                        # Condition lock + notify_all. Any waiter in
                        # `_score_one_request` that captured a smaller
                        # `_flush_seq_at_attempt` will see the new value
                        # and return from wait_for immediately —
                        # including a waiter that hadn't reached
                        # `wait_for` yet when this notify fired
                        # (the lost-wakeup case bare Event.set+clear
                        # had).
                        with rs.flush_cond:
                            rs.flush_seq += 1
                            rs.flush_cond.notify_all()
                        if rs.chain:
                            self._record_stored_groups(
                                rs.chain, rs.flushed_local)
                            _append_stored_chain_queue(
                                list(rs.chain), rs.flushed_local)

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
        # 2026-05-09 N2 deferral: drain any pending WriteGroup ok-bits
        # from the previous iter's pipeline-thread flushes BEFORE the
        # memcpy gate. This runs `_tp_broadcast_bool` on the forward
        # thread (the only legal place for collective NCCL on the TP
        # comm) and applies the symmetric ledger bumps. Closes the
        # TP>1 pipeline-thread NCCL collision the audit identified.
        self._drain_pending_flush_queue()
        if not (self._gpu_kv_caches and self._attn_metadata is not None
                and self._requests and not self._skip_extract):
            # Nothing to extract — still flip the prefill_done flag.
            if not self._prefill_done:
                self._reset_apply_caches_for_prefill_done()
                self._prefill_done = True
                logger.info("Prefill done. Switching to dense decode.")
            return

        # M4-A all-dense gate: once every active request has flipped to
        # dense_mode, the bitmap is saturated for each rs — no downstream
        # Score will reference future decode-token KV. Skip the extract +
        # flush closure (and the BLOCK_WRITES drain) entirely. Removes
        # the ~33 ms decode-iter spikes at every 16-token page boundary
        # (extract_and_record GPU→CPU + record_page × num_layers +
        # WriteGroup RPC). Decode-token write-back is parked pending a
        # multi-turn-reuse harness (see project_icms_decode_m3_status).
        if (self._prefill_done
                and all(getattr(rs, "dense_mode", False)
                        for rs in self._requests.values())):
            return

        # Snapshot references that the background task needs. The KV
        # cache tensors and request state are expected to remain valid
        # until on_request_finished drains the pipeline.
        snap_caches = dict(self._gpu_kv_caches)
        snap_meta   = self._attn_metadata
        snap_rids   = list(self._requests.keys())

        # 2026-05-09 multi-rid batched-mode KV-corruption fix:
        # Without gating, the pipeline thread's GPU→CPU memcpy races
        # against vLLM freeing the prefill's physical pages and the
        # NEXT request's apply WRITING ICMS-fetched KV into those same
        # pages. Result: deferred-extract reads corrupted bytes (the
        # next rid's apply data), ships them to ICMS as "this rid's
        # archived KV," and all future scoring on this rid's chain
        # returns garbage → repetitive/junk model output.
        #
        # Fix: signal a per-call memcpy_done event AFTER all
        # extract_and_record calls complete (i.e., after every .cpu()
        # has returned), and block this forward thread on that event
        # before letting vLLM proceed to free pages. The WriteGroup
        # RPCs + summary compute keep running async on the pipeline
        # thread — they don't touch GPU pages anymore.
        #
        # Default ON. Set ICMS_GATE_MEMCPY=0 to disable (debug only;
        # reintroduces the corruption in batched mode).
        _gate_memcpy = os.environ.get("ICMS_GATE_MEMCPY", "1") == "1"
        memcpy_done = threading.Event() if _gate_memcpy else None

        def _task():
            self._do_deferred_extract_and_flush(
                snap_caches, snap_meta, snap_rids,
                memcpy_done_event=memcpy_done)

        self._write_pipeline.submit(_task, tag="wait_for_save",
                                     rids=snap_rids)

        # Block the forward thread until the memcpy stage of the task
        # completes. This is the page-safety gate: vLLM cannot free the
        # prefill's KV blocks until wait_for_save returns, so once we
        # pass this point the GPU pages have been fully drained to
        # host memory and are safe to release. Bounded by a generous
        # timeout so a stuck pipeline never permanently hangs the
        # forward thread (the timeout falls back to the legacy "free
        # eagerly" behavior, accepting the corruption risk over a hang).
        if memcpy_done is not None:
            try:
                _memcpy_timeout_s = float(os.environ.get(
                    "ICMS_GATE_MEMCPY_TIMEOUT_S", "120.0"))
            except ValueError:
                _memcpy_timeout_s = 120.0
            if not memcpy_done.wait(timeout=_memcpy_timeout_s):
                logger.warning(
                    "ICMS memcpy gate timed out (%.1fs) — falling "
                    "through; GPU pages may be freed before extract "
                    "completes (cross-rid KV corruption risk).",
                    _memcpy_timeout_s)

        # Optional measurement mode: block until this prefill's writes
        # complete before returning. Default behavior leaves writes async
        # in the background, which is faster but means iter N's writes
        # can race iter N+1's fetches on the same RDMA link, polluting
        # slack measurements (per-layer fetch arrival times shift later
        # under contention). With ICMS_BLOCK_WRITES=1, all writes are on
        # the current prefill's TTFT critical path — the next prefill
        # begins with an idle link.
        #
        # Fix H (2026-04-29): at TP>1, default to blocking. The pipeline
        # thread does an AllGather on the TP NCCL group; if the next
        # iter's main-thread forward starts before the AllGather + .cpu()
        # sync completes, both threads hammer the same NCCL group
        # concurrently and deadlock at TP=2 (worker stuck in
        # extract_and_record's .cpu(), engine core's shm_broadcast times
        # out at 60s). Set ICMS_BLOCK_WRITES=0 to override.
        # 2026-05-08 race-audit follow-up (Step 1 of "C"): batched mode
        # at TP=1 (max_num_seqs > 1, ICMS_ALLOW_BATCH=1) hits the same
        # write/read ordering race as TP>1 — iter N's write_pipeline can
        # lag iter N+1's main-thread Score, and the connector's retry-
        # on-ENOENT silently falls through to dense (no flush_local
        # progress means Score reads from a chain prefix the server
        # hasn't seen yet). Empirical evidence: llama3 vt 32K batched
        # b=0.05 with BLOCK_WRITES=0 → 0.260 acc; BLOCK_WRITES=1 → 0.600.
        # Extend the default-on gate to batched mode regardless of TP
        # so paper accuracy runs are correct out of the box. The
        # ICMS_BLOCK_WRITES env knob still overrides if the user wants
        # raw async perf at the known correctness cost.
        # 2026-05-09 (Item C audit): consolidated to _is_multi_rid_mode
        # for consistency with the other batched-mode gates. The
        # logical-OR semantics are identical to the earlier inline
        # expression; helper just centralizes the policy.
        # 2026-05-09: BLOCK_WRITES default policy after introducing the
        # memcpy-done gate (above):
        #   * TP=1 batched: gate alone is sufficient. Section 1
        #     (extract+memcpy) is awaited by the gate; Section 2
        #     (_flush_group → WriteGroup RPC) doesn't touch GPU pages
        #     and stays async. Default OFF.
        #   * TP>1: Section 2's `_tp_broadcast_bool` is an NCCL op on
        #     the TP communicator. After the gate releases the main
        #     thread, iter N+1's forward starts — its NCCL ops on the
        #     same rank's same communicator collide with the pipeline
        #     thread's Section 2 NCCL → CUDA hang → execute_model RPC
        #     timeout. Until _tp_broadcast_bool is moved to the forward
        #     thread, keep BLOCK_WRITES default ON for TP>1 to drain
        #     Section 2 before returning. (TP=1 single-rid is also OFF
        #     since there's no batching contention either.)
        # BLOCK_WRITES=1 is honored as an opt-in fallback even at TP=1
        # (useful for diagnosing async-RPC-related vs memcpy-related
        # symptoms; mirrors what BLOCK_WRITES=1 did pre-fix).
        _batched = self._is_multi_rid_mode()
        _block_default = "1" if self._tp_size > 1 else "0"
        if os.environ.get("ICMS_BLOCK_WRITES", _block_default) == "1":
            # Drain timeout configurable (2026-05-01). Default raised from
            # 60s → 180s: Llama-3.1-8B (32 layers × 8 KV heads × 128k ctx)
            # generates 16+ GiB of WriteGroup traffic per prefill, which
            # at the link's effective ~2-4 GiB/s takes well over 60s. Pre-
            # fix the drain timed out, returned without finishing the
            # pipeline AllGather, and the next iter's main-thread forward
            # collided with the still-running pipeline thread on the same
            # NCCL group → shm_broadcast hang → sample_tokens RPC timeout
            # → EngineDeadError. Mirrors the existing 90 s timeout in
            # on_request_finished but with more headroom for larger
            # models. Set ICMS_BLOCK_WRITES_DRAIN_TIMEOUT_S to override.
            try:
                _drain_timeout_s = float(os.environ.get(
                    "ICMS_BLOCK_WRITES_DRAIN_TIMEOUT_S", "180.0"))
            except ValueError:
                _drain_timeout_s = 180.0
            try:
                drained = self._write_pipeline.drain(timeout=_drain_timeout_s)
                if not drained:
                    logger.warning(
                        "ICMS_BLOCK_WRITES drain timed out (%.1fs)",
                        _drain_timeout_s)
            except Exception:
                logger.exception("ICMS_BLOCK_WRITES drain failed")

        if not self._prefill_done:
            self._reset_apply_caches_for_prefill_done()
            self._prefill_done = True
            logger.info("Prefill done. Switching to dense decode.")

    def _diag_full_dense_flip_snapshot(self, rs, flipping_layer: int) -> None:
        """ICMS_DIAG_FULL: dump rs metadata at the moment dense_mode flips.

        Captures per-rs cache pointers, fetched_pages totals, _active
        state, and a sample of the most recent set_active block_table —
        the things most likely to carry stale state into the natural-bt
        decode path that runs after the flip.
        """
        from vllm.v1.attention import icms_fetch_state
        try:
            active = icms_fetch_state.get_active()
            active_bt_shape = (tuple(active.block_table.shape)
                                if active is not None else None)
            active_seq = (active.seq_lens.cpu().tolist()
                           if active is not None else None)
            active_max = active.max_seq_len if active is not None else None
            active_kp = (hex(active.key_cache.data_ptr())
                         if active is not None else None)
        except Exception as _e:
            active_bt_shape = active_seq = active_max = active_kp = f"err:{_e!r}"
        fp_summary = {k: len(v) for k, v in rs.fetched_pages.items()}
        cached_bt_shape = (tuple(rs._apply_cached_new_bt.shape)
                           if rs._apply_cached_new_bt is not None else None)
        logger.info(
            "[diag-dense-flip] rid=%s flip_at_layer=%d stored_grp=%d "
            "num_grp_written=%d cache_layer_start=%d cache_actual_k=%d "
            "cache_new_bt_shape=%s cache_max_seq_len=%d "
            "fetched_pages_sizes=%s _active_bt_shape=%s _active_seq=%s "
            "_active_max=%s _active_key_ptr=%s",
            rs.request_id, flipping_layer,
            rs.stored_groups, rs.num_groups_written,
            rs._apply_cached_layer_start, rs._apply_cached_actual_k,
            cached_bt_shape, rs._apply_cached_max_seq_len,
            fp_summary, active_bt_shape, active_seq,
            active_max, active_kp,
        )

    def _diag_full_iter_metadata(
        self, layer_name: str, attn_metadata, where: str
    ) -> None:
        """ICMS_DIAG_FULL: dump natural attn_metadata at well-known points.

        For each active rs that's post-dense-flip (rs._post_dense_iter in
        0..2), log block_table[0][:8/-8], seq_lens, slot_mapping[:4/-4],
        max_seq_len, num_actual_tokens. This lets us watch metadata
        evolve across the first 3 forwards after the flip — the regime
        most likely to expose stale-pointer / stale-bt issues.
        """
        try:
            am = attn_metadata
            if am is None or not isinstance(am, dict):
                # Some backends pass dict, some pass single object; handle both.
                if am is None:
                    return
                am_local = am
            else:
                am_local = am.get(layer_name, None)
                if am_local is None:
                    return
            for rs in self._requests.values():
                pdi = getattr(rs, "_post_dense_iter", -1)
                if pdi < 0 or pdi > 2:
                    continue
                bt_local = getattr(am_local, "block_table", None)
                sl_local = getattr(am_local, "seq_lens", None)
                sm_local = getattr(am_local, "slot_mapping", None)
                msk_local = getattr(am_local, "max_seq_len", None)
                nat_local = getattr(am_local, "num_actual_tokens", None)
                bt_first = (bt_local[0][:8].cpu().tolist()
                            if bt_local is not None else None)
                bt_last = (bt_local[0][-8:].cpu().tolist()
                            if bt_local is not None else None)
                bt_shape = tuple(bt_local.shape) if bt_local is not None else None
                bt_ptr = (hex(bt_local.data_ptr())
                          if bt_local is not None else None)
                sl_list = (sl_local.cpu().tolist()
                           if sl_local is not None else None)
                sm_first = (sm_local[:4].cpu().tolist()
                            if sm_local is not None else None)
                sm_last = (sm_local[-4:].cpu().tolist()
                            if sm_local is not None else None)
                logger.info(
                    "[diag-postflip] where=%s rid=%s pdi=%d "
                    "layer=%s bt_shape=%s bt_ptr=%s "
                    "bt[0][:8]=%s bt[0][-8:]=%s seq_lens=%s "
                    "slot_map[:4]=%s slot_map[-4:]=%s "
                    "max_seq_len=%s num_actual_tokens=%s",
                    where, rs.request_id, pdi, layer_name,
                    bt_shape, bt_ptr, bt_first, bt_last, sl_list,
                    sm_first, sm_last, msk_local, nat_local)
        except Exception as _e:
            logger.warning("[diag-postflip] snapshot failed: %r", _e)
    def _reset_apply_caches_for_prefill_done(self):
        """Bug 11 (2026-04-29) audit fix #1: invalidate per-rs apply
        caches at the prefill→decode transition.

        The fast-path cache (_apply_cached_*) was populated by the LAST
        scored layer of prefill (typically layer N-1 of the last stride
        group, e.g., layer 42 with stride=6). Without invalidation, the
        first decode iter's reuse-layer wait_for_layer calls would hit
        the fast path with prefill's stale phys_blocks/page_idx/new_bt
        — scattering this iter's smaller decode Score reply via prefill's
        actual_k stride into wrong sink slots and using prefill's trimmed
        block_table for set_active. Resetting layer_start to -1 forces
        the slow path to re-run on the first decode-iter scored layer,
        rebuilding the cache with current data."""
        from vllm.v1.attention import icms_fetch_state
        for rs in self._requests.values():
            rs._apply_cached_layer_start = -1
            rs._apply_cached_phys_blocks_dev = None
            rs._apply_cached_page_idx_dev = None
            rs._apply_cached_actual_k = 0
            rs._apply_cached_new_bt = None
            rs._apply_cached_new_sl = None
            rs._apply_cached_max_seq_len = 0
            rs._apply_cached_filled_count = 0
            rs._apply_cached_cont_idx_dev = None
            rs._apply_cached_cont_idx_range = (0, 0)
            rs._apply_cached_seq_len = None
            rs._apply_cached_attn_md = None
        # The trimmed bt/key_cache pointers stored in icms_fetch_state
        # also reference prefill state — clear so the first decode-iter
        # layer starts from a clean slate.
        icms_fetch_state.clear()

    def _do_deferred_extract_and_flush(
        self,
        caches: dict,
        meta,
        rids: list[str],
        memcpy_done_event: "threading.Event | None" = None,
    ):
        """Runs on the write-pipeline worker thread.

        Does the heavy lifting (GPU→CPU copies inside extract_and_record,
        summary compute, bytearray fills, and WriteGroup RPCs) without
        holding up the vLLM forward-pass return. Touches shared client
        state under self._score_lock to serialize with main-thread
        Score/FetchAll RPCs.

        memcpy_done_event: optional event signaled AFTER all per-layer
        extract_and_record calls complete (i.e., after every .cpu() has
        returned). The forward thread waits on this event before letting
        vLLM free GPU pages — see wait_for_pending_writes. Always set in
        a finally block so an exception in the extract loop cannot
        deadlock vLLM."""
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
        # CRITICAL: any return / exception from this section MUST signal
        # memcpy_done_event so wait_for_pending_writes does not deadlock
        # the forward thread. Use try/finally.
        try:
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
        finally:
            # All .cpu() calls have returned by this point (PyTorch
            # tensor.cpu() is a synchronous GPU→CPU copy that blocks the
            # caller until the bytes are on host). The GPU pages backing
            # this prefill are now safe to release.
            if memcpy_done_event is not None:
                memcpy_done_event.set()

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
        # Record `flushed_local` (count of groups this request actually
        # flushed), NOT `num_groups_written` (which may be inflated by
        # the inherited-prefix elision path in extract_and_record). The
        # ledger must reflect what's truly in the trie or downstream
        # Score RPCs hit ENOENT.
        for rid in rids:
            rs = self._requests.get(rid)
            if rs is None:
                continue
            if rs.chain and rs.flushed_local > 0:
                self._record_stored_groups(rs.chain, rs.flushed_local)
                # Bug #5 fix (race-audit 2026-05-08): single atomic
                # operation under `_stored_chain_cond`. Append + bump
                # are now indivisible from a scheduler-thread drain's
                # snapshot+clear, closing the iter/clear race.
                _append_stored_chain_queue(
                    list(rs.chain), rs.flushed_local)

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
        # Multi-rid path (ICMS_ALLOW_BATCH=1): the global short-circuit
        # is too coarse — one non-skip rid in a batch should not force
        # extract for the skip-eligible rids. Skip the early-return
        # here; the plan-builder below filters per-rid via
        # self._skip_extract_rids instead.
        # 2026-05-09 (Item C audit): use _is_multi_rid_mode() so the
        # global short-circuit is also bypassed when max_num_seqs > 1
        # without ICMS_ALLOW_BATCH=1. Otherwise one non-skip rid in a
        # batch would force-skip every other rid's extraction.
        if self._skip_extract and not self._is_multi_rid_mode():
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

        # KV cache layout: standard FA backend uses
        #   [2 (K+V), num_blocks, block_size, num_kv_heads, head_dim]
        # but TRITON_ATTN backend (used for gemma-3 sliding-window-attention
        # models) uses
        #   [num_blocks, 2 (K+V), block_size, num_kv_heads, head_dim]
        # — the K/V dimension is on axis 1, not axis 0. Detect both. The
        # K/V slice in either layout is [num_blocks, block_size,
        # num_kv_heads, head_dim] which is what the rest of this routine
        # expects.
        if kv_layer.ndim != 5:
            if _diag_n13 and layer_idx == 0:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=bad_kv_shape "
                            "ndim=%d", self._tp_rank, layer_name, kv_layer.ndim)
            return
        if kv_layer.shape[0] == 2:
            # Standard FA layout.
            k_cache = kv_layer[0]
            v_cache = kv_layer[1]
        elif kv_layer.shape[1] == 2:
            # TRITON_ATTN layout (gemma-3, etc.). kv_layer[:, 0] selects K
            # across all blocks; result has the same downstream shape as
            # the standard layout.
            k_cache = kv_layer[:, 0]
            v_cache = kv_layer[:, 1]
        else:
            if _diag_n13 and layer_idx == 0:
                logger.info(
                    "[diag-extract] rank=%d layer=%s SKIP=unknown_kv_layout "
                    "shape=%s", self._tp_rank, layer_name,
                    tuple(kv_layer.shape))
            return

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

        # block_table may be a tensor on GPU; move to CPU for indexing.
        if isinstance(block_table, torch.Tensor):
            bt_cpu = block_table.cpu()
        else:
            bt_cpu = block_table
        if isinstance(seq_lens, torch.Tensor):
            sl_cpu = seq_lens.cpu()
        else:
            sl_cpu = seq_lens

        # Build the per-rid plan. Single-rid path (legacy): only the
        # first active request, row 0. Multi-rid path (ICMS_ALLOW_BATCH
        # =1): walk batch positions in self._last_step_requests order
        # so both TP ranks symmetrically reach every collective.
        plan: list = []  # list of (rid, rs, req_idx_in_bt)
        # 2026-05-09 (Item C audit): use _is_multi_rid_mode() instead
        # of _allow_batch() so launchers with max_num_seqs > 1 but no
        # ICMS_ALLOW_BATCH=1 take the multi-rid plan-build path. Pre-fix
        # the legacy "first rid only" branch fired and silently dropped
        # the second rid's KV — every rid past the first contributed
        # garbage downstream Score/apply with zeroed sink bytes.
        if self._is_multi_rid_mode() and self._last_step_requests:
            # 2026-05-12 multi-rid row-mapping fix: prefer the
            # authoritative `self._rid_to_bt_row` populated from vLLM's
            # `input_batch.req_ids` (the row order FA actually uses for
            # bt / seq_lens / qsl). The legacy `enumerate(_last_step_requests)`
            # order matches meta.requests (new+cached append) which does
            # NOT match FA when input_batch reorders — pre-fix this read
            # the wrong rid's K/V at extract for every rid in the batch
            # whose position differed. See docs/multi_rid_root_cause_2026-05-12.md.
            # `getattr` keeps the access safe for callers (tests, mocks)
            # that don't initialize the attribute.
            _rid_to_bt_row = getattr(self, "_rid_to_bt_row", None) or {}
            # 2026-05-15 invariant flag: surface the structural hazard
            # of falling back to enumerate order. Cheap; runs once per
            # extract; gated by _is_multi_rid_mode so the single-rid
            # path doesn't pay the cost.
            try:
                _active = [prs.request_id
                           for prs in self._last_step_requests
                           if prs.request_id not in self._skip_extract_rids
                           and self._requests.get(prs.request_id) is not None
                           and self._requests[prs.request_id].chain]
                if _active:
                    self._check_rid_to_bt_row_present(
                        _active, source="extract.multi_rid")
            except Exception:
                # Never let the invariant check itself break extract.
                pass
            for prs in self._last_step_requests:
                rid_i = prs.request_id
                if rid_i in self._skip_extract_rids:
                    continue
                rs_i = self._requests.get(rid_i)
                if rs_i is None or not rs_i.chain:
                    continue
                bt_row_idx = _rid_to_bt_row.get(rid_i)
                if bt_row_idx is None:
                    # Pre-fix path: no input_batch ordering plumbed
                    # through. Fall back to legacy enumerate order.
                    bt_row_idx = next(
                        (i for i, p in enumerate(self._last_step_requests)
                         if p.request_id == rid_i), 0)
                plan.append((rid_i, rs_i, bt_row_idx))
        else:
            # Legacy single-rid path: first active request, row 0.
            # 2026-05-10 audit fix #19: pick from `_last_step_requests`
            # (broadcast scheduler metadata, identical order on every
            # rank) instead of `self._requests.items()` (Python dict
            # insertion order, which can drift if a retry path on one
            # rank inserts an extra rs that the other rank doesn't
            # have).  Pre-fix, dict-iteration drift caused different
            # ranks to pick different rids → asymmetric K bytes
            # shipped to MemBackend → silent corruption (same shape as
            # the bug we just root-caused).  Falls back to the legacy
            # dict-iter only when `_last_step_requests` is empty (rare
            # edge case during connector startup before any
            # scheduler step has been observed).
            if self._last_step_requests:
                for prs in self._last_step_requests:
                    rid_i = prs.request_id
                    rs_i = self._requests.get(rid_i)
                    if rs_i is None or not rs_i.chain:
                        continue
                    plan.append((rid_i, rs_i, 0))
                    break
            else:
                for r, s in self._requests.items():
                    if s is None or not s.chain:
                        continue
                    plan.append((r, s, 0))
                    break
        if not plan:
            return

        # ICMS_DIAG_FULLTRACE: cross-rid block-table snapshot at L=0.
        # Dumps every active rid's bt row + seq_len so set-intersection
        # across rid timelines reveals physical-block aliasing.
        if _ICMS_FULLTRACE_ENABLED and layer_idx == 0:
            try:
                per_rid_rows: list = []
                for _rid, _rs, _bt_idx in plan:
                    if hasattr(sl_cpu, '__len__') and len(sl_cpu) > _bt_idx:
                        _sl = int(sl_cpu[_bt_idx]) if isinstance(
                            sl_cpu[_bt_idx],
                            (int, float, torch.Tensor)) else 0
                    else:
                        _sl = 0
                    _nb = (_sl + block_size - 1) // block_size
                    if bt_cpu.ndim >= 2:
                        _bts = int(bt_cpu.shape[1])
                        _nb = min(_nb, _bts)
                        _bt_row_full = bt_cpu[_bt_idx, :_nb].tolist()
                    elif bt_cpu.ndim == 1:
                        _bts = int(bt_cpu.shape[0])
                        _nb = min(_nb, _bts)
                        _bt_row_full = bt_cpu[:_nb].tolist()
                    else:
                        _bt_row_full = []
                    per_rid_rows.append({
                        "rid": str(_rid),
                        "bt_row_idx": int(_bt_idx),
                        "seq_len": int(_sl),
                        "num_blocks": int(_nb),
                        "chain_len": int(
                            len(_rs.chain) if _rs.chain else 0),
                        "stored_groups": int(getattr(
                            _rs, "stored_groups", -1)),
                        "num_groups_written": int(getattr(
                            _rs, "num_groups_written", -1)),
                        "bt_row_first16": [int(x)
                                             for x in _bt_row_full[:16]],
                        "bt_row_last4": [int(x)
                                           for x in _bt_row_full[-4:]],
                        "bt_row_full": [int(x) for x in _bt_row_full[:2048]],
                    })
                # Cross-rid intersection counts: for each pair (i, j),
                # count how many phys blocks they share.
                _bts_per_rid = [set(int(x) for x in r["bt_row_full"])
                                 for r in per_rid_rows]
                _bts_per_rid_full = []
                for r in per_rid_rows:
                    _bts_per_rid_full.append(set(int(x) for x in
                                                  r["bt_row_full"]))
                pair_intersections: list = []
                for i in range(len(per_rid_rows)):
                    for j in range(i + 1, len(per_rid_rows)):
                        _inter = _bts_per_rid_full[i] & _bts_per_rid_full[j]
                        pair_intersections.append({
                            "i_rid": per_rid_rows[i]["rid"],
                            "j_rid": per_rid_rows[j]["rid"],
                            "intersection_count": int(len(_inter)),
                            "intersection_head": sorted(list(_inter))[:8],
                        })
                _icms_fulltrace(
                    "iter_bt_snapshot", rid="", layer=int(layer_idx),
                    n_rids=len(per_rid_rows),
                    per_rid=per_rid_rows,
                    pair_intersections=pair_intersections,
                    bt_shape=list(bt_cpu.shape) if hasattr(bt_cpu, 'shape')
                        else [],
                    skip_extract_rids=sorted(list(self._skip_extract_rids)),
                )
            except Exception:
                pass

        # Process each rid's new blocks. At TP>1, AllGather fires once
        # per rid (per layer); both ranks iterate plan in identical
        # order (derived from broadcast metadata) so collective ordering
        # is symmetric. Future optimization: stack per-rid K/V then
        # AllGather once per layer (Phase B).
        for rid, rs, bt_row_idx in plan:
            self._extract_one_rid(layer_idx, layer_name, k_cache, v_cache,
                                  bt_cpu, sl_cpu, block_size, rid, rs,
                                  bt_row_idx, t0)
        return

    def _extract_one_rid(self, layer_idx, layer_name, k_cache, v_cache,
                         bt_cpu, sl_cpu, block_size, rid, rs, bt_row_idx,
                         t0):
        # Number of blocks for this rid.
        if hasattr(sl_cpu, '__len__') and len(sl_cpu) > bt_row_idx:
            seq_len = int(sl_cpu[bt_row_idx]) if isinstance(
                sl_cpu[bt_row_idx], (int, float, torch.Tensor)) else 0
        else:
            seq_len = 0
        if seq_len <= 0:
            return
        num_blocks = (seq_len + block_size - 1) // block_size

        # BUG-N1 fix: under prefix elision (TP>1 + ICMS), vLLM only
        # allocates the trailing slice of the block table, but seq_lens
        # still reports the full prompt length. Trailing slots in
        # bt_cpu[row, k:] hold stale IDs from another request's
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

        # Extract this rid's block IDs from the block table row.
        if bt_cpu.ndim >= 2:
            req_block_ids = bt_cpu[bt_row_idx, :num_blocks].tolist()
        else:
            req_block_ids = bt_cpu[:num_blocks].tolist()

        # ICMS_TRACE: which physical GPU pages does this rid's bt_row map
        # to? If two different rids share physical pages here, we have
        # cross-rid block_table contamination at extract time.
        if _ICMS_TRACE_ENABLED:
            try:
                _icms_trace(
                    "extract_bt", rid, layer=layer_idx, chain_fp="",
                    extra={
                        "bt_row_idx": int(bt_row_idx),
                        "num_blocks": int(num_blocks),
                        "seq_len": int(seq_len),
                        "bt_row_first16": [int(x) for x in req_block_ids[:16]],
                        "bt_row_last4": [int(x) for x in req_block_ids[-4:]],
                    })
            except Exception:
                pass

        # Only process blocks we haven't recorded yet for this layer.
        recorded_key = (rid, layer_idx)
        if not hasattr(rs, '_recorded_blocks'):
            rs._recorded_blocks = {}
        already_recorded = rs._recorded_blocks.get(recorded_key, 0)

        # Dedup-aware skip: blocks in groups already stored under the
        # request's chain prefix don't need re-extraction. The server
        # would dedup the write anyway, but the GPU→CPU copy + summary
        # compute are wasted client-side work on the critical path.
        # 2026-05-10 TP>1 stored_groups asymmetry fix (extract-side):
        # Read the symmetrized value populated on the forward thread
        # by _update_stored_groups_from_meta (which all-reduces under
        # the main-thread NCCL group). This loop runs on the pipeline
        # thread, so we MUST NOT issue NCCL here — it would collide
        # with vLLM's per-layer all_reduce on the main thread (verified
        # to hang the engine at first prefill). The forward-thread
        # symmetrize ensures both ranks see the same value.
        effective_stored_groups = max(
            int(rs.stored_groups),
            int(getattr(rs, "_effective_stored_groups", 0)))
        stored_blocks = effective_stored_groups * _GROUP_BLOCKS
        effective_start = max(already_recorded, stored_blocks)
        if effective_start > already_recorded:
            # Advance num_groups_written so later _flush_group calls and
            # the stored-prefix push correctly reflect "stored + new".
            if effective_stored_groups > rs.num_groups_written:
                rs.num_groups_written = effective_stored_groups

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
                # T2 fix (race-audit, 2026-05-08): route through the
                # ICMS sep-NCCL group when ICMS_USE_SEPARATE_NCCL_GROUP=1
                # so the pipeline thread doesn't share the comm with
                # vLLM's main-thread per-layer all_reduce. Pre-fix this
                # site hardcoded tp_group.device_group, defeating the
                # sep-NCCL knob for the K/V AllGather and explaining
                # llama3 TP=2 hangs even with sep-NCCL on.
                dev_group = _get_icms_nccl_group() or tp_group.device_group
                gk = [torch.empty_like(k_batch_gpu)
                      for _ in range(self._tp_size)]
                gv = [torch.empty_like(v_batch_gpu)
                      for _ in range(self._tp_size)]
                _trace = os.environ.get("ICMS_NCCL_TRACE", "0") == "1"
                _diag_order = (os.environ.get("ICMS_DIAG_HEAD_ORDER", "0") == "1"
                               and not getattr(self, "_diag_head_order_fired", False))
                if _diag_order:
                    pre_hash = int(k_batch_gpu.contiguous().view(torch.uint8).sum().item())
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
                if _diag_order:
                    try:
                        try:
                            grp_rank_dist = dist.get_rank(group=dev_group)
                        except Exception:
                            grp_rank_dist = -1
                        grp_rank_attr = getattr(tp_group, "rank_in_group", -1)
                        per_rank_hashes = [
                            int(t.contiguous().view(torch.uint8).sum().item())
                            for t in gk
                        ]
                        head0_first = [
                            int(gk[i].contiguous().view(-1).view(torch.uint8)[0].item())
                            for i in range(self._tp_size)
                        ]
                        logger.info(
                            "[icms-diag-head-order] self_tp_rank=%d "
                            "tp_group.rank_in_group=%s dist.get_rank(dev_group)=%s "
                            "pre_allgather_hash=%d gk_hashes=%s "
                            "gk_head0_first_byte=%s layer=%d k_shape=%s",
                            self._tp_rank, grp_rank_attr, grp_rank_dist,
                            pre_hash, per_rank_hashes, head0_first,
                            layer_idx, list(k_batch_gpu.shape))
                    except Exception as _diag_e:
                        logger.warning("[icms-diag-head-order] diag failed (non-fatal): %s", _diag_e)
                    self._diag_head_order_fired = True
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
                # 2026-05-10 audit fix #4: re-raise instead of falling
                # back to per-rank K/V.  Pre-fix this swallowed the
                # exception and proceeded with the rank-LOCAL nkv-heads
                # slice — write_group then shipped a half-head blob to
                # the server, silently corrupting MemBackend's view of
                # the chain.  The other rank may have completed
                # AllGather and shipped FULL-head data → ranks now
                # disagree on what's stored under the chain.  Surface
                # the failure loudly via the connector's existing
                # exception path; the deferred-extract pipeline already
                # logs + propagates exceptions to wait_for_pending_writes.
                logger.error(
                    "extract_and_record: TP AllGather(K/V) failed "
                    "rank=%d layer=%d: %s — re-raising. Silent per-rank "
                    "fallback would write half-head garbage to the "
                    "server and silently desynchronize ranks.",
                    self._tp_rank, layer_idx, e, exc_info=True)
                raise

        k_batch = k_batch_gpu.cpu()
        v_batch = v_batch_gpu.cpu()

        # 2026-05-10 TP=2 KV-content corruption diag: hash the K bytes
        # of the FIRST extracted page at L=0 only. Lets us compare the
        # extracted-K hash against the apply-side K hash for the same
        # logical page → if they differ, the corruption is between
        # write-to-ICMS and read-from-ICMS at TP=2.
        if _ICMS_TRACE_ENABLED and layer_idx == 0 and len(valid_ids) > 0:
            try:
                import hashlib as _hl
                _kb = k_batch[0].numpy().tobytes()[:128]
                _vb = v_batch[0].numpy().tobytes()[:128]
                _icms_trace(
                    "extract_kv_hash", rid, layer=int(layer_idx),
                    chain_fp="",
                    extra={
                        "physical_page": int(req_block_ids[
                            effective_start + 0]) if effective_start < len(
                            req_block_ids) else -1,
                        "intra_idx": int(effective_start),
                        "k_bytes_hash": _hl.sha1(_kb).hexdigest()[:16],
                        "v_bytes_hash": _hl.sha1(_vb).hexdigest()[:16],
                        "k_first_8_bytes_hex": _kb[:8].hex(),
                        "k_shape": list(k_batch[0].shape),
                        "tp_rank": int(self._tp_rank),
                        "tp_size": int(self._tp_size),
                        "n_pages_extracted": int(len(valid_ids)),
                    })
                # 2026-05-10 follow-up audit trace: full-page V hash,
                # plus a sampled set across the extracted batch (pages
                # 0, mid, last) so we don't only validate page 0. K
                # was previously shown bit-identical at TP=2 batched;
                # V wasn't checked, so add an explicit trace event.
                # Also full-bytes (not just first 128) so we don't
                # miss a tail-region corruption.
                _kf = k_batch[0].numpy().tobytes()
                _vf = v_batch[0].numpy().tobytes()
                samples: list = []
                _idxs = [0]
                if len(valid_ids) > 1:
                    _idxs.append(len(valid_ids) // 2)
                if len(valid_ids) > 2:
                    _idxs.append(len(valid_ids) - 1)
                for _i in _idxs:
                    _kbi = k_batch[_i].numpy().tobytes()
                    _vbi = v_batch[_i].numpy().tobytes()
                    samples.append({
                        "intra_idx": int(_i),
                        "physical_page": int(req_block_ids[
                            effective_start + _i]) if (
                            effective_start + _i < len(req_block_ids)
                        ) else -1,
                        "k_full_hash": _hl.sha1(_kbi).hexdigest()[:16],
                        "v_full_hash": _hl.sha1(_vbi).hexdigest()[:16],
                    })
                _icms_trace(
                    "extract_v_hash", rid, layer=int(layer_idx),
                    chain_fp="",
                    extra={
                        "tp_rank": int(self._tp_rank),
                        "tp_size": int(self._tp_size),
                        "n_pages_extracted": int(len(valid_ids)),
                        "page0_v_full_hash":
                            _hl.sha1(_vf).hexdigest()[:16],
                        "page0_k_full_hash":
                            _hl.sha1(_kf).hexdigest()[:16],
                        "samples": samples,
                    })
            except Exception:
                pass

        # 2026-05-10 TP>1 byte-layout diag: log key_block shape at L=0
        # so we can verify it's full K (post-AllGather) vs per-rank K.
        if (layer_idx == 0 and len(valid_ids) > 0
                and not getattr(self, "_diag_keyblock_shape_fired", False)):
            try:
                logger.warning(
                    "[diag-keyblock-shape] tp_rank=%d tp_size=%d "
                    "k_batch.shape=%s k_batch[0].shape=%s "
                    "expected_full_per_geom=%dx%d k_batch[0].nbytes=%d",
                    self._tp_rank, self._tp_size,
                    list(k_batch.shape), list(k_batch[0].shape),
                    self._geom.num_kv_heads if self._geom else -1,
                    self._geom.head_dim if self._geom else -1,
                    int(k_batch[0].numel() * k_batch[0].element_size()))
                self._diag_keyblock_shape_fired = True
            except Exception:
                pass

        for i, intra_idx in enumerate(range(effective_start, effective_start + len(valid_ids))):
            self.record_page(rid, intra_idx, layer_idx, k_batch[i], v_batch[i])

        rs._recorded_blocks[recorded_key] = len(req_block_ids)
        self.stats.record_extract(
            (time.perf_counter() - t0) * 1e6, len(valid_ids))
        # ICMS_DIAG_FULLTRACE: per-rid per-layer extract summary. Layer 0
        # also emits first/last/sample K+V SHAs over the extracted batch
        # so we can cross-reference Phase 1 against what comes back at
        # apply time.
        if _ICMS_FULLTRACE_ENABLED:
            try:
                import hashlib as _hl_ft
                extra: dict = {
                    "bt_row_idx": int(bt_row_idx),
                    "seq_len": int(seq_len),
                    "num_blocks": int(num_blocks),
                    "effective_start": int(effective_start),
                    "stored_blocks": int(stored_blocks),
                    "n_valid": int(len(valid_ids)),
                    "valid_ids_head": [int(x) for x in valid_ids[:8]],
                    "valid_ids_tail": [int(x) for x in valid_ids[-8:]],
                    "req_block_ids_head": [int(x)
                                             for x in req_block_ids[:8]],
                    "req_block_ids_tail": [int(x)
                                             for x in req_block_ids[-8:]],
                    "num_groups_written": int(rs.num_groups_written),
                    "stored_groups": int(rs.stored_groups),
                    "chain_len": int(len(rs.chain) if rs.chain else 0),
                }
                if layer_idx == 0 and len(valid_ids) > 0:
                    samples: list = []
                    _idxs = [0]
                    if len(valid_ids) > 1:
                        _idxs.append(len(valid_ids) // 2)
                    if len(valid_ids) > 2:
                        _idxs.append(len(valid_ids) - 1)
                    for _i in _idxs:
                        _kbi = k_batch[_i].contiguous().view(
                            torch.uint8).numpy().tobytes()
                        _vbi = v_batch[_i].contiguous().view(
                            torch.uint8).numpy().tobytes()
                        samples.append({
                            "intra_idx": int(effective_start + _i),
                            "phys_block": int(req_block_ids[
                                effective_start + _i])
                                if effective_start + _i < len(
                                    req_block_ids) else -1,
                            "k_sha": _hl_ft.sha1(_kbi).hexdigest()[:16],
                            "v_sha": _hl_ft.sha1(_vbi).hexdigest()[:16],
                            "k_head8": _kbi[:8].hex(),
                        })
                    extra["samples"] = samples
                _icms_fulltrace("extract", rid=rid, layer=int(layer_idx),
                                **extra)
            except Exception:
                pass

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
        # Bug 11 family fix (2026-04-30): preserve full model precision by
        # serializing raw model_dtype bytes (no fp16 down-cast). The
        # legacy `.to(torch.float16)` was a bf16↔fp16 round-trip for bf16
        # models like qwen3, losing ~3 mantissa bits per value. At 32k
        # chain × 16 tokens × 4 KV heads × 128 dim × 48 layers = ~745M
        # values per request, the accumulated drift derails attention
        # enough that multi-key NIAH retrieval fails. Both bf16 and fp16
        # are 2 bytes per element; view(uint8) reinterprets the buffer
        # as numpy-compatible bytes without any conversion. Apply side
        # must view back as model_dtype (not fp16) — see lines 3805+,
        # 3448+, 3957+ for the corresponding read-side updates.
        k_bytes = key_block.contiguous().view(torch.uint8).numpy().tobytes()
        v_bytes = value_block.contiguous().view(torch.uint8).numpy().tobytes()
        kv_bytes = k_bytes + v_bytes

        # ICMS_DIAG_CANARY=1: per-write fingerprint for a few logical
        # pages of layer 0 (we recompute target pages from intra_idx →
        # group_idx*32+page_in_group). Pair with [diag-canary-read]
        # for the same pid in _apply_selective_attention.
        # Multi-layer canary (2026-04-30): also fire for scored layers
        # 6,12,18,24,30,36,42 so we can verify the FAPS slow-path
        # per-layer offset arithmetic. If layer-0 hashes match write↔read
        # but layer-6+ don't, the server's per-layer pack ordering is
        # the bug. Restricted to one probe pid per layer to keep volume
        # low.
        if (os.environ.get("ICMS_DIAG_CANARY") == "1"
                and layer_idx in (0, 6, 12, 18, 24, 30, 36, 42)):
            absolute_pid = group_idx * _GROUP_BLOCKS + page_in_group
            # Layer 0 keeps full probe set for byte-path correctness.
            # Higher layers only sample pid=0 / 17 / 100 to keep log size
            # manageable across 7 extra layers.
            _probe_set = (
                (0, 1, 2, 3, 4, 8, 16, 17, 24, 31, 32, 33, 100, 500)
                if layer_idx == 0 else (0, 17, 100))
            if absolute_pid in _probe_set:
                import hashlib
                chain_head = rs.chain[:1] if rs.chain else []
                kh = hashlib.sha1(k_bytes).hexdigest()[:16]
                vh = hashlib.sha1(v_bytes).hexdigest()[:16]
                khead = k_bytes[:32].hex()
                logger.info("[diag-canary-write] rid=%s chain_head=%s "
                             "abs_pid=%d gidx=%d page_in_group=%d layer=%d "
                             "k_len=%d v_len=%d k_sha=%s v_sha=%s "
                             "k_head=%s",
                             rs.request_id, chain_head, absolute_pid,
                             group_idx, page_in_group, layer_idx,
                             len(k_bytes), len(v_bytes), kh, vh, khead)
                # Per-rank-subset canary: hash the slice of key_block /
                # value_block that EACH rank's apply read-side will see
                # under PRS=1. key_block here has shape
                # [block_size=16, num_kv_heads=4, head_dim=128]. Each
                # rank's slice is heads [r*nkv_local, (r+1)*nkv_local).
                # Direct-compare sub_sha against [diag-canary-read]
                # k_sha at the same (layer, pid).
                if self._tp_size > 1:
                    nkv_local = (self._geom.num_kv_heads // self._tp_size)
                    for _r in range(self._tp_size):
                        _s = _r * nkv_local
                        _sub_k = (key_block[:, _s:_s + nkv_local, :]
                                  .contiguous().view(torch.uint8)
                                  .numpy().tobytes())
                        _sub_v = (value_block[:, _s:_s + nkv_local, :]
                                  .contiguous().view(torch.uint8)
                                  .numpy().tobytes())
                        _sub_kh = hashlib.sha1(_sub_k).hexdigest()[:16]
                        _sub_vh = hashlib.sha1(_sub_v).hexdigest()[:16]
                        logger.info(
                            "[diag-canary-write-rank] rid=%s abs_pid=%d "
                            "layer=%d rank=%d k_sub_len=%d v_sub_len=%d "
                            "k_sub_sha=%s v_sub_sha=%s k_sub_head=%s",
                            rs.request_id, absolute_pid, layer_idx, _r,
                            len(_sub_k), len(_sub_v),
                            _sub_kh, _sub_vh, _sub_k[:32].hex())

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

        # Retain GPU-side per-(KV-head) summaries (shape [num_kv_heads,
        # head_dim] fp16, indexed by absolute page id) when EITHER:
        #   ICMS_ORIGINAL_QUEST=1   — local Quest scorer path
        #   ICMS_DIAG_SCORE_DUMP=<> — regular sparse path dump for offline
        #                            scoring-algo analysis (added 2026-05-14)
        # Default path is byte-identical when both envs are unset.
        if (os.environ.get("ICMS_ORIGINAL_QUEST", "0") == "1"
                or os.environ.get("ICMS_DIAG_SCORE_DUMP", "")):
            keys_gpu = key_block.detach()
            if keys_gpu.ndim == 3:
                # [block_size, num_kv_heads, head_dim] → reduce over tokens
                kmin_gpu = keys_gpu.amin(dim=0).to(torch.float16)
                kmax_gpu = keys_gpu.amax(dim=0).to(torch.float16)
                abs_pid = group_idx * _GROUP_BLOCKS + page_in_group
                rs.quest_gpu_summaries.setdefault(layer_idx, []).append(
                    (abs_pid, kmin_gpu, kmax_gpu))

        buf.filled.add((layer_idx, page_in_group))
        # NOTE: no inline flush on completion. Flushing is deferred to
        # wait_for_pending_writes (called at the end of each forward
        # pass by vLLM) so that write_group RPCs and the extraction
        # work don't stall per-layer save_kv_layer calls during the
        # prefill forward pass.

    def _flush_write_batch_now(self, request_id: str):
        """2026-05-11 Option-1: drain the per-rid write batch buffer
        with ONE WriteGroup RPC carrying K = len(buffer) tail groups.

        Records the post-RPC outcome for each buffered group with the
        SAME `ok_local` (server commits the K groups atomically; one
        RPC error → all K marked failed; one RPC ok → all K marked
        committed). Forwards into `_pending_flush_q` so the existing
        `_drain_pending_flush_queue` machinery bumps `flushed_local`
        and `num_groups_written` once per group, just as it would have
        for K individual single-group RPCs."""
        # Defense-in-depth rank gate (2026-05-11): the callers in
        # `_flush_group` (line ~9077, 9130) and `on_request_finished`
        # (line ~9301) ALREADY gate on rank-0, so this should never
        # fire. Kept inline so the `test_tp_rank_gating` AST inspector
        # can see the gate without traversing the call graph, and to
        # protect future callers from accidentally bypassing it.
        if self._tp_size > 1 and self._tp_rank != 0:
            return
        # 2026-05-11 write-batching-audit Finding 2: pop under the
        # batch-buf lock so a concurrent pipeline-thread append doesn't
        # race the forward-thread drain.
        with self._write_batch_buf_lock:
            buf = self._write_batch_buf.pop(request_id, [])
        if not buf:
            return
        buf.sort(key=lambda b: b['group_idx'])
        # Sanity: every buffered group must have a strictly-consecutive
        # group_idx (the wire protocol's "extend" semantic appends K
        # tails under a single parent — gaps would corrupt the chain).
        # If not consecutive, fall back to N single-group RPCs.
        first_gidx = buf[0]['group_idx']
        for i, b in enumerate(buf):
            if b['group_idx'] != first_gidx + i:
                logger.warning(
                    "[icms-batch] non-consecutive group_idxs in batch "
                    "for rid=%s: %s. Falling back to single-RPC per "
                    "group.", request_id,
                    [b2['group_idx'] for b2 in buf])
                self._fallback_single_rpcs(request_id, buf)
                return
        K = len(buf)
        # 2026-05-11 wire-size guard: the wire frame header is
        # `<IB` (uint32 length + byte type — protocol.py:63), so a
        # single RPC's payload must fit in uint32 (< 4 GiB). For
        # gemma-3-27b at 16 KV heads × 128 head_dim × 62 layers,
        # one group's KV ≈ 500 MB; K=16 → 8 GB → frame-length
        # overflow → struct.pack("<I", length) fails with
        # `'I' format requires 0 <= number <= 4294967295`. If the
        # accumulated payload would exceed the safe cap, split the
        # buffer in half and recurse — each half sends its own
        # WriteGroup RPC. Choice of 3 GiB cap (vs 4 GiB max) leaves
        # headroom for the chain-hash + header bytes and avoids
        # bumping right against the limit.
        _MAX_FRAME_PAYLOAD = int(os.environ.get(
            "ICMS_WRITE_BATCH_MAX_PAYLOAD_BYTES",
            str(3 * 1024 * 1024 * 1024)))
        # Halve K repeatedly until the payload fits in one frame. Each
        # iteration drops half of the buffer back into the pending
        # queue (preserving group_idx order) and re-checks the
        # remaining first chunk. For gemma-3-27b at K=16/8GB, this
        # converges at K=4 (~2GB). Bounded by log2(K) iterations.
        while True:
            _total_payload = sum(
                len(b['summary_blob']) + len(b['kv_blob']) for b in buf)
            if K <= 1 or _total_payload <= _MAX_FRAME_PAYLOAD:
                break
            mid = K // 2
            logger.info(
                "[icms-batch] payload %.2f GiB > cap %.2f GiB for "
                "rid=%s K=%d first_gidx=%d — splitting (sending "
                "K=%d first, deferring K=%d)",
                _total_payload / (1024 ** 3),
                _MAX_FRAME_PAYLOAD / (1024 ** 3),
                request_id, K, first_gidx, mid, K - mid)
            first_half = buf[:mid]
            second_half = buf[mid:]
            with self._write_batch_buf_lock:
                # Prepend second_half so subsequent _flush_group call
                # ordering is preserved (lower group_idxs at front).
                # Any new entries appended concurrently for this rid
                # have higher group_idxs than `second_half`, so
                # prepending keeps the sorted invariant.
                existing = self._write_batch_buf.setdefault(
                    request_id, [])
                self._write_batch_buf[request_id] = second_half + existing
            buf = first_half
            K = len(buf)
        # Build parent + tail chains from the LAST buffered group's
        # `chain_prefix` (= rs.chain[:last_gidx+1] at flush time).
        last_chain_prefix = buf[-1]['chain_prefix']
        rank_chain = self._rank_chain(last_chain_prefix)
        if first_gidx == 0:
            parent_chain: list[int] = []
            new_tail_groups = list(rank_chain[:K])
        else:
            parent_chain = list(rank_chain[:first_gidx])
            new_tail_groups = list(rank_chain[first_gidx:first_gidx + K])
        # Concatenated blobs — server handles K-tail layout per
        # handlers.cc:1099-1115 (summary_blob_base then kv_blob_base
        # at offsets keyed by K, num_scored, num_layers).
        summary_blob = b''.join(b['summary_blob'] for b in buf)
        kv_blob = b''.join(b['kv_blob'] for b in buf)
        # pages_in_group on the wire applies to the LAST tail per
        # client.py:418-425 ("earlier tails must be 32-page-full ...
        # split the bulk write across calls if the LAST tail isn't").
        # Since we only buffer partial=False, all are 32-page-full —
        # send the same pages count for last (a no-op equivalence).
        pages_in_group = buf[-1]['pages']
        t_send = time.perf_counter()
        try:
            with self._rpc_lock:
                self._client.write_group(
                    parent_chain, new_tail_groups,
                    summary_blob, kv_blob,
                    pages_in_group=pages_in_group,
                )
        except Exception as e:
            logger.warning(
                "[icms-batch] Batched WriteGroup failed for rid=%s "
                "K=%d first_gidx=%d: %s — falling through to per-"
                "group single-RPC retry for accurate per-tail ok_local.",
                request_id, K, first_gidx, e)
            # 2026-05-11 write-batching-audit Finding 3: on batched-
            # RPC failure, retry per-tail via `_fallback_single_rpcs`
            # so each group gets its own ok_local. Server's WriteGroup
            # is idempotent on already-committed tails (handlers.cc
            # ~L1146-1173: `trie::extend → insert(parent+tails)`
            # fall-through finds existing match and no-ops), so
            # retries don't double-write KV bytes. Pre-fix, the
            # whole batch was marked failed → subsequent same-prefix
            # requests didn't elide ANY of the K groups even if the
            # server had committed k<K before erroring → latency hit
            # proportional to the wasted re-prefill.
            self._fallback_single_rpcs(request_id, buf)
            return
        send_elapsed_us = (time.perf_counter() - t_send) * 1e6
        # Amortize the batched RPC time across the K groups so the
        # histogram remains comparable to the N=1 baseline.
        per_group_lat_us = send_elapsed_us / max(1, K)
        for b in buf:
            self.stats.record_flush(per_group_lat_us)
            with self._pending_flush_lock:
                self._pending_flush_q.append((
                    request_id, int(b['group_idx']), True,
                    bool(b['partial']), int(b['pages']),
                    tuple(b['chain_prefix']),
                ))

    def _fallback_single_rpcs(self, request_id: str,
                               buf: "list[dict]"):
        """Fallback when batch buffer can't be sent atomically
        (non-consecutive group_idxs). Sends N single-group RPCs and
        records the same per-group outcome as the pre-batching path.
        Should be rare — buffer is built by consecutive `_flush_group`
        calls during haystack prefill so non-consecutive only happens
        on weird interleaving (which would be a separate bug)."""
        # Defense-in-depth rank gate (mirrors _flush_write_batch_now).
        if self._tp_size > 1 and self._tp_rank != 0:
            return
        for b in buf:
            rank_chain = self._rank_chain(b['chain_prefix'])
            gidx = b['group_idx']
            if gidx == 0:
                parent_chain: list[int] = []
                new_tail_groups = list(rank_chain)
            else:
                parent_chain = list(rank_chain[:-1])
                new_tail_groups = [rank_chain[-1]]
            ok_local = True
            try:
                with self._rpc_lock:
                    self._client.write_group(
                        parent_chain, new_tail_groups,
                        b['summary_blob'], b['kv_blob'],
                        pages_in_group=b['pages'],
                    )
            except Exception as e:
                ok_local = False
                logger.warning(
                    "[icms-batch] fallback WriteGroup failed for "
                    "rid=%s gidx=%d: %s", request_id, gidx, e)
            self.stats.record_flush(
                (time.perf_counter() - b['t0']) * 1e6)
            with self._pending_flush_lock:
                self._pending_flush_q.append((
                    request_id, int(gidx), bool(ok_local),
                    bool(b['partial']), int(b['pages']),
                    tuple(b['chain_prefix']),
                ))

    def _flush_group(self, request_id: str, group_idx: int, partial: bool = False):
        """Issue WriteGroup for a completed (or partial) group buffer."""
        # Diag (2026-05-02): ICMS_SKIP_WRITES=1 short-circuits the
        # WriteGroup RPC entirely. Used to disambiguate whether a
        # server-side LOC_LEN_ERR occurs on a write-path SEND
        # (WriteGroup ACK) or a read-path SEND (Score reply / FetchAll).
        # Also lets us measure decode-only TPOT without Phase-2 KV
        # persistence — the prefix won't be reusable across sweeps but
        # the in-flight request still sees correct attention because
        # apply scattered the K pages into vLLM's local kv_cache before
        # WriteGroup fires.
        if os.environ.get("ICMS_SKIP_WRITES") == "1":
            rs = self._requests.get(request_id)
            if rs is not None:
                rs.active_group_buffers.pop(group_idx, None)
            return
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
        # ICMS_DIAG_CANARY=1: hash bytes at layer 0 / specific page
        # offsets right before shipping. Pair with [diag-canary-write]
        # (record_page side) and [server-canary-recv]. If these match
        # write canaries but server differs → wire transfer dropped
        # bytes. If they differ from write canaries → record_page
        # stored at wrong offset OR something clobbered buf.kv_blob.
        if os.environ.get("ICMS_DIAG_CANARY") == "1":
            import hashlib as _hl
            kv_blob_bytes = bytes(buf.kv_blob)
            kpb = self._geom.kv_page_bytes if self._geom else 32768
            kgb = self._geom.kv_group_bytes if self._geom else 32 * kpb
            for probe_pid in (0, 1, 2, 3, 4, 8, 16, 17, 24, 31, 32, 33, 100, 500):
                pg_idx = probe_pid // _GROUP_BLOCKS
                if pg_idx != group_idx:
                    continue
                page_in_group = probe_pid % _GROUP_BLOCKS
                off = 0 * kgb + page_in_group * kpb  # layer=0
                if off + 32 > len(kv_blob_bytes):
                    continue
                head32 = kv_blob_bytes[off:off + 32].hex()
                k_sha = _hl.sha1(
                    kv_blob_bytes[off:off + kpb // 2]).hexdigest()[:16]
                logger.info("[diag-flush-bytes] rid=%s gidx=%d "
                             "probe_pid=%d layer=0 off=%d "
                             "k_sha=%s head32=%s",
                             request_id, group_idx, probe_pid, off,
                             k_sha, head32)
        # Design B (2026-05-08): track local RPC success in a flag,
        # broadcast OUTSIDE the try, then gate ledger bumps on the
        # broadcasted (cross-rank-symmetric) value. Replaces the prior
        # in-try bumps that were skipped on rank-0 when write_group
        # raised, while rank>0's no-op SKIP path took the bumps —
        # producing divergent flushed_local / num_groups_written and
        # the TP=2 multi-rid `sample_tokens` hang. See
        # project_tp2_writegroup_asymmetry_2026-05-07 + Design B in
        # docs/icms_open_bugs_audit_2026-05-08.md.
        ok_local = True
        try:
            # Option W: only rank 0 sends to the server. All ranks still
            # update local bookkeeping (_record_stored_groups etc.) below.
            _diag_tp_write = os.environ.get("ICMS_DIAG_TP_WRITE", "0") == "1"
            if self._tp_size > 1 and self._tp_rank != 0:
                if _diag_tp_write:
                    logger.info(
                        "[diag-tp-write] rank=%d SKIP rid=%s gidx=%d "
                        "chain_prefix_len=%d partial=%s pages=%d "
                        "summary_bytes=%d kv_bytes=%d",
                        self._tp_rank, request_id, group_idx,
                        len(chain_prefix), partial, pages,
                        len(buf.summary_blob), len(buf.kv_blob))
            else:
                # 2026-05-11 Option-1: if this is a partial flush AND we
                # have buffered full groups for this rid, drain them
                # FIRST so the chain is built parent→tails in order.
                # The partial group below appends on top of the
                # committed batched groups.
                # Finding 2: dict read under `_write_batch_buf_lock`
                # for snapshot consistency. `_flush_write_batch_now`
                # itself re-acquires the lock for the pop.
                with self._write_batch_buf_lock:
                    _has_buffered = bool(self._write_batch_buf.get(
                        request_id))
                if partial and _has_buffered:
                    self._flush_write_batch_now(request_id)
                # v6 wire. WG#1 of a rid (group_idx==0) is a fresh
                # registration: parent=[], tail=full_rank_chain. Server
                # routes to insert() which handles cross-rid prefix
                # split + ref bumping. WG#k>1 is streaming extension:
                # parent=rank_chain[:-1], tail=[last]. Server routes to
                # extend() with swap-on-extend semantics. Both keep
                # per-rid contribution to each node at +1, so one Evict
                # at request-finish frees the whole chain.
                rank_chain = self._rank_chain(chain_prefix)
                if group_idx == 0:
                    parent_chain = []
                    new_tail_groups = list(rank_chain)
                else:
                    parent_chain = rank_chain[:-1]
                    new_tail_groups = [rank_chain[-1]]
                if _diag_tp_write:
                    _ck_first = rank_chain[0] if rank_chain else None
                    _ck_last = rank_chain[-1] if rank_chain else None
                    logger.info(
                        "[diag-tp-write] rank=%d WRITE rid=%s gidx=%d "
                        "tp_size=%d chain_len=%d ck_first=%s ck_last=%s "
                        "parent_len=%d tail_len=%d pages=%d "
                        "summary_bytes=%d kv_bytes=%d partial=%s",
                        self._tp_rank, request_id, group_idx,
                        self._tp_size, len(rank_chain),
                        _ck_first, _ck_last,
                        len(parent_chain), len(new_tail_groups),
                        pages, len(buf.summary_blob),
                        len(buf.kv_blob), partial)
                # 2026-05-11 Option-1: batched WriteGroup. Buffer full
                # consecutive groups (partial=False) and send K-at-a-time.
                # Default N=1 → no batching. N>1 → defer RPC + post-RPC
                # bookkeeping to `_flush_write_batch_now`, which handles
                # all N groups atomically (one RPC + one bookkeeping pass
                # per group). Partial flushes always go through the
                # single-RPC path; if there's a pending batch when a
                # partial flush hits, the partial path drains the batch
                # FIRST (see end of this function).
                batch_n = int(os.environ.get("ICMS_WRITE_BATCH_N", "1"))
                if batch_n > 1 and not partial:
                    # Finding 2: setdefault+append + length-check must
                    # be ONE atomic critical section, else two threads
                    # could each observe `len < batch_n` and both
                    # decide NOT to drain even when their combined
                    # count crosses the threshold. Compute the
                    # post-append count under the lock; perform the
                    # drain decision OUTSIDE the lock so
                    # `_flush_write_batch_now`'s own lock acquire
                    # (for the pop) doesn't dead-nest.
                    with self._write_batch_buf_lock:
                        self._write_batch_buf.setdefault(
                            request_id, []).append({
                            'group_idx': int(group_idx),
                            'summary_blob': bytes(buf.summary_blob),
                            'kv_blob': bytes(buf.kv_blob),
                            'pages': int(pages),
                            'chain_prefix': list(chain_prefix),
                            'partial': bool(partial),
                            't0': float(t0),
                        })
                        _post_append_n = len(
                            self._write_batch_buf[request_id])
                    if _post_append_n >= batch_n:
                        # Drain now — sends one batched RPC + records
                        # outcome for every buffered group.
                        self._flush_write_batch_now(request_id)
                    # In either case, the bookkeeping for this group is
                    # handled by `_flush_write_batch_now` (now or later).
                    # Skip the per-group post-RPC tail below.
                    return
                with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                    self._client.write_group(
                        parent_chain, new_tail_groups,
                        bytes(buf.summary_blob), bytes(buf.kv_blob),
                        pages_in_group=pages,
                    )
        except Exception as e:
            ok_local = False
            logger.warning("WriteGroup failed for req=%s group=%d: %s",
                           request_id, group_idx, e)
            # ICMS_TRACE: failure branch trace.
            if _ICMS_TRACE_ENABLED:
                try:
                    _wg_pids_fail = [int(group_idx) * _GROUP_BLOCKS + i
                                     for i in range(_GROUP_BLOCKS)][:16]
                    _icms_trace(
                        "writegroup_commit", request_id, layer=-1,
                        chain_fp=_icms_chain_fp(chain_prefix),
                        extra={
                            "group_idx": int(group_idx),
                            "page_ids_in_group": _wg_pids_fail,
                            "success": False,
                            "err": str(e),
                            "partial": bool(partial),
                        })
                except Exception:
                    pass

        # Always record flush latency — this is observation, not a
        # bookkeeping bump, so it should fire whether or not the RPC
        # succeeded so timing histograms stay representative.
        self.stats.record_flush((time.perf_counter() - t0) * 1e6)
        # ICMS_DIAG_FULLTRACE: emit per-flush summary so we can see exactly
        # which (rid, group_idx) made it to the server (locally) before
        # the deferred TP broadcast.
        if _ICMS_FULLTRACE_ENABLED:
            try:
                _icms_fulltrace(
                    "flush_group", rid=request_id,
                    group_idx=int(group_idx),
                    ok_local=bool(ok_local),
                    partial=bool(partial),
                    pages=int(pages),
                    chain_prefix_len=int(len(chain_prefix)),
                    chain_prefix_head=[int(x) for x in chain_prefix[:4]],
                    chain_prefix_tail=[int(x) for x in chain_prefix[-4:]],
                    chain_fp=_icms_chain_fp(list(chain_prefix)),
                    num_groups_written_pre=int(rs.num_groups_written),
                    flushed_local_pre=int(getattr(rs, "flushed_local", -1)),
                )
            except Exception:
                pass

        # 2026-05-09 N2 deferral: instead of running `_tp_broadcast_bool`
        # inline (which is a NCCL op on the TP comm and races iter N+1's
        # forward-thread NCCL → CUDA hang at TP>1), enqueue an entry
        # for the forward thread to broadcast and apply the bumps on
        # the next `wait_for_pending_writes` call. The forward thread
        # is the only legal place for collective NCCL ops on the TP
        # comm (vLLM owns that thread's stream).
        #
        # Snapshot rs.chain at flush time — rs.chain may mutate before
        # the forward thread drains the queue. chain_prefix is already
        # a slice of rs.chain (line ~7882), so we copy it explicitly.
        with self._pending_flush_lock:
            self._pending_flush_q.append((
                request_id, int(group_idx), bool(ok_local),
                bool(partial), int(pages),
                tuple(chain_prefix),
            ))

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
        # B2 (2026-05-05): per-rid drain. Pre-batching, this used to
        # `drain()` the entire pipeline, forcing a finishing rid to wait
        # on still-active rids' writes. With submit() now tagging tasks
        # by rid, drain_rid() awaits only this rid's tasks; tasks that
        # don't touch this rid are left running for their owners.
        # At N=1 / single-rid path this collapses to the legacy
        # behavior (only one rid in flight).
        #
        # 2026-05-07 BUG FIX: 90s timeout was sized for 30K ctx
        # (~70 GiB writes ⇒ ~7s drain). At 128K ctx the writes are
        # ~300 GiB ⇒ 30+ s drain plus overhead → 90s fires routinely.
        # When the timeout fires, `flushed_local` is then recorded as
        # the ENQUEUED count (not acked), so the scheduler-side
        # `_stored_chains` claims groups the server's trie hasn't
        # committed → next turn's Score returns "no resolvable groups"
        # (server has 0 groups for chain) or DIVERGENCE (off-by-N).
        # The original 90s was kept because longer drains on `drain()`
        # (whole pipeline) collided with the next request's NCCL; with
        # per-rid drain that collision class is gone, so we can safely
        # extend. Default 600s (10× more headroom). Set
        # ICMS_DRAIN_TIMEOUT_S=N to override.
        try:
            _drain_timeout_s = float(
                os.environ.get("ICMS_DRAIN_TIMEOUT_S", "600.0"))
        except ValueError:
            _drain_timeout_s = 600.0
        ok = self._write_pipeline.drain_rid(
            request_id, timeout=_drain_timeout_s)
        drain_us = (time.perf_counter() - t0) * 1e6
        if not ok:
            logger.warning(
                "on_request_finished: write-pipeline drain_rid TIMED OUT "
                "(>%.0fs, rid=%s, %d tasks pending globally); "
                "request-finish proceeding — some writes may be lost. "
                "Set ICMS_DRAIN_TIMEOUT_S to a higher value to avoid "
                "this on long-context runs.",
                _drain_timeout_s, request_id,
                self._write_pipeline.pending())
        elif drain_us > 1000:
            logger.info(
                "on_request_finished: write-pipeline drain_rid(%s) took %.1f ms",
                request_id, drain_us / 1000.0)

        rs = self._requests.get(request_id)
        if rs is None:
            return
        # Flush any partial group buffers first (must happen BEFORE we
        # pop rs, since _flush_group re-reads self._requests[rid]).
        # BUG-N5: no _score_lock here — the icms client serializes
        # QP access internally; the lock would only block dict
        # mutations in wait_for_layer / on_layer_score for no gain.
        #
        # 2026-05-10 audit #12: distinguish actually-partial (buffer
        # holds < _GROUP_BLOCKS pages) from full-but-not-yet-flushed
        # buffers. Pre-fix, every remaining buffer was flagged
        # partial=True, which made `_drain_pending_flush_queue` skip
        # the flushed_local bump (`not partial` gate at the drain).
        # The `_record_stored_groups(rs.chain, rs.flushed_local)` call
        # below then under-counted: a buffer that happened to be a
        # FULL group (exactly _GROUP_BLOCKS pages) but hadn't been
        # flushed yet would never advance flushed_local → trie
        # stored-prefix was incomplete → next request's
        # get_num_new_matched_tokens elided one fewer group of
        # prefill (latency hit), and the partial group's bytes were
        # silently orphaned in the trie's flushed_local tally.
        for gidx in list(rs.active_group_buffers.keys()):
            buf = rs.active_group_buffers[gidx]
            # 2026-05-11 bug-fix-of-the-fix (write-batching audit
            # Finding 1): the original audit-#12 fix wrote
            # `getattr(buf, "pages_filled", 0) < _GROUP_BLOCKS`,
            # but `_GroupBuffer` defines `filled` (a set of
            # (layer, page) tuples) — NOT `pages_filled`. The
            # `getattr` default returned 0, so `is_partial` was
            # always True and the audit-#12 fix silently reverted
            # to its pre-fix behavior (every buffer flagged partial
            # → drain skips bump → stored-prefix undercounts).
            # `buf.is_complete()` is the canonical "all layers ×
            # all pages filled" check (line 701); negate to get
            # "this buffer hasn't reached its full target."
            is_partial = not buf.is_complete()
            self._flush_group(request_id, gidx, partial=is_partial)
        # 2026-05-11 Option-1: drain any pending batched writes for
        # this rid BEFORE we read `flushed_local` / pop the rs. The
        # for-loop above may have buffered full groups (partial=False
        # cases) that haven't reached the N threshold and would
        # otherwise be silently dropped on rs cleanup.
        # Finding 2: snapshot the dict membership under the lock so
        # the conditional drain doesn't race a concurrent pipeline
        # append.
        with self._write_batch_buf_lock:
            _has_buffered = bool(self._write_batch_buf.get(request_id))
        if _has_buffered:
            self._flush_write_batch_now(request_id)
        # 2026-05-10 audit #12: drain the pending flush queue
        # synchronously so the bumps from the partial=False flushes
        # above (full-buffer-via-this-path) reach `rs.flushed_local`
        # BEFORE we read it for `_record_stored_groups`. Without this
        # drain, the bumps land in the next `wait_for_pending_writes`
        # forward-thread call — by which point `rs` has been popped
        # (`self._requests.pop(request_id, ...)` below), so the drain
        # finds `rs is None` and silently skips the bump.
        #
        # `on_request_finished` is called per-rank by
        # `get_finished` (line ~1397), so the drain's `_tp_*`
        # collectives fire in lockstep across ranks (mirrors the
        # existing call site at the top of
        # `wait_for_pending_writes`).
        self._drain_pending_flush_queue()
        # Push the final group-count to the scheduler's prefix index so
        # a subsequent request with the same prefix can skip prefill.
        # wait_for_pending_writes is NOT called after on_request_finished,
        # so the last partial group's contribution to num_groups_written
        # would otherwise be lost.
        # Use flushed_local — the count of groups this request actually
        # wrote via _flush_group success. num_groups_written is inflated
        # by the inherited-prefix elision path in extract_and_record and
        # would cause the ledger to advertise groups the trie doesn't have.
        if rs.chain and rs.flushed_local > 0:
            self._record_stored_groups(rs.chain, rs.flushed_local)
            # Bug #5 fix (race-audit 2026-05-08): single atomic operation.
            # See _append_stored_chain_queue's docstring for rationale.
            _append_stored_chain_queue(
                list(rs.chain), rs.flushed_local)
            logger.debug("[diag-finish] rid=%s chain_len=%d flushed_local=%d "
                        "pushed to stored-prefix index",
                        request_id, len(rs.chain), rs.flushed_local)
        else:
            logger.debug("[diag-finish] rid=%s chain_len=%d flushed_local=%d "
                        "(NOT pushed — chain empty or no groups)",
                        request_id, len(rs.chain) if rs.chain else 0,
                        rs.flushed_local)
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
        # Step 2 per-rid Condition (2026-05-09 → 2026-05-10 audit #5):
        # wake any _score_one_request still waiting on this rid's
        # flush_cond before we drop the state. Without this, a Score
        # retry could block on the condition for the full timeout
        # even though the rs is gone. Bump flush_seq so any pending
        # wait_for(flush_seq > snapshot) predicate becomes true; the
        # waiter wakes, returns from wait_for, sees rs is gone via
        # downstream lookup, and falls through cleanly.
        _rs = self._requests.get(request_id)
        if _rs is not None:
            with _rs.flush_cond:
                _rs.flush_seq += 1
                _rs.flush_cond.notify_all()
        # ICMS_DIAG_SCORE_DUMP: save per-(rid) summary snapshot for
        # offline replay of alternate scoring algorithms. Pairs with the
        # per-(rid, layer) Q+picked_pages dumps written during Score
        # (~line 5592). quest_gpu_summaries is populated by record_page
        # under the same env gate (~line 9864). Score fires mid-forward-
        # pass before extract_and_record (end-of-pass) populates summaries
        # for the current chunk, so the per-Score `kmin/kmax` fields can
        # be empty even when this final snapshot has full data. Offline
        # consumer reads BOTH files and cross-references by (rid, layer).
        _dump_dir = os.environ.get("ICMS_DIAG_SCORE_DUMP", "")
        # 2026-05-15 ICMS_DIAG_SCORE_DUMP_PICKED_ONLY=1: skip cache-rid
        # dumps entirely (require last_q_by_layer, which only generate
        # rids have) AND drop the heavy `summaries` kmin/kmax payload
        # from the saved file. Used when the only downstream analysis
        # is membership (picked_by_layer is enough). Drops dump-dir
        # size by ~99% (per-rid file goes from ~195 MB → ~100 KB).
        # Forward-fill / alt-scoring analyses MUST run without this flag.
        _picked_only = os.environ.get(
            "ICMS_DIAG_SCORE_DUMP_PICKED_ONLY", "") in ("1", "true",
                                                          "True")
        # 2026-05-14 fix: ALSO dump when last_q_by_layer is non-empty
        # (Score fired even if extract_and_record didn't populate
        # quest_gpu_summaries — happens for "generate" rids that hit
        # prefix-cache and don't write fresh KV). Without this, mk2's
        # 5 generate rids dumped no q at all because the cache hit
        # zeroed quest_gpu_summaries.
        _dump_eligible = (
            rs.last_q_by_layer if _picked_only
            else (rs.quest_gpu_summaries or rs.last_q_by_layer))
        if _dump_dir and int(self._tp_rank) == 0 and _dump_eligible:
            try:
                import os as _os
                _os.makedirs(_dump_dir, exist_ok=True)
                _safe_rid = str(request_id).replace("/", "_")
                _path = _os.path.join(
                    _dump_dir, f"{_safe_rid}_summaries.pt")
                _by_layer = {}
                if not _picked_only:
                    for _lyr, _entries in rs.quest_gpu_summaries.items():
                        _items = sorted(_entries, key=lambda t: t[0])
                        _by_layer[int(_lyr)] = {
                            "abs_pids": [int(p) for p, _, _ in _items],
                            "kmin": torch.stack(
                                [m for _, m, _ in _items], dim=0).cpu(),
                            "kmax": torch.stack(
                                [m for _, _, m in _items], dim=0).cpu(),
                        }
                # 2026-05-14: include the per-(scored-layer) Q tensor
                # snapshot stashed by _score_one_request, plus the
                # picked_page_ids that Score returned at that layer. This
                # is what previously only lived in the per-(rid,layer).pt
                # files — and those only got written for the first rid
                # because the per-layer write path had silent failures
                # for subsequent rids. Embedding everything in the
                # per-rid summaries dump makes the data complete for
                # every rid in one place.
                _q_by_layer = {
                    int(L): t for L, t in
                    getattr(rs, "last_q_by_layer", {}).items()}
                _picked_by_layer = dict(
                    getattr(rs, "last_picked_by_layer", {}))
                _scores_by_layer = dict(
                    getattr(rs, "last_scores_by_layer", {}))
                torch.save({
                    "rid": str(request_id),
                    "summaries": _by_layer,
                    "q_by_layer": _q_by_layer,
                    "picked_by_layer": _picked_by_layer,
                    "scores_by_layer": _scores_by_layer,
                    "tp_rank": int(self._tp_rank),
                    "tp_size": int(self._tp_size),
                }, _path)
            except Exception as _e:
                logger.warning(
                    "[icms_diag_score_dump] summaries save failed "
                    "for rid=%s: %s", request_id, _e)

        # Now pop the request state and reset prefill_done.
        self._requests.pop(request_id, None)
        self._prefill_done = False
        # Recompute the all-dense cache after removing this rs. With
        # max_num_seqs=1 _requests is now empty and the cache becomes
        # False (no active rs to be dense for).
        self._cached_all_dense = (bool(self._requests) and all(
            getattr(r, "dense_mode", False)
            for r in self._requests.values()))
        # KV data is NOT evicted — it persists for prefix reuse by
        # subsequent requests. Eviction is managed by the server's LRU
        # when capacity is full.

        # Unregister from adaptive bandwidth allocator.
        if self._adaptive_allocator is not None:
            self._adaptive_allocator.unregister_request(request_id)

        # Fire RequestFinished to the server unconditionally so it can
        # release this rid's per-conn sink slot (and, at TP>1, walk
        # tp_groups_ to release peer ranks' slots — see handlers.cc:2290).
        # 2026-05-05 fix: was previously nested inside the
        # adaptive_allocator gate above, which meant adaptive_bandwidth=False
        # (bench default) silently skipped the RPC. Result: server slot map
        # accumulated across rids → 217 [sink-slots] warnings on a
        # TP=1 4-ex run, and the TP=2 fan-out leak fix #4 had nothing to
        # release because rank 0 never sent the frame.
        if self._client is not None and (
                self._tp_size <= 1 or self._tp_rank == 0):
            try:
                icms_rid = self._icms_request_id(request_id, 0)
                with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                    self._client.request_finished(icms_rid)
            except Exception as e:
                logger.debug(
                    "request_finished RPC failed for rid=%s: %s "
                    "(harmless; on_closed will GC the entry)",
                    request_id, e)
            # ICMS_QUEST_MODE=per_kv_head uses a separate registry on
            # the server side. Best-effort cleanup; the call is silent
            # (no reply) and a no-op on transports without per-head
            # support.
            if (os.environ.get("ICMS_QUEST_MODE", "") == "per_kv_head"
                    and hasattr(self._client, "request_finished_per_head")):
                try:
                    icms_rid = self._icms_request_id(request_id, 0)
                    with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                        self._client.request_finished_per_head(icms_rid)
                except Exception as e:
                    logger.debug(
                        "request_finished_per_head failed for rid=%s: %s",
                        request_id, e)

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

    def direct_write_group(self, request_id: str, chain: list[int],
                            summary_blob: bytes, kv_blob: bytes):
        rs = self._requests.setdefault(
            request_id, _RequestState(request_id=request_id))
        rs.chain = list(chain)
        # Option W: only rank 0 RPCs; other ranks no-op but keep local state.
        if self._tp_size > 1 and self._tp_rank != 0:
            return None
        # v6 wire: direct write is always a fresh registration
        # (parent=[], tail=full chain). Used by smoke tests that put
        # a complete chain in one shot.
        rank_chain = self._rank_chain(chain)
        with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
            return self._client.write_group(
                [], list(rank_chain),
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
        with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
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
