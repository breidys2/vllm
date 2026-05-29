# SPDX-License-Identifier: Apache-2.0
"""ICMS connector TP/NCCL collectives (rank-tagging, separate-NCCL-group cache,
score-reply broadcast, int/bool all-reduce/broadcast helpers).

Extracted verbatim from icms_connector.py (behavior-preserving split). All are
module-level functions already; imported back + re-exported by icms_connector
so the public + test import surface is unchanged.
"""
from __future__ import annotations

import os
import struct
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


_RANK_TAG_MAGIC = 0xE1C5A4EE00000000


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

    # 2026-05-29: fused payload broadcast. Pre-fix did 3 separate
    # broadcasts (pids, offs, scs) and 3 separate .cpu().tolist() drains
    # (each .cpu() implicitly drains the default CUDA stream against the
    # broadcast collective). At C ctx=126k pf=64 this fired 3 collectives
    # + 3 default-stream drains per scored boundary × 8 boundaries = 24
    # drain points per iter — measured as part of the ~150 ms Score-tail
    # cost in the workflow synthesis (wf_eae7e00b-a21).
    # Fused approach: pack pids(i64) + offs(i64) + scs(f32) into ONE
    # uint8 byte buffer; do ONE broadcast and ONE .cpu() sync; .view()
    # back to original dtypes on the receive side. Drops 3 broadcasts
    # to 1 and 3 .cpu() drains to 1 per scored boundary. Semantically
    # identical to the pre-fix 3-broadcast scheme (same bytes shipped,
    # same reconstruction order). Header broadcast at line 473 is
    # unchanged because its n_pages value gates buffer sizing on the
    # receive side.
    pids_bytes = 8 * n_pages
    offs_bytes = 8 * n_pages
    scs_bytes  = 4 * n_pages
    total_bytes = pids_bytes + offs_bytes + scs_bytes
    if tp_rank == 0:
        pids_t = torch.tensor(list(reply.page_ids),
                              dtype=torch.int64, device=dev)
        offs_t = torch.tensor(list(reply.sink_offsets),
                              dtype=torch.int64, device=dev)
        scs_t  = torch.tensor(list(reply.scores),
                              dtype=torch.float32, device=dev)
        payload = torch.cat([
            pids_t.view(torch.uint8),
            offs_t.view(torch.uint8),
            scs_t.view(torch.uint8),
        ])
    else:
        payload = torch.empty(total_bytes, dtype=torch.uint8, device=dev)
    dist.broadcast(payload, src=tp_group.first_rank, group=dev_group)

    # SINGLE host-sync: drain the entire fused buffer in one .cpu()
    # rather than three. tolist() on the CPU-side .view() slices is
    # pure host work — no GPU sync.
    payload_cpu = payload.cpu()
    pids_cpu = payload_cpu[:pids_bytes].view(torch.int64).tolist()
    offs_cpu = (payload_cpu[pids_bytes:pids_bytes + offs_bytes]
                .view(torch.int64).tolist())
    scs_cpu  = (payload_cpu[pids_bytes + offs_bytes:]
                .view(torch.float32).tolist())

    from icms_client.protocol import ScoreReply
    return ScoreReply(
        request_id=_i64_to_u64(int(hdr[9])),
        status=int(hdr[0]), trie_walk_ns=int(hdr[1]),
        summary_read_ns=int(hdr[2]), score_ns=int(hdr[3]),
        sink_write_ns=int(hdr[4]), cache_hit=bool(hdr[5]),
        concurrent_requests=int(hdr[6]),
        server_ingest_to_ready_ns=int(hdr[7]),
        effective_supply_bps=int(hdr[8]),
        page_ids=pids_cpu,
        scores=scs_cpu,
        sink_offsets=offs_cpu,
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


def _tp_broadcast_bool_list(values: list[bool], tp_rank: int,
                              tp_size: int) -> list[bool]:
    """Batched _tp_broadcast_bool: ONE broadcast for N bits instead of N.

    2026-05-28: introduced because `_drain_pending_flush_queue` was
    calling `_tp_broadcast_bool` once per pending-flush entry
    (~128 entries/iter at qwen3 ctx=65k pf=4096). Each per-entry
    broadcast does `int(t.item())` which drains the default CUDA
    stream — so per iter we paid 128 × default-stream-drain on the
    TTFT critical path. Measured at 1388 ms/iter via INSTR
    (drain_pending_flush_queue sum_ms=18045 across 13 iters); ≈85%
    of the 5200 ms TTFT regression vs Apr 27's 1163 ms.

    Single broadcast packs all bits into one int64 tensor of length
    N, broadcasts once, unpacks via a single `.tolist()` host-sync.
    Net: 1 stream drain instead of N → ~1.3 s/iter savings at this
    scale.

    TP=1 → returns input unchanged (no collective). Empty input →
    returns []. Failure mode: returns [False]*N on collective
    failure so all ranks symmetrically skip the bump (matches the
    legacy per-bit helper's failure semantics).
    """
    n = len(values)
    if tp_size <= 1 or n == 0:
        return [bool(v) for v in values]
    import torch.distributed as dist  # noqa: E402
    from vllm.distributed.parallel_state import get_tp_group
    try:
        tp_group = get_tp_group()
        dev_group = _get_icms_nccl_group() or tp_group.device_group
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        if tp_rank == 0:
            t = torch.tensor(
                [1 if v else 0 for v in values],
                dtype=torch.int64, device=dev)
        else:
            t = torch.zeros(n, dtype=torch.int64, device=dev)
        dist.broadcast(t, src=tp_group.first_rank, group=dev_group)
        # Single host sync to drain — ONE default-stream drain for the
        # whole batch instead of N.
        return [bool(x) for x in t.tolist()]
    except Exception as e:
        logger.warning(
            "[icms-tp] _tp_broadcast_bool_list failed (rank=%d, n=%d): "
            "%r — treating ALL as False to keep ledgers symmetric",
            tp_rank, n, e)
        return [False] * n


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
