# SPDX-License-Identifier: Apache-2.0
"""ICMS connector _Worker mixin: _WorkerFetchApplyMixin.

Extracted verbatim from icms_connector.py (behavior-preserving split).
Methods reference self.* attributes set by _WorkerBase.__init__ and call
sibling-mixin methods via the _Worker MRO; imports resolve from the neutral
helper modules so there is no cycle back into icms_connector.
"""
from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import IcmsConnectorMetadata
from icms_client.geometry import PAGE_TOKENS
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _ICMS_FULLTRACE_ENABLED
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _ICMS_TRACE_ENABLED
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import _RequestState
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_chain_fp
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_fulltrace
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_trace
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_allreduce_max_int
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_allreduce_min_int
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_broadcast_score_reply
from vllm.distributed.kv_transfer.kv_connector.v1 import icms_provenance
from contextlib import nullcontext
import os
import time
import torch
from vllm.logger import init_logger

# Log under the original connector logger name (behavior-preserving
# split): all split modules share one logger so log-name filtering,
# grep, and assertLogs see the same name as before.
logger = init_logger("vllm.distributed.kv_transfer.kv_connector.v1.icms_connector")
from icms_client.geometry import GROUP_PAGES  # noqa: E402
_GROUP_BLOCKS = GROUP_PAGES


class _WorkerFetchApplyMixin:
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
        # 2026-05-28 INSTR-FA: tag entry with layer + initial path. Path
        # is filled in below as the branch is taken (fast/slow/invalidated/
        # no-pending).
        _instr_fa = os.environ.get("ICMS_INSTR", "0") == "1"
        _t_fa_enter = time.perf_counter() if _instr_fa else 0.0
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
        # 2026-05-28 INVALIDATION GATE: the audit-Finding-4 invalidation
        # is correct for CHUNKED prefill where successive chunks
        # commit new ctx-prefix groups, BUT in non-chunked single-shot
        # prefill (paper bench's perf-sweep at TP=2 with
        # max_num_batched_tokens ≥ ctx+pf) the per-layer
        # save_kv_layer of THIS forward's pf tokens grows the
        # counter mid-forward. Each scored layer (6/12/18/24/30/36/42)
        # then sees `_committed_now > _committed_at_dispatch`, drops
        # `_fetch_all_complete`, and the slow path re-issues fetch_all
        # → 7 redundant fetches per iter, ~400 ms each = ~2.8 s of
        # serial waits on the TTFT critical path. Observed at qwen3
        # ctx=65k pf=4096 = 5226 ms vs Apr 27's 1163 ms.
        # Fix: gate the invalidation on `ICMS_FAPS_INVALIDATE_ON_GROWTH=1`
        # opt-in (chunked-prefill paths can re-enable). Default off so
        # single-shot prefill keeps the fast-path hit.
        _faps_inv_enabled = (
            os.environ.get("ICMS_FAPS_INVALIDATE_ON_GROWTH", "0") == "1")
        if (_faps_inv_enabled
                and getattr(rs, "_fetch_all_complete", False)
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
                if _instr_fa:
                    logger.info(
                        "[INSTR-FA] layer=%d path=fast-MISS dt_us=%.1f",
                        next_layer_idx,
                        (time.perf_counter() - _t_fa_enter) * 1e6)
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
            if _instr_fa:
                _n_reply = (len(reply.page_ids)
                            if reply is not None else 0)
                logger.info(
                    "[INSTR-FA] layer=%d path=fast-HIT dt_us=%.1f "
                    "n_pages_reply=%d (NO WIRE — reuse-cache hit)",
                    next_layer_idx,
                    (time.perf_counter() - _t_fa_enter) * 1e6,
                    _n_reply)
            return

        # Slow path: first scoring boundary for this request — issue
        # the single full-request FetchAll covering every layer.
        icms_rid = self._icms_request_id(rid, 0)
        # BUG-N7: read the cached per-request value populated by
        # on_step_start instead of rescanning _stored_chain_groups
        # (O(N stored × chain_len)) on every layer in the hot path.
        # 2026-05-28: prefer eff_groups_synced cache populated in
        # on_step_start (once per step) over per-call allreduce.
        # Cache-hit branch fires symmetrically across TP ranks
        # because the populating loop iterates meta.new_chains, which
        # is scheduler-broadcast (both ranks see same set). When the
        # cache is None (continuing request, chunked-prefill chunk
        # 1+, or rid not in this step's new_chains) we fall back to
        # the legacy per-call allreduce — semantically identical and
        # NCCL-safe since the None state is also symmetric.
        stored_groups = rs.stored_groups
        _cached_eff = getattr(rs, "eff_groups_synced", None)
        if _cached_eff is not None:
            effective_groups = int(_cached_eff)
        else:
            effective_groups = max(rs.num_groups_written, stored_groups)
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
                    _t_rpc0 = time.perf_counter() if _instr_fa else 0.0
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
                    if _instr_fa:
                        _rpc_dt_us = (time.perf_counter() - _t_rpc0) * 1e6
                        _n_reply = (len(reply.page_ids)
                                    if reply is not None else 0)
                        _layers_covered = (reuse_through
                                           - next_layer_idx + 1)
                        # Per-rank bytes: kv_page_bytes is FULL page (all KV
                        # heads). TP fan-out: each rank reads its share via
                        # kv_page_bytes / tp_size; assert at line 684-693
                        # guarantees clean divisibility.
                        _per_rank_page_bytes = (
                            int(self._geom.kv_page_bytes)
                            // max(1, int(self._tp_size)))
                        _bytes_per_rank = (_n_reply * _per_rank_page_bytes
                                           * _layers_covered)
                        _bytes_full = (_n_reply
                                       * int(self._geom.kv_page_bytes)
                                       * _layers_covered)
                        _wire_GBps = (_bytes_per_rank
                                      / max(1.0, _rpc_dt_us)
                                      * 1e6 / 1e9)
                        logger.info(
                            "[INSTR-FA] layer=%d path=slow-RPC dt_us=%.1f "
                            "total_pages_req=%d n_pages_reply=%d "
                            "layers_covered=%d bytes_per_rank_MB=%.2f "
                            "bytes_full_MB=%.2f wire_GBps=%.3f "
                            "reuse_through=%d use_flags=%d",
                            next_layer_idx, _rpc_dt_us, total_pages,
                            _n_reply, _layers_covered,
                            _bytes_per_rank / (1 << 20),
                            _bytes_full / (1 << 20), _wire_GBps,
                            reuse_through,
                            int(os.environ.get("ICMS_REPLY_EARLY", "1") == "1"))
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
            # 2026-05-19: with scored_layers_mask set (e.g. gemma-3 SWA),
            # the server's FetchAll worker only writes scored layers and
            # packs them by scored_rank in the sink. Reflect that here:
            # skip non-scored reuse layers (they short-circuit in
            # wait_for_layer anyway), and compute the per-layer offset
            # from scored_rank delta, not the abs_layer delta.
            _mask_set = self._geom.scored_layers_mask != 0
            # 2026-05-29 ICMS_FULL_FETCH=1: server's lifted Phase-2
            # writes ALL 48 layers (not just scored); populate
            # _pending_reuse for every reuse layer, not just scored
            # ones, and use raw abs_layer delta (sink slot index ==
            # layer index when sink is sized for num_layers).
            _full_fetch = (
                os.environ.get("ICMS_FULL_FETCH", "0") == "1")
            _skip_nonscored = _mask_set and not _full_fetch
            if _mask_set and not _full_fetch:
                _base_scored_rank = self._geom.scored_rank(next_layer_idx)
            for delta in range(1, reuse_through - next_layer_idx + 1):
                reuse_layer = next_layer_idx + delta
                if _skip_nonscored and not self._geom.is_scored(reuse_layer):
                    continue
                if _mask_set and not _full_fetch:
                    effective_delta = (
                        self._geom.scored_rank(reuse_layer)
                        - _base_scored_rank)
                else:
                    effective_delta = delta
                reuse_attn = f"model.layers.{reuse_layer}.self_attn.attn"
                reuse_offsets = [off + effective_delta * per_layer_bytes
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
    def wait_for_layer(self, layer_name: str):  # noqa: C901
        """Path B: fetch selected KV from icms → GPU + modify block table rows."""
        _instr_wfl = os.environ.get("ICMS_INSTR", "0") == "1"
        _t_wfl0 = time.perf_counter() if _instr_wfl else 0.0
        try:
            return self._wait_for_layer_impl(layer_name)
        finally:
            if _instr_wfl:
                _dt = (time.perf_counter() - _t_wfl0) * 1000.0
                # Per-scored-layer stall: the residual fetch+apply time NOT
                # hidden by ahead-of-time prefetch overlap — exactly the latency
                # stop-world would re-expose (it moves scoring onto this path).
                # Drop the >1ms gate for the ~8 scored/stride layers so
                # well-overlapped sub-ms scored layers are still captured.
                _idx = self._extract_layer_idx(layer_name)
                _scored = (
                    _idx is not None and self._geom is not None
                    and self._geom.is_scored(_idx)
                    and (self._geom.dense_layers_mask != 0
                         or (_idx % self._score_stride) == 0))
                if _scored:
                    logger.info("[INSTR] wait_scored_layer[%s]: %.3fms",
                                _idx, _dt)
                elif _dt > 1.0:  # non-scored: keep the >1ms gate
                    logger.info("[INSTR] wait_for_layer[%s]: %.2fms",
                                layer_name, _dt)
    def _wait_for_layer_impl(self, layer_name: str):
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
        # 2026-06-01 true post-flip bypass for mode (c) (DECODE_APPLY=1).
        # Once every active rs is dense_mode, the ONLY work the slow path
        # below does for this layer is `icms_fetch_state.clear()` to flush a
        # possibly-stale set_active(trimmed bt) — see the all-dense block at
        # ~L1000. That clear is a global dict .clear() (~µs) and idempotent,
        # so we do exactly it and early-return, skipping the per-layer
        # lock + _pending_scores pop + INSTR/diag traversal that otherwise
        # ran ×48 layers × every decode iter post-flip. That traversal was
        # the ~1.3 ms/token connector-hook tax leaving mode (c) decode
        # slower than the connector-free FR baseline. DECODE_APPLY=0 (mode d)
        # is handled by the fast path above; this targets DECODE_APPLY=1.
        if (self._cached_all_dense and self._prefill_done
                and os.environ.get("ICMS_DECODE_APPLY") != "0"):
            from vllm.v1.attention import icms_fetch_state
            icms_fetch_state.clear()
            return
        # 2026-05-09: with a non-zero scored_layers_mask, non-scored layers
        # never had Score fired (gated in on_layer_score) so _pending_scores
        # has nothing for them. But the prior scored layer's set_active()
        # leaves a trimmed bt active — vLLM would use that for this layer's
        # attention, corrupting it. Clear active state and short-circuit so
        # attention falls back to natural full-context bt (e.g. SW layers in
        # gemma-3 attend over the full prefill, with their own SW mask).
        _abs_layer_for_mask = self._extract_layer_idx(layer_name)
        # 2026-05-29 ICMS_FULL_FETCH=1: in B-full-fetch mode the server
        # writes ALL 48 layers' KV to the sink and _pending_reuse is
        # populated for every layer (see fetch_all reuse loop above),
        # so non-scored layers should NOT short-circuit — they must
        # fall through to pop+apply so attention reads from the sink
        # (where Phase-2 just landed full-page data) rather than from
        # vLLM's natural bt.
        # CRITICAL gate (2026-05-29 v2): the lift is ONLY safe when this
        # specific call has data pending for the layer (B fetch_all path
        # populates _pending_reuse for all 48 layers). For C (Score
        # path), only the scored layer + its reuse range get data; other
        # non-scored layers must still short-circuit or wait_for_layer
        # spins to its 5s flag-timeout. Probe the pending dicts to
        # distinguish: presence ⇒ B full-fetch, absence ⇒ C/legacy.
        _full_fetch_wfl = (
            os.environ.get("ICMS_FULL_FETCH", "0") == "1")
        _has_pending_for_layer = False
        if _full_fetch_wfl and _abs_layer_for_mask is not None:
            _canonical = (
                f"model.layers.{_abs_layer_for_mask}.self_attn.attn")
            with self._score_lock:
                _has_pending_for_layer = (
                    _canonical in self._pending_scores
                    or _canonical in self._pending_reuse)
        if (_abs_layer_for_mask is not None
                and not self._geom.is_scored(_abs_layer_for_mask)
                and not _has_pending_for_layer):
            # Contiguous-reuse mode (dense_layers_mask != 0): non-scored
            # layers should already have a pending reuse entry promoted
            # by on_layer_reuse from the per-layer hook chain. Fall
            # through to the normal pop+apply so they apply the cached
            # selection. Only short-circuit (legacy gemma-SWA behavior)
            # when no pending data exists.
            _fallthrough = False
            if self._geom.dense_layers_mask != 0:
                _canonical_key = (
                    f"model.layers.{_abs_layer_for_mask}.self_attn.attn")
                with self._score_lock:
                    _fallthrough = _canonical_key in self._pending_scores
            if not _fallthrough:
                from vllm.v1.attention import icms_fetch_state
                icms_fetch_state.clear()
                # KV-provenance: this is the non-scored short-circuit
                # path. Attention will read the NATURAL bt from
                # attn_metadata, which includes ext_comp blocks that
                # ICMS may not have populated for this layer.
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
        # 2026-05-28 INSTR-WFL: per-layer breakdown. Captures:
        #   - ready_at_call: was the per-layer flag already set on entry?
        #   - spin_us: time spinning on is_layer_ready (only counts when
        #     ready_at_call=False)
        # Followed by INSTR-WFL-POST below for everything after the spin.
        # Aggregator: /tmp/agg_instr.py -- grep "WFL " to filter.
        if os.environ.get("ICMS_INSTR", "0") == "1" and abs_layer is not None:
            logger.info(
                "[INSTR-WFL] layer=%d ready=%d spin_us=%.1f",
                abs_layer, int(ready_at_call), spin_us)
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
                # 2026-05-28 INSTR-APPLY: per-(rid, layer) apply timing.
                # Combined with INSTR-WFL `spin_us`, fully accounts for
                # wait_for_layer's wall time. Reply page count + reuse
                # offset count exposes per-layer transfer scale so we
                # can see if apply time scales with k or is fixed-cost.
                if os.environ.get("ICMS_INSTR", "0") == "1":
                    _abs_l = self._extract_layer_idx(layer_name)
                    logger.info(
                        "[INSTR-APPLY] layer=%s abs=%s rid=%s "
                        "apply_us=%.1f n_pages=%d n_sink_off=%d",
                        layer_name, str(_abs_l), rid[:8], apply_us,
                        len(reply.page_ids),
                        len(getattr(reply, "sink_offsets", []) or []))
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
        # 2026-05-28: prefer eff_groups_synced cache (set in
        # on_step_start) over per-call recompute + per-layer allreduce.
        # The per-layer .item() of the legacy path drained the default
        # CUDA stream which holds vLLM's per-layer attention NCCL, so
        # this apply blocked behind it for ~50 ms × N_dense_layers
        # between scored layers — observed at qwen3 ctx=65k pf=4096
        # as ~400 ms/scored-layer × 8 = 3.2 s/iter of pure stream wait
        # (4× TTFT regression). Cache is bit-identical to the legacy
        # per-call result within one forward pass (inputs are stable
        # mid-forward; see comment in on_step_start). Falls back to
        # per-call allreduce when cache is None (cold-start or rid
        # not in this step's new_chains, e.g. chunked-prefill chunk
        # 1+). Both ranks see the same None state because the
        # populating loop iterates the symmetric meta.new_chains
        # source, so the fallback fires symmetrically → NCCL-safe.
        stored_groups = rs.stored_groups
        _cached_eff = getattr(rs, "eff_groups_synced", None)
        if _cached_eff is not None:
            effective_groups = int(_cached_eff)
        else:
            effective_groups = max(rs.num_groups_written, stored_groups)
            if self._tp_size > 1:
                effective_groups = _tp_allreduce_max_int(
                    effective_groups, self._tp_size)
        if os.environ.get("ICMS_INSTR", "0") == "1":
            _abs_l_ic = self._extract_layer_idx(layer_name)
            logger.info(
                "[INSTR-APPLY-CACHE] layer=%s hit=%d cached=%s eff=%d",
                str(_abs_l_ic), int(_cached_eff is not None),
                str(_cached_eff), int(effective_groups))
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
                and rs._apply_cached_new_bt is not None
                # Faithful Quest: the fast-path cached bt carries no per-head
                # mask, so force the slow path (which rebuilds bt + head_mask)
                # for every faithful layer. Correctness over the stride-reuse
                # speedup; the baseline is not perf-sensitive.
                and os.environ.get("ICMS_SCORING_MODE") != "faithful_quest"):
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
        # 2026-05-28: gate on multi-rid mode. In single-rid mode (paper
        # bench) the comment above guarantees no asymmetry, so the
        # allreduce is pure overhead — and its `.item()` drains the
        # default CUDA stream, serializing apply behind vLLM's prior
        # per-layer attention NCCL (~400 ms/scored-layer at qwen3
        # ctx=65k). Multi-rid keeps the allreduce as the comment
        # intended. ICMS_SKIP_BT_ROW_ALLREDUCE=1 forces skip
        # regardless (diag knob for parity tests).
        if (self._tp_size > 1
                and self._is_multi_rid_mode()
                and os.environ.get("ICMS_SKIP_BT_ROW_ALLREDUCE", "0") != "1"):
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
            # ── Faithful Quest per-KV-head mask, aligned to new_bt columns ──
            # [#1 GPU-VALIDATION POINT] new_bt = [context blocks ++ continuation
            # blocks]. The context columns are in `cumulative_pids` order (see
            # the phys_blocks_for_bt_dev build, ~L2062-2068); continuation
            # columns (j >= n_ctx) are the query's own window → unmasked (1) for
            # every head, matching HF's locally_have ∪ partial-last-page.
            # _pending_faithful_masks[layer_name][rid] holds the full [H_kv, P]
            # page-id-keyed selection mask (None on every non-faithful path).
            _faithful_hm_state = None
            _fmasks = self._pending_faithful_masks.get(layer_name)
            _full_sel = _fmasks.get(rid) if _fmasks else None
            if _full_sel is not None:
                try:
                    _Hkv, _P = _full_sel.shape
                    _bt_ctx = cumulative_pids if cumulative_pids else valid_pids
                    _ncol = int(new_bt.shape[1])
                    _hm = torch.ones(_Hkv, _ncol, dtype=torch.int8,
                                     device=_full_sel.device)
                    _ctx_ids = [int(p) for p in (_bt_ctx or [])][:_ncol]
                    if _ctx_ids:
                        _cols = torch.as_tensor(
                            [p if 0 <= p < _P else 0 for p in _ctx_ids],
                            dtype=torch.long, device=_full_sel.device)
                        _hm[:, :_cols.numel()] = _full_sel.index_select(
                            1, _cols).to(torch.int8)
                    _faithful_hm_state = _hm.unsqueeze(0).contiguous()  # [1,H_kv,ncol]
                except Exception as _e_hm:
                    logger.warning(
                        "[faithful_quest] head_mask build failed layer=%s "
                        "rid=%s: %r — falling back to dense-over-union",
                        layer_name, rid[:8], _e_hm)
                    _faithful_hm_state = None
            # ICMS_FAITHFUL_DEBUG_BT=1: one-shot layer-0 dump to VERIFY the
            # bt-column ↔ page-id alignment (the #1 correctness risk). Compare
            # new_bt context columns against cumulative_pids/valid_pids; the
            # per-head mask sums should equal B_eff (+ continuation count).
            if (_faithful_hm_state is not None
                    and os.environ.get("ICMS_FAITHFUL_DEBUG_BT") == "1"
                    and layer_idx_for_cache == 0
                    and not getattr(self, "_faithful_bt_dbg_logged", False)):
                self._faithful_bt_dbg_logged = True
                try:
                    _nb0 = new_bt.detach().cpu().tolist()[0]
                    logger.info(
                        "[faithful-bt-dbg] layer0 rid=%s ctx_pages=%d ncol=%d "
                        "new_bt[:8]=%s cumulative_pids[:8]=%s "
                        "valid_pids[:8]=%s hm_sum_per_head=%s",
                        rid[:8], context_pages, int(new_bt.shape[1]),
                        _nb0[:8],
                        (list(cumulative_pids)[:8] if cumulative_pids else None),
                        list(valid_pids)[:8],
                        _faithful_hm_state[0].sum(dim=1).cpu().tolist())
                except Exception:
                    pass
            _state_slow = icms_fetch_state.IcmsFetchState(
                key_cache=main_key,
                value_cache=main_value,
                block_table=new_bt,
                seq_lens=new_sl,
                max_seq_len=new_seq_len,
                head_mask=_faithful_hm_state,
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
    def restore_attn_metadata(self, layer_name: str):
        """Clear the ICMS fetch state after attention ran with it."""
        from vllm.v1.attention import icms_fetch_state
        icms_fetch_state.clear()
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
