# SPDX-License-Identifier: Apache-2.0
"""ICMS connector _Worker mixin: _WorkerStateMixin.

Extracted verbatim from icms_connector.py (behavior-preserving split).
Methods reference self.* attributes set by _WorkerBase.__init__ and call
sibling-mixin methods via the _Worker MRO; imports resolve from the neutral
helper modules so there is no cycle back into icms_connector.
"""
from __future__ import annotations

# C++ ArrivalPoller (slack diagnostic), optional. Mirrors icms_connector.
try:
    from icms_client._icms_client import ArrivalPoller as _ArrivalPoller  # noqa: E402
    _HAVE_ARRIVAL_POLLER = True
except ImportError:
    _ArrivalPoller = None  # type: ignore[assignment]
    _HAVE_ARRIVAL_POLLER = False

from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import IcmsConnectorMetadata
from icms_client.geometry import PAGE_TOKENS
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _ICMS_TRACE_ENABLED
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import _RequestState
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _append_stored_chain_queue
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_trace
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _instr_timing
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_allreduce_max_int
from vllm.distributed.kv_transfer.kv_connector.v1 import icms_provenance
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


class _WorkerStateMixin:
    def _get_stored_context_groups(self, chain: list[int]) -> int:
        """Find the longest stored chain prefix matching the given chain.

        Returns the number of groups written for that prefix (0 if none).
        This allows a new request to know how many context pages were
        stored by a prior request with the same prefix.

        PR7a: read-side acquires self._stored_chain_groups_lock to
        snapshot the list, then iterates the snapshot lock-free. Under
        prefill mode there's no writeback daemon thread — the lock is
        uncontended and adds ~50 ns. Under eviction mode the daemon
        thread mutates the list from _flush_group's task closure, so
        the snapshot pattern avoids a torn read.
        """
        with self._stored_chain_groups_lock:
            snapshot = list(self._stored_chain_groups)
        best = 0
        best_src_len = 0
        best_src_n = 0
        for stored_chain, n_groups in snapshot:
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
                len(chain), len(snapshot),
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

        PR7a: write-side holds self._stored_chain_groups_lock for the
        scan + mutate (also the append fallback). The daemon thread
        (eviction-mode WriteGroup completion callback) calls this; the
        forward thread also calls this from the legacy prefill path
        and from _drain_pending_flush_queue. Without the lock, a torn
        write between max() and assignment could lose updates.
        """
        assert n_groups <= len(chain), (
            f"_record_stored_groups: n_groups={n_groups} > len(chain)="
            f"{len(chain)}; query side will silently cap at match_len. "
            f"Caller is recording more groups than chain elements — "
            f"likely a stale-chain or wrong-counter bug.")
        with self._stored_chain_groups_lock:
            for i, (sc, _) in enumerate(self._stored_chain_groups):
                if sc == chain:
                    self._stored_chain_groups[i] = (
                        chain, max(_, n_groups))
                    return
            self._stored_chain_groups.append((list(chain), n_groups))
    @_instr_timing("on_step_start")
    def on_step_start(self, meta: IcmsConnectorMetadata):
        """Called from start_load_kv. Drains scheduler metadata into caches."""
        t_step = time.perf_counter()
        self.stats.advance_step()

        # PR7a of ICMS eviction-mode refactor: bump step counter, prune
        # stale chain snapshots. Prune cadence: every
        # max_age_steps // 4 steps (at most). For default max_age=200
        # this is every 50 steps; for small max_age (tests), every
        # max(1, max_age // 4) steps. Effectively bounds the snapshot
        # dict memory to the rids whose chains finished within the
        # last max_age_steps. Prefill mode walks an empty dict ⇒ no-op.
        self._step_counter += 1
        _max_age = getattr(self, "_chain_snapshot_max_age_steps", 200)
        if (self._step_counter - self._last_chain_snapshot_prune_step
                >= max(1, _max_age // 4)
                and self._chain_snapshots):
            cutoff = self._step_counter - _max_age
            stale = [rid for rid, (_chain, fin_step)
                     in self._chain_snapshots.items()
                     if fin_step < cutoff]
            for rid in stale:
                del self._chain_snapshots[rid]
            if stale:
                self._chain_snapshots_pruned_total = (
                    getattr(self, "_chain_snapshots_pruned_total", 0)
                    + len(stale))
            self._last_chain_snapshot_prune_step = self._step_counter

        # PR3 of ICMS eviction-mode refactor (2026-05-31): receive
        # ChainLocator tuples ferried from the scheduler's BlockLocator
        # via IcmsConnectorMetadata. Under prefill mode this field is
        # always empty (scheduler-side `_pending_evicted_locators` is
        # never populated because the eviction callback registration is
        # gated on supports_eviction_writes=False), so the conditional
        # below is a free skip — no behavior change for prefill.
        #
        # PR5 (2026-05-31): dispatch to _EvictionExtractor which
        # accumulates per-(rid, group) buffers and enqueues complete
        # groups to the writeback queue (PR4). The extractor is
        # allocated only under eviction mode; absence under prefill
        # mode short-circuits with the getattr-default-None check.
        if (getattr(self, "_write_mode", "prefill") == "eviction"
                and getattr(meta, "evicted_chain_locators", None)):
            n = len(meta.evicted_chain_locators)
            # Cheap aggregate counter — surfaces in PR12 telemetry.
            self._eviction_locators_received_total = (
                getattr(self, "_eviction_locators_received_total", 0) + n)
            extractor = getattr(self, "_eviction_extractor", None)
            if extractor is not None:
                flushed = extractor.process_locators(
                    meta.evicted_chain_locators)
                if flushed:
                    logger.debug(
                        "[icms-eviction] PR5 step: %d locators in, "
                        "%d groups flushed to writeback queue "
                        "(extractor totals: pages=%d groups=%d drops=%d).",
                        n, flushed, extractor.pages_received,
                        extractor.groups_completed,
                        extractor.groups_dropped_writeback_full)
            else:
                logger.debug(
                    "[icms-eviction] PR3 worker received %d "
                    "ChainLocators (extractor not allocated — "
                    "PR5 incomplete).", n)
        # PR7b of ICMS eviction-mode refactor (2026-05-31): process new
        # waiters ferried from the scheduler. MUST run AFTER
        # process_locators above so any chain that completed THIS step
        # has already landed in _completed_chains_lru via the daemon
        # signal — waiters arriving in the SAME step as completion
        # then get routed immediately. Reverse order risks a stranded
        # waiter (chain completes between metadata-ferry-arrival and
        # this call, waiter sees no completed entry → registers in
        # _waiters_by_chain, daemon already drained → orphan).
        if (getattr(self, "_write_mode", "prefill") == "eviction"
                and getattr(meta, "chain_waiters", None)):
            try:
                self.pr7b_ingest_chain_waiters(meta.chain_waiters)
            except Exception:
                logger.exception(
                    "[icms-pr7b] ingest_chain_waiters failed for %d "
                    "entries — waiters may be stranded; vLLM will "
                    "timeout WAITING_FOR_REMOTE_KVS",
                    len(meta.chain_waiters))
        if (getattr(self, "_write_mode", "prefill") == "eviction"
                and getattr(meta, "scavenger_rids", None)):
            n = len(meta.scavenger_rids)
            self._scavenger_rids_received_total = (
                getattr(self, "_scavenger_rids_received_total", 0) + n)
            logger.debug(
                "[icms-eviction] PR3 worker received %d scavenger "
                "rids from scheduler bridge (total: %d).",
                n, self._scavenger_rids_received_total)
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

        # 2026-05-28 v2: once-per-step TP-allreduce-MAX of
        # effective_groups, cached on rs.eff_groups_synced. Iterates
        # meta.new_chains (scheduler-symmetric set; same dict-insertion
        # order on both ranks — already used by the n_stored allreduce
        # above) so NCCL participation is bit-symmetric. PRE-fix the
        # three hot-path call sites (apply/score/fetch_all) did this
        # per-layer; the helper's `int(t.item())` after `all_reduce`
        # drained the default CUDA stream which holds vLLM's per-layer
        # attention NCCL → ~400 ms/scored-layer × 8 = 3.2 s/iter of
        # serial stream-wait at qwen3 ctx=65k pf=4096.
        # v1 of this hoist iterated self._requests.items() and
        # DEADLOCKED within seconds: rid sets can transiently diverge
        # across ranks at iter boundaries (on_request_finished firing
        # asymmetrically). meta.new_chains is the only known-symmetric
        # source. For rids NOT in meta.new_chains (continuing requests
        # / chunked prefill chunks 1+), eff_groups_synced stays at its
        # prior value or None, and the hot-path readers fall back to
        # per-call allreduce — correctness-preserving.
        for _rid_es in meta.new_chains.keys():
            _rs_es = self._requests.get(_rid_es)
            if _rs_es is None:
                continue
            _eff_in = max(_rs_es.num_groups_written, _rs_es.stored_groups)
            if self._tp_size > 1:
                _eff_in = _tp_allreduce_max_int(_eff_in, self._tp_size)
            _rs_es.eff_groups_synced = int(_eff_in)

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
    @_instr_timing("on_request_finished")
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
        _instr_on_orf = os.environ.get("ICMS_INSTR", "0") == "1"
        _t_drainrid0 = time.perf_counter()
        ok = self._write_pipeline.drain_rid(
            request_id, timeout=_drain_timeout_s)
        if _instr_on_orf:
            logger.info("[INSTR] on_request_finished.drain_rid: %.2fms",
                        (time.perf_counter() - _t_drainrid0) * 1000.0)
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

        # ──────────────────────────────────────────────────────────────
        # PR6 of ICMS eviction-mode refactor (2026-05-31): scavenger
        # fire-and-forget under WRITE_MODE=eviction. Reviewer 3 Option a
        # (round-DOWN): complete groups still buffered in
        # _EvictionExtractor get enqueued via the writeback queue with
        # priority='low'; incomplete groups get DROPPED. Reviewer 1
        # HIGH #4: capture rs.chain EAGERLY here (engine thread)
        # because the writeback queue's daemon thread fires the task
        # closure later, AFTER self._requests.pop has run. Reviewer 1
        # HIGH #5: pop ALL (rid, *) buffer keys so dead-rid late-
        # eviction callbacks (which can fire seconds after rid finishes
        # via BlockLocator snapshot retention) cannot orphan bytes.
        # Reviewer 1 MEDIUM #7: one-shot guard via
        # self._scavenger_fired_rids neutralizes the facade.request_finished
        # + facade.get_finished double-call.
        #
        # Reviewer 2 BLOCKER #1: NO _inflight_chains mark, NO is_async
        # path, NO finished_recving return — those all deferred to PR7
        # when the real WriteGroup RPC + proper completion mechanism
        # (KVConnectorOutput.finished_recving keyed on WAITING rids,
        # NOT FINISHING rids per Reviewer 1 BLOCKER #3) land together.
        # See docs/icms_env_precedence_matrix_v2.md (PR7 TODO).
        if (getattr(self, "_write_mode", "prefill") == "eviction"
                and getattr(self, "_eviction_extractor", None) is not None):
            _scav_set = self._scavenger_fired_rids
            if request_id in _scav_set:
                logger.debug(
                    "[icms-eviction] scavenger already fired for "
                    "rid=%s; skipping duplicate (facade get_finished "
                    "double-call)", request_id)
            else:
                _scav_set.add(request_id)
                _eager_chain = list(rs.chain) if rs.chain else None
                t_evf = time.perf_counter()
                try:
                    n_flushed, n_dropped = (
                        self._eviction_extractor
                            .flush_remaining_for_rid(
                                request_id, _eager_chain))
                except Exception:
                    logger.exception(
                        "[icms-eviction] scavenger flush_remaining_"
                        "for_rid failed for rid=%s", request_id)
                    n_flushed, n_dropped = 0, 0
                ev_us = (time.perf_counter() - t_evf) * 1e6
                logger.debug(
                    "[icms-eviction] scavenger flush rid=%s "
                    "flushed=%d dropped=%d (%.1f us)",
                    request_id, n_flushed, n_dropped, ev_us)
                # Soft-assert <10ms wall-time invariant (Reviewer 2
                # BLOCKER #1). Warning is operator-visible without
                # log scrape; CI can grep this as a regression canary.
                if ev_us > 10000.0:
                    logger.warning(
                        "[icms-eviction] scavenger flush slow: "
                        "%.1f us (target <10ms) — investigate "
                        "writeback queue put_or_drop contention.",
                        ev_us)

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
        # Faithful Quest per-head masks (dormant unless faithful_quest).
        _pfm = getattr(self, "_pending_faithful_masks", None)
        if _pfm:
            for inner in _pfm.values():
                inner.pop(request_id, None)
            self._pending_faithful_masks = {
                k: v for k, v in _pfm.items() if v}
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
                # picked_call_log: ordered list of every score call's
                # picks for this rid, with the (call_idx, layer,
                # is_decode, budget, k, total_pages, picked) tuple.
                # Analyzer should partition by is_decode to separate
                # prefill from decode-time picks.
                _picked_call_log = list(getattr(rs, "picked_call_log", []))
                torch.save({
                    "rid": str(request_id),
                    "summaries": _by_layer,
                    "q_by_layer": _q_by_layer,
                    "picked_by_layer": _picked_by_layer,
                    "picked_call_log": _picked_call_log,
                    "scores_by_layer": _scores_by_layer,
                    "tp_rank": int(self._tp_rank),
                    "tp_size": int(self._tp_size),
                }, _path)
            except Exception as _e:
                logger.warning(
                    "[icms_diag_score_dump] summaries save failed "
                    "for rid=%s: %s", request_id, _e)

        # PR7a of ICMS eviction-mode refactor (2026-05-31): snapshot
        # rs.chain into self._chain_snapshots BEFORE we pop the request
        # state, so eviction callbacks that fire many steps later can
        # still resolve rid → chain in process_locators. Snapshot is
        # pruned by age via _maybe_prune_chain_snapshots in on_step_start.
        # Allocated under both modes; in prefill the snapshot is never
        # consulted (no eviction extractor). Branch is the same one-line
        # write either way, so the prefill cost is constant.
        if rs.chain:
            self._chain_snapshots[request_id] = (
                list(rs.chain), int(self._step_counter))

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
