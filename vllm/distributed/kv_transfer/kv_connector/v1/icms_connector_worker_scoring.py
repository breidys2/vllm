# SPDX-License-Identifier: Apache-2.0
"""ICMS connector _Worker mixin: _WorkerScoringMixin.

Extracted verbatim from icms_connector.py (behavior-preserving split).
Methods reference self.* attributes set by _WorkerBase.__init__ and call
sibling-mixin methods via the _Worker MRO; imports resolve from the neutral
helper modules so there is no cycle back into icms_connector.
"""
from __future__ import annotations
from icms_client.sink import Sink

from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import IcmsConnectorMetadata
from icms_client.client import IcmsError
from icms_client.geometry import PAGE_TOKENS
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _ICMS_FULLTRACE_ENABLED
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _ICMS_TRACE_ENABLED
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import _RequestState
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _allow_batch
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_chain_fp
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_fulltrace
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_trace
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _pack_fetch_bitmap
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_allreduce_max_int
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_allreduce_max_tensor
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_broadcast_bool
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_broadcast_score_reply
from vllm.distributed.kv_transfer.kv_connector.v1 import icms_connector_trace as _trace
import errno
import numpy as np
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


class _WorkerScoringMixin:
    def _maybe_enable_async_score(self) -> bool:
        """Return True if the client is in async multiplexed scoring mode
        (so the caller drops the _rpc_lock and scores concurrently).

        Opt-in (ICMS_ASYNC_SCORE=1) and only flipped on when the
        SKIP_ARCHIVE sentinel is present — i.e. measure-time, when writes
        are suppressed, so no sync write/evict RPC can race the dispatcher
        for the CQ. The flip is one-time and done under _rpc_lock so no
        in-flight sync RPC overlaps it. Idempotent + cheap on the hot path.
        """
        cli = getattr(self, "_client", None)
        # UNCONDITIONAL first-call probe (before ANY early return) — last run the
        # dbg never logged, meaning we bailed at the hasattr/async_enabled gate.
        if not getattr(self, "_async_dbg_logged", False):
            self._async_dbg_logged = True
            try:
                _ae = cli.async_enabled() if (cli is not None and hasattr(cli, "async_enabled")) else "NO_METHOD"
            except Exception as _e:  # noqa: BLE001
                _ae = "THREW:%s" % _e
            logger.info("[async-score][dbg] cli_type=%s has_enable_async=%s "
                        "has_async_enabled=%s async_enabled=%s env=%s file=%s",
                        type(cli).__name__ if cli is not None else None,
                        hasattr(cli, "enable_async") if cli is not None else False,
                        hasattr(cli, "async_enabled") if cli is not None else False,
                        _ae,
                        os.environ.get("ICMS_ASYNC_SCORE") == "1",
                        os.path.exists("/tmp/icms_async_enable"))
        if cli is None or not hasattr(cli, "async_enabled"):
            return False
        try:
            if cli.async_enabled():
                return True
        except Exception:
            return False
        # Opt-in gate. ICMS_ASYNC_SCORE env vars do NOT reliably reach the
        # vLLM worker process (spawned clean; only a subset of os.environ is
        # injected at runtime), so ALSO honor a file sentinel which provably
        # does reach the worker (same mechanism SKIP_ARCHIVE uses).
        _async_env = os.environ.get("ICMS_ASYNC_SCORE") == "1"
        _async_file = os.path.exists("/tmp/icms_async_enable")
        sentinel = os.environ.get("ICMS_SKIP_ARCHIVE_SENTINEL", "")
        _sent_ok = bool(sentinel and os.path.exists(sentinel))
        if not (_async_env or _async_file):
            return False
        if not _sent_ok:
            return False  # not measure-time → writes may still fire; stay sync
        with self._rpc_lock:  # ensure no sync RPC is mid-flight during the flip
            try:
                if not cli.async_enabled():
                    _ok = cli.enable_async()
                    logger.info("[async-score] enable_async()=%s (SKIP_ARCHIVE "
                                "active): scores now %s", _ok,
                                "concurrent" if _ok else "STILL SYNC")
            except Exception as e:  # noqa: BLE001
                logger.warning("[async-score] enable_async failed: %s", e)
        try:
            return cli.async_enabled()
        except Exception:
            return False

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
        #
        # 2026-05-19: contiguous-reuse mode (dense_layers_mask != 0) instead
        # routes non-scored layers through on_layer_reuse so they apply the
        # cached page selection. Required for full-attention models (qwen3)
        # where the gemma-SWA fallback path corrupts hidden states.
        _contiguous_reuse = self._geom.dense_layers_mask != 0
        if not self._geom.is_scored(next_layer_idx):
            if _contiguous_reuse:
                self.on_layer_reuse(next_layer_idx, budget, stats)
            self._slack_probe_pre_hook(next_layer_idx)
            return
        try:
            # Stride modular gate: only consulted in legacy mode. With
            # contiguous-reuse every scored layer fires Score and the next
            # scored layer in the mask delimits the reuse window.
            #
            # 2026-05-30 fix (workflow wrrwqsdto): when scored_layers_mask
            # is set, the is_scored() filter at line 91 already enforced
            # selectivity and the mask itself encodes the (possibly
            # non-uniform) stride. The absolute-index modulo here is then
            # redundant AND wrong for masks not aligned to score_stride
            # — e.g. gemma-3's hybrid mask {5,11,17,23,29,35,41,47,53,59}
            # under stride=6: every scored layer satisfies idx%6==5, so
            # EVERY scored gemma layer was being routed to on_layer_reuse,
            # bypassing _on_layer_score_impl and the register_request call
            # → adaptive allocator never saw a single gemma-3 request →
            # C-config effectively ran at full budget. qwen3 + mistral
            # scored masks are stride-6-aligned by construction, so for
            # them the gate is already a no-op on every scored layer and
            # this change is byte-identical.
            _mask_is_stride = self._geom.scored_layers_mask != 0
            if (not _contiguous_reuse
                    and not _mask_is_stride
                    and (next_layer_idx % self._score_stride) != 0):
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
            # Per-layer dense budget override: ICMS_DENSE_LAYERS forces
            # budget=1.0 on the listed layers (still subject to the mask
            # gate above). Picks all pages → set_active leaves bt at
            # natural length → effectively dense attention for L0,L1.
            # Skip the override under ICMS_SCORING_MODE=subset_max: in
            # that mode dense_layers_mask is used purely to activate
            # contiguous-reuse (line 4368 gate); the scored layers
            # themselves must keep their sparse subset_max budget.
            if (self._geom.is_dense(next_layer_idx)
                    and os.environ.get("ICMS_SCORING_MODE", "")
                        != "subset_max"):
                budget = 1.0
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
        # 2026-05-29 single-pass INSTR: wrap whole call to bound the full
        # Score-path overhead per scored-layer boundary. Inner INSTRs
        # (Q-CPU, SCORE-RPC, TP-BCAST) decompose where the time goes.
        _instr_sco = os.environ.get("ICMS_INSTR", "0") == "1"
        _t_sco_enter = time.perf_counter() if _instr_sco else 0.0
        try:
            return self.__score_one_request_impl(
                rid, rs, req_idx, next_layer_idx,
                quest_query, budget, stats, connector_meta)
        finally:
            if _instr_sco:
                logger.info(
                    "[INSTR-SCORE-TOTAL] layer=%d rid=%s total_us=%.1f",
                    next_layer_idx, rid[:8],
                    (time.perf_counter() - _t_sco_enter) * 1e6)

    def __score_one_request_impl(self, rid, rs, req_idx, next_layer_idx,
                                  quest_query, budget, stats,
                                  connector_meta):
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
        # 2026-05-28: prefer eff_groups_synced cache (set in on_step_start)
        # over per-call allreduce. Cache-hit branch fires symmetrically
        # across TP ranks; falls back to per-call when None.
        stored_groups = rs.stored_groups
        _cached_eff = getattr(rs, "eff_groups_synced", None)
        if _cached_eff is not None:
            effective_groups = int(_cached_eff)
        else:
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
        # 2026-05-28: skip when eff_groups_synced cache is populated
        # (both ranks see the same None-or-int state because the
        # populating loop in on_step_start iterates the symmetric
        # meta.new_chains source). The cache-hit branch fires
        # symmetrically → NCCL-safe; cache-miss falls through to
        # the legacy unconditional allreduce.
        if _cached_eff is None and self._tp_size > 1:
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
            _flushed_local_v = int(getattr(rs, "flushed_local", -1))
            _eff_stored_v = int(getattr(rs, "_effective_stored_groups", -1))
            logger.info(
                "[diag-chain] PRE rid=%s layer=%d chain_len=%d "
                "stored_groups=%d num_groups_written=%d "
                "flushed_local=%d _effective_stored_groups=%d "
                "effective_groups=%d client_total_pages=%d",
                rid, next_layer_idx, len(rs.chain),
                stored_groups, rs.num_groups_written,
                _flushed_local_v, _eff_stored_v,
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
        # 2026-05-19 THRESHOLD-FLIP DIAG: report why the threshold flip path
        # is/isn't firing each Score call. Useful when bitmap should be
        # saturating but flip never fires.
        if os.environ.get("ICMS_DIAG_BITMAP_GROWTH", "0") == "1" and is_decode:
            _cond_decode = is_decode
            _cond_threshold = len(already_fetched) >= _flip_threshold
            _cond_safe = _flip_safe
            _already_flipped = rs.dense_mode
            logger.info(
                "[diag-flip-thresh] rid=%s layer=%d "
                "len_fetched=%d total_pages=%d _flip_threshold=%d _flip_frac=%.3f "
                "decode=%s thresh_met=%s safe=%s already_flipped=%s "
                "WILL_FIRE=%s",
                rid, next_layer_idx,
                len(already_fetched), total_pages, _flip_threshold, _flip_frac,
                _cond_decode, _cond_threshold, _cond_safe, _already_flipped,
                (_cond_decode and _cond_threshold and _cond_safe
                 and not _already_flipped))
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

        # ICMS_SCORING_MODE=subset_max — per-layer per-Q-head subset_max
        # scoring (2026-05-19). For each scored layer, score with a small
        # set of Q heads (e.g. 1-2 per layer, calibrated offline) and
        # take per-page MAX across the subset, then top-K. Validated
        # offline on qwen3-30b mk1/mk2/mk3: catches needle in 80-100% of
        # examples at b=0.20 vs ~50-60% with baseline_sum32, at ~1/4 the
        # FLOPs. Requires ICMS_SUBSET_HEADS_JSON pointing to a layer→
        # subset mapping. See scripts/utils/subset_max_scoring.py.
        if os.environ.get("ICMS_SCORING_MODE", "") == "subset_max":
            if not getattr(self, "_subset_max_entry_logged", False):
                logger.info("[icms_subset_max] env detected; entering "
                             "subset_max path on layer=%d rid=%s",
                             next_layer_idx, rid)
                self._subset_max_entry_logged = True
            self._subset_max_score_one_layer(
                rid, rs, req_idx, next_layer_idx, quest_query, budget,
                total_pages, already_fetched)
            return

        # ICMS_SCORING_MODE=per_layer_max_kv — Quest-faithful client-side
        # scoring on per-head summaries (fetched via opcode 25/27, same as
        # subset_max). The K=16 random q-token sampling + per-token MAX +
        # per-KV-head MAX collapse produces the +50pp lift over baseline
        # sparse on qwen3-30b mk2/mk3 at h=32k/64k (validated by the HF
        # cleanroom impl at scripts/utils/quest_faithful_hf.py).
        #
        # MUST PRECEDE the per_kv_head branch below so that setting both
        # ICMS_SCORING_MODE=per_layer_max_kv AND ICMS_QUEST_MODE=per_kv_head
        # routes to the new mode (the per_kv_head branch otherwise wins).
        # ICMS_SCORING_MODE=faithful_quest — FAITHFUL Quest baseline
        # (per_kv_head selection + per-head exclusivity mask, matches
        # scripts/utils/quest_faithful_hf.py). Reuses the per_layer_max_kv
        # plumbing verbatim (summaries fetch → score → fetch_union → apply);
        # only the scoring call is swapped (per_layer_max_kv_per_head_topk →
        # union materialize + [H_kv,P] per-head mask) and the mask is stashed
        # for the Triton attention backend. TP=1 only (per-head selection is
        # NOT MAX-allreduce-composable).
        _scoring_mode = os.environ.get("ICMS_SCORING_MODE", "")
        if _scoring_mode in ("per_layer_max_kv", "faithful_quest"):
            self._per_layer_max_kv_score_one_layer(
                rid, rs, req_idx, next_layer_idx, quest_query, budget,
                total_pages, already_fetched,
                faithful=(_scoring_mode == "faithful_quest"))
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
        # 2026-05-19: in contiguous-reuse mode (dense_layers_mask != 0),
        # FetchAll's _pending_reuse population skips non-scored layers
        # (icms_connector.py:4101 _mask_set filter), so a dense L0/L1
        # redirect via ceiling-snap would leave the intervening reuse
        # layers (L3-L7 etc.) empty → wait_for_layer short-circuits →
        # gemma-SWA fallback on qwen3 → corrupted hidden states. Keep
        # dense scored layers on the Score path; its mask-driven
        # reuse_through (line 5390) populates every intervening layer.
        ceiling_snap = (effective_budget >= 1.0 and total_pages > 0
                         and not is_decode
                         and getattr(self, "_supports_fetch_all", False)
                         and self._geom.dense_layers_mask == 0)
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
        if os.environ.get("ICMS_DIAG_GEOM", "0") == "1" and not getattr(
                self, "_diag_geom_logged", False):
            self._diag_geom_logged = True
            logger.warning(
                "[diag-geom-entry] rank=%d tp=%d model=%s geom_is_None=%s "
                "geom.num_kv_heads=%s geom.num_layers=%s qq_type=%s qq_ndim=%s "
                "qq_shape=%s",
                self._tp_rank, self._tp_size, self._model_name,
                geom is None,
                (geom.num_kv_heads if geom is not None else "N/A"),
                (geom.num_layers if geom is not None else "N/A"),
                type(quest_query).__name__,
                (quest_query.ndim if isinstance(quest_query, torch.Tensor)
                 else "n/a"),
                (tuple(quest_query.shape)
                 if isinstance(quest_query, torch.Tensor) else "n/a"))
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
            # 2026-05-26: Q tail-slice for accuracy benches. When the bench
            # knows the question (+ answer_prefix) is the last N tokens of
            # the prompt and sets ICMS_Q_TAIL_TOKENS=N, slice quest_query
            # to its last N tokens BEFORE pooling. This solves the Q-mean
            # dilution problem on configs where vLLM's prefix cache does
            # not fully elide the haystack (e.g., gemma-3 with the bench's
            # chain-floor leaving a 256-token haystack-tail un-elided).
            # The slice fires only on prefill chunks whose Q includes the
            # question region (i.e., q.shape[0] > N tells us this is the
            # final chunk; smaller chunks earlier in prefill are pure
            # haystack and we let mean-pool run on them — Quest's score
            # accumulation across chunks is union-based so haystack
            # chunks' picks compose with the final chunk's picks).
            # NOTE: q.shape[0] <= N means we're INSIDE the question region
            # already (or the chunk is naturally smaller than the question);
            # no slicing needed in that case.
            if q.ndim == 3:
                try:
                    _q_tail = int(os.environ.get("ICMS_Q_TAIL_TOKENS", "0") or "0")
                except ValueError:
                    _q_tail = 0
                if _q_tail > 0 and q.shape[0] > _q_tail:
                    q = q[-_q_tail:]
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
                _pre_mean_shape = tuple(q.shape)
                if q.shape[0] == 1:
                    q = q.squeeze(0)
                elif _q_agg == "last":
                    q = q[-1]
                else:  # default 'mean'
                    q = q.mean(dim=0)
                # 2026-05-26: diagnostic — dump Q hash + norm per (rid, layer,
                # budget) so we can verify whether marker token changes the
                # mean-pooled Q across budget cycles. ALSO log pre-mean shape
                # to distinguish "differing num_tokens" from "differing values".
                if os.environ.get("ICMS_DIAG_Q_HASH") == "1":
                    try:
                        import hashlib as _hl
                        _q_b = q.detach().to(torch.float32).cpu().numpy().tobytes()
                        _q_h = _hl.sha1(_q_b).hexdigest()[:16]
                        _q_norm = float(q.float().norm().item())
                        _is_decode = bool(getattr(self, "_prefill_done", False))
                        logger.info("[diag-q] rid=%s layer=%d budget=%.3f "
                                    "pre_mean_shape=%s post_shape=%s decode=%s "
                                    "q_norm=%.6f q_sha=%s",
                                    rid, next_layer_idx,
                                    float(effective_budget),
                                    _pre_mean_shape, tuple(q.shape),
                                    _is_decode, _q_norm, _q_h)
                    except Exception as _qe:
                        logger.warning("[diag-q] failed: %s", _qe)
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
                    if os.environ.get("ICMS_DIAG_GEOM", "0") == "1":
                        logger.warning(
                            "[diag-geom] rank=%d tp=%d model=%s "
                            "geom.num_kv_heads(FULL?)=%d full_kv=%d "
                            "local_kv=%d q_heads=%d heads_per_group=%d "
                            "(EXPECT mistral full_kv=8 local_kv=4; "
                            "if full_kv=4 local_kv=2 => DOUBLE-DIVIDE BUG)",
                            self._tp_rank, self._tp_size, self._model_name,
                            geom.num_kv_heads, full_kv_heads, local_kv_heads,
                            num_heads, heads_per_group)
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
            # [INSTR-Q-CPU] GPU→CPU drain (also implicit cuda sync on
            # default stream). Suspected cost at high-ctx with many TP
            # callsites; isolate here so the next two milestones can
            # attribute SCORE-RPC vs Q-CPU separately.
            _instr_qcpu = os.environ.get("ICMS_INSTR", "0") == "1"
            _t_qcpu0 = time.perf_counter() if _instr_qcpu else 0.0
            q = q.reshape(-1).to(dtype=torch.float32, device="cpu")
            if _instr_qcpu:
                logger.info(
                    "[INSTR-Q-CPU] layer=%d q_drain_us=%.1f",
                    next_layer_idx,
                    (time.perf_counter() - _t_qcpu0) * 1e6)
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
        if (self._geom is not None
                and self._geom.dense_layers_mask != 0):
            # Contiguous-reuse: window extends up to the next scored layer
            # in the mask (exclusive), or to num_layers-1 if this is the
            # last scored layer. Populates _pending_reuse for every
            # intervening non-scored layer so they apply this page set.
            _nxt = self._geom.next_scored_layer_after(next_layer_idx)
            reuse_through = (_nxt - 1) if _nxt is not None else (num_layers - 1)
        else:
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
            # 2026-05-31 (ICMS_LOCAL_MASK_FROM_L1=1): also emit the
            # bitmap during prefill when worker_state.on_step_start
            # seeded `already_fetched` from `prov_local_cached_tokens`.
            # Lets BF2 mask the vLLM-L1-cached head pages to -INF so
            # the K picks land in the cold tail (bucket-B scenarios).
            _allow_prefill_bitmap = (
                os.environ.get("ICMS_LOCAL_MASK_FROM_L1", "0") == "1"
            )
            fetch_bitmap = (
                _pack_fetch_bitmap(already_fetched, total_pages)
                if ((is_decode or _allow_prefill_bitmap) and already_fetched)
                else b""
            )
            # 2026-05-19 BITMAP-WIRE DIAG: verify bitmap is actually being sent
            # to server (non-empty bytes + popcount == len(already_fetched)).
            if os.environ.get("ICMS_DIAG_BITMAP_GROWTH", "0") == "1":
                _bm_len = len(fetch_bitmap) if fetch_bitmap else 0
                _bm_popcount = sum(bin(b).count("1") for b in fetch_bitmap) if fetch_bitmap else 0
                logger.info(
                    "[diag-wire] rid=%s layer=%d is_decode=%s "
                    "already_fetched=%d total_pages=%d "
                    "bitmap_bytes=%d bitmap_popcount=%d k=%d",
                    rid, next_layer_idx, is_decode,
                    len(already_fetched), total_pages,
                    _bm_len, _bm_popcount, k)
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
                # Phase 1 matched-prefix (2026-06-03): when this rid was
                # routed through the matched-prefix bridge AND the bridge
                # deferred to per-layer Score (ICMS_MATCHED_PREFIX_BUDGET_AWARE=1),
                # set write_all_layers so the server bypasses its
                # scored_layers_mask filter — Phase-2 writes K/V for every
                # layer in [layer..reuse_through_layer], not just the scored
                # one. Required because matched-prefix elision leaves the
                # 40 non-scored layers (qwen3) with no other K/V source.
                _wall = bool(getattr(rs, "matched_prefix_request", False))
                if _wall and os.environ.get("ICMS_DIAG_MATCHED_PREFIX", "0") == "1":
                    logger.info(
                        "[diag-mp] score-rpc: rid=%s layer=%d k=%d "
                        "reuse_through=%d write_all_layers=1 "
                        "demand_bps=%d compute_supply_bps=%d",
                        rid[:8], next_layer_idx, k, reuse_through,
                        ab_demand_bps, ab_compute_supply_bps)
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
                    write_all_layers=_wall,
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
                        _instr_srpc = (os.environ.get("ICMS_INSTR", "0")
                                       == "1")
                        _t_srpc0 = (time.perf_counter()
                                    if _instr_srpc else 0.0)
                        _t_inlock = 0.0
                        # ASYNC SUBMIT (ICMS_ASYNC_SCORE=1, measure-time): once
                        # enabled the client multiplexes concurrent scores on
                        # one connection and demuxes by request_id, so we DROP
                        # the RTT-long _rpc_lock — the burst-serialization cork.
                        # Falls back to the locked sync path otherwise.
                        if self._maybe_enable_async_score():
                            # reply-early (use_flags) gives the score a two-phase
                            # completion (early reply + flag-based KV landing)
                            # that breaks the async mux's one-reply-per-
                            # request_id demux → score_get hangs the full
                            # timeout. Async already takes the score off the
                            # critical path, so the early-reply overlap is
                            # redundant. Force the single-reply (full-completion)
                            # path. Verified: REPLY_EARLY=0 async runs clean,
                            # REPLY_EARLY=1 async hangs 10s/score.
                            _score_kwargs["use_flags"] = False
                            _t_inlock = (time.perf_counter()
                                         if _instr_srpc else 0.0)
                            reply = self._client.score(**_score_kwargs)
                            _t_done = (time.perf_counter()
                                       if _instr_srpc else 0.0)
                        else:
                            with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                                # Capture lock-acquire boundary: _t_srpc0→here =
                                # LOCK-WAIT (serialization); here→after .score() =
                                # IN-LOCK (RTT + server queue + transport).
                                _t_inlock = (time.perf_counter()
                                             if _instr_srpc else 0.0)
                                reply = self._client.score(**_score_kwargs)
                                _t_done = (time.perf_counter()
                                           if _instr_srpc else 0.0)
                        if _instr_srpc:
                            _wire_us = (time.perf_counter() - _t_srpc0) * 1e6
                            _lock_wait_us = (_t_inlock - _t_srpc0) * 1e6
                            _in_lock_us = (_t_done - _t_inlock) * 1e6
                            _g = lambda f: (int(getattr(reply, f, 0)) // 1000
                                            if reply is not None else 0)
                            logger.info(
                                "[INSTR-SCORE-RPC] layer=%d wire_us=%.1f "
                                "lock_wait_us=%.1f in_lock_us=%.1f "
                                "srv_ready_us=%d srv_score_us=%d "
                                "srv_trie_us=%d srv_summary_us=%d "
                                "srv_sink_us=%d srv_concurrent=%d",
                                next_layer_idx, _wire_us,
                                _lock_wait_us, _in_lock_us,
                                _g("server_ingest_to_ready_ns"),
                                _g("score_ns"), _g("trie_walk_ns"),
                                _g("summary_read_ns"), _g("sink_write_ns"),
                                int(getattr(reply, "concurrent_requests", 0))
                                if reply is not None else 0)
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
                _instr_tpb = os.environ.get("ICMS_INSTR", "0") == "1"
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
                if _instr_tpb:
                    # [INSTR-TP-BCAST] Score-reply NCCL broadcast (rank-0
                    # → peers). Already fused 4→2 collectives earlier
                    # today; this confirms it's still cheap.
                    logger.info(
                        "[INSTR-TP-BCAST] layer=%d bcast_us=%.1f",
                        next_layer_idx, nccl_bcast_us)
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
                    # Per-score-call snapshot for offline analysis.
                    # Earlier version keyed by a forward-pass counter
                    # bumped at start_load_kv, but that callback's
                    # early-returns (worker=None, meta=None) skipped
                    # increments — so multiple score calls collapsed
                    # to the same key and overwrote each other.
                    # Switched to a per-call counter incremented here,
                    # guaranteed unique per scoring invocation, so
                    # prefill picks and decode picks always live in
                    # separate cells. Key = (call_idx, layer_idx) →
                    # also stash is_decode at record time so the
                    # analyzer can tell which calls fired pre- vs
                    # post-prefill_done flip.
                    if not hasattr(rs, "picked_call_log"):
                        rs.picked_call_log = []
                    _call_idx = len(rs.picked_call_log)
                    # Diagnostic: compute full per-page per-head Quest
                    # scores locally (using THIS call's Q + the
                    # currently-stored summaries). The server's reply
                    # only returns scores for the picked top-k, so to
                    # know the needle's rank we must reproduce the
                    # ranking. Identical math as the server-side AVX2
                    # kernel: score(p,h) = max(Q_h·kmin_p,h, Q_h·kmax_p,h).
                    # Stored as (P, H) fp32 tensor. ~32KB per call at
                    # h32k×4-heads. Gated to score-dump mode so the
                    # production hot path is unaffected.
                    _full_scores_ph = None
                    if os.environ.get("ICMS_DIAG_SCORE_DUMP"):
                        try:
                            _items_d = rs.quest_gpu_summaries.get(
                                next_layer_idx)
                            if (_items_d
                                    and isinstance(quest_query,
                                                    torch.Tensor)):
                                _items_d = sorted(_items_d,
                                                   key=lambda t: t[0])[
                                    :total_pages]
                                _kmin_t = torch.stack(
                                    [m for _, m, _ in _items_d], dim=0
                                ).float()  # (P, H, D)
                                _kmax_t = torch.stack(
                                    [m for _, _, m in _items_d], dim=0
                                ).float()
                                _q_t = quest_query.detach().cpu().float()
                                if _q_t.dim() == 3:
                                    # (tokens, H, D) → token-mean
                                    _q_t = _q_t.mean(dim=0)
                                if _q_t.dim() == 2:
                                    H_q = _q_t.shape[0]
                                    H_k = _kmin_t.shape[1]
                                    if H_q != H_k and H_q % H_k == 0:
                                        _q_t = _q_t.view(
                                            H_k, H_q // H_k, -1
                                            ).mean(dim=1)
                                    if _q_t.shape[0] == H_k:
                                        _s_lo = (_q_t.unsqueeze(0)
                                                  * _kmin_t).sum(dim=-1)
                                        _s_hi = (_q_t.unsqueeze(0)
                                                  * _kmax_t).sum(dim=-1)
                                        _full_scores_ph = (
                                            torch.maximum(_s_lo, _s_hi)
                                            .half())  # store fp16 to halve size
                        except Exception as _e_fsc:
                            logger.warning(
                                "[icms_diag_score_dump] full-scores "
                                "compute failed L=%d rid=%s: %s",
                                next_layer_idx, rid, _e_fsc)
                    rs.picked_call_log.append({
                        "call": _call_idx,
                        "layer": int(next_layer_idx),
                        "is_decode": bool(self._prefill_done),
                        "budget": float(effective_budget),
                        "k": int(k),
                        "total_pages": int(total_pages),
                        "picked": [int(p) for p in reply.page_ids],
                        "full_scores_per_head": _full_scores_ph,
                    })
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
            # 2026-05-19 BITMAP-GROWTH DIAG: per-Score-call log of how the
            # bitmap is growing. Helps diagnose why dense_mode flip doesn't
            # fire at default _flip_frac=1.0 for sparse-decode-with-budget.
            # Enable by setting ICMS_DIAG_BITMAP_GROWTH=1.
            if os.environ.get("ICMS_DIAG_BITMAP_GROWTH", "0") == "1":
                _reply_n = len(reply.page_ids) if reply.page_ids else 0
                _new_size = len(page_set)
                _delta = _new_size - prev_size
                _dups = _reply_n - _delta
                logger.info(
                    "[diag-bitmap] rid=%s layer=%d is_decode=%s "
                    "reply_n=%d new_pages=%d dups=%d "
                    "prev_size=%d new_size=%d total_pages=%d frac=%.3f",
                    rid, next_layer_idx, is_decode,
                    _reply_n, _delta, _dups,
                    prev_size, _new_size, total_pages,
                    (_new_size / total_pages if total_pages else 0.0))
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
            # 2026-05-19 M4 NEAR-MISS DIAG: log every Score call where M4
            # CONDITION (page_set == prev_size, no new pages) is met,
            # regardless of whether _m4_flip_enabled. Lets us see if
            # saturation IS detected but only gating is blocking the flip.
            if (os.environ.get("ICMS_DIAG_BITMAP_GROWTH", "0") == "1"
                    and is_decode and len(page_set) == prev_size):
                logger.info(
                    "[diag-m4-near] rid=%s layer=%d "
                    "len_fetched=%d total_pages=%d "
                    "m4_enabled=%s already_flipped=%s "
                    "(page_set == prev_size: bitmap saturated)",
                    rid, next_layer_idx,
                    len(page_set), total_pages,
                    _m4_flip_enabled, rs.dense_mode)
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
                # Phase 1 matched-prefix (2026-06-03): mirror
                # `force_apply_all_layers` from _fetch_all_one_request.
                # Server with kScoreFlagWriteAllLayers wrote K/V for
                # EVERY layer in [layer..reuse_through], not just scored.
                # The reuse loop must populate _pending_scores too so
                # wait_for_layer's apply fires for non-scored reuse
                # layers (otherwise on_layer_reuse → quest_hooks gate
                # leaves them unread → garbage attention for the matched
                # range). Safe: sink is sized for num_layers when
                # matched_prefix is in effect.
                _matched_prefix = bool(
                    getattr(rs, "matched_prefix_request", False))
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
                        if _matched_prefix:
                            import copy as _copy_mp
                            _promoted = _copy_mp.copy(reply)
                            _promoted.sink_offsets = reuse_offsets
                            self._assert_pending_scores_no_clobber(
                                reuse_attn, rid,
                                source="score-reply-matched-prefix-reuse")
                            self._pending_scores.setdefault(
                                reuse_attn, {})[rid] = (_promoted, req_idx)
                if (_matched_prefix
                        and os.environ.get("ICMS_DIAG_MATCHED_PREFIX", "0") == "1"):
                    logger.info(
                        "[diag-mp] score-reuse-populate: rid=%s scored_layer=%d "
                        "reuse_through=%d n_reuse_layers=%d k=%d "
                        "(matched-prefix branch: pending_scores+pending_reuse "
                        "set for all reuse layers)",
                        rid[:8], next_layer_idx, reuse_through,
                        reuse_through - next_layer_idx, actual_k)
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
    def _load_subset_heads_config(self):
        """Lazy-load the per-layer Q-head subset mapping for the
        ICMS_SCORING_MODE=subset_max path. Cached on self.

        File schema (JSON):
          {"0": [10, 15], "6": [12, 4], ...}
            or
          {"layers": {...}, "k_heads": 2, "model": "qwen3"}

        Layer indices not present in the mapping fall back to the
        default ICMS_SUBSET_DEFAULT_HEADS (single int q-head) or are
        skipped entirely (no scoring → dense fall-through).
        """
        cached = getattr(self, "_subset_heads_cfg", None)
        if cached is not None:
            return cached
        path = os.environ.get("ICMS_SUBSET_HEADS_JSON", "")
        cfg: dict[int, list[int]] = {}
        if path:
            try:
                import json as _json
                with open(path) as _f:
                    raw = _json.load(_f)
                if isinstance(raw, dict) and "layers" in raw:
                    raw = raw["layers"]
                if isinstance(raw, dict):
                    for k, v in raw.items():
                        try:
                            li = int(k)
                        except (TypeError, ValueError):
                            continue
                        if (isinstance(v, list)
                                and all(isinstance(x, int) for x in v)):
                            cfg[li] = list(v)
                if not cfg:
                    logger.warning(
                        "[icms_subset_max] %s parsed but produced empty"
                        " layer→subset mapping", path)
            except Exception as _e:
                logger.warning(
                    "[icms_subset_max] failed to load %s: %s",
                    path, _e)
        else:
            logger.warning(
                "[icms_subset_max] ICMS_SUBSET_HEADS_JSON not set; "
                "subset_max scoring will fall back to all-heads sum.")
        # Optional default for layers not in the mapping.
        try:
            default_h = os.environ.get("ICMS_SUBSET_DEFAULT_HEADS", "")
            if default_h:
                cfg.setdefault("__default__",  # type: ignore[arg-type]
                                [int(x) for x in default_h.split(",")])
        except Exception:
            pass
        self._subset_heads_cfg = cfg
        return cfg
    def _subset_max_score_one_layer(self, rid, rs, req_idx,
                                      next_layer_idx,
                                      quest_query, budget,
                                      total_pages, already_fetched):
        """ICMS_SCORING_MODE=subset_max path (2026-05-19, server-fetch
        version). Per-layer per-Q-head subset_max top-K.

        Mirrors _quest_per_kv_head_score_one_layer's data plane:
          1. Fetch per-page summaries (kmin, kmax) for THIS layer from
             the server via fetch_summaries_per_head — the chain on the
             server has them written by the cache rid's prefill.
          2. Reshape sink bytes → (kmin, kmax) tensors.
          3. Apply subset_max scoring: per-channel-max-summed Quest
             score per Q-head in the configured subset, take per-page
             MAX across the subset, top-K of that = picks.
          4. Fetch picked pages' KV via fetch_union_per_head — bytes
             land in the main KV sink at known offsets.
          5. Synthesize a ScoreReply with the picks + sink_offsets and
             route through _pending_scores so the existing apply path
             (_apply_selective_attention) trims attention to the picks.

        TP=1 only — same constraint as ICMS_ORIGINAL_QUEST and
        ICMS_QUEST_MODE=per_kv_head. unix-socket client only (the
        per-head opcodes don't exist on RDMA/shmem clients).

        Empirical (qwen3-30b, h32k, mk1/mk2/mk3, 2026-05-19 offline):
          * K_heads=1 per layer: ~75% needle pickup at b=0.20.
          * K_heads=2 per layer: ~85-100% pickup.
          * K_heads>=4 starts hurting (max() noise). Stay at 1-3.

        Configuration:
          ICMS_SCORING_MODE=subset_max
          ICMS_SUBSET_HEADS_JSON=path.json
          ICMS_SUBSET_DEFAULT_HEADS=10,15
        """
        if not isinstance(quest_query, torch.Tensor) or quest_query.ndim < 2:
            if not getattr(self, "_subset_max_qskip_logged", False):
                logger.warning("[icms_subset_max] skipping layer=%d: "
                                "quest_query not 2D Tensor (got %s)",
                                next_layer_idx, type(quest_query).__name__)
                self._subset_max_qskip_logged = True
            return
        if self._tp_size > 1:
            if not getattr(self, "_subset_max_tp_warned", False):
                logger.warning(
                    "[icms_subset_max] TP=1 only; got tp_size=%d. "
                    "Falling through.", self._tp_size)
                self._subset_max_tp_warned = True
            return
        if total_pages <= 0 or self._client is None or self._geom is None:
            return
        if not hasattr(self._client, "fetch_summaries_per_head"):
            if not getattr(self, "_subset_max_client_unsupported_warned",
                            False):
                logger.warning(
                    "[icms_subset_max] active client (%s) lacks "
                    "fetch_summaries_per_head; subset_max requires the "
                    "unix-socket IcmsClient. No-op.",
                    type(self._client).__name__)
                self._subset_max_client_unsupported_warned = True
            return

        cfg = self._load_subset_heads_config()
        subset_for_layer = cfg.get(int(next_layer_idx))
        if subset_for_layer is None:
            subset_for_layer = cfg.get("__default__")  # type: ignore[arg-type]
        if not subset_for_layer:
            if not hasattr(self, "_subset_max_missing_warned"):
                self._subset_max_missing_warned = set()
            if next_layer_idx not in self._subset_max_missing_warned:
                self._subset_max_missing_warned.add(next_layer_idx)
                logger.warning(
                    "[icms_subset_max] no subset configured for layer %d; "
                    "skipping scoring (dense fall-through).",
                    next_layer_idx)
            return

        sum_sink = self._ensure_summary_sink(total_pages)
        if sum_sink is None:
            return
        if not getattr(self, "_subset_max_first_call_logged", False):
            logger.info(
                "[icms_subset_max] FIRST CALL — subset_max active "
                "(rid=%s layer=%d total_pages=%d budget=%.3f subset=%s)",
                rid[:8], next_layer_idx, total_pages, float(budget),
                subset_for_layer)
            self._subset_max_first_call_logged = True

        chain = self._rank_chain(rs.chain)
        icms_rid = self._icms_request_id(rid, 0)
        t0 = time.perf_counter()

        # ── 1. Fetch per-page summaries for THIS layer from server ──
        try:
            with self._rpc_lock:
                sum_reply = self._client.fetch_summaries_per_head(
                    request_id=icms_rid, chain=chain,
                    layer=next_layer_idx,
                    sink=sum_sink, reuse_through_layer=next_layer_idx)
        except IcmsError as e:
            logger.warning(
                "[icms_subset_max] FetchSummariesPerHead failed at "
                "layer %d (rid=%s): %r",
                next_layer_idx, rid[:8], e)
            return

        P = len(sum_reply.page_ids)
        if P == 0:
            return
        # Pin the page_id ordering invariant: server MUST return
        # page_ids = [0, 1, ..., P-1] so scorer-returned indices (row
        # indices into kmin/kmax) coincide with real page IDs we pass
        # back to fetch_union_per_head. Same invariant the per-kv-head
        # path pins (see :6947-6953).
        if (sum_reply.page_ids[0] != 0 or sum_reply.page_ids[-1] != P - 1
                or len(sum_reply.page_ids) != P):
            logger.warning(
                "[icms_subset_max] sum_reply.page_ids violates [0..P-1] "
                "contract at layer %d (P=%d, head=%s, tail=%s); aborting "
                "to avoid corrupted KV reads.",
                next_layer_idx, P, sum_reply.page_ids[:4],
                sum_reply.page_ids[-4:])
            return

        # ── 2. Reshape sink bytes → (kmin, kmax) on the GPU ──
        # Layout: kmin.tobytes() ++ kmax.tobytes(), each [H_kv, D] fp16,
        # page p at offset slot_base + p * spb. Mirror lines 6975-7035
        # of the per-kv-head path.
        H_kv = int(self._geom.num_kv_heads)
        D = int(self._geom.head_dim)
        elem = int(self._geom.elem_bytes)
        spb = int(self._geom.summary_page_bytes)
        if elem != 2:
            logger.warning(
                "[icms_subset_max] elem_bytes=%d not supported in v1 "
                "(fp16 only); no-op.", elem)
            return
        slot_base = int(sum_reply.sink_offsets[0])
        total_bytes = P * spb
        contiguous = all(
            int(sum_reply.sink_offsets[i]) == slot_base + i * spb
            for i in range(P))
        device = quest_query.device if quest_query.is_cuda else "cuda"
        if getattr(sum_sink, "is_gpu_direct", False):
            gpu_tensor = getattr(sum_sink, "gpu_tensor", None)
            if gpu_tensor is None:
                logger.warning(
                    "[icms_subset_max] GPU-IPC summary sink missing "
                    "gpu_tensor; aborting layer %d", next_layer_idx)
                return
            if contiguous:
                flat_u8 = gpu_tensor[
                    slot_base:slot_base + total_bytes]
            else:
                flat_u8 = torch.empty(total_bytes, dtype=torch.uint8,
                                      device=gpu_tensor.device)
                for i, off in enumerate(sum_reply.sink_offsets):
                    flat_u8[i * spb:(i + 1) * spb] = \
                        gpu_tensor[off:off + spb]
            flat_gpu = flat_u8.view(torch.float16).reshape(
                P, 2, H_kv, D)
            if flat_gpu.device != torch.device(device):
                flat_gpu = flat_gpu.to(device=device, non_blocking=True)
        else:
            view = sum_sink.view()
            if contiguous:
                host_bytes = bytes(view[slot_base:slot_base + total_bytes])
            else:
                host_buf = bytearray(total_bytes)
                for i, off in enumerate(sum_reply.sink_offsets):
                    host_buf[i * spb:(i + 1) * spb] = bytes(
                        view[off:off + spb])
                host_bytes = bytes(host_buf)
            flat_cpu = torch.frombuffer(host_bytes, dtype=torch.float16,
                                          count=P * 2 * H_kv * D).reshape(
                P, 2, H_kv, D)
            flat_gpu = flat_cpu.to(device=device, non_blocking=True)
        kmin = flat_gpu[:, 0, :, :].contiguous()
        kmax = flat_gpu[:, 1, :, :].contiguous()

        # ── 3. Subset_max scoring on GPU ──
        # Q pooling — token-mean across the prefill chunk's tokens.
        q = quest_query.detach()
        if q.ndim == 3:
            q = q.mean(dim=0)
        if q.ndim != 2:
            return
        q_f = q.to(torch.float32).to(device)
        kmin_f = kmin.to(torch.float32)
        kmax_f = kmax.to(torch.float32)
        H_q = q_f.shape[0]
        if H_q % H_kv != 0:
            logger.warning(
                "[icms_subset_max] GQA mismatch H_q=%d H_kv=%d; skip.",
                H_q, H_kv)
            return
        group = H_q // H_kv

        # Filter subset to valid head indices.
        subset_valid = [int(h) for h in subset_for_layer
                         if 0 <= int(h) < H_q]
        if not subset_valid:
            return

        # subset_max: score per Q head in subset, take per-page MAX,
        # then top-K of that. Picks live in chain-page space (0..P-1).
        k = max(1, int(P * float(budget)))
        picked: list[int] = []
        try:
            # Quest's upper-bound formula is PER-CHANNEL max, summed
            # across D — see scripts/utils/subset_max_scoring.py for the
            # discussion. Matches storage_service/src/scoring/
            # quest_kernel.h exactly.
            kmin_exp = kmin_f.repeat_interleave(group, dim=1)
            kmax_exp = kmax_f.repeat_interleave(group, dim=1)
            q_b = q_f.unsqueeze(0)
            per_channel_max = torch.maximum(q_b * kmin_exp,
                                              q_b * kmax_exp)
            per_qh = per_channel_max.sum(dim=-1)  # (P, H_q)
            sub_idx = torch.as_tensor(subset_valid, dtype=torch.long,
                                        device=per_qh.device)
            sub_scores = per_qh.index_select(1, sub_idx)
            page_scores = sub_scores.max(dim=-1).values
            # Exclude already-fetched pages by setting their score to
            # -inf so top-K skips them.
            if already_fetched:
                for ap in already_fetched:
                    if 0 <= ap < page_scores.numel():
                        page_scores[ap] = float("-inf")
            P = page_scores.numel()
            k = min(k, P)
            # vAttention hybrid: split budget between top-K (top_idx) and
            # uniform random tail. ICMS_VATTN_ALPHA=1.0 (default) = pure
            # subset_max top-K. ICMS_VATTN_ALPHA=0.5 = 50% top-K + 50% random.
            # ICMS_VATTN_RANDOM_MODE=per_layer (default) freshly samples
            # the random tail at each scored layer; "per_rid" picks once
            # per request and reuses across layers.
            _alpha = float(os.environ.get("ICMS_VATTN_ALPHA", "1.0"))
            _rand_mode = os.environ.get("ICMS_VATTN_RANDOM_MODE", "per_layer")
            _alpha = max(0.0, min(1.0, _alpha))
            if _alpha >= 1.0 - 1e-6:
                k_topk = k
                k_sample = 0
            else:
                k_topk = max(1, int(round(k * _alpha))) if _alpha > 0 else 0
                k_sample = k - k_topk
            top_idx = torch.topk(page_scores, k_topk).indices.tolist() if k_topk > 0 else []
            picked_topk = [int(i) for i in top_idx
                            if page_scores[i].item() != float("-inf")]
            # Fix (a) k_topk under-fill: if -inf masking shrank picked_topk
            # below the requested k_topk, absorb the deficit into k_sample
            # so total picks stay near k.
            _deficit = k_topk - len(picked_topk)
            if _deficit > 0 and _alpha < 1.0 - 1e-6:
                k_sample = min(k_sample + _deficit, P - len(picked_topk))
            picked_sample: list[int] = []
            if k_sample > 0:
                import random as _random
                # already_excluded = already_fetched (downstream skip)
                # + top-K picks (avoid dup)
                _exclude = set(picked_topk)
                if already_fetched:
                    _exclude |= set(int(p) for p in already_fetched)
                _cache_attr = "_vattn_random_sample"
                # Fix (b) per-rid seeded RNG: decouple our sample from
                # vLLM's set_random_seed(0). For per_rid: seed by rid so all
                # layers in this request share a stream. For per_layer: seed
                # by (rid, layer) so each layer gets a different but
                # reproducible sample.
                if _rand_mode == "per_rid":
                    _rng = _random.Random(hash(rid))
                else:
                    _rng = _random.Random(hash((rid, int(next_layer_idx))))
                if _rand_mode == "per_rid":
                    cached = getattr(rs, _cache_attr, None)
                    if cached is None:
                        _cands = [p for p in range(P) if p not in _exclude]
                        picked_sample = _rng.sample(_cands,
                                                     min(k_sample, len(_cands)))
                        setattr(rs, _cache_attr, list(picked_sample))
                    else:
                        # Re-use cached sample; drop any in current exclude set
                        picked_sample = [p for p in cached if p not in _exclude]
                        if len(picked_sample) < k_sample:
                            _need = k_sample - len(picked_sample)
                            _more = [p for p in range(P)
                                     if p not in _exclude
                                     and p not in set(picked_sample)]
                            picked_sample += _rng.sample(_more,
                                                          min(_need, len(_more)))
                        # Fix (c): persist the augmented sample so next
                        # layer's cache reflects what we actually used.
                        setattr(rs, _cache_attr, list(picked_sample))
                else:  # per_layer
                    _cands = [p for p in range(P) if p not in _exclude]
                    picked_sample = _rng.sample(_cands,
                                                 min(k_sample, len(_cands)))
            picked = picked_topk + [int(p) for p in picked_sample]
            if (_alpha < 1.0 - 1e-6
                    and not getattr(self, "_vattn_first_logged", False)):
                logger.info(
                    "[icms_vattn] FIRST CALL — alpha=%.2f mode=%s "
                    "k=%d k_topk=%d k_sample=%d rid=%s layer=%d",
                    _alpha, _rand_mode, k, k_topk, k_sample, rid[:8],
                    next_layer_idx)
                self._vattn_first_logged = True
        except Exception as e:
            logger.warning("[icms_subset_max] scoring failed at "
                            "layer %d rid=%s: %s",
                            next_layer_idx, rid, e)

        if not picked:
            if not getattr(self, "_subset_max_empty_picks_logged", False):
                logger.info(
                    "[icms_subset_max] layer %d picks empty rid=%s — "
                    "scoring fired but selected 0 pages", next_layer_idx,
                    rid[:8])
                self._subset_max_empty_picks_logged = True
            return

        if not getattr(self, "_subset_max_picks_logged_layers", set()):
            self._subset_max_picks_logged_layers = set()
        if next_layer_idx not in self._subset_max_picks_logged_layers:
            self._subset_max_picks_logged_layers.add(next_layer_idx)
            logger.info(
                "[icms_subset_max] layer %d rid=%s picked %d pages, "
                "subset=%s, first10=%s",
                next_layer_idx, rid[:8], len(picked), subset_valid,
                picked[:10])

        if os.environ.get("ICMS_DIAG_SCORE_DUMP", "") and int(self._tp_rank) == 0:
            try:
                rs.last_picked_by_layer[int(next_layer_idx)] = list(picked)
                if int(next_layer_idx) not in rs.last_q_by_layer:
                    rs.last_q_by_layer[int(next_layer_idx)] = torch.zeros(
                        1, dtype=torch.uint8)
                if not hasattr(rs, "picked_call_log"):
                    rs.picked_call_log = []
                rs.picked_call_log.append({
                    "call": len(rs.picked_call_log),
                    "layer": int(next_layer_idx),
                    "is_decode": bool(self._prefill_done),
                    "budget": float(budget),
                    "k": int(k),
                    "total_pages": int(total_pages),
                    "picked": list(picked),
                    "full_scores_per_head": None,
                    "scored_heads_subset": list(subset_valid),
                })
            except Exception as _e_stash:
                logger.warning(
                    "[icms_subset_max] dump stash failed at layer %d "
                    "rid=%s: %s", next_layer_idx, rid, _e_stash)

        # === APPLY PATH WIRING (2026-05-19 fix per audit) ===
        # Computing picks alone isn't enough — the attention kernel reads
        # picked KV bytes from a contiguous "sink" buffer via offsets
        # carried in _pending_scores[layer][rid]. Without populating
        # _pending_scores with valid sink_offsets, the apply path
        # (_apply_selective_attention) never runs and the attention
        # kernel falls through to the natural (full) block_table —
        # under ICMS_BENCH_PREFIX_CACHE=0 the haystack pages aren't
        # all written yet → garbled attention → gibberish output.
        #
        # Fix: mirror _quest_per_kv_head_score_one_layer's pattern.
        # Fire fetch_union_per_head to ask the server to scatter the
        # picked pages' KV into our sink, then synthesize a ScoreReply
        # and route through the existing _pending_scores plumbing.
        if (self._sink_pool is None
                or not hasattr(self._client, "fetch_union_per_head")):
            if not getattr(self, "_subset_max_apply_unsupported_warned",
                            False):
                logger.warning(
                    "[icms_subset_max] client lacks fetch_union_per_head "
                    "or sink_pool missing; apply path cannot be wired.")
                self._subset_max_apply_unsupported_warned = True
            return

        # Fetch THIS layer + the next stride-1 layers' KV in one RPC, so
        # the unscored layers in the stride window have their KV in the
        # sink at known offsets. Without this, only the scored layer
        # gets KV; downstream apply for unscored layers reads stale
        # offsets → garbled attention. Mirrors the BF2 reuse_through
        # mechanism (line ~4100).
        num_layers = int(self._geom.num_layers)
        if self._geom.dense_layers_mask != 0:
            _nxt = self._geom.next_scored_layer_after(next_layer_idx)
            reuse_through = (_nxt - 1) if _nxt is not None else (num_layers - 1)
        else:
            reuse_through = min(next_layer_idx + self._score_stride - 1,
                                  num_layers - 1)
        try:
            with self._rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                union_reply = self._client.fetch_union_per_head(
                    request_id=icms_rid, chain=chain,
                    layer=next_layer_idx,
                    sink=self._sink_pool.sink, page_ids=picked,
                    reuse_through_layer=reuse_through)
        except IcmsError as e:
            logger.warning(
                "[icms_subset_max] FetchUnionPerHead failed at "
                "layer %d (rid=%s, k=%d, reuse_through=%d): %r",
                next_layer_idx, rid[:8], len(picked), reuse_through, e)
            return
        except Exception as e:
            logger.warning(
                "[icms_subset_max] FetchUnionPerHead unexpected at "
                "layer %d (rid=%s, k=%d): %r",
                next_layer_idx, rid[:8], len(picked), e)
            return

        # Synthesize a ScoreReply to drop into _pending_scores. The
        # apply path doesn't care that we computed picks locally — it
        # only needs (page_ids, sink_offsets, status). Scores are
        # decorative (used for perf-sweep histograms only).
        try:
            from icms_client.protocol import ScoreReply
            synth = ScoreReply(
                request_id=icms_rid,
                status=0,
                trie_walk_ns=0,
                summary_read_ns=0,
                score_ns=int((time.perf_counter() - t0) * 1e9),
                sink_write_ns=int(getattr(union_reply,
                                           "sink_write_ns", 0)),
                cache_hit=False,
                concurrent_requests=int(getattr(
                    union_reply, "concurrent_requests", 0)),
                server_ingest_to_ready_ns=int(getattr(
                    union_reply, "server_ingest_to_ready_ns", 0)),
                effective_supply_bps=int(getattr(
                    union_reply, "effective_supply_bps", 0)),
                page_ids=list(picked),
                scores=[0.0] * len(picked),
                sink_offsets=list(union_reply.sink_offsets),
            )
        except Exception as e:
            logger.warning(
                "[icms_subset_max] synth ScoreReply failed at "
                "layer %d: %s", next_layer_idx, e)
            return

        attn_layer_name = f"model.layers.{next_layer_idx}.self_attn.attn"
        with self._score_lock:
            self._assert_pending_scores_no_clobber(
                attn_layer_name, rid, source="subset_max-score-landing")
            self._pending_scores.setdefault(
                attn_layer_name, {})[rid] = (synth, req_idx)

        # Suspect #2 probe (audit a6818fa, 2026-05-20): is the apply path
        # silently filtering picks via `pid >= context_pages` because
        # rs.stored_groups is 0 at write time? context_pages = stored_groups*32.
        if not getattr(self, "_subset_max_ctx_probe_logged", False):
            try:
                _sg = int(getattr(rs, "stored_groups", 0))
                _ngw = int(getattr(rs, "num_groups_written", 0))
                _ctx = max(_sg, _ngw) * 32
                _above = sum(1 for p in picked if p >= _ctx) if _ctx > 0 else len(picked)
                logger.info(
                    "[icms_subset_max] CTX-PROBE rid=%s layer=%d P=%d k=%d "
                    "stored_groups=%d ngw=%d ctx_pages=%d picks_above_ctx=%d "
                    "picks_minmax=(%d,%d)",
                    rid[:8], next_layer_idx, P, k, _sg, _ngw, _ctx, _above,
                    min(picked), max(picked))
                self._subset_max_ctx_probe_logged = True
            except Exception:
                pass

        # Populate _pending_reuse for the stride/contig-reuse window.
        # on_layer_reuse at each unscored layer pops the entry and promotes
        # it to _pending_scores with the adjusted sink offsets. Unlike the
        # FetchAll path (which packs scored-only in the sink, hence
        # scored_rank delta), fetch_union_per_head packs ALL layers in
        # [layer..reuse_through_layer] sequentially — see
        # handlers_per_head.cc handle_fetch_union_per_head's L-loop — so
        # we use abs-layer delta and populate EVERY layer in the window
        # (including non-scored gap layers in shifted mode, which need
        # reuse entries so on_layer_reuse promotes them).
        actual_k = len(picked)
        per_layer_bytes = actual_k * int(self._geom.kv_page_bytes)
        for delta in range(1, reuse_through - next_layer_idx + 1):
            reuse_layer = next_layer_idx + delta
            reuse_attn = f"model.layers.{reuse_layer}.self_attn.attn"
            reuse_offsets = [off + delta * per_layer_bytes
                              for off in union_reply.sink_offsets]
            with self._score_lock:
                self._pending_reuse.setdefault(reuse_attn, {})[rid] = (
                    synth, reuse_offsets, req_idx)

        wall_us = (time.perf_counter() - t0) * 1e6
        rs.fetched_pages.setdefault(next_layer_idx, set()).update(picked)
        self.stats.record_score(wall_us, bool(picked), picked,
                                  next_layer_idx)
        # 2026-05-20: removed the level==0 force-bump band-aid here —
        # record_score now increments total_score_calls unconditionally,
        # so this would double-count in the subset_max path.
        logger.debug(
            "[icms_subset_max] layer=%d total_pages=%d k=%d "
            "subset=%s n_picked=%d sink_offsets=%d wall_us=%.1f",
            next_layer_idx, total_pages, k, subset_valid, len(picked),
            len(union_reply.sink_offsets), wall_us)
    def _get_q_token_sample_generator(self, device):
        """Lazy-init the per-worker torch.Generator for q-token sampling.

        Generator is bound to a single device (host vs cuda); recreated +
        re-seeded if a later call presents a different device. The seed
        is read from ``ICMS_PER_LAYER_SEED_BASE`` at ``__init__`` time and
        cached on ``self._q_token_sample_seed`` — runtime env changes are
        ignored to preserve "single process → reproducible trajectory".
        """
        g = self._q_token_sample_generator
        if g is None or g.device != device:
            g = torch.Generator(device=device)
            g.manual_seed(int(self._q_token_sample_seed))
            self._q_token_sample_generator = g
        return g
    def _per_layer_max_kv_score_one_layer(self, rid, rs, req_idx,
                                            next_layer_idx,
                                            quest_query, budget,
                                            total_pages, already_fetched,
                                            faithful=False):
        """ICMS_SCORING_MODE=per_layer_max_kv path (2026-05-28).

        When ``faithful=True`` (ICMS_SCORING_MODE=faithful_quest) the scoring
        call is swapped to per-KV-head distinct selection
        (``per_layer_max_kv_per_head_topk``): the materialized page-set becomes
        the UNION of per-head picks (fed to the unchanged ``fetch_union_per_head``)
        and the per-head ``[H_kv, P]`` selection mask is stashed in
        ``_pending_faithful_masks`` for the Triton attention backend to enforce
        per-head exclusivity. Faithful mode is TP=1 only.

        Quest-faithful client-side scoring on per-head summaries with
        optional K random q-token sampling. Mirrors
        ``_subset_max_score_one_layer``'s plumbing verbatim; only the
        scoring math is swapped. See
        ``scripts/utils/per_layer_max_kv_scoring.py`` for the math and
        ``scripts/utils/quest_faithful_hf.py`` for the equivalence
        reference (asserted in
        ``tests/test_per_layer_max_kv_scoring.py``).

        Algorithm (paper config: qtok_pool=max, qhd_pool=mean, K=16
        random, kv_reduce=max):
          1. Fetch per-page (kmin, kmax) summaries for THIS layer.
          2. Optionally sub-sample K q-token positions per call.
          3. Per-(page, KV head) Quest score; collapse across KV via MAX.
          4. Top-B pages once at layer level; broadcast to all KV heads.
          5. fetch_union_per_head + _pending_scores plumbing (subset_max
             pattern).

        TP=1 only — server is not rank-aware in v1, so we hard-fail at
        TP>1 instead of falling through to legacy. Synthesis review
        flagged the legacy fallback as a silent correctness hazard
        (per_layer_max_kv with no allreduce diverges across ranks).

        Configuration:
          ICMS_SCORING_MODE=per_layer_max_kv (REQUIRED to enter this path)
          ICMS_Q_TOKEN_POOL=max|mean              (default: "max")
          ICMS_Q_HEAD_POOL=mean|max               (default: "mean")
          ICMS_Q_TOKEN_SAMPLE_COUNT=int           (default: 0 = no sample)
          ICMS_Q_TOKEN_SAMPLE_MODE=none|strided|random  (default: "random")
          ICMS_PER_LAYER_SEED_BASE=int            (default: 0; PROCESS-WIDE)
          ICMS_PER_LAYER_MAX_KV_REDUCE=max|sum|mean     (default: "max")
        """
        # ── 0. Guards (mirror subset_max) ──────────────────────────────
        if not isinstance(quest_query, torch.Tensor) or quest_query.ndim < 2:
            if not self._per_layer_max_kv_q_len_warned:
                logger.warning("[icms_per_layer_max_kv] skipping layer=%d: "
                                "quest_query not 2/3-D Tensor (got %s)",
                                next_layer_idx, type(quest_query).__name__)
                self._per_layer_max_kv_q_len_warned = True
            return
        if faithful and self._tp_size > 1:
            raise RuntimeError(
                "[faithful_quest] per-KV-head distinct selection is NOT "
                "MAX-allreduce-composable across ranks (unlike per_layer_max_kv); "
                "faithful_quest is TP=1 only. Got tp_size=%d — run the baseline "
                "at TP=1, or implement the per-rank per-head all-gather first."
                % self._tp_size)
        if self._tp_size > 1 and not self._per_layer_max_kv_tp_warned:
            logger.info(
                "[icms_per_layer_max_kv] TP=%d: cross-rank allreduce(MAX) of "
                "per-page scores ENABLED (accuracy-path TP support; one extra "
                "NCCL all-reduce per scored layer, not perf-optimized). All "
                "ranks score rank-local KV heads, MAX-combine, then top-k → "
                "identical picks.",
                self._tp_size)
            self._per_layer_max_kv_tp_warned = True
        # Hard-assert no adaptive allocator: per_layer_max_kv interacts
        # badly with per-layer budget changes (R1 from synthesis review)
        # — picked sets become non-monotonic across layers in the same
        # request → cumulative already_fetched contains pages that
        # wouldn't be picked at the higher budget. Defer adaptive support
        # to a follow-up PR; loud-fail here so nobody silently combines
        # the two in production.
        if self._adaptive_allocator is not None:
            raise RuntimeError(
                "[icms_per_layer_max_kv] _adaptive_allocator is not None: "
                "per_layer_max_kv is incompatible with adaptive budgeting "
                "in v1. Unset the adaptive flag or pin a static budget.")
        if total_pages <= 0 or self._client is None or self._geom is None:
            return
        if not hasattr(self._client, "fetch_summaries_per_head"):
            if not self._per_layer_max_kv_client_unsupported_warned:
                logger.warning(
                    "[icms_per_layer_max_kv] active client (%s) lacks "
                    "fetch_summaries_per_head; per_layer_max_kv requires "
                    "the unix-socket IcmsClient. No-op.",
                    type(self._client).__name__)
                self._per_layer_max_kv_client_unsupported_warned = True
            return

        sum_sink = self._ensure_summary_sink(total_pages)
        if sum_sink is None:
            return

        # Env knobs (read each call so tests can flip them per-fixture;
        # the seed is NOT re-read — see _get_q_token_sample_generator).
        qtok_pool = os.environ.get("ICMS_Q_TOKEN_POOL", "max")
        # Faithful per_kv_head uses per-Q-head scoring (q_head_pool='max', the
        # HF install_quest default); per_layer_max_kv defaults to 'mean'.
        qhd_pool = os.environ.get(
            "ICMS_Q_HEAD_POOL", "max" if faithful else "mean")
        k_sample = int(os.environ.get("ICMS_Q_TOKEN_SAMPLE_COUNT", "0"))
        sample_mode = os.environ.get("ICMS_Q_TOKEN_SAMPLE_MODE", "random")
        kv_reduce = os.environ.get("ICMS_PER_LAYER_MAX_KV_REDUCE", "max")

        if not self._per_layer_max_kv_first_call_logged:
            logger.info(
                "[icms_per_layer_max_kv] FIRST CALL — env active "
                "(rid=%s layer=%d total_pages=%d budget=%.3f "
                "qtok_pool=%s qhd_pool=%s K=%d sample_mode=%s "
                "kv_reduce=%s seed_base=%d q.shape=%s)",
                rid[:8], next_layer_idx, total_pages, float(budget),
                qtok_pool, qhd_pool, k_sample, sample_mode, kv_reduce,
                int(self._q_token_sample_seed), tuple(quest_query.shape))
            self._per_layer_max_kv_first_call_logged = True

        chain = self._rank_chain(rs.chain)
        icms_rid = self._icms_request_id(rid, 0)
        t0 = time.perf_counter()

        # ── 1. Fetch per-page summaries for THIS layer from server ──
        try:
            with self._rpc_lock:
                sum_reply = self._client.fetch_summaries_per_head(
                    request_id=icms_rid, chain=chain,
                    layer=next_layer_idx,
                    sink=sum_sink, reuse_through_layer=next_layer_idx)
        except IcmsError as e:
            logger.warning(
                "[icms_per_layer_max_kv] FetchSummariesPerHead failed at "
                "layer %d (rid=%s): %r",
                next_layer_idx, rid[:8], e)
            return

        P = len(sum_reply.page_ids)
        if P == 0:
            return
        if (sum_reply.page_ids[0] != 0
                or sum_reply.page_ids[-1] != P - 1
                or len(sum_reply.page_ids) != P):
            logger.warning(
                "[icms_per_layer_max_kv] sum_reply.page_ids violates "
                "[0..P-1] contract at layer %d (P=%d, head=%s, tail=%s); "
                "aborting to avoid corrupted KV reads.",
                next_layer_idx, P, sum_reply.page_ids[:4],
                sum_reply.page_ids[-4:])
            return

        # ── 2. Reshape sink bytes → (kmin, kmax) on the GPU ──
        H_kv = int(self._geom.num_kv_heads)
        D = int(self._geom.head_dim)
        elem = int(self._geom.elem_bytes)
        spb = int(self._geom.summary_page_bytes)
        if os.environ.get("ICMS_DIAG_GEOM", "0") == "1" and not getattr(
                self, "_diag_plmk_logged", False):
            self._diag_plmk_logged = True
            _qqs = (tuple(quest_query.shape)
                    if isinstance(quest_query, torch.Tensor) else "n/a")
            logger.warning(
                "[diag-plmk-geom] rank=%d tp=%d model=%s "
                "H_kv(geom.num_kv_heads)=%d D=%d spb=%d elem=%d "
                "total_pages=%d quest_query.shape=%s "
                "(EXPECT mistral H_kv=8; H_kv=4 => geom WRONG => "
                "summary reshape+rank-slice both broken)",
                self._tp_rank, self._tp_size, self._model_name,
                H_kv, D, spb, elem, total_pages, _qqs)
        if elem != 2:
            logger.warning(
                "[icms_per_layer_max_kv] elem_bytes=%d not supported in "
                "v1 (fp16 only); no-op.", elem)
            return
        slot_base = int(sum_reply.sink_offsets[0])
        total_bytes = P * spb
        contiguous = all(
            int(sum_reply.sink_offsets[i]) == slot_base + i * spb
            for i in range(P))
        device = quest_query.device if quest_query.is_cuda else "cuda"
        if getattr(sum_sink, "is_gpu_direct", False):
            gpu_tensor = getattr(sum_sink, "gpu_tensor", None)
            if gpu_tensor is None:
                logger.warning(
                    "[icms_per_layer_max_kv] GPU-IPC summary sink missing "
                    "gpu_tensor; aborting layer %d", next_layer_idx)
                return
            if contiguous:
                flat_u8 = gpu_tensor[
                    slot_base:slot_base + total_bytes]
            else:
                flat_u8 = torch.empty(total_bytes, dtype=torch.uint8,
                                      device=gpu_tensor.device)
                for i, off in enumerate(sum_reply.sink_offsets):
                    flat_u8[i * spb:(i + 1) * spb] = \
                        gpu_tensor[off:off + spb]
            flat_gpu = flat_u8.view(torch.float16).reshape(
                P, 2, H_kv, D)
            if flat_gpu.device != torch.device(device):
                flat_gpu = flat_gpu.to(device=device, non_blocking=True)
        else:
            view = sum_sink.view()
            if contiguous:
                host_bytes = bytes(view[slot_base:slot_base + total_bytes])
            else:
                host_buf = bytearray(total_bytes)
                for i, off in enumerate(sum_reply.sink_offsets):
                    host_buf[i * spb:(i + 1) * spb] = bytes(
                        view[off:off + spb])
                host_bytes = bytes(host_buf)
            flat_cpu = torch.frombuffer(host_bytes, dtype=torch.float16,
                                          count=P * 2 * H_kv * D).reshape(
                P, 2, H_kv, D)
            flat_gpu = flat_cpu.to(device=device, non_blocking=True)
        kmin = flat_gpu[:, 0, :, :].contiguous()
        kmax = flat_gpu[:, 1, :, :].contiguous()

        # ── per_layer_max_kv TP>1 KV-head slice (accuracy-path fix 2026-05-29) ──
        # The server is NOT rank-aware: the summary fetch returns ALL H_kv KV
        # heads to every rank ([P, H_kv, D] with H_kv = full num_kv_heads). But
        # the captured scoring Q is rank-LOCAL (H_q/tp_size heads). If we score
        # the rank-local Q against the full-H_kv summaries, the scorer computes
        # GQA group = H_q_local // H_kv_full (e.g. 16//4=4) instead of the true
        # 8, mis-pairing each rank's Q heads across ALL kv heads → wrong page
        # scores (verified: page_scores diverge by ~110, top-403 overlap 215/403
        # vs TP=1; slicing → bit-identical, 403/403). So restrict the summaries
        # to this rank's local KV heads; rank r owns kv[r*Hl:(r+1)*Hl]. After the
        # slice group = H_q_local // H_kv_local = 8 (correct) and each rank scores
        # its local Q↔local KV; the downstream allreduce(MAX) over the [P] page
        # scores then MAX-combines ranks → global max over all kv heads = TP=1.
        if self._tp_size > 1 and H_kv % self._tp_size == 0:
            _hkv_local = H_kv // self._tp_size
            _lo = int(self._tp_rank) * _hkv_local
            kmin = kmin[:, _lo:_lo + _hkv_local, :].contiguous()
            kmax = kmax[:, _lo:_lo + _hkv_local, :].contiguous()

        # Full-tensor summary dump (cross-impl numerical diff). Fires ONCE at
        # the first layer-0 scoring call. Writes [P, H_kv, D] kmin/kmax so we
        # can numerically compare vLLM's server-fetched summaries against the
        # HF cleanroom's locally-computed summaries.
        _tdump = os.environ.get("ICMS_PER_LAYER_MAX_KV_TENSOR_DUMP", "")
        if (_tdump and int(next_layer_idx) == 0
                and not getattr(self, "_per_layer_tensor_dumped", False)):
            try:
                self._per_layer_tensor_dumped = True
                _qd = quest_query.detach()
                if _qd.ndim == 2:
                    _qd = _qd.unsqueeze(0)
                # ── DIAGNOSTIC rank-aware variant (commented 2026-05-29; revive
                # for TP>1 dumps so ranks don't clobber one file — used to prove
                # the KV-slice fix: concat per-rank kmin → bit-identical TP=1).
                # Uncomment these two and swap _tdump → _tdump_rank below:
                # _tdump_rank = f"{_tdump}.rank{int(self._tp_rank)}"
                _tp_meta = {"tp_rank": int(self._tp_rank),
                            "tp_size": int(self._tp_size)}
                torch.save({
                    "impl": "vllm", "layer": 0, "P": int(P), **_tp_meta,
                    "kmin": kmin.detach().to(torch.float32).cpu(),
                    "kmax": kmax.detach().to(torch.float32).cpu(),
                    "q": _qd.to(torch.float32).cpu(),  # [q_len, H_q(_local), D]
                }, _tdump)  # revive TP>1: _tdump → _tdump_rank
                logger.info("[icms_per_layer_max_kv] tensor dump → %s "
                            "(rank=%d P=%d kmin=%s q=%s)", _tdump,
                            int(self._tp_rank), P, tuple(kmin.shape),
                            tuple(_qd.shape))
            except Exception as _e_td:
                logger.warning("[icms_per_layer_max_kv] tensor dump failed: %s",
                               _e_td)

        # ── 3. per_layer_max_kv scoring on GPU ──
        # quest_query shape: (q_len, H_q, D) at prefill scoring, or
        # (1, H_q, D) at decode. Caller is responsible for the q-token
        # dimension; we sample/aggregate inside the scorer helper.
        q = quest_query.detach()
        if q.ndim == 2:
            # decode: synthesize the q_len=1 dim.
            q = q.unsqueeze(0)
        if q.ndim != 3:
            return
        # Question-only-Q: when the bench's chain cache leaves a haystack tail
        # un-elided, phase-2 re-prefills it, so this prefill quest_query pools
        # (q_token_pool=max) over up to hundreds of *haystack* tokens and
        # pollutes page selection — the dominant cause of vLLM<HF on
        # per_layer_max_kv. Restrict scoring to the last N (question) tokens.
        #   ICMS_Q_TAIL_FILE=<path>: EXACT per-example mode. The bench writes
        #     the precise question token count (len(scored)-len(ctx)) to this
        #     file before each phase-2 generate; read it fresh each call so
        #     every example trims to exactly its own question (no guesswork,
        #     no over-trim). Safe under max_num_seqs=1 (requests serialized).
        #   ICMS_Q_TAIL_TOKENS=N: fixed fallback when no file is set.
        # Decode (q_len=1) is never sliced. Default off = existing path
        # byte-identical. The q.shape[0] > N guard means earlier pure-haystack
        # prefill chunks (q_len <= N) are left alone.
        # 2026-05-31: drop NaN (TP>1 padding/uninit) rows from the FULL captured q
        # BEFORE the question-tail trim, so the trim grabs the genuine question
        # tokens even when the trailing rows are all padding (otherwise the last-N
        # window can be entirely uninitialized → all-NaN → 0-pick prefill → gib).
        if q.ndim == 3 and q.shape[0] > 1 and bool(torch.isnan(q).any()):
            _ft_full = ~torch.isnan(q).any(dim=2).any(dim=1)
            if 1 <= int(_ft_full.sum()) < int(q.shape[0]):
                q = q[_ft_full]
        _q_tail = 0
        try:
            _qfile = os.environ.get("ICMS_Q_TAIL_FILE", "")
            if _qfile:
                _q_tail = int((open(_qfile).read().strip() or "0"))
            else:
                _q_tail = int(os.environ.get("ICMS_Q_TAIL_TOKENS", "0") or "0")
        except (ValueError, OSError):
            _q_tail = 0
        if _q_tail > 0 and q.shape[0] > _q_tail:
            q = q[-_q_tail:]
        # ── 2026-05-31 TP>1 prefill Q-capture NaN guard (root-cause fix for the
        # mistral TP=2 gibberish) ──────────────────────────────────────────────
        # At TP>1 the stop-world / real-Q capture intermittently leaves 1-2 of the
        # PREFILL question-token rows uninitialized (NaN) — verified via
        # ICMS_PER_LAYER_MAX_KV_DUMP: TP2 prefill q_shape=[7,16,128] had 2/7 tokens
        # all-NaN while TP1 captured all 7 clean. With q_token_pool=max a single
        # NaN token poisons EVERY per-page score → topk drops all → 0 picks at the
        # prefill → empty page selection → the first decode token attends nothing →
        # gibberish cascade (decode then re-does the selection one iteration late).
        # The NaN lives in the SCORING Q only (not the model's KV/attention), so
        # dropping the NaN token-rows is safe and strictly improves selection.
        # No-op when no NaN (default path byte-identical). Decode (q_len=1) keeps
        # its row unless it's the only one and NaN (then we leave it untouched).
        if q.ndim == 3 and bool(torch.isnan(q).any()):
            _finite_tok = ~torch.isnan(q).any(dim=2).any(dim=1)  # [q_len]
            _nvalid = int(_finite_tok.sum())
            if 1 <= _nvalid < int(q.shape[0]):
                # Preferred: keep only the genuinely-captured question tokens.
                q = q[_finite_tok]
            # Residual NaN (ALL rows NaN — heavy trailing padding at TP>1 — or
            # partial-channel NaN inside a kept row) → zero it so page scores stay
            # finite. Worst case yields a uniform (non-empty) prefill selection,
            # which decode then refines — strictly better than the 0-pick prefill
            # that left the first decode token attending nothing → gibberish.
            if bool(torch.isnan(q).any()):
                q = torch.nan_to_num(q, nan=0.0)
            if not getattr(self, "_nan_q_drop_logged", False):
                self._nan_q_drop_logged = True
                logger.warning(
                    "[icms_per_layer_max_kv] NaN scoring-Q guard FIRED: "
                    "%d/%d question-token rows valid at layer %d tp_rank=%d "
                    "(TP>1 prefill Q-capture leaves uninit/NaN rows; was causing "
                    "0-pick prefill → gibberish)",
                    _nvalid, int(q.shape[0]), next_layer_idx, self._tp_rank)
        # ── DIAGNOSTIC (commented 2026-05-29; revive by uncommenting) ──
        # One-time Q-TAIL trim probe — used to prove the question-only-Q trim
        # fires at TP>1 (pre_qlen=32295→post_qlen=38). Confirmed working; muted
        # to keep production logs clean.
        # if not getattr(self, "_q_tail_diag_logged", False):
        #     self._q_tail_diag_logged = True
        #     logger.info(
        #         "[icms_per_layer_max_kv] Q-TAIL DIAG: ICMS_Q_TAIL_FILE=%r "
        #         "_q_tail=%d pre_qlen=%d post_qlen=%d (trim %s)",
        #         os.environ.get("ICMS_Q_TAIL_FILE", ""), int(_q_tail),
        #         int(quest_query.shape[0]) if quest_query.ndim >= 1 else -1,
        #         int(q.shape[0]),
        #         "FIRED" if (_q_tail > 0 and int(q.shape[0]) <= int(_q_tail))
        #         else "NO-OP")
        q_dev = q.to(device=device, dtype=torch.float32, non_blocking=True)

        # q_norm liveness probe: log per-head RMS spread of the scoring Q at
        # layer 0, first call. q_norm normalizes every head to ~unit RMS; the
        # broken (pre-q_norm) Q has per-head RMS varying ~15x. Fires once.
        if (os.environ.get("ICMS_PER_LAYER_QNORM_PROBE", "") == "1"
                and int(next_layer_idx) == 0
                and not getattr(self, "_qnorm_probe_logged", False)):
            try:
                self._qnorm_probe_logged = True
                # q_dev: [q_len, H_q, D] → per-head RMS over (q_len, D)
                _rms = q_dev.pow(2).mean(dim=(0, 2)).sqrt()  # [H_q]
                logger.info(
                    "[qnorm-probe] layer0 Q per-head RMS: min=%.4f max=%.4f "
                    "mean=%.4f max/min=%.2f (uniform≈q_norm-LIVE, "
                    "high-ratio≈q_norm-MISSING)",
                    float(_rms.min()), float(_rms.max()),
                    float(_rms.mean()), float(_rms.max() / (_rms.min() + 1e-9)))
            except Exception:
                pass

        from scripts.utils.per_layer_max_kv_scoring import (
            per_layer_max_kv_topk,
            per_layer_max_kv_per_head_topk,
        )
        # Two K=16 random-sample seeding modes:
        #   stream   (default): one process-wide torch.Generator advances
        #                       monotonically across all scoring calls. Same
        #                       seed + identical call order → identical
        #                       samples. Call order matters.
        #   per_call:           fresh torch.Generator per (rid, layer, call_idx)
        #                       seeded by hash((seed_base, rid_int, layer_idx,
        #                       call_idx)). Order-independent across calls;
        #                       designed for cross-impl parity testing vs the
        #                       HF cleanroom (scripts/utils/quest_faithful_hf.py).
        seed_mode = os.environ.get("ICMS_PER_LAYER_SAMPLE_SEED_MODE", "stream")
        try:
            if sample_mode != "random":
                generator = None
            elif seed_mode == "per_call":
                # Track (rid, layer)-local call index on the rs.
                if not hasattr(rs, "_per_layer_max_kv_call_idx"):
                    rs._per_layer_max_kv_call_idx = {}
                _layer_key = int(next_layer_idx)
                call_idx = rs._per_layer_max_kv_call_idx.get(_layer_key, 0)
                rs._per_layer_max_kv_call_idx[_layer_key] = call_idx + 1
                # Stable per-request int from the first 8 hex chars of rid.
                try:
                    rid_int = int(str(rid)[:8].replace("-", "0"), 16)
                except Exception:
                    rid_int = hash(str(rid)) & 0xFFFFFFFF
                sub_seed = (hash((int(self._q_token_sample_seed),
                                   int(rid_int) & 0xFFFFFFFF,
                                   _layer_key, int(call_idx)))
                             & 0x7FFFFFFF)
                generator = torch.Generator(device=q_dev.device)
                generator.manual_seed(int(sub_seed))
            else:
                generator = self._get_q_token_sample_generator(q_dev.device)
            # Optional dump capture (cross-impl parity diagnostic).
            _dump_path = os.environ.get("ICMS_PER_LAYER_MAX_KV_DUMP", "")
            cap = {} if _dump_path else None
            # TP>1: inject the cross-rank MAX-combine of per-page scores so all
            # ranks pick identical pages (rank-local heads → global via NCCL
            # all-reduce). None at TP=1 (byte-identical). Fires symmetrically:
            # all ranks score the same scored layer in index order.
            _tp_arm = self._tp_size
            _tp_fn = ((lambda _t: _tp_allreduce_max_tensor(_t, _tp_arm))
                      if _tp_arm > 1 else None)
            if faithful:
                # FAITHFUL per-KV-head selection: each KV head picks its own
                # top-B; materialize the UNION (fed to the unchanged fetch),
                # stash the [H_kv, P] per-head mask for the attention kernel.
                # No tp_allreduce (per-head selection is not MAX-composable;
                # TP=1 guarded above).
                # TODO(faithful): per-head cumulative exclude_per_head to mirror
                # HF state.selected masking across decode iterations; for the
                # single-shot prefill scoring already_fetched is empty.
                union_t, _faithful_sel = per_layer_max_kv_per_head_topk(
                    q_dev, kmin, kmax,
                    budget=float(budget),
                    q_token_pool=qtok_pool, q_head_pool=qhd_pool,
                    q_token_sample_count=k_sample,
                    q_token_sample_mode=sample_mode,
                    generator=generator,
                    exclude_per_head=None,
                    capture=cap,
                )
                picks_t = union_t
            else:
                _faithful_sel = None
                picks_t = per_layer_max_kv_topk(
                    q_dev, kmin, kmax,
                    budget=float(budget),
                    q_token_pool=qtok_pool, q_head_pool=qhd_pool,
                    kv_reduce=kv_reduce,
                    q_token_sample_count=k_sample,
                    q_token_sample_mode=sample_mode,
                    generator=generator,
                    exclude_pages=already_fetched if already_fetched else None,
                    capture=cap,
                    tp_allreduce_max_fn=_tp_fn,
                )
            picked = [int(p) for p in picks_t.tolist()]
            k = picks_t.numel()
            # ── Write JSONL dump line ──────────────────────────────────
            if _dump_path:
                try:
                    import hashlib as _hl, json as _json
                    # Per-(rid, layer) call index from rs._per_layer_max_kv_call_idx,
                    # which is also incremented by the per_call seed mode. Track a
                    # parallel counter for dump so we don't conflict if per_call mode
                    # is off (counter wasn't created yet).
                    if not hasattr(rs, "_per_layer_max_kv_dump_call_idx"):
                        rs._per_layer_max_kv_dump_call_idx = {}
                    _lk = int(next_layer_idx)
                    _ci = rs._per_layer_max_kv_dump_call_idx.get(_lk, 0)
                    rs._per_layer_max_kv_dump_call_idx[_lk] = _ci + 1
                    import base64 as _b64
                    q_bytes = q_dev.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
                    kmin_bytes = kmin.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
                    kmax_bytes = kmax.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
                    q_first10 = q_dev.detach().flatten()[:10].to(torch.float32).cpu().tolist()
                    picks_sorted = sorted(picked)

                    def _f16b64(t):
                        # fp16 little-endian base64 (compact, lossless-enough for
                        # rank/score profiling). Returns ("", []) on failure.
                        try:
                            arr = t.detach().to(torch.float16).contiguous().cpu().numpy()
                            return (_b64.b64encode(arr.tobytes()).decode("ascii"),
                                    list(arr.shape))
                        except Exception:
                            return ("", [])

                    # Full post-exclusion page-score vector (from scorer capture)
                    # so the offline profiler can recover the answer-page rank.
                    _ps = (cap or {}).get("page_scores")
                    _ps_b64, _ps_shape = _f16b64(_ps) if _ps is not None else ("", [])
                    # Full q vector [q_len, H_q, D] (decode q_len=1 → tiny).
                    _q_b64, _q_shape = _f16b64(q_dev)
                    if not hasattr(self, "_diag_rid_ord"):
                        self._diag_rid_ord = {}
                    _rord = self._diag_rid_ord.setdefault(
                        str(rid), len(self._diag_rid_ord))
                    rec = {
                        "impl": "vllm_connector",
                        "tp_rank": int(self._tp_rank),
                        "tp_size": int(self._tp_size),
                        "req_ord": int(_rord),
                        "rid_short": str(rid)[:8],
                        "rid": str(rid),
                        "is_decode": bool(self._prefill_done),
                        "cum_selected": int(len(already_fetched)) if already_fetched else 0,
                        "layer": _lk,
                        "call_idx": int(_ci),
                        "q_shape": list(q_dev.shape),
                        "q_first10": q_first10,
                        "q_sha": _hl.sha256(q_bytes).hexdigest()[:32],
                        "kmin_sha": _hl.sha256(kmin_bytes).hexdigest()[:32],
                        "kmax_sha": _hl.sha256(kmax_bytes).hexdigest()[:32],
                        "P": int(P),
                        "B_eff": int(k),
                        "sampled_positions": (cap or {}).get("sampled_positions"),
                        "picks": picks_sorted,
                        "picks_first50": picks_sorted[:50],
                        "n_picks": len(picks_sorted),
                        "picks_sha": _hl.sha256(
                            _json.dumps(picks_sorted).encode()).hexdigest()[:32],
                        "page_scores_b64": _ps_b64,
                        "page_scores_shape": _ps_shape,
                        "q_b64": _q_b64,
                        "q_b64_shape": _q_shape,
                        "qtok_pool": qtok_pool, "qhd_pool": qhd_pool,
                        "kv_reduce": kv_reduce,
                        "k_sample": int(k_sample),
                        "sample_mode": sample_mode,
                        "seed_mode": seed_mode,
                        "seed_base": int(self._q_token_sample_seed),
                    }
                    with open(_dump_path, "a") as _df:
                        _df.write(_json.dumps(rec) + "\n")
                except Exception:
                    pass
        except Exception as e:
            logger.warning("[icms_per_layer_max_kv] scoring failed at "
                            "layer %d rid=%s: %s",
                            next_layer_idx, rid, e)
            return

        if not picked:
            if not self._per_layer_max_kv_empty_picks_logged:
                logger.info(
                    "[icms_per_layer_max_kv] layer %d picks empty "
                    "rid=%s — scoring fired but selected 0 pages",
                    next_layer_idx, rid[:8])
                self._per_layer_max_kv_empty_picks_logged = True
            return

        if next_layer_idx not in self._per_layer_max_kv_picks_logged_layers:
            self._per_layer_max_kv_picks_logged_layers.add(next_layer_idx)
            logger.info(
                "[icms_per_layer_max_kv] layer %d rid=%s picked %d pages, "
                "first10=%s",
                next_layer_idx, rid[:8], len(picked), picked[:10])

        if os.environ.get("ICMS_DIAG_SCORE_DUMP", "") \
                and int(self._tp_rank) == 0:
            try:
                rs.last_picked_by_layer[int(next_layer_idx)] = list(picked)
                if int(next_layer_idx) not in rs.last_q_by_layer:
                    rs.last_q_by_layer[int(next_layer_idx)] = torch.zeros(
                        1, dtype=torch.uint8)
                if not hasattr(rs, "picked_call_log"):
                    rs.picked_call_log = []
                rs.picked_call_log.append({
                    "call": len(rs.picked_call_log),
                    "layer": int(next_layer_idx),
                    "is_decode": bool(self._prefill_done),
                    "budget": float(budget),
                    "k": int(k),
                    "total_pages": int(total_pages),
                    "picked": list(picked),
                    "full_scores_per_head": None,
                    "mode": "per_layer_max_kv",
                })
            except Exception as _e_stash:
                logger.warning(
                    "[icms_per_layer_max_kv] dump stash failed at layer "
                    "%d rid=%s: %s", next_layer_idx, rid, _e_stash)

        # ── 4. Apply path wiring (mirror subset_max verbatim) ──
        if (self._sink_pool is None
                or not hasattr(self._client, "fetch_union_per_head")):
            if not self._per_layer_max_kv_apply_unsupported_warned:
                logger.warning(
                    "[icms_per_layer_max_kv] client lacks "
                    "fetch_union_per_head or sink_pool missing; apply "
                    "path cannot be wired.")
                self._per_layer_max_kv_apply_unsupported_warned = True
            return

        num_layers = int(self._geom.num_layers)
        if self._geom.dense_layers_mask != 0:
            _nxt = self._geom.next_scored_layer_after(next_layer_idx)
            reuse_through = (_nxt - 1) if _nxt is not None \
                else (num_layers - 1)
        else:
            reuse_through = min(
                next_layer_idx + self._score_stride - 1,
                num_layers - 1)
        try:
            with self._rpc_lock:
                union_reply = self._client.fetch_union_per_head(
                    request_id=icms_rid, chain=chain,
                    layer=next_layer_idx,
                    sink=self._sink_pool.sink, page_ids=picked,
                    reuse_through_layer=reuse_through)
        except IcmsError as e:
            logger.warning(
                "[icms_per_layer_max_kv] FetchUnionPerHead failed at "
                "layer %d (rid=%s, k=%d, reuse_through=%d): %r",
                next_layer_idx, rid[:8], len(picked), reuse_through, e)
            return
        except Exception as e:
            logger.warning(
                "[icms_per_layer_max_kv] FetchUnionPerHead unexpected at "
                "layer %d (rid=%s, k=%d): %r",
                next_layer_idx, rid[:8], len(picked), e)
            return

        try:
            from icms_client.protocol import ScoreReply
            synth = ScoreReply(
                request_id=icms_rid,
                status=0,
                trie_walk_ns=0,
                summary_read_ns=0,
                score_ns=int((time.perf_counter() - t0) * 1e9),
                sink_write_ns=int(getattr(union_reply,
                                           "sink_write_ns", 0)),
                cache_hit=False,
                concurrent_requests=int(getattr(
                    union_reply, "concurrent_requests", 0)),
                server_ingest_to_ready_ns=int(getattr(
                    union_reply, "server_ingest_to_ready_ns", 0)),
                effective_supply_bps=int(getattr(
                    union_reply, "effective_supply_bps", 0)),
                page_ids=list(picked),
                scores=[0.0] * len(picked),
                sink_offsets=list(union_reply.sink_offsets),
            )
        except Exception as e:
            logger.warning(
                "[icms_per_layer_max_kv] synth ScoreReply failed at "
                "layer %d: %s", next_layer_idx, e)
            return

        attn_layer_name = f"model.layers.{next_layer_idx}.self_attn.attn"
        # Faithful Quest: stash the FULL [H_kv, P] page-id-keyed mask (NOT
        # union-slot order). The apply path indexes it by the trimmed
        # block_table's context page order (valid_pids / cumulative_pids), so
        # the mask stays correct even when the union is filtered/reordered
        # downstream. Robust against the audit's #1 risk (mask must key to the
        # post-fetch block-table order).
        _faithful_hm = None
        if faithful and _faithful_sel is not None:
            _faithful_hm = _faithful_sel.contiguous()  # [H_kv, P] bool
        with self._score_lock:
            self._assert_pending_scores_no_clobber(
                attn_layer_name, rid,
                source="per_layer_max_kv-score-landing")
            self._pending_scores.setdefault(
                attn_layer_name, {})[rid] = (synth, req_idx)
            if _faithful_hm is not None:
                self._pending_faithful_masks.setdefault(
                    attn_layer_name, {})[rid] = _faithful_hm

        if not self._per_layer_max_kv_ctx_probe_logged:
            try:
                _sg = int(getattr(rs, "stored_groups", 0))
                _ngw = int(getattr(rs, "num_groups_written", 0))
                _ctx = max(_sg, _ngw) * 32
                _above = (sum(1 for p in picked if p >= _ctx)
                          if _ctx > 0 else len(picked))
                logger.info(
                    "[icms_per_layer_max_kv] CTX-PROBE rid=%s layer=%d "
                    "P=%d k=%d stored_groups=%d ngw=%d ctx_pages=%d "
                    "picks_above_ctx=%d picks_minmax=(%d,%d)",
                    rid[:8], next_layer_idx, P, k, _sg, _ngw, _ctx,
                    _above, min(picked), max(picked))
                self._per_layer_max_kv_ctx_probe_logged = True
            except Exception:
                pass

        # _pending_reuse for stride/contig-reuse window.
        actual_k = len(picked)
        per_layer_bytes = actual_k * int(self._geom.kv_page_bytes)
        for delta in range(1, reuse_through - next_layer_idx + 1):
            reuse_layer = next_layer_idx + delta
            reuse_attn = f"model.layers.{reuse_layer}.self_attn.attn"
            reuse_offsets = [off + delta * per_layer_bytes
                              for off in union_reply.sink_offsets]
            with self._score_lock:
                self._pending_reuse.setdefault(reuse_attn, {})[rid] = (
                    synth, reuse_offsets, req_idx)
                # Faithful: the scored layer's per-head mask reuses across the
                # stride window (same union order → same [H_kv, U] mask).
                if _faithful_hm is not None:
                    self._pending_faithful_masks.setdefault(
                        reuse_attn, {})[rid] = _faithful_hm

        wall_us = (time.perf_counter() - t0) * 1e6
        rs.fetched_pages.setdefault(next_layer_idx, set()).update(picked)
        self.stats.record_score(wall_us, bool(picked), picked,
                                  next_layer_idx)
        logger.debug(
            "[icms_per_layer_max_kv] layer=%d total_pages=%d k=%d "
            "n_picked=%d sink_offsets=%d wall_us=%.1f",
            next_layer_idx, total_pages, k, len(picked),
            len(union_reply.sink_offsets), wall_us)
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
