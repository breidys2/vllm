# SPDX-License-Identifier: Apache-2.0
"""ICMS connector _Worker mixin: _WorkerDiagMixin.

Extracted verbatim from icms_connector.py (behavior-preserving split).
Methods reference self.* attributes set by _WorkerBase.__init__ and call
sibling-mixin methods via the _Worker MRO; imports resolve from the neutral
helper modules so there is no cycle back into icms_connector.
"""
from __future__ import annotations

from icms_client.geometry import GROUP_PAGES
from dataclasses import field
from vllm.distributed.kv_transfer.kv_connector.v1 import icms_provenance
import os
import time
from vllm.logger import init_logger

# Log under the original connector logger name (behavior-preserving
# split): all split modules share one logger so log-name filtering,
# grep, and assertLogs see the same name as before.
logger = init_logger("vllm.distributed.kv_transfer.kv_connector.v1.icms_connector")


class _WorkerDiagMixin:
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
            logger.info(
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
        _flag = os.environ.get("ICMS_RID_TO_BT_ROW_FLAG", "warn")
        if _flag == "strict":
            raise RuntimeError(msg)
        if _flag == "silent":
            # 2026-05-20: at batch=4+, this WARNING fires per (layer ×
            # budget × batch) and dominates log volume (~1M+ lines/hr,
            # 500MB+ logs), serializing the worker on synchronous disk
            # I/O and capping decode throughput. The check is a known
            # benign noise source post-fix; set ICMS_RID_TO_BT_ROW_FLAG
            # =silent on long sparse sweeps to skip the log entirely.
            return
        logger.warning("%s", msg)
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
