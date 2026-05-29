# SPDX-License-Identifier: Apache-2.0
"""ICMS connector dataclasses (scheduler<->worker payload, per-request state,
write pipeline, sink slot pool, timing stats).

Extracted verbatim from icms_connector.py (behavior-preserving split). Imported
back + re-exported by icms_connector for the public/test import surface.
"""
from __future__ import annotations

import queue as _queue
import threading
from dataclasses import dataclass, field
from typing import Any

import torch

from icms_client.geometry import GROUP_PAGES
from icms_client.sink import Sink
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger

logger = init_logger(__name__)


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
        # total_score_calls must increment on EVERY Score call regardless
        # of stats level — it's read by the bench's silent-fallback guard
        # to detect engine death (0 Score calls over a non-trivial cell
        # → engine is dead, dense-attention rows silently labelled as
        # ICMS budget). Pre-2026-05-20 fix this was gated on `level >= 1`,
        # so default runs (level=0) had total_score_calls stuck at 0,
        # spuriously tripping the guard on short-prompt benchmarks where
        # Score legitimately ran (MMLU/GSM8K smoke 2026-05-20). The
        # detailed timing/cache-hit fields below stay gated on level
        # since they're expensive (per-call lists at level=2).
        self.total_score_calls += 1
        if cache_hit:
            self.total_score_cache_hits += 1
        if self.level == 0 and not self.log_selections:
            return
        if self.level >= 1:
            self.total_score_us += us
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
    # 2026-05-28 v2: TP-symmetrized effective_groups cache. Populated
    # ONCE per step in on_step_start for rids in meta.new_chains (the
    # scheduler-broadcast set both ranks agree on). Apply/Score/
    # FetchAll read this when not None, falling back to per-call
    # allreduce otherwise. Eliminates the per-layer `.item()` that
    # was draining the default CUDA stream and serializing apply
    # behind vLLM's per-layer attention NCCL.
    eff_groups_synced: object = None  # Optional[int]; None = not synced
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
