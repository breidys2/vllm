# SPDX-License-Identifier: Apache-2.0
"""ICMS connector scheduler-side half (_Scheduler).

Extracted verbatim from icms_connector.py (behavior-preserving split). Imports
its deps from the neutral helper modules (types/trace) so there is no circular
import back into icms_connector; re-exported by icms_connector.
"""
from __future__ import annotations

import os

from icms_client.geometry import GROUP_PAGES, PAGE_TOKENS
from vllm.distributed.kv_transfer.kv_connector.v1 import icms_provenance
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    icms_connector_trace as _trace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import (
    _ICMS_FULLTRACE_ENABLED,
    _allow_batch,
    _drain_stored_chain_queue,
    _icms_chain_fp,
    _icms_fulltrace,
)
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import (
    IcmsConnectorMetadata,
    _PerRequestStep,
)

# Log under the original connector logger name (behavior-preserving
# split): all split modules share one logger so log-name filtering,
# grep, and assertLogs see the same name as before.
logger = init_logger("vllm.distributed.kv_transfer.kv_connector.v1.icms_connector")
_GROUP_BLOCKS = GROUP_PAGES   # blocks per group (= pages per group); see connector


class _Scheduler:
    """Scheduler-side state. Reads block_hashes from Request, packs metadata."""

    def __init__(self, vllm_config):
        self._vllm_config = vllm_config
        # 2026-05-26: snapshot vLLM's prefix-cache flag so
        # get_num_new_matched_tokens can short-circuit when prefix
        # caching is OFF and the model has unscored (SW) layers — see
        # hybrid-cold-fallback comment in get_num_new_matched_tokens.
        _cc = getattr(vllm_config, "cache_config", None)
        self._vllm_prefix_cache_enabled = bool(
            _cc is not None and getattr(_cc, "enable_prefix_caching", False))
        self._hybrid_cold_logged = False
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
        # 2026-05-26 HYBRID-SW COLD FALLBACK: when vLLM's prefix cache is
        # OFF and the model is actually hybrid attention (e.g. gemma-3
        # sliding-window), reporting matched_tokens > 0 tells vLLM to skip
        # haystack prefill. ICMS apply only restores K/V for the scored
        # (dense) layers via the trimmed bt; the SW layers' K/V is never
        # written into vLLM's freshly-allocated blocks → SW attention
        # reads stale/uninitialized memory → garbage output that
        # alternates per cycle as the vLLM block allocator hands back
        # different block IDs. Force cold-prefill so all 62 gemma-3
        # layers get fresh K/V; ICMS apply still trims the dense layers
        # to top-K from the sink. Cost: one full haystack re-prefill
        # per budget cycle (~1 s at h32k, ~5 s at h128k) — negligible
        # for sparse RULER cells. ICMS_HYBRID_COLD_DISABLE=1 disables
        # this safety net for diagnostic A/B.
        # 2026-05-28 FIX: gate was previously
        # `ICMS_SCORED_LAYERS != ""`, but uniform models (qwen3,
        # mistral-small) ALSO set ICMS_SCORED_LAYERS to pick which
        # subset of dense layers to score-and-store. The mistargeted
        # gate forced cold-prefill EVERY iter for these models —
        # turning a ~80 ms pf=4096 compute (with ICMS-fetched ctx) into
        # a ~3.4 s full ctx+pf recompute → 4x TTFT regression at
        # qwen3 ctx=65k (4605 vs ~1163 ms Apr 27). Re-gated on the
        # explicit `ICMS_HYBRID_MODEL=1` env var (already set by
        # `run_full_ttft_sweep.sh` only for gemma-3) so uniform models
        # are unaffected. ICMS_HYBRID_COLD_DISABLE=1 still overrides.
        _hybrid_cold = (
            not getattr(self, "_vllm_prefix_cache_enabled", True)
            and os.environ.get("ICMS_HYBRID_MODEL", "0") == "1"
            and os.environ.get("ICMS_HYBRID_COLD_DISABLE") != "1"
        )
        if _hybrid_cold:
            if not getattr(self, "_hybrid_cold_logged", False):
                logger.info(
                    "[icms] hybrid-cold-fallback active: prefix_cache=OFF "
                    "and ICMS_SCORED_LAYERS set; returning 0 matched_tokens "
                    "so vLLM cold-prefills all layers (preserves SW-layer "
                    "KV in models like gemma-3). Set "
                    "ICMS_HYBRID_COLD_DISABLE=1 to skip this safety net.")
                self._hybrid_cold_logged = True
            if _ICMS_FULLTRACE_ENABLED:
                _icms_fulltrace(
                    "gnnmt_exit_hybrid_cold", rid=request.request_id,
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
        # 2026-05-29 TP>1 prefix-cache bypass: at TP>1 with
        # VLLM_WORKER_MULTIPROC_METHOD=spawn the worker and scheduler run
        # in SEPARATE processes; _stored_chain_queue / _stored_chain_cond /
        # _stored_chain_generation live in icms_connector_trace as Python
        # module globals which do NOT cross process boundaries. The
        # wait_for predicate `_stored_chain_generation > gen_at_wait` can
        # therefore never fire at TP>1, so this loop always full-timeouts
        # at 3 x ICMS_STORED_CHAIN_WAIT_MS = 600ms per warm-prefix lookup.
        # When vllm_prefix_cache_enabled=True, vLLM's own
        # find_longest_cache_hit (called BEFORE the connector at
        # vllm/v1/core/sched/scheduler.py:601-613) already covers any
        # reusable haystack from prior iters within the same run, so
        # returning matched_groups=0 here is benign — vLLM elides the
        # haystack on its own. The single-rid TP=1 warm-prefix path also
        # doesn't need this wait (the eager _lookup_stored_prefix above
        # already resolves on iter 2+ because on_finished_record_stored
        # ran synchronously between iters). The only regime where this
        # wait is genuinely needed is the 2026-05-09 v2 race: TP=1 +
        # max_num_seqs>=2 batched per-batch/per-budget + BLOCK_WRITES=1 +
        # prefix_caching=False; that path keeps _vllm_prefix_cache_enabled
        # =False and so still runs the wait below.
        #
        # Defense-in-depth: getattr default is False (preserve legacy wait
        # when attribute is missing). _Scheduler.__init__ unconditionally
        # sets the attr at line 51-52, so the default is dead code today;
        # but if a future refactor drops that init, defaulting to False
        # silently keeps the v2 race protection rather than skipping it.
        # ICMS_STORED_CHAIN_WAIT_SKIP_DISABLE=1 restores the original wait
        # for diagnostics or if a future regression emerges.
        _skip_stored_chain_wait = (
            bool(getattr(self, "_vllm_prefix_cache_enabled", False))
            and os.environ.get(
                "ICMS_STORED_CHAIN_WAIT_SKIP_DISABLE", "0") != "1")
        if matched_groups == 0 and len(chain) >= 4 and not _skip_stored_chain_wait:
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
                # _stored_chain_generation now lives in icms_connector_trace and
                # is REBOUND by _bump/_append on the worker thread — read it via
                # the _trace module object so this wait observes the live value
                # (a `from .. import _stored_chain_generation` copy would be
                # frozen at import and never wake → cross-thread bridge stall).
                with _trace._stored_chain_cond:
                    gen_at_wait = _trace._stored_chain_generation
                    _trace._stored_chain_cond.wait_for(
                        lambda: _trace._stored_chain_generation > gen_at_wait,
                        timeout=_wait_timeout_s)
        elif matched_groups == 0 and len(chain) >= 4 and _skip_stored_chain_wait:
            # When skipping the cross-process wait, opportunistically drain
            # any queued chains that landed BETWEEN the line-403 drain and
            # this point (TP=1 worker-thread same-process race) and
            # re-lookup once. Zero added latency vs. the wait path.
            for c, n in _drain_stored_chain_queue():
                self.record_stored_chain(c, n)
            matched_groups = self._lookup_stored_prefix(chain)
            if not getattr(self, "_stored_chain_wait_skip_logged", False):
                logger.info(
                    "[icms-stored-chain-wait-skip] active: vLLM prefix "
                    "cache is ON; skipping cross-iter stored-chain "
                    "wait_for (dead code at TP>1 spawn; same-process drain "
                    "preserved). Set ICMS_STORED_CHAIN_WAIT_SKIP_DISABLE=1 "
                    "to restore the original wait.")
                self._stored_chain_wait_skip_logged = True
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
            # 2026-05-28 TP>1 CROSS-PROCESS BRIDGE RESTORATION:
            # At TP>1 with VLLM_WORKER_MULTIPROC_METHOD=spawn the worker
            # and scheduler are SEPARATE processes; the module-global
            # _stored_chain_queue at icms_connector.py:683 doesn't cross
            # process boundaries. So worker_n_groups will be 0 here on
            # EVERY request at TP>1, and the legacy early-return makes
            # _stored_chains permanently empty → get_num_new_matched_tokens
            # returns 0 forever → vLLM re-prefills the full ctx on every
            # iter → 4-5x TTFT regression vs Apr 27 baseline.
            # The Apr 27 milestone commit (3c0855a89, "ttft numbers
            # looking good") handled this by deterministically recording
            # n_groups = len(token_ids) // group_tokens — the scheduler
            # has all token_ids and the chain hash itself; n_groups is a
            # tight upper bound because the worker only writes whole
            # groups that fit under the prompt length. Apr 28 commit
            # 5a0200a91 replaced that with the worker-bounded gate above
            # to avoid Score ENOENT when WriteGroup partially fails at
            # TP=1 — a legitimate TP=1 correctness concern that broke
            # the TP>1 cross-process bridge as collateral damage.
            # Fallback gated on ICMS_TRUST_PROMPT_LEN_N_GROUPS=1 (set by
            # run_full_ttft_sweep.sh) so the legacy TP=1 ENOENT-safe
            # behavior is preserved by default. Safe for budget=1.0
            # fetch_all bench (B config) regardless: fetch_all returns
            # exactly what the trie has, so over-promising the elided
            # ctx prefix is corrected by the actual fetch reply.
            if (os.environ.get("ICMS_TRUST_PROMPT_LEN_N_GROUPS", "0")
                    == "1"):
                token_ids = getattr(request, "all_token_ids", None)
                if token_ids is None or len(token_ids) == 0:
                    token_ids = getattr(request, "prompt_token_ids", [])
                if token_ids:
                    group_tokens = _GROUP_BLOCKS * PAGE_TOKENS
                    n_groups_est = len(token_ids) // group_tokens
                    if n_groups_est > 0:
                        # Bound at chain length so we never claim more
                        # groups than the chain actually has.
                        n_groups_est = min(n_groups_est, len(chain))
                        self.record_stored_chain(chain, n_groups_est)
                        logger.debug(
                            "on_finished_record_stored: rid=%s chain_len=%d "
                            "TRUST_PROMPT_LEN fallback recorded n_groups=%d "
                            "(token_ids=%d, group_tokens=%d)",
                            rid, len(chain), n_groups_est,
                            len(token_ids), group_tokens)
                        return
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
