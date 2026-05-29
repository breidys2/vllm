# SPDX-License-Identifier: Apache-2.0
"""ICMS connector public facade (IcmsConnector, KVConnectorBase_V1 subclass).

Extracted verbatim from icms_connector.py (behavior-preserving split). Imports
its deps from the neutral helper modules; _Worker still lives in icms_connector
(until the _Worker mixin split), so it is imported FUNCTION-LOCALLY at its
instantiation site to avoid an import cycle. Re-exported by icms_connector so
the public path 'vllm...v1.icms_connector.IcmsConnector' is unchanged.
"""
from __future__ import annotations

import os
import time
from typing import Any

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

from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_scheduler import (
    _Scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import (
    _allow_batch,
    _instr_timing,
)
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import (
    IcmsConnectorMetadata,
)

# Log under the original connector logger name (behavior-preserving
# split): all split modules share one logger so log-name filtering,
# grep, and assertLogs see the same name as before.
logger = init_logger("vllm.distributed.kv_transfer.kv_connector.v1.icms_connector")


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
            # _Worker still lives in icms_connector.py (until the mixin
            # split); import it here to avoid an import cycle.
            from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector import (  # noqa: E402
                _Worker,
            )
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

    @_instr_timing("start_load_kv")
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

    @_instr_timing("wait_for_layer_load")
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

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        _instr_skv = os.environ.get("ICMS_INSTR", "0") == "1"
        _t_skv0 = time.perf_counter() if _instr_skv else 0.0
        try:
            return self._save_kv_layer_impl(layer_name, kv_layer, attn_metadata, **kwargs)
        finally:
            if _instr_skv:
                _dt = (time.perf_counter() - _t_skv0) * 1000.0
                # >1ms catches prefill; the layer_ prefix catches the per-token
                # quest-hook scoring/reuse dispatches (scored layers arrive as
                # save_kv_layer[layer_{S-1}]) so we can size the decode scoring
                # cost that stop-world would relocate onto the critical path.
                if _dt > 1.0 or str(layer_name).startswith("layer_"):
                    logger.info("[INSTR] save_kv_layer[%s]: %.3fms", layer_name, _dt)

    def _save_kv_layer_impl(  # noqa: C901
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

    @_instr_timing("wait_for_save")
    def wait_for_save(self) -> None:
        if self._worker is not None:
            self._worker.wait_for_pending_writes()

    @_instr_timing("get_finished")
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

    @_instr_timing("build_connector_meta")
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        if self._sched is not None:
            return self._sched.build_meta(scheduler_output)
        return IcmsConnectorMetadata()

    @_instr_timing("request_finished")
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
