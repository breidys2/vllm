# SPDX-License-Identifier: Apache-2.0
"""ICMS connector _Worker mixin: _WorkerBase.

Extracted verbatim from icms_connector.py (behavior-preserving split).
Methods reference self.* attributes set by _WorkerBase.__init__ and call
sibling-mixin methods via the _Worker MRO; imports resolve from the neutral
helper modules so there is no cycle back into icms_connector.
"""
from __future__ import annotations

# RDMA client (optional — pyverbs must be installed). Mirrors icms_connector.
try:
    from icms_client.rdma_client import RdmaIcmsClient  # noqa: E402
    from icms_client.rdma_transport import RdmaTransportConfig  # noqa: E402
    _HAVE_RDMA = True
except ImportError:
    _HAVE_RDMA = False
from icms_client.sink import Sink

from icms_client import IcmsClient
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import IcmsTimingStats
from icms_client.geometry import KvLayout
from icms_client.geometry import ModelGeometry
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import _RequestState
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import _SinkSlotPool
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import _WritePipeline
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _allow_batch
import dataclasses as _dataclasses
from icms_client.geometry import find_model
import os
from icms_client.geometry import parse_scored_layers
import threading
import torch
from vllm.logger import init_logger

# Log under the original connector logger name (behavior-preserving
# split): all split modules share one logger so log-name filtering,
# grep, and assertLogs see the same name as before.
logger = init_logger("vllm.distributed.kv_transfer.kv_connector.v1.icms_connector")


class _WorkerBase:
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

        # ICMS_SCORING_MODE=per_layer_max_kv: client-side Quest-faithful
        # scoring on per-head summaries, with optional K random q-token
        # sampling. The torch.Generator is lazy-allocated on first use so
        # the scoring device (host vs cuda) doesn't have to be known at
        # __init__ time. NOT reset between requests — generator state
        # advances monotonically across all per-layer-max-kv scoring
        # calls in this process; reproducibility is "same seed → same
        # full trajectory" (the HF cleanroom impl follows the same rule).
        self._q_token_sample_generator = None  # type: ignore[var-annotated]
        self._q_token_sample_seed: int = int(
            os.environ.get("ICMS_PER_LAYER_SEED_BASE", "0"))
        self._per_layer_max_kv_first_call_logged: bool = False
        self._per_layer_max_kv_tp_warned: bool = False
        self._per_layer_max_kv_client_unsupported_warned: bool = False
        self._per_layer_max_kv_q_len_warned: bool = False
        self._per_layer_max_kv_empty_picks_logged: bool = False
        self._per_layer_max_kv_apply_unsupported_warned: bool = False
        self._per_layer_max_kv_picks_logged_layers: set[int] = set()
        self._per_layer_max_kv_ctx_probe_logged: bool = False

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
        # ICMS_DENSE_LAYERS: subset of scored_layers that fire Score with
        # budget=1.0 (full-fetch). Activates contiguous-reuse mode: reuse
        # window walks to the next scored layer instead of stride-1, and
        # wait_for_layer's non-scored short-circuit falls through when a
        # reuse entry was promoted. Required for hybrid schedules like
        # "L0,L1 dense + L2,L8,... Quest @ 0.20" on full-attention models
        # (qwen3) where every layer must apply the selected page set.
        _dense_spec = os.environ.get("ICMS_DENSE_LAYERS", "")
        _dense_mask = parse_scored_layers(_dense_spec)
        if _dense_mask != 0 and self._geom.num_layers < 64:
            _dense_mask &= ((1 << self._geom.num_layers) - 1)
        if _dense_mask != 0:
            _bad = _dense_mask & ~self._geom.scored_layers_mask
            if _bad:
                raise ValueError(
                    f"ICMS_DENSE_LAYERS bitmask 0x{_dense_mask:x} contains "
                    f"layers not in ICMS_SCORED_LAYERS (extra bits "
                    f"0x{_bad:x}). Dense layers must be a subset of scored "
                    f"layers.")
            self._geom = _dataclasses.replace(self._geom,
                                               dense_layers_mask=_dense_mask)
            logger.info("[icms] dense_layers mask=0x%x popcount=%d "
                         "(contiguous-reuse enabled)",
                         _dense_mask, bin(_dense_mask).count("1"))
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
        #
        # 2026-05-27: for hybrid attention models (gemma-3: 10 dense full-
        # attention layers + 52 sliding-window layers), only the dense
        # layers participate in the ICMS K/V path — SW layers use vLLM's
        # local cache and don't transfer K/V at all. Mirror the sink-sizing
        # condition at line 2933-2936: when scored_layers_mask is set AND
        # dense_layers_mask is 0 (i.e., NOT in contiguous-reuse mode used
        # by qwen3/llama/mistral), use num_scored_layers so the allocator's
        # demand math counts only the K/V actually transferred. Without
        # this, gemma-3's allocator over-counts demand by 6.2× (62/10) and
        # snaps the C budget too low — making C config look closer to A
        # than it actually is.
        # The matching slack table must also be dense-only (see
        # profile_compute_slack.py --scored-layers).
        if self._adaptive_allocator is not None:
            self._adaptive_allocator._kv_page_bytes = self._geom.kv_page_bytes
            # 2026-05-28 Bug 1 fix: the prior condition checked
            # `dense_layers_mask == 0` and fired for BOTH qwen3 and
            # gemma-3 (neither sets ICMS_DENSE_LAYERS), so qwen3's
            # demand was undercounted by 6× (8 scored vs 48 actual
            # transferring layers). The correct discriminator is the
            # ICMS_HYBRID_MODEL env var (set externally for gemma-3
            # only): hybrid → only scored (=dense) layers transfer;
            # uniform → all layers transfer using strided selection.
            _hybrid = os.environ.get("ICMS_HYBRID_MODEL", "0") == "1"
            if _hybrid:
                _alloc_layers = int(self._geom.num_scored_layers)
            else:
                _alloc_layers = int(self._geom.num_layers)
            self._adaptive_allocator._num_layers = _alloc_layers
            # 2026-05-28 Bug 3 fix: per-rank demand registered against
            # the FULL link bandwidth, but the physical BF2 link is
            # shared by all TP ranks. Divide link share by tp_size so
            # each rank's `my_share` reflects its true fraction of the
            # shared link. Without this, demand never crowds the link
            # and adaptive picks budget=1.0 everywhere (paper bench:
            # qwen3 ctx=128k → 4.7 GB/s "demand" vs 11 GB/s "share"
            # → budget=1.0; with fix: per-rank share = 5.5 GB/s → real
            # page-skipping engages).
            if self._tp_size > 1:
                self._adaptive_allocator._link_bw = (
                    self._adaptive_allocator._link_bw / float(self._tp_size))
                logger.info("[icms] adaptive link_bw per-rank: %.1f MB/s "
                            "(full link / tp_size=%d)",
                            self._adaptive_allocator._link_bw / 1e6,
                            self._tp_size)
            logger.info("[icms] adaptive num_layers=%d "
                        "(hybrid=%s, num_scored=%d, num_full=%d)",
                        _alloc_layers, _hybrid,
                        self._geom.num_scored_layers,
                        self._geom.num_layers)

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
        # 2026-05-19: when scored_layers_mask is set (e.g. ICMS_SCORED_LAYERS
        # for gemma-3 SWA models), the matching server patch makes FetchAll
        # skip non-scored layers AND lay out sink slots by scored_rank. So
        # we only need num_scored_layers × per_layer_bytes here instead of
        # num_layers × per_layer_bytes. For gemma-3-27b that drops sink
        # from 62 layers to 10, freeing ~22 GiB of GPU memory per worker.
        per_layer_bytes = self._k * self._geom.kv_page_bytes
        # 2026-05-19: in contiguous-reuse mode (dense_layers_mask != 0),
        # the server side runs without --scored-layers so it writes K/V
        # for ALL num_layers slots (not just num_scored_layers). The
        # client's reuse offset math (off + delta * per_layer_bytes for
        # non-scored reuse layers) reads from slot abs_layer — so the
        # sink MUST be sized for num_layers. Sizing for num_scored_layers
        # would silently truncate reuse reads on layers > scored_rank.
        _eff_layers = (int(self._geom.num_scored_layers)
                       if (self._geom.scored_layers_mask != 0
                           and self._geom.dense_layers_mask == 0)
                       else int(self._geom.num_layers))
        sink_layers = max(self._score_stride, _eff_layers)
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
    @staticmethod
    def _icms_request_id(request_id: str, num_computed_tokens: int) -> int:
        """Derive a u64 icms request_id from vLLM request_id + chunk position.

        Per Q6/D5: per-prefill-chunk namespacing so the score cache is fresh
        per chunk. For decode (num_computed_tokens doesn't change the hash
        for the same prefill chunk), subsequent layers reuse the same cache.
        """
        h = hash((request_id, num_computed_tokens)) & 0xFFFFFFFFFFFFFFFF
        return h
