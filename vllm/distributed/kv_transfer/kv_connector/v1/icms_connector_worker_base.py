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
                 local_gpu_direct: bool = False,
                 write_mode: str = "prefill"):
        # ICMS_WRITE_MODE plumb-through (PR1 of the eviction-mode
        # refactor). Facade validates mutex + auto-sets FULL_FETCH
        # before constructing the worker; the worker carries the
        # resolved value through so PR2-PR10 dispatch sites can branch
        # on it without re-reading the env. Default "prefill"
        # preserves byte-identical behavior with pre-refactor builds.
        if write_mode not in ("prefill", "eviction"):
            raise ValueError(
                f"_WorkerBase write_mode={write_mode!r} invalid; "
                "expected 'prefill' or 'eviction'.")
        self._write_mode: str = write_mode
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

        # Stash the raw __init__ param for the [INSTR-MODEL] startup dump
        # (which fires after geom resolution in _connect()). The adaptive
        # allocator divides this by tp_size at line 660-662, so we keep the
        # raw full-link value here for visibility into the original wire
        # budget regardless of whether adaptive is on.
        self._link_bandwidth_bps: float = float(link_bandwidth_bps)

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
        # ICMS_USE_SPLIT_CLIENTS=1 (landmine #1 fix, 2026-06-01): a
        # separate IcmsClient for WriteGroup so the eviction-daemon
        # writeback thread doesn't race the forward-thread Score/Fetch
        # on a shared `tx_/rx_` buffer (the C++ CallGuard aborts on
        # overlap). When OFF (default), _write_client is aliased to
        # _client and behavior is byte-identical. See
        # tests/test_landmine_1_single_client_race.py for the race
        # reproduction.
        self._write_client: IcmsClient | None = None
        self._use_split_clients = (
            os.environ.get("ICMS_USE_SPLIT_CLIENTS", "0") == "1")
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
        # PR7a of ICMS eviction-mode refactor (2026-05-31): cross-thread
        # lock for _stored_chain_groups. Under eviction mode the
        # writeback daemon thread mutates this list from the WriteGroup
        # completion callback; the forward thread reads it from
        # _get_stored_context_groups (NMT match path) and writes from
        # _drain_pending_flush_queue + legacy prefill paths. Without the
        # lock, concurrent append + scan in _record_stored_groups loses
        # updates (Plan-review BLOCKER #2). Under prefill, the lock is
        # only held by the forward thread → uncontended ~50 ns / call.
        self._stored_chain_groups_lock: threading.RLock = threading.RLock()

        # Timing / debug stats.
        self.stats = IcmsTimingStats(level=stats_level, log_selections=log_selections)

        # Pending async score results (C9).
        # Key: layer_name → dict[request_id → (reply, req_idx_in_batch)]
        self._pending_scores: dict[str, dict[str, tuple]] = {}
        self._pending_reuse: dict[str, dict[str, tuple]] = {}
        # Faithful Quest (ICMS_SCORING_MODE=faithful_quest): per-KV-head
        # selection mask, union-slot-ordered, stashed alongside _pending_scores
        # so the apply path can attach it to IcmsFetchState.head_mask.
        # Key: layer_name → dict[request_id → head_mask_pk (Tensor [H_kv, U])].
        # Empty/unused on every non-faithful path.
        self._pending_faithful_masks: dict[str, dict[str, "torch.Tensor"]] = {}
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
        # Landmine #1 split (2026-06-01): under split-clients mode the
        # write daemon uses its own IcmsClient + its own lock — so the
        # WG tx_/rx_ buffer is independent of Score's. Default-OFF →
        # alias to _rpc_lock so single-client mode is byte-identical.
        if self._use_split_clients:
            if os.environ.get("ICMS_WRITE_RPC_MUTEX", "0") == "1":
                self._write_rpc_lock = threading.Lock()
            else:
                self._write_rpc_lock = _ctxlib.nullcontext()
        else:
            self._write_rpc_lock = self._rpc_lock

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

        # Landmine #1 split (2026-06-01): build a separate IcmsClient
        # for the WriteGroup path so the daemon writeback doesn't share
        # tx_/rx_ buffers with forward-thread Score/Fetch. The write
        # client sends Hello with deployment_id=0 so the server's
        # `tp_groups_[deployment_id][tp_rank] = conn_id` clobber
        # (handlers.cc:983, guarded by `tp_size > 1 && deployment_id != 0`)
        # is skipped — Score/Fetch fan-out keeps the read client's
        # conn_id. Default-OFF aliases _write_client = _client.
        if self._use_split_clients and self._use_rdma:
            for _wattempt in range(_max_attempts):
                try:
                    self._write_client = RdmaIcmsClient(cfg)
                    self._write_client.connect()
                    with self._write_rpc_lock:
                        self._write_client.hello(
                            self._model_name,
                            tp_rank=self._tp_rank,
                            tp_size=self._tp_size,
                            # deployment_id=0 → skip server's tp_groups_
                            # fan-out registration; write client doesn't
                            # need to receive Score/Fetch RDMA writeback.
                            deployment_id=0,
                            sink_slot_count=0)
                    break
                except Exception as _e:
                    if _wattempt == _max_attempts - 1:
                        raise
                    logger.warning(
                        "[icms-connect] write-client attempt %d/%d "
                        "failed: %s (tp_rank=%d)",
                        _wattempt + 1, _max_attempts, _e, self._tp_rank)
                    _t.sleep(1.0 + 0.5 * _wattempt)
            logger.info(
                "[icms-split] write client connected (tp_rank=%d, "
                "deployment_id=0); read+write CallGuards independent",
                self._tp_rank)
        else:
            # Single-client mode: writes share the read client → behavior
            # byte-identical to pre-split code.
            self._write_client = self._client

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
        # 2026-05-29 ICMS_SW_LAYERS: bitmask of SW (sliding-window)
        # layers. The connector skips K/V serialization for them in
        # WriteGroup; server must mirror via --sw-layers. Required for
        # gemma-3 hybrid (52 SW layers) where the previous null-block
        # K/V garbage inflated WriteGroup frames to 263 MB > 128 MiB
        # wire cap. For uniform models (qwen3, mistral), leave unset
        # (mask=0 → byte-identical legacy behavior).
        _sw_spec = os.environ.get("ICMS_SW_LAYERS", "")
        _sw_mask = parse_scored_layers(_sw_spec)
        if _sw_mask != 0 and self._geom.num_layers < 64:
            _sw_mask &= ((1 << self._geom.num_layers) - 1)
        if _sw_mask != 0:
            # SW and scored masks must be disjoint (a layer can't be
            # both scored and SW).
            _bad_sw = _sw_mask & self._geom.scored_layers_mask
            if _bad_sw:
                raise ValueError(
                    f"ICMS_SW_LAYERS bitmask 0x{_sw_mask:x} overlaps "
                    f"ICMS_SCORED_LAYERS (conflict bits 0x{_bad_sw:x}). "
                    f"A layer cannot be both scored and SW.")
            self._geom = _dataclasses.replace(self._geom,
                                               sw_layers_mask=_sw_mask)
            logger.info(
                "[icms] sw_layers mask=0x%x popcount=%d "
                "(K/V write-skip enabled; %d K/V layers in WriteGroup)",
                _sw_mask, bin(_sw_mask).count("1"),
                self._geom.num_kv_layers)
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

        # [INSTR-MODEL] one-shot dump of finalized geometry + link bw so
        # the smoke can ground the transfer-overhead math on actual values
        # (instead of assumed num_layers / num_kv_heads / link_bw).
        if os.environ.get("ICMS_INSTR", "0") == "1":
            _per_rank_page_bytes = (
                int(self._geom.kv_page_bytes)
                // max(1, int(self._tp_size)))
            logger.info(
                "[INSTR-MODEL] name=%s num_layers=%d num_scored_layers=%d "
                "num_kv_heads=%d head_dim=%d elem_bytes=%d "
                "kv_page_bytes=%d kv_page_bytes_per_rank=%d tp_size=%d "
                "link_bandwidth_bps=%.3e link_GBps=%.3f",
                self._geom.name, int(self._geom.num_layers),
                int(self._geom.num_scored_layers),
                int(self._geom.num_kv_heads), int(self._geom.head_dim),
                int(self._geom.elem_bytes),
                int(self._geom.kv_page_bytes), _per_rank_page_bytes,
                int(self._tp_size), float(self._link_bandwidth_bps),
                float(self._link_bandwidth_bps) / 1e9)

        # 2026-05-29 ICMS_MAX_PAGES sink-clamp: self._k is set from
        # max_model_len / PAGE_TOKENS at __init__ to ensure the sink can
        # hold a worst-case full-ctx fetch. For shorter ctx cells (e.g.
        # mistral-small ctx=65k with max_model_len=131k) this over-
        # provisions the sink by ~2× and pushes the B-full-fetch sink to
        # 20 GB/rank — OOM at gpu_memory_utilization=0.80. ICMS_MAX_PAGES
        # lets the operator cap self._k to the actual pages exercised in
        # this run; per_layer_bytes shrinks proportionally. Unset = no-op
        # (preserves byte-identical behavior). Set per-cell by
        # run_full_ttft_sweep.sh as `ctx_max // PAGE_TOKENS + slack`.
        try:
            _max_pages_env = int(
                os.environ.get("ICMS_MAX_PAGES", "0") or 0)
        except ValueError:
            _max_pages_env = 0
        if _max_pages_env > 0 and _max_pages_env < self._k:
            logger.info(
                "[icms-max-pages] clamping self._k %d -> %d (sink "
                "per_layer_bytes drops %.1fx; check the sweep harness "
                "if you didn't expect this)",
                self._k, _max_pages_env, self._k / _max_pages_env)
            self._k = _max_pages_env

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
        # 2026-05-29 ICMS_FULL_FETCH=1 (B-full-fetch path): force the
        # sink to hold num_layers (not num_scored_layers) so the server's
        # lifted Phase-2 RDMA write loop (worker_pool.cc:1417-1429 +
        # 1797-1834 with !fetch_all_layers gate) has room for all 48
        # layers' KV. Guards:
        #   - reject hybrid SWA models (dense_layers_mask != 0):
        #     gemma-3 SW layers have undefined apply semantics under
        #     full-fetch.
        #   - clamp _sink_slots to 1: at 6× more layers, _sink_slots>1
        #     would 6× the GPU memory pressure (>60 GB/rank, infeasible).
        #     Also disables multi-rid (ICMS_ALLOW_BATCH) safely.
        _full_fetch_mode = (
            os.environ.get("ICMS_FULL_FETCH", "0") == "1")
        if _full_fetch_mode:
            if self._geom.dense_layers_mask != 0:
                # Mode-aware advice (PR1 eviction-mode refactor): under
                # ICMS_WRITE_MODE=eviction, FULL_FETCH=1 is auto-set, so
                # the "use FULL_FETCH=0" advice is wrong. Direct the
                # operator to the underlying limitation instead.
                if self._write_mode == "eviction":
                    raise RuntimeError(
                        "ICMS_WRITE_MODE=eviction is not supported "
                        "for hybrid SWA models (dense_layers_mask != 0). "
                        "Gemma-3's SW-layer eviction integration is "
                        "out of v1 eviction-mode scope. Use "
                        "ICMS_WRITE_MODE=prefill (default) for "
                        "hybrid models.")
                raise RuntimeError(
                    "ICMS_FULL_FETCH=1 is incompatible with hybrid "
                    "models (dense_layers_mask != 0). gemma-3 SW layer "
                    "apply path is undefined under full-fetch. "
                    "Use ICMS_FULL_FETCH=0 for hybrid models.")
            if self._is_multi_rid_mode() and self._write_mode != "eviction":
                # Under WRITE_MODE=eviction the sink sizing semantics are
                # different (writeback queue, not single in-flight slot)
                # so the multi-rid prohibition does NOT apply. PR4 lands
                # the writeback queue + correctly-sized depth; until
                # then keep the prohibition active for prefill mode.
                raise RuntimeError(
                    "ICMS_FULL_FETCH=1 requires single-rid mode "
                    "(max_num_seqs=1 AND ICMS_ALLOW_BATCH unset). "
                    "Multi-rid at 6× sink size would OOM the GPU.")
            # 2026-05-30: use num_kv_layers, NOT num_layers. For uniform
            # models (sw_mask=0) num_kv_layers == num_layers so this is
            # byte-identical. For SW models like gemma-3 (sw_mask!=0),
            # num_kv_layers=10 instead of 62 → sink shrinks from 62 GB
            # to 10 GB per rank. The original 2026-05-29 SW-skip refactor
            # missed this site, causing 62 GB sink + 25 GB model + 32 GB
            # vLLM KV = 119 GB > 93 GB cap → OOM at first kv_cache alloc.
            _eff_layers = int(self._geom.num_kv_layers)
            if self._sink_slots != 1:
                logger.warning(
                    "[icms-full-fetch] clamping _sink_slots %d -> 1 "
                    "to fit %d-layer sink in GPU memory",
                    self._sink_slots, _eff_layers)
                self._sink_slots = 1
            logger.info(
                "[icms-full-fetch] ACTIVE: sink sized for num_layers=%d "
                "(was num_scored=%d); _sink_slots=1",
                _eff_layers, int(self._geom.num_scored_layers))
        else:
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
        if _full_fetch_mode:
            logger.info(
                "[icms-full-fetch] sink total: %.2f GB per rank "
                "(num_layers=%d * per_layer_bytes=%d * slots=%d)",
                total_sink / (1 << 30), sink_layers, per_layer_bytes,
                self._sink_slots)
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

        # ──────────────────────────────────────────────────────────────
        # PR4 of ICMS eviction-mode refactor (2026-05-31): writeback
        # queue + dedicated CUDA stream + background drain thread.
        # ──────────────────────────────────────────────────────────────
        # Allocated ONLY when write_mode=eviction; under prefill mode
        # these attributes stay None so the legacy _WritePipeline path
        # is unaffected (PR0a fixture invariant).
        #
        # Queue capacity is sized in CHAIN units per Reviewer 3 MEDIUM:
        # per-page sizing (~1.25 MB) would drop every put under
        # realistic eviction batches. Default = N inflight chains × the
        # FULL-KV chain bytes derived from worker geometry.
        #
        # chain_bytes = icms_k × num_kv_layers × kv_page_bytes
        # where icms_k = pages_per_ctx ≈ max_model_len / PAGE_TOKENS,
        # which is the upper bound on a single chain's page count.
        # Under eviction-mode FULL_FETCH=1 is auto-set so this is the
        # right scale for the chain footprint.
        self._writeback_queue = None
        self._writeback_stream = None
        if self._write_mode == "eviction":
            from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import (  # noqa: E501
                _WriteBackQueue,
            )
            kv_page_bytes = int(self._geom.kv_page_bytes)
            chain_bytes = (int(self._k) *
                           int(self._geom.num_kv_layers) *
                           kv_page_bytes)
            inflight_chains = int(os.environ.get(
                "ICMS_WRITE_QUEUE_INFLIGHT_CHAINS", "4"))
            inflight_chains = max(1, inflight_chains)
            # Explicit override: ICMS_WRITE_QUEUE_DEPTH (bytes) wins
            # over the derived default if set.
            queue_bytes_env = os.environ.get("ICMS_WRITE_QUEUE_DEPTH")
            if queue_bytes_env:
                queue_capacity = int(queue_bytes_env)
            else:
                queue_capacity = inflight_chains * chain_bytes
            self._writeback_queue = _WriteBackQueue(
                capacity_bytes=queue_capacity,
                name=f"icms-writeback-tp{self._tp_rank}")
            # Dedicated CUDA stream for GPU→CPU eviction memcpy so the
            # extract path (PR5) doesn't serialize behind the default
            # stream's prefill ops. Stream is allocated on the same
            # device as the sink (gpu_dev resolved above).
            try:
                _dev_idx = int(gpu_dev.split(":")[1]) if gpu_dev.startswith(
                    "cuda:") else 0
                self._writeback_stream = torch.cuda.Stream(
                    device=_dev_idx)
            except Exception as _e:
                logger.warning(
                    "[icms-eviction] PR4 failed to allocate dedicated "
                    "CUDA stream: %s — falling back to default stream; "
                    "extract path will serialize with prefill compute.",
                    _e)
                self._writeback_stream = None
            logger.info(
                "[icms-eviction] PR4 writeback queue allocated: "
                "capacity=%.2f GiB (%d chains × %.2f GiB), "
                "stream=%s, drain_thread=%s",
                queue_capacity / (1 << 30),
                inflight_chains, chain_bytes / (1 << 30),
                "dedicated" if self._writeback_stream is not None
                else "default",
                self._writeback_queue._t.name)
            # PR5: extractor turns ChainLocators (from PR3 bridge) into
            # writeback-queue tasks. Allocated only under eviction mode
            # so prefill is untouched.
            from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_worker_write import (  # noqa: E501
                _EvictionExtractor,
            )
            self._eviction_extractor = _EvictionExtractor(self)
            logger.info(
                "[icms-eviction] PR5 extractor instantiated "
                "(per-group buffer accumulator → writeback queue).")
        else:
            self._eviction_extractor = None

        # PR6 of ICMS eviction-mode refactor (2026-05-31): one-shot
        # guard set so the facade.request_finished + facade.get_finished
        # double-call doesn't fire the scavenger flush twice for the
        # same rid. Allocated under BOTH modes (~56 bytes empty set —
        # eliminates a None-check branch in on_request_finished); under
        # prefill the set is allocated but never written.
        # See icms_connector_worker_state.py:on_request_finished for
        # the consumer site (Reviewer 1 MEDIUM #7).
        self._scavenger_fired_rids: set[str] = set()

        # PR7a of ICMS eviction-mode refactor (2026-05-31): chain
        # snapshots for finished-before-evict resolution. Eviction
        # callbacks for a rid can fire many steps after the rid
        # finishes (BlockLocator's snapshot retention bounds this to
        # max_age_steps). When the worker's process_locators needs
        # rid → chain for the WriteGroup RPC, self._requests[rid] may
        # already be gone — so we snapshot rs.chain at
        # on_request_finished time, keyed by rid, with the finish step.
        # Pruned by _maybe_prune_chain_snapshots once per step against
        # ICMS_EVICTION_CHAIN_SNAPSHOT_MAX_AGE_STEPS (default 200,
        # matching BlockLocator's snapshot prune cadence).
        #
        # Allocated under BOTH modes (empty dict ≈ 64 B) so the
        # eviction-only-write branch in on_request_finished doesn't
        # need a None-check. Under prefill the dict is never written.
        self._chain_snapshots: dict[str, tuple[list[int], int]] = {}

        # PR7b of ICMS eviction-mode refactor (2026-05-31): async cross-
        # turn signal state. All three structures are LOCK-PROTECTED by
        # self._pr7b_lock because the writeback daemon thread mutates
        # _completed_chains_lru + _pending_finished_recving on RPC
        # completion, while the engine/forward thread reads them in
        # on_step_start and drains _pending_finished_recving in
        # get_finished. RLock for re-entrant safety (logging callbacks
        # may take the lock while a logger format string evaluates).
        #
        # _waiters_by_chain: chain_tuple → set of waiter rids the
        #   scheduler ferried via metadata.chain_waiters. Drained by
        #   the daemon thread when the chain's last group RPC
        #   succeeds. NOT consulted by get_finished — completed-chain
        #   waiters are pushed to _pending_finished_recving immediately.
        #
        # _completed_chains_lru: OrderedDict of chain_tuple → None,
        #   tracking chains whose writeback has fully completed (last
        #   group RPC succeeded). LRU-bounded to ICMS_PR7B_COMPLETED_LRU
        #   entries (default 4096); when full, the oldest entry is
        #   evicted via popitem(last=False). Consulted by on_step_start
        #   when new waiters arrive via metadata — a waiter whose
        #   chain is already completed gets immediately routed to
        #   _pending_finished_recving (closes the race where worker
        #   completes BEFORE scheduler-side waiter ferry arrives).
        #
        # _pending_finished_recving: set of rids to return in the next
        #   get_finished call's finished_recving slot.
        from collections import OrderedDict
        self._waiters_by_chain: dict[tuple[int, ...], set[str]] = {}
        self._completed_chains_lru: OrderedDict[
            tuple[int, ...], None] = OrderedDict()
        self._pending_finished_recving: set[str] = set()
        self._pr7b_lock: threading.RLock = threading.RLock()
        try:
            self._pr7b_completed_lru_cap: int = int(
                os.environ.get("ICMS_PR7B_COMPLETED_LRU", "4096"))
        except ValueError:
            self._pr7b_completed_lru_cap = 4096
        # Telemetry — surfaced in shutdown stats dump.
        self._pr7b_stats = {
            "waiters_received_total": 0,
            "waiters_resolved_immediate_total": 0,  # arrived after completion
            "waiters_resolved_on_completion_total": 0,  # arrived before
            "chains_completed_total": 0,
            "completed_lru_evictions_total": 0,
        }
        # Step counter, bumped once per on_step_start. Used by
        # chain-snapshot prune-by-age. Increment is harmless under
        # prefill (counter just walks forward, no consumer).
        self._step_counter: int = 0
        # Last prune step (so prune fires at most once per K steps).
        self._last_chain_snapshot_prune_step: int = 0
        try:
            self._chain_snapshot_max_age_steps: int = int(
                os.environ.get(
                    "ICMS_EVICTION_CHAIN_SNAPSHOT_MAX_AGE_STEPS", "200"))
        except ValueError:
            self._chain_snapshot_max_age_steps = 200

    def compute_eviction_drain_timeout_s(self, pending_bytes: int,
                                          override: "float | None",
                                          base_s: float,
                                          bw_gb_per_s: float,
                                          floor_s: float,
                                          ceiling_s: float) -> float:
        """PR7a per-chain-size derivation of drain timeout.

        timeout = max(floor, min(ceiling,
                                  base + pending_bytes / link_bw_bytes_per_s))
        with override hard-clamping to a fixed value if set.
        Returns seconds.

        Reviewer 2 HIGH #6: 30s default was too aggressive for h128k
        multi-GB writes (e.g., 24 GiB pending @ 12 GB/s = 2s; +5s base
        = 7s minimum but if h128k unlocks 48 GiB then 9s/etc., still
        well within ceiling). Floor ensures we don't drop below 10s
        even for tiny pending.
        """
        if override is not None and override > 0.0:
            return float(override)
        if bw_gb_per_s <= 0.0:
            bw_gb_per_s = 12.0
        bw_bytes_per_s = bw_gb_per_s * (1024.0 ** 3)
        derived = base_s + float(pending_bytes) / bw_bytes_per_s
        return max(floor_s, min(ceiling_s, derived))

    def pr7b_ingest_chain_waiters(
        self, chain_waiters: "list[tuple[tuple[int, ...], list[str]]]"
    ) -> None:
        """PR7b: process new waiters ferried from the scheduler.

        Called from worker's on_step_start AFTER process_locators (so
        chains completed THIS step have already been added to the LRU).

        For each (chain, [waiter_rids]) entry:
          - If the chain is already in _completed_chains_lru, route the
            waiter_rids straight to _pending_finished_recving (race
            window where worker completed BEFORE the scheduler ferried
            the waiter). Get_finished returns them next call.
          - Else stash in _waiters_by_chain — the daemon thread will
            drain on chain completion.

        Lock-protected: contends with the daemon thread's completion
        handler that mutates the same structures.
        """
        if not chain_waiters:
            return
        with self._pr7b_lock:
            for chain_tuple, waiter_rids in chain_waiters:
                if not chain_tuple or not waiter_rids:
                    continue
                self._pr7b_stats[
                    "waiters_received_total"] += len(waiter_rids)
                if chain_tuple in self._completed_chains_lru:
                    # Race: chain completed before this waiter arrived.
                    # Mark waiter_rids as ready immediately.
                    self._pending_finished_recving.update(waiter_rids)
                    self._pr7b_stats[
                        "waiters_resolved_immediate_total"] += len(
                            waiter_rids)
                else:
                    self._waiters_by_chain.setdefault(
                        chain_tuple, set()).update(waiter_rids)

    def pr7b_signal_chain_completed(
        self, chain_tuple: tuple[int, ...]
    ) -> None:
        """PR7b: called from the daemon thread's _flush_group task
        closure on RPC success of the LAST group in the chain
        (gidx == len(chain) - 1).

        Updates _completed_chains_lru (with bounded eviction) and
        drains _waiters_by_chain[chain_tuple] into
        _pending_finished_recving so get_finished returns them.

        Idempotent on re-signaling for the same chain (LRU touch).
        Lock-protected.
        """
        with self._pr7b_lock:
            # LRU touch / insert.
            if chain_tuple in self._completed_chains_lru:
                self._completed_chains_lru.move_to_end(chain_tuple)
            else:
                self._completed_chains_lru[chain_tuple] = None
                self._pr7b_stats["chains_completed_total"] += 1
                # Bound the LRU.
                while (len(self._completed_chains_lru)
                        > self._pr7b_completed_lru_cap):
                    self._completed_chains_lru.popitem(last=False)
                    self._pr7b_stats[
                        "completed_lru_evictions_total"] += 1
            # Drain any waiters that were already registered.
            waiters = self._waiters_by_chain.pop(chain_tuple, None)
            if waiters:
                self._pending_finished_recving.update(waiters)
                self._pr7b_stats[
                    "waiters_resolved_on_completion_total"] += len(
                        waiters)

    def drain_pending_finished_recving(self) -> "set[str]":
        """PR7b: called from facade.get_finished on the worker side.

        Snapshots and clears _pending_finished_recving under the lock.
        The returned set is passed as the second element of
        get_finished's tuple → KVConnectorOutput.finished_recving →
        consumed by vLLM core's _update_from_kv_xfer_finished.

        Returns an empty set under prefill mode (set never populated).
        """
        with self._pr7b_lock:
            if not self._pending_finished_recving:
                return set()
            drained = self._pending_finished_recving
            self._pending_finished_recving = set()
            return drained

    def _chain_for_rid(self, request_id: str):
        """PR7a: resolve rid → trie chain for the WriteGroup RPC.

        Eviction-mode locators may arrive for a rid whose request state
        was popped many steps ago. Lookup priority:
        1. self._requests[rid].chain — live rid still in flight.
        2. self._chain_snapshots[rid] — finished but within max_age.

        Returns None if neither is available (orphan locator). The
        caller (extractor _flush_group) MUST handle None by dropping
        the locator and incrementing a counter — it's the contract.
        """
        rs = self._requests.get(request_id)
        if rs is not None and rs.chain:
            return rs.chain
        snap = self._chain_snapshots.get(request_id)
        if snap is not None:
            return snap[0]
        return None

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
        # PR6 of ICMS eviction-mode refactor (2026-05-31): bound memory
        # at long-run-end by clearing the one-shot scavenger guard set.
        try:
            getattr(self, "_scavenger_fired_rids", set()).clear()
        except Exception:
            pass
        # PR7a: clear chain snapshots at shutdown.
        try:
            getattr(self, "_chain_snapshots", {}).clear()
        except Exception:
            pass
        # PR7b: clear async-signal state at shutdown.
        try:
            with self._pr7b_lock:
                self._waiters_by_chain.clear()
                self._completed_chains_lru.clear()
                self._pending_finished_recving.clear()
        except Exception:
            pass
        # PR4 of eviction-mode refactor: drain + shutdown the writeback
        # queue (if it was allocated). Under prefill mode this is a
        # no-op since the queue was never instantiated. Snapshot the
        # queue's stats BEFORE shutdown so the stats dump below (PR6
        # telemetry surface) sees the final state — drops_total +
        # pending_bytes at shutdown are the operator's stuck-BF2
        # canary per Reviewer 2 HIGH #5.
        wbq = getattr(self, "_writeback_queue", None)
        _wbq_stats: dict | None = None
        if wbq is not None:
            try:
                _wbq_stats = {
                    "capacity_bytes": wbq.capacity_bytes,
                    "pending_bytes_at_shutdown": wbq.pending_bytes,
                    "puts_total": wbq.puts_total,
                    "drops_total": wbq.drops_total,
                    "bytes_put_total": wbq.bytes_put_total,
                    "puts_high_total": wbq.puts_high_total,
                    "puts_low_total": wbq.puts_low_total,
                }
            except Exception:
                _wbq_stats = None
            # PR7a per-chain-size drain timeout. Pulls parameters from
            # the facade attributes set in __init__ (defaults wired
            # in icms_connector_facade.py:208ish). Falls back to a
            # conservative 30s if the facade attributes aren't reachable
            # (only happens in unit tests that mock worker_base.shutdown).
            try:
                _facade_obj = getattr(self, "_facade", None) or self
                _override = getattr(
                    _facade_obj,
                    "_eviction_drain_timeout_s_override", None)
                _base = getattr(
                    _facade_obj,
                    "_eviction_drain_timeout_base_s", 5.0)
                _bw = getattr(
                    _facade_obj,
                    "_eviction_drain_bw_gb_per_s", 12.0)
                _floor = getattr(
                    _facade_obj,
                    "_eviction_drain_timeout_floor_s", 10.0)
                _ceiling = getattr(
                    _facade_obj,
                    "_eviction_drain_timeout_ceiling_s", 600.0)
                _pending = int(getattr(wbq, "pending_bytes", 0))
                _timeout_s = self.compute_eviction_drain_timeout_s(
                    pending_bytes=_pending,
                    override=_override,
                    base_s=_base,
                    bw_gb_per_s=_bw,
                    floor_s=_floor,
                    ceiling_s=_ceiling)
                logger.info(
                    "[icms-eviction] PR7a derived drain timeout: "
                    "%.1f s (pending=%d B, base=%.1f, bw=%.1f GB/s, "
                    "floor=%.1f, ceiling=%.1f, override=%s)",
                    _timeout_s, _pending, _base, _bw, _floor, _ceiling,
                    "None" if _override is None else f"{_override:.1f}")
            except Exception:
                _timeout_s = 30.0
                logger.warning(
                    "[icms-eviction] PR7a drain timeout derivation "
                    "failed; falling back to 30.0 s",
                    exc_info=True)
            try:
                wbq.shutdown(timeout=_timeout_s)
            except Exception:
                logger.exception(
                    "[icms-eviction] writeback queue shutdown failed")
            self._writeback_queue = None
        # PR6: snapshot extractor stats too (for the stats dump below).
        _evx_stats: dict | None = None
        evx = getattr(self, "_eviction_extractor", None)
        if evx is not None:
            try:
                _evx_stats = {
                    "pages_received": evx.pages_received,
                    "groups_completed": evx.groups_completed,
                    "groups_dropped_writeback_full": (
                        evx.groups_dropped_writeback_full),
                    "scavenger_calls_total": evx.scavenger_calls_total,
                    "groups_flushed_on_finish": evx.groups_flushed_on_finish,
                    "groups_dropped_partial_on_finish": (
                        evx.groups_dropped_partial_on_finish),
                    "puts_high_total": evx.puts_high_total,
                    "puts_low_total": evx.puts_low_total,
                    "pages_at_snapshot_step": evx.pages_at_snapshot_step,
                    # PR7a counters — monotonic-gidx gate + RPC outcomes.
                    "out_of_order_drops_total": getattr(
                        evx, "out_of_order_drops_total", 0),
                    "orphan_chain_drops_total": getattr(
                        evx, "orphan_chain_drops_total", 0),
                    "write_group_rpc_successes_total": getattr(
                        evx, "write_group_rpc_successes_total", 0),
                    "write_group_rpc_failures_total": getattr(
                        evx, "write_group_rpc_failures_total", 0),
                }
            except Exception:
                _evx_stats = None
        # PR7b worker counters (live on _WorkerBase, not the extractor).
        _pr7b_worker_stats: dict | None = None
        try:
            _pr7b_worker_stats = {
                **getattr(self, "_pr7b_stats", {}),
                "waiters_by_chain_live": len(
                    getattr(self, "_waiters_by_chain", {})),
                "completed_chains_lru_size": len(
                    getattr(self, "_completed_chains_lru", {})),
                "pending_finished_recving_at_shutdown": len(
                    getattr(self, "_pending_finished_recving", set())),
            }
        except Exception:
            _pr7b_worker_stats = None
        # Dump connector-side timing stats to a file before teardown.
        if self.stats.level > 0 or self.stats.log_selections:
            try:
                import json
                pid = os.getpid()
                stats_path = f"/tmp/icms_connector_stats_{pid}.json"
                _payload = self.stats.to_dict()
                # PR6 telemetry surface (PR12 prerequisite per Reviewer
                # 2 MEDIUM #11). Always present even if extractor/queue
                # were never allocated (None → omitted via if-guard).
                if _evx_stats is not None:
                    _payload["eviction_extractor"] = _evx_stats
                if _wbq_stats is not None:
                    _payload["writeback_queue"] = _wbq_stats
                # PR7b telemetry surface.
                if _pr7b_worker_stats is not None:
                    _payload["pr7b_worker"] = _pr7b_worker_stats
                with open(stats_path, "w") as f:
                    json.dump(_payload, f, indent=2)
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
            # Landmine #1 split: close the write client first (no sinks,
            # cheap) then read client. When unsplit, _write_client IS
            # _client → skip the redundant close.
            if (self._use_split_clients
                    and self._write_client is not None
                    and self._write_client is not self._client):
                try:
                    self._write_client.close()
                except Exception:
                    pass
                self._write_client = None
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
