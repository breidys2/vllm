# SPDX-License-Identifier: Apache-2.0
"""ICMS connector _Worker mixin: _WorkerWritePipelineMixin.

Extracted verbatim from icms_connector.py (behavior-preserving split).
Methods reference self.* attributes set by _WorkerBase.__init__ and call
sibling-mixin methods via the _Worker MRO; imports resolve from the neutral
helper modules so there is no cycle back into icms_connector.
"""
from __future__ import annotations

from icms_client.geometry import KvLayout
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import _GroupBuffer
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _ICMS_FULLTRACE_ENABLED
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _ICMS_TRACE_ENABLED
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_types import _RequestState
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _append_stored_chain_queue
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _get_icms_nccl_group
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_chain_fp
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_fulltrace
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_trace import _icms_trace
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_allreduce_max_int
from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector_tp import _tp_broadcast_bool_list
from vllm.distributed.kv_transfer.kv_connector.v1 import icms_connector_trace as _trace
import os
import threading
import time
import torch
from vllm.logger import init_logger

# Log under the original connector logger name (behavior-preserving
# split): all split modules share one logger so log-name filtering,
# grep, and assertLogs see the same name as before.
logger = init_logger("vllm.distributed.kv_transfer.kv_connector.v1.icms_connector")
from icms_client.geometry import GROUP_PAGES  # noqa: E402
_GROUP_BLOCKS = GROUP_PAGES


# ═══════════════════════════════════════════════════════════════════════
# PR5 of ICMS eviction-mode refactor: per-rid/per-group page accumulator
# that feeds the writeback queue (PR4).
#
# Receives ChainLocator tuples from the scheduler→worker bridge (PR3)
# in on_step_start. For each tuple (rid, group_idx, page_in_group):
#   1. Mark the page filled in the per-(rid, group) _GroupBuffer.
#   2. When the buffer is complete (all GROUP_PAGES pages across all
#      kv layers filled), enqueue a WriteGroup task to the writeback
#      queue (PR4) and clear the buffer.
#
# PR5 STAGING NOTE: the actual GPU→CPU memcpy of KV bytes from vLLM's
# pool + the WriteGroup RPC dispatch are STUBBED here (the task just
# logs + counts). PR6 lands the real memcpy on the dedicated writeback
# stream + the WriteGroup RPC dispatch. This staging lets PR5 prove
# the wire (PR2 callback → PR3 bridge → PR4 queue) end-to-end without
# changing the actual bytes-to-BF2 picture, so the closed-loop bench
# remains byte-identical under prefill mode (the queue is never
# allocated under prefill) and eviction mode produces no garbage KV
# (no writes leave the worker process).
# ═══════════════════════════════════════════════════════════════════════


class _EvictionExtractor:
    """Per-(rid, group_idx) buffer accumulator for eviction-mode KV
    writeback. Owned by _WorkerBase under WRITE_MODE=eviction.

    Single-producer (engine thread, via on_step_start) +
    single-consumer (the writeback queue's drain thread). The internal
    state is mutated only from the engine thread; the drain thread
    only reads the task closures we enqueued.
    """

    def __init__(self, worker):
        self._worker = worker  # _WorkerBase ref
        # Per-(rid, group_idx) → _GroupBuffer
        self._buffers: dict[tuple[str, int], _GroupBuffer] = {}
        # Telemetry — surfaces in PR12.
        self.pages_received: int = 0
        self.groups_completed: int = 0
        self.groups_dropped_writeback_full: int = 0
        self.orphan_locators_total: int = 0  # PR3 already counts these
        self.pages_at_snapshot_step: int = 0
        # PR6 of ICMS eviction-mode refactor (2026-05-31): scavenger
        # lifecycle counters. NO _rid_priority dict — both reviewers
        # demanded counter-only telemetry to avoid the lifecycle
        # duality / leak class that a per-rid priority map would carry.
        self.groups_flushed_on_finish: int = 0
        self.groups_dropped_partial_on_finish: int = 0
        self.scavenger_calls_total: int = 0
        self.puts_high_total: int = 0
        self.puts_low_total: int = 0
        # PR7a: monotonic-gidx push gate. Per-chain (tuple of trie node
        # ids) → max gidx already pushed. Server's trie.extend requires
        # the parent chain to exist; pushing gidx=K without 0..K-1 first
        # would either fail (if extend can't find parent) or fall back
        # to insert(full_chain) which advertises chain[0..K-1] in the
        # ledger without their KV → cross-turn fetches return garbage.
        # Out-of-order group flushes are DROPPED + counted, not
        # buffered, for PR7a simplicity. A future PR can add buffering
        # if the dropped rate is non-trivial under eviction pressure.
        self._chain_max_pushed: dict[tuple, int] = {}
        self.out_of_order_drops_total: int = 0
        self.orphan_chain_drops_total: int = 0
        self.write_group_rpc_failures_total: int = 0
        self.write_group_rpc_successes_total: int = 0
        # Landmine #2 fix (2026-06-01): protect `self._buffers` against
        # the cross-thread access that DEFER_CPU mode introduces.
        # Forward thread does pass-1 staging (write); writeback daemon
        # does pass-2 CPU finalize (write) + pass-3 flush (read+pop).
        # RLock because _flush_group → _record_stored_groups can
        # re-enter via the same code path.
        self._buffers_lock = threading.RLock()
        # Counters for the new code paths (surface in shutdown dump):
        self.finalize_inline_count: int = 0      # legacy + SINGLE_SYNC
        self.finalize_deferred_count: int = 0    # DEFER_CPU submitted
        self.finalize_submit_failures_total: int = 0  # queue overflow

    def flush_remaining_for_rid(self, rid: str,
                                chain: "list[int] | None"
                                ) -> tuple[int, int]:
        """PR6 scavenger fire-and-forget: Reviewer 3 Option a (round-DOWN).

        Enqueue all COMPLETE buffered groups for `rid` via the writeback
        queue with priority='low'; DROP any incomplete groups. Also pop
        every (rid, *) key so dead-rid late-eviction callbacks (which
        could fire seconds after rid finishes via BlockLocator snapshot
        retention) cannot orphan bytes in self._buffers (Reviewer 1
        HIGH #5 fix). Non-blocking: every put_or_drop call goes through
        the bounded queue's lock for ~microseconds; the heavy work runs
        on the daemon thread (icms-writeback-tp{N}).

        `chain` is captured by the engine-thread caller BEFORE the
        worker pops self._requests[rid]; passing it in means the daemon
        thread never reads-back state that may have already been GC'd
        (Reviewer 1 HIGH #4 fix).

        Returns (n_flushed, n_dropped).
        """
        self.scavenger_calls_total += 1
        flushed = 0
        dropped = 0
        # Landmine #2 review finding #4 (2026-06-01): take
        # `_buffers_lock` so the scavenger doesn't race the writeback
        # daemon's `_finalize_cpu_per_layer` under DEFER_CPU mode.
        # Pre-refactor this was single-threaded so the lock was a
        # no-op; under DEFER_CPU the daemon writes into
        # `_buffers[buf_key].kv_blob` concurrently. The atomic
        # capture-keys + pop-and-flush block must be a critical
        # section against both finalize and flush-completed.
        with self._buffers_lock:
            keys = [k for k in self._buffers.keys() if k[0] == rid]
            for key in keys:
                buf = self._buffers.pop(key, None)
                if buf is None:
                    continue
                _, group_idx = key
                if buf.is_complete():
                    self._flush_group(rid, group_idx, buf,
                                      chain=chain, priority='low')
                    flushed += 1
                else:
                    dropped += 1
                    self.groups_dropped_partial_on_finish += 1
        self.groups_flushed_on_finish += flushed
        return flushed, dropped

    def _new_buffer(self) -> _GroupBuffer:
        geom = self._worker._geom
        num_kv_layers = int(geom.num_kv_layers)
        # PR7a fix (2026-05-31): summary_blob is sized by
        # num_scored_layers, NOT num_kv_layers. The server's
        # handle_write_group reads `K * num_scored * summary_group_bytes`
        # bytes of summary then `K * num_kv * kv_group_bytes` bytes of
        # KV at known offsets (handlers.cc:1241-1247). PR5 stub never
        # wrote real summary bytes so the wrong sizing was invisible;
        # PR7a's real RPC ships the over-sized summary, which pushes
        # the kv_blob_base offset on the server side INTO our summary
        # region → server reads garbage as KV → kError "WriteGroup
        # failed" on every flush.
        #
        # Matches prefill mode's record_page buffer sizing
        # (worker_write.py:1458-1462) byte-for-byte.
        num_scored_layers = int(
            getattr(geom, "num_scored_layers", num_kv_layers))
        kv_page_bytes = int(geom.kv_page_bytes)
        kv_group_bytes = int(getattr(
            geom, "kv_group_bytes",
            kv_page_bytes * GROUP_PAGES))
        summary_page_bytes = int(geom.summary_page_bytes)
        summary_group_bytes = int(getattr(
            geom, "summary_group_bytes",
            summary_page_bytes * GROUP_PAGES))
        return _GroupBuffer(
            summary_blob=bytearray(
                num_scored_layers * summary_group_bytes),
            kv_blob=bytearray(
                num_kv_layers * kv_group_bytes),
            filled=set(),
            num_layers=num_kv_layers,
            pages_in_group=GROUP_PAGES,
        )

    def process_locators(
        self,
        locators: list[tuple[int, str, int, int, int]],
    ) -> int:
        """Record this step's evicted ChainLocators into per-group
        buffers; do batched per-layer GPU→CPU memcpy + summary
        computation; flush any group that becomes complete.

        Tuple shape (PR7a 5-tuple):
          (block_id, rid, group_idx, page_in_group, snapshot_step)

        Returns the count of completed groups flushed this call.

        PR7a of ICMS eviction-mode refactor (2026-05-31):
        replaces PR5/PR6 stub. The flow:

        1. Stage locators into per-(rid, group) buffers, building a
           reverse map block_id → list of (buffer, page_in_group)
           targets so multiple locators sharing a block_id (rare —
           PR0b's evict_lookup pops, so duplicates only arise if the
           same block_id appears for different rids in one step) are
           handled by one gather.

        2. For each KV layer L:
             a. Gather K and V slices via tensor indexing —
                `k_cache[block_id_tensor]` produces one batched
                [N, block_size, num_kv_heads_local, head_dim] tensor.
                ONE indexed gather per layer per step (replaces the
                per-page loop that was 10k × 48 = 480k tiny memcpys
                in plan-review BLOCKER #3).
             b. TP>1 AllGather to recombine per-rank head shards into
                full-head bytes — same shape the prefill path writes
                so eviction-written chains read back correctly via
                the existing apply path. Routed through the ICMS
                sep-NCCL group when enabled to avoid colliding with
                vLLM's forward-pass NCCL on the main TP group.
             c. .to('cpu') on the writeback CUDA stream (PR4); sync
                once after the gather so the bytes are safely on host.
             d. For scored layers: compute per-page (kmin, kmax)
                summary over the block_size axis, fp16-cast.
             e. Scatter K||V bytes + summary bytes into each target
                buffer's blob at the correct offset (matches
                record_page's offset math).

        3. After all layers, flush any groups that became complete.

        Timing window invariant (Plan-review BLOCKER #1 fix in
        scheduler.py): drain_eviction_callbacks runs BEFORE
        build_connector_meta, so locators for blocks freed this step
        ferry in THIS step's metadata. on_step_start (and thus
        process_locators) runs in start_load_kv BEFORE the forward
        pass writes new KV — so reading `kv_cache[block_id]` here
        returns the OLD owner's K/V bytes, even if vLLM has already
        reassigned the block to a new rid in this step's allocate.

        Landmine #2 fix (2026-06-01): two independent optimizations,
        gated and default-OFF (byte-identical):
          • ICMS_EVICTION_SINGLE_SYNC=1 — schedule all per-layer GPU
            ops first, sync the writeback stream ONCE at the end
            instead of N times. Still on forward thread but layers
            can overlap on the stream → ~Nx fewer sync points.
          • ICMS_EVICTION_DEFER_CPU=1 — implies SINGLE_SYNC; also
            moves the CPU summary + scatter + flush to the writeback
            daemon. Forward thread schedules GPU work, records a
            CUDA event, queues `main_stream.wait_event(event)` (no
            CPU sync), submits a daemon closure, returns. Cross-
            stream safety: main stream's NEXT kernel waits for our
            gather to complete before vLLM can reassign blocks.
        """
        if not locators:
            return 0
        worker = self._worker
        # Worker may not be fully ready (geom registered after first
        # bind_connector_metadata call). Defer until ready.
        if (not getattr(worker, "_gpu_kv_caches", None)
                or getattr(worker, "_geom", None) is None):
            return 0

        defer_cpu = (
            os.environ.get("ICMS_EVICTION_DEFER_CPU", "0") == "1")
        single_sync = (
            defer_cpu
            or os.environ.get("ICMS_EVICTION_SINGLE_SYNC", "0") == "1")

        # Pass 1: forward thread, CPU bookkeeping (cheap).
        staged = self._stage_buffers(locators)
        if staged is None:
            return 0

        # Pass 2 GPU: per-layer gather + AllGather + .to('cpu').
        # In legacy mode each layer syncs the writeback stream before
        # the next iteration so the CPU summary below sees fresh bytes.
        # In SINGLE_SYNC / DEFER_CPU modes the per-layer sync is
        # skipped; the caller handles the final sync (or wait_event).
        per_layer = self._schedule_gpu_per_layer(
            staged, per_layer_sync=not single_sync)

        if defer_cpu:
            # Off the forward thread entirely.
            self._submit_finalize_to_daemon(per_layer, staged)
            self.finalize_deferred_count += 1
            return 0  # daemon flushes asynchronously; caller doesn't see it

        if single_sync:
            # Replace N per-layer syncs with one at the end.
            ws = getattr(worker, "_writeback_stream", None)
            if ws is not None:
                ws.synchronize()

        # Inline finalize (legacy and SINGLE_SYNC paths).
        with self._buffers_lock:
            self._finalize_cpu_per_layer(per_layer, staged)
            self.finalize_inline_count += 1
            return self._flush_completed_buffers()

    # ──────────────────── helpers (landmine #2 refactor) ──────────

    def _stage_buffers(self, locators):
        """Pass 1: stage locators → per-(rid, gidx) buffers + reverse
        map. Returns a dict-of-dicts shaped staged record, or None
        when no work to do.

        Forward thread only (cheap CPU bookkeeping). Mutates
        self._buffers — caller takes `_buffers_lock` if running
        from a thread other than the forward thread (DEFER_CPU
        daemon path takes the lock only at finalize time; this
        method is always called from the forward thread).
        """
        per_buffer_targets: dict[
            tuple[str, int], list[tuple[int, int]]] = {}
        block_id_to_targets: dict[
            int, list[tuple[tuple[str, int], int]]] = {}
        for (block_id, rid, group_idx, page_in_group, snapshot_step
             ) in locators:
            self.pages_received += 1
            if snapshot_step != 0:
                self.pages_at_snapshot_step += 1
            key = (rid, group_idx)
            with self._buffers_lock:
                if key not in self._buffers:
                    self._buffers[key] = self._new_buffer()
            per_buffer_targets.setdefault(key, []).append(
                (int(block_id), int(page_in_group)))
            block_id_to_targets.setdefault(int(block_id), []).append(
                (key, int(page_in_group)))

        block_id_list = sorted(block_id_to_targets.keys())
        if not block_id_list:
            return None

        return {
            "per_buffer_targets": per_buffer_targets,
            "block_id_to_targets": block_id_to_targets,
            "block_id_list": block_id_list,
            "bid_to_pos": {bid: i for i, bid in enumerate(block_id_list)},
        }

    def _schedule_gpu_per_layer(self, staged, *, per_layer_sync: bool):
        """Pass 2 GPU: per-layer index_select + (TP>1 AllGather) +
        .to('cpu'). Returns a list of per-layer state tuples that the
        CPU finalize step consumes.

        When `per_layer_sync=False` the writeback stream is NOT
        synchronized between layers — caller must either synchronize
        once at the end (SINGLE_SYNC mode) or rely on a CUDA event +
        wait_event for cross-stream coherence (DEFER_CPU mode).

        Tuple shape: (layer_idx, k_cpu, v_cpu, is_scored, kv_rank,
                      scored_rank_or_None)
        k_cpu/v_cpu are CPU tensors (host memory). When the stream is
        a real CUDA stream and per_layer_sync=False, the host memory
        is NOT valid until the corresponding event fires — caller
        must enforce ordering.
        """
        worker = self._worker
        geom = worker._geom
        block_id_list = staged["block_id_list"]
        block_id_tensor = torch.tensor(block_id_list, dtype=torch.long)
        tp_size = int(getattr(worker, "_tp_size", 1) or 1)
        writeback_stream = getattr(worker, "_writeback_stream", None)

        per_layer_state: list[tuple] = []
        for layer_name, kv_layer in list(worker._gpu_kv_caches.items()):
            if kv_layer is None or kv_layer.ndim != 5:
                continue
            layer_idx = worker._parse_layer_idx(layer_name)
            if layer_idx is None or layer_idx >= geom.num_layers:
                continue
            if geom.is_sw(layer_idx):
                # SW layers carry no on-disk KV in the chain.
                continue

            # K/V layout dispatch (matches prefill extract_and_record).
            if kv_layer.shape[0] == 2:
                k_cache = kv_layer[0]
                v_cache = kv_layer[1]
            elif kv_layer.shape[1] == 2:
                k_cache = kv_layer[:, 0]
                v_cache = kv_layer[:, 1]
            else:
                continue

            bidx = block_id_tensor.to(k_cache.device, non_blocking=True)
            try:
                if writeback_stream is not None:
                    with torch.cuda.stream(writeback_stream):
                        k_gathered = k_cache.index_select(0, bidx).contiguous()
                        v_gathered = v_cache.index_select(0, bidx).contiguous()
                else:
                    k_gathered = k_cache.index_select(0, bidx).contiguous()
                    v_gathered = v_cache.index_select(0, bidx).contiguous()
            except Exception:
                logger.exception(
                    "[icms-eviction] PR7a per-layer gather failed for "
                    "layer=%s — skipping; orphan blocks counted",
                    layer_name)
                self.orphan_locators_total += len(block_id_list)
                continue

            # ── TP>1 AllGather to reconstruct full-head bytes. ──
            #
            # all_gather is a COLLECTIVE: every rank must participate for
            # the SAME layer or the ranks desync (a rank that skips a
            # layer while peers still wait on that layer's collective
            # hangs / corrupts the gathered K/V — 2026-05-10 audit #4).
            # So failures are split into two classes:
            #   (A) recoverable SETUP failures (buffer-alloc OOM, group
            #       lookup) — these can be asymmetric, so we vote on them
            #       via a tiny symmetric all-reduce and skip the layer on
            #       ALL ranks together (graceful degradation, balanced).
            #   (C) a failure of the all_gather collective ITSELF — the
            #       communicator is then in an undefined state with no
            #       safe per-rank recovery, so it propagates (FATAL), per
            #       the all_gather-must-be-fatal invariant. NOT swallowed.
            if tp_size > 1:
                import torch.distributed as dist  # noqa: E402
                from vllm.distributed.parallel_state import get_tp_group
                # Phase A: per-rank setup that may fail asymmetrically.
                # Capture the outcome locally — do NOT `continue` here, or
                # a one-rank skip would unbalance Phase C's collective.
                setup_ok = True
                gk = gv = dev_group = None
                try:
                    tp_group = get_tp_group()
                    dev_group = (_get_icms_nccl_group()
                                  or tp_group.device_group)
                    gk = [torch.empty_like(k_gathered)
                          for _ in range(tp_size)]
                    gv = [torch.empty_like(v_gathered)
                          for _ in range(tp_size)]
                except Exception:
                    logger.exception(
                        "[icms-eviction] PR7a AllGather setup failed for "
                        "layer=%s tp_size=%d — voting to skip this layer "
                        "on all ranks", layer_name, tp_size)
                    setup_ok = False
                # Phase B: symmetric barrier. All-reduce-MAX the local
                # failure bit on a fixed 1-element int — robust (unlike
                # the variable-shape K/V gather it can't fail from a
                # per-layer alloc), and EVERY rank reaches it regardless
                # of its own setup outcome, so the skip decision is
                # identical on all ranks. (If the group itself is dead,
                # this degrades to the rank-local value and Phase C's
                # all_gather then fails fatally — never silent.)
                any_rank_failed = _tp_allreduce_max_int(
                    0 if setup_ok else 1, tp_size)
                if any_rank_failed:
                    if setup_ok:
                        logger.warning(
                            "[icms-eviction] PR7a AllGather: a peer rank "
                            "failed setup for layer=%s — skipping on all "
                            "ranks (symmetric)", layer_name)
                    self.orphan_locators_total += len(block_id_list)
                    continue
                # Phase C: every rank prepared OK → the collective is
                # balanced. A raise here is FATAL by design (no try).
                if writeback_stream is not None:
                    with torch.cuda.stream(writeback_stream):
                        dist.all_gather(gk, k_gathered, group=dev_group)
                        dist.all_gather(gv, v_gathered, group=dev_group)
                        k_full = torch.cat(gk, dim=2)
                        v_full = torch.cat(gv, dim=2)
                else:
                    dist.all_gather(gk, k_gathered, group=dev_group)
                    dist.all_gather(gv, v_gathered, group=dev_group)
                    k_full = torch.cat(gk, dim=2)
                    v_full = torch.cat(gv, dim=2)
            else:
                k_full = k_gathered
                v_full = v_gathered

            if writeback_stream is not None:
                with torch.cuda.stream(writeback_stream):
                    k_cpu = k_full.to('cpu', non_blocking=True)
                    v_cpu = v_full.to('cpu', non_blocking=True)
                if per_layer_sync:
                    writeback_stream.synchronize()
            else:
                k_cpu = k_full.cpu()
                v_cpu = v_full.cpu()

            is_scored = geom.is_scored(layer_idx)
            scored_rank = geom.scored_rank(layer_idx) if is_scored else None
            kv_rank = geom.kv_layer_rank(layer_idx)
            per_layer_state.append(
                (int(layer_idx), k_cpu, v_cpu, is_scored,
                 int(kv_rank),
                 int(scored_rank) if scored_rank is not None else None))
        return per_layer_state

    def _finalize_cpu_per_layer(self, per_layer_state, staged):
        """Pass 2 CPU: summary computation + scatter into per-(rid,
        group) buffers. Caller MUST hold self._buffers_lock.

        When called from the writeback daemon (DEFER_CPU mode) the
        host memory backing k_cpu / v_cpu becomes valid only after
        the corresponding CUDA event fires — caller must have
        event.synchronize()'d before invoking this.
        """
        worker = self._worker
        geom = worker._geom
        spb = geom.summary_page_bytes
        kpb = geom.kv_page_bytes
        kv_group_bytes = geom.kv_group_bytes
        summary_group_bytes = geom.summary_group_bytes
        kv_layout = geom.kv_layout
        num_kv_layers_total = geom.num_kv_layers
        block_id_to_targets = staged["block_id_to_targets"]
        bid_to_pos = staged["bid_to_pos"]

        for (layer_idx, k_cpu, v_cpu, is_scored, kv_rank, scored_rank
             ) in per_layer_state:
            if is_scored:
                keys = k_cpu.to(dtype=torch.float32)
                keys = keys.reshape(keys.shape[0], keys.shape[1], -1)
                kmins = keys.min(dim=1).values.to(torch.float16)
                kmaxs = keys.max(dim=1).values.to(torch.float16)

            for block_id, targets in block_id_to_targets.items():
                pos = bid_to_pos[block_id]
                k_bytes = (k_cpu[pos].contiguous().view(torch.uint8)
                            .numpy().tobytes())
                v_bytes = (v_cpu[pos].contiguous().view(torch.uint8)
                            .numpy().tobytes())
                kv_bytes = k_bytes + v_bytes
                if is_scored:
                    summary_bytes = (
                        kmins[pos].numpy().tobytes()
                        + kmaxs[pos].numpy().tobytes())
                for (buf_key, page_in_group) in targets:
                    buf = self._buffers[buf_key]
                    if kv_layout == KvLayout.LAYER_MAJOR:
                        k_off = (kv_rank * kv_group_bytes
                                  + page_in_group * kpb)
                    else:
                        k_off = (page_in_group * num_kv_layers_total * kpb
                                  + kv_rank * kpb)
                    buf.kv_blob[k_off:k_off + len(kv_bytes)] = kv_bytes
                    if is_scored:
                        s_off = (scored_rank * summary_group_bytes
                                  + page_in_group * spb)
                        buf.summary_blob[
                            s_off:s_off + len(summary_bytes)] = summary_bytes
                    buf.filled.add((int(layer_idx), int(page_in_group)))

    def _flush_completed_buffers(self) -> int:
        """Pass 3: flush every buffer that has reached completeness.
        Caller MUST hold self._buffers_lock.

        Landmine #2 review finding #7 (2026-06-01): iterate in
        (rid, group_idx) ASCENDING order so the per-chain monotonic
        gate in `_flush_group` (`_chain_max_pushed` at write.py:99)
        sees groups in trie-extend order. Pre-refactor, multiple
        groups completing in a single call were rare (one per call
        typically) so dict-insertion-order was fine. Under
        DEFER_CPU's wider race window (multi-step accumulation
        before daemon drains) several gidx for the same chain can
        complete together; out-of-order iteration would trip the
        monotonic-gidx gate → out_of_order_drops_total + lost data.
        """
        flushed = 0
        for key in sorted(self._buffers.keys()):
            buf = self._buffers[key]
            if buf.is_complete():
                rid, group_idx = key
                self._flush_group(rid, group_idx, buf)
                del self._buffers[key]
                flushed += 1
        return flushed

    def _submit_finalize_to_daemon(self, per_layer_state, staged):
        """DEFER_CPU path: synchronize the writeback stream on the
        FORWARD thread (one sync; Win A), then submit a daemon closure
        that does the CPU summary + scatter + flush (Win B).

        SAFETY rationale (review finding 2026-06-01): the original
        design used `torch.cuda.current_stream().wait_event(event)` to
        queue a CUDA-side wait on the model-forward stream so vLLM
        couldn't reassign freed blocks before our gather landed. That
        was unsound under vLLM's CUDA-graph mode (and any backend that
        runs forward kernels on a non-default stream) — `current_stream()`
        at on_step_start time is NOT guaranteed to be the same stream
        vLLM later uses for model.forward(). Wrong stream → silent KV
        corruption.

        Safer design: synchronize the writeback stream on the forward
        thread BEFORE returning. Adds one ~1ms sync to the forward
        path (Win A already collapses N→1 — net cost unchanged). The
        CPU summary + scatter + flush still moves off the forward
        thread (Win B preserved). The daemon's `event.synchronize()`
        becomes a no-op when the event has already fired, but we
        still record + carry it for telemetry/debugging.

        Falls back to inline execution on:
          • writeback stream sync raise (CUDA failure) → inline
          • event creation raise → inline (NOT silent skip — review #6)
          • queue submit fail → inline
        """
        worker = self._worker
        writeback_stream = getattr(worker, "_writeback_stream", None)
        extractor = self

        # Defensive sync on the FORWARD thread before returning. After
        # this returns, all gathered K/V is on the CPU and vLLM is free
        # to reassign blocks. No race with vLLM's main forward stream
        # regardless of its stream config.
        sync_failed = False
        if writeback_stream is not None:
            try:
                writeback_stream.synchronize()
            except Exception:
                logger.exception(
                    "[icms-eviction] landmine #2 writeback sync failed "
                    "— defer-cpu degraded to inline")
                sync_failed = True

        if sync_failed:
            # Inline finalize — bytes are safe (no daemon waiting).
            with extractor._buffers_lock:
                extractor._finalize_cpu_per_layer(per_layer_state, staged)
                extractor._flush_completed_buffers()
            return

        def _finalize_task():
            try:
                with extractor._buffers_lock:
                    extractor._finalize_cpu_per_layer(
                        per_layer_state, staged)
                    extractor._flush_completed_buffers()
            except KeyError as e:
                # Review finding #9 (2026-06-01): scavenger
                # (flush_remaining_for_rid) may have popped the
                # `buf_key` between submit and daemon run. The pages
                # for that buffer are lost — count it as an orphan
                # rather than silently swallowing.
                self.orphan_locators_total += 1
                logger.warning(
                    "[icms-eviction] landmine #2 deferred finalize "
                    "hit KeyError (scavenger race): %s — bytes for "
                    "this buffer lost; orphan_locators_total bumped",
                    e)
            except Exception:
                logger.exception(
                    "[icms-eviction] landmine #2 deferred finalize "
                    "failed; data for this step's evicted locators "
                    "may be lost")

        # payload_bytes=1 so `_writeback_queue.drain_all()` actually
        # waits for this closure to run (queue is bytes-bounded; a
        # 0-byte task would let drain_all return before the closure
        # starts). Real KV bytes are accounted by `_flush_group`'s
        # own put_or_drop. SHUTDOWN GOTCHA (review #3): drain_all
        # semantics still work because once the finalize task runs it
        # synchronously submits the WriteGroup tasks (with real byte
        # counts) BEFORE its own byte count is released. The queue's
        # _pending_bytes never reaches 0 until both finalize AND the
        # WriteGroup tasks it spawned have completed.
        ok = worker._writeback_queue.put_or_drop(
            rid="<finalize>",
            payload_bytes=1,
            task_fn=_finalize_task,
            priority='high',
        )
        if not ok:
            self.finalize_submit_failures_total += 1
            # Fallback: run inline. Correctness over latency.
            _finalize_task()

    def _flush_group(self, rid: str, group_idx: int,
                     buf: _GroupBuffer,
                     chain: "list[int] | None" = None,
                     priority: str = 'high') -> None:
        """Enqueue a writeback task for one complete group.

        PR7a of ICMS eviction-mode refactor (2026-05-31): dispatches
        a real `self._worker._write_client.write_group(...)` RPC via
        the writeback queue. Replaces the PR5/PR6 stub closure that
        only logged. (Landmine #1 fix 2026-06-01: under
        ICMS_USE_SPLIT_CLIENTS=1, _write_client is a separate
        IcmsClient instance with its own CallGuard atomic so the
        daemon RPC cannot abort/stall the forward-thread Score.)

        Monotonic-gidx gate: only group_idx == _chain_max_pushed + 1
        is dispatched. Out-of-order groups are dropped + counted (no
        buffering in PR7a). Already-pushed groups (group_idx <=
        max_pushed) are idempotent skips. The gate prevents
        advertising chain[0..K-1] as stored when only chain[K] has
        the actual bytes on disk (which would corrupt cross-turn
        fetches).

        Chain resolution priority: explicit `chain` argument (eager
        capture from scavenger), then `self._worker._chain_for_rid`
        (live _requests[rid].chain or _chain_snapshots[rid]). None
        on both → orphan, dropped + counted.

        TP>1: all ranks update _stored_chain_groups symmetrically;
        only rank 0 dispatches the RPC. Failure asymmetry (rank 0
        RPC fails, rank>0 still bumps) is a small known risk;
        production BF2 is reliable enough this hasn't shown up in
        prefill mode's identical pattern.

        `priority` is observational only — counter increment, no FIFO
        order change (per PR6 / Reviewer 1 LOW).
        """
        # Resolve chain: explicit param wins, else worker lookup.
        if chain is None or len(chain) == 0:
            chain = self._worker._chain_for_rid(rid)
        if chain is None or group_idx >= len(chain):
            self.orphan_chain_drops_total += 1
            logger.debug(
                "[icms-eviction] PR7a orphan chain rid=%s gidx=%d "
                "(chain=%s)", rid, group_idx,
                "None" if chain is None else f"len={len(chain)}")
            return

        # Monotonic-gidx gate.
        chain_tuple = tuple(chain)
        cur_max = self._chain_max_pushed.get(chain_tuple, -1)
        if group_idx <= cur_max:
            # Already pushed (idempotent skip — e.g., the same chain
            # is shared across rids and another rid pushed it first).
            logger.debug(
                "[icms-eviction] PR7a idempotent-skip rid=%s gidx=%d "
                "max=%d", rid, group_idx, cur_max)
            return
        if group_idx > cur_max + 1:
            # Out-of-order: drop + count.
            self.out_of_order_drops_total += 1
            logger.debug(
                "[icms-eviction] PR7a out-of-order rid=%s gidx=%d "
                "expected=%d", rid, group_idx, cur_max + 1)
            return

        # Capture closure locals BEFORE handing off to daemon thread.
        # Tuple-of-int for chain; bytes for blobs (frozen copies).
        worker = self._worker
        rid_local = rid
        group_local = int(group_idx)
        parent_chain_local = list(chain[:group_local])
        new_tail_local = [chain[group_local]]
        chain_prefix_local = list(chain[:group_local + 1])
        n_groups_stored_local = int(group_local + 1)
        # bytes() copy detaches from the bytearray that lives in
        # self._buffers — important because the buffer dict pops the
        # entry after _flush_group returns (in process_locators) but
        # the daemon thread may still be holding the bytes.
        summary_blob_bytes = bytes(buf.summary_blob)
        kv_blob_bytes = bytes(buf.kv_blob)
        payload_bytes = len(kv_blob_bytes) + len(summary_blob_bytes)
        tp_rank = int(getattr(worker, "_tp_rank", 0) or 0)
        tp_size = int(getattr(worker, "_tp_size", 1) or 1)

        # Reserve max_pushed slot eagerly so a concurrent flush for
        # the next gidx sees cur_max=group_local (not the prior
        # value). The daemon thread completion will keep this as-is.
        self._chain_max_pushed[chain_tuple] = group_local

        def _task():
            rpc_ok = True
            if tp_size <= 1 or tp_rank == 0:
                try:
                    # Landmine #1 split: WG goes through _write_client +
                    # _write_rpc_lock. Default (split OFF) → both alias
                    # the read client / its lock → byte-identical.
                    with worker._write_rpc_lock:
                        worker._write_client.write_group(
                            parent_chain_local, new_tail_local,
                            summary_blob_bytes, kv_blob_bytes,
                            pages_in_group=GROUP_PAGES,
                        )
                except Exception:
                    rpc_ok = False
                    self.write_group_rpc_failures_total += 1
                    logger.warning(
                        "[icms-eviction] PR7a WriteGroup RPC failed "
                        "for rid=%s gidx=%d (chain_head=%s)",
                        rid_local, group_local,
                        parent_chain_local[
                            :min(4, len(parent_chain_local))],
                        exc_info=True)
            if rpc_ok:
                self.write_group_rpc_successes_total += 1
                # Advertise stored prefix so NMT can find the chain on
                # the next request. Both ranks update their per-rank
                # ledger; rank>0 didn't dispatch but the ledger still
                # needs to be symmetric for NMT determinism.
                try:
                    worker._record_stored_groups(
                        chain_prefix_local, n_groups_stored_local)
                    _append_stored_chain_queue(
                        chain_prefix_local, n_groups_stored_local)
                except Exception:
                    logger.exception(
                        "[icms-eviction] PR7a stored-chain ledger "
                        "update failed rid=%s gidx=%d",
                        rid_local, group_local)
                # PR7b of ICMS eviction-mode refactor (2026-05-31):
                # if this push completes the chain (gidx is the last
                # group index), signal the worker so any waiters
                # registered for this chain get drained into
                # _pending_finished_recving. get_finished returns
                # them as finished_recving → vLLM unblocks the
                # waiter rids from WAITING_FOR_REMOTE_KVS.
                #
                # `chain_tuple` here is the FULL producer chain, NOT
                # the prefix up to gidx — waiters were registered
                # against the full producer chain in NMT (matched as
                # a prefix of the waiter's own chain). Signal the
                # producer chain so waiter set lookup succeeds.
                if group_local == len(chain_tuple) - 1:
                    try:
                        worker.pr7b_signal_chain_completed(chain_tuple)
                    except Exception:
                        logger.exception(
                            "[icms-pr7b] signal_chain_completed "
                            "failed for chain head=%s",
                            chain_tuple[:min(4, len(chain_tuple))])

        ok = worker._writeback_queue.put_or_drop(
            rid, payload_bytes, _task, priority=priority)
        if ok:
            self.groups_completed += 1
            if priority == 'high':
                self.puts_high_total += 1
            else:
                self.puts_low_total += 1
        else:
            # Queue full — back out the reservation since we won't
            # actually push this group.
            if self._chain_max_pushed.get(chain_tuple) == group_local:
                self._chain_max_pushed[chain_tuple] = cur_max
            self.groups_dropped_writeback_full += 1
            logger.warning(
                "[icms-eviction] writeback queue FULL — dropping group "
                "rid=%s group=%d size=%d B priority=%s "
                "(drops_total=%d)",
                rid, group_idx, payload_bytes, priority,
                self._worker._writeback_queue.drops_total)


class _WorkerWritePipelineMixin:
    def _drain_pending_flush_queue(self):
        """Forward-thread drain of pipeline-thread WriteGroup ok-bits.

        Runs `_tp_broadcast_bool` on each enqueued (rid, group_idx,
        ok_local, partial, pages, chain_prefix) tuple, then applies the
        symmetric ledger bumps + `_record_stored_groups` push that the
        old inline path in `_flush_group` did. Called from
        `wait_for_pending_writes` BEFORE the memcpy gate so that all
        pipeline-thread NCCL is now on this (forward) thread → no
        same-rank concurrent NCCL with iter N+1's model.forward.

        Order is preserved per-rid because the pipeline thread is a
        single worker that flushes groups in monotonic gidx order; this
        drain processes them in FIFO order.

        2026-05-10 TP>1 asymmetric-queue fix:
        Pipeline progress is async per-rank, so at any given drain
        point one rank's `_pending_flush_q` may be shorter than the
        other's. Pre-fix, each rank called `_tp_broadcast_bool` once
        per local entry → asymmetric NCCL collective counts → CUDA
        hang. Post-fix, all-reduce-MAX the local count first; every
        rank then issues exactly `n` broadcasts, padding the local
        queue with sentinel entries (rid="", group_idx=-1) when
        local < n. The sentinel broadcasts use ok_local=True so the
        broadcast no-ops on rank-0; bumps are skipped via the
        sentinel check inside the loop. Pipeline catches up in
        future drains where the missed flush appears as a real entry.
        """
        # 1. Snapshot local queue (under lock).
        with self._pending_flush_lock:
            entries = self._pending_flush_q
            self._pending_flush_q = []
        # 2026-05-10 audit #11: sort by (rid, group_idx) so per-rank
        # ordering is canonical at the drain. Pre-fix, the queue
        # reflected pipeline-thread completion order which could
        # legitimately differ across ranks (rank-0 enqueues [A:0, B:0]
        # while rank-1 enqueues [B:0, A:0]) — broadcasting only
        # `ok_local` at lockstep i then meant rank-0's `ok_A` was
        # applied to rank-1's B (wrong-rid bump). Sort ensures both
        # ranks see the same entry at the same slot when they share
        # entries; combined with the sentinel-padding below this also
        # keeps lockstep when one rank has an entry the other doesn't
        # (the missing-side gets a sentinel at the would-be slot for
        # entries it doesn't have, and the present-side's bump lands
        # only on the rank that has the entry).
        #
        # Known residual limitation: if both ranks have non-empty
        # queues with strictly different entries (e.g., rank-0 only
        # has A, rank-1 only has B), sort+pad lines them up by sort
        # key, not by rid — broadcasts can still cross. The pipeline
        # producer is single-threaded per rank and driven by the same
        # ICMS server completion order, so production divergence here
        # is transient. A future tuple-broadcast (rid_hash, gidx, ok)
        # would close this fully; flagged in the audit doc.
        _instr_dpfq = os.environ.get("ICMS_INSTR", "0") == "1"
        _t_dpfq0 = time.perf_counter() if _instr_dpfq else 0.0
        entries = sorted(entries, key=lambda e: (str(e[0]), int(e[1])))
        if _instr_dpfq:
            _t_dpfq_sort = time.perf_counter()
        local_n = len(entries)
        # 3. Symmetrize across TP ranks. At TP=1 this is a no-op and
        #    falls through to the legacy fast-path (early-return on 0).
        n = _tp_allreduce_max_int(local_n, self._tp_size)
        if _instr_dpfq:
            _t_dpfq_ar = time.perf_counter()
        if n == 0:
            if _instr_dpfq:
                logger.info(
                    "[INSTR-DPFQ] n=0 local_n=%d sort_us=%.1f ar_us=%.1f",
                    local_n,
                    (_t_dpfq_sort - _t_dpfq0) * 1e6,
                    (_t_dpfq_ar - _t_dpfq_sort) * 1e6)
            return
        # 4. Pad local queue with sentinels if shorter than the global
        #    max. Sentinel `rid=""` + `group_idx=-1` is recognized
        #    inside the loop and skips the per-rid bump (no rs to find).
        if local_n < n:
            sentinel = ("", -1, True, False, 0, ())
            entries = list(entries) + [sentinel] * (n - local_n)
        # Broadcasts + bumps run lock-free — bumps mutate per-rid state
        # only, and rs is single-threaded (forward thread or
        # pipeline-thread-during-pipeline-task) so no contention here.
        # 2026-05-28: batch the N per-entry broadcasts into ONE collective.
        # Pre-fix each iter cost ~1.3 s here at qwen3 ctx=65k pf=4096
        # (128 entries × per-entry .item() drain of default CUDA stream).
        # Single-broadcast bulk packs all ok_locals into one int64 tensor,
        # broadcasts once, unpacks via one .tolist() — one drain instead
        # of N. Semantically identical: each entry's `ok` is still
        # rank-0's `ok_local` for that entry's slot (same source rank,
        # same payload mapping).
        ok_locals = [bool(e[2]) for e in entries]
        if _instr_dpfq:
            _t_dpfq_pre_bcast = time.perf_counter()
        oks_batch = _tp_broadcast_bool_list(
            ok_locals, self._tp_rank, self._tp_size)
        if _instr_dpfq:
            _t_dpfq_post_bcast = time.perf_counter()
        for (rid, group_idx, ok_local, partial, pages, chain_prefix), ok in zip(
                entries, oks_batch):
            # Sentinel entry from the symmetrize-padding above: keep
            # the broadcast (collective participation is the whole
            # reason it's there) but skip the bump path.
            if rid == "" and group_idx == -1:
                continue
            if ok:
                # Mirror the inline path's success-trace + bumps. Gating
                # on `ok` (broadcast result, not `ok_local`) preserves
                # Design B symmetry — all ranks bump or all skip.
                if _ICMS_TRACE_ENABLED:
                    try:
                        _wg_pids = [int(group_idx) * _GROUP_BLOCKS + i
                                    for i in range(_GROUP_BLOCKS)][:16]
                        _icms_trace(
                            "writegroup_commit", rid, layer=-1,
                            chain_fp=_icms_chain_fp(list(chain_prefix)),
                            extra={
                                "group_idx": int(group_idx),
                                "page_ids_in_group": _wg_pids,
                                "success": True,
                                "partial": bool(partial),
                                "pages": int(pages),
                                "deferred_broadcast": True,
                            })
                    except Exception:
                        pass
                self.stats.total_groups_written += 1
                rs = self._requests.get(rid)
                if rs is not None:
                    if group_idx >= rs.num_groups_written:
                        rs.num_groups_written = group_idx + 1
                    if group_idx >= rs.flushed_local and not partial:
                        _chain_len = (len(rs.chain) if rs.chain
                                      else group_idx + 1)
                        rs.flushed_local = min(
                            group_idx + 1, _chain_len)
                        # 2026-05-10 audit #5: bump flush_seq under the
                        # Condition lock + notify_all. Any waiter in
                        # `_score_one_request` that captured a smaller
                        # `_flush_seq_at_attempt` will see the new value
                        # and return from wait_for immediately —
                        # including a waiter that hadn't reached
                        # `wait_for` yet when this notify fired
                        # (the lost-wakeup case bare Event.set+clear
                        # had).
                        with rs.flush_cond:
                            rs.flush_seq += 1
                            rs.flush_cond.notify_all()
                        if rs.chain:
                            self._record_stored_groups(
                                rs.chain, rs.flushed_local)
                            _append_stored_chain_queue(
                                list(rs.chain), rs.flushed_local)
        if _instr_dpfq:
            _t_dpfq_end = time.perf_counter()
            logger.info(
                "[INSTR-DPFQ] n=%d local_n=%d sort_us=%.1f ar_us=%.1f "
                "pre_bcast_us=%.1f bcast_us=%.1f loop_us=%.1f total_us=%.1f",
                n, local_n,
                (_t_dpfq_sort - _t_dpfq0) * 1e6,
                (_t_dpfq_ar - _t_dpfq_sort) * 1e6,
                (_t_dpfq_pre_bcast - _t_dpfq_ar) * 1e6,
                (_t_dpfq_post_bcast - _t_dpfq_pre_bcast) * 1e6,
                (_t_dpfq_end - _t_dpfq_post_bcast) * 1e6,
                (_t_dpfq_end - _t_dpfq0) * 1e6)
    def wait_for_pending_writes(self):
        """Kick off deferred write work; DO NOT block on it.

        vLLM calls this at the end of a forward pass via wait_for_save;
        that call sits on the TTFT critical path. We must return fast:
        snapshot what the extract + flush work needs, submit a single
        closure to the background pipeline, and return.

        The drain (i.e., synchronous wait for the pipeline to finish
        this request's writes) happens in on_request_finished, which
        runs AFTER first-token emission and is off the TTFT critical
        path.

        Legacy inline behavior is still invoked when pipeline is absent
        (shouldn't happen in normal operation).
        """
        t_save_enter = time.perf_counter()
        self._ttft_emit(t_save_enter)
        self._slack_emit_and_reset()
        _instr_on = os.environ.get("ICMS_INSTR", "0") == "1"
        # 2026-05-28: ICMS_SKIP_WAIT_FOR_SAVE=1 makes this whole method a
        # no-op (after the cheap _ttft/slack emit). The deferred extract
        # + WriteGroup pipeline still runs on the background thread; the
        # per-rid drain happens in on_request_finished (which fires
        # AFTER first-token emission, off the TTFT critical path).
        # Safe for single-rid bench at TP>1 only: in batched mode the
        # cross-rid memcpy race re-emerges and in multi-iter prefill
        # bench the per-rid drain at on_request_finished still serializes
        # iter N's writes vs iter N+1's reads on the BF2 link.
        if (os.environ.get("ICMS_SKIP_WAIT_FOR_SAVE", "0") == "1"
                and not self._is_multi_rid_mode()):
            if _instr_on:
                logger.info("[INSTR] wait_for_pending_writes SKIPPED: %.2fms",
                            (time.perf_counter() - t_save_enter) * 1000.0)
            return
        # 2026-05-09 N2 deferral: drain any pending WriteGroup ok-bits
        # from the previous iter's pipeline-thread flushes BEFORE the
        # memcpy gate. This runs `_tp_broadcast_bool` on the forward
        # thread (the only legal place for collective NCCL on the TP
        # comm) and applies the symmetric ledger bumps. Closes the
        # TP>1 pipeline-thread NCCL collision the audit identified.
        _t_drain0 = time.perf_counter()
        self._drain_pending_flush_queue()
        if _instr_on:
            logger.info("[INSTR] drain_pending_flush_queue: %.2fms",
                        (time.perf_counter() - _t_drain0) * 1000.0)
        if not (self._gpu_kv_caches and self._attn_metadata is not None
                and self._requests and not self._skip_extract):
            # Nothing to extract — still flip the prefill_done flag.
            if not self._prefill_done:
                self._reset_apply_caches_for_prefill_done()
                self._prefill_done = True
                logger.info("Prefill done. Switching to dense decode.")
            return

        # M4-A all-dense gate: once every active request has flipped to
        # dense_mode, the bitmap is saturated for each rs — no downstream
        # Score will reference future decode-token KV. Skip the extract +
        # flush closure (and the BLOCK_WRITES drain) entirely. Removes
        # the ~33 ms decode-iter spikes at every 16-token page boundary
        # (extract_and_record GPU→CPU + record_page × num_layers +
        # WriteGroup RPC). Decode-token write-back is parked pending a
        # multi-turn-reuse harness (see project_icms_decode_m3_status).
        if (self._prefill_done
                and all(getattr(rs, "dense_mode", False)
                        for rs in self._requests.values())):
            return

        # Snapshot references that the background task needs. The KV
        # cache tensors and request state are expected to remain valid
        # until on_request_finished drains the pipeline.
        snap_caches = dict(self._gpu_kv_caches)
        snap_meta   = self._attn_metadata
        snap_rids   = list(self._requests.keys())

        # 2026-05-09 multi-rid batched-mode KV-corruption fix:
        # Without gating, the pipeline thread's GPU→CPU memcpy races
        # against vLLM freeing the prefill's physical pages and the
        # NEXT request's apply WRITING ICMS-fetched KV into those same
        # pages. Result: deferred-extract reads corrupted bytes (the
        # next rid's apply data), ships them to ICMS as "this rid's
        # archived KV," and all future scoring on this rid's chain
        # returns garbage → repetitive/junk model output.
        #
        # Fix: signal a per-call memcpy_done event AFTER all
        # extract_and_record calls complete (i.e., after every .cpu()
        # has returned), and block this forward thread on that event
        # before letting vLLM proceed to free pages. The WriteGroup
        # RPCs + summary compute keep running async on the pipeline
        # thread — they don't touch GPU pages anymore.
        #
        # 2026-05-28: default ON only in multi-rid (batched) mode where
        # the cross-rid race can actually occur. In single-rid mode
        # (max_num_seqs=1 and ICMS_ALLOW_BATCH=0) only one request's
        # apply can ever be in flight at a time, so freeing the prior
        # prefill's pages can't be racing the pipeline thread's memcpy.
        # The instrumented smoke (2026-05-28) measured this gate at
        # ~460 ms/iter on the TTFT critical path at qwen3 TP=2 ctx=2k
        # pf=4096 — pure overhead in single-rid mode.
        # ICMS_GATE_MEMCPY=1 still forces the gate ON regardless;
        # ICMS_GATE_MEMCPY=0 still forces it OFF (debug only — in
        # batched mode this reintroduces the corruption).
        _gate_default = "1" if self._is_multi_rid_mode() else "0"
        _gate_memcpy = os.environ.get("ICMS_GATE_MEMCPY", _gate_default) == "1"
        memcpy_done = threading.Event() if _gate_memcpy else None

        def _task():
            self._do_deferred_extract_and_flush(
                snap_caches, snap_meta, snap_rids,
                memcpy_done_event=memcpy_done)

        # PR6 of ICMS eviction-mode refactor (2026-05-31): under
        # eviction mode the prefill-style deferred-extract pipeline
        # must NOT be submitted. Eviction writes are exclusively driven
        # by the block_pool eviction callback (PR2) → ChainLocator
        # (PR3) → _EvictionExtractor (PR5) → writeback queue (PR4).
        # If we also submit the legacy _task here, the partial-flush
        # loop in on_request_finished
        # (icms_connector_worker_state.py:553-568) will see
        # rs.active_group_buffers populated and fire legacy
        # _flush_group (which performs NCCL allgather + flushed_local
        # accounting), racing with the eviction extractor and
        # producing dual-write paths. Reviewer 1 MEDIUM #8 + Reviewer
        # 2 HIGH #3 + HIGH #13 all demand this gate in PR6 (not PR7).
        # The prefill branch is BYTE-IDENTICAL — same submit call.
        if getattr(self, "_write_mode", "prefill") != "eviction":
            self._write_pipeline.submit(_task, tag="wait_for_save",
                                         rids=snap_rids)
        else:
            # No-op: eviction mode delegates writes to the eviction
            # extractor. The memcpy_done gate below is also irrelevant
            # under eviction (no GPU→CPU staging happens here), but
            # leaving it lets the wait_for_save signature stay
            # mode-independent — the event is never set under eviction
            # which is fine because no PR5 code waits on it.
            logger.debug(
                "[icms-eviction] wait_for_save: legacy submit skipped "
                "(eviction mode); writes via _EvictionExtractor.")

        # Block the forward thread until the memcpy stage of the task
        # completes. This is the page-safety gate: vLLM cannot free the
        # prefill's KV blocks until wait_for_save returns, so once we
        # pass this point the GPU pages have been fully drained to
        # host memory and are safe to release. Bounded by a generous
        # timeout so a stuck pipeline never permanently hangs the
        # forward thread (the timeout falls back to the legacy "free
        # eagerly" behavior, accepting the corruption risk over a hang).
        #
        # PR6 of ICMS eviction-mode refactor (2026-05-31): skip the
        # memcpy_done.wait() under eviction mode too — the event would
        # NEVER be set because the legacy _task that calls
        # memcpy_done.set() was gated out above. Under eviction mode,
        # GPU page safety is enforced differently: vLLM's block_pool
        # only fires the eviction callback (PR2) AFTER ref_cnt drops
        # to 0, by which time the request that owned the page is done
        # with prefill — so the page-safety contract is preserved by
        # the eviction-source choice in PR2, not by a synchronous
        # gate here. Without this gate-skip, wait_for_save deadlocks
        # the engine for ICMS_GATE_MEMCPY_TIMEOUT_S=120s per forward.
        if memcpy_done is not None and (
                getattr(self, "_write_mode", "prefill") != "eviction"):
            try:
                _memcpy_timeout_s = float(os.environ.get(
                    "ICMS_GATE_MEMCPY_TIMEOUT_S", "120.0"))
            except ValueError:
                _memcpy_timeout_s = 120.0
            _t_memcpy0 = time.perf_counter()
            if not memcpy_done.wait(timeout=_memcpy_timeout_s):
                logger.warning(
                    "ICMS memcpy gate timed out (%.1fs) — falling "
                    "through; GPU pages may be freed before extract "
                    "completes (cross-rid KV corruption risk).",
                    _memcpy_timeout_s)
            if _instr_on:
                logger.info("[INSTR] memcpy_done.wait: %.2fms",
                            (time.perf_counter() - _t_memcpy0) * 1000.0)

        # Optional measurement mode: block until this prefill's writes
        # complete before returning. Default behavior leaves writes async
        # in the background, which is faster but means iter N's writes
        # can race iter N+1's fetches on the same RDMA link, polluting
        # slack measurements (per-layer fetch arrival times shift later
        # under contention). With ICMS_BLOCK_WRITES=1, all writes are on
        # the current prefill's TTFT critical path — the next prefill
        # begins with an idle link.
        #
        # Fix H (2026-04-29): at TP>1, default to blocking. The pipeline
        # thread does an AllGather on the TP NCCL group; if the next
        # iter's main-thread forward starts before the AllGather + .cpu()
        # sync completes, both threads hammer the same NCCL group
        # concurrently and deadlock at TP=2 (worker stuck in
        # extract_and_record's .cpu(), engine core's shm_broadcast times
        # out at 60s). Set ICMS_BLOCK_WRITES=0 to override.
        # 2026-05-08 race-audit follow-up (Step 1 of "C"): batched mode
        # at TP=1 (max_num_seqs > 1, ICMS_ALLOW_BATCH=1) hits the same
        # write/read ordering race as TP>1 — iter N's write_pipeline can
        # lag iter N+1's main-thread Score, and the connector's retry-
        # on-ENOENT silently falls through to dense (no flush_local
        # progress means Score reads from a chain prefix the server
        # hasn't seen yet). Empirical evidence: llama3 vt 32K batched
        # b=0.05 with BLOCK_WRITES=0 → 0.260 acc; BLOCK_WRITES=1 → 0.600.
        # Extend the default-on gate to batched mode regardless of TP
        # so paper accuracy runs are correct out of the box. The
        # ICMS_BLOCK_WRITES env knob still overrides if the user wants
        # raw async perf at the known correctness cost.
        # 2026-05-09 (Item C audit): consolidated to _is_multi_rid_mode
        # for consistency with the other batched-mode gates. The
        # logical-OR semantics are identical to the earlier inline
        # expression; helper just centralizes the policy.
        # 2026-05-09: BLOCK_WRITES default policy after introducing the
        # memcpy-done gate (above):
        #   * TP=1 batched: gate alone is sufficient. Section 1
        #     (extract+memcpy) is awaited by the gate; Section 2
        #     (_flush_group → WriteGroup RPC) doesn't touch GPU pages
        #     and stays async. Default OFF.
        #   * TP>1: Section 2's `_tp_broadcast_bool` is an NCCL op on
        #     the TP communicator. After the gate releases the main
        #     thread, iter N+1's forward starts — its NCCL ops on the
        #     same rank's same communicator collide with the pipeline
        #     thread's Section 2 NCCL → CUDA hang → execute_model RPC
        #     timeout. Until _tp_broadcast_bool is moved to the forward
        #     thread, keep BLOCK_WRITES default ON for TP>1 to drain
        #     Section 2 before returning. (TP=1 single-rid is also OFF
        #     since there's no batching contention either.)
        # BLOCK_WRITES=1 is honored as an opt-in fallback even at TP=1
        # (useful for diagnosing async-RPC-related vs memcpy-related
        # symptoms; mirrors what BLOCK_WRITES=1 did pre-fix).
        _batched = self._is_multi_rid_mode()
        _block_default = "1" if self._tp_size > 1 else "0"
        if os.environ.get("ICMS_BLOCK_WRITES", _block_default) == "1":
            # Drain timeout configurable (2026-05-01). Default raised from
            # 60s → 180s: Llama-3.1-8B (32 layers × 8 KV heads × 128k ctx)
            # generates 16+ GiB of WriteGroup traffic per prefill, which
            # at the link's effective ~2-4 GiB/s takes well over 60s. Pre-
            # fix the drain timed out, returned without finishing the
            # pipeline AllGather, and the next iter's main-thread forward
            # collided with the still-running pipeline thread on the same
            # NCCL group → shm_broadcast hang → sample_tokens RPC timeout
            # → EngineDeadError. Mirrors the existing 90 s timeout in
            # on_request_finished but with more headroom for larger
            # models. Set ICMS_BLOCK_WRITES_DRAIN_TIMEOUT_S to override.
            try:
                _drain_timeout_s = float(os.environ.get(
                    "ICMS_BLOCK_WRITES_DRAIN_TIMEOUT_S", "180.0"))
            except ValueError:
                _drain_timeout_s = 180.0
            _t_blockw0 = time.perf_counter()
            try:
                drained = self._write_pipeline.drain(timeout=_drain_timeout_s)
                if not drained:
                    logger.warning(
                        "ICMS_BLOCK_WRITES drain timed out (%.1fs)",
                        _drain_timeout_s)
            except Exception:
                logger.exception("ICMS_BLOCK_WRITES drain failed")
            if _instr_on:
                logger.info("[INSTR] BLOCK_WRITES drain: %.2fms",
                            (time.perf_counter() - _t_blockw0) * 1000.0)

        if not self._prefill_done:
            self._reset_apply_caches_for_prefill_done()
            self._prefill_done = True
            logger.info("Prefill done. Switching to dense decode.")
        if _instr_on:
            logger.info("[INSTR] wait_for_pending_writes TOTAL: %.2fms",
                        (time.perf_counter() - t_save_enter) * 1000.0)
    def _do_deferred_extract_and_flush(
        self,
        caches: dict,
        meta,
        rids: list[str],
        memcpy_done_event: "threading.Event | None" = None,
    ):
        """Runs on the write-pipeline worker thread.

        Does the heavy lifting (GPU→CPU copies inside extract_and_record,
        summary compute, bytearray fills, and WriteGroup RPCs) without
        holding up the vLLM forward-pass return. Touches shared client
        state under self._score_lock to serialize with main-thread
        Score/FetchAll RPCs.

        memcpy_done_event: optional event signaled AFTER all per-layer
        extract_and_record calls complete (i.e., after every .cpu() has
        returned). The forward thread waits on this event before letting
        vLLM free GPU pages — see wait_for_pending_writes. Always set in
        a finally block so an exception in the extract loop cannot
        deadlock vLLM."""
        # BUG-N13 Phase 1 diag: entry trace. Captures pipeline-thread state
        # at the moment we attempt extract+flush. Toggle ICMS_DIAG_N13=1.
        if os.environ.get("ICMS_DIAG_N13", "0") == "1":
            meta_kind = "dict" if isinstance(meta, dict) else type(meta).__name__
            rid_summaries = []
            for rid in rids:
                rs_dbg = self._requests.get(rid)
                if rs_dbg is None:
                    rid_summaries.append(f"{rid}=missing")
                else:
                    rid_summaries.append(
                        f"{rid}=chain_len={len(rs_dbg.chain)} "
                        f"stored={rs_dbg.stored_groups} "
                        f"written={rs_dbg.num_groups_written} "
                        f"buffers={len(rs_dbg.active_group_buffers)}")
            logger.info(
                "[diag-extract-entry] rank=%d caches=%d meta=%s rids=[%s] "
                "skip_extract=%s",
                self._tp_rank, len(caches), meta_kind,
                "; ".join(rid_summaries), self._skip_extract)
        # ─── 1. Batch extraction
        # CRITICAL: any return / exception from this section MUST signal
        # memcpy_done_event so wait_for_pending_writes does not deadlock
        # the forward thread. Use try/finally.
        try:
            for layer_name, kv in caches.items():
                if kv is None or kv.ndim != 5:
                    continue
                if isinstance(meta, dict):
                    am = meta.get(layer_name)
                else:
                    am = meta
                if am is None:
                    continue
                try:
                    self.extract_and_record(layer_name, kv, am)
                except Exception:
                    logger.exception(
                        "deferred-extract: extract_and_record failed "
                        "for layer=%s", layer_name)
        finally:
            # All .cpu() calls have returned by this point (PyTorch
            # tensor.cpu() is a synchronous GPU→CPU copy that blocks the
            # caller until the bytes are on host). The GPU pages backing
            # this prefill are now safe to release.
            if memcpy_done_event is not None:
                memcpy_done_event.set()

        # ─── 2. Flush complete group buffers via WriteGroup RPCs.
        # BUG-N5: no _score_lock here. The icms client serializes
        # QP access internally (per-conn QP mutex); _score_lock only
        # guards _pending_scores / _pending_reuse dict mutations.
        # Holding it across a 170 ms write_group RPC needlessly
        # blocked main-thread wait_for_layer / on_layer_score dict
        # ops. (The original comment on this block claimed the lock
        # serialized the QP — that's stale: score()/fetch_all()
        # already run lock-free at the connector level.)
        for rid in rids:
            rs = self._requests.get(rid)
            if rs is None:
                continue
            for gidx in sorted(rs.active_group_buffers.keys()):
                buf = rs.active_group_buffers.get(gidx)
                if buf is None:
                    continue
                if buf.is_complete():
                    self._flush_group(rid, gidx, partial=False)

        # ─── 3. Push stored-prefix entries so subsequent requests can
        #       skip extraction for the matched prefix.
        # Record `flushed_local` (count of groups this request actually
        # flushed), NOT `num_groups_written` (which may be inflated by
        # the inherited-prefix elision path in extract_and_record). The
        # ledger must reflect what's truly in the trie or downstream
        # Score RPCs hit ENOENT.
        for rid in rids:
            rs = self._requests.get(rid)
            if rs is None:
                continue
            if rs.chain and rs.flushed_local > 0:
                self._record_stored_groups(rs.chain, rs.flushed_local)
                # Bug #5 fix (race-audit 2026-05-08): single atomic
                # operation under `_stored_chain_cond`. Append + bump
                # are now indivisible from a scheduler-thread drain's
                # snapshot+clear, closing the iter/clear race.
                _append_stored_chain_queue(
                    list(rs.chain), rs.flushed_local)
    def extract_and_record(self, layer_name: str, kv_layer: torch.Tensor,
                            attn_metadata) -> None:
        """Extract per-block K/V from the GPU KV cache and populate icms buffers.

        Called from save_kv_layer for each standard (non-Quest) per-layer call.

        kv_layer shape: [2 (K+V), num_blocks, block_size, num_kv_heads, head_dim]
        attn_metadata: FlashAttentionMetadata with block_table, seq_lens, etc.
        """
        # Diagnostic short-circuit: ICMS_SKIP_EXTRACT=1 makes this a no-op.
        # Used to verify whether extract_and_record's deferred GPU work
        # (AllGather K/V at TP>1, GPU->CPU memcpy) is silently serializing
        # with the model's forward via shared CUDA stream. If TTFT drops
        # noticeably with this set, those ops are on the critical path
        # despite running in a "background" thread.
        import os as _os
        # BUG-N13 Phase 1 diag: trace every entry + every early-return
        # path in extract_and_record. Only log on layer_idx==0 to avoid
        # 48× spam per request.
        _diag_n13 = _os.environ.get("ICMS_DIAG_N13", "0") == "1"
        if _os.environ.get("ICMS_SKIP_EXTRACT", "0") == "1":
            if _diag_n13:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=env_skip",
                            self._tp_rank, layer_name)
            return
        # BUG-N2: scheduler-decided per-step skip flag. Set by
        # on_step_start when every active request's stored prefix
        # already covers its complete-group count. Symmetric across TP
        # ranks because it's derived from broadcast metadata.
        # Multi-rid path (ICMS_ALLOW_BATCH=1): the global short-circuit
        # is too coarse — one non-skip rid in a batch should not force
        # extract for the skip-eligible rids. Skip the early-return
        # here; the plan-builder below filters per-rid via
        # self._skip_extract_rids instead.
        # 2026-05-09 (Item C audit): use _is_multi_rid_mode() so the
        # global short-circuit is also bypassed when max_num_seqs > 1
        # without ICMS_ALLOW_BATCH=1. Otherwise one non-skip rid in a
        # batch would force-skip every other rid's extraction.
        if self._skip_extract and not self._is_multi_rid_mode():
            if _diag_n13:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=N2_skip_extract",
                            self._tp_rank, layer_name)
            return
        t0 = time.perf_counter()
        logger.debug("extract_and_record: layer=%s kv_shape=%s", layer_name,
                      tuple(kv_layer.shape) if hasattr(kv_layer, 'shape') else '?')
        if not self._requests or self._geom is None:
            if _diag_n13:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=no_requests/geom "
                            "requests=%d geom=%s",
                            self._tp_rank, layer_name,
                            len(self._requests), self._geom is not None)
            return

        # Parse layer index from layer_name like "model.layers.5.self_attn.attn"
        layer_idx = self._parse_layer_idx(layer_name)
        if layer_idx is None or layer_idx >= self._geom.num_layers:
            if _diag_n13:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=bad_layer_idx",
                            self._tp_rank, layer_name)
            return

        # 2026-05-29 SW-skip: short-circuit BEFORE the GPU→CPU memcpy.
        # SW layers carry no K/V in the trie; the previous code path
        # uselessly transferred + serialized null-block garbage for
        # gemma-3's 52 SW layers, inflating the WriteGroup frame to
        # 263 MB > 128 MiB wire cap. For uniform models (sw_mask==0)
        # is_sw is always False → byte-identical legacy behavior.
        if self._geom.is_sw(layer_idx):
            if _diag_n13:
                logger.info(
                    "[diag-extract] rank=%d layer=%s SKIP=sw_layer",
                    self._tp_rank, layer_name)
            return

        # KV cache layout: standard FA backend uses
        #   [2 (K+V), num_blocks, block_size, num_kv_heads, head_dim]
        # but TRITON_ATTN backend (used for gemma-3 sliding-window-attention
        # models) uses
        #   [num_blocks, 2 (K+V), block_size, num_kv_heads, head_dim]
        # — the K/V dimension is on axis 1, not axis 0. Detect both. The
        # K/V slice in either layout is [num_blocks, block_size,
        # num_kv_heads, head_dim] which is what the rest of this routine
        # expects.
        if kv_layer.ndim != 5:
            if _diag_n13 and layer_idx == 0:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=bad_kv_shape "
                            "ndim=%d", self._tp_rank, layer_name, kv_layer.ndim)
            return
        if kv_layer.shape[0] == 2:
            # Standard FA layout.
            k_cache = kv_layer[0]
            v_cache = kv_layer[1]
        elif kv_layer.shape[1] == 2:
            # TRITON_ATTN layout (gemma-3, etc.). kv_layer[:, 0] selects K
            # across all blocks; result has the same downstream shape as
            # the standard layout.
            k_cache = kv_layer[:, 0]
            v_cache = kv_layer[:, 1]
        else:
            if _diag_n13 and layer_idx == 0:
                logger.info(
                    "[diag-extract] rank=%d layer=%s SKIP=unknown_kv_layout "
                    "shape=%s", self._tp_rank, layer_name,
                    tuple(kv_layer.shape))
            return

        # block_table: [num_reqs, max_num_blocks_per_req] — per-request block IDs
        block_table = getattr(attn_metadata, "block_table", None)
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if block_table is None or seq_lens is None:
            if _diag_n13 and layer_idx == 0:
                logger.info("[diag-extract] rank=%d layer=%s SKIP=missing_attn_md "
                            "bt=%s sl=%s", self._tp_rank, layer_name,
                            block_table is not None, seq_lens is not None)
            return
        if _diag_n13 and layer_idx == 0:
            sl0 = (int(sl_cpu[0]) if (hasattr(seq_lens, '__len__') and len(seq_lens) > 0)
                   else None) if False else None
            # Don't crash on weird metadata — just log shape
            logger.info("[diag-extract] rank=%d layer=%s ENTRY ok kv_shape=%s "
                        "bt_shape=%s",
                        self._tp_rank, layer_name,
                        tuple(kv_layer.shape),
                        tuple(block_table.shape) if hasattr(block_table, 'shape') else '?')

        # Block size from the KV cache tensor.
        block_size = k_cache.shape[1]

        # block_table may be a tensor on GPU; move to CPU for indexing.
        if isinstance(block_table, torch.Tensor):
            bt_cpu = block_table.cpu()
        else:
            bt_cpu = block_table
        if isinstance(seq_lens, torch.Tensor):
            sl_cpu = seq_lens.cpu()
        else:
            sl_cpu = seq_lens

        # Build the per-rid plan. Single-rid path (legacy): only the
        # first active request, row 0. Multi-rid path (ICMS_ALLOW_BATCH
        # =1): walk batch positions in self._last_step_requests order
        # so both TP ranks symmetrically reach every collective.
        plan: list = []  # list of (rid, rs, req_idx_in_bt)
        # 2026-05-09 (Item C audit): use _is_multi_rid_mode() instead
        # of _allow_batch() so launchers with max_num_seqs > 1 but no
        # ICMS_ALLOW_BATCH=1 take the multi-rid plan-build path. Pre-fix
        # the legacy "first rid only" branch fired and silently dropped
        # the second rid's KV — every rid past the first contributed
        # garbage downstream Score/apply with zeroed sink bytes.
        if self._is_multi_rid_mode() and self._last_step_requests:
            # 2026-05-12 multi-rid row-mapping fix: prefer the
            # authoritative `self._rid_to_bt_row` populated from vLLM's
            # `input_batch.req_ids` (the row order FA actually uses for
            # bt / seq_lens / qsl). The legacy `enumerate(_last_step_requests)`
            # order matches meta.requests (new+cached append) which does
            # NOT match FA when input_batch reorders — pre-fix this read
            # the wrong rid's K/V at extract for every rid in the batch
            # whose position differed. See docs/multi_rid_root_cause_2026-05-12.md.
            # `getattr` keeps the access safe for callers (tests, mocks)
            # that don't initialize the attribute.
            _rid_to_bt_row = getattr(self, "_rid_to_bt_row", None) or {}
            # 2026-05-15 invariant flag: surface the structural hazard
            # of falling back to enumerate order. Cheap; runs once per
            # extract; gated by _is_multi_rid_mode so the single-rid
            # path doesn't pay the cost.
            try:
                _active = [prs.request_id
                           for prs in self._last_step_requests
                           if prs.request_id not in self._skip_extract_rids
                           and self._requests.get(prs.request_id) is not None
                           and self._requests[prs.request_id].chain]
                if _active:
                    self._check_rid_to_bt_row_present(
                        _active, source="extract.multi_rid")
            except Exception:
                # Never let the invariant check itself break extract.
                pass
            for prs in self._last_step_requests:
                rid_i = prs.request_id
                if rid_i in self._skip_extract_rids:
                    continue
                rs_i = self._requests.get(rid_i)
                if rs_i is None or not rs_i.chain:
                    continue
                bt_row_idx = _rid_to_bt_row.get(rid_i)
                if bt_row_idx is None:
                    # Pre-fix path: no input_batch ordering plumbed
                    # through. Fall back to legacy enumerate order.
                    bt_row_idx = next(
                        (i for i, p in enumerate(self._last_step_requests)
                         if p.request_id == rid_i), 0)
                plan.append((rid_i, rs_i, bt_row_idx))
        else:
            # Legacy single-rid path: first active request, row 0.
            # 2026-05-10 audit fix #19: pick from `_last_step_requests`
            # (broadcast scheduler metadata, identical order on every
            # rank) instead of `self._requests.items()` (Python dict
            # insertion order, which can drift if a retry path on one
            # rank inserts an extra rs that the other rank doesn't
            # have).  Pre-fix, dict-iteration drift caused different
            # ranks to pick different rids → asymmetric K bytes
            # shipped to MemBackend → silent corruption (same shape as
            # the bug we just root-caused).  Falls back to the legacy
            # dict-iter only when `_last_step_requests` is empty (rare
            # edge case during connector startup before any
            # scheduler step has been observed).
            if self._last_step_requests:
                for prs in self._last_step_requests:
                    rid_i = prs.request_id
                    rs_i = self._requests.get(rid_i)
                    if rs_i is None or not rs_i.chain:
                        continue
                    plan.append((rid_i, rs_i, 0))
                    break
            else:
                for r, s in self._requests.items():
                    if s is None or not s.chain:
                        continue
                    plan.append((r, s, 0))
                    break
        if not plan:
            return

        # ICMS_DIAG_FULLTRACE: cross-rid block-table snapshot at L=0.
        # Dumps every active rid's bt row + seq_len so set-intersection
        # across rid timelines reveals physical-block aliasing.
        if _ICMS_FULLTRACE_ENABLED and layer_idx == 0:
            try:
                per_rid_rows: list = []
                for _rid, _rs, _bt_idx in plan:
                    if hasattr(sl_cpu, '__len__') and len(sl_cpu) > _bt_idx:
                        _sl = int(sl_cpu[_bt_idx]) if isinstance(
                            sl_cpu[_bt_idx],
                            (int, float, torch.Tensor)) else 0
                    else:
                        _sl = 0
                    _nb = (_sl + block_size - 1) // block_size
                    if bt_cpu.ndim >= 2:
                        _bts = int(bt_cpu.shape[1])
                        _nb = min(_nb, _bts)
                        _bt_row_full = bt_cpu[_bt_idx, :_nb].tolist()
                    elif bt_cpu.ndim == 1:
                        _bts = int(bt_cpu.shape[0])
                        _nb = min(_nb, _bts)
                        _bt_row_full = bt_cpu[:_nb].tolist()
                    else:
                        _bt_row_full = []
                    per_rid_rows.append({
                        "rid": str(_rid),
                        "bt_row_idx": int(_bt_idx),
                        "seq_len": int(_sl),
                        "num_blocks": int(_nb),
                        "chain_len": int(
                            len(_rs.chain) if _rs.chain else 0),
                        "stored_groups": int(getattr(
                            _rs, "stored_groups", -1)),
                        "num_groups_written": int(getattr(
                            _rs, "num_groups_written", -1)),
                        "bt_row_first16": [int(x)
                                             for x in _bt_row_full[:16]],
                        "bt_row_last4": [int(x)
                                           for x in _bt_row_full[-4:]],
                        "bt_row_full": [int(x) for x in _bt_row_full[:2048]],
                    })
                # Cross-rid intersection counts: for each pair (i, j),
                # count how many phys blocks they share.
                _bts_per_rid = [set(int(x) for x in r["bt_row_full"])
                                 for r in per_rid_rows]
                _bts_per_rid_full = []
                for r in per_rid_rows:
                    _bts_per_rid_full.append(set(int(x) for x in
                                                  r["bt_row_full"]))
                pair_intersections: list = []
                for i in range(len(per_rid_rows)):
                    for j in range(i + 1, len(per_rid_rows)):
                        _inter = _bts_per_rid_full[i] & _bts_per_rid_full[j]
                        pair_intersections.append({
                            "i_rid": per_rid_rows[i]["rid"],
                            "j_rid": per_rid_rows[j]["rid"],
                            "intersection_count": int(len(_inter)),
                            "intersection_head": sorted(list(_inter))[:8],
                        })
                _icms_fulltrace(
                    "iter_bt_snapshot", rid="", layer=int(layer_idx),
                    n_rids=len(per_rid_rows),
                    per_rid=per_rid_rows,
                    pair_intersections=pair_intersections,
                    bt_shape=list(bt_cpu.shape) if hasattr(bt_cpu, 'shape')
                        else [],
                    skip_extract_rids=sorted(list(self._skip_extract_rids)),
                )
            except Exception:
                pass

        # Process each rid's new blocks. At TP>1, AllGather fires once
        # per rid (per layer); both ranks iterate plan in identical
        # order (derived from broadcast metadata) so collective ordering
        # is symmetric. Future optimization: stack per-rid K/V then
        # AllGather once per layer (Phase B).
        for rid, rs, bt_row_idx in plan:
            self._extract_one_rid(layer_idx, layer_name, k_cache, v_cache,
                                  bt_cpu, sl_cpu, block_size, rid, rs,
                                  bt_row_idx, t0)
        return
    def _extract_one_rid(self, layer_idx, layer_name, k_cache, v_cache,
                         bt_cpu, sl_cpu, block_size, rid, rs, bt_row_idx,
                         t0):
        # Number of blocks for this rid.
        if hasattr(sl_cpu, '__len__') and len(sl_cpu) > bt_row_idx:
            seq_len = int(sl_cpu[bt_row_idx]) if isinstance(
                sl_cpu[bt_row_idx], (int, float, torch.Tensor)) else 0
        else:
            seq_len = 0
        if seq_len <= 0:
            return
        num_blocks = (seq_len + block_size - 1) // block_size

        # BUG-N1 fix: under prefix elision (TP>1 + ICMS), vLLM only
        # allocates the trailing slice of the block table, but seq_lens
        # still reports the full prompt length. Trailing slots in
        # bt_cpu[row, k:] hold stale IDs from another request's
        # allocation; indexing k_cache by those produces a CUDA OOB
        # that surfaces as a device-side assert at the next .cpu().
        # Clamp num_blocks to the actually-allocated row width so we
        # never read past the live region. Both ranks at TP>1 observe
        # bt_cpu shape symmetrically, so this clamp is collective-safe.
        # The existing dedup early-return at `effective_start >=
        # len(req_block_ids)` below then short-circuits when the
        # already-stored groups cover the (clamped) allocation — but
        # crucially that early-return only fires after the AllGather,
        # so an asymmetric skip *before* AllGather would deadlock.
        if bt_cpu.ndim >= 2:
            num_blocks = min(num_blocks, int(bt_cpu.shape[1]))
        elif bt_cpu.ndim == 1:
            num_blocks = min(num_blocks, int(bt_cpu.shape[0]))
        else:
            return

        # Extract this rid's block IDs from the block table row.
        if bt_cpu.ndim >= 2:
            req_block_ids = bt_cpu[bt_row_idx, :num_blocks].tolist()
        else:
            req_block_ids = bt_cpu[:num_blocks].tolist()

        # ICMS_TRACE: which physical GPU pages does this rid's bt_row map
        # to? If two different rids share physical pages here, we have
        # cross-rid block_table contamination at extract time.
        if _ICMS_TRACE_ENABLED:
            try:
                _icms_trace(
                    "extract_bt", rid, layer=layer_idx, chain_fp="",
                    extra={
                        "bt_row_idx": int(bt_row_idx),
                        "num_blocks": int(num_blocks),
                        "seq_len": int(seq_len),
                        "bt_row_first16": [int(x) for x in req_block_ids[:16]],
                        "bt_row_last4": [int(x) for x in req_block_ids[-4:]],
                    })
            except Exception:
                pass

        # Only process blocks we haven't recorded yet for this layer.
        recorded_key = (rid, layer_idx)
        if not hasattr(rs, '_recorded_blocks'):
            rs._recorded_blocks = {}
        already_recorded = rs._recorded_blocks.get(recorded_key, 0)

        # Dedup-aware skip: blocks in groups already stored under the
        # request's chain prefix don't need re-extraction. The server
        # would dedup the write anyway, but the GPU→CPU copy + summary
        # compute are wasted client-side work on the critical path.
        # 2026-05-10 TP>1 stored_groups asymmetry fix (extract-side):
        # Read the symmetrized value populated on the forward thread
        # by _update_stored_groups_from_meta (which all-reduces under
        # the main-thread NCCL group). This loop runs on the pipeline
        # thread, so we MUST NOT issue NCCL here — it would collide
        # with vLLM's per-layer all_reduce on the main thread (verified
        # to hang the engine at first prefill). The forward-thread
        # symmetrize ensures both ranks see the same value.
        effective_stored_groups = max(
            int(rs.stored_groups),
            int(getattr(rs, "_effective_stored_groups", 0)))
        stored_blocks = effective_stored_groups * _GROUP_BLOCKS
        effective_start = max(already_recorded, stored_blocks)
        if effective_start > already_recorded:
            # Advance num_groups_written so later _flush_group calls and
            # the stored-prefix push correctly reflect "stored + new".
            if effective_stored_groups > rs.num_groups_written:
                rs.num_groups_written = effective_stored_groups

        if effective_start >= len(req_block_ids):
            rs._recorded_blocks[recorded_key] = len(req_block_ids)
            return  # everything we'd process is already stored

        new_ids = req_block_ids[effective_start:]
        valid_ids = [bid for bid in new_ids if 0 <= bid < k_cache.shape[0]]
        if not valid_ids:
            rs._recorded_blocks[recorded_key] = len(req_block_ids)
            return

        # Batched GPU→CPU copy: index all new blocks at once.
        idx_tensor = torch.tensor(valid_ids, dtype=torch.long, device=k_cache.device)
        k_batch_gpu = k_cache[idx_tensor]
        v_batch_gpu = v_cache[idx_tensor]

        # Option W write-path: at TP>1 each rank's kv_cache holds only
        # its num_kv_heads_local slice. NCCL-AllGather across the TP
        # group along the num_kv_heads dim (dim=2 for
        # [N, block_size, num_kv_heads_local, head_dim]) so rank 0
        # can emit a full-head blob via write_group. Both ranks must
        # reach the collective: extract_and_record's early-returns
        # (self._requests empty / seq_len<=0 / nothing-to-record) are
        # derived from scheduler-propagated state and so fire
        # symmetrically at TP=2. If we ever see a hang here it means
        # that assumption is wrong — fall back by wrapping with a
        # pre-sync AllReduce(have-data) flag.
        if self._tp_size > 1:
            try:
                import torch.distributed as dist  # noqa: E402
                from vllm.distributed.parallel_state import get_tp_group
                tp_group = get_tp_group()
                # T2 fix (race-audit, 2026-05-08): route through the
                # ICMS sep-NCCL group when ICMS_USE_SEPARATE_NCCL_GROUP=1
                # so the pipeline thread doesn't share the comm with
                # vLLM's main-thread per-layer all_reduce. Pre-fix this
                # site hardcoded tp_group.device_group, defeating the
                # sep-NCCL knob for the K/V AllGather and explaining
                # llama3 TP=2 hangs even with sep-NCCL on.
                dev_group = _get_icms_nccl_group() or tp_group.device_group
                gk = [torch.empty_like(k_batch_gpu)
                      for _ in range(self._tp_size)]
                gv = [torch.empty_like(v_batch_gpu)
                      for _ in range(self._tp_size)]
                _trace = os.environ.get("ICMS_NCCL_TRACE", "0") == "1"
                _diag_order = (os.environ.get("ICMS_DIAG_HEAD_ORDER", "0") == "1"
                               and not getattr(self, "_diag_head_order_fired", False))
                if _diag_order:
                    pre_hash = int(k_batch_gpu.contiguous().view(torch.uint8).sum().item())
                if _trace:
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                dist.all_gather(gk, k_batch_gpu.contiguous(),
                                 group=dev_group)
                dist.all_gather(gv, v_batch_gpu.contiguous(),
                                 group=dev_group)
                if _trace:
                    torch.cuda.synchronize()
                _us = (time.perf_counter() - _t0) * 1e6
                if _diag_order:
                    try:
                        try:
                            grp_rank_dist = dist.get_rank(group=dev_group)
                        except Exception:
                            grp_rank_dist = -1
                        grp_rank_attr = getattr(tp_group, "rank_in_group", -1)
                        per_rank_hashes = [
                            int(t.contiguous().view(torch.uint8).sum().item())
                            for t in gk
                        ]
                        head0_first = [
                            int(gk[i].contiguous().view(-1).view(torch.uint8)[0].item())
                            for i in range(self._tp_size)
                        ]
                        logger.info(
                            "[icms-diag-head-order] self_tp_rank=%d "
                            "tp_group.rank_in_group=%s dist.get_rank(dev_group)=%s "
                            "pre_allgather_hash=%d gk_hashes=%s "
                            "gk_head0_first_byte=%s layer=%d k_shape=%s",
                            self._tp_rank, grp_rank_attr, grp_rank_dist,
                            pre_hash, per_rank_hashes, head0_first,
                            layer_idx, list(k_batch_gpu.shape))
                    except Exception as _diag_e:
                        logger.warning("[icms-diag-head-order] diag failed (non-fatal): %s", _diag_e)
                    self._diag_head_order_fired = True
                k_batch_gpu = torch.cat(gk, dim=2)
                v_batch_gpu = torch.cat(gv, dim=2)
                if _trace:
                    bytes_per_rank = int(
                        k_batch_gpu.numel() * k_batch_gpu.element_size()
                        + v_batch_gpu.numel() * v_batch_gpu.element_size())
                    logger.info("[nccl-trace] phase=extract_kv_allgather "
                                 "rank=%d layer=%d kv_bytes_per_rank=%d "
                                 "us=%.1f",
                                 self._tp_rank, layer_idx,
                                 bytes_per_rank, _us)
            except Exception as e:
                # 2026-05-10 audit fix #4: re-raise instead of falling
                # back to per-rank K/V.  Pre-fix this swallowed the
                # exception and proceeded with the rank-LOCAL nkv-heads
                # slice — write_group then shipped a half-head blob to
                # the server, silently corrupting MemBackend's view of
                # the chain.  The other rank may have completed
                # AllGather and shipped FULL-head data → ranks now
                # disagree on what's stored under the chain.  Surface
                # the failure loudly via the connector's existing
                # exception path; the deferred-extract pipeline already
                # logs + propagates exceptions to wait_for_pending_writes.
                logger.error(
                    "extract_and_record: TP AllGather(K/V) failed "
                    "rank=%d layer=%d: %s — re-raising. Silent per-rank "
                    "fallback would write half-head garbage to the "
                    "server and silently desynchronize ranks.",
                    self._tp_rank, layer_idx, e, exc_info=True)
                raise

        k_batch = k_batch_gpu.cpu()
        v_batch = v_batch_gpu.cpu()

        # 2026-05-10 TP=2 KV-content corruption diag: hash the K bytes
        # of the FIRST extracted page at L=0 only. Lets us compare the
        # extracted-K hash against the apply-side K hash for the same
        # logical page → if they differ, the corruption is between
        # write-to-ICMS and read-from-ICMS at TP=2.
        if _ICMS_TRACE_ENABLED and layer_idx == 0 and len(valid_ids) > 0:
            try:
                import hashlib as _hl
                _kb = k_batch[0].numpy().tobytes()[:128]
                _vb = v_batch[0].numpy().tobytes()[:128]
                _icms_trace(
                    "extract_kv_hash", rid, layer=int(layer_idx),
                    chain_fp="",
                    extra={
                        "physical_page": int(req_block_ids[
                            effective_start + 0]) if effective_start < len(
                            req_block_ids) else -1,
                        "intra_idx": int(effective_start),
                        "k_bytes_hash": _hl.sha1(_kb).hexdigest()[:16],
                        "v_bytes_hash": _hl.sha1(_vb).hexdigest()[:16],
                        "k_first_8_bytes_hex": _kb[:8].hex(),
                        "k_shape": list(k_batch[0].shape),
                        "tp_rank": int(self._tp_rank),
                        "tp_size": int(self._tp_size),
                        "n_pages_extracted": int(len(valid_ids)),
                    })
                # 2026-05-10 follow-up audit trace: full-page V hash,
                # plus a sampled set across the extracted batch (pages
                # 0, mid, last) so we don't only validate page 0. K
                # was previously shown bit-identical at TP=2 batched;
                # V wasn't checked, so add an explicit trace event.
                # Also full-bytes (not just first 128) so we don't
                # miss a tail-region corruption.
                _kf = k_batch[0].numpy().tobytes()
                _vf = v_batch[0].numpy().tobytes()
                samples: list = []
                _idxs = [0]
                if len(valid_ids) > 1:
                    _idxs.append(len(valid_ids) // 2)
                if len(valid_ids) > 2:
                    _idxs.append(len(valid_ids) - 1)
                for _i in _idxs:
                    _kbi = k_batch[_i].numpy().tobytes()
                    _vbi = v_batch[_i].numpy().tobytes()
                    samples.append({
                        "intra_idx": int(_i),
                        "physical_page": int(req_block_ids[
                            effective_start + _i]) if (
                            effective_start + _i < len(req_block_ids)
                        ) else -1,
                        "k_full_hash": _hl.sha1(_kbi).hexdigest()[:16],
                        "v_full_hash": _hl.sha1(_vbi).hexdigest()[:16],
                    })
                _icms_trace(
                    "extract_v_hash", rid, layer=int(layer_idx),
                    chain_fp="",
                    extra={
                        "tp_rank": int(self._tp_rank),
                        "tp_size": int(self._tp_size),
                        "n_pages_extracted": int(len(valid_ids)),
                        "page0_v_full_hash":
                            _hl.sha1(_vf).hexdigest()[:16],
                        "page0_k_full_hash":
                            _hl.sha1(_kf).hexdigest()[:16],
                        "samples": samples,
                    })
            except Exception:
                pass

        # 2026-05-10 TP>1 byte-layout diag: log key_block shape at L=0
        # so we can verify it's full K (post-AllGather) vs per-rank K.
        if (layer_idx == 0 and len(valid_ids) > 0
                and not getattr(self, "_diag_keyblock_shape_fired", False)):
            try:
                logger.warning(
                    "[diag-keyblock-shape] tp_rank=%d tp_size=%d "
                    "k_batch.shape=%s k_batch[0].shape=%s "
                    "expected_full_per_geom=%dx%d k_batch[0].nbytes=%d",
                    self._tp_rank, self._tp_size,
                    list(k_batch.shape), list(k_batch[0].shape),
                    self._geom.num_kv_heads if self._geom else -1,
                    self._geom.head_dim if self._geom else -1,
                    int(k_batch[0].numel() * k_batch[0].element_size()))
                self._diag_keyblock_shape_fired = True
            except Exception:
                pass

        for i, intra_idx in enumerate(range(effective_start, effective_start + len(valid_ids))):
            self.record_page(rid, intra_idx, layer_idx, k_batch[i], v_batch[i])

        rs._recorded_blocks[recorded_key] = len(req_block_ids)
        self.stats.record_extract(
            (time.perf_counter() - t0) * 1e6, len(valid_ids))
        # ICMS_DIAG_FULLTRACE: per-rid per-layer extract summary. Layer 0
        # also emits first/last/sample K+V SHAs over the extracted batch
        # so we can cross-reference Phase 1 against what comes back at
        # apply time.
        if _ICMS_FULLTRACE_ENABLED:
            try:
                import hashlib as _hl_ft
                extra: dict = {
                    "bt_row_idx": int(bt_row_idx),
                    "seq_len": int(seq_len),
                    "num_blocks": int(num_blocks),
                    "effective_start": int(effective_start),
                    "stored_blocks": int(stored_blocks),
                    "n_valid": int(len(valid_ids)),
                    "valid_ids_head": [int(x) for x in valid_ids[:8]],
                    "valid_ids_tail": [int(x) for x in valid_ids[-8:]],
                    "req_block_ids_head": [int(x)
                                             for x in req_block_ids[:8]],
                    "req_block_ids_tail": [int(x)
                                             for x in req_block_ids[-8:]],
                    "num_groups_written": int(rs.num_groups_written),
                    "stored_groups": int(rs.stored_groups),
                    "chain_len": int(len(rs.chain) if rs.chain else 0),
                }
                if layer_idx == 0 and len(valid_ids) > 0:
                    samples: list = []
                    _idxs = [0]
                    if len(valid_ids) > 1:
                        _idxs.append(len(valid_ids) // 2)
                    if len(valid_ids) > 2:
                        _idxs.append(len(valid_ids) - 1)
                    for _i in _idxs:
                        _kbi = k_batch[_i].contiguous().view(
                            torch.uint8).numpy().tobytes()
                        _vbi = v_batch[_i].contiguous().view(
                            torch.uint8).numpy().tobytes()
                        samples.append({
                            "intra_idx": int(effective_start + _i),
                            "phys_block": int(req_block_ids[
                                effective_start + _i])
                                if effective_start + _i < len(
                                    req_block_ids) else -1,
                            "k_sha": _hl_ft.sha1(_kbi).hexdigest()[:16],
                            "v_sha": _hl_ft.sha1(_vbi).hexdigest()[:16],
                            "k_head8": _kbi[:8].hex(),
                        })
                    extra["samples"] = samples
                _icms_fulltrace("extract", rid=rid, layer=int(layer_idx),
                                **extra)
            except Exception:
                pass
    def record_page(
        self,
        request_id: str,
        intra_request_block_idx: int,
        layer_idx: int,
        key_block: torch.Tensor,    # [block_size, num_kv_heads, head_dim]
        value_block: torch.Tensor,  # same shape
    ):
        """Record one page's K/V + summary into the per-(req, group) buffer.

        Called from the metadata-update-after-store equivalent in the
        connector's save path.
        """
        rs = self._requests.get(request_id)
        if rs is None:
            return
        geom = self._geom

        group_idx, page_in_group = divmod(intra_request_block_idx, _GROUP_BLOCKS)
        buf = rs.active_group_buffers.get(group_idx)
        if buf is None:
            # Summary region is packed by scored_rank; size accordingly.
            # 2026-05-29 SW-skip: kv_blob region is packed by num_kv_layers
            # (= num_layers - num_sw_layers). For uniform models with
            # sw_layers_mask==0, num_kv_layers == num_layers ⇒
            # byte-identical to legacy.
            buf = _GroupBuffer(
                summary_blob=bytearray(
                    geom.num_scored_layers * geom.summary_group_bytes),
                kv_blob=bytearray(
                    geom.num_kv_layers * geom.kv_group_bytes),
                filled=set(),
                # 2026-05-30 (re-RCA wj3qp74qx): use num_kv_layers, NOT
                # num_layers. SW layers short-circuit before filled.add()
                # below, so len(filled) maxes at num_kv_layers * 32 (=320
                # for gemma-3). With num_layers=62 here, is_complete()
                # required 1984 and was structurally False forever →
                # pipeline never auto-flushed → 128 groups serialized
                # through on_request_finished's forward-thread drain
                # (15s drain_rid) AND every flush is_partial=True →
                # trie undercount → no inter-iter elision → re-extract
                # full prefix every iter → 22s TTFT. For uniform models
                # (sw_mask=0) num_kv_layers == num_layers ⇒ byte-identical.
                num_layers=geom.num_kv_layers,
                pages_in_group=_GROUP_BLOCKS,
            )
            rs.active_group_buffers[group_idx] = buf

        # 2026-05-29 SW-skip: SW layers carry no K/V in the trie; skip
        # serialization. Server's per_group_bytes excludes SW layers
        # (mirror via --sw-layers). For uniform models (sw_mask==0)
        # this branch is never taken.
        if geom.is_sw(layer_idx):
            return

        # KV: K || V byte concatenation (C4). Stored for every non-SW layer.
        # Bug 11 family fix (2026-04-30): preserve full model precision by
        # serializing raw model_dtype bytes (no fp16 down-cast). The
        # legacy `.to(torch.float16)` was a bf16↔fp16 round-trip for bf16
        # models like qwen3, losing ~3 mantissa bits per value. At 32k
        # chain × 16 tokens × 4 KV heads × 128 dim × 48 layers = ~745M
        # values per request, the accumulated drift derails attention
        # enough that multi-key NIAH retrieval fails. Both bf16 and fp16
        # are 2 bytes per element; view(uint8) reinterprets the buffer
        # as numpy-compatible bytes without any conversion. Apply side
        # must view back as model_dtype (not fp16) — see lines 3805+,
        # 3448+, 3957+ for the corresponding read-side updates.
        k_bytes = key_block.contiguous().view(torch.uint8).numpy().tobytes()
        v_bytes = value_block.contiguous().view(torch.uint8).numpy().tobytes()
        kv_bytes = k_bytes + v_bytes

        # ICMS_DIAG_CANARY=1: per-write fingerprint for a few logical
        # pages of layer 0 (we recompute target pages from intra_idx →
        # group_idx*32+page_in_group). Pair with [diag-canary-read]
        # for the same pid in _apply_selective_attention.
        # Multi-layer canary (2026-04-30): also fire for scored layers
        # 6,12,18,24,30,36,42 so we can verify the FAPS slow-path
        # per-layer offset arithmetic. If layer-0 hashes match write↔read
        # but layer-6+ don't, the server's per-layer pack ordering is
        # the bug. Restricted to one probe pid per layer to keep volume
        # low.
        if (os.environ.get("ICMS_DIAG_CANARY") == "1"
                and layer_idx in (0, 6, 12, 18, 24, 30, 36, 42)):
            absolute_pid = group_idx * _GROUP_BLOCKS + page_in_group
            # Layer 0 keeps full probe set for byte-path correctness.
            # Higher layers only sample pid=0 / 17 / 100 to keep log size
            # manageable across 7 extra layers.
            _probe_set = (
                (0, 1, 2, 3, 4, 8, 16, 17, 24, 31, 32, 33, 100, 500)
                if layer_idx == 0 else (0, 17, 100))
            if absolute_pid in _probe_set:
                import hashlib
                chain_head = rs.chain[:1] if rs.chain else []
                kh = hashlib.sha1(k_bytes).hexdigest()[:16]
                vh = hashlib.sha1(v_bytes).hexdigest()[:16]
                khead = k_bytes[:32].hex()
                logger.info("[diag-canary-write] rid=%s chain_head=%s "
                             "abs_pid=%d gidx=%d page_in_group=%d layer=%d "
                             "k_len=%d v_len=%d k_sha=%s v_sha=%s "
                             "k_head=%s",
                             rs.request_id, chain_head, absolute_pid,
                             group_idx, page_in_group, layer_idx,
                             len(k_bytes), len(v_bytes), kh, vh, khead)
                # Per-rank-subset canary: hash the slice of key_block /
                # value_block that EACH rank's apply read-side will see
                # under PRS=1. key_block here has shape
                # [block_size=16, num_kv_heads=4, head_dim=128]. Each
                # rank's slice is heads [r*nkv_local, (r+1)*nkv_local).
                # Direct-compare sub_sha against [diag-canary-read]
                # k_sha at the same (layer, pid).
                if self._tp_size > 1:
                    nkv_local = (self._geom.num_kv_heads // self._tp_size)
                    for _r in range(self._tp_size):
                        _s = _r * nkv_local
                        _sub_k = (key_block[:, _s:_s + nkv_local, :]
                                  .contiguous().view(torch.uint8)
                                  .numpy().tobytes())
                        _sub_v = (value_block[:, _s:_s + nkv_local, :]
                                  .contiguous().view(torch.uint8)
                                  .numpy().tobytes())
                        _sub_kh = hashlib.sha1(_sub_k).hexdigest()[:16]
                        _sub_vh = hashlib.sha1(_sub_v).hexdigest()[:16]
                        logger.info(
                            "[diag-canary-write-rank] rid=%s abs_pid=%d "
                            "layer=%d rank=%d k_sub_len=%d v_sub_len=%d "
                            "k_sub_sha=%s v_sub_sha=%s k_sub_head=%s",
                            rs.request_id, absolute_pid, layer_idx, _r,
                            len(_sub_k), len(_sub_v),
                            _sub_kh, _sub_vh, _sub_k[:32].hex())

        spb = geom.summary_page_bytes
        kpb = geom.kv_page_bytes
        # Byte offset within buf.kv_blob for (layer_idx, page_in_group).
        # Must mirror the server's kv_offset() — both sides agree via
        # the kv_layout config. Offsets here are relative to the start
        # of the KV region (buf.kv_blob does not include summaries).
        # 2026-05-29 SW-skip: use kv_layer_rank (== layer when sw_mask==0)
        # so the KV region is packed by physical write rank, excluding
        # SW layers. Server-side per_group_bytes mirror must use the
        # same num_kv_layers count.
        _kv_rank = geom.kv_layer_rank(layer_idx)
        if geom.kv_layout == KvLayout.LAYER_MAJOR:
            k_off = _kv_rank * geom.kv_group_bytes + page_in_group * kpb
        else:  # PAGE_MAJOR
            k_off = page_in_group * geom.num_kv_layers * kpb + _kv_rank * kpb
        buf.kv_blob[k_off:k_off + len(kv_bytes)] = kv_bytes

        # Summary: only compute + store for scored layers. Non-scored layers
        # have no on-disk summary slot.
        if geom.is_scored(layer_idx):
            keys = key_block.to(dtype=torch.float32, device="cpu")
            if keys.ndim == 3:
                keys = keys.reshape(keys.shape[0], -1)  # [block_size, key_dim]
            kmin = keys.min(dim=0).values.to(torch.float16)
            kmax = keys.max(dim=0).values.to(torch.float16)
            summary_bytes = kmin.numpy().tobytes() + kmax.numpy().tobytes()
            rank = geom.scored_rank(layer_idx)
            s_off = rank * geom.summary_group_bytes + page_in_group * spb
            buf.summary_blob[s_off:s_off + len(summary_bytes)] = summary_bytes

        # Retain GPU-side per-(KV-head) summaries (shape [num_kv_heads,
        # head_dim] fp16, indexed by absolute page id) when EITHER:
        #   ICMS_ORIGINAL_QUEST=1   — local Quest scorer path
        #   ICMS_DIAG_SCORE_DUMP=<> — regular sparse path dump for offline
        #                            scoring-algo analysis (added 2026-05-14)
        # Default path is byte-identical when both envs are unset.
        if (os.environ.get("ICMS_ORIGINAL_QUEST", "0") == "1"
                or os.environ.get("ICMS_DIAG_SCORE_DUMP", "")):
            keys_gpu = key_block.detach()
            if keys_gpu.ndim == 3:
                # [block_size, num_kv_heads, head_dim] → reduce over tokens
                kmin_gpu = keys_gpu.amin(dim=0).to(torch.float16)
                kmax_gpu = keys_gpu.amax(dim=0).to(torch.float16)
                abs_pid = group_idx * _GROUP_BLOCKS + page_in_group
                rs.quest_gpu_summaries.setdefault(layer_idx, []).append(
                    (abs_pid, kmin_gpu, kmax_gpu))

        buf.filled.add((layer_idx, page_in_group))
    def _flush_write_batch_now(self, request_id: str):
        """2026-05-11 Option-1: drain the per-rid write batch buffer
        with ONE WriteGroup RPC carrying K = len(buffer) tail groups.

        Records the post-RPC outcome for each buffered group with the
        SAME `ok_local` (server commits the K groups atomically; one
        RPC error → all K marked failed; one RPC ok → all K marked
        committed). Forwards into `_pending_flush_q` so the existing
        `_drain_pending_flush_queue` machinery bumps `flushed_local`
        and `num_groups_written` once per group, just as it would have
        for K individual single-group RPCs."""
        # Defense-in-depth rank gate (2026-05-11): the callers in
        # `_flush_group` (line ~9077, 9130) and `on_request_finished`
        # (line ~9301) ALREADY gate on rank-0, so this should never
        # fire. Kept inline so the `test_tp_rank_gating` AST inspector
        # can see the gate without traversing the call graph, and to
        # protect future callers from accidentally bypassing it.
        if self._tp_size > 1 and self._tp_rank != 0:
            return
        # 2026-05-11 write-batching-audit Finding 2: pop under the
        # batch-buf lock so a concurrent pipeline-thread append doesn't
        # race the forward-thread drain.
        with self._write_batch_buf_lock:
            buf = self._write_batch_buf.pop(request_id, [])
        if not buf:
            return
        buf.sort(key=lambda b: b['group_idx'])
        # Sanity: every buffered group must have a strictly-consecutive
        # group_idx (the wire protocol's "extend" semantic appends K
        # tails under a single parent — gaps would corrupt the chain).
        # If not consecutive, fall back to N single-group RPCs.
        first_gidx = buf[0]['group_idx']
        for i, b in enumerate(buf):
            if b['group_idx'] != first_gidx + i:
                logger.warning(
                    "[icms-batch] non-consecutive group_idxs in batch "
                    "for rid=%s: %s. Falling back to single-RPC per "
                    "group.", request_id,
                    [b2['group_idx'] for b2 in buf])
                self._fallback_single_rpcs(request_id, buf)
                return
        K = len(buf)
        # 2026-05-11 wire-size guard: the wire frame header is
        # `<IB` (uint32 length + byte type — protocol.py:63), so a
        # single RPC's payload must fit in uint32 (< 4 GiB). For
        # gemma-3-27b at 16 KV heads × 128 head_dim × 62 layers,
        # one group's KV ≈ 500 MB; K=16 → 8 GB → frame-length
        # overflow → struct.pack("<I", length) fails with
        # `'I' format requires 0 <= number <= 4294967295`. If the
        # accumulated payload would exceed the safe cap, split the
        # buffer in half and recurse — each half sends its own
        # WriteGroup RPC. Choice of 3 GiB cap (vs 4 GiB max) leaves
        # headroom for the chain-hash + header bytes and avoids
        # bumping right against the limit.
        _MAX_FRAME_PAYLOAD = int(os.environ.get(
            "ICMS_WRITE_BATCH_MAX_PAYLOAD_BYTES",
            str(3 * 1024 * 1024 * 1024)))
        # Halve K repeatedly until the payload fits in one frame. Each
        # iteration drops half of the buffer back into the pending
        # queue (preserving group_idx order) and re-checks the
        # remaining first chunk. For gemma-3-27b at K=16/8GB, this
        # converges at K=4 (~2GB). Bounded by log2(K) iterations.
        while True:
            _total_payload = sum(
                len(b['summary_blob']) + len(b['kv_blob']) for b in buf)
            if K <= 1 or _total_payload <= _MAX_FRAME_PAYLOAD:
                break
            mid = K // 2
            logger.info(
                "[icms-batch] payload %.2f GiB > cap %.2f GiB for "
                "rid=%s K=%d first_gidx=%d — splitting (sending "
                "K=%d first, deferring K=%d)",
                _total_payload / (1024 ** 3),
                _MAX_FRAME_PAYLOAD / (1024 ** 3),
                request_id, K, first_gidx, mid, K - mid)
            first_half = buf[:mid]
            second_half = buf[mid:]
            with self._write_batch_buf_lock:
                # Prepend second_half so subsequent _flush_group call
                # ordering is preserved (lower group_idxs at front).
                # Any new entries appended concurrently for this rid
                # have higher group_idxs than `second_half`, so
                # prepending keeps the sorted invariant.
                existing = self._write_batch_buf.setdefault(
                    request_id, [])
                self._write_batch_buf[request_id] = second_half + existing
            buf = first_half
            K = len(buf)
        # Build parent + tail chains from the LAST buffered group's
        # `chain_prefix` (= rs.chain[:last_gidx+1] at flush time).
        last_chain_prefix = buf[-1]['chain_prefix']
        rank_chain = self._rank_chain(last_chain_prefix)
        if first_gidx == 0:
            parent_chain: list[int] = []
            new_tail_groups = list(rank_chain[:K])
        else:
            parent_chain = list(rank_chain[:first_gidx])
            new_tail_groups = list(rank_chain[first_gidx:first_gidx + K])
        # Concatenated blobs — server handles K-tail layout per
        # handlers.cc:1099-1115 (summary_blob_base then kv_blob_base
        # at offsets keyed by K, num_scored, num_layers).
        summary_blob = b''.join(b['summary_blob'] for b in buf)
        kv_blob = b''.join(b['kv_blob'] for b in buf)
        # pages_in_group on the wire applies to the LAST tail per
        # client.py:418-425 ("earlier tails must be 32-page-full ...
        # split the bulk write across calls if the LAST tail isn't").
        # Since we only buffer partial=False, all are 32-page-full —
        # send the same pages count for last (a no-op equivalence).
        pages_in_group = buf[-1]['pages']
        t_send = time.perf_counter()
        try:
            # Landmine #1 split: WG uses _write_client + _write_rpc_lock
            # (alias to _client / _rpc_lock when split mode OFF).
            with self._write_rpc_lock:
                self._write_client.write_group(
                    parent_chain, new_tail_groups,
                    summary_blob, kv_blob,
                    pages_in_group=pages_in_group,
                )
        except Exception as e:
            logger.warning(
                "[icms-batch] Batched WriteGroup failed for rid=%s "
                "K=%d first_gidx=%d: %s — falling through to per-"
                "group single-RPC retry for accurate per-tail ok_local.",
                request_id, K, first_gidx, e)
            # 2026-05-11 write-batching-audit Finding 3: on batched-
            # RPC failure, retry per-tail via `_fallback_single_rpcs`
            # so each group gets its own ok_local. Server's WriteGroup
            # is idempotent on already-committed tails (handlers.cc
            # ~L1146-1173: `trie::extend → insert(parent+tails)`
            # fall-through finds existing match and no-ops), so
            # retries don't double-write KV bytes. Pre-fix, the
            # whole batch was marked failed → subsequent same-prefix
            # requests didn't elide ANY of the K groups even if the
            # server had committed k<K before erroring → latency hit
            # proportional to the wasted re-prefill.
            self._fallback_single_rpcs(request_id, buf)
            return
        send_elapsed_us = (time.perf_counter() - t_send) * 1e6
        # Amortize the batched RPC time across the K groups so the
        # histogram remains comparable to the N=1 baseline.
        per_group_lat_us = send_elapsed_us / max(1, K)
        for b in buf:
            self.stats.record_flush(per_group_lat_us)
            with self._pending_flush_lock:
                self._pending_flush_q.append((
                    request_id, int(b['group_idx']), True,
                    bool(b['partial']), int(b['pages']),
                    tuple(b['chain_prefix']),
                ))
    def _fallback_single_rpcs(self, request_id: str,
                               buf: "list[dict]"):
        """Fallback when batch buffer can't be sent atomically
        (non-consecutive group_idxs). Sends N single-group RPCs and
        records the same per-group outcome as the pre-batching path.
        Should be rare — buffer is built by consecutive `_flush_group`
        calls during haystack prefill so non-consecutive only happens
        on weird interleaving (which would be a separate bug)."""
        # Defense-in-depth rank gate (mirrors _flush_write_batch_now).
        if self._tp_size > 1 and self._tp_rank != 0:
            return
        for b in buf:
            rank_chain = self._rank_chain(b['chain_prefix'])
            gidx = b['group_idx']
            if gidx == 0:
                parent_chain: list[int] = []
                new_tail_groups = list(rank_chain)
            else:
                parent_chain = list(rank_chain[:-1])
                new_tail_groups = [rank_chain[-1]]
            ok_local = True
            try:
                # Landmine #1 split: WG → _write_client + _write_rpc_lock.
                with self._write_rpc_lock:
                    self._write_client.write_group(
                        parent_chain, new_tail_groups,
                        b['summary_blob'], b['kv_blob'],
                        pages_in_group=b['pages'],
                    )
            except Exception as e:
                ok_local = False
                logger.warning(
                    "[icms-batch] fallback WriteGroup failed for "
                    "rid=%s gidx=%d: %s", request_id, gidx, e)
            self.stats.record_flush(
                (time.perf_counter() - b['t0']) * 1e6)
            with self._pending_flush_lock:
                self._pending_flush_q.append((
                    request_id, int(gidx), bool(ok_local),
                    bool(b['partial']), int(b['pages']),
                    tuple(b['chain_prefix']),
                ))
    def _flush_group(self, request_id: str, group_idx: int, partial: bool = False):
        """Issue WriteGroup for a completed (or partial) group buffer."""
        # Diag (2026-05-02): ICMS_SKIP_WRITES=1 short-circuits the
        # WriteGroup RPC entirely. Used to disambiguate whether a
        # server-side LOC_LEN_ERR occurs on a write-path SEND
        # (WriteGroup ACK) or a read-path SEND (Score reply / FetchAll).
        # Also lets us measure decode-only TPOT without Phase-2 KV
        # persistence — the prefix won't be reusable across sweeps but
        # the in-flight request still sees correct attention because
        # apply scattered the K pages into vLLM's local kv_cache before
        # WriteGroup fires.
        if os.environ.get("ICMS_SKIP_WRITES") == "1":
            rs = self._requests.get(request_id)
            if rs is not None:
                rs.active_group_buffers.pop(group_idx, None)
            return
        # BUG-N13 Phase 1 diag: catch the early-return paths.
        _diag_n13 = os.environ.get("ICMS_DIAG_N13", "0") == "1"
        rs = self._requests.get(request_id)
        if rs is None:
            if _diag_n13:
                logger.info("[diag-flush-attempted] rank=%d rid=%s gidx=%d "
                            "EARLY_RETURN=rs_missing", self._tp_rank,
                            request_id, group_idx)
            return
        buf = rs.active_group_buffers.pop(group_idx, None)
        if buf is None:
            if _diag_n13:
                logger.info("[diag-flush-attempted] rank=%d rid=%s gidx=%d "
                            "EARLY_RETURN=buf_missing buffers=%s",
                            self._tp_rank, request_id, group_idx,
                            sorted(rs.active_group_buffers.keys()))
            return
        chain_prefix = rs.chain[:group_idx + 1]
        if not chain_prefix:
            if _diag_n13:
                logger.info("[diag-flush-attempted] rank=%d rid=%s gidx=%d "
                            "EARLY_RETURN=empty_chain_prefix chain_len=%d",
                            self._tp_rank, request_id, group_idx,
                            len(rs.chain))
            return
        if _diag_n13:
            logger.info("[diag-flush-attempted] rank=%d rid=%s gidx=%d "
                        "partial=%s buf.filled=%d/%d chain_prefix_len=%d",
                        self._tp_rank, request_id, group_idx, partial,
                        len(buf.filled),
                        _GROUP_BLOCKS * (self._geom.num_layers if self._geom else 48),
                        len(chain_prefix))
        pages = _GROUP_BLOCKS if not partial else len({p for (_, p) in buf.filled})
        phase = "decode" if self._prefill_done else "prefill"
        logger.debug("[diag-flush] group=%d pages=%d partial=%s phase=%s "
                     "chain_len=%d num_groups_written(pre)=%d",
                     group_idx, pages, partial, phase,
                     len(chain_prefix), rs.num_groups_written)
        t0 = time.perf_counter()
        # ICMS_DIAG_CANARY=1: hash bytes at layer 0 / specific page
        # offsets right before shipping. Pair with [diag-canary-write]
        # (record_page side) and [server-canary-recv]. If these match
        # write canaries but server differs → wire transfer dropped
        # bytes. If they differ from write canaries → record_page
        # stored at wrong offset OR something clobbered buf.kv_blob.
        if os.environ.get("ICMS_DIAG_CANARY") == "1":
            import hashlib as _hl
            kv_blob_bytes = bytes(buf.kv_blob)
            kpb = self._geom.kv_page_bytes if self._geom else 32768
            kgb = self._geom.kv_group_bytes if self._geom else 32 * kpb
            for probe_pid in (0, 1, 2, 3, 4, 8, 16, 17, 24, 31, 32, 33, 100, 500):
                pg_idx = probe_pid // _GROUP_BLOCKS
                if pg_idx != group_idx:
                    continue
                page_in_group = probe_pid % _GROUP_BLOCKS
                off = 0 * kgb + page_in_group * kpb  # layer=0
                if off + 32 > len(kv_blob_bytes):
                    continue
                head32 = kv_blob_bytes[off:off + 32].hex()
                k_sha = _hl.sha1(
                    kv_blob_bytes[off:off + kpb // 2]).hexdigest()[:16]
                logger.info("[diag-flush-bytes] rid=%s gidx=%d "
                             "probe_pid=%d layer=0 off=%d "
                             "k_sha=%s head32=%s",
                             request_id, group_idx, probe_pid, off,
                             k_sha, head32)
        # Design B (2026-05-08): track local RPC success in a flag,
        # broadcast OUTSIDE the try, then gate ledger bumps on the
        # broadcasted (cross-rank-symmetric) value. Replaces the prior
        # in-try bumps that were skipped on rank-0 when write_group
        # raised, while rank>0's no-op SKIP path took the bumps —
        # producing divergent flushed_local / num_groups_written and
        # the TP=2 multi-rid `sample_tokens` hang. See
        # project_tp2_writegroup_asymmetry_2026-05-07 + Design B in
        # docs/icms_open_bugs_audit_2026-05-08.md.
        ok_local = True
        try:
            # Option W: only rank 0 sends to the server. All ranks still
            # update local bookkeeping (_record_stored_groups etc.) below.
            _diag_tp_write = os.environ.get("ICMS_DIAG_TP_WRITE", "0") == "1"
            if self._tp_size > 1 and self._tp_rank != 0:
                if _diag_tp_write:
                    logger.info(
                        "[diag-tp-write] rank=%d SKIP rid=%s gidx=%d "
                        "chain_prefix_len=%d partial=%s pages=%d "
                        "summary_bytes=%d kv_bytes=%d",
                        self._tp_rank, request_id, group_idx,
                        len(chain_prefix), partial, pages,
                        len(buf.summary_blob), len(buf.kv_blob))
            else:
                # 2026-05-11 Option-1: if this is a partial flush AND we
                # have buffered full groups for this rid, drain them
                # FIRST so the chain is built parent→tails in order.
                # The partial group below appends on top of the
                # committed batched groups.
                # Finding 2: dict read under `_write_batch_buf_lock`
                # for snapshot consistency. `_flush_write_batch_now`
                # itself re-acquires the lock for the pop.
                with self._write_batch_buf_lock:
                    _has_buffered = bool(self._write_batch_buf.get(
                        request_id))
                if partial and _has_buffered:
                    self._flush_write_batch_now(request_id)
                # v6 wire. WG#1 of a rid (group_idx==0) is a fresh
                # registration: parent=[], tail=full_rank_chain. Server
                # routes to insert() which handles cross-rid prefix
                # split + ref bumping. WG#k>1 is streaming extension:
                # parent=rank_chain[:-1], tail=[last]. Server routes to
                # extend() with swap-on-extend semantics. Both keep
                # per-rid contribution to each node at +1, so one Evict
                # at request-finish frees the whole chain.
                rank_chain = self._rank_chain(chain_prefix)
                if group_idx == 0:
                    parent_chain = []
                    new_tail_groups = list(rank_chain)
                else:
                    parent_chain = rank_chain[:-1]
                    new_tail_groups = [rank_chain[-1]]
                if _diag_tp_write:
                    _ck_first = rank_chain[0] if rank_chain else None
                    _ck_last = rank_chain[-1] if rank_chain else None
                    logger.info(
                        "[diag-tp-write] rank=%d WRITE rid=%s gidx=%d "
                        "tp_size=%d chain_len=%d ck_first=%s ck_last=%s "
                        "parent_len=%d tail_len=%d pages=%d "
                        "summary_bytes=%d kv_bytes=%d partial=%s",
                        self._tp_rank, request_id, group_idx,
                        self._tp_size, len(rank_chain),
                        _ck_first, _ck_last,
                        len(parent_chain), len(new_tail_groups),
                        pages, len(buf.summary_blob),
                        len(buf.kv_blob), partial)
                # 2026-05-11 Option-1: batched WriteGroup. Buffer full
                # consecutive groups (partial=False) and send K-at-a-time.
                # Default N=1 → no batching. N>1 → defer RPC + post-RPC
                # bookkeeping to `_flush_write_batch_now`, which handles
                # all N groups atomically (one RPC + one bookkeeping pass
                # per group). Partial flushes always go through the
                # single-RPC path; if there's a pending batch when a
                # partial flush hits, the partial path drains the batch
                # FIRST (see end of this function).
                batch_n = int(os.environ.get("ICMS_WRITE_BATCH_N", "1"))
                if batch_n > 1 and not partial:
                    # Finding 2: setdefault+append + length-check must
                    # be ONE atomic critical section, else two threads
                    # could each observe `len < batch_n` and both
                    # decide NOT to drain even when their combined
                    # count crosses the threshold. Compute the
                    # post-append count under the lock; perform the
                    # drain decision OUTSIDE the lock so
                    # `_flush_write_batch_now`'s own lock acquire
                    # (for the pop) doesn't dead-nest.
                    with self._write_batch_buf_lock:
                        self._write_batch_buf.setdefault(
                            request_id, []).append({
                            'group_idx': int(group_idx),
                            'summary_blob': bytes(buf.summary_blob),
                            'kv_blob': bytes(buf.kv_blob),
                            'pages': int(pages),
                            'chain_prefix': list(chain_prefix),
                            'partial': bool(partial),
                            't0': float(t0),
                        })
                        _post_append_n = len(
                            self._write_batch_buf[request_id])
                    if _post_append_n >= batch_n:
                        # Drain now — sends one batched RPC + records
                        # outcome for every buffered group.
                        self._flush_write_batch_now(request_id)
                    # In either case, the bookkeeping for this group is
                    # handled by `_flush_write_batch_now` (now or later).
                    # Skip the per-group post-RPC tail below.
                    return
                # Landmine #1 split: WG → _write_client + _write_rpc_lock.
                with self._write_rpc_lock:  # X1 (race-audit): serialize tx_/rx_
                    self._write_client.write_group(
                        parent_chain, new_tail_groups,
                        bytes(buf.summary_blob), bytes(buf.kv_blob),
                        pages_in_group=pages,
                    )
        except Exception as e:
            ok_local = False
            logger.warning("WriteGroup failed for req=%s group=%d: %s",
                           request_id, group_idx, e)
            # ICMS_TRACE: failure branch trace.
            if _ICMS_TRACE_ENABLED:
                try:
                    _wg_pids_fail = [int(group_idx) * _GROUP_BLOCKS + i
                                     for i in range(_GROUP_BLOCKS)][:16]
                    _icms_trace(
                        "writegroup_commit", request_id, layer=-1,
                        chain_fp=_icms_chain_fp(chain_prefix),
                        extra={
                            "group_idx": int(group_idx),
                            "page_ids_in_group": _wg_pids_fail,
                            "success": False,
                            "err": str(e),
                            "partial": bool(partial),
                        })
                except Exception:
                    pass

        # Always record flush latency — this is observation, not a
        # bookkeeping bump, so it should fire whether or not the RPC
        # succeeded so timing histograms stay representative.
        self.stats.record_flush((time.perf_counter() - t0) * 1e6)
        # ICMS_DIAG_FULLTRACE: emit per-flush summary so we can see exactly
        # which (rid, group_idx) made it to the server (locally) before
        # the deferred TP broadcast.
        if _ICMS_FULLTRACE_ENABLED:
            try:
                _icms_fulltrace(
                    "flush_group", rid=request_id,
                    group_idx=int(group_idx),
                    ok_local=bool(ok_local),
                    partial=bool(partial),
                    pages=int(pages),
                    chain_prefix_len=int(len(chain_prefix)),
                    chain_prefix_head=[int(x) for x in chain_prefix[:4]],
                    chain_prefix_tail=[int(x) for x in chain_prefix[-4:]],
                    chain_fp=_icms_chain_fp(list(chain_prefix)),
                    num_groups_written_pre=int(rs.num_groups_written),
                    flushed_local_pre=int(getattr(rs, "flushed_local", -1)),
                )
            except Exception:
                pass

        # 2026-05-09 N2 deferral: instead of running `_tp_broadcast_bool`
        # inline (which is a NCCL op on the TP comm and races iter N+1's
        # forward-thread NCCL → CUDA hang at TP>1), enqueue an entry
        # for the forward thread to broadcast and apply the bumps on
        # the next `wait_for_pending_writes` call. The forward thread
        # is the only legal place for collective NCCL ops on the TP
        # comm (vLLM owns that thread's stream).
        #
        # Snapshot rs.chain at flush time — rs.chain may mutate before
        # the forward thread drains the queue. chain_prefix is already
        # a slice of rs.chain (line ~7882), so we copy it explicitly.
        with self._pending_flush_lock:
            self._pending_flush_q.append((
                request_id, int(group_idx), bool(ok_local),
                bool(partial), int(pages),
                tuple(chain_prefix),
            ))
    def direct_write_group(self, request_id: str, chain: list[int],
                            summary_blob: bytes, kv_blob: bytes):
        rs = self._requests.setdefault(
            request_id, _RequestState(request_id=request_id))
        rs.chain = list(chain)
        # Option W: only rank 0 RPCs; other ranks no-op but keep local state.
        if self._tp_size > 1 and self._tp_rank != 0:
            return None
        # v6 wire: direct write is always a fresh registration
        # (parent=[], tail=full chain). Used by smoke tests that put
        # a complete chain in one shot.
        rank_chain = self._rank_chain(chain)
        # Landmine #1 split: WG → _write_client + _write_rpc_lock.
        with self._write_rpc_lock:  # X1 (race-audit): serialize tx_/rx_
            return self._write_client.write_group(
                [], list(rank_chain),
                summary_blob, kv_blob,
                pages_in_group=_GROUP_BLOCKS)
