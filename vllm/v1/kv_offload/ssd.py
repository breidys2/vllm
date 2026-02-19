# SPDX-License-Identifier: Apache-2.0
"""SSD-backed KV cache offloading with two transfer modes.

Mode 1 — CPU bounce buffer (default):
  Store: GPU -> CPU pinned (swap_blocks) -> SSD file (O_DIRECT pwrite)
  Load:  SSD file (O_DIRECT pread) -> CPU pinned -> GPU (swap_blocks)

Mode 2 — GPUDirect Storage (use_gds=True, requires kvikio):
  Store: GPU -> SSD file (cuFile DMA, no CPU in data path)
  Load:  SSD file -> GPU  (cuFile DMA, no CPU in data path)

Storage is a set of pre-allocated files (one per KV cache tensor) on a
filesystem mount point.  Block IDs map directly to file offsets.

For NVMe-oF: mount remote targets at the same path -- no code changes.

Usage::

    kv_connector_extra_config={
        "spec_name": "SSDOffloadingSpec",
        "ssd_path": "/mnt/kv_cache",
        "ssd_bytes_to_use": 100 * 1024**3,
        "block_size": 48,
        "use_gds": True,  # optional: enable GPUDirect Storage
    }
"""

import concurrent.futures
import ctypes
import os
import pathlib
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

try:
    import kvikio
    HAS_KVIKIO = True
except ImportError:
    HAS_KVIKIO = False

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Medium / block types
# ---------------------------------------------------------------------------

class SSDLoadStoreSpec(BlockIDsLoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "SSD"


class SSDBlockStatus(BlockStatus):
    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)]

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id


# ---------------------------------------------------------------------------
# Backend (scheduler-side block allocator)
# ---------------------------------------------------------------------------

class SSDBackend(Backend):
    def __init__(self, block_size: int, num_blocks: int):
        super().__init__(block_size=block_size, medium=SSDLoadStoreSpec.medium())
        self.num_blocks = num_blocks
        self.num_allocated_blocks = 0
        self.allocated_blocks_free_list: list[int] = []

    def get_num_free_blocks(self):
        return (
            len(self.allocated_blocks_free_list)
            + self.num_blocks
            - self.num_allocated_blocks
        )

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        num_fresh = min(
            len(block_hashes), self.num_blocks - self.num_allocated_blocks
        )
        num_reused = len(block_hashes) - num_fresh
        assert len(self.allocated_blocks_free_list) >= num_reused

        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(SSDBlockStatus(self.num_allocated_blocks))
            self.num_allocated_blocks += 1
        for _ in range(num_reused):
            blocks.append(SSDBlockStatus(self.allocated_blocks_free_list.pop()))
        return blocks

    def free(self, block: BlockStatus):
        assert isinstance(block, SSDBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus],
    ) -> LoadStoreSpec:
        return SSDLoadStoreSpec([b.block_id for b in blocks])  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _tensor_block_buf(tensor: torch.Tensor, sub_block_id: int,
                      stride_bytes: int, total_bytes: int) -> ctypes.Array:
    """Writable ctypes buffer pointing into a pinned CPU tensor.

    Args:
        tensor: The CPU tensor backing the buffer.
        sub_block_id: Index of the first sub-block to expose.
        stride_bytes: Byte size of ONE sub-block (used to compute offset).
        total_bytes: Total number of bytes to expose (may span multiple
            sub-blocks, i.e. ``total_bytes >= stride_bytes``).
    """
    ptr = tensor.data_ptr() + sub_block_id * stride_bytes
    return (ctypes.c_char * total_bytes).from_address(ptr)


# ---------------------------------------------------------------------------
# Transfer handlers (worker-side)
# ---------------------------------------------------------------------------

@dataclass
class _SsdTransfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    wall_elapsed: float | None = None  # precomputed wall-clock time (seconds)


class SsdStoreHandler(OffloadingHandler):
    """GPU -> CPU bounce -> SSD file."""

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        ssd_fds: list[int],
        gpu_block_size_factor: int,
        ssd_block_size_factor: int,
        block_sizes_bytes: list[int],
    ):
        self.gpu_tensors = gpu_tensors
        self.cpu_tensors = cpu_tensors
        self.ssd_fds = ssd_fds
        self.gpu_bsf = gpu_block_size_factor
        self.ssd_bsf = ssd_block_size_factor
        self.block_sizes = block_sizes_bytes
        self.total_block_size = sum(block_sizes_bytes)
        self._transfers: deque[_SsdTransfer] = deque()
        self._transfer_events: dict[int, torch.Event] = {}
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def _get_stream(self) -> torch.cuda.Stream:
        return self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()

    def _get_event(self) -> torch.Event:
        return (self._event_pool.pop() if self._event_pool
                else torch.Event(enable_timing=True))

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        gpu_spec, ssd_spec = spec
        gpu_blocks = gpu_spec.block_ids
        ssd_blocks = ssd_spec.block_ids

        # Build sub-block mapping (GPU sub-blocks -> CPU bounce sub-blocks).
        # CPU bounce and SSD file share the same sub-block numbering.
        gpu_sub_count = gpu_blocks.size * self.gpu_bsf
        ssd_sub_count = ssd_blocks.size * self.ssd_bsf
        skip = -ssd_blocks.size % self.gpu_bsf
        assert ssd_sub_count == gpu_sub_count - skip

        src_to_dst = np.empty((ssd_sub_count, 2), dtype=np.int64)
        expand_block_ids(gpu_blocks, self.gpu_bsf, src_to_dst[:, 0],
                         skip_count=skip)
        expand_block_ids(ssd_blocks, self.ssd_bsf, src_to_dst[:, 1])
        mapping = torch.from_numpy(src_to_dst)

        stream = self._get_stream()
        start_event = self._get_event()
        end_event = self._get_event()

        wall_start = time.perf_counter()

        # Step 1: GPU -> CPU bounce (async on CUDA stream).
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            start_event.record(stream)
            for gpu_t, cpu_t, bsz in zip(
                    self.gpu_tensors, self.cpu_tensors, self.block_sizes):
                ops.swap_blocks(gpu_t, cpu_t, bsz, mapping)

        # Wait for GPU->CPU DMA to complete.
        stream.synchronize()

        # Step 2: CPU bounce -> SSD (O_DIRECT pwrite, one I/O per block).
        total_bytes = 0
        for cpu_t, fd, bsz in zip(
                self.cpu_tensors, self.ssd_fds, self.block_sizes):
            for ssd_blk in ssd_blocks:
                start_sub = int(ssd_blk) * self.ssd_bsf
                nbytes = self.ssd_bsf * bsz
                buf = _tensor_block_buf(cpu_t, start_sub, bsz, nbytes)
                os.pwrite(fd, buf, start_sub * bsz)
                total_bytes += nbytes

        wall_elapsed = time.perf_counter() - wall_start

        # Record end event on the same stream as start for consistent
        # query()/synchronize() semantics.
        with torch.cuda.stream(stream):
            end_event.record(stream)
        self._transfer_events[job_id] = end_event
        self._transfers.append(
            _SsdTransfer(job_id, stream, start_event, end_event, total_bytes,
                         wall_elapsed=wall_elapsed))
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            t = self._transfers.popleft()
            # Use wall-clock time (includes GPU DMA + CPU pwrite) when
            # available; fall back to CUDA event timing.
            elapsed = (t.wall_elapsed if t.wall_elapsed is not None
                       else t.start_event.elapsed_time(t.end_event) * 1e-3)
            results.append(TransferResult(
                job_id=t.job_id, success=True,
                transfer_size=t.num_bytes, transfer_time=elapsed,
                transfer_type=("GPU", "SSD")))
            self._stream_pool.append(t.stream)
            self._event_pool.extend([t.start_event, t.end_event])
            del self._transfer_events[t.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for jid in job_ids:
            ev = self._transfer_events.get(jid)
            if ev is not None:
                ev.synchronize()


class SsdLoadHandler(OffloadingHandler):
    """SSD file -> CPU bounce -> GPU."""

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        ssd_fds: list[int],
        gpu_block_size_factor: int,
        ssd_block_size_factor: int,
        block_sizes_bytes: list[int],
    ):
        self.gpu_tensors = gpu_tensors
        self.cpu_tensors = cpu_tensors
        self.ssd_fds = ssd_fds
        self.gpu_bsf = gpu_block_size_factor
        self.ssd_bsf = ssd_block_size_factor
        self.block_sizes = block_sizes_bytes
        self.total_block_size = sum(block_sizes_bytes)
        self._transfers: deque[_SsdTransfer] = deque()
        self._transfer_events: dict[int, torch.Event] = {}
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def _get_stream(self) -> torch.cuda.Stream:
        return self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()

    def _get_event(self) -> torch.Event:
        return (self._event_pool.pop() if self._event_pool
                else torch.Event(enable_timing=True))

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        ssd_spec, gpu_spec = spec
        ssd_blocks = ssd_spec.block_ids
        gpu_blocks = gpu_spec.block_ids

        stream = self._get_stream()
        start_event = self._get_event()
        end_event = self._get_event()

        # Record start on the transfer stream so both events share the
        # same stream — this makes elapsed_time, query, and synchronize
        # work correctly.
        with torch.cuda.stream(stream):
            start_event.record(stream)

        # Step 1: SSD -> CPU bounce (blocking pread into pinned tensor).
        # Read directly into the page-aligned pinned CPU tensor so that
        # O_DIRECT alignment requirements are satisfied (os.pread returns
        # a heap-allocated bytes object that is NOT page-aligned).
        total_bytes = 0
        for cpu_t, fd, bsz in zip(
                self.cpu_tensors, self.ssd_fds, self.block_sizes):
            for ssd_blk in ssd_blocks:
                start_sub = int(ssd_blk) * self.ssd_bsf
                nbytes = self.ssd_bsf * bsz
                file_offset = start_sub * bsz
                buf = _tensor_block_buf(cpu_t, start_sub, bsz, nbytes)
                os.preadv(fd, [buf], file_offset)
                total_bytes += nbytes

        # Step 2: CPU bounce -> GPU (async on CUDA stream).
        ssd_sub_count = ssd_blocks.size * self.ssd_bsf
        gpu_sub_count = gpu_blocks.size * self.gpu_bsf
        skip = -gpu_blocks.size % self.ssd_bsf
        assert gpu_sub_count == ssd_sub_count - skip

        src_to_dst = np.empty((gpu_sub_count, 2), dtype=np.int64)
        expand_block_ids(ssd_blocks, self.ssd_bsf, src_to_dst[:, 0],
                         skip_count=skip)
        expand_block_ids(gpu_blocks, self.gpu_bsf, src_to_dst[:, 1])
        mapping = torch.from_numpy(src_to_dst)

        with torch.cuda.stream(stream):
            for cpu_t, gpu_t, bsz in zip(
                    self.cpu_tensors, self.gpu_tensors, self.block_sizes):
                ops.swap_blocks(cpu_t, gpu_t, bsz, mapping)
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            _SsdTransfer(job_id, stream, start_event, end_event, total_bytes))
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            t = self._transfers.popleft()
            # Both events are on the transfer stream.  elapsed_time
            # captures SSD pread + CPU->GPU DMA end-to-end.
            elapsed = t.start_event.elapsed_time(t.end_event) * 1e-3
            results.append(TransferResult(
                job_id=t.job_id, success=True,
                transfer_size=t.num_bytes, transfer_time=elapsed,
                transfer_type=("SSD", "GPU")))
            self._stream_pool.append(t.stream)
            self._event_pool.extend([t.start_event, t.end_event])
            del self._transfer_events[t.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for jid in job_ids:
            ev = self._transfer_events.get(jid)
            if ev is not None:
                ev.synchronize()


# ---------------------------------------------------------------------------
# GDS transfer handlers (worker-side, no CPU bounce)
# ---------------------------------------------------------------------------

class GdsStoreHandler(OffloadingHandler):
    """GPU -> SSD direct via GPUDirect Storage (cuFile)."""

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        ssd_files: list["kvikio.CuFile"],
        gpu_block_size_factor: int,
        ssd_block_size_factor: int,
        block_sizes_bytes: list[int],
        num_io_threads: int = 4,
    ):
        self.gpu_tensors = gpu_tensors
        self.ssd_files = ssd_files
        self.gpu_bsf = gpu_block_size_factor
        self.ssd_bsf = ssd_block_size_factor
        self.block_sizes = block_sizes_bytes
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_io_threads)
        self._pending: dict[int, concurrent.futures.Future] = {}

    def _do_store(self, job_id: int, mapping: np.ndarray,
                  total_bytes: int) -> TransferResult:
        t0 = time.perf_counter()
        for gpu_t, ssd_file, bsz in zip(
                self.gpu_tensors, self.ssd_files, self.block_sizes):
            for i in range(len(mapping)):
                gpu_sub = int(mapping[i, 0])
                ssd_sub = int(mapping[i, 1])
                ssd_file.write(
                    gpu_t,
                    size=bsz,
                    file_offset=ssd_sub * bsz,
                    dev_offset=gpu_sub * bsz,
                )
        elapsed = time.perf_counter() - t0
        return TransferResult(
            job_id=job_id, success=True,
            transfer_size=total_bytes, transfer_time=elapsed,
            transfer_type=("GPU", "SSD"))

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        gpu_spec, ssd_spec = spec
        gpu_blocks = gpu_spec.block_ids
        ssd_blocks = ssd_spec.block_ids

        gpu_sub_count = gpu_blocks.size * self.gpu_bsf
        ssd_sub_count = ssd_blocks.size * self.ssd_bsf
        skip = -ssd_blocks.size % self.gpu_bsf
        assert ssd_sub_count == gpu_sub_count - skip

        src_to_dst = np.empty((ssd_sub_count, 2), dtype=np.int64)
        expand_block_ids(gpu_blocks, self.gpu_bsf, src_to_dst[:, 0],
                         skip_count=skip)
        expand_block_ids(ssd_blocks, self.ssd_bsf, src_to_dst[:, 1])

        total_bytes = ssd_sub_count * sum(self.block_sizes)

        # Ensure GPU data is stable before cuFile reads it via DMA.
        torch.cuda.current_stream().synchronize()

        future = self._executor.submit(
            self._do_store, job_id, src_to_dst, total_bytes)
        self._pending[job_id] = future
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        done_ids: list[int] = []
        for jid, future in self._pending.items():
            if future.done():
                done_ids.append(jid)
                try:
                    results.append(future.result())
                except Exception:
                    logger.exception("GDS store job %d failed", jid)
                    results.append(TransferResult(
                        job_id=jid, success=False,
                        transfer_type=("GPU", "SSD")))
        for jid in done_ids:
            del self._pending[jid]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for jid in job_ids:
            future = self._pending.pop(jid, None)
            if future is not None:
                future.result()


class GdsLoadHandler(OffloadingHandler):
    """SSD -> GPU direct via GPUDirect Storage (cuFile)."""

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        ssd_files: list["kvikio.CuFile"],
        gpu_block_size_factor: int,
        ssd_block_size_factor: int,
        block_sizes_bytes: list[int],
        num_io_threads: int = 4,
    ):
        self.gpu_tensors = gpu_tensors
        self.ssd_files = ssd_files
        self.gpu_bsf = gpu_block_size_factor
        self.ssd_bsf = ssd_block_size_factor
        self.block_sizes = block_sizes_bytes
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_io_threads)
        self._pending: dict[int, concurrent.futures.Future] = {}

    def _do_load(self, job_id: int, mapping: np.ndarray,
                 total_bytes: int) -> TransferResult:
        t0 = time.perf_counter()
        for gpu_t, ssd_file, bsz in zip(
                self.gpu_tensors, self.ssd_files, self.block_sizes):
            for i in range(len(mapping)):
                ssd_sub = int(mapping[i, 0])
                gpu_sub = int(mapping[i, 1])
                ssd_file.read(
                    gpu_t,
                    size=bsz,
                    file_offset=ssd_sub * bsz,
                    dev_offset=gpu_sub * bsz,
                )
        elapsed = time.perf_counter() - t0
        return TransferResult(
            job_id=job_id, success=True,
            transfer_size=total_bytes, transfer_time=elapsed,
            transfer_type=("SSD", "GPU"))

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        ssd_spec, gpu_spec = spec
        ssd_blocks = ssd_spec.block_ids
        gpu_blocks = gpu_spec.block_ids

        ssd_sub_count = ssd_blocks.size * self.ssd_bsf
        gpu_sub_count = gpu_blocks.size * self.gpu_bsf
        skip = -gpu_blocks.size % self.ssd_bsf
        assert gpu_sub_count == ssd_sub_count - skip

        src_to_dst = np.empty((gpu_sub_count, 2), dtype=np.int64)
        expand_block_ids(ssd_blocks, self.ssd_bsf, src_to_dst[:, 0],
                         skip_count=skip)
        expand_block_ids(gpu_blocks, self.gpu_bsf, src_to_dst[:, 1])

        total_bytes = gpu_sub_count * sum(self.block_sizes)

        future = self._executor.submit(
            self._do_load, job_id, src_to_dst, total_bytes)
        self._pending[job_id] = future
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        done_ids: list[int] = []
        for jid, future in self._pending.items():
            if future.done():
                done_ids.append(jid)
                try:
                    results.append(future.result())
                except Exception:
                    logger.exception("GDS load job %d failed", jid)
                    results.append(TransferResult(
                        job_id=jid, success=False,
                        transfer_type=("SSD", "GPU")))
        for jid in done_ids:
            del self._pending[jid]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for jid in job_ids:
            future = self._pending.pop(jid, None)
            if future is not None:
                future.result()


# ---------------------------------------------------------------------------
# Handlers factory (allocates CPU bounce + opens SSD files)
# ---------------------------------------------------------------------------

class SsdGpuOffloadingHandlers:
    """Creates GPU<->SSD handlers.

    Two modes:
      use_gds=False (default): CPU pinned bounce buffers + O_DIRECT pwrite/pread
      use_gds=True:            GPUDirect Storage via kvikio (cuFile DMA, no CPU
                               in data path)
    """

    def __init__(
        self,
        gpu_block_size: int,
        ssd_block_size: int,
        num_ssd_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        ssd_path: str,
        use_o_direct: bool = True,
        use_gds: bool = False,
        gds_num_threads: int = 4,
    ):
        assert gpu_caches
        assert ssd_block_size % gpu_block_size == 0

        if use_gds and not HAS_KVIKIO:
            raise ImportError(
                "use_gds=True requires kvikio. "
                "Install with: pip install kvikio-cu12")

        # --- Parse GPU tensors (mirrors CpuGpuOffloadingHandlers) ---
        kernel_block_size: int | None = None
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []

        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256)

            has_layers_dim = False
            split_k_and_v = False
            if len(gpu_shape) != len(test_shape):
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                assert test_shape[0] == 2 and test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            try:
                stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=has_layers_dim)
                assert len(stride_order) == len(gpu_shape)
            except (AttributeError, NotImplementedError):
                stride_order = tuple(range(len(gpu_shape)))

            permuted = tuple(test_shape[i] for i in stride_order)
            bs_idx = permuted.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[bs_idx]
            else:
                kernel_block_size = gpu_shape[bs_idx]
                assert gpu_block_size % kernel_block_size == 0

            parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))

        assert kernel_block_size is not None
        ssd_block_size_factor = ssd_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size
        num_ssd_kernel_blocks = num_ssd_blocks * ssd_block_size_factor

        # --- Flatten GPU tensors (split K/V if needed) ---
        gpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            gpu_tensors.extend(
                gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor])

        min_bsf = min(gpu_block_size_factor, ssd_block_size_factor)
        block_sizes_bytes = [
            t.element_size() * t.stride(0) * min_bsf for t in gpu_tensors
        ]

        # --- Normalised block-size factors ---
        norm_gpu_bsf = gpu_block_size_factor // min_bsf
        norm_ssd_bsf = ssd_block_size_factor // min_bsf

        ssd_dir = pathlib.Path(ssd_path)
        ssd_dir.mkdir(parents=True, exist_ok=True)

        if use_gds:
            self._init_gds(
                gpu_tensors, ssd_dir, num_ssd_kernel_blocks,
                block_sizes_bytes, norm_gpu_bsf, norm_ssd_bsf,
                gds_num_threads)
        else:
            self._init_bounce(
                gpu_tensors, parsed_gpu_tensors, ssd_dir,
                num_ssd_kernel_blocks, block_sizes_bytes,
                norm_gpu_bsf, norm_ssd_bsf, use_o_direct)

    def _init_gds(
        self,
        gpu_tensors: list[torch.Tensor],
        ssd_dir: pathlib.Path,
        num_ssd_kernel_blocks: int,
        block_sizes_bytes: list[int],
        norm_gpu_bsf: int,
        norm_ssd_bsf: int,
        num_threads: int,
    ) -> None:
        """Initialise GPUDirect Storage mode (no CPU bounce buffers)."""
        logger.info("SSD offload: using GPUDirect Storage (kvikio) — "
                     "no CPU bounce buffers")

        self._ssd_fds = []  # not used in GDS mode
        self._gds_files: list[kvikio.CuFile] = []

        for i, bsz in enumerate(block_sizes_bytes):
            filepath = ssd_dir / f"kv_offload_{i}.bin"
            file_size = num_ssd_kernel_blocks * bsz
            # Pre-allocate the file (kvikio needs an existing file for r+).
            fd = os.open(str(filepath), os.O_RDWR | os.O_CREAT, 0o644)
            os.ftruncate(fd, file_size)
            os.close(fd)
            # Open with kvikio for cuFile DMA.
            gds_file = kvikio.CuFile(str(filepath), "r+")
            self._gds_files.append(gds_file)
            logger.info("  GDS file %s: %d blocks, %.1f MB (direct_io=%s)",
                        filepath, num_ssd_kernel_blocks, file_size / 1e6,
                        gds_file.is_direct_io_supported())

        self.gpu_to_ssd_handler = GdsStoreHandler(
            gpu_tensors, self._gds_files,
            norm_gpu_bsf, norm_ssd_bsf, block_sizes_bytes,
            num_io_threads=num_threads)
        self.ssd_to_gpu_handler = GdsLoadHandler(
            gpu_tensors, self._gds_files,
            norm_gpu_bsf, norm_ssd_bsf, block_sizes_bytes,
            num_io_threads=num_threads)

    def _init_bounce(
        self,
        gpu_tensors: list[torch.Tensor],
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]],
        ssd_dir: pathlib.Path,
        num_ssd_kernel_blocks: int,
        block_sizes_bytes: list[int],
        norm_gpu_bsf: int,
        norm_ssd_bsf: int,
        use_o_direct: bool,
    ) -> None:
        """Initialise CPU-bounce-buffer mode (original path)."""
        pin_memory = is_pin_memory_available()
        logger.info("SSD offload: using CPU bounce buffers (pinned=%s)",
                     pin_memory)

        cpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            cpu_shape = list(gpu_tensor.shape)
            cpu_shape[1 if split_k_and_v else 0] = num_ssd_kernel_blocks
            cpu_tensor = torch.zeros(
                cpu_shape, dtype=gpu_tensor.dtype,
                device="cpu", pin_memory=pin_memory)
            cpu_tensors.extend(
                cpu_tensor.unbind(0) if split_k_and_v else [cpu_tensor])

        flags = os.O_RDWR | os.O_CREAT
        if use_o_direct:
            try:
                flags |= os.O_DIRECT
            except AttributeError:
                logger.warning("O_DIRECT not available on this platform; "
                               "SSD I/O will go through page cache")

        self._gds_files = []  # not used in bounce mode
        self._ssd_fds: list[int] = []
        for i, bsz in enumerate(block_sizes_bytes):
            filepath = str(ssd_dir / f"kv_offload_{i}.bin")
            fd = os.open(filepath, flags, 0o644)
            file_size = num_ssd_kernel_blocks * bsz
            os.ftruncate(fd, file_size)
            self._ssd_fds.append(fd)
            logger.info("  SSD file %s: %d blocks, %.1f MB",
                        filepath, num_ssd_kernel_blocks, file_size / 1e6)

        self.gpu_to_ssd_handler = SsdStoreHandler(
            gpu_tensors, cpu_tensors, self._ssd_fds,
            norm_gpu_bsf, norm_ssd_bsf, block_sizes_bytes)
        self.ssd_to_gpu_handler = SsdLoadHandler(
            gpu_tensors, cpu_tensors, self._ssd_fds,
            norm_gpu_bsf, norm_ssd_bsf, block_sizes_bytes)

    def __del__(self):
        for fd in getattr(self, "_ssd_fds", []):
            try:
                os.close(fd)
            except OSError:
                pass
        for gf in getattr(self, "_gds_files", []):
            try:
                gf.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Offloading spec (entry point for the framework)
# ---------------------------------------------------------------------------

class SSDOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)

        ssd_path = self.extra_config.get("ssd_path")
        if not ssd_path:
            raise ValueError(
                "ssd_path must be specified in kv_connector_extra_config")

        ssd_bytes = self.extra_config.get("ssd_bytes_to_use")
        if not ssd_bytes:
            raise ValueError(
                "ssd_bytes_to_use must be specified in kv_connector_extra_config")

        # Calculate number of blocks.
        assert kv_cache_config is not None
        page_sizes = {
            g.kv_cache_spec.page_size_bytes
            for g in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )
        self.num_blocks = (
            int(ssd_bytes) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0 else 0
        )

        self._ssd_path = ssd_path
        self._use_o_direct = self.extra_config.get("use_o_direct", True)
        self._use_gds = self.extra_config.get("use_gds", False)
        self._gds_num_threads = self.extra_config.get("gds_num_threads", 4)
        self._manager: OffloadingManager | None = None
        self._handlers: SsdGpuOffloadingHandlers | None = None
        self.eviction_policy: str = self.extra_config.get(
            "eviction_policy", "lru")

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )
            backend = SSDBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_blocks,
            )
            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events)
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events)
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}")
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec],
                        OffloadingHandler]]:
        if not self._handlers:
            self._handlers = SsdGpuOffloadingHandlers(
                gpu_block_size=self.gpu_block_size,
                ssd_block_size=self.offloaded_block_size,
                num_ssd_blocks=self.num_blocks,
                gpu_caches=kv_caches,
                attn_backends=attn_backends,
                ssd_path=self._ssd_path,
                use_o_direct=self._use_o_direct,
                use_gds=self._use_gds,
                gds_num_threads=self._gds_num_threads,
            )
        assert self._handlers is not None
        yield GPULoadStoreSpec, SSDLoadStoreSpec, self._handlers.gpu_to_ssd_handler
        yield SSDLoadStoreSpec, GPULoadStoreSpec, self._handlers.ssd_to_gpu_handler
