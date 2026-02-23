# SPDX-License-Identifier: Apache-2.0
"""Self-contained io_uring wrapper using raw Linux syscalls via ctypes.

Works on any Linux 5.6+ kernel (x86_64).  Zero external dependencies.
Only supports IORING_OP_READ for the SSD KV-cache offload read path.

Usage::

    ring = IoUring(entries=128)
    ring.prep_read(fd, buf_addr, nbytes, file_offset, user_data=0)
    ring.prep_read(fd, buf_addr2, nbytes2, file_offset2, user_data=1)
    results = ring.submit_and_wait(2)   # [(user_data, bytes_read), ...]
    ring.close()
"""

import ctypes
import ctypes.util
import os
from typing import ClassVar

# ---------------------------------------------------------------------------
# Syscall numbers (x86_64)
# ---------------------------------------------------------------------------
SYS_io_uring_setup = 425
SYS_io_uring_enter = 426

# ---------------------------------------------------------------------------
# io_uring constants
# ---------------------------------------------------------------------------
IORING_OP_READ = 22
IORING_ENTER_GETEVENTS = 1

# mmap offsets (from include/uapi/linux/io_uring.h)
IORING_OFF_SQ_RING = 0
IORING_OFF_CQ_RING = 0x8000000
IORING_OFF_SQES = 0x10000000

# Feature flags
IORING_FEAT_SINGLE_MMAP = 1 << 0

# mmap constants
MAP_SHARED = 0x01
MAP_POPULATE = 0x08000
PROT_READ = 0x1
PROT_WRITE = 0x2

# ---------------------------------------------------------------------------
# libc bindings
# ---------------------------------------------------------------------------
_libc = ctypes.CDLL(
    ctypes.util.find_library("c") or "libc.so.6", use_errno=True)

_syscall = _libc.syscall
_syscall.restype = ctypes.c_long

_mmap = _libc.mmap
_mmap.restype = ctypes.c_void_p
_mmap.argtypes = [
    ctypes.c_void_p,  # addr
    ctypes.c_size_t,  # length
    ctypes.c_int,     # prot
    ctypes.c_int,     # flags
    ctypes.c_int,     # fd
    ctypes.c_long,    # offset (off_t)
]

_munmap = _libc.munmap
_munmap.restype = ctypes.c_int
_munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


# ---------------------------------------------------------------------------
# Kernel ABI structs
# ---------------------------------------------------------------------------

class _SqringOffsets(ctypes.Structure):
    _fields_ = [
        ("head",         ctypes.c_uint32),
        ("tail",         ctypes.c_uint32),
        ("ring_mask",    ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("flags",        ctypes.c_uint32),
        ("dropped",      ctypes.c_uint32),
        ("array",        ctypes.c_uint32),
        ("resv1",        ctypes.c_uint32),
        ("resv2",        ctypes.c_uint64),
    ]


class _CqringOffsets(ctypes.Structure):
    _fields_ = [
        ("head",         ctypes.c_uint32),
        ("tail",         ctypes.c_uint32),
        ("ring_mask",    ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("overflow",     ctypes.c_uint32),
        ("cqes",         ctypes.c_uint32),
        ("flags",        ctypes.c_uint32),
        ("resv1",        ctypes.c_uint32),
        ("resv2",        ctypes.c_uint64),
    ]


class _IoUringParams(ctypes.Structure):
    _fields_ = [
        ("sq_entries",     ctypes.c_uint32),
        ("cq_entries",     ctypes.c_uint32),
        ("flags",          ctypes.c_uint32),
        ("sq_thread_cpu",  ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32),
        ("features",       ctypes.c_uint32),
        ("wq_fd",          ctypes.c_uint32),
        ("resv",           ctypes.c_uint32 * 3),
        ("sq_off",         _SqringOffsets),
        ("cq_off",         _CqringOffsets),
    ]


class _IoUringSqe(ctypes.Structure):
    """Submission Queue Entry — 64 bytes."""
    _fields_ = [
        ("opcode",        ctypes.c_uint8),    # 0
        ("flags",         ctypes.c_uint8),    # 1
        ("ioprio",        ctypes.c_uint16),   # 2
        ("fd",            ctypes.c_int32),    # 4
        ("off",           ctypes.c_uint64),   # 8
        ("addr",          ctypes.c_uint64),   # 16
        ("len",           ctypes.c_uint32),   # 24
        ("rw_flags",      ctypes.c_uint32),   # 28
        ("user_data",     ctypes.c_uint64),   # 32
        ("buf_index",     ctypes.c_uint16),   # 40
        ("personality",   ctypes.c_uint16),   # 42
        ("splice_fd_in",  ctypes.c_int32),    # 44
        ("addr3",         ctypes.c_uint64),   # 48
        ("__pad2",        ctypes.c_uint64),   # 56
    ]

assert ctypes.sizeof(_IoUringSqe) == 64


class _IoUringCqe(ctypes.Structure):
    """Completion Queue Entry — 16 bytes."""
    _fields_ = [
        ("user_data", ctypes.c_uint64),  # 0
        ("res",       ctypes.c_int32),   # 8
        ("flags",     ctypes.c_uint32),  # 12
    ]

assert ctypes.sizeof(_IoUringCqe) == 16


# ---------------------------------------------------------------------------
# IoUring class
# ---------------------------------------------------------------------------

class IoUring:
    """Minimal io_uring wrapper for batched O_DIRECT reads.

    Thread safety: NOT thread-safe.  Use one instance per worker thread.
    """

    _SQE_SIZE: ClassVar[int] = ctypes.sizeof(_IoUringSqe)
    _CQE_SIZE: ClassVar[int] = ctypes.sizeof(_IoUringCqe)

    def __init__(self, entries: int = 128):
        self._ring_fd: int = -1
        self._mmap_regions: list[tuple[int, int]] = []  # (addr, size)

        # ---- io_uring_setup ----
        params = _IoUringParams()
        ctypes.memset(ctypes.addressof(params), 0, ctypes.sizeof(params))

        ret = _syscall(
            ctypes.c_long(SYS_io_uring_setup),
            ctypes.c_uint(entries),
            ctypes.byref(params),
        )
        if ret < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"io_uring_setup failed: {os.strerror(errno)}")

        self._ring_fd = ret
        self.sq_entries: int = params.sq_entries
        self.cq_entries: int = params.cq_entries

        # ---- mmap the rings ----
        prot = PROT_READ | PROT_WRITE
        mflags = MAP_SHARED | MAP_POPULATE

        sq_ring_sz = params.sq_off.array + params.sq_entries * 4
        cq_ring_sz = params.cq_off.cqes + params.cq_entries * self._CQE_SIZE
        sqes_sz = params.sq_entries * self._SQE_SIZE

        single_mmap = bool(params.features & IORING_FEAT_SINGLE_MMAP)

        if single_mmap:
            ring_sz = max(sq_ring_sz, cq_ring_sz)
            sq_ring_ptr = _mmap(None, ring_sz, prot, mflags,
                                self._ring_fd, IORING_OFF_SQ_RING)
            if sq_ring_ptr == ctypes.c_void_p(-1).value:
                self._cleanup()
                raise OSError(ctypes.get_errno(), "mmap SQ/CQ ring failed")
            self._mmap_regions.append((sq_ring_ptr, ring_sz))
            cq_ring_ptr = sq_ring_ptr
        else:
            sq_ring_ptr = _mmap(None, sq_ring_sz, prot, mflags,
                                self._ring_fd, IORING_OFF_SQ_RING)
            if sq_ring_ptr == ctypes.c_void_p(-1).value:
                self._cleanup()
                raise OSError(ctypes.get_errno(), "mmap SQ ring failed")
            self._mmap_regions.append((sq_ring_ptr, sq_ring_sz))

            cq_ring_ptr = _mmap(None, cq_ring_sz, prot, mflags,
                                self._ring_fd, IORING_OFF_CQ_RING)
            if cq_ring_ptr == ctypes.c_void_p(-1).value:
                self._cleanup()
                raise OSError(ctypes.get_errno(), "mmap CQ ring failed")
            self._mmap_regions.append((cq_ring_ptr, cq_ring_sz))

        sqes_ptr = _mmap(None, sqes_sz, prot, mflags,
                         self._ring_fd, IORING_OFF_SQES)
        if sqes_ptr == ctypes.c_void_p(-1).value:
            self._cleanup()
            raise OSError(ctypes.get_errno(), "mmap SQEs failed")
        self._mmap_regions.append((sqes_ptr, sqes_sz))

        # ---- Compute pointers into the mmap'd regions ----
        sq_off = params.sq_off
        cq_off = params.cq_off

        # SQ ring pointers
        self._sq_head = ctypes.c_uint32.from_address(sq_ring_ptr + sq_off.head)
        self._sq_tail = ctypes.c_uint32.from_address(sq_ring_ptr + sq_off.tail)
        self._sq_mask = ctypes.c_uint32.from_address(
            sq_ring_ptr + sq_off.ring_mask).value
        SqArray = ctypes.c_uint32 * params.sq_entries
        self._sq_array = SqArray.from_address(sq_ring_ptr + sq_off.array)

        # CQ ring pointers
        self._cq_head = ctypes.c_uint32.from_address(cq_ring_ptr + cq_off.head)
        self._cq_tail = ctypes.c_uint32.from_address(cq_ring_ptr + cq_off.tail)
        self._cq_mask = ctypes.c_uint32.from_address(
            cq_ring_ptr + cq_off.ring_mask).value

        # SQE and CQE arrays
        SqeArray = _IoUringSqe * params.sq_entries
        self._sqes = SqeArray.from_address(sqes_ptr)

        CqeArray = _IoUringCqe * params.cq_entries
        self._cqes = CqeArray.from_address(cq_ring_ptr + cq_off.cqes)

        # Local tail tracker (only written to shared tail on submit)
        self._local_sq_tail: int = 0

    def prep_read(self, fd: int, buf_addr: int, nbytes: int,
                  offset: int, user_data: int = 0) -> None:
        """Prepare one IORING_OP_READ SQE.

        Args:
            fd: File descriptor (must be opened with O_DIRECT for DIO).
            buf_addr: Integer address of the destination buffer
                (e.g. ``tensor.data_ptr() + byte_offset``).
                Must be page-aligned for O_DIRECT.
            nbytes: Number of bytes to read.
            offset: File offset to read from.
            user_data: Opaque tag returned in the CQE.
        """
        idx = self._local_sq_tail & self._sq_mask
        sqe = self._sqes[idx]
        # Zero the SQE to clear any stale fields.
        ctypes.memset(ctypes.addressof(sqe), 0, self._SQE_SIZE)
        sqe.opcode = IORING_OP_READ
        sqe.fd = fd
        sqe.off = offset
        sqe.addr = buf_addr
        sqe.len = nbytes
        sqe.user_data = user_data
        self._sq_array[idx] = idx
        self._local_sq_tail += 1

    def submit_and_wait(self, nr_submit: int) -> list[tuple[int, int]]:
        """Submit *nr_submit* SQEs and wait for all completions.

        Returns:
            List of ``(user_data, result)`` tuples.
            ``result`` is the number of bytes read on success,
            or a negative errno on failure.
        """
        return self.submit_and_wait_partial(nr_submit, nr_submit)

    def submit_and_wait_partial(self, nr_submit: int,
                                min_complete: int) -> list[tuple[int, int]]:
        """Submit *nr_submit* SQEs and wait for at least *min_complete* CQEs.

        Returns all CQEs available at the time of return (may be more than
        *min_complete*).

        Args:
            nr_submit: Number of new SQEs to submit (may be 0 to just wait).
            min_complete: Minimum CQEs to wait for before returning.
        """
        # Publish the SQEs by updating the shared tail.
        self._sq_tail.value = self._local_sq_tail

        flags = IORING_ENTER_GETEVENTS if min_complete > 0 else 0
        ret = _syscall(
            ctypes.c_long(SYS_io_uring_enter),
            ctypes.c_uint(self._ring_fd),
            ctypes.c_uint(nr_submit),
            ctypes.c_uint(min_complete),
            ctypes.c_uint(flags),
            ctypes.c_void_p(0),  # sigset
            ctypes.c_size_t(0),  # sigset size
        )
        if ret < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"io_uring_enter failed: {os.strerror(errno)}")

        return self._harvest_cqes()

    def _harvest_cqes(self) -> list[tuple[int, int]]:
        """Drain all available CQEs from the completion queue (non-blocking).

        Returns:
            List of ``(user_data, result)`` tuples.
        """
        results: list[tuple[int, int]] = []
        head = self._cq_head.value
        tail = self._cq_tail.value
        while head != tail:
            idx = head & self._cq_mask
            cqe = self._cqes[idx]
            results.append((cqe.user_data, cqe.res))
            head += 1
        # Advance the CQ head so the kernel can reuse these slots.
        self._cq_head.value = head
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        for addr, size in self._mmap_regions:
            _munmap(addr, size)
        self._mmap_regions.clear()
        if self._ring_fd >= 0:
            os.close(self._ring_fd)
            self._ring_fd = -1

    def close(self) -> None:
        """Release the io_uring ring and all mmap'd memory."""
        self._cleanup()

    def __del__(self) -> None:
        self._cleanup()

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @staticmethod
    def available() -> bool:
        """Return True if io_uring is supported on this kernel."""
        params = _IoUringParams()
        ctypes.memset(ctypes.addressof(params), 0, ctypes.sizeof(params))
        ret = _syscall(
            ctypes.c_long(SYS_io_uring_setup),
            ctypes.c_uint(1),
            ctypes.byref(params),
        )
        if ret < 0:
            return False
        # Got a valid ring fd — close it immediately.
        os.close(ret)
        return True
