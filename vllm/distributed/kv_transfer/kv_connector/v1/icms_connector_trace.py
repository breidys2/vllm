# SPDX-License-Identifier: Apache-2.0
"""ICMS connector tracing helpers + process-local stored-chain-queue bridge.

Extracted verbatim from icms_connector.py (behavior-preserving split). The
stored-chain queue (worker pipeline thread -> scheduler) lives here as the
single canonical owner so both icms_connector and icms_connector_scheduler
share ONE queue/cond/generation instance. Read _stored_chain_generation live
via this module's attribute (it is rebound by _bump/_append).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time

from vllm.logger import init_logger

# Log under the original connector logger name (behavior-preserving
# split): all split modules share one logger so log-name filtering,
# grep, and assertLogs see the same name as before.
logger = init_logger("vllm.distributed.kv_transfer.kv_connector.v1.icms_connector")


def _instr_timing(name):
    """Decorator: wrap a connector method with [INSTR] timing when
    ICMS_INSTR=1 is set. Logs only if duration > 1ms to avoid spam.
    Used to identify hot blockers on the TTFT critical path. Can be
    safely left in code — checks env var at every call, near-zero
    overhead when disabled.
    """
    import functools as _ft
    def _deco(fn):
        @_ft.wraps(fn)
        def _wrapper(self, *args, **kwargs):
            if os.environ.get("ICMS_INSTR", "0") != "1":
                return fn(self, *args, **kwargs)
            _t0 = time.perf_counter()
            try:
                return fn(self, *args, **kwargs)
            finally:
                _dt = (time.perf_counter() - _t0) * 1000.0
                if _dt > 1.0:
                    logger.info("[INSTR] %s: %.2fms", name, _dt)
        return _wrapper
    return _deco
_ICMS_TRACE_ENABLED = os.environ.get("ICMS_TRACE") == "1"
_ICMS_TRACE_LOCK = threading.Lock()
_ICMS_TRACE_FH = None
def _icms_trace(op, rid, layer=-1, chain_fp="", pc_before=-1, pc_after=-1,
                extra=None):
    """Append one JSON line to the ICMS trace file.

    Schema (must match C++ side):
      {"ts_ns":<int>,"side":"connector","op":"<op>","rid":"<rid>",
       "layer":<int_or_-1>,"chain_fp":"<hex>","pc_before":<int>,
       "pc_after":<int>,"tid":<int>,"extra":{...}}

    No-op when ICMS_TRACE != "1". Caller pays only the env-cached bool
    check on the hot path.
    """
    if not _ICMS_TRACE_ENABLED:
        return
    global _ICMS_TRACE_FH
    import json as _json_trace
    rec = {
        "ts_ns": time.monotonic_ns(),
        "side": "connector",
        "op": op,
        "rid": str(rid),
        "layer": int(layer) if layer is not None else -1,
        "chain_fp": chain_fp or "",
        "pc_before": int(pc_before) if pc_before is not None else -1,
        "pc_after": int(pc_after) if pc_after is not None else -1,
        "tid": threading.get_ident(),
        "extra": extra if extra is not None else {},
    }
    try:
        line = _json_trace.dumps(rec)
    except (TypeError, ValueError):
        # Fallback for non-serializable extras: stringify them.
        rec["extra"] = {k: str(v) for k, v in (extra or {}).items()}
        line = _json_trace.dumps(rec)
    with _ICMS_TRACE_LOCK:
        if _ICMS_TRACE_FH is None:
            _path = os.environ.get(
                "ICMS_TRACE_FILE", "/tmp/icms_trace_connector.jsonl")
            _ICMS_TRACE_FH = open(_path, "a", buffering=1)
        _ICMS_TRACE_FH.write(line + "\n")
_ICMS_FULLTRACE_ENABLED = os.environ.get("ICMS_DIAG_FULLTRACE") == "1"
_ICMS_FULLTRACE_LOCK = threading.Lock()
_ICMS_FULLTRACE_FH = None
def _icms_fulltrace(op, rid="", layer=-1, **extra):
    """Append a structured JSONL record for ICMS_DIAG_FULLTRACE=1.

    Emits to `${ICMS_DIAG_FULLTRACE_FILE}` (default
    /tmp/icms_fulltrace.jsonl). One line per call. All `extra` kwargs
    are JSON-serialized; non-serializable values are stringified.
    Caller pays only the env-cached bool check on the hot path.
    """
    if not _ICMS_FULLTRACE_ENABLED:
        return
    global _ICMS_FULLTRACE_FH
    import json as _json_ft
    rec = {
        "ts_ns": time.monotonic_ns(),
        "op": op,
        "rid": str(rid),
        "layer": int(layer) if layer is not None else -1,
        "tid": threading.get_ident(),
        "pid": os.getpid(),
        **extra,
    }
    try:
        line = _json_ft.dumps(rec)
    except (TypeError, ValueError):
        safe = {k: (v if isinstance(v, (int, float, str, bool, list, dict,
                                         type(None))) else str(v))
                for k, v in rec.items()}
        try:
            line = _json_ft.dumps(safe)
        except Exception:
            line = _json_ft.dumps({"op": op, "rid": str(rid),
                                    "layer": int(layer or -1),
                                    "_serialize_error": True})
    with _ICMS_FULLTRACE_LOCK:
        if _ICMS_FULLTRACE_FH is None:
            _path = os.environ.get(
                "ICMS_DIAG_FULLTRACE_FILE", "/tmp/icms_fulltrace.jsonl")
            _ICMS_FULLTRACE_FH = open(_path, "a", buffering=1)
        _ICMS_FULLTRACE_FH.write(line + "\n")
def _icms_chain_fp(chain) -> str:
    """Best-effort fingerprint of a chain (list of ints). Returns hex of
    a short blake2b digest of the chain's repr. Empty string on failure
    or empty input.
    """
    if not chain:
        return ""
    try:
        import hashlib as _hl_fp
        # chain entries are 64-bit ints; pack as decimal-comma string.
        # Cheap and stable across runs.
        b = ",".join(str(int(x)) for x in chain).encode("ascii")
        return _hl_fp.blake2b(b, digest_size=8).hexdigest()
    except Exception:
        return ""
def _allow_batch() -> bool:
    """ICMS_ALLOW_BATCH=1 gates the multi-rid (N>=2) batching path.

    When unset (default), the connector behaves exactly as the
    single-request implementation: per-rid `set_active`, first-rid-only
    extract, global `_skip_extract`. When set:
      - wait_for_layer aggregates per-rid trimmed states into one
        multi-row IcmsFetchState per layer (see icms_fetch_state.py).
      - extract_and_record walks every active rid (per-layer single
        AllGather at TP>1).
      - _skip_extract becomes a per-rid set, not a global bool.

    The flag is read fresh on each call so tests can flip it without a
    process restart. See docs/icms_vllm_integration_audit_2026-05-05.md
    for the Phase A landing plan.
    """
    return os.environ.get("ICMS_ALLOW_BATCH") == "1"
def _pack_fetch_bitmap(fetched: "set[int]", total_pages: int) -> bytes:
    """Pack a set of page IDs into the decode-mode bitmap wire format.

    bit n of byte n/8 (LSB-first) ⇒ page_id n is "already fetched" on
    the client. Sized to ceil(total_pages / 8) bytes. Page IDs outside
    [0, total_pages) are ignored. M3 calls this on a stride-group's
    fetched_pages set to build the wire suffix for decode-mode Score.
    """
    n_bytes = (total_pages + 7) // 8
    if n_bytes == 0 or not fetched:
        return b"\x00" * n_bytes
    bm = bytearray(n_bytes)
    for pid in fetched:
        if 0 <= pid < total_pages:
            bm[pid >> 3] |= 1 << (pid & 0x7)
    return bytes(bm)
_stored_chain_queue: list[tuple[list[int], int]] = []
_stored_chain_lock: threading.Lock = threading.Lock()
_stored_chain_cond: threading.Condition = threading.Condition(_stored_chain_lock)
_stored_chain_generation: int = 0
def _bump_stored_chain_generation() -> None:
    """Increment the generation counter and wake any waiter. Called by
    the worker pipeline thread immediately after every
    `_stored_chain_queue.append`."""
    global _stored_chain_generation
    with _stored_chain_cond:
        _stored_chain_generation += 1
        _stored_chain_cond.notify_all()
def _append_stored_chain_queue(chain: list[int], n_groups: int) -> None:
    """Atomically append to the worker→scheduler chain queue, bump the
    generation counter, and wake any waiter. Producers must use this
    helper rather than bare `_stored_chain_queue.append(...)` so a
    concurrent drain can't iterate-then-clear past this notification."""
    global _stored_chain_generation
    with _stored_chain_cond:
        _stored_chain_queue.append((chain, n_groups))
        _stored_chain_generation += 1
        _stored_chain_cond.notify_all()
def _drain_stored_chain_queue() -> list[tuple[list[int], int]]:
    """Atomically snapshot + clear the worker→scheduler chain queue.
    Returns the snapshot for the caller to process OUTSIDE the lock
    (so the lock isn't held across `record_stored_chain` work).
    Replaces the bare `for ... in _stored_chain_queue: ... ;
    _stored_chain_queue.clear()` pattern, which was racy: a producer
    append landing between the iteration's end and the clear was
    silently wiped (Bug #5)."""
    with _stored_chain_cond:
        if not _stored_chain_queue:
            return []
        snapshot = list(_stored_chain_queue)
        _stored_chain_queue.clear()
    return snapshot
