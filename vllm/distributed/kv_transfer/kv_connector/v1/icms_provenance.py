"""KV-block provenance tracing for ICMS.

Gated by env `ICMS_TRACE_KV_PROVENANCE`:
  - unset / empty / "0": disabled (zero cost).
  - "1" or "warn":       log violations to stderr; continue.
  - "raise":             log + raise AssertionError on first violation.

The tracker records, per request:
  * `allocated`: every block_id vLLM allocated (in order).
  * `ext_comp_tokens` + `local_cached_tokens` + `page_tokens`: enough to
    derive the ext_comp slice of `allocated` — the range that vLLM
    leaves uninitialized expecting ICMS to populate it.
  * `icms_populated[layer_idx]`: block_ids that ICMS apply scattered
    into for that layer.

Invariant (checked at `wait_for_layer` end, per layer):
    For every block in the attention bt that lies in the ext_comp
    range, it MUST be in `icms_populated[this_layer]`.

A violation means attention will read uninitialized / stale free-pool
KV at those block positions. Most commonly fires for non-scored layers
(when `scored_layers_mask` is set) because `wait_for_layer` short-
circuits without populating ext_comp slots, leaving the natural bt
attended over uninitialized blocks.

Paper launchers (mask=0) trip none of these because every layer's bt
is trimmed by ICMS to its top-k pages (all ICMS-populated), so the
ext_comp ∩ bt set equals the populated set. The tracer makes that
property *verified* rather than *assumed*.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import defaultdict
from typing import Iterable, Optional

logger = logging.getLogger("icms.provenance")


def _mode() -> str:
    return os.environ.get("ICMS_TRACE_KV_PROVENANCE", "").strip().lower()


def is_enabled() -> bool:
    return _mode() in ("1", "true", "warn", "raise")


def is_raise() -> bool:
    return _mode() == "raise"


class ProvenanceTracker:
    """Per-process tracker, thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # rid → in-order block_ids
        self.ordered: dict[str, list[int]] = {}
        # rid → counts
        self.ext_comp_tokens: dict[str, int] = {}
        self.local_cached_tokens: dict[str, int] = {}
        self.page_tokens: dict[str, int] = {}
        # rid → layer_idx → set[block_id]
        self.icms_populated: dict[str, dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set))
        # rolling stats
        self.violations: int = 0
        self.layers_checked: int = 0
        self.records_alloc: int = 0
        self.records_scatter: int = 0
        # per-request running violation count (for per-request summary)
        self.violations_by_rid: dict[str, int] = defaultdict(int)

    # ── recording sites ─────────────────────────────────────────────

    def record_alloc(
        self,
        rid: str,
        block_ids: Iterable[int],
        num_external_tokens: int,
        num_local_cached_tokens: int,
        page_tokens: int,
    ) -> None:
        if not is_enabled():
            return
        with self._lock:
            self.ordered[rid] = [int(b) for b in block_ids]
            self.ext_comp_tokens[rid] = int(num_external_tokens)
            self.local_cached_tokens[rid] = int(num_local_cached_tokens)
            self.page_tokens[rid] = int(page_tokens)
            self.records_alloc += 1

    def record_icms_populated(
        self,
        rid: str,
        layer_idx: int,
        block_ids: Iterable[int],
    ) -> None:
        if not is_enabled():
            return
        if layer_idx is None or layer_idx < 0:
            return
        with self._lock:
            self.icms_populated[rid][int(layer_idx)].update(
                int(b) for b in block_ids if int(b) >= 0)
            self.records_scatter += 1

    # ── check site ───────────────────────────────────────────────────

    def _ext_comp_block_set(self, rid: str) -> set[int]:
        """Derive the ext_comp slice of allocated block_ids."""
        ord_bids = self.ordered.get(rid)
        pt = self.page_tokens.get(rid, 0)
        n_local = self.local_cached_tokens.get(rid, 0)
        n_ext = self.ext_comp_tokens.get(rid, 0)
        if not ord_bids or pt <= 0 or n_ext <= 0:
            return set()
        start = n_local // pt
        # ceil for the upper bound — partial trailing pages still
        # need population
        end = (n_local + n_ext + pt - 1) // pt
        if end > len(ord_bids):
            end = len(ord_bids)
        return set(ord_bids[start:end])

    def check_bt(
        self,
        rid: str,
        layer_idx: Optional[int],
        bt_block_ids: Iterable[int],
        path: str = "?",
    ) -> int:
        """Return count of ext_comp blocks in bt that ICMS did not
        populate for this layer. Logs a violation when >0."""
        if not is_enabled():
            return 0
        if layer_idx is None or layer_idx < 0:
            return 0
        with self._lock:
            self.layers_checked += 1
            bt = {int(b) for b in bt_block_ids if int(b) >= 0}
            if not bt:
                return 0
            ext_blocks = self._ext_comp_block_set(rid)
            if not ext_blocks:
                return 0
            icms_pop = self.icms_populated.get(rid, {}).get(
                int(layer_idx), set())
            unpopulated_ext = (bt & ext_blocks) - icms_pop
            if not unpopulated_ext:
                return 0
            self.violations += 1
            self.violations_by_rid[rid] += 1
            sample = sorted(unpopulated_ext)[:8]
            msg = (
                f"[ICMS_PROVENANCE] VIOLATION rid={rid[:8]} "
                f"layer={layer_idx} path={path} "
                f"n_unpop_ext={len(unpopulated_ext)} "
                f"bt_size={len(bt)} ext_size={len(ext_blocks)} "
                f"icms_pop_size={len(icms_pop)} sample={sample}"
                f"{'...' if len(unpopulated_ext) > 8 else ''}"
            )
            print(msg, file=sys.stderr, flush=True)
            logger.warning(msg)
            if is_raise():
                raise AssertionError(msg)
            return len(unpopulated_ext)

    # ── lifecycle ────────────────────────────────────────────────────

    def clear_request(self, rid: str) -> None:
        if not is_enabled():
            return
        with self._lock:
            self.ordered.pop(rid, None)
            self.ext_comp_tokens.pop(rid, None)
            self.local_cached_tokens.pop(rid, None)
            self.page_tokens.pop(rid, None)
            self.icms_populated.pop(rid, None)
            self.violations_by_rid.pop(rid, None)

    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": is_enabled(),
                "mode": _mode() or "off",
                "violations": int(self.violations),
                "layers_checked": int(self.layers_checked),
                "records_alloc": int(self.records_alloc),
                "records_scatter": int(self.records_scatter),
                "tracked_rids": len(self.ordered),
            }


_TRACKER = ProvenanceTracker()


def tracker() -> ProvenanceTracker:
    return _TRACKER


def reset_for_tests() -> None:
    """Test-only: wipe state."""
    global _TRACKER
    _TRACKER = ProvenanceTracker()
