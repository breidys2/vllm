"""icms_block_locator — inverse map block_id → ChainLocator for the
ICMS eviction-mode write path (PR0b of the eviction refactor).

WHY THIS EXISTS
---------------
Under prefill-mode ICMS, the writer knows (rid, group_idx, page_in_group)
at save_kv_layer time because the prefill loop's hook supplies them
directly. Under eviction-mode, the trigger source is
vLLM's block_pool.free_blocks callback, which only knows the raw
scheduler-side block_id. This module is the inverse index that lets the
connector resolve a freshly-evicted block_id back into a ChainLocator
that the BF2 server understands.

The map is GREENFIELD per PR0b — icms_provenance is per-rid metadata
and does NOT contain a block_id → locator inversion. The reviewers
flagged three adversarial cases that drive the API:

  1. **rid_clear_before_evict** — the rid finishes (its scheduler-side
     metadata is cleared) BEFORE its blocks evict. Without a snapshot,
     the lookup at eviction time has nowhere to resolve. Fix:
     `clear_request(rid)` snapshots the rid's locator entries into a
     bounded retention store; `evict_lookup` consults live map first,
     then snapshot.

  2. **block_id reuse** — vLLM may reassign a block_id to a different
     rid (either after eviction OR via direct touch reassignment without
     going through eviction). Fix: `insert(block_id, ...)` is a write —
     last-writer-wins. The previous owner's entry must have been
     snapshotted already if its rid finished, or it was evicted before
     reuse (in which case `evict_lookup` already cleared it). An assert
     catches the "alive-rid still owns block_id" overlap as a bug.

  3. **TP-rank-local KV partitions** — block_id is shared across TP
     ranks (it's vLLM's logical block ID); the per-rank KV bytes differ
     but the locator (rid, group_idx, page_in_group) is identical across
     ranks. This module is rank-agnostic; the per-rank byte copy lives
     in icms_connector_worker_write under PR5.

LIFECYCLE
---------
  insert(block_id, rid, group_idx, page_in_group) ─►
     map[block_id] = ChainLocator(rid, group_idx, page_in_group, ts)

  clear_request(rid) ─►
     for each block_id whose locator.rid == rid:
        snapshot[block_id] = map[block_id]
     # do NOT remove from map yet — block may still get reused or
     # evicted before reuse, both paths handle map correctly.

  evict_lookup(block_id) ─► Optional[ChainLocator]
     loc = map.pop(block_id) or snapshot.pop(block_id) or None
     return loc

  prune_snapshots(now_step, max_age_steps) ─► int
     remove any snapshot entry whose ts < now_step - max_age_steps

CONCURRENCY
-----------
vLLM v1's scheduler is single-threaded; the KV cache manager + connector
both run on it. No internal locking is required. The asserts catch
"called from wrong thread" bugs by assuming single-writer semantics.

GROUP_PAGES = 32 (per storage_service/python/icms_client/geometry.py).
A vLLM block = one ICMS page = 16 tokens. group_idx is the floor-division
of the page's logical position in the prompt by GROUP_PAGES;
page_in_group is the modulo. Both are bounded ≥ 0.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Mirror of storage_service/python/icms_client/geometry.py:13.
# Duplicated here so this module does not import icms_client at module
# load time (icms_client requires the _ensure_icms_client_on_path
# bootstrap in icms_connector.py to have run first; see [Connector
# decomposition 2026-05-29] for the entry-order trap).
_GROUP_PAGES = 32


@dataclass(frozen=True, slots=True)
class ChainLocator:
    """Identifies one KV page within an ICMS chain.

    rid             — vLLM request id this page was first allocated for.
                      Survives request finish via snapshot.
    group_idx       — floor(logical_page_idx / GROUP_PAGES).
    page_in_group   — logical_page_idx % GROUP_PAGES, in [0, GROUP_PAGES).
    snapshot_step   — set by clear_request; the engine step at which the
                      rid finished. Used by prune_snapshots to bound
                      retention. 0 means "still live (not snapshotted)".
    """

    rid: str
    group_idx: int
    page_in_group: int
    snapshot_step: int = 0


class BlockLocator:
    """Inverse map block_id → ChainLocator, with rid-aware snapshot
    retention so finished-rid blocks resolve correctly when they
    finally evict.

    All methods are O(1) amortized. The retention store is keyed by
    block_id; if the same block_id is reused for a new rid before the
    snapshot is consulted, the live-map insert overwrites correctly
    (new rid wins; the old snapshot under that block_id is dropped
    only when `evict_lookup` pops it OR when the new rid touch-replaces
    it via `insert` — see _maybe_drop_snapshot).
    """

    def __init__(self):
        # Live map: block_id → ChainLocator with snapshot_step == 0.
        self._live: dict[int, ChainLocator] = {}
        # Snapshot map: block_id → ChainLocator with snapshot_step > 0.
        # Populated by clear_request, drained by evict_lookup or
        # prune_snapshots.
        self._snapshot: dict[int, ChainLocator] = {}
        # rid → set of block_ids it owned at last clear_request.
        # Used so clear_request is O(per-rid block count) instead of
        # O(live map size). Updated incrementally in insert / evict.
        self._rid_blocks: dict[str, set[int]] = {}
        # Stats — for telemetry surface PR12 / Reviewer 2 recommendation.
        self.stats: dict[str, int] = {
            "inserts": 0,
            "evict_hits_live": 0,
            "evict_hits_snapshot": 0,
            "evict_misses": 0,
            "snapshots_created": 0,
            "snapshots_dropped_by_reuse": 0,
            "snapshots_pruned_by_age": 0,
            "alive_rid_overlap": 0,
        }

    # ─────────────────── Mutation ───────────────────

    def insert(self, block_id: int, rid: str,
               group_idx: int, page_in_group: int) -> None:
        """Record that block_id now holds (rid, group_idx, page_in_group).

        Overwrites any prior live entry — last-writer-wins. If the prior
        owner's rid is the same as `rid`, that's a re-insertion (no-op
        semantically); if different, see _maybe_drop_snapshot for the
        reuse-collision handling.
        """
        if block_id < 0:
            raise ValueError(f"block_id must be non-negative, got {block_id}")
        if not (0 <= page_in_group < _GROUP_PAGES):
            raise ValueError(
                f"page_in_group {page_in_group} out of [0, {_GROUP_PAGES})")
        if group_idx < 0:
            raise ValueError(f"group_idx must be non-negative, got {group_idx}")

        prior = self._live.get(block_id)
        if prior is not None and prior.rid != rid:
            # Block_id reused without going through eviction.
            #
            # PR7a of ICMS eviction-mode refactor: move the OLD owner's
            # entry to snapshot. The reuse case happens when vLLM
            # LRU-evicts a cached block belonging to rid A and
            # immediately reassigns it to rid B inside the same
            # schedule() call (allocate path). The GPU page at this
            # block_id still holds A's KV bytes — B's forward pass
            # will overwrite them only LATER in this same step. Until
            # the eviction callback drains, the inverse map MUST be
            # able to resolve block_id → A's ChainLocator so the
            # worker can copy A's KV to BF2 before B's forward
            # clobbers it. Pre-PR7a this entry was lost (live
            # overwritten, snapshot dropped), so eviction-mode would
            # ferry incorrect (B's) provenance + corrupt cross-turn
            # cache reads for A's chain.
            self.stats["alive_rid_overlap"] += 1
            blocks = self._rid_blocks.get(prior.rid)
            if blocks is not None:
                blocks.discard(block_id)
                if not blocks:
                    del self._rid_blocks[prior.rid]
            # snapshot_step=0 distinguishes "live-from-reuse" from
            # clear_request snapshots (which carry the step number).
            # Either kind is acceptable to evict_lookup, but tooling
            # can differentiate via the field.
            self._snapshot[block_id] = ChainLocator(
                prior.rid, prior.group_idx, prior.page_in_group, 0)
            self.stats["snapshots_created_by_reuse"] = (
                self.stats.get("snapshots_created_by_reuse", 0) + 1)
        elif block_id in self._snapshot:
            # Same-rid re-insertion: a no-op semantically. Drop any
            # stale snapshot entry from a PRIOR (older) reuse cycle
            # under this same rid (snapshot for rid B becomes stale
            # if rid B re-inserts; only the most recent live entry
            # matters going forward, and same-rid implies no chain
            # mismatch).
            existing_snap = self._snapshot[block_id]
            if existing_snap.rid == rid:
                del self._snapshot[block_id]
                self.stats["snapshots_dropped_by_reuse"] += 1

        self._live[block_id] = ChainLocator(rid, group_idx, page_in_group, 0)
        self._rid_blocks.setdefault(rid, set()).add(block_id)
        self.stats["inserts"] += 1

    def evict_lookup(self, block_id: int) -> Optional[ChainLocator]:
        """Pop and return the ChainLocator for block_id, or None.

        PR7a of ICMS eviction-mode refactor: consult **snapshot first**.

        When block_id was freed in this same schedule() step and is in
        the eviction pipeline, the live entry may have been overwritten
        by a same-step reuse (rid B's allocate claimed the block before
        drain ran). The OLD owner's locator was moved to snapshot by
        insert(). The GPU page still holds A's KV until B's forward
        writes, so the worker needs A's chain — that's what's in
        snapshot. Returning live's B-entry would ferry B's locator
        under A's KV bytes (cross-contaminated chain), which would
        leak B's KV under A's chain on cache hit.

        After draining the snapshot entry, leave any live entry intact
        — future evictions of this block_id (B's normal eviction lifecycle)
        will resolve via the standard live → snapshot path.
        """
        loc = self._snapshot.pop(block_id, None)
        if loc is not None:
            self.stats["evict_hits_snapshot"] += 1
            return loc
        loc = self._live.pop(block_id, None)
        if loc is not None:
            self.stats["evict_hits_live"] += 1
            blocks = self._rid_blocks.get(loc.rid)
            if blocks is not None:
                blocks.discard(block_id)
                if not blocks:
                    del self._rid_blocks[loc.rid]
            return loc
        self.stats["evict_misses"] += 1
        return None

    def clear_request(self, rid: str, now_step: int) -> int:
        """Snapshot all live blocks owned by rid as finished-at now_step.

        Returns the count of blocks snapshotted. Subsequent eviction
        of any of these block_ids will resolve via the snapshot path
        until either (a) `evict_lookup` pops them or (b) `insert`
        reuses the block_id for a different rid (snapshot dropped per
        _maybe_drop_snapshot) or (c) `prune_snapshots` ages them out.
        """
        blocks = self._rid_blocks.pop(rid, None)
        if blocks is None:
            return 0
        snapped = 0
        # Materialize the iterable so we can mutate self._live safely.
        for block_id in list(blocks):
            loc = self._live.pop(block_id, None)
            if loc is None:
                # Already evicted between insert and clear_request — fine.
                continue
            self._snapshot[block_id] = ChainLocator(
                loc.rid, loc.group_idx, loc.page_in_group, now_step or 1)
            snapped += 1
        self.stats["snapshots_created"] += snapped
        return snapped

    def prune_snapshots(self, now_step: int, max_age_steps: int) -> int:
        """Drop snapshot entries older than max_age_steps.

        Returns the count pruned. Called periodically by the connector
        (e.g., from get_finished or on_step_end) to bound the snapshot
        store under long-running sessions. max_age_steps should be
        chosen ≥ the worst-case lag between request_finish and
        block_pool.free_blocks calling the eviction callback for that
        block — under vLLM v1 LRU this is typically <100 steps.
        """
        if not self._snapshot:
            return 0
        cutoff = now_step - max_age_steps
        stale = [bid for bid, loc in self._snapshot.items()
                 if loc.snapshot_step < cutoff]
        for bid in stale:
            del self._snapshot[bid]
        n = len(stale)
        self.stats["snapshots_pruned_by_age"] += n
        return n

    # ─────────────────── Read-only ───────────────────

    def __len__(self) -> int:
        return len(self._live) + len(self._snapshot)

    @property
    def live_count(self) -> int:
        return len(self._live)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshot)

    @property
    def tracked_rids(self) -> int:
        return len(self._rid_blocks)

    def peek_live(self, block_id: int) -> Optional[ChainLocator]:
        """Non-destructive lookup — for testing and stats."""
        return self._live.get(block_id)

    def peek_snapshot(self, block_id: int) -> Optional[ChainLocator]:
        return self._snapshot.get(block_id)

    # ─────────────────── PR5 bulk-insert helper ───────────────────

    def insert_request_blocks(self, rid: str,
                              block_ids: list[int]) -> int:
        """PR5 of ICMS eviction-mode refactor: bulk-insert a request's
        block_ids in chain order.

        vLLM allocates blocks to a request sequentially: the i-th
        block_id covers logical page i of the request's prompt. Page i
        belongs to group floor(i / GROUP_PAGES) at position
        i % GROUP_PAGES inside that group. This helper applies that
        mapping in one call so the scheduler doesn't have to compute
        it at every allocation site.

        Returns the count of blocks inserted (mirrors `insert` calls).
        Idempotent re-insertion of the SAME (block_id, rid, group, page)
        is a no-op semantically — the underlying `insert` is
        last-writer-wins so calling this repeatedly with the same
        sequence has no negative effect.
        """
        n = 0
        for i, bid in enumerate(block_ids):
            if bid < 0:
                continue
            group_idx = i // _GROUP_PAGES
            page_in_group = i % _GROUP_PAGES
            self.insert(bid, rid, group_idx, page_in_group)
            n += 1
        return n
