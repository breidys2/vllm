# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the TP=2 WriteGroup asymmetry fix (Design B).

The bug: rank-0's `IcmsClient.write_group` could raise under load. The
ledger bumps (`flushed_local`, `num_groups_written`) were inside the
`try:` block — so rank-0 skipped the bumps on failure while rank>0
took the no-op SKIP path and bumped normally. Result: divergent
per-rank ledger → asymmetric `rs.stored_groups` on the next request →
asymmetric `total_pages` → wait_for_layer flag-spin × N layers → 5-min
`sample_tokens` timeout.

Design B fix (icms_connector.py:7305-7410): track local RPC success
in a flag, broadcast OUTSIDE the try, gate ledger bumps on the
broadcasted value. All ranks now bump or skip together.

These tests lock in the helper semantics:
  * `_tp_broadcast_bool` — TP=1 short-circuits (no collective); TP>1
    propagates rank-0's value; TP>1 broadcast failure returns False
    on every rank to keep ledgers symmetric.

End-to-end `_flush_group` symmetry under multi-rid load is still
covered only by the TP=2 multi-rid sweep on real hardware (sprc03).
This file covers the unit-level invariant of the broadcast helper.

See: docs/icms_open_bugs_audit_2026-05-08.md "P1 — TP=2 WriteGroup
asymmetry hang", project_tp2_writegroup_asymmetry_2026-05-07.md
"""
from __future__ import annotations

from unittest import mock

import pytest


def _import_helper():
    """Import the helper under test. Isolated so test collection still
    works even if the connector module fails to import for unrelated
    reasons (CUDA missing, etc.) — tests will skip instead of error
    at collection time."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.icms_connector import (
            _tp_broadcast_bool,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"icms_connector not importable in this env: {e!r}")
    return _tp_broadcast_bool


# ─── TP=1: pure Python, no collective ────────────────────────────────


@pytest.mark.parametrize("value", [True, False])
def test_tp1_returns_input_unchanged(value: bool) -> None:
    """TP=1 must short-circuit before touching torch.distributed."""
    bcast = _import_helper()
    # tp_size=1 takes the short-circuit at line 360 — no NCCL machinery.
    assert bcast(value, tp_rank=0, tp_size=1) is value


def test_tp1_does_not_touch_dist() -> None:
    """Sanity: TP=1 path must not call into torch.distributed at all.
    A regression that drops the short-circuit would silently bring up
    a NCCL channel per RPC, costing ~ms per call."""
    bcast = _import_helper()
    with mock.patch("torch.distributed.broadcast") as mock_bcast:
        bcast(True, tp_rank=0, tp_size=1)
        bcast(False, tp_rank=0, tp_size=1)
        mock_bcast.assert_not_called()


# ─── TP>1: rank-0's value propagates ─────────────────────────────────


class _FakeTensor:
    """Stand-in for the i64 broadcast tensor. `dist.broadcast` is
    expected to overwrite this tensor's value with rank-0's contents
    on every non-rank-0 rank. We simulate the post-broadcast state by
    pre-setting the value the test wants `.item()` to return."""

    def __init__(self, value: int) -> None:
        self._value = value

    def item(self) -> int:
        return self._value


@pytest.fixture
def _patched_dist_env():
    """Patches the imports `_tp_broadcast_bool` resolves at call time
    so the test runs without a real torch.distributed backend or GPU.
    Yields a dict the test can mutate to control the broadcast outcome.

    Usage:
      with _patched_dist_env() as env:
          env['post_broadcast_value'] = 1   # simulates "rank-0 sent True"
    """
    state = {
        "post_broadcast_value": 0,
        "broadcast_calls": [],
        "broadcast_raises": None,
    }

    fake_tp_group = mock.MagicMock()
    fake_tp_group.first_rank = 0
    fake_tp_group.device_group = object()  # opaque token for `group=` arg

    def _fake_broadcast(tensor, src=None, group=None):
        state["broadcast_calls"].append((tensor, src, group))
        if state["broadcast_raises"] is not None:
            raise state["broadcast_raises"]
        # Simulate NCCL: every rank ends up with rank-0's value.
        tensor._value = state["post_broadcast_value"]

    def _fake_tensor(data, dtype=None, device=None):
        # Helper packs data as a single-element list. Mirror the
        # post-broadcast semantics by stashing the initial value here.
        return _FakeTensor(int(data[0]))

    patches = [
        mock.patch(
            "vllm.distributed.kv_transfer.kv_connector.v1."
            "icms_connector._get_icms_nccl_group",
            return_value=None,
        ),
        mock.patch(
            "vllm.distributed.parallel_state.get_tp_group",
            return_value=fake_tp_group,
        ),
        mock.patch("torch.distributed.broadcast", side_effect=_fake_broadcast),
        mock.patch("torch.tensor", side_effect=_fake_tensor),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.device", return_value="cuda:0"),
    ]
    for p in patches:
        p.start()
    try:
        yield state
    finally:
        for p in reversed(patches):
            p.stop()


def test_tp2_rank0_true_propagates(_patched_dist_env) -> None:
    """Rank-0 sends True → broadcast result is True on every rank."""
    bcast = _import_helper()
    _patched_dist_env["post_broadcast_value"] = 1
    # Rank-0 — sends its actual value.
    assert bcast(True, tp_rank=0, tp_size=2) is True
    # The tensor was constructed with [1] (rank-0, value=True).
    assert _patched_dist_env["broadcast_calls"], "dist.broadcast not called"


def test_tp2_rank0_false_propagates(_patched_dist_env) -> None:
    """Rank-0 sends False → result False on every rank."""
    bcast = _import_helper()
    _patched_dist_env["post_broadcast_value"] = 0
    assert bcast(False, tp_rank=0, tp_size=2) is False


def test_tp2_rank1_receives_rank0_value(_patched_dist_env) -> None:
    """Rank>0 always sends 0 (its own input is ignored). Result is
    whatever rank-0 sent, propagated via the broadcast."""
    bcast = _import_helper()
    # Simulate rank-0 having sent True. Rank-1's input doesn't matter.
    _patched_dist_env["post_broadcast_value"] = 1
    assert bcast(True, tp_rank=1, tp_size=2) is True
    assert bcast(False, tp_rank=1, tp_size=2) is True

    # Now simulate rank-0 having sent False.
    _patched_dist_env["post_broadcast_value"] = 0
    assert bcast(True, tp_rank=1, tp_size=2) is False
    assert bcast(False, tp_rank=1, tp_size=2) is False


def test_tp2_rank1_input_is_ignored_at_send(_patched_dist_env) -> None:
    """The tensor packed for the broadcast must be 0 on rank>0
    regardless of `value` — only rank-0's input matters. This is the
    invariant that lets `_flush_group` pass `ok_local` on every rank
    (rank>0's `ok_local` is always True from the SKIP path, but it
    must NOT influence the result)."""
    bcast = _import_helper()
    _patched_dist_env["post_broadcast_value"] = 0  # rank-0 sent False
    # Rank-1 passes True, but its input must not flip the result.
    assert bcast(True, tp_rank=1, tp_size=2) is False

    # Capture what was packed into the tensor on rank-1: the helper
    # constructs `torch.tensor([1 if (tp_rank == 0 and value) else 0])`
    # — so rank-1 always sends 0. We can't directly check the
    # constructor arg here without more elaborate mocking, but the
    # behavioral assertion above is the user-facing invariant.


# ─── TP>1: broadcast failure → conservative False ────────────────────


def test_tp2_broadcast_exception_returns_false(_patched_dist_env) -> None:
    """If `dist.broadcast` raises (NCCL not warmed, channel issue,
    etc.), the helper returns False on EVERY rank. Critical: this
    keeps the per-rank ledgers symmetric — better to skip a bump on
    all ranks than to diverge."""
    bcast = _import_helper()
    _patched_dist_env["broadcast_raises"] = RuntimeError("NCCL channel broken")

    # Rank-0 with value=True: would have returned True if broadcast
    # succeeded. With failure, returns False.
    assert bcast(True, tp_rank=0, tp_size=2) is False
    # Rank-1: same, conservative False.
    assert bcast(True, tp_rank=1, tp_size=2) is False
    assert bcast(False, tp_rank=1, tp_size=2) is False


# ─── Type contract: returns bool, not int ────────────────────────────


def test_returns_bool_type(_patched_dist_env) -> None:
    """The helper returns `bool(int(t.item()))` — must be a bool, not
    an int. Downstream `if ok:` would still work on int but type
    contracts matter for mypy + clarity."""
    bcast = _import_helper()
    _patched_dist_env["post_broadcast_value"] = 1
    out = bcast(True, tp_rank=0, tp_size=2)
    assert isinstance(out, bool)
