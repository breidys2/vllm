# SPDX-License-Identifier: Apache-2.0
"""Regression test for the pfcache=ON vs pfcache=OFF page-ID divergence bug.

The bug
-------
Under `enable_prefix_caching=True`, when vLLM's local prefix cache covers
the same range of tokens that ICMS has stored as a chain, the connector's
``get_num_new_matched_tokens`` subtracts ``num_computed_tokens`` from
``matched_tokens`` (icms_connector.py:2107) and returns ext_tokens=0.

Consequence: Path B is **skipped**, the Quest hook does NOT fire over
the haystack tokens (because vLLM has no forward pass to run over them),
and the only Q tensor that ever feeds Score is over the trailing
question tokens (typ. 48). Page-ID selection for the haystack is then
made on a Q that has zero haystack context, biased by Quest+RoPE
recency toward end-of-prompt pages.

Under ``enable_prefix_caching=False`` the same code path hands back
``ext_tokens = matched_tokens`` (because ``num_computed_tokens == 0``),
vLLM treats the chain prefix as ICMS-owned, allocates fresh blocks
for the remainder, runs forward over those (~32k tokens), and Quest
fires per stride over haystack content.  Page-IDs diverge.

Empirical witness (2026-05-14, sprc01 GPU 0, TP=1, qwen3 niah_multikey_2
h64000, seed=42):

  pfcache=ON  layer=0 first Score reply_pids[:5] = [1718, 1719, 1837, 1857, 2083]
  pfcache=OFF layer=0 first Score reply_pids[:5] = [1807, 1837, 1857, 2083, 2133]

  60/60 (idx, budget) tuples produce different generated text.
  pfcache=ON  predicts "8443229" (wrong);  pfcache=OFF predicts "1682375" (correct).
  total_pages = 4000 in BOTH (universe identical — H1 falsified).
  scored layers = {0,6,12,...,42} in BOTH (aggregation identical — H5 falsified).

  See results_accuracy/paper_run_2026-05-10/_probes/qwen3_pfcache_topk_div_2026-05-14/

What this test pins
-------------------
The semantic that two configs with identical chain state must produce
identical (matched_tokens, ext_tokens) flow into the Quest hook. The
test directly drives ``_Scheduler.get_num_new_matched_tokens`` with
the two ``num_computed_tokens`` values the bench presents and asserts
the connector reports a contract that does not allow ICMS Path B to
be silently elided when vLLM happens to have the same prefix cached.

FAILS on current main (the subtraction at L2107 zeroes ext_tokens when
``num_computed_tokens >= matched_tokens``).  PASSES on the proposed
fix where the connector either always returns ``matched_tokens`` or
otherwise signals vLLM to release the locally-cached blocks for the
chain range.
"""

from __future__ import annotations

import types
from unittest import mock

import pytest


def _import_sched():
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import icms_connector
    except Exception as e:  # pragma: no cover
        pytest.skip(f"icms_connector not importable: {e!r}")
    return icms_connector


def _make_request(rid: str, num_prompt_tokens: int):
    """Mimic the subset of vllm.v1.request.Request used by the path."""
    req = types.SimpleNamespace()
    req.request_id = rid
    req.num_tokens = num_prompt_tokens
    # The chain-hash code reads either `all_token_ids` or `prompt_token_ids`.
    req.all_token_ids = list(range(num_prompt_tokens))
    req.prompt_token_ids = req.all_token_ids
    req.num_computed_tokens = 0
    return req


def _make_sched_with_stored_chain(ic_mod, chain_groups: int):
    """Build a _Scheduler with a pre-populated stored-chain index that
    matches a deterministic chain we re-derive from a fixed prompt."""
    cfg = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(enable_prefix_caching=True),
    )
    sched = ic_mod._Scheduler.__new__(ic_mod._Scheduler)
    # Hand-build minimal state — avoids needing a full vllm_config.
    sched._vllm_config = cfg
    sched._chains = {}
    sched._block_ids = {}
    sched._pending_chain_sends = set()
    sched._prov_alloc = {}
    # `_stored_chains` is a list[(tuple_chain, n_groups)] (see
    # _lookup_stored_prefix), not a dict.
    sched._stored_chains = []
    sched._sched_step_count = 0
    # Pre-stash a chain of `chain_groups` group-hashes (matching the
    # prompt we will pass below) so `_lookup_stored_prefix` returns
    # `chain_groups`.
    GROUP_TOKENS = ic_mod._GROUP_BLOCKS * ic_mod.PAGE_TOKENS
    n_prompt = chain_groups * GROUP_TOKENS + 48  # +48 question tail
    import hashlib

    prev = b""
    full_chain: list[int] = []
    tokens = list(range(n_prompt))
    for g in range(chain_groups):
        start, end = g * GROUP_TOKENS, (g + 1) * GROUP_TOKENS
        h = hashlib.sha256()
        h.update(prev)
        h.update(repr(tuple(tokens[start:end])).encode())
        d = h.digest()
        full_chain.append(int.from_bytes(d[:8], "little"))
        prev = d
    # Stash so `_lookup_stored_prefix(chain)` returns chain_groups.
    sched._stored_chains.append((tuple(full_chain), chain_groups))
    return sched, n_prompt, full_chain


def test_pfcache_on_off_path_b_engagement_is_identical():
    """Contract: under a shared chain match, the two pfcache configs
    must hand the same ``ext_tokens`` to vLLM. Anything else means
    Quest stride hooks get different Q-coverage between the two configs
    (pfcache=ON sees only the 48-token question tail; pfcache=OFF sees
    the full 32k haystack tail), and the top-k page-ID set diverges
    deterministically across 60/60 examples.

    Empirical witness (2026-05-14, sprc01 GPU 0, TP=1, qwen3
    niah_multikey_2 h64000 seed=42; see results_accuracy/
    paper_run_2026-05-10/_probes/qwen3_pfcache_topk_div_2026-05-14/):

      pfcache=ON  [icms-nmt]:  matched=125 num_computed=64272 → ext=0
                              → Path B SKIPPED → Q over 48 tokens only
                              → predicts "8443229" (wrong)
      pfcache=OFF [icms-nmt]: matched=125 num_computed=0     → ext=32000
                              → Path B fires → Q over 32k haystack tail
                              → predicts "1682375" (correct)

    Status: FAILS on current main (ext_on=0 vs ext_off=64000).

    Status of the proposed fix at L2107: a naive ``ext_tokens =
    matched_tokens`` makes this unit test PASS but breaks vLLM end-to-
    end (scheduler.py:671 ``assert num_new_tokens > 0`` fires because
    local_match + ext > prompt_len). A production-safe fix must either
    plumb prefix-cache invalidation (``kv_cache_manager.evict_blocks``)
    to ``_Scheduler.on_alloc`` or have the connector signal to vLLM
    that the chain range is ICMS-owned even when local cache is
    present. See the END-TO-END probe at
    scripts/launchers/paper_run_2026-05-10/
    probe_qwen3_pfcache_topk_divergence_2026-05-14.sh for the smoke
    test that exercises the full path."""
    ic = _import_sched()
    sched, n_prompt, _chain = _make_sched_with_stored_chain(ic, chain_groups=125)

    req_on = _make_request("rid-pfcache-on", n_prompt)
    req_off = _make_request("rid-pfcache-off", n_prompt)

    # pfcache=ON: vLLM reports n_prompt-48 local match (its prefix cache
    # owns the chain prefix range).
    ext_on, _ = sched.get_num_new_matched_tokens(req_on, n_prompt - 48)
    # pfcache=OFF: no local prefix cache.
    ext_off, _ = sched.get_num_new_matched_tokens(req_off, 0)

    # Both regimes share the same server-side chain match. Path B
    # engagement must not silently disengage in one but not the other.
    # FAILS on current main: ext_on=0 != ext_off=64000.
    assert ext_on > 0, (
        "pfcache=ON returned ext_tokens=0 — Path B silently elided. "
        "Quest stride hooks will only see the 48-token question tail, "
        f"biasing page selection. (ext_off={ext_off})"
    )
    assert ext_on == ext_off, (
        f"Path B engagement asymmetric: ext_on={ext_on} vs "
        f"ext_off={ext_off}. These two configs share a chain match "
        "but route through different code paths in Quest."
    )
