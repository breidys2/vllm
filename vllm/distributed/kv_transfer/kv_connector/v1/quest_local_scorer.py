# SPDX-License-Identifier: Apache-2.0
"""Local (GPU/CPU) Quest-style page scorer.

This module implements the original Quest [Wang et al. 2024,
mit-han-lab/Quest] per-(KV-head) page-importance scoring as a pure
torch function, *independent* of the ICMS connector's pooled-Q +
RPC-to-BF2 path. It is gated behind ``ICMS_ORIGINAL_QUEST=1`` and
intended **only** for accuracy comparison: same downstream attention
path (per-layer page set fed into vLLM's PagedAttention), only the
page-selection algorithm differs.

Algorithm (per scored layer):

    score[h, p] = sum_d max(q[h, d] * kmin[p, h, d],
                            q[h, d] * kmax[p, h, d])

For GQA: query heads in a KV-head group are mean-pooled (matching the
ICMS path's GQA reduction so the comparison isolates Q-head pooling
vs per-(KV-head) scoring, not GQA awareness).

Tokens within the prefill chunk are mean-pooled (Quest reference does
*full attention* at prefill, so there's no "true" Quest behavior to
mimic at prefill; pooling tokens keeps the scoring tractable and
consistent with ICMS's prefill-chunk semantics).

Then: top-K per KV-head → set-union across heads → return the union
as a sorted list of page IDs. This is "option 1" from the design
discussion: per-layer union, fed to PagedAttention as-is. Since the
union is a strict superset of any single head's selection, accuracy
is an upper bound on Quest's faithful per-head sparse attention —
which is the right framing for "what did our scoring optimizations
cost vs Quest-style scoring", not "vs Quest end-to-end".

NOT in scope here:
  * Per-iter decode-time fresh top-K (Quest's actual decode path).
    Use mode (b/c) FetchAll-on-warm-prefill instead — decode runs
    dense over all selected pages, identical between modes.
  * Custom per-head paged-attention kernel. We intentionally feed
    the union to vLLM's existing per-layer PagedAttention.
"""
from __future__ import annotations

from typing import Iterable

import torch


def quest_score_local(
    q: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    k: int,
    *,
    num_kv_heads: int | None = None,
    exclude_pages: Iterable[int] | None = None,
) -> list[int]:
    """Quest-style per-(KV-head) top-K page selection, union across heads.

    Args:
      q: [T, H_q, D] (prefill chunk's queries) or [H_q, D] (already
         token-pooled). Will be mean-pooled across tokens if 3-D.
      kmin: [P, H_kv, D] fp16/fp32 (per-page key min over block).
      kmax: [P, H_kv, D] fp16/fp32 (per-page key max over block).
      k: top-K page budget *per KV head*. The returned union may have
         up to (k * H_kv) unique pages, but is bounded by P-|exclude|.
      num_kv_heads: if q has H_q != H_kv (GQA), pool query heads
         within each KV-head group via mean. Defaults to kmin.shape[1].
      exclude_pages: optional iterable of page IDs to omit from the
         result (the connector's per-layer fetched-bitmap, when
         applicable).

    Returns:
      Sorted list of unique page IDs in the union of per-head top-K.
    """
    if k <= 0 or kmin.shape[0] == 0:
        return []
    if num_kv_heads is None:
        num_kv_heads = kmin.shape[1]

    # ── 1. Normalize q to [H_kv, D] ───────────────────────────────────
    # Pool tokens (if present), then pool Q heads within each GQA group.
    if q.ndim == 3:
        q2 = q.mean(dim=0)  # [H_q, D]
    elif q.ndim == 2:
        q2 = q
    else:
        raise ValueError(f"q must be 2D [H_q,D] or 3D [T,H_q,D]; got {q.shape}")
    h_q, head_dim = q2.shape
    if h_q != num_kv_heads:
        if h_q < num_kv_heads or h_q % num_kv_heads != 0:
            raise ValueError(
                f"q has {h_q} heads, not divisible by num_kv_heads={num_kv_heads}")
        group = h_q // num_kv_heads
        q2 = q2.reshape(num_kv_heads, group, head_dim).mean(dim=1)  # [H_kv, D]

    # Align device first (so a CPU-tensor q with GPU summaries — or vice
    # versa — fails fast with a clear up-cast rather than later in a
    # broadcast multiply with a cryptic device-mismatch trace).
    q2 = q2.to(device=kmin.device, dtype=torch.float32)  # [H_kv, D]
    kmin_f = kmin.to(dtype=torch.float32)      # [P, H_kv, D]
    kmax_f = kmax.to(dtype=torch.float32)
    P = kmin_f.shape[0]

    # ── 2. Per-(h, p) score: sum_d max(q*kmin, q*kmax) ────────────────
    #   q2:    [H_kv, D]            → broadcast as [1, H_kv, D]
    #   kmin:  [P, H_kv, D]
    # term_min, term_max: [P, H_kv, D] → max → sum_d → [P, H_kv]
    qb = q2.unsqueeze(0)                                # [1, H_kv, D]
    term_min = qb * kmin_f                              # [P, H_kv, D]
    term_max = qb * kmax_f
    scores = torch.maximum(term_min, term_max).sum(dim=-1)  # [P, H_kv]

    # ── 3. Apply exclude mask (set excluded scores to -inf) ───────────
    if exclude_pages:
        excl = torch.as_tensor(
            sorted(set(int(p) for p in exclude_pages if 0 <= int(p) < P)),
            dtype=torch.long, device=scores.device)
        if excl.numel() > 0:
            scores[excl] = float("-inf")

    # ── 4. Top-K per KV head → union ──────────────────────────────────
    k_eff = min(k, P)
    if k_eff <= 0:
        return []
    # scores: [P, H_kv] → topk along dim=0
    topk_pid = torch.topk(scores, k_eff, dim=0).indices  # [k_eff, H_kv]
    # Filter out -inf entries (only matter when |non-excluded| < k_eff).
    topk_scores = torch.gather(scores, 0, topk_pid)
    valid_mask = torch.isfinite(topk_scores)
    union = set(int(pid.item())
                for pid, ok in zip(topk_pid.flatten(), valid_mask.flatten())
                if bool(ok))
    return sorted(union)


def quest_score_local_chunked(
    q: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    k: int,
    *,
    num_kv_heads: int | None = None,
    exclude_pages: Iterable[int] | None = None,
    page_chunk: int = 4096,
) -> list[int]:
    """Memory-aware variant: chunks over pages so peak [P, H_kv, D] never
    exceeds page_chunk * H_kv * D float32 elements. Useful at very large
    P (e.g., 128k tokens / 16 = 8k pages per layer). Otherwise identical.
    """
    if k <= 0 or kmin.shape[0] == 0:
        return []
    P = kmin.shape[0]
    if P <= page_chunk:
        return quest_score_local(
            q, kmin, kmax, k,
            num_kv_heads=num_kv_heads, exclude_pages=exclude_pages)

    # Score in chunks, accumulate per-head top-K candidates across chunks.
    # Final union is top-K per head across all chunks.
    if num_kv_heads is None:
        num_kv_heads = kmin.shape[1]
    excl_set = set(int(p) for p in (exclude_pages or [])
                   if 0 <= int(p) < P)
    # Score each chunk independently, keep top-K candidates per head as
    # (score, pid) pairs, then merge at the end.
    cand_scores = []   # list of [k_eff, H_kv]
    cand_pids   = []   # list of [k_eff, H_kv]
    k_eff = min(k, P)
    # Normalize q once for all chunks. (Mirror the validation from the
    # non-chunked path: H_q must be a positive multiple of num_kv_heads.)
    if q.ndim == 3:
        q2 = q.mean(dim=0)
    elif q.ndim == 2:
        q2 = q
    else:
        raise ValueError(f"q must be 2D [H_q,D] or 3D [T,H_q,D]; got {q.shape}")
    h_q, head_dim = q2.shape
    if h_q != num_kv_heads:
        if h_q < num_kv_heads or h_q % num_kv_heads != 0:
            raise ValueError(
                f"q has {h_q} heads, not divisible by num_kv_heads={num_kv_heads}")
        group = h_q // num_kv_heads
        q2 = q2.reshape(num_kv_heads, group, head_dim).mean(dim=1)
    q2 = q2.to(device=kmin.device, dtype=torch.float32).unsqueeze(0)  # [1, H_kv, D]
    for start in range(0, P, page_chunk):
        end = min(P, start + page_chunk)
        kmin_c = kmin[start:end].to(dtype=torch.float32)
        kmax_c = kmax[start:end].to(dtype=torch.float32)
        sc = torch.maximum(q2 * kmin_c, q2 * kmax_c).sum(dim=-1)  # [chunk,H_kv]
        # Apply exclude within this chunk.
        for pid in excl_set:
            if start <= pid < end:
                sc[pid - start] = float("-inf")
        kk = min(k_eff, end - start)
        top = torch.topk(sc, kk, dim=0)
        cand_scores.append(top.values)
        cand_pids.append(top.indices + start)
    all_scores = torch.cat(cand_scores, dim=0)        # [sum_kk, H_kv]
    all_pids   = torch.cat(cand_pids,   dim=0)
    final = torch.topk(all_scores, min(k_eff, all_scores.shape[0]), dim=0)
    sel_pids = torch.gather(all_pids, 0, final.indices)
    valid = torch.isfinite(final.values)
    return sorted(set(int(p.item())
                      for p, v in zip(sel_pids.flatten(), valid.flatten())
                      if bool(v)))


def quest_score_pages_max(
    q: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    pids: list[int],
    *,
    num_kv_heads: int | None = None,
) -> dict[int, float]:
    """Compute ``max_h score[pid, h]`` for each pid in ``pids``,
    using the same Quest scoring formula as ``quest_score_local`` but
    restricted to a precomputed pick list. Returns ``{pid: score}``.

    Used by the per-KV-head Quest connector path to attach real
    score values to a synthesized ``ScoreReply`` (instead of the
    degenerate all-zero placeholder). Picks themselves are NOT
    re-derived — this function does NOT change selection semantics.
    Cost: O(|pids| × H_kv × D), small relative to the upstream
    full-page scoring kernel (which already touches O(P × H_kv × D)).
    """
    if not pids or kmin.shape[0] == 0:
        return {}
    if num_kv_heads is None:
        num_kv_heads = kmin.shape[1]

    # Normalize q (mirrors quest_score_local).
    if q.ndim == 3:
        q2 = q.mean(dim=0)
    elif q.ndim == 2:
        q2 = q
    else:
        raise ValueError(
            f"q must be 2D [H_q,D] or 3D [T,H_q,D]; got {q.shape}")
    h_q, head_dim = q2.shape
    if h_q != num_kv_heads:
        if h_q < num_kv_heads or h_q % num_kv_heads != 0:
            raise ValueError(
                f"q has {h_q} heads, not divisible by num_kv_heads={num_kv_heads}")
        group = h_q // num_kv_heads
        q2 = q2.reshape(num_kv_heads, group, head_dim).mean(dim=1)
    q2 = q2.to(device=kmin.device, dtype=torch.float32).unsqueeze(0)

    pick_idx = torch.as_tensor(pids, dtype=torch.long, device=kmin.device)
    kmin_pick = kmin.index_select(0, pick_idx).to(dtype=torch.float32)
    kmax_pick = kmax.index_select(0, pick_idx).to(dtype=torch.float32)
    # [|pids|, H_kv, D] → sum_d max(q*kmin, q*kmax) → [|pids|, H_kv]
    scores = torch.maximum(q2 * kmin_pick, q2 * kmax_pick).sum(dim=-1)
    per_pid_max, _ = scores.max(dim=-1)  # [|pids|]
    return {int(p): float(s.item())
            for p, s in zip(pids, per_pid_max.flatten())}
