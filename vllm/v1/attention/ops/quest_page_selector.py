# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quest Page Selector — Core algorithm from Quest (ICML '24).

Quest organises the KV cache into pages of S tokens and maintains
per-page channel-wise min/max key statistics (GPU-resident).  Given a
query vector Q, it computes an upper-bound attention score per page:

    score(page_i) = sum_c max(Q_c * m_{i,c}, Q_c * M_{i,c})

where m and M are the channel-wise min and max key values for page i.
This is provably an upper bound on the true dot-product attention score
for any token in the page.  The top-K pages are selected and their full
K,V tensors are loaded from CPU to GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class QuestConfig:
    """Per-instance configuration for Quest page selection."""

    page_size: int = 16
    """Number of tokens per page (S).  Should match vLLM block_size."""

    budget: float = 0.2
    """Base fraction of pages to select per layer."""

    alpha: float = 5.0
    """Threshold scaling factor for dynamic budget computation."""

    dynamic_budget: bool = True
    """Adapt budget per-layer based on score distribution."""


class PageMetadata:
    """GPU-resident per-page min/max key statistics for one layer.

    For each page p in [0, num_pages) and each channel c in
    [0, num_kv_heads * head_dim), stores:
      - key_min[p, c] = min over tokens in page p of K[t, c]
      - key_max[p, c] = max over tokens in page p of K[t, c]

    Memory: 2 * num_pages * key_dim * element_size  (~2/S of key cache).
    """

    def __init__(
        self,
        num_pages: int,
        key_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
    ):
        self.num_pages = num_pages
        self.key_dim = key_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Initialise min to +inf and max to -inf so first update sets values
        self.key_min = torch.full(
            (num_pages, key_dim), float("inf"), dtype=dtype, device=self.device
        )
        self.key_max = torch.full(
            (num_pages, key_dim), float("-inf"), dtype=dtype, device=self.device
        )
        # Track which pages have been written
        self._initialised = torch.zeros(
            num_pages, dtype=torch.bool, device=self.device
        )

    def update_page(
        self,
        page_idx: int,
        key_vectors: torch.Tensor,
    ) -> None:
        """Update min/max for a single page from key vectors.

        Args:
            page_idx: Index of the page to update.
            key_vectors: Key tensor of shape ``[num_tokens, key_dim]``.
                The number of tokens may be less than page_size for
                partial pages.
        """
        if key_vectors.numel() == 0:
            return
        # Compute min/max in FP32 for precision, then store in self.dtype.
        keys = key_vectors.to(device=self.device).float()
        if keys.ndim == 1:
            keys = keys.unsqueeze(0)
        # Flatten to [num_tokens, key_dim] if needed
        if keys.shape[-1] != self.key_dim:
            keys = keys.reshape(-1, self.key_dim)

        page_min = keys.min(dim=0).values.to(self.dtype)
        page_max = keys.max(dim=0).values.to(self.dtype)

        if self._initialised[page_idx]:
            self.key_min[page_idx] = torch.min(
                self.key_min[page_idx], page_min
            )
            self.key_max[page_idx] = torch.max(
                self.key_max[page_idx], page_max
            )
        else:
            self.key_min[page_idx] = page_min
            self.key_max[page_idx] = page_max
            self._initialised[page_idx] = True

    def update_pages_batch(
        self,
        page_indices: torch.Tensor,
        key_vectors: torch.Tensor,
    ) -> None:
        """Batch update min/max for multiple pages.

        Args:
            page_indices: 1-D tensor of page indices, shape ``[P]``.
            key_vectors: Key tensor of shape ``[P, S, key_dim]`` where
                S is the page size (number of tokens per page).
        """
        if page_indices.numel() == 0:
            return
        # Compute min/max in FP32 for precision, then store in self.dtype.
        keys = key_vectors.to(device=self.device).float()
        if keys.ndim == 2:
            # [P, key_dim] — single token per page
            keys = keys.unsqueeze(1)

        # keys: [P, S, key_dim]
        if keys.shape[-1] != self.key_dim:
            keys = keys.reshape(keys.shape[0], -1, self.key_dim)

        batch_min = keys.min(dim=1).values.to(self.dtype)  # [P, key_dim]
        batch_max = keys.max(dim=1).values.to(self.dtype)  # [P, key_dim]

        idx = page_indices.to(self.device).long()
        already_init = self._initialised[idx]

        # For pages already initialised: element-wise min/max
        if already_init.any():
            mask = already_init.unsqueeze(1)  # [P, 1]
            existing_min = self.key_min[idx]
            existing_max = self.key_max[idx]
            new_min = torch.where(mask, torch.min(existing_min, batch_min), batch_min)
            new_max = torch.where(mask, torch.max(existing_max, batch_max), batch_max)
        else:
            new_min = batch_min
            new_max = batch_max

        self.key_min[idx] = new_min
        self.key_max[idx] = new_max
        self._initialised[idx] = True

    def reset(self) -> None:
        """Reset all metadata (e.g. after cache flush)."""
        self.key_min.fill_(float("inf"))
        self.key_max.fill_(float("-inf"))
        self._initialised.fill_(False)

    @property
    def num_initialised(self) -> int:
        return int(self._initialised.sum().item())


class QuestPageSelector:
    """Core Quest scoring algorithm.

    Maintains per-layer PageMetadata and provides page selection
    given a query vector.

    One ``QuestPageSelector`` instance is shared across layers.
    """

    def __init__(self, config: QuestConfig, num_layers: int):
        self.config = config
        self.num_layers = num_layers
        self.page_metadata: dict[int, PageMetadata] = {}

    def init_layer_metadata(
        self,
        layer_idx: int,
        num_pages: int,
        key_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
    ) -> PageMetadata:
        """Allocate GPU-resident page metadata for a layer."""
        meta = PageMetadata(
            num_pages=num_pages,
            key_dim=key_dim,
            dtype=dtype,
            device=device,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.page_metadata[layer_idx] = meta
        return meta

    def update_metadata_from_kv_store(
        self,
        layer_idx: int,
        block_ids: list[int],
        key_cache: torch.Tensor,
        block_size: int,
    ) -> None:
        """Compute and store page metadata when blocks are offloaded to CPU.

        Called during the GPU→CPU store path.  Extracts key vectors from
        the GPU cache before they leave GPU and updates per-page min/max.

        Args:
            layer_idx: Transformer layer index.
            block_ids: CPU-side block IDs being stored.
            key_cache: GPU key cache tensor.  Layout depends on the
                attention backend but typically
                ``[num_blocks, block_size, num_kv_heads, head_dim]``.
            block_size: Number of tokens per block (should equal page_size).
        """
        meta = self.page_metadata.get(layer_idx)
        if meta is None:
            return

        for i, block_id in enumerate(block_ids):
            if block_id >= key_cache.shape[0]:
                continue
            # Extract key vectors for this block: [block_size, num_kv_heads, head_dim]
            keys_block = key_cache[block_id]
            # Flatten to [block_size, num_kv_heads * head_dim]
            keys_flat = keys_block.reshape(block_size, -1)
            meta.update_page(block_id, keys_flat)

    # -- Scoring and Selection -----------------------------------------------

    @torch.no_grad()
    def compute_page_scores(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute upper-bound attention score per page.

        For each page i and channel c::

            U_{i,c} = max(Q_c * m_{i,c}, Q_c * M_{i,c})
            score_i = sum_c U_{i,c}

        This is a provable upper bound on the maximum dot-product
        attention score between Q and any token in page i.

        When the model uses GQA (num_heads > num_kv_heads), the query
        heads that share each KV-head are **mean-pooled** to produce a
        single representative query of dimension ``key_dim =
        num_kv_heads * head_dim``.  This matches the benchmark scorer
        from ``quest_selection_quality.py``.

        Args:
            query: Query vector(s).  Shape ``[B, key_dim]``,
                ``[B, num_kv_heads, head_dim]``, or
                ``[B, num_heads, head_dim]`` (GQA — will be mean-pooled
                across the heads-per-group that share each KV-head).
            layer_idx: The layer whose metadata to score against.

        Returns:
            Scores of shape ``[num_pages]`` (aggregated over batch).
        """
        meta = self.page_metadata.get(layer_idx)
        if meta is None:
            return torch.empty(0)

        device = meta.device

        # On CPU, use FP32 for scoring precision; on GPU use meta.dtype.
        scoring_dtype = torch.float32 if device.type == "cpu" else meta.dtype

        # Normalise query to [B, key_dim]
        q = query.to(device=device, dtype=scoring_dtype)
        if q.ndim == 3:
            B, H, D = q.shape
            num_kv_heads = meta.num_kv_heads
            head_dim = meta.head_dim

            if (num_kv_heads is not None
                    and head_dim is not None
                    and H > num_kv_heads):
                # GQA: mean-pool the heads_per_group Q-heads that share
                # each KV-head, then concatenate across KV-heads.
                # [B, H, D] -> [B, KVH, heads_per_group, D] -> mean
                #            -> [B, KVH, D] -> [B, key_dim]
                heads_per_group = H // num_kv_heads
                q = (
                    q.reshape(B, num_kv_heads, heads_per_group, D)
                    .mean(dim=2)
                    .reshape(B, num_kv_heads * D)
                )
            else:
                # No GQA or already key_dim-aligned
                q = q.reshape(B, -1)
        if q.ndim == 1:
            q = q.unsqueeze(0)  # [1, key_dim]

        # Trim q to match key_dim (safety fallback)
        if q.shape[-1] != meta.key_dim:
            min_dim = min(q.shape[-1], meta.key_dim)
            q = q[:, :min_dim]
            key_min = meta.key_min[:, :min_dim].to(scoring_dtype)
            key_max = meta.key_max[:, :min_dim].to(scoring_dtype)
        else:
            key_min = meta.key_min.to(scoring_dtype)
            key_max = meta.key_max.to(scoring_dtype)

        # Upper-bound per channel: max(Q_c * m_c, Q_c * M_c)
        # q: [B, D], key_min/key_max: [P, D]
        # Broadcast: [B, 1, D] * [1, P, D] -> [B, P, D]
        q_expanded = q.unsqueeze(1)  # [B, 1, D]
        min_expanded = key_min.unsqueeze(0)  # [1, P, D]
        max_expanded = key_max.unsqueeze(0)  # [1, P, D]

        score_with_min = q_expanded * min_expanded  # [B, P, D]
        score_with_max = q_expanded * max_expanded  # [B, P, D]

        upper_bound = torch.max(score_with_min, score_with_max)  # [B, P, D]
        page_scores = upper_bound.sum(dim=-1)  # [B, P]

        # Aggregate over batch (mean)
        if page_scores.shape[0] > 1:
            page_scores = page_scores.mean(dim=0)  # [P]
        else:
            page_scores = page_scores.squeeze(0)  # [P]

        # Mask out uninitialised pages
        uninit = ~meta._initialised
        if uninit.any():
            page_scores[uninit] = float("-inf")

        return page_scores

    @torch.no_grad()
    def select_pages(
        self,
        query: torch.Tensor,
        layer_idx: int,
        budget: float | None = None,
        stats: "PrefetchStats | None" = None,
    ) -> torch.Tensor:
        """Select top-K pages based on upper-bound scores.

        Args:
            query: Query vector(s), same as ``compute_page_scores``.
            layer_idx: Layer index.
            budget: Override for ``config.budget``.
            stats: Optional stats accumulator (reuses PrefetchStats;
                "rehearsal" maps to page scoring time).

        Returns:
            Selected page indices of shape ``[K]``, sorted ascending.
            Returns all initialised page indices if budget >= 1.0 or
            if K >= num_initialised_pages.
        """
        budget = budget if budget is not None else self.config.budget

        meta = self.page_metadata.get(layer_idx)
        if meta is None:
            return torch.empty(0, dtype=torch.long)

        num_init = meta.num_initialised
        if num_init == 0:
            if stats is not None:
                stats.end_rehearsal(layer_idx, 0, 0)
            return torch.empty(0, dtype=torch.long)

        k = max(1, int(num_init * budget))

        if stats is not None:
            stats.begin_rehearsal(layer_idx)

        if k >= num_init:
            indices = meta._initialised.nonzero(as_tuple=False).squeeze(-1)
            if stats is not None:
                stats.end_rehearsal(layer_idx, num_init, num_init)
            return indices.sort().values

        scores = self.compute_page_scores(query, layer_idx)

        _, top_indices = scores.topk(k)
        selected = top_indices.sort().values

        if stats is not None:
            stats.end_rehearsal(
                layer_idx,
                tokens_cached=num_init * self.config.page_size,
                tokens_selected=k * self.config.page_size,
            )

        return selected
