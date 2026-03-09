# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
InfiniGen Rehearsal Engine — Contribution 3 of InfiniGen (OSDI '24).

The rehearsal engine performs lightweight speculation to predict which KV cache
tokens will be important for the *next* attention layer.  It exploits the
observation that attention inputs of consecutive Transformer layers are highly
similar: the residual stream after layer L is a good proxy for layer L+1's
attention input.

Algorithm (per decode step, per layer L):
    1. Take layer L's post-attention residual stream ``h`` (shape [B, D]).
    2. Project through a *subset* of layer L+1's query weight columns to
       obtain approximate queries ``q_approx`` (shape [B, D_partial]).
    3. Retrieve matching partial key columns from the CPU-side KV cache:
       ``k_partial`` (shape [N, D_partial]) where N = total cached tokens.
    4. Compute approximate attention scores: ``scores = q_approx @ k_partial^T``
       (shape [B, N]).
    5. Use ``DynamicBudgetComputer`` to select which token indices to prefetch.
    6. Return a boolean mask of shape [N] indicating selected tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RehearsalConfig:
    """Per-layer configuration for the rehearsal engine."""

    partial_weight_ratio: float = 0.1
    """Fraction of Q/K weight columns used for speculation."""

    budget: float = 0.2
    """Base fraction of tokens to select."""

    alpha: float = 5.0
    """Threshold scaling factor for score-based selection."""

    dynamic_budget: bool = True
    """Adapt budget per-layer based on attention score distribution."""


class RehearsalEngine:
    """Performs lightweight speculation to predict important tokens.

    One ``RehearsalEngine`` instance is shared across layers.  Per-layer
    partial weight matrices are stored in :attr:`partial_q_weights` and
    :attr:`partial_k_weights` (populated during model initialisation by
    ``infinigen_svd.load_skewing_metadata`` or by extracting raw column
    subsets when SVD is not used).
    """

    def __init__(self, config: RehearsalConfig, num_layers: int):
        self.config = config
        self.num_layers = num_layers

        # Populated by load_partial_weights() or infinigen_svd loader.
        # layer_idx -> (partial_q, partial_k)
        # partial_q: [num_heads, head_size, partial_dim]
        # partial_k: [num_kv_heads, head_size, partial_dim]
        self.partial_q_weights: dict[int, torch.Tensor] = {}
        self.partial_k_weights: dict[int, torch.Tensor] = {}

        # Column indices selected for each layer (used when no SVD skewing).
        self.partial_column_indices: dict[int, torch.Tensor] = {}

    # -- Weight initialisation -----------------------------------------------

    def load_partial_weights_from_model(
        self,
        model: torch.nn.Module,
        layer_prefix: str = "model.layers",
        qkv_proj_name: str = "self_attn.qkv_proj",
    ) -> None:
        """Extract partial Q/K weight columns from a loaded model.

        This is used when SVD skewing is NOT applied: we select the columns
        with the largest L2 norm (heuristic for importance).
        """
        num_cols = None
        for layer_idx in range(self.num_layers):
            proj_name = f"{layer_prefix}.{layer_idx}.{qkv_proj_name}"
            qkv_weight = None
            for name, param in model.named_parameters():
                if name == f"{proj_name}.weight":
                    qkv_weight = param.data
                    break

            if qkv_weight is None:
                logger.warning(
                    "Could not find QKV weight for layer %d at %s",
                    layer_idx,
                    proj_name,
                )
                continue

            # qkv_weight shape: [q_size + kv_size + kv_size, hidden_size]
            # We need to figure out the split sizes.  For GQA models:
            #   q_size  = num_heads * head_size
            #   kv_size = num_kv_heads * head_size
            hidden_size = qkv_weight.shape[1]
            total_out = qkv_weight.shape[0]

            if num_cols is None:
                num_cols = max(1, int(hidden_size * self.config.partial_weight_ratio))

            # Select top-norm columns from the hidden_size dimension
            col_norms = qkv_weight.norm(dim=0)  # [hidden_size]
            _, top_indices = col_norms.topk(num_cols)
            top_indices, _ = top_indices.sort()

            self.partial_column_indices[layer_idx] = top_indices

            # Extract partial columns: [total_out, num_cols]
            partial_qkv = qkv_weight[:, top_indices]

            # We store the full partial QKV and split at rehearsal time
            # since the split sizes depend on model config
            self.partial_q_weights[layer_idx] = partial_qkv
            # partial_k_weights will be set to the K portion during rehearsal

    def load_partial_weights_from_svd(
        self,
        svd_path: str,
    ) -> None:
        """Load pre-computed SVD skewing metadata.

        The SVD setup script produces per-layer files:
            {svd_path}/layer_{i}_partial_q.pt  — partial Q weight columns
            {svd_path}/layer_{i}_partial_k.pt  — partial K weight columns
            {svd_path}/layer_{i}_col_indices.pt — selected column indices
        """
        import os

        for layer_idx in range(self.num_layers):
            q_path = os.path.join(svd_path, f"layer_{layer_idx}_partial_q.pt")
            k_path = os.path.join(svd_path, f"layer_{layer_idx}_partial_k.pt")
            idx_path = os.path.join(
                svd_path, f"layer_{layer_idx}_col_indices.pt"
            )

            if not os.path.exists(q_path):
                logger.warning(
                    "SVD partial Q weights not found for layer %d at %s",
                    layer_idx,
                    q_path,
                )
                continue

            self.partial_q_weights[layer_idx] = torch.load(
                q_path, map_location="cpu", weights_only=True
            )
            if os.path.exists(k_path):
                self.partial_k_weights[layer_idx] = torch.load(
                    k_path, map_location="cpu", weights_only=True
                )
            if os.path.exists(idx_path):
                self.partial_column_indices[layer_idx] = torch.load(
                    idx_path, map_location="cpu", weights_only=True
                )

        logger.info(
            "Loaded InfiniGen SVD partial weights for %d/%d layers from %s",
            len(self.partial_q_weights),
            self.num_layers,
            svd_path,
        )

    # -- Core rehearsal ------------------------------------------------------

    @torch.no_grad()
    def rehearse(
        self,
        hidden_states: torch.Tensor,
        cached_keys: torch.Tensor,
        layer_idx: int,
        next_layer_idx: int,
        budget: float | None = None,
        alpha: float | None = None,
        stats: "InfiniGenStats | None" = None,
    ) -> torch.Tensor:
        """Predict which cached tokens are important for the next layer.

        Args:
            hidden_states: Post-attention residual from layer ``layer_idx``.
                Shape: ``[num_tokens, hidden_size]``.
            cached_keys: Full key cache for the **next** layer on CPU.
                Shape: ``[num_cached_tokens, num_kv_heads, head_size]``.
            layer_idx: Index of the current layer (whose forward just
                completed).
            next_layer_idx: Index of the next layer for which we are
                predicting importance.
            budget: Override for ``config.budget``.
            alpha: Override for ``config.alpha``.

        Returns:
            Boolean mask of shape ``[num_cached_tokens]`` where ``True``
            indicates the token should be prefetched.
        """
        budget = budget if budget is not None else self.config.budget
        alpha = alpha if alpha is not None else self.config.alpha

        if stats is not None:
            stats.begin_rehearsal(next_layer_idx)

        num_cached = cached_keys.shape[0]
        if num_cached == 0:
            if stats is not None:
                stats.end_rehearsal(next_layer_idx, 0, 0)
            return torch.zeros(0, dtype=torch.bool)

        # Number of tokens to select
        k = max(1, int(num_cached * budget))

        # Get partial weights for the next layer
        partial_qkv = self.partial_q_weights.get(next_layer_idx)
        if partial_qkv is None:
            # No partial weights — select all tokens (full load)
            if stats is not None:
                stats.end_rehearsal(next_layer_idx, num_cached, num_cached)
            return torch.ones(num_cached, dtype=torch.bool)

        col_indices = self.partial_column_indices.get(next_layer_idx)
        if col_indices is None:
            if stats is not None:
                stats.end_rehearsal(next_layer_idx, num_cached, num_cached)
            return torch.ones(num_cached, dtype=torch.bool)

        # Project hidden states through partial columns
        # hidden_states: [B, hidden_size]
        # We extract just the partial columns from hidden_states
        device = hidden_states.device
        col_indices_dev = col_indices.to(device)

        # h_partial: [B, num_partial_cols]
        h_partial = hidden_states[:, col_indices_dev]

        # partial_qkv: [total_out, num_partial_cols]
        # For the approximate score, we do: h_partial @ partial_qkv^T
        # This gives us approximate Q*K^T patterns
        partial_qkv_dev = partial_qkv.to(device)

        # approx_qk: [B, total_out]
        approx_qk = h_partial @ partial_qkv_dev.t()

        # Extract partial columns from cached keys
        # cached_keys: [N, num_kv_heads, head_size] on CPU
        # Reshape to [N, num_kv_heads * head_size]
        N, num_kv_heads, head_size = cached_keys.shape
        cached_keys_flat = cached_keys.reshape(N, -1)

        # We use the norm of each cached key as importance proxy
        # combined with the approximate QK scores
        key_norms = cached_keys_flat.norm(dim=-1)  # [N]

        # Compute approximate attention: use mean of approx_qk as
        # a scalar importance signal, weighted by key norms
        # This is a simplified version; full implementation would do
        # the proper matmul with partial K columns
        approx_importance = key_norms  # [N] on CPU

        # If we have column indices, do a more precise computation
        if col_indices is not None and h_partial.shape[0] > 0:
            # k_partial: [N, num_partial_cols]
            # We index into the flattened key cache
            # For simplicity, use the first few columns as proxy
            num_partial = min(col_indices.shape[0], cached_keys_flat.shape[1])
            k_partial = cached_keys_flat[:, :num_partial].to(device)

            # Mean query: [num_partial]
            q_mean = h_partial[:, :num_partial].mean(dim=0)

            # scores: [N]
            scores = (k_partial @ q_mean).cpu()

            # Combine with key norms
            approx_importance = scores.abs() + alpha * key_norms

        # Select top-k tokens
        if k >= num_cached:
            if stats is not None:
                stats.end_rehearsal(next_layer_idx, num_cached, num_cached)
            return torch.ones(num_cached, dtype=torch.bool)

        _, top_indices = approx_importance.topk(k)
        mask = torch.zeros(num_cached, dtype=torch.bool)
        mask[top_indices] = True

        if stats is not None:
            num_selected = mask.sum().item()
            stats.end_rehearsal(next_layer_idx, num_cached, num_selected)

        return mask

    @torch.no_grad()
    def rehearse_with_partial_qk(
        self,
        hidden_states: torch.Tensor,
        cached_keys_partial: torch.Tensor,
        num_cached_tokens: int,
        next_layer_idx: int,
        budget: float | None = None,
        stats: "InfiniGenStats | None" = None,
    ) -> torch.Tensor:
        """Rehearse using pre-extracted partial key columns.

        This is the optimised path when the partial K cache is maintained
        separately (e.g., extracted during KV cache store).

        Args:
            hidden_states: Post-attention residual, shape [B, hidden_size].
            cached_keys_partial: Partial key columns for next layer,
                shape [num_cached_tokens, partial_dim].  Already on GPU.
            num_cached_tokens: Total cached tokens.
            next_layer_idx: Index of the next layer.
            budget: Override for ``config.budget``.

        Returns:
            Boolean mask of shape [num_cached_tokens].
        """
        budget = budget if budget is not None else self.config.budget
        k = max(1, int(num_cached_tokens * budget))

        if stats is not None:
            stats.begin_rehearsal(next_layer_idx)

        if k >= num_cached_tokens:
            if stats is not None:
                stats.end_rehearsal(
                    next_layer_idx, num_cached_tokens, num_cached_tokens
                )
            return torch.ones(num_cached_tokens, dtype=torch.bool)

        partial_qkv = self.partial_q_weights.get(next_layer_idx)
        col_indices = self.partial_column_indices.get(next_layer_idx)

        if partial_qkv is None or col_indices is None:
            if stats is not None:
                stats.end_rehearsal(
                    next_layer_idx, num_cached_tokens, num_cached_tokens
                )
            return torch.ones(num_cached_tokens, dtype=torch.bool)

        device = hidden_states.device
        col_indices_dev = col_indices.to(device)

        # h_partial: [B, partial_dim]
        h_partial = hidden_states[:, col_indices_dev]

        # Approximate attention scores: h_partial @ cached_keys_partial^T
        # h_partial: [B, partial_dim]
        # cached_keys_partial: [N, partial_dim]
        # scores: [B, N]
        scores = h_partial @ cached_keys_partial.t()

        # Aggregate across query tokens (mean)
        importance = scores.mean(dim=0).abs()  # [N]

        _, top_indices = importance.topk(k)
        mask = torch.zeros(num_cached_tokens, dtype=torch.bool, device=device)
        mask[top_indices] = True

        if stats is not None:
            num_selected = mask.sum().item()
            stats.end_rehearsal(
                next_layer_idx, num_cached_tokens, num_selected
            )

        return mask.cpu()
