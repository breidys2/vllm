# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
InfiniGen Dynamic Budget Computer — Contribution 5 of InfiniGen (OSDI '24).

Rather than using a fixed token selection budget for all layers, InfiniGen
dynamically varies the number of selected tokens per layer and per query
token.  Layers with focused (sparse) attention need fewer tokens; layers
with diffuse attention need more.

The budget computer operates on the approximate attention scores produced
by the rehearsal engine and determines how many tokens to select for each
layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class BudgetConfig:
    """Configuration for the dynamic budget computer."""

    base_budget: float = 0.2
    """Default fraction of tokens to select per layer."""

    alpha: float = 5.0
    """Threshold scaling factor.  Higher values → fewer tokens selected."""

    min_budget: float = 0.05
    """Minimum fraction of tokens to select (floor)."""

    max_budget: float = 1.0
    """Maximum fraction of tokens to select (cap)."""

    dynamic: bool = True
    """When True, adapt budget per layer; otherwise use base_budget."""


class DynamicBudgetComputer:
    """Computes per-layer token selection budgets.

    In *dynamic* mode, the budget is adjusted based on the entropy of the
    approximate attention score distribution:

    - **Low entropy** (concentrated scores): the layer's attention is focused
      on a few tokens → use a *lower* budget (fewer tokens suffice).
    - **High entropy** (uniform scores): attention is diffuse → use a
      *higher* budget (more tokens needed to cover the distribution).

    In *static* mode, ``base_budget`` is returned for every layer.
    """

    def __init__(self, config: BudgetConfig):
        self.config = config

    @torch.no_grad()
    def compute_budget(
        self,
        approximate_scores: torch.Tensor,
        layer_idx: int,
        num_layers: int,
    ) -> float:
        """Compute the token selection budget for a single layer.

        Args:
            approximate_scores: Approximate attention scores from the
                rehearsal engine, shape ``[num_query_tokens, num_kv_tokens]``.
            layer_idx: Index of the layer.
            num_layers: Total number of layers in the model.

        Returns:
            Fraction of tokens to select (in ``[min_budget, max_budget]``).
        """
        if not self.config.dynamic:
            return self.config.base_budget

        if approximate_scores.numel() == 0:
            return self.config.base_budget

        # Compute softmax to get attention probability distribution
        probs = torch.softmax(approximate_scores, dim=-1)  # [B, N]

        # Compute entropy per query token
        # H = -sum(p * log(p))  where p > 0
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]

        # Maximum possible entropy for this sequence length
        N = approximate_scores.shape[-1]
        max_entropy = torch.log(torch.tensor(float(N), device=entropy.device))

        if max_entropy <= 0:
            return self.config.base_budget

        # Normalised entropy in [0, 1]
        norm_entropy = (entropy.mean() / max_entropy).item()

        # Map entropy to budget:
        # norm_entropy ~ 0 (focused) → budget close to min_budget
        # norm_entropy ~ 1 (diffuse) → budget close to max_budget
        budget = (
            self.config.min_budget
            + (self.config.max_budget - self.config.min_budget) * norm_entropy
        )

        return max(self.config.min_budget, min(self.config.max_budget, budget))

    @torch.no_grad()
    def compute_per_query_token_counts(
        self,
        approximate_scores: torch.Tensor,
        layer_idx: int,
        num_layers: int,
    ) -> torch.Tensor:
        """Compute per-query-token selection counts.

        For each query token, determines how many KV tokens to select based
        on the concentration of its approximate attention distribution.

        Args:
            approximate_scores: Shape ``[num_query_tokens, num_kv_tokens]``.
            layer_idx: Index of the layer.
            num_layers: Total number of layers.

        Returns:
            Integer tensor of shape ``[num_query_tokens]`` with the number of
            KV tokens to select for each query token.
        """
        B, N = approximate_scores.shape

        if not self.config.dynamic:
            k = max(1, int(N * self.config.base_budget))
            return torch.full((B,), k, dtype=torch.long)

        # Compute per-query entropy
        probs = torch.softmax(approximate_scores, dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]

        max_entropy = torch.log(
            torch.tensor(float(N), device=entropy.device)
        )

        if max_entropy <= 0:
            k = max(1, int(N * self.config.base_budget))
            return torch.full((B,), k, dtype=torch.long)

        norm_entropy = entropy / max_entropy  # [B] in [0, 1]

        # Map to token counts
        min_k = max(1, int(N * self.config.min_budget))
        max_k = max(1, int(N * self.config.max_budget))

        counts = (min_k + (max_k - min_k) * norm_entropy).long()
        counts = counts.clamp(min=min_k, max=max_k)

        return counts

    def compute_static_layer_budgets(
        self,
        num_layers: int,
    ) -> list[float]:
        """Compute static per-layer budgets using a pyramid schedule.

        Lower layers (near input) get higher budgets (more diffuse attention);
        upper layers (near output) get lower budgets (more focused attention).
        This is inspired by PyramidKV's observation about layer-wise sparsity.

        Returns:
            List of floats of length ``num_layers``.
        """
        budgets = []
        for i in range(num_layers):
            # Linear interpolation: lower layers → max_budget,
            # upper layers → min_budget
            t = i / max(1, num_layers - 1)
            budget = (
                self.config.max_budget * (1 - t)
                + self.config.min_budget * t
            )
            budget = max(self.config.min_budget, min(self.config.max_budget, budget))
            budgets.append(budget)
        return budgets
