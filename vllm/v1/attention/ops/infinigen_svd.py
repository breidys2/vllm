# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
InfiniGen SVD Skewing — Contribution 4 of InfiniGen (OSDI '24).

This module handles **loading** of pre-computed SVD skewing matrices at
runtime.  The actual offline SVD computation is in
``scripts/infinigen_svd_setup.py``.

The SVD transformation concentrates important information into fewer
columns of the Q/K weight matrices, making the rehearsal engine's
partial-column speculation more accurate.  The transformation is
mathematically equivalent — attention outputs are identical.

SVD directory layout (produced by ``infinigen_svd_setup.py``):
    {svd_path}/
        metadata.pt                     — global metadata dict
        layer_{i}_partial_q.pt          — partial Q weight columns [out, partial_dim]
        layer_{i}_partial_k.pt          — partial K weight columns [out, partial_dim]
        layer_{i}_col_indices.pt        — selected column indices [partial_dim]
        layer_{i}_skewing_matrix.pt     — skewing matrix M [hidden, hidden]
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SVDMetadata:
    """Global metadata from the SVD setup process."""

    model_name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    partial_weight_ratio: float
    partial_dim: int


class InfiniGenSVDLoader:
    """Loads pre-computed SVD skewing data and attaches it to the model."""

    def __init__(self, svd_path: str):
        self.svd_path = svd_path
        self.metadata: SVDMetadata | None = None

        # Per-layer data (loaded lazily)
        self._partial_q: dict[int, torch.Tensor] = {}
        self._partial_k: dict[int, torch.Tensor] = {}
        self._col_indices: dict[int, torch.Tensor] = {}
        self._skewing_matrices: dict[int, torch.Tensor] = {}

    def load_metadata(self) -> SVDMetadata:
        """Load global SVD metadata."""
        meta_path = os.path.join(self.svd_path, "metadata.pt")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"InfiniGen SVD metadata not found at {meta_path}. "
                f"Run scripts/infinigen_svd_setup.py first."
            )

        meta_dict = torch.load(meta_path, map_location="cpu", weights_only=True)
        self.metadata = SVDMetadata(**meta_dict)
        logger.info(
            "Loaded InfiniGen SVD metadata: model=%s, layers=%d, "
            "partial_ratio=%.2f, partial_dim=%d",
            self.metadata.model_name,
            self.metadata.num_layers,
            self.metadata.partial_weight_ratio,
            self.metadata.partial_dim,
        )
        return self.metadata

    def load_layer(self, layer_idx: int) -> None:
        """Load SVD data for a specific layer."""
        prefix = os.path.join(self.svd_path, f"layer_{layer_idx}")

        q_path = f"{prefix}_partial_q.pt"
        k_path = f"{prefix}_partial_k.pt"
        idx_path = f"{prefix}_col_indices.pt"
        skew_path = f"{prefix}_skewing_matrix.pt"

        if os.path.exists(q_path):
            self._partial_q[layer_idx] = torch.load(
                q_path, map_location="cpu", weights_only=True
            )
        if os.path.exists(k_path):
            self._partial_k[layer_idx] = torch.load(
                k_path, map_location="cpu", weights_only=True
            )
        if os.path.exists(idx_path):
            self._col_indices[layer_idx] = torch.load(
                idx_path, map_location="cpu", weights_only=True
            )
        if os.path.exists(skew_path):
            self._skewing_matrices[layer_idx] = torch.load(
                skew_path, map_location="cpu", weights_only=True
            )

    def load_all_layers(self) -> None:
        """Load SVD data for all layers."""
        if self.metadata is None:
            self.load_metadata()
        assert self.metadata is not None

        for i in range(self.metadata.num_layers):
            self.load_layer(i)

        logger.info(
            "Loaded InfiniGen SVD data for %d layers",
            len(self._partial_q),
        )

    def get_partial_q_weights(self, layer_idx: int) -> torch.Tensor | None:
        """Get partial Q weight columns for a layer."""
        return self._partial_q.get(layer_idx)

    def get_partial_k_weights(self, layer_idx: int) -> torch.Tensor | None:
        """Get partial K weight columns for a layer."""
        return self._partial_k.get(layer_idx)

    def get_column_indices(self, layer_idx: int) -> torch.Tensor | None:
        """Get selected column indices for a layer."""
        return self._col_indices.get(layer_idx)

    def get_skewing_matrix(self, layer_idx: int) -> torch.Tensor | None:
        """Get the skewing matrix for a layer."""
        return self._skewing_matrices.get(layer_idx)


def load_skewing_metadata(
    rehearsal_engine: "RehearsalEngine",  # noqa: F821
    svd_path: str,
) -> SVDMetadata:
    """Convenience function: load SVD data and populate a RehearsalEngine.

    Args:
        rehearsal_engine: The rehearsal engine to populate with partial
            weights and column indices.
        svd_path: Path to the SVD directory.

    Returns:
        The loaded SVDMetadata.
    """
    loader = InfiniGenSVDLoader(svd_path)
    metadata = loader.load_metadata()
    loader.load_all_layers()

    # Populate the rehearsal engine
    for layer_idx in range(metadata.num_layers):
        q = loader.get_partial_q_weights(layer_idx)
        k = loader.get_partial_k_weights(layer_idx)
        idx = loader.get_column_indices(layer_idx)

        if q is not None:
            rehearsal_engine.partial_q_weights[layer_idx] = q
        if k is not None:
            rehearsal_engine.partial_k_weights[layer_idx] = k
        if idx is not None:
            rehearsal_engine.partial_column_indices[layer_idx] = idx

    return metadata
