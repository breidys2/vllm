# SPDX-License-Identifier: Apache-2.0
"""Per-layer ICMS fetch state for Path B selective attention.

Module-level state set by the connector before each layer's attention.
Contains the trimmed block_table for selective context attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class IcmsFetchState:
    """Active fetch state for one layer's attention.

    Shape contract:
      block_table: [N, max_k]  — N rids in the batch, padded to max trim length
      seq_lens:    [N]          — per-rid effective seq_len (post-trim)
      max_seq_len: int          — max(seq_lens)
    For the legacy single-rid path (ICMS_ALLOW_BATCH unset / N==1) shapes
    are [1, k+c] and [1]. Under ICMS_ALLOW_BATCH=1 with batch>=2, the
    aggregator in icms_connector.wait_for_layer produces a multi-row state
    in one set_active() call per layer.
    """
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int


_active: Optional[IcmsFetchState] = None


def set_active(state: IcmsFetchState) -> None:
    global _active
    _active = state


def get_active() -> Optional[IcmsFetchState]:
    return _active


def clear() -> None:
    global _active
    _active = None
