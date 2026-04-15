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
    """Active fetch state for one layer's attention."""
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
