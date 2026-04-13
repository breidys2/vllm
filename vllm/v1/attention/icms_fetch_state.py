# SPDX-License-Identifier: Apache-2.0
"""Per-layer ICMS fetch state for Path B selective attention.

This module provides a thread-local state that the IcmsConnector sets
before each layer's attention forward and FlashAttention reads. It
contains the k-page fetch buffer (separate GPU tensors) that FlashAttention
uses INSTEAD of the main KV cache when active.

The connector sets the state in wait_for_layer_load and clears it in
save_kv_layer (which fires after attention within the kv_transfer_utils
wrapper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

# Module-level state. Set by the connector, read by FlashAttention.
# Since both run in the same thread (the model forward thread), this
# is safe without locks.

@dataclass
class IcmsFetchState:
    """Active fetch state for one layer's attention."""
    key_cache: torch.Tensor       # [k, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor     # same shape
    block_table: torch.Tensor     # [1, k] — identity mapping [0, 1, ..., k-1]
    seq_lens: torch.Tensor        # [1] — k * block_size
    max_seq_len: int
    scheduler_metadata: object = None  # override to None for fetch state


_active: Optional[IcmsFetchState] = None


def set_active(state: IcmsFetchState) -> None:
    global _active
    _active = state


def get_active() -> Optional[IcmsFetchState]:
    return _active


def clear() -> None:
    global _active
    _active = None
