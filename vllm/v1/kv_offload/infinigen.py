# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
InfiniGen Offloading Spec and Manager.

InfiniGen (OSDI '24) stores the full KV cache in CPU memory and selectively
prefetches only essential entries to GPU per-layer during decode, using a
lightweight rehearsal mechanism to predict which tokens each layer will attend
to.

Unlike the standard CPUOffloadingSpec which loads entire blocks before the
forward pass, InfiniGen operates at token granularity, guided by per-layer
importance masks produced by the rehearsal engine.
"""

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.backends.cpu import CPUBackend
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# InfiniGen-specific LoadStoreSpec that carries per-layer token masks
# ---------------------------------------------------------------------------

@dataclass
class InfiniGenLoadSpec(LoadStoreSpec):
    """Extends block-level load spec with per-layer token selection masks.

    Attributes:
        block_ids: The CPU-side block IDs to load from (same as
            CPULoadStoreSpec).
        token_masks: Optional dict mapping ``layer_idx`` to a 1-D boolean
            ``np.ndarray`` of length ``num_tokens_in_blocks``.  When ``None``,
            all tokens in the blocks are loaded (equivalent to full-block
            transfer).
    """
    block_ids: np.ndarray
    token_masks: dict[int, np.ndarray] | None = None

    def __init__(
        self,
        block_ids: list[int],
        token_masks: dict[int, np.ndarray] | None = None,
    ):
        self.block_ids = np.array(block_ids, dtype=np.int64)
        self.token_masks = token_masks

    @staticmethod
    def medium() -> str:
        return "CPU"


# ---------------------------------------------------------------------------
# InfiniGen Offloading Manager (scheduler-side)
# ---------------------------------------------------------------------------

class InfiniGenOffloadingManager(OffloadingManager):
    """Scheduler-side manager for InfiniGen-style token-selective offloading.

    The manager tracks which blocks are offloaded in CPU memory using an LRU
    policy (identical to ``LRUOffloadingManager``), and additionally maintains
    per-layer token importance masks that are updated each decode step by the
    rehearsal engine.

    The key extension over ``LRUOffloadingManager`` is
    :meth:`set_layer_token_mask` which allows the rehearsal engine to specify
    which tokens within the offloaded blocks should be prefetched for each
    layer.
    """

    def __init__(
        self,
        backend: Backend,
        num_layers: int,
        enable_events: bool = False,
    ):
        self.backend = backend
        self.num_layers = num_layers
        # block_hash -> BlockStatus
        self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
        self.events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )

        # Per-layer token masks set by the rehearsal engine.
        # layer_idx -> (block_hashes, bool mask) — updated each decode step.
        self._layer_token_masks: dict[int, tuple[list[BlockHash], np.ndarray]] = {}

    # -- Rehearsal interface -------------------------------------------------

    def set_layer_token_mask(
        self,
        layer_idx: int,
        block_hashes: list[BlockHash],
        mask: np.ndarray,
    ) -> None:
        """Store the token importance mask for a specific layer.

        Called by the rehearsal engine after computing approximate attention
        scores.  The mask is consumed by :meth:`prepare_load_layer`.
        """
        self._layer_token_masks[layer_idx] = (block_hashes, mask)

    def get_layer_token_mask(
        self,
        layer_idx: int,
    ) -> tuple[list[BlockHash], np.ndarray] | None:
        """Retrieve the stored token mask for a layer (if any)."""
        return self._layer_token_masks.get(layer_idx)

    def clear_layer_token_masks(self) -> None:
        """Clear all per-layer masks (called at the end of each step)."""
        self._layer_token_masks.clear()

    # -- Standard OffloadingManager interface --------------------------------

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        hit_count = 0
        for block_hash in block_hashes:
            block = self.blocks.get(block_hash)
            if block is None or not block.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(
        self, block_hashes: Iterable[BlockHash]
    ) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        return self.backend.get_load_store_spec(block_hashes, blocks)

    def prepare_load_layer(
        self,
        block_hashes: Iterable[BlockHash],
        layer_idx: int,
    ) -> InfiniGenLoadSpec:
        """Prepare a layer-specific load with token mask.

        Falls back to full-block load if no mask is set for the layer.
        """
        block_hashes_list = list(block_hashes)
        blocks = []
        for block_hash in block_hashes_list:
            block = self.blocks[block_hash]
            assert block.is_ready
            block.ref_cnt += 1
            blocks.append(block)

        block_ids = [block.block_id for block in blocks]  # type: ignore[attr-defined]
        token_mask_entry = self._layer_token_masks.get(layer_idx)
        token_masks = (
            {layer_idx: token_mask_entry[1]}
            if token_mask_entry is not None
            else None
        )
        return InfiniGenLoadSpec(block_ids, token_masks=token_masks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        for block_hash in reversed(list(block_hashes)):
            if self.blocks.get(block_hash):
                self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            block = self.blocks[block_hash]
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        # Filter out blocks that are already stored
        block_hashes_to_store = [
            bh for bh in block_hashes if bh not in self.blocks
        ]

        num_blocks_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        to_evict: list[BlockHash] = []
        if num_blocks_to_evict > 0:
            for block_hash, block in self.blocks.items():
                if block.ref_cnt == 0:
                    to_evict.append(block_hash)
                    num_blocks_to_evict -= 1
                    if num_blocks_to_evict == 0:
                        break
            else:
                return None  # not enough evictable blocks

        # Perform eviction
        for block_hash in to_evict:
            self.backend.free(self.blocks.pop(block_hash))

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        blocks = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(blocks) == len(block_hashes_to_store)

        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.blocks[block_hash] = block

        store_spec = self.backend.get_load_store_spec(
            block_hashes_to_store, blocks
        )

        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ):
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                block = self.blocks[block_hash]
                if not block.is_ready:
                    self.backend.free(block)
                    del self.blocks[block_hash]

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()


# ---------------------------------------------------------------------------
# InfiniGen Offloading Spec (top-level entry point)
# ---------------------------------------------------------------------------

class InfiniGenOffloadingSpec(OffloadingSpec):
    """Offloading spec for InfiniGen-style token-selective KV cache loading.

    Configuration is provided via ``kv_connector_extra_config`` in the
    ``KVTransferConfig``.  Recognised keys (in addition to standard ones):

    - ``cpu_bytes_to_use`` (int, required): Total CPU memory budget in bytes.
    - ``infinigen_budget`` (float, default 0.2): Base token selection budget
      as a fraction of total cached tokens.
    - ``infinigen_alpha`` (float, default 5.0): Threshold scaling factor for
      dynamic budget computation.
    - ``infinigen_dynamic_budget`` (bool, default True): Whether to use
      per-layer dynamic budgets.
    - ``eviction_policy`` (str, default "lru"): CPU-side block eviction
      policy ("lru" supported).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig | None,
    ):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in "
                "kv_connector_extra_config for InfiniGen"
            )

        # Compute bytes per offloaded block (same logic as CPUOffloadingSpec)
        assert kv_cache_config is not None
        page_sizes = {
            group.kv_cache_spec.page_size_bytes
            for group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # InfiniGen-specific parameters
        self.infinigen_budget: float = float(
            self.extra_config.get("infinigen_budget", 0.2)
        )
        self.infinigen_alpha: float = float(
            self.extra_config.get("infinigen_alpha", 5.0)
        )
        self.infinigen_dynamic_budget: bool = bool(
            self.extra_config.get("infinigen_dynamic_budget", True)
        )

        # Determine number of layers from model config
        self.num_layers: int = (
            vllm_config.model_config.hf_config.num_hidden_layers
        )

        self.eviction_policy: str = self.extra_config.get(
            "eviction_policy", "lru"
        )

        # Lazily initialised
        self._manager: InfiniGenOffloadingManager | None = None
        self._handlers: CpuGpuOffloadingHandlers | None = None

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            backend = CPUBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_blocks,
            )

            if self.eviction_policy != "lru":
                raise ValueError(
                    f"InfiniGen currently only supports 'lru' eviction "
                    f"policy, got: {self.eviction_policy}"
                )

            self._manager = InfiniGenOffloadingManager(
                backend=backend,
                num_layers=self.num_layers,
                enable_events=enable_events,
            )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]:
        if self._handlers is None:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "InfiniGen offloading is currently only supported "
                    "on CUDA-alike GPUs"
                )

            self._handlers = CpuGpuOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                cpu_block_size=self.offloaded_block_size,
                num_cpu_blocks=self.num_blocks,
                gpu_caches=kv_caches,
            )

        assert self._handlers is not None
        yield (
            GPULoadStoreSpec,
            CPULoadStoreSpec,
            self._handlers.gpu_to_cpu_handler,
        )
        yield (
            CPULoadStoreSpec,
            GPULoadStoreSpec,
            self._handlers.cpu_to_gpu_handler,
        )
