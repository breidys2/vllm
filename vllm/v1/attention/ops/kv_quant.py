# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extensible KV cache quantization framework for vLLM.

This module provides a registry-based framework for KV cache quantization
methods. Each method defines how to quantize incoming K/V tensors into a
compressed format and how to dequantize them back for attention computation.

The framework supports sub-byte quantization (4-bit, 2-bit) with per-group
asymmetric quantization (scale + zero-point) and inline fp16 metadata stored
alongside packed data.

Quantized cache layout per head per token:
    [ packed_data (head_size // pack_factor bytes) |
      scales (n_groups * 2 bytes, fp16) |
      zero_points (n_groups * 2 bytes, fp16) ]

Usage:
    from vllm.v1.attention.ops.kv_quant import (
        get_kv_quant_method,
        is_kv_quant_dtype,
    )

    quant_method = get_kv_quant_method("kivi_4bit", group_size=128)
    quant_method.quantize_and_store(key, value, kv_cache, slot_mapping, ...)
    key_bf16, value_bf16 = quant_method.dequantize(kv_cache, head_size=128)
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# ─── Registry ────────────────────────────────────────────────────────────────

_KV_QUANT_REGISTRY: dict[str, type["KVQuantMethod"]] = {}


def register_kv_quant(prefix: str):
    """Decorator to register a KV cache quantization method.

    Args:
        prefix: The dtype string prefix that triggers this method.
                e.g. "kivi_" matches "kivi_4bit", "kivi_2bit".
    """

    def decorator(cls: type["KVQuantMethod"]) -> type["KVQuantMethod"]:
        _KV_QUANT_REGISTRY[prefix] = cls
        return cls

    return decorator


def get_kv_quant_method(
    kv_cache_dtype: str,
    group_size: int = 0,
) -> "KVQuantMethod | None":
    """Get the quantization method for a given cache dtype string.

    Args:
        kv_cache_dtype: The cache dtype string (e.g. "kivi_4bit").
        group_size: Quantization group size. 0 means use head_size.

    Returns:
        A KVQuantMethod instance, or None if not a quantized dtype.
    """
    for prefix, cls in _KV_QUANT_REGISTRY.items():
        if kv_cache_dtype.startswith(prefix):
            return cls(kv_cache_dtype=kv_cache_dtype, group_size=group_size)
    return None


def is_kv_quant_dtype(kv_cache_dtype: str) -> bool:
    """Check if the cache dtype uses sub-byte quantization."""
    return any(kv_cache_dtype.startswith(p) for p in _KV_QUANT_REGISTRY)


# ─── Base class ──────────────────────────────────────────────────────────────


@dataclass
class KVQuantMethod(ABC):
    """Base class for KV cache quantization methods.

    Subclasses must implement:
        - bits: number of bits per quantized element
        - quantize_and_store: quantize K/V and write to paged cache
        - dequantize: dequantize cache to bf16 for attention computation

    The base class provides:
        - pack_factor: elements per byte
        - storage_dtype: torch.uint8
        - quant_head_bytes: bytes per head per token (data + scales)
        - page_size_bytes: total bytes per block (K + V)
        - resolve_group_size: resolve 0 → head_size
    """

    kv_cache_dtype: str
    group_size: int

    @property
    @abstractmethod
    def bits(self) -> int:
        """Number of bits per quantized element."""
        ...

    @property
    def pack_factor(self) -> int:
        """Number of quantized elements packed per byte."""
        return 8 // self.bits

    @property
    def storage_dtype(self) -> torch.dtype:
        """The torch dtype used for cache tensor storage."""
        return torch.uint8

    def resolve_group_size(self, head_size: int) -> int:
        """Resolve effective group size. 0 or negative → head_size."""
        if self.group_size <= 0:
            return head_size
        return min(self.group_size, head_size)

    def quant_head_bytes(self, head_size: int) -> int:
        """Bytes per head per token in quantized form.

        Layout: [packed_data | fp16_scales | fp16_zero_points]
        - packed_data: ceil(head_size / pack_factor) bytes
        - fp16_scales: n_groups * 2 bytes
        - fp16_zero_points: n_groups * 2 bytes
        """
        gs = self.resolve_group_size(head_size)
        n_groups = math.ceil(head_size / gs)
        data_bytes = math.ceil(head_size / self.pack_factor)
        scale_bytes = n_groups * 2  # fp16 per group
        zp_bytes = n_groups * 2     # fp16 per group
        return data_bytes + scale_bytes + zp_bytes

    def page_size_bytes(
        self,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> int:
        """Total page size in bytes for one block (K + V)."""
        qhb = self.quant_head_bytes(head_size)
        return 2 * block_size * num_kv_heads * qhb

    @abstractmethod
    def quantize_and_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        key_channel_scales: torch.Tensor | None = None,
        key_channel_mins: torch.Tensor | None = None,
        key_channel_maxs: torch.Tensor | None = None,
    ) -> None:
        """Quantize K/V and store into the paged cache.

        Args:
            key: [num_tokens, num_kv_heads, head_size] in model dtype.
            value: [num_tokens, num_kv_heads, head_size] in model dtype.
            kv_cache: [2, num_blocks, block_size, num_kv_heads,
                       quant_head_bytes] in uint8.
            slot_mapping: [num_tokens] int32, -1 for padding.
            k_scale: Per-head key scale (unused for KIVI, kept for API compat).
            v_scale: Per-head value scale (unused for KIVI, kept for API compat).
            key_channel_scales: [num_kv_heads, head_size] fp16, optional.
                Derived per-channel scales for keys, updated in-place.
            key_channel_mins: [num_kv_heads, head_size] fp16, optional.
                Running per-channel minimums for keys, updated in-place.
            key_channel_maxs: [num_kv_heads, head_size] fp16, optional.
                Running per-channel maximums for keys, updated in-place.
        """
        ...

    @abstractmethod
    def dequantize(
        self,
        kv_cache: torch.Tensor,
        head_size: int,
        output_dtype: torch.dtype = torch.bfloat16,
        key_channel_scales: torch.Tensor | None = None,
        key_channel_zero_points: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize cache to floating-point for attention computation.

        Args:
            kv_cache: [2, num_blocks, block_size, num_kv_heads,
                       quant_head_bytes] in uint8.
            head_size: Original head dimension size.
            output_dtype: Target dtype for dequantized tensors.
            key_channel_scales: [num_kv_heads, head_size] fp16, optional.
                Per-channel scales for key dequantization.
            key_channel_zero_points: [num_kv_heads, head_size] fp16, optional.
                Per-channel zero-points for key dequantization.

        Returns:
            (key_cache, value_cache) each
            [num_blocks, block_size, num_kv_heads, head_size] in output_dtype.
        """
        ...


# ─── Quantization helpers ────────────────────────────────────────────────────


def _symmetric_quantize(
    tensor: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-group quantization.

    Args:
        tensor: [..., head_size] in float/bf16.
        bits: Quantization bit width (2 or 4).
        group_size: Number of elements per quantization group.

    Returns:
        (qvals, scales):
            qvals: [..., head_size] in int8 (only lower `bits` significant).
            scales: [..., n_groups] in float16.
    """
    head_size = tensor.shape[-1]
    n_groups = math.ceil(head_size / group_size)

    # Pad head_size to multiple of group_size for clean reshape
    pad_size = n_groups * group_size - head_size
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))

    shape_prefix = tensor.shape[:-1]
    grouped = tensor.reshape(*shape_prefix, n_groups, group_size)

    # Per-group scales: symmetric quantization
    max_int = (1 << (bits - 1)) - 1  # 7 for 4-bit, 1 for 2-bit
    amax = grouped.abs().amax(dim=-1, keepdim=True)  # [..., n_groups, 1]
    scales = amax / max_int
    scales = scales.clamp(min=1e-10)

    # Quantize
    qvals = (grouped / scales).round().clamp(-(max_int + 1), max_int)
    qvals = qvals.to(torch.int8)

    # Remove padding and flatten
    qvals = qvals.reshape(*shape_prefix, n_groups * group_size)
    if pad_size > 0:
        qvals = qvals[..., :head_size]
    scales = scales.squeeze(-1).to(torch.float16)  # [..., n_groups]

    return qvals, scales


def _asymmetric_quantize(
    tensor: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-group quantization (unsigned).

    Args:
        tensor: [..., head_size] in float/bf16.
        bits: Quantization bit width (2 or 4).
        group_size: Number of elements per quantization group.

    Returns:
        (qvals, scales, zero_points):
            qvals: [..., head_size] in uint8 (only lower `bits` significant).
            scales: [..., n_groups] in float16.
            zero_points: [..., n_groups] in float16.
    """
    head_size = tensor.shape[-1]
    n_groups = math.ceil(head_size / group_size)

    # Pad head_size to multiple of group_size for clean reshape
    pad_size = n_groups * group_size - head_size
    if pad_size > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_size))

    shape_prefix = tensor.shape[:-1]
    grouped = tensor.reshape(*shape_prefix, n_groups, group_size)

    # Per-group asymmetric quantization
    max_uint = (1 << bits) - 1  # 15 for 4-bit, 3 for 2-bit
    vmin = grouped.amin(dim=-1, keepdim=True)  # [..., n_groups, 1]
    vmax = grouped.amax(dim=-1, keepdim=True)  # [..., n_groups, 1]

    scale = (vmax - vmin) / max_uint
    scale = scale.clamp(min=1e-10)

    zero_point = (-vmin / scale).round().clamp(0, max_uint)

    # Quantize
    qvals = (grouped / scale + zero_point).round().clamp(0, max_uint)
    qvals = qvals.to(torch.uint8)

    # Remove padding and flatten
    qvals = qvals.reshape(*shape_prefix, n_groups * group_size)
    if pad_size > 0:
        qvals = qvals[..., :head_size]
    scale = scale.squeeze(-1).to(torch.float16)           # [..., n_groups]
    zero_point = zero_point.squeeze(-1).to(torch.float16)  # [..., n_groups]

    return qvals, scale, zero_point


def compute_channel_scales(
    tensor: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """Compute per-channel scales for a batch of tokens.

    Per-channel means one scale per (head, head_size-element) pair,
    computed across the token dimension.

    Args:
        tensor: [N, num_kv_heads, head_size] in float/bf16.
        bits: Quantization bit width (2 or 4).

    Returns:
        scales: [num_kv_heads, head_size] in float16.
    """
    max_int = (1 << (bits - 1)) - 1
    amax = tensor.abs().amax(dim=0)  # [num_kv_heads, head_size]
    scales = amax / max_int
    scales = scales.clamp(min=1e-10).to(torch.float16)
    return scales


def _per_channel_quantize(
    tensor: torch.Tensor,
    bits: int,
    channel_scales: torch.Tensor,
) -> torch.Tensor:
    """Per-channel symmetric quantization using pre-computed scales.

    Unlike _symmetric_quantize which computes per-group scales along head_size,
    this uses externally-provided per-channel scales (one per head_size element,
    shared across all tokens).

    Args:
        tensor: [N, num_kv_heads, head_size] in float/bf16.
        bits: Quantization bit width (2 or 4).
        channel_scales: [num_kv_heads, head_size] in float16.

    Returns:
        qvals: [N, num_kv_heads, head_size] in int8.
    """
    max_int = (1 << (bits - 1)) - 1
    scales = channel_scales.unsqueeze(0)  # [1, num_kv_heads, head_size]
    qvals = (tensor / scales).round().clamp(-(max_int + 1), max_int)
    qvals = qvals.to(torch.int8)
    return qvals


def compute_channel_min_max(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel min and max for a batch of tokens.

    Per-channel means one min/max per (head, head_size-element) pair,
    computed across the token dimension.

    Args:
        tensor: [N, num_kv_heads, head_size] in float/bf16.

    Returns:
        (channel_mins, channel_maxs): each [num_kv_heads, head_size] in float16.
    """
    channel_mins = tensor.amin(dim=0).to(torch.float16)
    channel_maxs = tensor.amax(dim=0).to(torch.float16)
    return channel_mins, channel_maxs


def _per_channel_quantize_asymmetric(
    tensor: torch.Tensor,
    bits: int,
    channel_mins: torch.Tensor,
    channel_maxs: torch.Tensor,
) -> torch.Tensor:
    """Per-channel asymmetric quantization using pre-computed min/max.

    Args:
        tensor: [N, num_kv_heads, head_size] in float/bf16.
        bits: Quantization bit width (2 or 4).
        channel_mins: [num_kv_heads, head_size] in float16 (running min).
        channel_maxs: [num_kv_heads, head_size] in float16 (running max).

    Returns:
        qvals: [N, num_kv_heads, head_size] in uint8.
    """
    max_uint = (1 << bits) - 1
    mins = channel_mins.unsqueeze(0)  # [1, num_kv_heads, head_size]
    maxs = channel_maxs.unsqueeze(0)

    scale = (maxs - mins) / max_uint
    scale = scale.clamp(min=1e-10)
    zero_point = (-mins / scale).round().clamp(0, max_uint)

    qvals = (tensor / scale + zero_point).round().clamp(0, max_uint)
    qvals = qvals.to(torch.uint8)
    return qvals


def _pack_int4(qvals: torch.Tensor) -> torch.Tensor:
    """Pack pairs of signed 4-bit values into uint8 bytes.

    Layout: low nibble first (even indices), high nibble second (odd indices).

    Args:
        qvals: [..., N] in int8 where N is even. Values in [-8, 7].

    Returns:
        [..., N//2] in uint8.
    """
    even = qvals[..., 0::2] & 0x0F
    odd = (qvals[..., 1::2] & 0x0F) << 4
    return (even | odd).to(torch.uint8)


def _unpack_int4(packed: torch.Tensor, head_size: int) -> torch.Tensor:
    """Unpack uint8 bytes into pairs of signed 4-bit values.

    Args:
        packed: [..., N//2] in uint8.
        head_size: Original head size (for trimming).

    Returns:
        [..., head_size] in int8, sign-extended.
    """
    lo = (packed & 0x0F).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    hi = torch.where(hi > 7, hi - 16, hi)
    # Interleave: [lo0, hi0, lo1, hi1, ...]
    interleaved = torch.stack([lo, hi], dim=-1)
    result = interleaved.reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return result[..., :head_size]


def _pack_int2(qvals: torch.Tensor) -> torch.Tensor:
    """Pack groups of 4 signed 2-bit values into uint8 bytes.

    Args:
        qvals: [..., N] in int8 where N is divisible by 4. Values in [-2, 1].

    Returns:
        [..., N//4] in uint8.
    """
    v0 = qvals[..., 0::4] & 0x03
    v1 = (qvals[..., 1::4] & 0x03) << 2
    v2 = (qvals[..., 2::4] & 0x03) << 4
    v3 = (qvals[..., 3::4] & 0x03) << 6
    return (v0 | v1 | v2 | v3).to(torch.uint8)


def _unpack_int2(packed: torch.Tensor, head_size: int) -> torch.Tensor:
    """Unpack uint8 bytes into groups of 4 signed 2-bit values.

    Args:
        packed: [..., N//4] in uint8.
        head_size: Original head size (for trimming).

    Returns:
        [..., head_size] in int8, sign-extended.
    """
    v0 = (packed & 0x03).to(torch.int8)
    v0 = torch.where(v0 > 1, v0 - 4, v0)
    v1 = ((packed >> 2) & 0x03).to(torch.int8)
    v1 = torch.where(v1 > 1, v1 - 4, v1)
    v2 = ((packed >> 4) & 0x03).to(torch.int8)
    v2 = torch.where(v2 > 1, v2 - 4, v2)
    v3 = ((packed >> 6) & 0x03).to(torch.int8)
    v3 = torch.where(v3 > 1, v3 - 4, v3)
    interleaved = torch.stack([v0, v1, v2, v3], dim=-1)
    result = interleaved.reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    return result[..., :head_size]


def _unpack_uint4(packed: torch.Tensor, head_size: int) -> torch.Tensor:
    """Unpack uint8 bytes into pairs of unsigned 4-bit values.

    Args:
        packed: [..., N//2] in uint8.
        head_size: Original head size (for trimming).

    Returns:
        [..., head_size] in uint8, values in [0, 15].
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    interleaved = torch.stack([lo, hi], dim=-1)
    result = interleaved.reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return result[..., :head_size]


def _unpack_uint2(packed: torch.Tensor, head_size: int) -> torch.Tensor:
    """Unpack uint8 bytes into groups of 4 unsigned 2-bit values.

    Args:
        packed: [..., N//4] in uint8.
        head_size: Original head size (for trimming).

    Returns:
        [..., head_size] in uint8, values in [0, 3].
    """
    v0 = packed & 0x03
    v1 = (packed >> 2) & 0x03
    v2 = (packed >> 4) & 0x03
    v3 = (packed >> 6) & 0x03
    interleaved = torch.stack([v0, v1, v2, v3], dim=-1)
    result = interleaved.reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    return result[..., :head_size]


# ─── KIVI Implementation ────────────────────────────────────────────────────


@register_kv_quant("kivi_")
@dataclass
class KIVIQuantMethod(KVQuantMethod):
    """KIVI: Tuning-Free 2-Bit/4-Bit KV Cache Quantization.

    Reference: https://arxiv.org/abs/2402.02750

    Keys use per-channel asymmetric quantization (scale + zero-point per
    head_size element, computed across tokens via running min/max). Values
    use per-token asymmetric quantization (per-group scales and zero-points
    along head_size).

    Supported dtype strings:
        - "kivi_4bit": 4-bit quantization (~3.9x compression)
        - "kivi_2bit": 2-bit quantization (~7.5x compression)
    """

    @property
    def bits(self) -> int:
        if "2bit" in self.kv_cache_dtype:
            return 2
        return 4

    def quantize_and_store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        key_channel_scales: torch.Tensor | None = None,
        key_channel_mins: torch.Tensor | None = None,
        key_channel_maxs: torch.Tensor | None = None,
    ) -> None:
        head_size = key.shape[-1]
        block_size = kv_cache.shape[2]
        group_size = self.resolve_group_size(head_size)
        n_groups = math.ceil(head_size / group_size)
        data_bytes = math.ceil(head_size / self.pack_factor)

        # Filter out padding tokens
        valid_mask = slot_mapping >= 0
        valid_slots = slot_mapping[valid_mask]
        if valid_slots.numel() == 0:
            return

        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

        pack_fn = _pack_int4 if self.bits == 4 else _pack_int2
        pad_size = (
            self.pack_factor - head_size % self.pack_factor
        ) % self.pack_factor

        # ── Keys: per-channel asymmetric quantization ───────────
        valid_key = key[valid_mask]  # [N, num_kv_heads, head_size]
        N = valid_key.shape[0]
        num_kv_heads = valid_key.shape[1]

        if key_channel_mins is not None and key_channel_maxs is not None:
            # Update running per-channel min/max
            batch_mins, batch_maxs = compute_channel_min_max(valid_key)
            torch.minimum(
                key_channel_mins, batch_mins, out=key_channel_mins
            )
            torch.maximum(
                key_channel_maxs, batch_maxs, out=key_channel_maxs
            )
            # Quantize keys using per-channel asymmetric
            k_qvals = _per_channel_quantize_asymmetric(
                valid_key, self.bits, key_channel_mins, key_channel_maxs
            )
            # Derive and store channel scales/zero_points for dequantize
            max_uint = (1 << self.bits) - 1
            ch_scale = (key_channel_maxs - key_channel_mins) / max_uint
            ch_scale = ch_scale.clamp(min=1e-10)
            ch_zp = (-key_channel_mins / ch_scale).round().clamp(0, max_uint)
            key_channel_scales.copy_(ch_scale)
            k_inline_scales = None
            k_inline_zps = None
        else:
            # Fallback: per-token asymmetric quantization
            k_qvals, k_inline_scales, k_inline_zps = _asymmetric_quantize(
                valid_key, self.bits, group_size
            )

        # Pack key quantized values
        if pad_size > 0:
            k_qvals = torch.nn.functional.pad(k_qvals, (0, pad_size))
        k_packed = pack_fn(k_qvals)[..., :data_bytes]

        # K slot: packed data + scale area + zero_point area
        if k_inline_scales is None:
            # Per-channel: scales stored separately, zero-pad inline area
            k_meta_area = torch.zeros(
                N, num_kv_heads, n_groups * 4,
                dtype=torch.uint8, device=k_packed.device,
            )
        else:
            # Per-token fallback: store inline scales and zero_points
            k_scales_bytes = k_inline_scales.view(torch.uint8).reshape(
                N, num_kv_heads, n_groups * 2
            )
            k_zps_bytes = k_inline_zps.view(torch.uint8).reshape(
                N, num_kv_heads, n_groups * 2
            )
            k_meta_area = torch.cat([k_scales_bytes, k_zps_bytes], dim=-1)

        k_output = torch.cat([k_packed, k_meta_area], dim=-1)
        kv_cache[0, block_indices, block_offsets] = k_output

        # ── Values: per-token asymmetric quantization ───────────
        valid_value = value[valid_mask]
        v_qvals, v_scales, v_zps = _asymmetric_quantize(
            valid_value, self.bits, group_size
        )

        if pad_size > 0:
            v_qvals = torch.nn.functional.pad(v_qvals, (0, pad_size))
        v_packed = pack_fn(v_qvals)[..., :data_bytes]

        v_scales_bytes = v_scales.view(torch.uint8).reshape(
            N, num_kv_heads, n_groups * 2
        )
        v_zps_bytes = v_zps.view(torch.uint8).reshape(
            N, num_kv_heads, n_groups * 2
        )
        v_output = torch.cat(
            [v_packed, v_scales_bytes, v_zps_bytes], dim=-1
        )
        kv_cache[1, block_indices, block_offsets] = v_output

    def dequantize(
        self,
        kv_cache: torch.Tensor,
        head_size: int,
        output_dtype: torch.dtype = torch.bfloat16,
        key_channel_scales: torch.Tensor | None = None,
        key_channel_zero_points: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_size = self.resolve_group_size(head_size)
        n_groups = math.ceil(head_size / group_size)
        data_bytes = math.ceil(head_size / self.pack_factor)
        scale_end = data_bytes + n_groups * 2
        zp_end = scale_end + n_groups * 2

        # Unsigned unpack (no sign extension)
        unpack_fn = _unpack_uint4 if self.bits == 4 else _unpack_uint2

        # ── Dequantize keys ─────────────────────────────────────
        k_cache = kv_cache[0]
        k_packed = k_cache[..., :data_bytes]
        k_unpacked = unpack_fn(k_packed, head_size)

        if (key_channel_scales is not None
                and key_channel_zero_points is not None):
            # Per-channel: broadcast [num_kv_heads, head_size] over
            # [num_blocks, block_size] dimensions
            k_scales = key_channel_scales.unsqueeze(0).unsqueeze(0)
            k_zps = key_channel_zero_points.unsqueeze(0).unsqueeze(0)
            key_dequantized = (
                (k_unpacked.to(output_dtype) - k_zps.to(output_dtype))
                * k_scales.to(output_dtype)
            )
        else:
            # Fallback: per-token inline scales and zero_points
            k_scales_raw = k_cache[..., data_bytes:scale_end].contiguous()
            k_zps_raw = k_cache[..., scale_end:zp_end].contiguous()
            k_scales = k_scales_raw.view(torch.float16)
            k_zps = k_zps_raw.view(torch.float16)

            k_scales_expanded = k_scales.unsqueeze(-1).expand(
                *k_scales.shape, group_size
            ).reshape(
                *k_scales.shape[:-1], n_groups * group_size
            )[..., :head_size]
            k_zps_expanded = k_zps.unsqueeze(-1).expand(
                *k_zps.shape, group_size
            ).reshape(
                *k_zps.shape[:-1], n_groups * group_size
            )[..., :head_size]

            key_dequantized = (
                (k_unpacked.to(output_dtype) - k_zps_expanded.to(output_dtype))
                * k_scales_expanded.to(output_dtype)
            )

        # ── Dequantize values (per-token asymmetric) ────────────
        v_cache = kv_cache[1]
        v_packed = v_cache[..., :data_bytes]
        v_scales_raw = v_cache[..., data_bytes:scale_end].contiguous()
        v_zps_raw = v_cache[..., scale_end:zp_end].contiguous()
        v_unpacked = unpack_fn(v_packed, head_size)
        v_scales = v_scales_raw.view(torch.float16)
        v_zps = v_zps_raw.view(torch.float16)

        v_scales_expanded = v_scales.unsqueeze(-1).expand(
            *v_scales.shape, group_size
        ).reshape(
            *v_scales.shape[:-1], n_groups * group_size
        )[..., :head_size]
        v_zps_expanded = v_zps.unsqueeze(-1).expand(
            *v_zps.shape, group_size
        ).reshape(
            *v_zps.shape[:-1], n_groups * group_size
        )[..., :head_size]

        value_dequantized = (
            (v_unpacked.to(output_dtype) - v_zps_expanded.to(output_dtype))
            * v_scales_expanded.to(output_dtype)
        )

        return key_dequantized, value_dequantized
