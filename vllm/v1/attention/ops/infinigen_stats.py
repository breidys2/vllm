# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility shim — re-exports from prefetch_stats under old names.

Existing code that does ``from ...infinigen_stats import InfiniGenStats``
continues to work.  New code should import from ``prefetch_stats`` directly.
"""

from vllm.v1.attention.ops.prefetch_stats import (  # noqa: F401
    PrefetchLayerStats as InfiniGenLayerStats,
    PrefetchStats as InfiniGenStats,
    PrefetchStepStats as InfiniGenStepStats,
)
