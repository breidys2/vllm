# SPDX-License-Identifier: Apache-2.0
"""
Quest layer-callback registry — torch.compile / CUDAGraph compatible
replacement for register_forward_hook.

Why this exists
---------------
ICMS (Quest) hooks need Python to run between decoder layers — to compute
the exact next-layer Q from the captured residual, fire a Score RPC at
stride boundaries, and push selected pages into the GPU KV cache before
the next layer's attention. Originally these were ``register_forward_hook``
callbacks on each decoder layer.

Forward hooks ARE stripped by ``torch.compile`` / Dynamo. So under
``cudagraph_mode=PIECEWISE`` the hooks silently stop firing — ICMS
becomes a no-op while looking healthy.

The replacement mechanism here:

1. Two custom torch ops, ``vllm::quest_fire_pre_layer`` and
   ``vllm::quest_fire_post_layer``, registered via vLLM's
   ``direct_register_custom_op`` helper. Their implementations look up a
   per-model :class:`QuestLayerCallbackRegistry` and dispatch.
2. These op names are added to vLLM's compilation ``splitting_ops`` list
   so PIECEWISE compilation breaks the captured graph at every call,
   leaving Python free to run the registered callbacks between
   per-layer captured pieces.
3. Each model class (Qwen3Moe, Llama, ...) is patched once to call
   the ops inside its forward loop, between consecutive ``layer(...)``
   calls.
4. Quest registers a single set of callbacks on the registry; the
   registry dispatches to all of them with the layer's residual stream.

Net effect: TP=2 + PIECEWISE captures NCCL + per-layer compute into
graphs (eliminating the eager-mode Python kernel-launch tax), while
ICMS callbacks still fire reliably between layers.

API
---
- :class:`QuestLayerCallbackRegistry`: holds the pre/post callback list.
- :func:`attach_registry`: stash a registry on a model with a stable id.
- :func:`detach_registry`: remove (idempotent).
- ``torch.ops.vllm.quest_fire_pre_layer(model_id, layer_idx, positions,
  hidden_states, residual)``: graph-break op the patched models call.
- ``torch.ops.vllm.quest_fire_post_layer(...)``: same for post-layer.
- :data:`SPLITTING_OPS`: the list to merge into ``compilation_config.splitting_ops``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from vllm.utils.torch_utils import direct_register_custom_op

logger = logging.getLogger(__name__)


# ─── Registry ───────────────────────────────────────────────────────────

# Callback signature:
#   (layer_idx, positions, hidden_states, residual, model_marker)
# `model_marker` is the integer key used when this callback was attached;
# callers can ignore it. ``residual`` may be ``None`` (e.g. before layer 0).
LayerCallback = Callable[
    [int, torch.Tensor, torch.Tensor, "torch.Tensor | None", int],
    None,
]


class QuestLayerCallbackRegistry:
    """Owns the set of pre/post layer callbacks for one model instance."""

    def __init__(self) -> None:
        self._pre: list[LayerCallback] = []
        self._post: list[LayerCallback] = []
        self._enabled: bool = True

    def register_pre(self, fn: LayerCallback) -> None:
        self._pre.append(fn)

    def register_post(self, fn: LayerCallback) -> None:
        self._post.append(fn)

    def clear(self) -> None:
        self._pre.clear()
        self._post.clear()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def has_callbacks(self) -> bool:
        return bool(self._pre) or bool(self._post)

    def fire_pre(self, layer_idx, positions, hidden_states, residual, key):
        if not self._enabled:
            return
        for fn in self._pre:
            try:
                fn(layer_idx, positions, hidden_states, residual, key)
            except Exception:
                logger.exception(
                    "quest pre-layer callback %s failed at layer=%d",
                    fn, layer_idx)

    def fire_post(self, layer_idx, positions, hidden_states, residual, key):
        if not self._enabled:
            return
        for fn in self._post:
            try:
                fn(layer_idx, positions, hidden_states, residual, key)
            except Exception:
                logger.exception(
                    "quest post-layer callback %s failed at layer=%d",
                    fn, layer_idx)


# Process-global "current registry" — only one model per worker process
# in practice, so a single global is simpler than a marker→registry map
# and avoids Dynamo constant-folding the marker at trace time. (We tried
# storing the marker as a model attribute and passing it as an int op
# arg; Dynamo evaluated getattr() at trace time and baked in 0.)
_REGISTRIES: dict[int, QuestLayerCallbackRegistry] = {}
_CURRENT_REGISTRY: QuestLayerCallbackRegistry | None = None

# Debug counters — tracks how many times the impls actually run.
# Bumped only when the op body executes (not during fake-tensor tracing).
_DEBUG_PRE_CALLS = [0]
_DEBUG_POST_CALLS = [0]


def _debug_log_first_call(which: str, marker: int, layer_idx: int) -> None:
    """Logs the first invocation of each op impl, then once every 50 calls.

    Helps verify whether the impl is actually being called when ICMS
    is configured but appears not to fire (e.g. under PIECEWISE).
    """
    counter = _DEBUG_PRE_CALLS if which == "pre" else _DEBUG_POST_CALLS
    counter[0] += 1
    n = counter[0]
    if n == 1 or n % 50 == 0:
        logger.debug(
            "[quest-debug] %s op invoked: call#=%d marker=%d layer=%d "
            "registries=%d", which, n, marker, layer_idx,
            len(_REGISTRIES))


def attach_registry(model, registry: QuestLayerCallbackRegistry) -> int:
    """Attach `registry` to `model` and set it as the process-global
    current registry. The ops look up `_CURRENT_REGISTRY` at runtime,
    bypassing Dynamo constant folding."""
    global _CURRENT_REGISTRY
    key = id(model)
    _REGISTRIES[key] = registry
    _CURRENT_REGISTRY = registry
    setattr(model, "quest_layer_callbacks", registry)
    setattr(model, "quest_layer_callbacks_marker", key)
    return key


def detach_registry(model) -> None:
    global _CURRENT_REGISTRY
    key = getattr(model, "quest_layer_callbacks_marker", None)
    if key is not None:
        _REGISTRIES.pop(key, None)
    if _CURRENT_REGISTRY is not None and id(_CURRENT_REGISTRY) and \
            getattr(model, "quest_layer_callbacks", None) is _CURRENT_REGISTRY:
        _CURRENT_REGISTRY = None
    if hasattr(model, "quest_layer_callbacks"):
        delattr(model, "quest_layer_callbacks")
    if hasattr(model, "quest_layer_callbacks_marker"):
        delattr(model, "quest_layer_callbacks_marker")


# ─── Custom torch ops ───────────────────────────────────────────────────
# These are the entry points the patched models call. They're defined
# under the vllm:: namespace so they can be added to
# compilation_config.splitting_ops, causing PIECEWISE compilation to
# break the captured graph at every call site. The op bodies run Python
# (registry dispatch) — torch.compile never traces inside them.

# Schemas use Tensor (not Tensor?) for residual; callers must pass a
# scalar zero placeholder when there is no residual yet. (None / Optional
# Tensor schemas are awkward to express via direct_register_custom_op's
# infer_schema helper.)


def _quest_fire_pre_layer_impl(
    marker: int,
    layer_idx: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    has_residual: bool,
) -> None:
    # Side-effecting op (declared mutates_args=["hidden_states"]). We do
    # not actually mutate the tensor; the declaration just keeps Inductor
    # from DCE'ing the op. `marker` is unused now (kept for schema
    # stability); the registry is looked up via the process-global
    # _CURRENT_REGISTRY because Dynamo constant-folded the per-model
    # marker at trace time.
    _debug_log_first_call("pre", marker, layer_idx)
    reg = _CURRENT_REGISTRY
    if reg is not None:
        res = residual if has_residual else None
        reg.fire_pre(layer_idx, positions, hidden_states, res, marker)


def _quest_fire_pre_layer_fake(
    marker: int,
    layer_idx: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    has_residual: bool,
) -> None:
    return None


def _quest_fire_post_layer_impl(
    marker: int,
    layer_idx: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    has_residual: bool,
) -> None:
    _debug_log_first_call("post", marker, layer_idx)
    reg = _CURRENT_REGISTRY
    if reg is not None:
        res = residual if has_residual else None
        reg.fire_post(layer_idx, positions, hidden_states, res, marker)


def _quest_fire_post_layer_fake(
    marker: int,
    layer_idx: int,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    has_residual: bool,
) -> None:
    return None


_REGISTERED = False


def _ensure_ops_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    # Declare hidden_states as mutated even though we don't actually
    # change it. This forces Inductor to treat the op as side-effecting
    # and keeps it in the captured graph; otherwise both ops are DCE'd
    # since their outputs are functionally equivalent to identity.
    direct_register_custom_op(
        op_name="quest_fire_pre_layer",
        op_func=_quest_fire_pre_layer_impl,
        mutates_args=["hidden_states"],
        fake_impl=_quest_fire_pre_layer_fake,
    )
    direct_register_custom_op(
        op_name="quest_fire_post_layer",
        op_func=_quest_fire_post_layer_impl,
        mutates_args=["hidden_states"],
        fake_impl=_quest_fire_post_layer_fake,
    )
    _REGISTERED = True


# Eager registration at import time so the ops are available before any
# model.forward gets traced.
_ensure_ops_registered()


# Names to merge into compilation_config.splitting_ops. Including these
# in the splitting_ops list tells vLLM's piecewise compilation backend
# to break the captured graph at each invocation, which is what lets our
# Python registry dispatch run between captured per-layer pieces.
SPLITTING_OPS: list[str] = [
    "vllm::quest_fire_pre_layer",
    "vllm::quest_fire_post_layer",
]


# ─── Convenience callers (used by patched model.forward) ────────────────


def fire_pre_layer(model: Any, layer_idx: int, positions: torch.Tensor,
                    hidden_states: torch.Tensor,
                    residual: torch.Tensor | None) -> None:
    """Always emits the torch op call so Dynamo can't bake in an
    early-return at trace time. Marker=0 is the runtime "no registry"
    sentinel — the op impl handles it."""
    marker = getattr(model, "quest_layer_callbacks_marker", 0) or 0
    if residual is None:
        placeholder = torch.empty(0, device=hidden_states.device,
                                   dtype=hidden_states.dtype)
        torch.ops.vllm.quest_fire_pre_layer(
            marker, layer_idx, positions, hidden_states, placeholder, False)
        return
    torch.ops.vllm.quest_fire_pre_layer(
        marker, layer_idx, positions, hidden_states, residual, True)


def fire_post_layer(model: Any, layer_idx: int, positions: torch.Tensor,
                     hidden_states: torch.Tensor,
                     residual: torch.Tensor | None) -> None:
    marker = getattr(model, "quest_layer_callbacks_marker", 0) or 0
    if residual is None:
        placeholder = torch.empty(0, device=hidden_states.device,
                                   dtype=hidden_states.dtype)
        torch.ops.vllm.quest_fire_post_layer(
            marker, layer_idx, positions, hidden_states, placeholder, False)
        return
    torch.ops.vllm.quest_fire_post_layer(
        marker, layer_idx, positions, hidden_states, residual, True)
