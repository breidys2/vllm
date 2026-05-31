# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quest Forward Hooks — Pipelined page-level prefetch with exact Q projection.

After layer L's forward pass completes, the hook:
  1. Extracts the post-attention residual stream h_L.
  2. Computes LayerNorm_{L+1}(h_L) using layer L+1's actual RMSNorm weights.
  3. Projects through layer L+1's actual Q weight matrix → exact Q_{L+1}.
  4. Forwards Q_{L+1} to the page selector / KV connector.
  5. Triggers an async CPU→GPU prefetch of the selected pages.

The hook computes the **exact** Q projection for the next layer. Cost
is one RMSNorm + one matmul per layer — negligible compared to attention
over thousands of tokens. Page selection itself happens externally
(e.g. BF2-side in ICMS); this module only handles the Q-projection
plumbing.

Usage:
    hook_manager = QuestHookManager(
        page_selector=selector,
        budget_computer=budget_computer,
        num_layers=32,
    )
    hook_manager.register_hooks(model)
    # ... run forward pass ...
    hook_manager.remove_hooks()
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


class QuestHookManager:
    """Manages forward hooks on decoder layers for Quest page prefetch.

    Attributes:
        page_selector: Optional in-process page selector (None when
            scoring is delegated externally, e.g. BF2-side in ICMS).
        budget_computer: Optional in-process budget computer.
        num_layers: Number of decoder layers.
        kv_connector: Reference to the active KV connector
            (e.g. IcmsConnector).
        stats: Optional stats collector for timing.
    """

    def __init__(
        self,
        page_selector: Any,
        budget_computer: Any,
        num_layers: int,
        kv_connector: Any | None = None,
        stats: Any | None = None,
        cpu_scoring: bool = True,
    ):
        self.page_selector = page_selector
        self.budget_computer = budget_computer
        self.num_layers = num_layers
        self.kv_connector = kv_connector
        self.stats = stats
        self.cpu_scoring = cpu_scoring
        self.single_layer_scoring = False  # set after __init__ if needed
        # Stride gating — fresh Q is only needed at stride-aligned layers
        # (next_layer_idx ∈ {1, 1+stride, 1+2*stride, ...}); non-stride
        # layers reuse the prior stride group's selection and never read
        # Q. Set by the callsite from the connector's icms_score_stride.
        # Default 1 = every layer scored (no stride gating).
        self.score_stride = 1
        # 2026-05-30 fix (workflow wrrwqsdto): explicit set of scored layer
        # indices for hybrid models whose dense layers do NOT align to the
        # absolute-index modulo (e.g. gemma-3 dense layers {5,11,17,...,59}
        # under score_stride=6). When non-None, the stride-modular gates at
        # ~line 404 and ~line 724 treat any layer IN this set as
        # stride-aligned regardless of the modulo check, restoring the
        # save_kv_layer(quest_query=...) path so the connector's adaptive
        # allocator can register_request. For uniform models the harness
        # leaves this None (legacy modulo-only behavior preserved).
        self.scored_layers_set: frozenset[int] | None = None
        # When True, budget >= 1.0 shortcuts to the FetchAll path (kFetchAll
        # on the ICMS server). Set to False for adaptive allocators whose
        # "1.0 on idle" output should still exercise the stride-gated Score
        # path — the callsite (model_runner) flips this per config.
        self.allow_all_pages_shortcut = True

        # Registered hook handles (for cleanup)
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []

        # Cached layer weights for exact Q computation
        # layer_idx -> (layernorm_weight, eps)
        self._layernorm_weights: dict[int, tuple[torch.Tensor, float]] = {}
        # layer_idx -> Q weight matrix [q_size, hidden_size]
        self._q_proj_weights: dict[int, torch.Tensor] = {}
        # layer_idx -> rotary embedding module (for post-rope Q)
        self._rotary_embs: dict[int, nn.Module] = {}
        # layer_idx -> (q_norm_weight [head_dim], eps) for models with QK-Norm
        # (qwen3, qwen3-moe, gemma3). Absent for llama/mistral. The real
        # forward applies this per-head RMSNorm over head_dim BETWEEN q_proj
        # and RoPE (qwen3_moe.py:341-348); omitting it dots a pre-q_norm Q
        # against post-k_norm K summaries → scrambled Quest page ranking.
        self._q_norm_weights: dict[int, tuple[torch.Tensor, float]] = {}

        # Per-step state: layer_idx -> selected page indices
        self._current_selections: dict[int, torch.Tensor] = {}

        self._active = False
        self._num_heads: int | None = None
        self._num_kv_heads: int | None = None
        self._head_dim: int | None = None
        # Set by register_callbacks(); None when using the legacy
        # register_hooks() path.
        self._registry: Any | None = None
        # ICMS_REAL_Q_CAPTURE=1: capture the model's genuine post-q_norm/
        # post-RoPE query at self.attn(q,k,v) and score Quest pages with it
        # instead of reconstructing via _compute_exact_q (which diverges,
        # cos 0.966; see project_quest_q_reconstruction_bug_2026-05-28).
        import os as _os_q
        self._real_q_capture: bool = (
            _os_q.environ.get("ICMS_REAL_Q_CAPTURE", "0") == "1")
        # layer_idx -> captured real query [num_tokens, H_q, D]; cleared per step.
        self._captured_q: dict[int, torch.Tensor] = {}
        # ICMS_STOP_WORLD_SCORE=1: score each scored layer S from its capture
        # callback (which fires just before S's attention → wait_for_layer(S))
        # using the GENUINE q_S, instead of scoring S a layer early in
        # _on_layer_complete with a reconstructed/adjacent query. The
        # adjacent-layer offset is fatal (HF offset=1 -> 0.0 acc; see
        # project_quest_q_reconstruction_bug_2026-05-28). Stop-world implies
        # real-q capture. Default OFF -> connector + hook behavior unchanged.
        self._stop_world: bool = (
            _os_q.environ.get("ICMS_STOP_WORLD_SCORE", "0") == "1")
        if self._stop_world:
            self._real_q_capture = True

    def register_hooks(
        self,
        model: nn.Module,
        decoder_layer_cls: type | None = None,
    ) -> int:
        """Register forward hooks on all decoder layers.

        Also extracts and caches LayerNorm and Q projection weights
        for exact Q computation.

        Args:
            model: The full model (e.g. LlamaForCausalLM).
            decoder_layer_cls: The decoder layer class to hook.

        Returns:
            Number of hooks registered.
        """
        if self._active:
            logger.warning("Quest hooks already registered, skipping")
            return 0

        if decoder_layer_cls is None:
            decoder_layer_cls = self._detect_decoder_layer_cls(model)

        if decoder_layer_cls is None:
            logger.warning(
                "Could not auto-detect decoder layer class. "
                "Quest hooks not registered."
            )
            return 0

        # Extract weights before registering hooks
        self._extract_layer_weights(model, decoder_layer_cls=decoder_layer_cls)

        count = 0
        pre_hook_layer0 = 0
        for name, module in model.named_modules():
            if isinstance(module, decoder_layer_cls):
                layer_idx = self._extract_layer_idx(name)
                if layer_idx is None:
                    continue

                handle = module.register_forward_hook(
                    self._make_post_layer_hook(layer_idx)
                )
                self._hook_handles.append(handle)
                count += 1

                # Additional pre-hook on layer 0 ONLY — post-hooks on
                # layer L-1 can only reconstruct Q_L for L>=1, so layer
                # 0 has no Q source otherwise. Pre-hook runs before
                # layer 0's forward with the embedding output as input
                # and recovers Q_0 using the already-cached layer-0
                # LayerNorm + Q-proj + RoPE weights. Keeping the rest
                # on post-hooks minimises surface area across decoder
                # variants (MLA etc.) that share the post-hook path.
                if layer_idx == 0:
                    pre_handle = module.register_forward_pre_hook(
                        self._make_pre_layer0_hook()
                    )
                    self._hook_handles.append(pre_handle)
                    pre_hook_layer0 += 1

        self._active = True
        logger.info(
            "Registered %d Quest forward hooks on %s layers "
            "(Q weights cached for %d layers)",
            count,
            decoder_layer_cls.__name__,
            len(self._q_proj_weights),
        )
        return count

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        if self._registry is not None:
            self._registry.clear()
        self._active = False
        self._current_selections.clear()

    # ------------------------------------------------------------------
    # New registry-based path (torch.compile / PIECEWISE-CUDAGraph compatible).
    #
    # Instead of register_forward_hook (which Dynamo strips during
    # compilation), we attach a QuestLayerCallbackRegistry on the model
    # and register two callbacks: pre-layer-0 and post-every-layer. The
    # patched model.forward calls fire_pre_layer / fire_post_layer (both
    # @torch._dynamo.disable'd) between consecutive layer(...) calls, so
    # Python runs in eager between captured per-layer pieces.
    # ------------------------------------------------------------------
    def register_callbacks(
        self,
        model: nn.Module,
        decoder_layer_cls: type | None = None,
    ) -> int:
        """Register Quest callbacks via the layer-callback registry.

        Equivalent to ``register_hooks`` but routes via the new registry
        path so callbacks survive ``torch.compile`` (PIECEWISE CUDAGraphs).

        Returns the number of decoder layers detected (0 if disabled).
        """
        from vllm.v1.attention.ops.quest_layer_callbacks import (
            QuestLayerCallbackRegistry,
            attach_registry,
        )

        if self._active:
            logger.warning(
                "Quest callbacks already registered, skipping")
            return 0

        if decoder_layer_cls is None:
            decoder_layer_cls = self._detect_decoder_layer_cls(model)
        if decoder_layer_cls is None:
            logger.warning(
                "Could not auto-detect decoder layer class. "
                "Quest callbacks not registered.")
            return 0

        # Cache LayerNorm + Q-proj + rotary weights for every layer.
        self._extract_layer_weights(model, decoder_layer_cls=decoder_layer_cls)

        # Attach the registry to the module that owns the for-layer loop.
        # The patched forward calls torch.ops.vllm.quest_fire_*_layer with
        # self.quest_layer_callbacks_marker (set by attach_registry); the
        # custom-op impl looks up the registry from a process-global table
        # using that marker. We attach to the *inner* model (Qwen3MoeModel,
        # LlamaModel, ...) since that's where the for-layer loop lives.
        target_module = self._find_layers_parent(model, decoder_layer_cls=decoder_layer_cls)
        if target_module is None:
            target_module = model
        registry = getattr(target_module, "quest_layer_callbacks", None)
        if registry is None:
            registry = QuestLayerCallbackRegistry()
        else:
            registry.clear()
        attach_registry(target_module, registry)
        self._registry = registry

        # Count layers (matches the old register_hooks return semantics).
        count = 0
        for _name, module in model.named_modules():
            if isinstance(module, decoder_layer_cls):
                count += 1

        # Register two adapters that reuse the existing hook bodies.
        registry.register_pre(self._registry_pre_callback)
        registry.register_post(self._registry_post_callback)

        # Real-query capture: register the capture callback and tag each
        # attention module so its forward emits the capture op (gated on
        # the per-module _quest_capture_enabled flag). target_module bears
        # the registry marker (attach_registry set it above).
        if self._real_q_capture:
            registry.register_capture_q(self._registry_capture_q_callback)
            _li = 0
            for _name, _mod in model.named_modules():
                if isinstance(_mod, decoder_layer_cls):
                    _attn = getattr(_mod, "self_attn", None)
                    if _attn is not None:
                        _attn._quest_marker_model = target_module
                        _attn._quest_layer_idx = _li
                        _attn._quest_capture_enabled = True
                    _li += 1
            logger.info(
                "[quest] real-query capture ENABLED on %d attention modules "
                "(ICMS_REAL_Q_CAPTURE=1) — scoring will use the model's "
                "genuine post-q_norm/post-RoPE query.", _li)

        self._active = True
        logger.info(
            "Registered Quest callbacks via registry on %s "
            "(%d layers detected, Q weights cached for %d)",
            decoder_layer_cls.__name__, count, len(self._q_proj_weights))
        return count

    @staticmethod
    def _find_layers_parent(
        model: nn.Module,
        decoder_layer_cls: type | None = None,
    ) -> nn.Module | None:
        """Locate the inner module whose ``self.layers`` is the decoder
        ModuleList. Mirrors ``_extract_layer_weights``'s walk.

        2026-05-30 fix: for multimodal hosts (e.g. Gemma3ForConditionalGeneration)
        the named_modules DFS visits self.vision_tower BEFORE self.language_model,
        so the FIRST ``.layers`` match is the vision encoder's
        ``vision_tower.vision_model.encoder.layers`` (27 SigLip layers), NOT the
        62 Gemma3DecoderLayer language layers. When decoder_layer_cls is
        provided, scan ALL candidates and prefer the one whose first element
        is an instance of decoder_layer_cls. Falls back to first-match if no
        class-typed candidate exists (legacy byte-identical for uniform models)."""
        candidates = []
        for _name, mod in model.named_modules():
            layers = getattr(mod, "layers", None)
            if layers is None:
                continue
            if not (isinstance(layers, (nn.ModuleList, nn.Sequential))
                    and len(layers) > 0):
                continue
            candidates.append(mod)
        if decoder_layer_cls is not None:
            for mod in candidates:
                if isinstance(mod.layers[0], decoder_layer_cls):
                    return mod
        return candidates[0] if candidates else None

    def _registry_pre_callback(self, layer_idx, positions, hidden_states,
                                residual, model):
        """Adapter from registry signature → existing pre-hook logic.

        Only layer 0 fires the actual pre-hook body; for L>=1 the
        post-hook on L-1 already covers the next-layer Q computation.
        """
        if layer_idx != 0:
            return
        # Reuse existing _on_layer0_pre body by synthesizing the input
        # tuple it expects: (positions, hidden_states, residual).
        synthetic_input = (positions, hidden_states, residual)
        self._on_layer0_pre(module=None, input=synthetic_input)

    def _registry_post_callback(self, layer_idx, positions, hidden_states,
                                 residual, model):
        """Adapter from registry signature → existing post-hook logic.

        Synthesizes the (input, output) tuple that ``_on_layer_complete``
        expects: input=(positions, ...), output=(hidden_states, residual).
        Since the post-hook reconstructs the residual stream from
        ``output[0] + output[1]``, passing the layer's true (h, r)
        return tuple is sufficient.
        """
        synthetic_input = (positions,)
        synthetic_output = (hidden_states, residual)
        self._on_layer_complete(
            layer_idx=layer_idx,
            module=None,
            input=synthetic_input,
            output=synthetic_output,
        )

    def _registry_capture_q_callback(self, layer_idx, q):
        """Stash the model's genuine post-q_norm/post-RoPE query for a layer.

        ``q`` is the first arg to ``self.attn(q, k, v)`` — shape
        ``[num_tokens, H_q*head_dim]`` (rank-local under TP). Reshape to
        ``[num_tokens, H_q, head_dim]`` to match the connector's scoring
        consumer (and the HF cleanroom layout). detach() only — no D2H, no
        sync; cheap enough to leave on the forward path.
        """
        if self._head_dim is None:
            return
        try:
            H = q.shape[-1] // self._head_dim  # rank-local H_q
            self._captured_q[int(layer_idx)] = (
                q.detach().view(-1, H, self._head_dim))
        except Exception:
            logger.exception("quest capture-q stash failed at layer=%d",
                             layer_idx)
            return
        # Stop-world: score THIS layer now with its genuine query. This runs
        # before self.attn(layer_idx) -> wait_for_layer(layer_idx), so the
        # connector's pending-score entry is populated in time for the wait to
        # pop + apply it (no connector changes). Gated; OFF by default.
        if self._stop_world:
            self._stop_world_score(int(layer_idx),
                                   self._captured_q[int(layer_idx)])

    def _stop_world_score(self, target_idx: int, query: "torch.Tensor") -> None:
        """Fire the Quest Score for the CURRENT scored layer using its genuine
        captured query (exact-layer q), mirroring _on_layer_complete's dispatch.

        Only the b<1.0 q-based score is fired here; the b>=1.0 all-pages
        prefetch and non-stride reuse stay in _on_layer_complete /
        _on_layer0_pre (fired ahead-of-time, no q needed) so they keep their
        compute/IO overlap and are not double-fired. The connector's
        save_kv_layer / wait / apply path is unchanged."""
        if self.kv_connector is None or query is None:
            return
        # M4 dense-mode short-circuit (same as _on_layer_complete).
        if (getattr(self.kv_connector, "is_dense_for_active_request", None)
                is not None
                and self.kv_connector.is_dense_for_active_request()):
            return
        # Only stride-aligned (scored) layers score here; non-stride layers
        # reuse the owner's selection via _on_layer_complete's reuse path.
        # 2026-05-30: scored_layers_set short-circuits the absolute modulo
        # for hybrid models (e.g. gemma-3) whose dense layers don't align
        # to score_stride. Uniform models leave scored_layers_set=None.
        _idx = int(target_idx)
        is_stride = (
            (self.scored_layers_set is not None and _idx in self.scored_layers_set)
            or self.score_stride <= 1
            or (_idx % self.score_stride) == 0)
        if not is_stride:
            return
        budget = self.budget_computer.compute_budget(
            approximate_scores=torch.empty(0),
            layer_idx=target_idx,
            num_layers=self.num_layers,
        )
        # b>=1.0 all-pages is fired ahead-of-time by _on_layer_complete /
        # _on_layer0_pre — don't double-fire (and it needs no query).
        if budget >= 1.0 and self.allow_all_pages_shortcut:
            return
        layer_name = f"layer_{target_idx}"
        if self.cpu_scoring:
            try:
                self.kv_connector.save_kv_layer(
                    layer_name=layer_name,
                    kv_layer=torch.empty(0),
                    attn_metadata=None,
                    quest_query=query,
                    next_layer_idx=target_idx,
                    budget=budget,
                    quest_stats=self.stats,
                )
            except Exception:
                logger.exception(
                    "stop-world cpu-score failed at layer=%d", target_idx)
        else:
            selected = self.page_selector.select_pages(
                query=query,
                layer_idx=target_idx,
                budget=budget,
                stats=self.stats,
            )
            self._current_selections[target_idx] = selected
            try:
                self.kv_connector.save_kv_layer(
                    layer_name=layer_name,
                    kv_layer=torch.empty(0),
                    attn_metadata=None,
                    next_layer_idx=target_idx,
                    budget=budget,
                    quest_selected_pages=selected,
                    quest_stats=self.stats,
                )
            except Exception:
                logger.exception(
                    "stop-world gpu-score failed at layer=%d", target_idx)

    def get_selection(self, layer_idx: int) -> torch.Tensor | None:
        """Get the computed page selection for a layer (if available)."""
        return self._current_selections.get(layer_idx)

    def clear_selections(self) -> None:
        """Clear all computed selections (called at end of each step)."""
        self._current_selections.clear()
        self._captured_q.clear()

    # -- Weight extraction ---------------------------------------------------

    def _extract_layer_weights(
        self,
        model: nn.Module,
        decoder_layer_cls: type | None = None,
    ) -> None:
        """Walk the model to extract LayerNorm and Q projection weights.

        For each layer i, caches:
        - self._layernorm_weights[i] = (weight, eps)
        - self._q_proj_weights[i] = W_Q  [q_size, hidden_size]

        2026-05-30 fix: for multimodal hosts (gemma-3), the FIRST ``.layers``
        match under model.named_modules() is the vision encoder's
        ModuleList (27 SigLip layers), NOT the language decoder (62 layers).
        Walking the first match cached vision-tower Q-projection weights
        (head_dim=72) into _q_proj_weights instead of the language model's
        Gemma3DecoderLayer Q-projections (head_dim=128), so every
        language-layer forward hook missed the lookup and save_kv_layer
        was never dispatched → adaptive bypassed entirely.
        When decoder_layer_cls is provided, prefer the candidate whose
        first element is an instance of that class."""
        layers_module = None
        layer_candidates = []
        for name, mod in model.named_modules():
            if name.endswith(".layers") or name == "layers":
                layer_candidates.append((name, mod))
        if decoder_layer_cls is not None:
            for _name, mod in layer_candidates:
                if (hasattr(mod, "__len__") and len(mod) > 0
                        and isinstance(mod[0], decoder_layer_cls)):
                    layers_module = mod
                    break
        if layers_module is None and layer_candidates:
            # Legacy fallback (uniform models): first match.
            layers_module = layer_candidates[0][1]

        if layers_module is None:
            logger.warning(
                "Could not find 'layers' module in model. "
                "Quest exact Q computation will be disabled."
            )
            return

        for layer_idx, layer in enumerate(layers_module):
            # Extract input_layernorm
            ln = getattr(layer, "input_layernorm", None)
            if ln is not None:
                weight = getattr(ln, "weight", None)
                eps = getattr(ln, "variance_epsilon", None)
                if eps is None:
                    eps = getattr(ln, "eps", 1e-6)
                if weight is not None:
                    self._layernorm_weights[layer_idx] = (
                        weight.data,
                        float(eps),
                    )

            # Extract Q projection from fused QKV
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue

            qkv_proj = getattr(self_attn, "qkv_proj", None)
            if qkv_proj is None:
                continue

            qkv_weight = getattr(qkv_proj, "weight", None)
            if qkv_weight is None:
                continue

            # Determine Q size from attention config
            num_heads = getattr(self_attn, "num_heads", None)
            head_dim = getattr(self_attn, "head_dim", None)
            num_kv_heads = getattr(self_attn, "num_kv_heads", None)

            if num_heads is None or head_dim is None:
                continue

            if self._num_heads is None:
                self._num_heads = num_heads
                self._head_dim = head_dim
                self._num_kv_heads = num_kv_heads

            q_size = num_heads * head_dim
            # Q portion is the first q_size rows of the fused QKV weight
            self._q_proj_weights[layer_idx] = (
                qkv_weight.data[:q_size, :]
            )

            # Cache rotary embedding module so captured Q is post-rope,
            # matching the K cache the attention kernel scores against.
            rotary_emb = getattr(self_attn, "rotary_emb", None)
            if rotary_emb is not None:
                self._rotary_embs[layer_idx] = rotary_emb

            # Cache q_norm (QK-Norm) for qwen3/qwen3-moe/gemma3. The real
            # forward applies this per-head RMSNorm over head_dim AFTER
            # q_proj and BEFORE RoPE (qwen3_moe.py:341-348). Without it the
            # scoring Q is scale-mismatched against the post-k_norm K
            # summaries the server computes. No-op for llama/mistral (the
            # attribute is absent → dict miss → skipped in _compute_exact_q).
            q_norm = getattr(self_attn, "q_norm", None)
            if q_norm is not None:
                qn_weight = getattr(q_norm, "weight", None)
                qn_eps = getattr(q_norm, "variance_epsilon", None)
                if qn_eps is None:
                    qn_eps = getattr(q_norm, "eps", 1e-6)
                if qn_weight is not None:
                    self._q_norm_weights[layer_idx] = (
                        qn_weight.data,
                        float(qn_eps),
                    )

        logger.info(
            "Extracted Quest Q weights for %d/%d layers "
            "(num_heads=%s, num_kv_heads=%s, head_dim=%s, rotary_embs=%d)",
            len(self._q_proj_weights),
            self.num_layers,
            self._num_heads,
            self._num_kv_heads,
            self._head_dim,
            len(self._rotary_embs),
        )

    # -- Core hook logic -----------------------------------------------------

    def _make_post_layer_hook(self, layer_idx: int):
        """Create a forward hook closure for a specific layer."""

        def hook(
            module: nn.Module,
            input: tuple,
            output: Any,
        ) -> None:
            self._on_layer_complete(layer_idx, module, input, output)

        return hook

    def _make_pre_layer0_hook(self):
        """Pre-hook for layer 0 so Q_0 can be captured.

        Post-hooks on layer L-1 can only reconstruct Q_L for L>=1 (there
        is no layer -1 whose output feeds layer 0). Without this pre-hook
        layer 0 falls through to vLLM's default block table and attends
        to whatever KV the scheduler had reserved, instead of the
        stride-group-0 selected pages. A pre-hook on layer 0's decoder
        module fires with the embedding output in `input`, lets us
        apply layer 0's cached LayerNorm + Q-proj + RoPE weights to
        compute Q_0, and dispatches via save_kv_layer with
        next_layer_idx=0 — the stride check fires for layer 0 under
        the {0, stride, 2*stride, …} pattern.
        """

        def pre_hook(module: nn.Module, input: tuple) -> None:
            self._on_layer0_pre(module, input)

        return pre_hook

    def _on_layer_complete(
        self,
        layer_idx: int,
        module: nn.Module,
        input: tuple,
        output: Any,
    ) -> None:
        """Called after each decoder layer's forward pass.

        Computes exact Q_{L+1} from the residual stream and selects
        pages for the next layer.
        """
        if layer_idx >= self.num_layers - 1:
            return

        # M4: connector flips rs.dense_mode once a bitmap-filtered
        # decode-mode Score returns no net-new pages. Skip Q compute
        # + every save_kv_layer dispatch for the rest of the request
        # so dense decode runs without per-layer Python callbacks.
        if (self.kv_connector is not None
                and getattr(self.kv_connector,
                            "is_dense_for_active_request", None) is not None
                and self.kv_connector.is_dense_for_active_request()):
            # M4 verification (ICMS_DIAG_DENSE=1): count + log Quest-hook
            # skips post-flip so we can confirm both the connector AND
            # the hook side observe dense_mode and bail out.
            import os as _os_d
            if _os_d.environ.get("ICMS_DIAG_DENSE") == "1":
                if not hasattr(self, "_quest_dense_skip_count"):
                    self._quest_dense_skip_count = 0
                self._quest_dense_skip_count += 1
                if (self._quest_dense_skip_count <= 5
                        or self._quest_dense_skip_count % 100 == 0):
                    import logging as _logging_d
                    _logging_d.getLogger(__name__).info(
                        "[quest] dense_skip layer=%d skip_count=%d",
                        layer_idx, self._quest_dense_skip_count)
            return

        next_layer_idx = layer_idx + 1

        # Can't compute Q without weights
        if next_layer_idx not in self._q_proj_weights:
            return
        if next_layer_idx not in self._layernorm_weights:
            return

        # Reconstruct the full residual stream that layer L+1's
        # input_layernorm will see.  vLLM's decoder layer returns
        # (hidden_states, residual) where `residual` is the stream
        # BEFORE the MLP add — the next layer's fused-add-norm does
        # `x = hidden_states + residual` inside input_layernorm and
        # then applies RMSNorm to that sum.  Using just `output[1]`
        # here drops the layer-L MLP output from the input to Q_{L+1}.
        if isinstance(output, tuple) and len(output) >= 2:
            hidden_states_out = output[0]
            residual_out = output[1]
            if hidden_states_out is not None and residual_out is not None:
                residual = hidden_states_out + residual_out
            elif residual_out is not None:
                residual = residual_out
            elif hidden_states_out is not None:
                residual = hidden_states_out
            else:
                return
        elif isinstance(output, torch.Tensor):
            residual = output
        else:
            return

        if residual is None:
            return

        # Extract positions from the layer's forward input.
        # LlamaDecoderLayer.forward(positions, hidden_states, residual).
        positions: torch.Tensor | None = None
        if isinstance(input, tuple) and len(input) >= 1:
            first = input[0]
            if isinstance(first, torch.Tensor):
                positions = first

        # Compute budget for this layer
        budget = self.budget_computer.compute_budget(
            approximate_scores=torch.empty(0),
            layer_idx=next_layer_idx,
            num_layers=self.num_layers,
        )

        # ICMS_DISABLE_ALL_PAGES_SHORTCUT=1: when set, b>=1.0 falls
        # through to the regular Score+select-K path with K=total_pages
        # instead of the FetchAll shortcut. Use to localize the
        # 2026-05-08 bug where qwen3 niah_multikey_2 b=1.0=0.067 <
        # b=0.50=0.400 — if this env knob recovers b=1.0 ≈ dense, the
        # bug is in the FetchAll path; if not, it's elsewhere.
        import os as _os_aps
        _aps_disabled = (
            _os_aps.environ.get("ICMS_DISABLE_ALL_PAGES_SHORTCUT", "0") == "1")
        if budget >= 1.0 and self.allow_all_pages_shortcut and not _aps_disabled:
            # Baseline pipelined path: transfer ALL pages for the next
            # layer without computing Q or scoring.  This gives the
            # baseline full compute/IO overlap (layer-pipelined) without
            # paying for the Q projection or CPU scoring overhead.
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_all_pages=True,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
            return

        # Stride gating: only stride-aligned layers need a fresh Q.
        # Scored set = {0, stride, 2*stride, …}. Layer 0 is handled by a
        # dedicated pre-hook (see _make_pre_layer0_hook); non-stride
        # layers reuse the prior stride group's Score result and route
        # through the connector's reuse path without computing Q.
        # Independent of budget — non-stride layers never consult Q.
        # 2026-05-30: scored_layers_set short-circuits the absolute
        # modulo for hybrid models — see __init__ note for rationale.
        is_stride_layer = (
            (self.scored_layers_set is not None
             and next_layer_idx in self.scored_layers_set)
            or self.score_stride <= 1
            or (next_layer_idx % self.score_stride) == 0
        )
        if not is_stride_layer:
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_reuse_selection=True,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
            return

        if self.single_layer_scoring and layer_idx > 0:
            # Single-layer scoring baseline: layer 0 computed Q and scored
            # pages; all subsequent layers reuse that selection.  The connector
            # looks up its cached _quest_last_selection and immediately starts
            # the transfer — no Q computation or scoring in the critical path.
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_reuse_selection=True,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
            return

        # Stop-world: do NOT score this stride layer a step early with a
        # reconstructed/adjacent query. Its genuine q_{next_layer_idx} is
        # captured at next_layer_idx's own forward and scored from the capture
        # callback (_stop_world_score) just before its attention. We've already
        # let the b>=1.0 all-pages prefetch + non-stride reuse fire above, so
        # only the b<1.0 q-based score is skipped here.
        if self._stop_world:
            return

        # Compute exact Q for the next layer (timed for stats).
        if self.stats is not None:
            self.stats.begin_q_compute(next_layer_idx)

        # Real-query capture: prefer the model's genuine query over the
        # reconstruction. At fire_post_layer(L) we have captured_q[L] (layer
        # L's true post-q_norm/post-RoPE query) but score next_layer_idx=L+1.
        # Using L's real q for L+1's scoring is the "adjacent-layer real q"
        # variant: it eliminates the q_norm + residual-ordering + RoPE
        # reconstruction errors (cos 0.62/0.966) at the cost of a 1-layer
        # query offset (adjacent queries are highly correlated). Falls back
        # to reconstruction when capture is off or the layer wasn't captured.
        query = None
        if self._real_q_capture:
            query = self._captured_q.get(layer_idx)
        if query is None:
            query = self._compute_exact_q(residual, next_layer_idx, positions)

        if self.stats is not None:
            self.stats.end_q_compute(next_layer_idx)

        if query is None:
            return

        if self.cpu_scoring:
            # CPU-side scoring: send the query vector to the connector.
            # The connector will copy Q to CPU, score against CPU-resident
            # page metadata, select pages, and issue a selective transfer.
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_query=query,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
        else:
            # GPU-side scoring: score and select pages on GPU, then send
            # the selected page indices to the connector for transfer.
            selected = self.page_selector.select_pages(
                query=query,
                layer_idx=next_layer_idx,
                budget=budget,
                stats=self.stats,
            )

            self._current_selections[next_layer_idx] = selected

            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name=f"layer_{layer_idx}",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        hidden_states=residual,
                        next_layer_idx=next_layer_idx,
                        budget=budget,
                        quest_selected_pages=selected,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass

    def _on_layer0_pre(
        self,
        module: nn.Module,
        input: tuple,
    ) -> None:
        """Pre-hook for layer 0.

        Mirrors _on_layer_complete's control flow (budget gating, stride
        check, FetchAll shortcut, Score dispatch) but reads the residual
        stream directly from `input` instead of reconstructing it from a
        post-hook output. vLLM's decoder layer takes
        (positions, hidden_states, residual); for layer 0 residual is
        typically None so the layer's input_layernorm operates on
        hidden_states alone.
        """
        # M4: dense-mode short-circuit — see _on_layer_complete.
        if (self.kv_connector is not None
                and getattr(self.kv_connector,
                            "is_dense_for_active_request", None) is not None
                and self.kv_connector.is_dense_for_active_request()):
            return

        if 0 not in self._q_proj_weights or 0 not in self._layernorm_weights:
            return

        if not isinstance(input, tuple) or len(input) < 2:
            return

        positions: torch.Tensor | None = None
        if isinstance(input[0], torch.Tensor):
            positions = input[0]

        hidden_states = input[1]
        if not isinstance(hidden_states, torch.Tensor):
            return

        residual_in = input[2] if len(input) >= 3 else None
        if isinstance(residual_in, torch.Tensor):
            stream = hidden_states + residual_in
        else:
            stream = hidden_states

        # Budget for layer 0.
        budget = self.budget_computer.compute_budget(
            approximate_scores=torch.empty(0),
            layer_idx=0,
            num_layers=self.num_layers,
        )

        # FetchAll shortcut (Config B's path): fire once; subsequent
        # layers' post-hook shortcut also fires, which is harmless under
        # quest_all_pages=True. Matches _on_layer_complete semantics.
        # ICMS_DISABLE_ALL_PAGES_SHORTCUT=1 also disables this layer-0
        # entry-point — see _on_layer_complete for rationale.
        import os as _os_aps0
        _aps0_disabled = (
            _os_aps0.environ.get("ICMS_DISABLE_ALL_PAGES_SHORTCUT", "0") == "1")
        if budget >= 1.0 and self.allow_all_pages_shortcut and not _aps0_disabled:
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name="layer_pre_0",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_all_pages=True,
                        next_layer_idx=0,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
            return

        # Stop-world: layer 0's genuine q_0 is captured at layer 0's own
        # forward and scored from the capture callback (_stop_world_score) just
        # before layer 0's attention — this finally retires the broken layer-0
        # _compute_exact_q reconstruction. (b>=1.0 all-pages already fired
        # above.) Skip the ahead-of-time reconstruction score here.
        if self._stop_world:
            return

        # Layer 0 is always a stride-aligned layer under the
        # {0, stride, 2*stride, …} pattern, so always compute Q here.
        if self.stats is not None:
            self.stats.begin_q_compute(0)

        query = self._compute_exact_q(stream, 0, positions)

        if self.stats is not None:
            self.stats.end_q_compute(0)

        if query is None:
            return

        if self.cpu_scoring:
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name="layer_pre_0",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        quest_query=query,
                        next_layer_idx=0,
                        budget=budget,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass
        else:
            if self.page_selector is None:
                return
            selected = self.page_selector.select_pages(
                query=query,
                layer_idx=0,
                budget=budget,
                stats=self.stats,
            )
            self._current_selections[0] = selected
            if self.kv_connector is not None:
                try:
                    self.kv_connector.save_kv_layer(
                        layer_name="layer_pre_0",
                        kv_layer=torch.empty(0),
                        attn_metadata=None,
                        hidden_states=stream,
                        next_layer_idx=0,
                        budget=budget,
                        quest_selected_pages=selected,
                        quest_stats=self.stats,
                    )
                except Exception:
                    pass

    @torch.no_grad()
    def _compute_exact_q(
        self,
        residual: torch.Tensor,
        next_layer_idx: int,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Compute exact Q_{L+1} from the residual stream.

        Steps:
        1. Apply RMSNorm using layer L+1's layernorm weights.
        2. Project through layer L+1's Q weight matrix.
        3. Apply rotary position embedding so Q matches the
           post-rope K that the attention kernel scores against.
        4. Reshape to [B, num_heads, head_dim].

        Args:
            residual: Post-layer residual, shape [B, hidden_size] or
                [num_tokens, hidden_size].
            next_layer_idx: Index of the next layer.
            positions: Per-token positions for rotary embedding
                (shape [num_tokens]). If None, RoPE is skipped and the
                returned Q is pre-rope (legacy behavior).

        Returns:
            Q tensor of shape [B, num_heads, head_dim], or None on error.
        """
        ln_weight, eps = self._layernorm_weights[next_layer_idx]
        q_weight = self._q_proj_weights[next_layer_idx]

        device = residual.device
        ln_w = ln_weight.to(device)
        q_w = q_weight.to(device)

        # RMSNorm: x * (1/rms) * weight
        # rms = sqrt(mean(x^2) + eps)
        variance = residual.to(torch.float32).pow(2).mean(-1, keepdim=True)
        h_normed = residual * torch.rsqrt(variance + eps)
        h_normed = (h_normed * ln_w).to(residual.dtype)

        # Q projection: [B, hidden_size] @ [q_size, hidden_size]^T
        # -> [B, q_size]
        q = h_normed @ q_w.t()

        # QK-Norm: qwen3/qwen3-moe/gemma3 apply a per-head RMSNorm over
        # head_dim AFTER q_proj and BEFORE RoPE (qwen3_moe.py:341-348). The K
        # cache the server summarizes is post-k_norm, so the scoring Q MUST be
        # post-q_norm or the per-channel scales mismatch and Quest page
        # ranking is scrambled on every scored layer. No-op when q_norm wasn't
        # captured (llama/mistral have none).
        qn = self._q_norm_weights.get(next_layer_idx)
        if qn is not None and self._head_dim is not None:
            qn_w, qn_eps = qn
            qn_w = qn_w.to(device)
            # Reshape flat [num_tokens, q_size] → [num_tokens, n_heads, head_dim],
            # RMSNorm over the last dim, reshape back — mirrors the real
            # forward's view(..., n_heads, head_dim) → q_norm → view(q.shape).
            orig_shape = q.shape
            q_bh = q.view(-1, q.shape[-1] // self._head_dim, self._head_dim)
            qn_var = q_bh.to(torch.float32).pow(2).mean(-1, keepdim=True)
            q_bh = q_bh * torch.rsqrt(qn_var + qn_eps)
            q_bh = (q_bh * qn_w).to(q.dtype)
            q = q_bh.view(orig_shape)

        # Apply rotary embedding so the captured Q is post-rope, aligned
        # with the K cache. The attention path does rope in-place on the
        # flat [num_tokens, q_size] tensor; mirror that here.
        rotary_emb = self._rotary_embs.get(next_layer_idx)
        # ROPE PROBE: log the positions fed to RoPE once at layer 0. If these
        # are chunk-RELATIVE (0..N) rather than ABSOLUTE (~haystack_len), the
        # scoring Q is rotated at the wrong frequencies → diverges from the
        # model's true query (2026-05-28 investigation).
        import os as _os_rp
        if (_os_rp.environ.get("ICMS_ROPE_PROBE", "") == "1"
                and next_layer_idx == 0
                and getattr(self, "_rope_probe_n", 0) < 8
                and positions is not None):
            self._rope_probe_n = getattr(self, "_rope_probe_n", 0) + 1
            try:
                _p = positions.flatten()
                logger.info(
                    "[rope-probe #%d] layer0 _compute_exact_q positions: "
                    "n=%d first3=%s last3=%s min=%d max=%d",
                    self._rope_probe_n,
                    _p.numel(), _p[:3].tolist(), _p[-3:].tolist(),
                    int(_p.min()), int(_p.max()))
            except Exception:
                pass
        if (
            rotary_emb is not None
            and positions is not None
            and self._num_kv_heads is not None
            and self._head_dim is not None
        ):
            q = q.contiguous()
            k_dummy = torch.empty(
                q.shape[0],
                self._num_kv_heads * self._head_dim,
                dtype=q.dtype,
                device=q.device,
            )
            q, _ = rotary_emb(positions, q, k_dummy)

        # Reshape to [B, num_heads, head_dim]
        if self._num_heads is not None and self._head_dim is not None:
            q = q.view(-1, self._num_heads, self._head_dim)

        return q

    # -- Static helpers ------------------------------------------------------

    @staticmethod
    def _detect_decoder_layer_cls(model: nn.Module) -> type | None:
        """Auto-detect the decoder layer class."""
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if "DecoderLayer" in cls_name:
                return type(module)
        return None

    @staticmethod
    def _extract_layer_idx(module_name: str) -> int | None:
        """Extract layer index from module name like 'model.layers.5'."""
        parts = module_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None
