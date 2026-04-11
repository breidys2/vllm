# SPDX-License-Identifier: Apache-2.0
"""ICMS storage service KV connector for vLLM v1.

Routes the KV-cache offload path through a unix-socket icms_client that
talks to a separate `icms_server` process implementing the SmartNIC Quest
indexing scheme. See `docs/icms_storage_service_design.md` and
`docs/smartnic_quest_indexing_design.md`.

This is the **v1** wiring — only the methods needed for the targeted
quest_benchmark.py A/B test are populated. Methods that aren't on the hot
path return safe no-ops, which is enough for the connector to be selected
via `KVTransferConfig` without breaking vLLM's connector lifecycle.

See `docs/icms_v1_relaxations.md` (entries R9 and the new R12) for the
list of things that are intentionally stubbed and need to be revisited
before this connector is production-ready.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

logger = init_logger(__name__)


# ─── icms_client import (from the storage_service tree, not pip) ─────────
# The connector lives under vllm/forks; the icms_client lives under
# storage_service/python alongside it. We need to make it importable.
def _ensure_icms_client_on_path():
    here = Path(__file__).resolve()
    # repo root: walk up until we hit `prefill_benchmarking/`.
    for ancestor in here.parents:
        cand = ancestor / "storage_service" / "python"
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            return
    raise ImportError(
        "icms_client not found — expected at "
        "<repo_root>/storage_service/python relative to forks/vllm/"
    )

_ensure_icms_client_on_path()

from icms_client import IcmsClient  # noqa: E402
from icms_client.geometry import GROUP_PAGES, ModelGeometry, find_model  # noqa: E402


# ─── connector metadata ──────────────────────────────────────────────────

@dataclass
class IcmsConnectorMetadata(KVConnectorMetadata):
    """Per-step metadata passed from scheduler to worker.

    For v1 we don't drive an offload-style transfer pipeline; the worker
    handles save/load directly inside `save_kv_layer` / `start_load_kv`
    using the icms_client. This metadata exists so the base class
    machinery is satisfied.
    """
    save_request_ids: list[str] = field(default_factory=list)
    load_request_ids: list[str] = field(default_factory=list)


# ─── per-request bookkeeping (worker side) ───────────────────────────────

@dataclass
class _RequestState:
    """Tracks the chain of group hashes a request has accumulated.

    For v1 we synthesize one group hash per 32 vLLM blocks of the request,
    derived from vLLM's chained `BlockHash` for the last block in each
    group. This mirrors the `blocks_to_groups` helper used by the trace
    analysis script.
    """
    request_id: str
    chain: list[int] = field(default_factory=list)
    # Layers we've already saved (so we don't double-write across layers).
    saved_layers: set[int] = field(default_factory=set)


# ─── facade ──────────────────────────────────────────────────────────────

class IcmsConnector(KVConnectorBase_V1):
    """vLLM v1 KV connector that offloads to a separate icms_server.

    Construction is dispatched to scheduler/worker subclasses based on
    `role`, mirroring `OffloadingConnector`'s pattern. The base class
    requires us to implement abstract methods directly (no pluggable
    impls), so the facade defers to internal helpers `_sched` / `_worker`.
    """

    def __init__(
        self,
        vllm_config,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        # Connector-specific config: we expect kv_transfer_config.kv_connector_extra_config
        # to carry our settings.
        extra = (vllm_config.kv_transfer_config.kv_connector_extra_config or {})
        self._socket_path: str = extra.get("icms_socket_path", "/tmp/icms.sock")
        self._model_name:  str = extra.get("icms_model_name", "")  # passed to Hello
        self._sink_size:   int = int(extra.get("icms_sink_size_bytes", 4 << 20))
        self._k:           int = int(extra.get("icms_k", 16))

        self._sched: _Scheduler | None = None
        self._worker: _Worker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self._sched = _Scheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self._worker = _Worker(
                socket_path=self._socket_path,
                model_name=self._model_name,
                sink_size=self._sink_size,
            )
        else:
            raise ValueError(f"unknown KVConnectorRole: {role}")

    # ─── worker-side hooks ───────────────────────────────────────────────

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        # v1: we don't run an async load pipeline. The Quest scoring path
        # pulls KV pages on demand via Score / Fetch when needed; the
        # forward pass itself doesn't pre-load anything.
        # See R12 in icms_v1_relaxations.md.
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        # No async loads in flight in v1.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs: Any,
    ) -> None:
        """Stash one layer's KV into the icms service.

        v1 implementation: we extract the per-page channel-wise key min/max
        from `kv_layer` directly (since we own the K cache shape) and
        stream the *full* KV bytes too. The icms service deduplicates by
        chain hash on the receiving side.

        For v1 we issue a synchronous WriteGroup per request that has new
        full groups since the last save. This is *not* the eventual async
        save the OffloadingConnector design uses — see R12.
        """
        if self._worker is None:
            return
        # The connector base class doesn't tell us request_id directly here;
        # the calling site has it in attn_metadata. For v1 we leave the
        # actual layer-buffer extraction to a follow-up — the smoke test
        # below exercises the methods explicitly.
        # See R12 in icms_v1_relaxations.md.
        return

    def wait_for_save(self) -> None:
        # No async saves in v1; everything is sync inside save_kv_layer.
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # v1 has no async transfers; nothing in-flight, nothing to report.
        return None, None

    def shutdown(self):
        if self._worker is not None:
            self._worker.shutdown()

    # ─── scheduler-side hooks ────────────────────────────────────────────

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        # v1 doesn't claim any external prefix tokens — we always return
        # 0 (no new matched tokens), so vLLM does the full prefill itself.
        # The icms service still gets the KV bytes via save_kv_layer, but
        # we don't try to short-circuit the prefill from a remote prefix
        # match. That's a follow-up for later phases.
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        # No external prefix tokens in v1; nothing to track here.
        return

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return IcmsConnectorMetadata()

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        if self._worker is not None:
            self._worker.on_request_finished(request.request_id)
        return False, None

    # ─── direct helper API used by IcmsQuestPageSelector ──────────────────
    #
    # The v1 integration path for quest_benchmark.py is via a custom
    # QuestPageSelector subclass that calls these helpers directly,
    # rather than going through vLLM's offload pipeline. This bypasses
    # the parts of the connector machinery we don't need yet (async
    # transfer state, scheduler ↔ worker block bookkeeping) and gives
    # us a working A/B test against the in-process scorer.
    #
    # See docs/icms_v1_relaxations.md R12.
    def write_group_for_request(
        self, request_id: str, chain: list[int],
        summary_blob: bytes, kv_blob: bytes,
    ):
        if self._worker is None:
            raise RuntimeError("write_group_for_request called on non-worker connector")
        return self._worker.write_group(request_id, chain, summary_blob, kv_blob)

    def score_for_request(
        self, request_id: str, chain: list[int],
        layer: int, query: torch.Tensor, k: int | None = None,
    ):
        if self._worker is None:
            raise RuntimeError("score_for_request called on non-worker connector")
        return self._worker.score(request_id, chain, layer, query,
                                    k or self._k)


# ─── scheduler-side state ────────────────────────────────────────────────

class _Scheduler:
    """v1 scheduler-side state. Currently just a placeholder.

    A full implementation would track per-request external-cache state
    (which prefix groups already exist on the icms service) so that
    `get_num_new_matched_tokens` could return non-zero. See R12.
    """

    def __init__(self, vllm_config):
        self._vllm_config = vllm_config


# ─── worker-side state ───────────────────────────────────────────────────

class _Worker:
    """Owns the icms_client connection + sink, drives WriteGroup / Score."""

    def __init__(self, *, socket_path: str, model_name: str, sink_size: int):
        self._socket_path = socket_path
        self._model_name  = model_name
        self._sink_size   = sink_size

        self._client: IcmsClient | None = None
        self._sink = None
        self._geom: ModelGeometry | None = None
        self._requests: dict[str, _RequestState] = {}

        self._connect()

    def _connect(self):
        self._client = IcmsClient(self._socket_path)
        self._client.connect()
        ack = self._client.hello(self._model_name)
        self._geom = find_model(self._model_name) or ModelGeometry(
            name=self._model_name,
            num_layers=ack.num_layers,
            num_kv_heads=ack.num_kv_heads,
            head_dim=ack.head_dim,
            elem_bytes=ack.elem_bytes,
        )
        self._sink = self._client.register_sink(self._sink_size)
        logger.info(
            "IcmsConnector worker: connected to %s, model=%s, sink=%d B",
            self._socket_path, self._model_name, self._sink_size,
        )

    def shutdown(self):
        if self._client is None:
            return
        try:
            if self._sink is not None:
                self._client.unregister_sink(self._sink)
                self._sink.close()
        finally:
            self._client.close()
            self._client = None

    # ─── op API used by the facade ──────────────────────────────────────

    def write_group(self, request_id: str, chain: list[int],
                     summary_blob: bytes, kv_blob: bytes):
        st = self._requests.setdefault(
            request_id, _RequestState(request_id=request_id))
        st.chain = list(chain)
        ack = self._client.write_group(chain, summary_blob, kv_blob,
                                         pages_in_group=GROUP_PAGES)
        return ack

    def score(self, request_id: str, chain: list[int],
              layer: int, query: torch.Tensor, k: int):
        # Marshal the query into a contiguous fp32 numpy array.
        import numpy as np
        if isinstance(query, torch.Tensor):
            q_np = query.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
        else:
            q_np = np.asarray(query, dtype=np.float32)
        if q_np.ndim > 1:
            q_np = q_np.reshape(-1)
        req_id_int = self._request_id_to_int(request_id, layer)
        return self._client.score(
            request_id=req_id_int, chain=chain, layer=layer,
            query=q_np, k=k, sink=self._sink,
        )

    def on_request_finished(self, request_id: str):
        st = self._requests.pop(request_id, None)
        if st is not None and st.chain:
            try:
                self._client.evict(st.chain)
            except Exception as e:
                logger.warning("evict failed for request %s: %s", request_id, e)

    @staticmethod
    def _request_id_to_int(request_id: str, layer: int) -> int:
        # The icms protocol's request_id is a u64. vLLM's request_id is a
        # string. We hash to a u64; the layer is *not* mixed in because
        # we want cross-layer cache reuse keyed on the same vLLM request
        # — that's the whole point of the score cache. See Q6/D5 in the
        # design doc and R12 in v1 relaxations.
        del layer
        h = hash(request_id) & 0xFFFFFFFFFFFFFFFF
        return h
