from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

import torch


_ACTIVATION_NAMES = (
    "vllm_ascend_moe_activation",
    "vllm_ascend_expert_activation",
    "moe_activation",
    "expert_activation",
    "expert_histogram",
    "moe_expert_ids",
    "expert_ids",
)
_COST_NAMES = (
    "vllm_ascend_expert_cost",
    "expert_cost",
    "scheduler_cost",
    "priority",
)
_SOURCE_NAMES = (
    "metrics",
    "sampling_params",
    "request_metadata",
    "metadata",
    "extra_args",
    "extra_body",
)
_VALID_POLICIES = ("auto", "fifo", "throughput", "expert", "offload")


class RequestPhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class SchedulerParameters:
    policy: str
    reorder_window: int
    decode_reserve_ratio: float
    activation_similarity_threshold: float
    max_profile_experts: int
    offload_count: int
    uses_cold_buffer: bool


@dataclass(frozen=True)
class ActivationProfile:
    """Sparse, L2-normalized MoE activation histogram."""

    weights: tuple[tuple[str, float], ...]

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        max_experts: int = 32,
    ) -> ActivationProfile | None:
        counts: dict[str, float] = defaultdict(float)
        _collect_activation(value, counts)
        positive = [(key, weight) for key, weight in counts.items()
                    if math.isfinite(weight) and weight > 0]
        if not positive:
            return None
        positive.sort(key=lambda item: (-item[1], item[0]))
        positive = positive[:max(1, int(max_experts))]
        norm = math.sqrt(sum(weight * weight for _, weight in positive))
        if norm <= 0:
            return None
        return cls(tuple(sorted((key, weight / norm)
                                for key, weight in positive)))

    def similarity(self, other: ActivationProfile | None) -> float:
        if other is None:
            return 0.0
        right = dict(other.weights)
        return max(0.0, min(1.0, sum(
            value * right.get(key, 0.0) for key, value in self.weights
        )))

    def merge(
        self,
        other: ActivationProfile,
        *,
        max_experts: int,
    ) -> ActivationProfile:
        values: dict[str, float] = defaultdict(float)
        for key, value in (*self.weights, *other.weights):
            values[key] += value
        merged = ActivationProfile.from_value(values, max_experts=max_experts)
        assert merged is not None
        return merged


class ActivationRegistry:
    """Scheduler-process store for request profiles learned at runtime."""

    def __init__(self) -> None:
        self._profiles: dict[str, ActivationProfile] = {}
        self._lock = threading.Lock()

    def record(
        self,
        request_id: str,
        activation: Any,
        *,
        max_experts: int = 32,
    ) -> ActivationProfile | None:
        profile = ActivationProfile.from_value(
            activation, max_experts=max_experts)
        if profile is None:
            return None
        key = str(request_id)
        with self._lock:
            previous = self._profiles.get(key)
            if previous is not None:
                profile = previous.merge(profile, max_experts=max_experts)
            self._profiles[key] = profile
        return profile

    def get(self, request_id: str) -> ActivationProfile | None:
        with self._lock:
            return self._profiles.get(str(request_id))

    def discard(self, request_id: str) -> None:
        with self._lock:
            self._profiles.pop(str(request_id), None)

    def clear(self) -> None:
        with self._lock:
            self._profiles.clear()


ACTIVATION_REGISTRY = ActivationRegistry()


class WorkerActivationCollector:
    """Aggregate per-request MoE routing on the model worker.

    Token-to-request ranges come from the V1 model runner. Expert histograms
    stay on device until the end of the model step, then only top experts are
    copied to CPU for the ModelRunnerOutput payload.
    """

    def __init__(self) -> None:
        self._runtime_config: Any = None
        self._step = 0
        self._req_ids: tuple[str, ...] = ()
        self._token_counts: tuple[int, ...] = ()
        self._collect_requests: set[int] = set()
        self._selected_layers: set[int] = set()
        self._layer_counts: dict[int, torch.Tensor] = {}
        self._request_indices: dict[str, torch.Tensor] = {}
        self._seen_requests: set[str] = set()

    def configure(self, runtime_config: Any) -> None:
        self._runtime_config = runtime_config

    def begin_batch(
        self,
        req_ids: Iterable[str],
        token_counts: Iterable[int],
    ) -> None:
        self._step += 1
        self._req_ids = tuple(str(req_id) for req_id in req_ids)
        self._token_counts = tuple(max(0, int(value))
                                   for value in token_counts)
        self._layer_counts.clear()
        self._request_indices.clear()

        interval = self._feedback_interval()
        periodic = self._step % interval == 0
        self._collect_requests = {
            index for index, req_id in enumerate(self._req_ids)
            if periodic or req_id not in self._seen_requests
        }
        self._selected_layers = self._feedback_layers()

    def record(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> None:
        layer_id = int(layer_id)
        if (not self._collect_requests
                or layer_id not in self._selected_layers
                or topk_ids.ndim != 2
                or num_experts <= 0):
            return
        valid_tokens = sum(self._token_counts)
        # ACLGraph padding is appended and can be ignored. A shorter tensor
        # means SP/another transform changed token ownership, so request ranges
        # are no longer trustworthy.
        if valid_tokens <= 0 or topk_ids.shape[0] < valid_tokens:
            return

        routed = topk_ids[:valid_tokens].to(dtype=torch.long)
        request_index = self._request_index(routed.device)
        top_k = int(routed.shape[1])
        flat_requests = request_index.repeat_interleave(top_k)
        flat_experts = routed.reshape(-1)
        valid = (flat_experts >= 0) & (flat_experts < num_experts)
        combined = flat_requests[valid] * num_experts + flat_experts[valid]
        counts = torch.zeros(
            len(self._req_ids) * num_experts,
            dtype=torch.long,
            device=routed.device,
        )
        counts.index_add_(0, combined, torch.ones_like(combined))
        counts = counts.reshape(len(self._req_ids), num_experts)
        previous = self._layer_counts.get(layer_id)
        self._layer_counts[layer_id] = (
            counts if previous is None else previous + counts)

    def finish_batch(self) -> dict[str, dict[str, int]]:
        if not self._layer_counts or not self._collect_requests:
            self._reset_batch()
            return {}

        max_experts = self._max_profile_experts()
        payload: dict[str, dict[str, int]] = {
            self._req_ids[index]: {} for index in self._collect_requests
        }
        for layer_id, counts in self._layer_counts.items():
            width = min(max_experts, int(counts.shape[1]))
            values, experts = torch.topk(counts, k=width, dim=1)
            values_cpu = values.detach().cpu().tolist()
            experts_cpu = experts.detach().cpu().tolist()
            for request_index in self._collect_requests:
                request_profile = payload[self._req_ids[request_index]]
                for expert_id, count in zip(
                        experts_cpu[request_index],
                        values_cpu[request_index]):
                    if int(count) > 0:
                        request_profile[f"{layer_id}/{int(expert_id)}"] = int(
                            count)

        payload = {req_id: profile for req_id, profile in payload.items()
                   if profile}
        self._seen_requests.update(payload)
        self._reset_batch()
        return payload

    def discard(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            self._seen_requests.discard(str(request_id))

    def _request_index(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._request_indices.get(key)
        if cached is None:
            counts = torch.tensor(
                self._token_counts, dtype=torch.long, device=device)
            cached = torch.repeat_interleave(
                torch.arange(len(self._req_ids), device=device), counts)
            self._request_indices[key] = cached
        return cached

    def _feedback_interval(self) -> int:
        explicit = getattr(
            self._runtime_config, "scheduler_feedback_interval", None)
        if explicit is not None:
            return max(1, int(explicit))
        offload_count = int(getattr(
            self._runtime_config, "offload_count", 0))
        return max(4, 16 - min(8, offload_count))

    def _feedback_layers(self) -> set[int]:
        layer_ids = sorted(set(getattr(
            self._runtime_config, "runtime_layer_ids", []) or []))
        if not layer_ids:
            return set()
        explicit = getattr(
            self._runtime_config, "scheduler_feedback_max_layers", None)
        limit = max(1, int(explicit)) if explicit is not None else 4
        if len(layer_ids) <= limit:
            return set(layer_ids)
        if limit == 1:
            return {layer_ids[len(layer_ids) // 2]}
        indices = {
            round(index * (len(layer_ids) - 1) / (limit - 1))
            for index in range(limit)
        }
        return {layer_ids[index] for index in indices}

    def _max_profile_experts(self) -> int:
        explicit = getattr(
            self._runtime_config, "scheduler_max_profile_experts", None)
        if explicit is not None:
            return max(1, int(explicit))
        offload_count = int(getattr(
            self._runtime_config, "offload_count", 0))
        return max(8, min(64, 8 + offload_count * 4))

    def _reset_batch(self) -> None:
        self._req_ids = ()
        self._token_counts = ()
        self._collect_requests.clear()
        self._selected_layers.clear()
        self._layer_counts.clear()
        self._request_indices.clear()


WORKER_ACTIVATION_COLLECTOR = WorkerActivationCollector()


def record_request_activation(
    request_id: str,
    activation: Any,
    *,
    max_experts: int = 32,
) -> ActivationProfile | None:
    return ACTIVATION_REGISTRY.record(
        request_id, activation, max_experts=max_experts)


def clear_request_activation(request_id: str | None = None) -> None:
    if request_id is None:
        ACTIVATION_REGISTRY.clear()
    else:
        ACTIVATION_REGISTRY.discard(request_id)


def configure_worker_activation(runtime_config: Any) -> None:
    WORKER_ACTIVATION_COLLECTOR.configure(runtime_config)


def begin_worker_activation_batch(
    req_ids: Iterable[str],
    token_counts: Iterable[int],
) -> None:
    WORKER_ACTIVATION_COLLECTOR.begin_batch(req_ids, token_counts)


def record_worker_expert_activation(
    layer_id: int,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> None:
    WORKER_ACTIVATION_COLLECTOR.record(layer_id, topk_ids, num_experts)


def finish_worker_activation_batch() -> dict[str, dict[str, int]]:
    return WORKER_ACTIVATION_COLLECTOR.finish_batch()


def discard_worker_activation_requests(request_ids: Iterable[str]) -> None:
    WORKER_ACTIVATION_COLLECTOR.discard(request_ids)


def derive_scheduler_parameters(
    runtime_config: Any,
    scheduler_config: Any,
) -> SchedulerParameters:
    """Resolve scheduling parameters from vLLM capacity and MoE layout."""

    max_seqs = max(1, int(getattr(scheduler_config, "max_num_seqs", 128)))
    offload_count = max(0, int(getattr(runtime_config, "offload_count", 0)))
    uses_cold_buffer = bool(getattr(
        runtime_config, "uses_cold_buffer", offload_count > 0))
    buffers = max(1, int(getattr(runtime_config, "num_buffers", 1)))
    layer_count = max(
        1, len(getattr(runtime_config, "runtime_layer_ids", []) or []))

    requested_policy = str(getattr(
        runtime_config, "scheduler_policy", "auto")).strip().lower()
    if requested_policy not in _VALID_POLICIES:
        requested_policy = "auto"
    policy = (("offload" if uses_cold_buffer else "expert")
              if requested_policy == "auto" else requested_policy)

    transfer_pressure = min(
        1.0, (offload_count * layer_count) / float(8 * buffers))
    explicit_window = getattr(runtime_config, "scheduler_reorder_window", None)
    reorder_window = (max(1, int(explicit_window))
                      if explicit_window is not None else
                      max(8, min(256, round(
                          max_seqs * (2.0 + transfer_pressure)))))

    explicit_decode = getattr(
        runtime_config, "scheduler_decode_reserve_ratio", None)
    decode_reserve = (float(explicit_decode) if explicit_decode is not None
                      else 0.45 - 0.15 * transfer_pressure)

    explicit_similarity = getattr(
        runtime_config, "scheduler_activation_similarity_threshold", None)
    similarity = (float(explicit_similarity)
                  if explicit_similarity is not None
                  else 0.80 - 0.10 * transfer_pressure)

    explicit_max_experts = getattr(
        runtime_config, "scheduler_max_profile_experts", None)
    max_profile_experts = (int(explicit_max_experts)
                           if explicit_max_experts is not None
                           else max(8, min(64, 8 + offload_count * 4)))

    return SchedulerParameters(
        policy=policy,
        reorder_window=reorder_window,
        decode_reserve_ratio=max(0.0, min(1.0, decode_reserve)),
        activation_similarity_threshold=max(0.0, min(1.0, similarity)),
        max_profile_experts=max(1, max_profile_experts),
        offload_count=offload_count,
        uses_cold_buffer=uses_cold_buffer,
    )


@dataclass(frozen=True)
class _RequestFeatures:
    request: Any
    index: int
    phase: RequestPhase
    tokens: int
    expert_cost: float | None
    activation: ActivationProfile | None


@dataclass
class _Cluster:
    phase: RequestPhase
    items: list[_RequestFeatures]
    centroid: ActivationProfile | None

    @property
    def tokens(self) -> int:
        return sum(item.tokens for item in self.items)


def reorder_requests(
    requests: list[Any],
    parameters: SchedulerParameters,
    *,
    waiting: bool,
) -> list[Any]:
    if parameters.policy == "fifo" or len(requests) <= 1:
        return list(requests)
    features = [_request_features(
        request, index, parameters, waiting=waiting)
                for index, request in enumerate(requests)]
    if parameters.policy == "throughput":
        features.sort(key=lambda item: (
            _phase_rank(item.phase), -item.tokens, item.index))
        return [item.request for item in features]
    if parameters.policy == "expert" and not any(
            item.activation is not None for item in features):
        features.sort(key=lambda item: (
            _phase_rank(item.phase),
            item.expert_cost is None,
            float("inf") if item.expert_cost is None else item.expert_cost,
            -item.tokens,
            item.index,
        ))
        return [item.request for item in features]
    return [item.request for item in _cluster_order(
        features, parameters, waiting=waiting)]


def _request_features(
    request: Any,
    index: int,
    parameters: SchedulerParameters,
    *,
    waiting: bool,
) -> _RequestFeatures:
    request_id = str(getattr(request, "request_id", index))
    phase = _detect_phase(request)
    return _RequestFeatures(
        request=request,
        index=index,
        phase=phase,
        tokens=_estimate_tokens(request, phase),
        expert_cost=_expert_cost(request),
        activation=_extract_activation_profile(
            request,
            request_id=request_id,
            max_experts=parameters.max_profile_experts,
        ),
    )


def _cluster_order(
    features: list[_RequestFeatures],
    parameters: SchedulerParameters,
    *,
    waiting: bool,
) -> list[_RequestFeatures]:
    clusters: list[_Cluster] = []
    for item in features:
        best: _Cluster | None = None
        best_similarity = -1.0
        if item.activation is not None:
            for cluster in clusters:
                if cluster.phase != item.phase or cluster.centroid is None:
                    continue
                similarity = item.activation.similarity(cluster.centroid)
                if (similarity >= parameters.activation_similarity_threshold
                        and similarity > best_similarity):
                    best = cluster
                    best_similarity = similarity
        if best is None:
            clusters.append(_Cluster(item.phase, [item], item.activation))
            continue
        best.items.append(item)
        best.centroid = best.centroid.merge(
            item.activation, max_experts=parameters.max_profile_experts)

    for cluster in clusters:
        if cluster.phase == RequestPhase.PREFILL:
            cluster.items.sort(key=lambda item: (-item.tokens, item.index))
        else:
            cluster.items.sort(key=lambda item: item.index)
    clusters.sort(key=lambda cluster: (
        _phase_rank(cluster.phase),
        cluster.centroid is None,
        -len(cluster.items) if cluster.phase == RequestPhase.DECODE else
        -cluster.tokens,
        min(item.index for item in cluster.items),
    ))
    if waiting:
        return [item for cluster in clusters for item in cluster.items]

    decode = [cluster for cluster in clusters
              if cluster.phase == RequestPhase.DECODE]
    prefills = [cluster for cluster in clusters
                if cluster.phase == RequestPhase.PREFILL]
    target = math.ceil(len(features) * parameters.decode_reserve_ratio)
    prefix: list[_Cluster] = []
    prefix_size = 0
    while decode and prefix_size < target:
        cluster = decode.pop(0)
        prefix.append(cluster)
        prefix_size += len(cluster.items)
    return [item for cluster in (*prefix, *prefills, *decode)
            for item in cluster.items]


def _detect_phase(request: Any) -> RequestPhase:
    computed = int(getattr(request, "num_computed_tokens", 0))
    prompt = int(getattr(request, "num_prompt_tokens", computed))
    return RequestPhase.PREFILL if computed < prompt else RequestPhase.DECODE


def _estimate_tokens(request: Any, phase: RequestPhase) -> int:
    computed = int(getattr(request, "num_computed_tokens", 0))
    if phase == RequestPhase.PREFILL:
        return max(0, int(getattr(request, "num_prompt_tokens", computed))
                   - computed)
    total = int(getattr(request, "num_tokens_with_spec", computed + 1))
    return max(1, total - computed)


def _extract_activation_profile(
    request: Any,
    *,
    request_id: str,
    max_experts: int,
) -> ActivationProfile | None:
    runtime_profile = ACTIVATION_REGISTRY.get(request_id)
    if runtime_profile is not None:
        return runtime_profile
    for source in _iter_sources(request):
        for name in _ACTIVATION_NAMES:
            value = (source.get(name) if isinstance(source, Mapping)
                     else getattr(source, name, None))
            if value is None:
                continue
            profile = ActivationProfile.from_value(
                value, max_experts=max_experts)
            if profile is not None:
                return profile
    return None


def _expert_cost(request: Any) -> float | None:
    for source in _iter_sources(request):
        for name in _COST_NAMES:
            value = (source.get(name) if isinstance(source, Mapping)
                     else getattr(source, name, None))
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _iter_sources(root: Any) -> Iterable[Any]:
    seen: set[int] = set()
    pending = [root]
    while pending:
        source = pending.pop(0)
        if source is None or id(source) in seen:
            continue
        seen.add(id(source))
        yield source
        for name in _SOURCE_NAMES:
            child = (source.get(name) if isinstance(source, Mapping)
                     else getattr(source, name, None))
            if child is not None:
                pending.append(child)


def _collect_activation(
    value: Any,
    counts: dict[str, float],
    prefix: str = "",
) -> None:
    if isinstance(value, Mapping):
        for key, weight in value.items():
            item_key = f"{prefix}{key}"
            if isinstance(weight, (Mapping, list, tuple, set)):
                _collect_activation(weight, counts, prefix=f"{item_key}/")
            else:
                try:
                    counts[item_key] += float(weight)
                except (TypeError, ValueError):
                    pass
        return
    if isinstance(value, (list, tuple, set)):
        for expert_id in value:
            if isinstance(expert_id, (Mapping, list, tuple, set)):
                _collect_activation(expert_id, counts, prefix=prefix)
            else:
                counts[f"{prefix}{expert_id}"] += 1.0
        return
    if value is not None:
        counts[f"{prefix}{value}"] += 1.0


def _phase_rank(phase: RequestPhase) -> int:
    return 0 if phase == RequestPhase.DECODE else 1
