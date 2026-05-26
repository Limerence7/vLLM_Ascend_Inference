from __future__ import annotations

from typing import Dict, Optional, Set

from ..config import OffloadConfig, get_offload_config

from .predictor import SameExpertNextLayerPredictor
from .scheduler import ExpertWisePlan, ExpertWiseScheduler


class ExpertWiseManager:
    _active: Optional["ExpertWiseManager"] = None

    def __init__(self, config: OffloadConfig):
        self.config = config
        self.scheduler = ExpertWiseScheduler(config)
        self.predictor = SameExpertNextLayerPredictor()
        self.layers: Dict[int, object] = {}

    @classmethod
    def activate(cls, config: Optional[OffloadConfig] = None) -> "ExpertWiseManager":
        cls._active = cls(config or get_offload_config())
        return cls._active

    @classmethod
    def active(cls) -> "ExpertWiseManager":
        if cls._active is None:
            cls.activate()
        return cls._active

    def register_layer(self, layer_idx: Optional[int], fused_moe: object) -> None:
        if layer_idx is not None:
            self.layers[layer_idx] = fused_moe

    def record_activated(
        self,
        layer_idx: int,
        expert_ids: Set[int],
        target_layer_idx: Optional[int] = None,
    ) -> None:
        if self.config.expert_wise.enable_prediction:
            self.predictor.record_activated(
                layer_idx,
                expert_ids,
                target_layer_idx=target_layer_idx,
            )

    def build_plan(
        self,
        *,
        routed_experts: Set[int],
        current_offloaded_experts: Set[int],
    ) -> ExpertWisePlan:
        return self.scheduler.build_plan(
            routed_experts=routed_experts,
            current_offloaded_experts=current_offloaded_experts,
        )

    def schedule_next_prefetch(
        self,
        *,
        layer_idx: Optional[int],
        activated_experts: Set[int],
        plan: ExpertWisePlan,
    ) -> None:
        if layer_idx is None:
            return

        if not self.config.overlap or self.config.prefetch_distance <= 0:
            return

        next_layer_idx = self.scheduler.next_prefetch_layer(layer_idx)
        if next_layer_idx is None:
            return

        self.record_activated(
            layer_idx,
            activated_experts,
            target_layer_idx=next_layer_idx,
        )
        if plan.prefetch_all_next:
            self.prefetch_layer(next_layer_idx, all_offloaded=True)
            return

        predicted = plan.prefetch_next_experts
        if not predicted and self.config.expert_wise.enable_prediction:
            predicted = self.predictor.predicted_for(next_layer_idx)
        self.prefetch_layer(next_layer_idx, expert_ids=predicted)

    def prefetch_layer(
        self,
        layer_idx: int,
        expert_ids: Optional[Set[int]] = None,
        all_offloaded: bool = False,
    ) -> None:
        layer = self.layers.get(layer_idx)
        if layer is None:
            return

        if all_offloaded and hasattr(layer, "prefetch_all_offloaded_experts"):
            layer.prefetch_all_offloaded_experts()
            return

        if expert_ids and hasattr(layer, "prefetch_experts"):
            layer.prefetch_experts(expert_ids)

    def summary(self) -> Dict[int, object]:
        summaries = {}
        for layer_idx, layer in sorted(self.layers.items()):
            if hasattr(layer, "expert_wise_summary"):
                summaries[layer_idx] = layer.expert_wise_summary()
        return {
            "mode": "expert_wise",
            "layers": summaries,
            "total_cpu_store_bytes": sum(
                item.get("cpu_store_bytes", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_copy_count": sum(
                item.get("copy_count", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_prefetch_count": sum(
                item.get("prefetch_count", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_prefetch_wait_count": sum(
                item.get("prefetch_wait_count", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_prefetch_capacity_skips": sum(
                item.get("prefetch_capacity_skips", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_chunked_compact_forward_count": sum(
                item.get("chunked_compact_forward_count", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_chunked_compact_piece_count": sum(
                item.get("chunked_compact_piece_count", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_chunked_compact_token_count": sum(
                item.get("chunked_compact_token_count", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_chunked_compact_full_token_count": sum(
                item.get("chunked_compact_full_token_count", 0)
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_no_shrink_skipped_layers": sum(
                item.get("skip_reason") == "no_shrink"
                for item in summaries.values()
                if isinstance(item, dict)
            ),
            "total_capacity_unsafe_skipped_layers": sum(
                item.get("skip_reason") == "capacity_unsafe"
                for item in summaries.values()
                if isinstance(item, dict)
            ),
        }
