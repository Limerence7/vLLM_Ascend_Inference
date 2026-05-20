import re
from types import MethodType
from typing import Dict, Optional, Set

import torch

from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE

from ..offload.config import get_offload_config
from ..offload.expert_wise_memory import ExpertWiseExpertStore
from .compact_dispatch import CompactAllGatherDispatchPatch


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


class ExpertWiseAscendFusedMoE(AscendFusedMoE):
    """
    Ascend FusedMoE wrapper for expert-wise offload.

    The wrapper owns routing integration and delegates CPU storage / NPU cache
    bookkeeping to ExpertWiseExpertStore.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runtime_offload_config = get_offload_config()
        self.expert_wise_config = self.runtime_offload_config.expert_wise
        self.decoder_layer_idx = parse_decoder_layer_idx(self.layer_name)
        self.expert_store = ExpertWiseExpertStore(
            layer_idx=self.decoder_layer_idx,
            config=self.expert_wise_config,
        )
        self.expert_wise_enabled = (
            self.runtime_offload_config.mode == "expert_wise"
            and self.decoder_layer_idx is not None
        )

        if self.expert_wise_enabled:
            self._install_weight_postprocess_hook()

    def _install_weight_postprocess_hook(self) -> None:
        original = self.quant_method.process_weights_after_loading

        def wrapped_process_weights_after_loading(quant_method, layer):
            original(layer)
            layer.init_expert_wise_cpu_store()

        self.quant_method.process_weights_after_loading = MethodType(
            wrapped_process_weights_after_loading,
            self.quant_method,
        )

    def init_expert_wise_cpu_store(self) -> None:
        selected_experts = self._selected_global_expert_ids()
        if not selected_experts:
            return

        self.expert_store.init_cpu_store(
            fused_moe=self,
            selected_global_expert_ids=selected_experts,
            local_expert_id_fn=self._local_expert_id,
        )
        if self.expert_store.enabled:
            self.expert_store.maybe_compact_npu_weights(self)
            print(
                "[Plugin] Expert-wise CPU store initialized: "
                f"layer={self.decoder_layer_idx}, "
                f"experts={sorted(self.expert_store.offloaded_expert_ids())}"
            )

    def _selected_global_expert_ids(self) -> Set[int]:
        selection = self.expert_wise_config.offloaded_experts
        if self.decoder_layer_idx is None or not selection:
            return set()

        if self.decoder_layer_idx not in selection:
            return set()

        selected = selection[self.decoder_layer_idx]
        if selected is None:
            return set(range(self._logical_num_experts()))

        return set(selected)

    def _logical_num_experts(self) -> int:
        return int(
            getattr(
                self.moe_config,
                "original_num_experts",
                getattr(self, "logical_num_experts", self.global_num_experts),
            )
        )

    def _local_expert_id(self, global_expert_id: int) -> Optional[int]:
        if global_expert_id < 0 or global_expert_id >= self.global_num_experts:
            return None

        local_expert_id = self._map_global_expert_id_to_local_expert_id(
            global_expert_id
        )
        if local_expert_id == -1:
            return None

        if local_expert_id >= self.w13_weight.shape[0]:
            return None

        return int(local_expert_id)

    def expert_wise_summary(self) -> Dict[str, object]:
        summary = self.expert_store.summary()
        summary["enabled"] = self.expert_wise_enabled
        summary["layer_name"] = self.layer_name
        return summary

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if self.expert_wise_enabled and self.expert_store.enabled:
            routed_experts = self._routed_offloaded_experts(router_logits)
            self.expert_store.restore_routed_experts(self, routed_experts)
            try:
                if self.expert_store.is_compact():
                    with CompactAllGatherDispatchPatch(
                        num_compact_slots=int(self.w13_weight.shape[0]),
                    ):
                        return super().forward_impl(hidden_states, router_logits)
                return super().forward_impl(hidden_states, router_logits)
            finally:
                self.expert_store.mark_loaded_experts_evicted_if_needed(self)

        return super().forward_impl(hidden_states, router_logits)

    def _routed_offloaded_experts(self, router_logits: torch.Tensor) -> Set[int]:
        _, topk_ids = select_experts(
            hidden_states=torch.empty(
                router_logits.shape[0],
                self.hidden_size,
                device=router_logits.device,
                dtype=router_logits.dtype,
            ),
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            global_num_experts=self.global_num_experts,
        )
        routed_experts = {
            int(expert_id)
            for expert_id in topk_ids.detach().cpu().flatten()
        }
        return routed_experts.intersection(self.expert_store.offloaded_expert_ids())


def parse_decoder_layer_idx(layer_name: str) -> Optional[int]:
    match = _LAYER_INDEX_RE.search(layer_name)
    return None if match is None else int(match.group(1))
