import re
from types import MethodType
from typing import Dict, Optional, Set

import torch

from vllm.forward_context import get_forward_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE

from ..config import get_offload_config
from ..expert_wise import (
    ExpertWiseExpertStore,
    ExpertWiseManager,
    ExpertWisePrefetcher,
    ExpertWiseScheduler,
)

from .compact_dispatch import (
    activate_compact_allgather_dispatch,
    deactivate_compact_allgather_dispatch,
)
from .expert_loader import ExpertLoader
from .overlap_executor import build_executor


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_DENSE_RESTORE_WARNED = False
_NO_SHRINK_SKIP_WARNED = False
_CAPACITY_UNSAFE_SKIP_WARNED = False


class ExpertWiseAscendFusedMoE(AscendFusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runtime_offload_config = get_offload_config()
        self.expert_wise_config = self.runtime_offload_config.expert_wise
        self.decoder_layer_idx = parse_decoder_layer_idx(self.layer_name)
        self.scheduler = ExpertWiseScheduler(self.runtime_offload_config)
        self.expert_store = ExpertWiseExpertStore(
            layer_idx=self.decoder_layer_idx,
            config=self.expert_wise_config,
        )
        self.expert_loader = ExpertLoader(self.expert_store)
        self.overlap_executor = build_executor(self.runtime_offload_config.overlap)
        self.expert_manager = ExpertWiseManager.active()
        self.prefetcher = ExpertWisePrefetcher(self.expert_manager)
        self.routing_select_count = 0
        self.routing_large_batch_fast_path_count = 0
        self._expert_wise_single_select_active = False
        self.expert_wise_skip_reason = None
        self.expert_manager.register_layer(self.decoder_layer_idx, self)

        self.expert_wise_enabled = (
            self.runtime_offload_config.mode == "expert_wise"
            and self.scheduler.should_offload_layer(self.decoder_layer_idx)
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
        if self._should_skip_no_shrink_expert_wise(selected_experts):
            return

        self.expert_store.init_cpu_store(
            fused_moe=self,
            selected_global_expert_ids=selected_experts,
            local_expert_id_fn=self._local_expert_id,
        )
        if self.expert_store.enabled:
            self.expert_store.maybe_compact_npu_weights(self)
            global _DENSE_RESTORE_WARNED
            if not self.expert_store.is_compact() and not _DENSE_RESTORE_WARNED:
                _DENSE_RESTORE_WARNED = True
                print(
                    "[Plugin] Expert-wise dense restore mode initialized; "
                    "this validates load/barrier/compute logic but does not "
                    "shrink resident NPU expert weight tensors. Enable "
                    "compact_npu_cache for actual expert-weight memory savings."
                )
            print(
                "[Plugin] Expert-wise CPU store initialized: "
                f"layer={self.decoder_layer_idx}, "
                f"resident_experts={self.expert_wise_config.resident_experts}, "
                f"offloaded_experts={sorted(self.expert_store.offloaded_expert_ids())}"
            )

    def _should_skip_no_shrink_expert_wise(self, selected_experts: Set[int]) -> bool:
        config = self.expert_wise_config
        if not (
            config.compact_npu_cache
            and config.skip_no_shrink_compact
            and self.expert_wise_enabled
        ):
            return False
        if self.expert_map is None:
            return False

        local_global_to_slot = ExpertWiseExpertStore._local_global_to_slot(
            self.expert_map
        )
        if not local_global_to_slot:
            return False

        local_offloaded_experts = selected_experts.intersection(local_global_to_slot)
        if not local_offloaded_experts:
            return False

        resident_local_count = len(local_global_to_slot) - len(local_offloaded_experts)
        if len(local_offloaded_experts) > config.npu_cache_capacity:
            self.expert_wise_enabled = False
            self.expert_wise_skip_reason = "capacity_unsafe"
            global _CAPACITY_UNSAFE_SKIP_WARNED
            if not _CAPACITY_UNSAFE_SKIP_WARNED:
                _CAPACITY_UNSAFE_SKIP_WARNED = True
                print(
                    "[Plugin] Expert-wise compact offload skipped because "
                    "offloaded local experts exceed cache capacity: "
                    f"resident_local={resident_local_count}, "
                    f"offloaded_local={len(local_offloaded_experts)}, "
                    f"cache_capacity={config.npu_cache_capacity}, "
                    f"original_local_slots={len(local_global_to_slot)}. "
                    "Increase NPU_CACHE_CAPACITY or reduce offloaded experts "
                    "to force compact offload safely."
                )
            return True

        compact_slots = resident_local_count + config.npu_cache_capacity
        if compact_slots < len(local_global_to_slot):
            return False

        self.expert_wise_enabled = False
        self.expert_wise_skip_reason = "no_shrink"
        global _NO_SHRINK_SKIP_WARNED
        if not _NO_SHRINK_SKIP_WARNED:
            _NO_SHRINK_SKIP_WARNED = True
            print(
                "[Plugin] Expert-wise compact offload skipped on no-shrink "
                "rank/layers: "
                f"resident_local={resident_local_count}, "
                f"offloaded_local={len(local_offloaded_experts)}, "
                f"cache_capacity={config.npu_cache_capacity}, "
                f"original_local_slots={len(local_global_to_slot)}. "
                "Disable SKIP_NO_SHRINK_COMPACT to force compact rebuild."
            )
        return True

    def _selected_global_expert_ids(self) -> Set[int]:
        if not self.expert_wise_enabled:
            return set()
        if self.expert_wise_config.partition_scope == "local_rank":
            if self.expert_map is None:
                return set()
            return self.scheduler.offloaded_local_experts(
                local_global_to_slot=ExpertWiseExpertStore._local_global_to_slot(
                    self.expert_map,
                ),
                total_experts=self._logical_num_experts(),
            )
        return self.scheduler.offloaded_experts(self._logical_num_experts())

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
        summary["executor"] = type(self.overlap_executor).__name__
        summary["resident_overlap"] = self.overlap_executor.supports_resident_overlap
        summary["routing_select_count"] = self.routing_select_count
        summary["routing_large_batch_fast_path_count"] = (
            self.routing_large_batch_fast_path_count
        )
        summary["skip_reason"] = self.expert_wise_skip_reason
        summary["partition_scope"] = self.expert_wise_config.partition_scope
        return summary

    def prefetch_all_offloaded_experts(self) -> None:
        self.prefetch_experts(self.expert_store.offloaded_expert_ids())

    def prefetch_experts(self, predicted_global_expert_ids: Set[int]) -> None:
        if not self.expert_wise_enabled or not self.expert_store.enabled:
            return

        offloaded = predicted_global_expert_ids.intersection(
            self.expert_store.offloaded_expert_ids()
        )
        if offloaded:
            self.expert_loader.prefetch(self, offloaded)

    def forward(self, *args, **kwargs):
        if self.expert_wise_enabled and self.expert_store.enabled:
            hidden_states, router_logits = self._extract_forward_inputs(args, kwargs)
            current_offloaded_experts = self.expert_store.offloaded_expert_ids()
            can_use_large_batch_fast_path = self._can_use_large_batch_fast_path(
                router_logits=router_logits,
                current_offloaded_experts=current_offloaded_experts,
            )
            if (
                not can_use_large_batch_fast_path
                and self._can_use_single_select_forward()
            ):
                try:
                    self._expert_wise_single_select_active = True
                    return super(ExpertWiseAscendFusedMoE, self).forward(
                        *args,
                        **kwargs,
                    )
                finally:
                    self._expert_wise_single_select_active = False
                    self.expert_store.mark_loaded_experts_evicted_if_needed(self)

            routed_experts, plan = self._build_expert_plan(
                router_logits=router_logits,
                current_offloaded_experts=current_offloaded_experts,
                can_use_large_batch_fast_path=can_use_large_batch_fast_path,
            )

            try:
                if not self.overlap_executor.supports_resident_overlap:
                    self.expert_loader.load_for_compute(self, plan.load_experts)
                    return self._forward_after_expert_load(
                        args=args,
                        kwargs=kwargs,
                        routed_experts=routed_experts,
                        plan=plan,
                    )

                def load_fn() -> None:
                    self.expert_loader.load_for_compute(self, plan.load_experts)

                def compute_fn():
                    return self._forward_after_expert_load(
                        args=args,
                        kwargs=kwargs,
                        routed_experts=routed_experts,
                        plan=plan,
                    )

                return self.overlap_executor.run(load_fn, compute_fn)
            finally:
                self.expert_store.mark_loaded_experts_evicted_if_needed(self)

        return super().forward(*args, **kwargs)

    def _can_use_single_select_forward(self) -> bool:
        config = self.expert_wise_config
        if not config.enable_single_select_forward:
            return False
        if self.expert_store.is_compact():
            offloaded_count = len(self.expert_store.offloaded_expert_ids())
            if offloaded_count > config.npu_cache_capacity:
                return False
        if self.multistream_overlap_gate:
            return False
        if self.dynamic_eplb:
            return False
        if self.quant_type.name != "NONE":
            return False
        return True

    def _forward_after_expert_load(self, *, args, kwargs, routed_experts, plan):
        self.prefetcher.schedule_next(
            self.decoder_layer_idx,
            routed_experts,
            plan,
        )
        if self.expert_store.is_compact():
            dispatcher = activate_compact_allgather_dispatch(
                num_compact_slots=int(self.w13_weight.shape[0]),
            )
            try:
                return super(ExpertWiseAscendFusedMoE, self).forward(*args, **kwargs)
            finally:
                deactivate_compact_allgather_dispatch(dispatcher)
        return super(ExpertWiseAscendFusedMoE, self).forward(*args, **kwargs)

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        if self._expert_wise_single_select_active:
            return self._forward_impl_single_select(hidden_states, router_logits)
        return super().forward_impl(hidden_states, router_logits)

    def _forward_impl_single_select(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        assert self.quant_method is not None

        forward_context = get_forward_context()
        if forward_context.in_profile_run:
            return super().forward_impl(hidden_states, router_logits)

        hidden_states, router_logits, mc2_mask, context_metadata = (
            forward_context.moe_comm_method.prepare(
                hidden_states=hidden_states,
                router_logits=router_logits,
                replace_allreduce=forward_context.sp_enabled,
                enable_shared_expert_dp=self.enable_shared_expert_dp,
                quant_type=self.quant_type,
            )
        )

        if isinstance(hidden_states, tuple):
            hidden_states, pertoken_scale = hidden_states
        else:
            pertoken_scale = None

        self.routing_select_count += 1
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
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
        topk_weights = topk_weights.to(hidden_states.dtype)

        routed_experts = {
            int(expert_id) for expert_id in topk_ids.detach().cpu().flatten()
        }
        current_offloaded_experts = self.expert_store.offloaded_expert_ids()
        plan = self.expert_manager.build_plan(
            routed_experts=routed_experts,
            current_offloaded_experts=current_offloaded_experts,
        )

        self.expert_loader.load_for_compute(self, plan.load_experts)
        self.prefetcher.schedule_next(
            self.decoder_layer_idx,
            routed_experts,
            plan,
        )

        dispatcher = None
        if self.expert_store.is_compact():
            dispatcher = activate_compact_allgather_dispatch(
                num_compact_slots=int(self.w13_weight.shape[0]),
            )
        try:
            final_hidden_states = get_forward_context().moe_comm_method.fused_experts(
                hidden_states=hidden_states,
                w1=self.w13_weight,
                w2=self.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                shared_experts=None,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                dynamic_eplb=False,
                mc2_mask=mc2_mask,
                pertoken_scale=pertoken_scale,
            )
        finally:
            deactivate_compact_allgather_dispatch(dispatcher)

        return forward_context.moe_comm_method.finalize(
            hidden_states=final_hidden_states,
            reduce_results=self.reduce_results,
            context_metadata=context_metadata,
        )

    def _build_expert_plan(
        self,
        *,
        router_logits: torch.Tensor,
        current_offloaded_experts: Set[int],
        can_use_large_batch_fast_path: Optional[bool] = None,
    ):
        if can_use_large_batch_fast_path is None:
            can_use_large_batch_fast_path = self._can_use_large_batch_fast_path(
                router_logits=router_logits,
                current_offloaded_experts=current_offloaded_experts,
            )
        if can_use_large_batch_fast_path:
            self.routing_large_batch_fast_path_count += 1
            routed_experts = set(current_offloaded_experts)
            plan = self.scheduler.build_large_batch_plan(
                current_offloaded_experts=current_offloaded_experts,
            )
            return routed_experts, plan

        self.routing_select_count += 1
        routed_experts = self._routed_global_experts(router_logits)
        plan = self.expert_manager.build_plan(
            routed_experts=routed_experts,
            current_offloaded_experts=current_offloaded_experts,
        )
        return routed_experts, plan

    def _can_use_large_batch_fast_path(
        self,
        *,
        router_logits: torch.Tensor,
        current_offloaded_experts: Set[int],
    ) -> bool:
        offloaded_count = len(current_offloaded_experts)
        if not self.scheduler.should_use_large_batch_plan(
            num_tokens=int(router_logits.shape[0]),
            offloaded_count=offloaded_count,
        ):
            return False

        if (
            self.expert_store.is_compact()
            and offloaded_count > self.expert_wise_config.npu_cache_capacity
        ):
            return False

        return True

    @staticmethod
    def _extract_forward_inputs(args, kwargs):
        if len(args) >= 2:
            return args[0], args[1]
        if "hidden_states" in kwargs and "router_logits" in kwargs:
            return kwargs["hidden_states"], kwargs["router_logits"]
        raise RuntimeError(
            "Expert-wise FusedMoE.forward expects hidden_states and router_logits."
        )

    def _routed_global_experts(self, router_logits: torch.Tensor) -> Set[int]:
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
        return {int(expert_id) for expert_id in topk_ids.detach().cpu().flatten()}


def parse_decoder_layer_idx(layer_name: str) -> Optional[int]:
    match = _LAYER_INDEX_RE.search(layer_name)
    return None if match is None else int(match.group(1))
