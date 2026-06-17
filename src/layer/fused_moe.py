import os.path
from typing import Callable

import torch
import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group, get_pcp_group,
                              get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, determine_expert_map, get_compressed_expert_map,
    maybe_roundup_hidden_size)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import determine_default_log2phy_map
from vllm_ascend.eplb.utils import moe_load_async_stream
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.ops.fused_moe.prepare_finalize import QuantType
from vllm_ascend.quantization.w4a8_dynamic import \
    AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.w8a8_dynamic import \
    AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.utils import npu_stream_switch

from ..offload_config import get_offload_config
from .quant_method import (OffloadUnquantizedFusedMoEMethod,
                           OffloadW8A8DynamicFusedMoEMethod,
                           wrap_quant_method)
from .routing import map_expert_ids


EXPERT_WEIGHT_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w13_weight_offset",
    "w2_weight_scale",
    "w2_weight_offset",
)


class OffloadAscendFusedMoE(FusedMoE):
    moe_counter = -1
    gate_stream: torch.npu.Stream | None = None
    executor = None

    @classmethod
    def reset_runtime(cls) -> None:
        cls.moe_counter = -1
        cls.gate_stream = None
        cls.executor = None

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config=None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        pcp_size: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        is_act_and_mul: bool = True,
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: int | None = 0,
        zero_expert_type: str | None = None,
        expert_mapping=None,
        n_shared_experts: int | None = None,
        routing_method_type: int | None = None,
    ):
        CustomOp.__init__(self)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        moe_in_dtype = (vllm_config.model_config.dtype
                        if vllm_config.model_config is not None else
                        params_dtype)

        tp_size_ = (tp_size if tp_size is not None else
                    get_tensor_model_parallel_world_size())
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
        pcp_size_ = (pcp_size if pcp_size is not None else
                     get_pcp_group().world_size)

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1
        self.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            pcp_size_=pcp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        self.logical_num_experts = num_experts
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type
        self.expert_mapping = expert_mapping
        self.enable_eplb = enable_eplb
        self.expert_load_view = None
        self.logical_to_physical_map = None
        self.logical_replica_count = None
        self.expert_placement_strategy = (
            vllm_config.parallel_config.expert_placement_strategy)
        self.shared_experts_stream = None
        self.rocm_aiter_fmoe_enabled = False
        self.aiter_fmoe_shared_expert_enabled = False
        self.num_fused_shared_experts = 0

        hidden_size = maybe_roundup_hidden_size(
            hidden_size,
            moe_in_dtype,
            quant_config,
            self.moe_parallel_config,
            is_lora_enabled=vllm_config.lora_config is not None,
        )
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.top_k = top_k
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation
        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for non-grouped topk."
            )
        if routing_method_type is not None:
            self.routing_method_type = routing_method_type
        elif scoring_func == "sigmoid":
            self.routing_method_type = (
                RoutingMethodType.DeepSeekV3 if self.use_grouped_topk else
                RoutingMethodType.Llama4
                if self.top_k == 1 else RoutingMethodType.TopK)
        elif scoring_func == "softmax":
            self.routing_method_type = (
                RoutingMethodType.Renormalize if not self.renormalize else
                RoutingMethodType.RenormalizeNaive)
        else:
            self.routing_method_type = RoutingMethodType.TopK

        self.moe_config = FusedMoEConfig(
            num_experts=num_experts + num_redundant_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=0,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
            is_act_and_mul=is_act_and_mul,
            is_lora_enabled=vllm_config.lora_config is not None,
        )
        self.moe_config_use_flashinfer_cutlass_kernels = (
            self.moe_config.use_flashinfer_cutlass_kernels)
        self.quant_config = quant_config
        self.batched_hidden_states = None
        self.batched_router_logits = None

        OffloadAscendFusedMoE.moe_counter += 1
        self.moe_instance_id = OffloadAscendFusedMoE.moe_counter

        self._expert_map = None
        self.full_expert_map = None
        self.full_local_num_experts = 0
        self.resident_local_num_experts = 0
        self._resident_maps_by_device: dict[torch.device, torch.Tensor] = {}
        self.log2phy = None

        self.quant_method = (
            OffloadUnquantizedFusedMoEMethod(self.moe_config)
            if self.quant_config is None else wrap_quant_method(
                self.quant_config.get_quant_method(self, self.layer_name)))

        assert self.quant_method is not None
        self.uses_w8a8 = isinstance(self.quant_method,
                                    OffloadW8A8DynamicFusedMoEMethod)
        self.offload_config = get_offload_config()
        if OffloadAscendFusedMoE.executor is None:
            from ..offload.executor import OffloadExecutor

            OffloadAscendFusedMoE.executor = OffloadExecutor(
                self.offload_config, self.uses_w8a8)
        self.offload_executor = OffloadAscendFusedMoE.executor

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        ascend_config = get_ascend_config()
        self.dynamic_eplb = (
            ascend_config.dynamic_eplb
            or ascend_config.expert_map_record_path)
        self.expert_map_path = ascend_config.expert_map_path
        self.global_redundant_expert_num = ascend_config.init_redundancy_expert
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        self.multistream_overlap_gate = False
        if (self.custom_routing_function is None
                and self.e_score_correction_bias is not None):
            vllm_config = get_current_vllm_config()
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype)

        self.local_num_experts, self._expert_map, _ = determine_expert_map(
            self.ep_size, self.ep_rank, self.global_num_experts)
        init_eplb_enable = False
        if self.expert_map_path and os.path.exists(
                self.expert_map_path) and os.access(self.expert_map_path,
                                                    os.R_OK):
            self.expert_load_balancer = ExpertLoadBalancer(
                self.expert_map_path, num_experts)
            self.expert_load_balancer.check_expert_map_tensor()
            self.global_redundant_expert_num = (
                self.expert_load_balancer.get_global_redundant_expert_num())
            self.global_num_experts = num_experts + self.global_redundant_expert_num
            try:
                self.local_num_experts, self._expert_map = (
                    self.expert_load_balancer.get_rank_placement_map(
                    self.moe_instance_id, self.ep_rank))
                self.log2phy = self.expert_load_balancer.get_rank_log2phy_map(
                    self.moe_instance_id, self.ep_rank).npu()
                init_eplb_enable = True
            except Exception as e:
                logger.warning(
                    f"Init expert map of mtp/eagle when using sample.{e}")
                self.log2phy = determine_default_log2phy_map(
                    self.global_num_experts, self.ep_size, self.ep_rank).npu()
        elif self.dynamic_eplb:
            self.log2phy = determine_default_log2phy_map(
                self.global_num_experts, self.ep_size, self.ep_rank).npu()
        self.full_expert_map = self._expert_map
        self.full_local_num_experts = int(
            torch.sum(self.full_expert_map != -1).item()
            if self.full_expert_map is not None else self.global_num_experts)
        self.resident_local_num_experts = self._resident_local_num_experts()
        self.expert_placement = self.offload_executor.init_layer_placement(
            self)
        self._expert_map = self._build_resident_expert_map()
        self.local_num_experts = self.resident_local_num_experts

        if self.full_expert_map is not None:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.", self.ep_rank, self.ep_size, self.full_local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self.full_expert_map))
        if self.dynamic_eplb:
            self.moe_load = torch.zeros(self.resident_local_num_experts,
                                        dtype=torch.int64).npu()
        else:
            self.moe_load = None

        if init_eplb_enable and not self.uses_w8a8:
            raise ValueError("Eplb supports only w8a8_dynamic quantization.")

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.resident_local_num_experts
        self.moe_config.original_num_experts = num_experts

        moe_quant_params = {
            "num_experts": self.resident_local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size
        self.quant_method.create_weights(layer=self, **moe_quant_params)
        self.offload_executor.register_layer(self)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        self.moe_config.num_local_experts = max(
            1, self.resident_local_num_experts)
        setup_moe_comm_method(self.moe_config)
        self.moe_config.num_local_experts = self.resident_local_num_experts
        self.quant_type = self._get_quant_type()

    def _resident_local_num_experts(self) -> int:
        if not self.offload_executor.should_offload_layer(self):
            return self.full_local_num_experts
        if self.offload_executor.offload_full_layer:
            return 0
        return min(int(self.offload_config.num_hot_experts),
                   self.full_local_num_experts)

    def _build_resident_expert_map(self) -> torch.Tensor | None:
        resident_ids = self.expert_placement.resident_expert_ids
        if (self.resident_local_num_experts == self.full_local_num_experts
                and resident_ids == list(range(self.full_local_num_experts))):
            return self.full_expert_map

        expert_map = torch.full((self.global_num_experts, ),
                                -1,
                                dtype=torch.int32)
        resident_slots = self.expert_placement.resident_slots
        if self.full_expert_map is None:
            for local_id, slot in resident_slots.items():
                expert_map[local_id] = slot
            return expert_map

        full_map = self.full_expert_map.detach().cpu()
        for global_id, local_id in enumerate(full_map.tolist()):
            slot = resident_slots.get(int(local_id))
            if slot is not None:
                expert_map[global_id] = slot
        return expert_map

    def map_global_expert_id_to_full_local_expert_id(
            self, expert_id: int) -> int:
        if self.full_expert_map is None:
            return expert_id
        return self.full_expert_map[expert_id].item()

    def build_hot_local_routing(self, topk_ids: torch.Tensor,
                                cold_mask: torch.Tensor) -> tuple[
                                    torch.Tensor, torch.Tensor]:
        if self.resident_local_num_experts <= 0:
            return torch.zeros_like(topk_ids), torch.zeros_like(
                cold_mask, dtype=torch.bool)

        expert_map = self._resident_maps_by_device.get(topk_ids.device)
        if expert_map is None:
            if self._expert_map is None:
                expert_map = torch.arange(self.global_num_experts,
                                          dtype=torch.long)
            else:
                expert_map = self._expert_map.detach().to(dtype=torch.long)
            expert_map = expert_map.to(device=topk_ids.device,
                                       non_blocking=True)
            self._resident_maps_by_device[topk_ids.device] = expert_map

        local_ids, is_local = map_expert_ids(topk_ids, expert_map)
        hot_mask = is_local & ~cold_mask
        hot_ids = local_ids.masked_fill(~hot_mask, 0)
        return hot_ids.to(topk_ids.dtype), hot_mask

    def _get_quant_type(self) -> QuantType:
        method = getattr(self.quant_method, "quant_method", None)
        if method is None:
            return QuantType.NONE

        if isinstance(method, AscendW8A8DynamicFusedMoEMethod):
            return QuantType.W8A8
        if isinstance(method, AscendW4A8DynamicFusedMoEMethod):
            return QuantType.W4A8
        return QuantType.NONE

    def update_expert_map(self, new_expert_map):
        self._expert_map = new_expert_map

    def apply_expert_placement(self, placement) -> None:
        self.expert_placement = placement
        self._expert_map = self._build_resident_expert_map()
        self._resident_maps_by_device.clear()

    def get_log2phy_map(self):
        return self.log2phy

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()

    def _record_moe_load(self, group_list_type: int,
                         expert_tokens: torch.Tensor) -> None:
        moe_load_stream = moe_load_async_stream()
        current_stream = torch.npu.current_stream()

        moe_load_stream.wait_stream(current_stream)
        with npu_stream_switch(moe_load_stream):
            if group_list_type != 1:
                expert_tokens = torch.cat([
                    expert_tokens[:1],
                    expert_tokens[1:] - expert_tokens[:-1],
                ])
            self.moe_load += expert_tokens
        current_stream.wait_stream(moe_load_stream)

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        """NOTE(Yizhou): This is to override the parent class method. In `mc2commimpl`,
        and `alltoallcommimpl`, we do not need to all-reduce the final outputs since
        the outputs are already aggregated across tensor parallel ranks in the
        `finalize` function. In `allgathercommimpl`, we still need to all-reduce the
        outputs since each rank only has partial outputs.
        """
        return torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int,
                      return_success: bool = False) -> bool | None:
        param_name = next((name for name in EXPERT_WEIGHT_NAMES
                           if getattr(self, name, None) is param), None)
        load_offloaded_weights = self.offload_executor.load_weight(
            self, param_name or weight_name, shard_id, expert_id,
            loaded_weight)
        if load_offloaded_weights:
            return True if return_success else None

        return super().weight_loader(param, loaded_weight, weight_name,
                                     shard_id, expert_id, return_success)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        forward_context = get_forward_context()
        enable_force_load_balance = forward_context.in_profile_run

        hidden_states, router_logits, mc2_mask, context_metadata = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=forward_context.sp_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type)

        if isinstance(hidden_states, tuple):
            hidden_states, pertoken_scale = hidden_states
        else:
            pertoken_scale = None

        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            pertoken_scale=pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            quantized_x_for_share=None,
            dynamic_scale_for_share=None,
            shared_experts=None,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask)

        if isinstance(final_hidden_states, tuple):
            final_hidden_states, group_list_type, expert_tokens = final_hidden_states
            if self.dynamic_eplb:
                self._record_moe_load(group_list_type, expert_tokens)

        final_hidden_states = forward_context.moe_comm_method.finalize(
            hidden_states=final_hidden_states,
            reduce_results=self.reduce_results,
            context_metadata=context_metadata)

        return final_hidden_states
