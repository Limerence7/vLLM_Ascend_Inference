from typing import Any, Callable

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.quantization.quant_config import AscendFusedMoEMethod
from vllm_ascend.quantization.w8a8_dynamic import \
    AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.utils import (AscendDeviceType, get_ascend_device_type,
                               maybe_trans_nz)

from .routing import (add_routing_output, build_routing_view,
                      select_optional_rows)
from .token_dispatcher import dispatch_with_local_experts


class RuntimeFusedMoEMethod:
    """Runtime routing flow for supported MoE weight formats."""

    dynamic_eplb: bool

    def _fused_experts(self, layer, moe_comm_method,
                       experts: torch.nn.Module, hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                       global_num_experts: int,
                       expert_map: torch.Tensor | None,
                       log2phy: torch.Tensor | None,
                       global_redundant_expert_num: int,
                       shared_experts: Any | None,
                       apply_router_weight_on_input: bool,
                       dynamic_eplb: bool,
                       mc2_mask: torch.Tensor | None,
                       pertoken_scale: torch.Tensor | None) -> Any:
        raise NotImplementedError

    def _run_experts(self, layer, moe_comm_method, experts, routing,
                     global_num_experts, expert_map, log2phy,
                     global_redundant_expert_num, shared_experts,
                     apply_router_weight_on_input, dynamic_eplb,
                     mc2_mask, pertoken_scale):
        return self._fused_experts(
            layer=layer,
            moe_comm_method=moe_comm_method,
            experts=experts,
            hidden_states=routing.hidden_states,
            topk_weights=routing.topk_weights,
            topk_ids=routing.topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=dynamic_eplb,
            mc2_mask=mc2_mask,
            pertoken_scale=pertoken_scale,
        )

    @staticmethod
    def _record_and_unwrap(layer, result,
                           slot_to_global: torch.Tensor | None):
        if not isinstance(result, tuple):
            return result

        output, group_list_type, expert_tokens = result
        layer.exo_executor.record_slot_expert_tokens(
            layer,
            group_list_type,
            expert_tokens,
            slot_to_global,
        )
        return output

    def _apply_native_runtime(self, layer, moe_comm_method, x, topk_weights,
                              topk_ids, global_num_experts, expert_map,
                              log2phy, global_redundant_expert_num,
                              shared_experts, apply_router_weight_on_input,
                              mc2_mask, pertoken_scale):
        collect_load = layer.runtime_config.runtime_mode == "profile"
        return self._fused_experts(
            layer=layer,
            moe_comm_method=moe_comm_method,
            experts=layer,
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=collect_load,
            mc2_mask=mc2_mask,
            pertoken_scale=pertoken_scale,
        )

    def _apply_split_offload(self, layer, moe_comm_method, x, topk_weights,
                             topk_ids, cold_experts, cold_topk_ids, cold_mask,
                             shared_experts, apply_router_weight_on_input,
                             mc2_mask, pertoken_scale):
        output = torch.zeros_like(x)
        num_rows = x.size(0)

        hot_topk_ids, hot_mask = layer.build_hot_local_routing(topk_ids,
                                                               cold_mask)
        hot_routing = build_routing_view(
            hidden_states=x,
            topk_ids=hot_topk_ids,
            topk_weights=topk_weights.masked_fill(~hot_mask, 0),
            row_mask=hot_mask.any(dim=1),
        )
        if hot_routing is not None:
            hot_output = self._run_experts(
                layer, moe_comm_method, layer, hot_routing,
                layer.resident_local_num_experts, None, None, 0,
                shared_experts, apply_router_weight_on_input, True,
                select_optional_rows(mc2_mask, hot_routing.row_indices,
                                     num_rows),
                select_optional_rows(pertoken_scale, hot_routing.row_indices,
                                     num_rows))
            hot_output = self._record_and_unwrap(
                layer, hot_output,
                layer.exo_executor.resident_slot_to_global(layer))
            add_routing_output(output, hot_routing, hot_output)

        cold_experts.wait()
        layer.exo_executor.prefetch_next_layers(layer)
        cold_routing = build_routing_view(
            hidden_states=x,
            topk_ids=cold_topk_ids,
            topk_weights=topk_weights.masked_fill(~cold_mask, 0),
            row_mask=cold_mask.any(dim=1),
        )
        if cold_routing is not None:
            cold_output = self._run_experts(
                layer, moe_comm_method, cold_experts, cold_routing,
                cold_experts.num_experts, None, None, 0, None,
                apply_router_weight_on_input, True,
                select_optional_rows(mc2_mask, cold_routing.row_indices,
                                     num_rows),
                select_optional_rows(pertoken_scale, cold_routing.row_indices,
                                     num_rows))
            cold_output = self._record_and_unwrap(
                layer, cold_output,
                layer.exo_executor.cold_slot_to_global(layer))
            add_routing_output(output, cold_routing, cold_output)
        return output

    def _apply_full_layer_offload(self, layer, moe_comm_method, x,
                                  topk_weights, topk_ids, cold_experts,
                                  shared_experts,
                                  apply_router_weight_on_input, mc2_mask,
                                  pertoken_scale):
        cold_experts.wait()
        layer.exo_executor.prefetch_next_layers(layer)
        result = self._fused_experts(
            layer=layer,
            moe_comm_method=moe_comm_method,
            experts=cold_experts,
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.full_expert_map,
            log2phy=None,
            global_redundant_expert_num=0,
            shared_experts=shared_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=True,
            mc2_mask=mc2_mask,
            pertoken_scale=pertoken_scale,
        )
        return self._record_and_unwrap(
            layer, result, layer.exo_executor.cold_slot_to_global(layer))

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              use_grouped_topk: bool,
              top_k: int,
              router_logits: torch.Tensor,
              renormalize: bool,
              topk_group: int | None = None,
              num_expert_group: int | None = None,
              custom_routing_function: Callable | None = None,
              scoring_func: str = "softmax",
              routed_scaling_factor: float = 1.0,
              e_score_correction_bias: torch.Tensor | None = None,
              global_num_experts: int = -1,
              expert_map: torch.Tensor | None = None,
              apply_router_weight_on_input: bool = False,
              enable_force_load_balance: bool = False,
              shared_experts: Any | None = None,
              pertoken_scale: torch.Tensor | None = None,
              **kwargs) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)
        topk_weights = topk_weights.to(x.dtype)

        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0),
                                       global_num_experts,
                                       device=topk_ids.device)
            topk_ids = torch.argsort(
                random_matrix, dim=1)[:, :topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        mc2_mask = kwargs.get("mc2_mask")
        log2phy = kwargs.get("log2phy")
        global_redundant_expert_num = int(
            kwargs.get("global_redundant_expert_num", 0))

        if layer.runtime_config.runtime_mode != "offload":
            return self._apply_native_runtime(
                layer, moe_comm_method, x, topk_weights, topk_ids,
                global_num_experts, expert_map, log2phy,
                global_redundant_expert_num, shared_experts,
                apply_router_weight_on_input, mc2_mask, pertoken_scale)

        prepared_cold_experts = layer.exo_executor.prepare_cold_experts(
            layer, topk_ids)

        cold_experts = prepared_cold_experts.experts
        if layer.exo_executor.offload_full_layer:
            return self._apply_full_layer_offload(
                layer, moe_comm_method, x, topk_weights, topk_ids,
                cold_experts, shared_experts, apply_router_weight_on_input,
                mc2_mask, pertoken_scale)

        return self._apply_split_offload(
            layer, moe_comm_method, x, topk_weights, topk_ids, cold_experts,
            prepared_cold_experts.topk_ids, prepared_cold_experts.mask,
            shared_experts,
            apply_router_weight_on_input, mc2_mask, pertoken_scale)


class RuntimeUnquantizedFusedMoEMethod(RuntimeFusedMoEMethod,
                                      UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig = None):
        super().__init__(moe=moe)
        self.dynamic_eplb = True

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)
        for name in ("w13_weight", "w2_weight"):
            data = self._maybe_pad_weight(getattr(layer, name).data)
            data = data.transpose(1, 2).contiguous()
            setattr(layer, name, torch.nn.Parameter(data, requires_grad=False))
            if get_ascend_device_type() != AscendDeviceType._310P:
                getattr(layer, name).data = maybe_trans_nz(
                    getattr(layer, name).data)
        layer.exo_executor.process_layer_after_loading(layer, self)

    def process_offloaded_weights(self,
                                  tensors: dict[str, torch.Tensor]) -> None:
        for name in ("w13_weight", "w2_weight"):
            tensors[name] = self._maybe_pad_weight(
                tensors[name]).transpose(1, 2).contiguous()

    def _fused_experts(self, layer, moe_comm_method, experts, hidden_states,
                       topk_weights, topk_ids, global_num_experts, expert_map,
                       log2phy, global_redundant_expert_num,
                       shared_experts, apply_router_weight_on_input,
                       dynamic_eplb, mc2_mask, pertoken_scale):
        local_num_experts = experts.w13_weight.shape[0]
        eplb_kwargs = {}
        if log2phy is not None:
            eplb_kwargs = {
                "log2phy": log2phy,
                "global_redundant_expert_num": global_redundant_expert_num,
            }
        with dispatch_with_local_experts(moe_comm_method, local_num_experts):
            return moe_comm_method.fused_experts(
                hidden_states=hidden_states,
                w1=experts.w13_weight,
                w2=experts.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                shared_experts=shared_experts,
                apply_router_weight_on_input=apply_router_weight_on_input,
                dynamic_eplb=dynamic_eplb,
                mc2_mask=mc2_mask,
                **eplb_kwargs)


class RuntimeW8A8DynamicFusedMoEMethod(RuntimeFusedMoEMethod,
                                      QuantizeMethodBase):
    """Runtime adapter around vLLM-Ascend's W8A8 dynamic MoE method."""

    def __init__(self, method: AscendFusedMoEMethod):
        self.base_method = method
        self.quant_method = method.quant_method
        self.dynamic_eplb = self.quant_method.dynamic_eplb

    def create_weights(self, *args, **kwargs):
        return self.base_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer):
        self.base_method.process_weights_after_loading(layer)
        layer.exo_executor.process_layer_after_loading(layer, self)

    def process_offloaded_weights(self,
                                  tensors: dict[str, torch.Tensor]) -> None:
        for name in ("w13_weight", "w2_weight"):
            tensors[name] = tensors[name].transpose(1, 2).contiguous()
        for name in ("w13_weight_scale", "w13_weight_offset",
                     "w2_weight_scale", "w2_weight_offset"):
            tensors[name] = tensors[name].view(tensors[name].shape[0],
                                               -1).contiguous()

    def _fused_experts(self, layer, moe_comm_method, experts, hidden_states,
                       topk_weights, topk_ids, global_num_experts, expert_map,
                       log2phy, global_redundant_expert_num,
                       shared_experts, apply_router_weight_on_input,
                       dynamic_eplb, mc2_mask, pertoken_scale):
        context = get_forward_context()
        w2_scale = (experts.w2_weight_scale_fp32
                    if context.moe_comm_type == MoECommType.FUSED_MC2
                    and envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2 else
                    experts.w2_weight_scale)
        local_num_experts = int(experts.w13_weight.shape[0])
        eplb_kwargs = {}
        if log2phy is not None:
            eplb_kwargs = {
                "log2phy": log2phy,
                "global_redundant_expert_num": global_redundant_expert_num,
            }
        with dispatch_with_local_experts(moe_comm_method, local_num_experts):
            return moe_comm_method.fused_experts(
                hidden_states=hidden_states,
                pertoken_scale=pertoken_scale,
                w1=[experts.w13_weight],
                w1_scale=[experts.w13_weight_scale_fp32],
                w2=[experts.w2_weight],
                w2_scale=[w2_scale],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=global_num_experts,
                use_int8_w8a8=True,
                expert_map=expert_map,
                shared_experts=shared_experts,
                dynamic_eplb=dynamic_eplb,
                mc2_mask=mc2_mask,
                **eplb_kwargs)


def wrap_quant_method(method):
    if (isinstance(method, AscendFusedMoEMethod)
            and isinstance(method.quant_method,
                           AscendW8A8DynamicFusedMoEMethod)):
        return RuntimeW8A8DynamicFusedMoEMethod(method)
    return method
