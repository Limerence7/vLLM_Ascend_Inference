from typing import Any, Callable

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    build_fused_experts_input, build_token_dispatch_input)
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8_dynamic import \
    AscendW8A8DynamicFusedMoEMethod, scale_from_float_to_int64
from vllm_ascend.utils import (AscendDeviceType, get_ascend_device_type,
                               maybe_trans_nz)

from .routing import dispatch_with_local_experts
from .moe_mlp import unified_apply_mlp as runtime_unified_apply_mlp


class RuntimeFusedMoEMethod:
    """Runtime routing flow for supported MoE weight formats."""

    dynamic_eplb: bool

    def _fused_experts(self, layer, moe_comm_method,
                       experts: torch.nn.Module, hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                       global_num_experts: int,
                       expert_map: torch.Tensor,
                       log2phy: torch.Tensor | None,
                       global_redundant_expert_num: int,
                       shared_experts: Any | None,
                       apply_router_weight_on_input: bool,
                       dynamic_eplb: bool,
                       mc2_mask: torch.Tensor | None,
                       pertoken_scale: torch.Tensor | None) -> Any:
        raise NotImplementedError

    @staticmethod
    def _record_and_unwrap(layer, result,
                           slot_to_global: torch.Tensor):
        if not isinstance(result, tuple):
            return result

        output, group_list_type, expert_tokens = result
        layer.runtime_core.record_slot_expert_tokens(
            layer,
            group_list_type,
            expert_tokens,
            slot_to_global,
        )
        return output

    @staticmethod
    def _unwrap_result(result):
        return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def _normalize_fused_result(result, dynamic_eplb: bool):
        if not hasattr(result, "routed_out"):
            return result
        if dynamic_eplb:
            return (result.routed_out, result.group_list_type,
                    result.expert_tokens)
        return result.routed_out

    @staticmethod
    def _build_fused_input(layer, hidden_states, topk_weights, topk_ids,
                           w1, w2, expert_map, log2phy,
                           global_redundant_expert_num, mc2_mask,
                           apply_router_weight_on_input, dynamic_eplb,
                           pertoken_scale, w1_scale=None, w2_scale=None):
        return build_fused_experts_input(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            quant_type=layer.quant_type,
            dynamic_eplb=dynamic_eplb,
            expert_map=expert_map,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
            activation=layer.activation.value,
        )

    def _wait_and_prefetch_next(self, layer, cold_experts) -> None:
        cold_experts.wait()
        layer.runtime_core.prefetch_next_layers(layer)

    def _supports_unified_cold_buffer(self, layer) -> bool:
        config = layer.runtime_core.config
        if config.runtime_mode not in ("offload", "balance"):
            return False
        has_w8a8_lists = all(
            hasattr(layer, name) for name in (
                "w13_weight_list",
                "w2_weight_list",
                "w13_weight_scale_fp32_list",
                "w2_weight_scale_list",
            ))
        has_unquantized_tensors = all(
            hasattr(layer, name) for name in ("w13_weight", "w2_weight"))
        return has_w8a8_lists or has_unquantized_tensors

    def _apply_unified_cold_buffer(self, layer, moe_comm_method, x,
                                   topk_weights, topk_ids, shared_experts,
                                   apply_router_weight_on_input, mc2_mask,
                                   pertoken_scale, collect_load: bool):
        prepared = layer.runtime_core.prepare_combined_experts(
            layer, topk_ids.device)
        self._wait_and_prefetch_next(layer, prepared.experts.cold_experts)

        output = self._fused_experts(
            layer=layer,
            moe_comm_method=moe_comm_method,
            experts=prepared.experts,
            hidden_states=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=layer.global_num_experts,
            expert_map=prepared.expert_map,
            log2phy=None,
            global_redundant_expert_num=0,
            shared_experts=shared_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=collect_load,
            mc2_mask=mc2_mask,
            pertoken_scale=pertoken_scale)
        if collect_load:
            return self._record_and_unwrap(layer, output,
                                           prepared.slot_to_global)
        return self._unwrap_result(output)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              expert_map: torch.Tensor,
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
        layer.runtime_core.record_request_experts(layer, topk_ids)

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
        collect_load = layer.runtime_core.begin_forward_collect(layer)

        if not layer.runtime_core.uses_cold_buffer_for(layer):
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

        if not self._supports_unified_cold_buffer(layer):
            raise NotImplementedError(
                "Cold-buffer runtime requires unified expert compute with either "
                "W8A8 expert lists or unquantized expert tensors.")
        return self._apply_unified_cold_buffer(
            layer, moe_comm_method, x, topk_weights, topk_ids,
            shared_experts, apply_router_weight_on_input, mc2_mask,
            pertoken_scale, collect_load)


class RuntimeUnquantizedFusedMoEMethod(RuntimeFusedMoEMethod,
                                      UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig = None):
        super().__init__(moe=moe)
        self.dynamic_eplb = True

    def process_weights_after_loading(self, layer):
        if _all_experts_offloaded(layer):
            layer.runtime_core.process_layer_after_loading(layer, self)
            return

        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)
        for name in ("w13_weight", "w2_weight"):
            data = self._maybe_pad_weight(getattr(layer, name).data)
            data = data.transpose(1, 2).contiguous()
            setattr(layer, name, torch.nn.Parameter(data, requires_grad=False))
            if get_ascend_device_type() != AscendDeviceType._310P:
                getattr(layer, name).data = maybe_trans_nz(
                    getattr(layer, name).data)
        if layer.runtime_core.config.runtime_mode in ("offload", "balance"):
            layer.runtime_core.process_layer_after_loading(layer, self)

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
        if getattr(experts, "runtime_combined_experts", False):
            with dispatch_with_local_experts(moe_comm_method,
                                             experts.num_experts):
                return self._runtime_fused_experts(
                    layer=layer,
                    moe_comm_method=moe_comm_method,
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                    w1=experts.w13_weight_list,
                    w2=experts.w2_weight_list,
                    shared_experts=shared_experts,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    dynamic_eplb=dynamic_eplb,
                    mc2_mask=mc2_mask,
                    pertoken_scale=pertoken_scale,
                    log2phy=log2phy,
                    global_redundant_expert_num=global_redundant_expert_num)

        local_num_experts = experts.w13_weight.shape[0]
        with dispatch_with_local_experts(moe_comm_method, local_num_experts):
            fused_input = self._build_fused_input(
                layer, hidden_states, topk_weights, topk_ids,
                w1=experts.w13_weight,
                w2=experts.w2_weight,
                expert_map=expert_map,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                dynamic_eplb=dynamic_eplb,
                pertoken_scale=pertoken_scale)
            result = moe_comm_method.fused_experts(
                fused_experts_input=fused_input)
            return self._normalize_fused_result(result, dynamic_eplb)

    @staticmethod
    def _runtime_fused_experts(
            layer,
            moe_comm_method,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            global_num_experts: int,
            expert_map: torch.Tensor,
            w1: list[torch.Tensor],
            w2: list[torch.Tensor],
            shared_experts: Any | None,
            apply_router_weight_on_input: bool,
            dynamic_eplb: bool,
            mc2_mask: torch.Tensor | None,
            pertoken_scale: torch.Tensor | None,
            log2phy: torch.Tensor | None = None,
            global_redundant_expert_num: int = 0):
        fused_input = RuntimeFusedMoEMethod._build_fused_input(
            layer=layer, hidden_states=hidden_states,
            topk_weights=topk_weights, topk_ids=topk_ids, w1=w1, w2=w2,
            expert_map=expert_map, log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=dynamic_eplb, pertoken_scale=pertoken_scale)
        results = moe_comm_method.token_dispatcher.token_dispatch(
            token_dispatch_input=build_token_dispatch_input(
                fused_experts_input=fused_input))

        expert_tokens = results.group_list
        group_list_type = results.group_list_type
        mlp_output = runtime_unified_apply_mlp(
            hidden_states=results.hidden_states,
            w1=w1,
            w2=w2,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            topk_scales=results.topk_scales,
            with_quant=False,
            dynamic_eplb=dynamic_eplb)
        final_hidden_states = moe_comm_method.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=results.combine_metadata)
        if dynamic_eplb:
            return final_hidden_states, group_list_type, expert_tokens
        return final_hidden_states


class RuntimeW8A8DynamicFusedMoEMethod(RuntimeFusedMoEMethod,
                                      QuantizeMethodBase):
    """Runtime adapter around vLLM-Ascend's W8A8 dynamic MoE method."""

    def __init__(self, method: AscendFusedMoEMethod):
        self.base_method = method
        self.quant_method = method.quant_method
        self.dynamic_eplb = self.quant_method.dynamic_eplb

    def disable_native_dynamic_eplb(self) -> None:
        self.dynamic_eplb = False
        self.quant_method.dynamic_eplb = False

    def create_weights(self, *args, **kwargs):
        return self.base_method.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer):
        if _all_experts_offloaded(layer):
            layer.runtime_core.process_layer_after_loading(layer, self)
            self._prepare_empty_weight_lists(layer)
            return

        self.base_method.process_weights_after_loading(layer)
        if layer.runtime_core.config.uses_cold_buffer:
            layer.runtime_core.process_layer_after_loading(layer, self)
            self._prepare_weight_lists(layer)
        elif layer.runtime_core.config.runtime_mode == "balance":
            layer.runtime_core.process_layer_after_loading(layer, self)

    def process_offloaded_weights(self,
                                  tensors: dict[str, torch.Tensor]) -> None:
        for name in ("w13_weight", "w2_weight"):
            tensors[name] = tensors[name].transpose(1, 2).contiguous()
        for name in ("w13_weight_scale", "w2_weight_scale"):
            tensors[name] = tensors[name].view(tensors[name].shape[0],
                                               -1).contiguous()
        tensors.pop("w13_weight_offset", None)
        tensors.pop("w2_weight_offset", None)

    def _fused_experts(self, layer, moe_comm_method, experts, hidden_states,
                       topk_weights, topk_ids, global_num_experts, expert_map,
                       log2phy, global_redundant_expert_num,
                       shared_experts, apply_router_weight_on_input,
                       dynamic_eplb, mc2_mask, pertoken_scale):
        context = get_forward_context()
        combined_experts = getattr(experts, "runtime_combined_experts", False)
        fused_mc2 = (context.moe_comm_type == MoECommType.FUSED_MC2
                     and not combined_experts)
        w1 = self._weight_arg(experts, "w13_weight_list", "w13_weight",
                              fused_mc2)
        w2 = self._weight_arg(experts, "w2_weight_list", "w2_weight",
                              fused_mc2)
        w2_scale_name = (
            "w2_weight_scale_fp32"
            if fused_mc2 and envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2 else
            "w2_weight_scale")
        w2_scale_list_name = (
            "w2_weight_scale_fp32_list"
            if w2_scale_name == "w2_weight_scale_fp32" else
            "w2_weight_scale_list")
        w1_scale = self._scale_arg(experts, "w13_weight_scale_fp32_list",
                                   "w13_weight_scale_fp32", fused_mc2)
        w2_scale = self._scale_arg(experts, w2_scale_list_name,
                                   w2_scale_name, fused_mc2)
        local_num_experts = int(w1[0].shape[0]) if fused_mc2 else len(w1)
        if combined_experts:
            with dispatch_with_local_experts(moe_comm_method, local_num_experts):
                return self._runtime_fused_experts(
                    layer=layer,
                    moe_comm_method=moe_comm_method,
                    hidden_states=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    global_num_experts=global_num_experts,
                    expert_map=expert_map,
                    w1=w1,
                    w1_scale=w1_scale,
                    w2=w2,
                    w2_scale=w2_scale,
                    shared_experts=shared_experts,
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    dynamic_eplb=dynamic_eplb,
                    mc2_mask=mc2_mask,
                    pertoken_scale=pertoken_scale,
                    log2phy=log2phy,
                    global_redundant_expert_num=global_redundant_expert_num)
        with dispatch_with_local_experts(moe_comm_method, local_num_experts):
            fused_input = self._build_fused_input(
                layer, hidden_states, topk_weights, topk_ids,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                expert_map=expert_map,
                log2phy=log2phy,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                dynamic_eplb=dynamic_eplb,
                pertoken_scale=pertoken_scale)
            result = moe_comm_method.fused_experts(
                fused_experts_input=fused_input)
            return self._normalize_fused_result(result, dynamic_eplb)

    @staticmethod
    def _runtime_fused_experts(
            layer,
            moe_comm_method,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            global_num_experts: int,
            expert_map: torch.Tensor,
            w1: list[torch.Tensor],
            w1_scale: list[torch.Tensor],
            w2: list[torch.Tensor],
            w2_scale: list[torch.Tensor],
            shared_experts: Any | None,
            apply_router_weight_on_input: bool,
            dynamic_eplb: bool,
            mc2_mask: torch.Tensor | None,
            pertoken_scale: torch.Tensor | None,
            log2phy: torch.Tensor | None = None,
            global_redundant_expert_num: int = 0):
        fused_input = RuntimeFusedMoEMethod._build_fused_input(
            layer=layer, hidden_states=hidden_states,
            topk_weights=topk_weights, topk_ids=topk_ids,
            w1=w1, w1_scale=w1_scale, w2=w2, w2_scale=w2_scale,
            expert_map=expert_map, log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=dynamic_eplb, pertoken_scale=pertoken_scale)
        results = moe_comm_method.token_dispatcher.token_dispatch(
            token_dispatch_input=build_token_dispatch_input(
                fused_experts_input=fused_input))

        expert_tokens = results.group_list
        group_list_type = results.group_list_type
        mlp_output = runtime_unified_apply_mlp(
            hidden_states=results.hidden_states,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            group_list=expert_tokens,
            dynamic_scale=results.dynamic_scale,
            group_list_type=group_list_type,
            topk_scales=results.topk_scales,
            with_quant=True,
            fusion=True,
            dynamic_eplb=dynamic_eplb)
        final_hidden_states = moe_comm_method.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=results.combine_metadata)
        if dynamic_eplb:
            return final_hidden_states, group_list_type, expert_tokens
        return final_hidden_states

    @staticmethod
    def _tensor_list(experts, list_name: str, tensor_name: str) -> list:
        split_dim = 1 if "scale" in tensor_name or "offset" in tensor_name else 4
        tensor_list = getattr(experts, list_name, None)
        if tensor_list is not None:
            if tensor_list and tensor_list[0].dim() > split_dim:
                tensor_list = list(tensor_list[0].unbind(dim=0))
                setattr(experts, list_name, tensor_list)
            return tensor_list
        tensor = getattr(experts, tensor_name)
        if tensor.dim() > split_dim:
            tensor_list = list(tensor.unbind(dim=0))
            setattr(experts, list_name, tensor_list)
            return tensor_list
        return [tensor]

    @classmethod
    def _weight_arg(cls, experts, list_name: str, tensor_name: str,
                    fused_mc2: bool) -> list:
        if not fused_mc2:
            return cls._tensor_list(experts, list_name, tensor_name)
        tensor = getattr(experts, tensor_name, None)
        if tensor is None:
            tensor = torch.stack(list(getattr(experts, list_name)))
        return [tensor]

    @classmethod
    def _scale_arg(cls, experts, list_name: str, tensor_name: str,
                   fused_mc2: bool) -> list:
        if not fused_mc2:
            return cls._tensor_list(experts, list_name, tensor_name)

        fused_name = {
            "w13_weight_scale_fp32": "fused_w1_scale",
            "w13_weight_scale": "fused_w1_scale",
            "w2_weight_scale_fp32": "fused_w2_scale",
            "w2_weight_scale": "fused_w2_scale",
        }.get(tensor_name)
        fused_scale = (
            getattr(experts, fused_name, None)
            if fused_name is not None else None)
        if fused_scale is not None:
            return [fused_scale]

        tensor = getattr(experts, tensor_name, None)
        if tensor is None:
            tensor = torch.stack(list(getattr(experts, list_name)))
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            tensor = scale_from_float_to_int64(tensor)
        return [tensor]

    @staticmethod
    def _prepare_weight_lists(layer) -> None:
        layer.w13_weight_list = [
            weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)
        ]
        layer.w2_weight_list = [
            weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)
        ]
        layer.w13_weight_scale_fp32_list = [
            weight.clone()
            for weight in layer.w13_weight_scale_fp32.data.unbind(dim=0)
        ]
        layer.w2_weight_scale_list = [
            weight.clone()
            for weight in layer.w2_weight_scale.data.unbind(dim=0)
        ]
        if hasattr(layer, "w2_weight_scale_fp32"):
            layer.w2_weight_scale_fp32_list = [
                weight.clone()
                for weight in layer.w2_weight_scale_fp32.data.unbind(dim=0)
            ]
        _release_layer_weights(
            layer,
            (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w13_weight_scale_fp32",
                "w13_weight_offset",
                "w2_weight_scale",
                "w2_weight_scale_fp32",
                "w2_weight_offset",
            ),
        )

    @staticmethod
    def _prepare_empty_weight_lists(layer) -> None:
        layer.w13_weight_list = []
        layer.w2_weight_list = []
        layer.w13_weight_scale_fp32_list = []
        layer.w2_weight_scale_list = []
        if hasattr(layer, "w2_weight_scale_fp32"):
            layer.w2_weight_scale_fp32_list = []
        _release_layer_weights(
            layer,
            (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w13_weight_scale_fp32",
                "w13_weight_offset",
                "w2_weight_scale",
                "w2_weight_scale_fp32",
                "w2_weight_offset",
            ),
        )

def _release_layer_weights(layer, names: tuple[str, ...]) -> None:
    for name in names:
        if hasattr(layer, name):
            delattr(layer, name)
    torch.npu.synchronize()
    torch.npu.empty_cache()


def _all_experts_offloaded(layer) -> bool:
    return (layer.runtime_core.uses_cold_buffer_for(layer)
            and int(layer.local_num_experts) == 0)


def wrap_quant_method(method):
    if (isinstance(method, AscendFusedMoEMethod)
            and isinstance(method.quant_method,
                           AscendW8A8DynamicFusedMoEMethod)):
        return RuntimeW8A8DynamicFusedMoEMethod(method)
    return method
