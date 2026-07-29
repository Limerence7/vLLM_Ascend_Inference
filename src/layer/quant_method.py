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
    AscendW8A8DynamicFusedMoEMethod, scale_from_float_to_int64
from vllm_ascend.utils import (AscendDeviceType, get_ascend_device_type,
                               maybe_trans_nz)

from .routing import (add_routing_output, build_routing_view,
                      dispatch_with_local_experts, select_optional_rows)


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
    def _slot_to_global(layer, cold: bool) -> torch.Tensor:
        expert_map, hot_count = layer.runtime_core.layout(layer)
        expert_ids = (expert_map[hot_count:]
                      if cold else expert_map[:hot_count])
        return torch.tensor(expert_ids, dtype=torch.long)

    @staticmethod
    def _unwrap_result(result):
        return result[0] if isinstance(result, tuple) else result

    def _offload_dynamic_eplb(self, layer) -> bool:
        return layer.runtime_core.collect_load

    def _wait_and_prefetch_next(self, layer, cold_experts) -> None:
        cold_experts.wait()
        layer.runtime_core.prefetch_next_layers(layer)

    def _run_cold_experts(self, layer, moe_comm_method, x, topk_weights,
                          cold_topk_ids, cold_mask, cold_experts,
                          apply_router_weight_on_input, dynamic_eplb,
                          mc2_mask, pertoken_scale):
        num_rows = x.size(0)
        cold_routing = build_routing_view(
            hidden_states=x,
            topk_ids=cold_topk_ids,
            topk_weights=topk_weights.masked_fill(~cold_mask, 0),
            row_mask=cold_mask.any(dim=1),
        )
        if cold_routing is None:
            return None, None

        cold_output = self._run_experts(
            layer, moe_comm_method, cold_experts, cold_routing,
            cold_experts.num_experts,
            None, None, 0, None,
            apply_router_weight_on_input, dynamic_eplb,
            select_optional_rows(mc2_mask, cold_routing.row_indices,
                                 num_rows),
            select_optional_rows(pertoken_scale, cold_routing.row_indices,
                                 num_rows))
        if layer.runtime_core.collect_load:
            cold_output = self._record_and_unwrap(
                layer, cold_output,
                self._slot_to_global(layer, cold=True))
        else:
            cold_output = self._unwrap_result(cold_output)
        return cold_routing, cold_output

    def _apply_cold_only(self, layer, moe_comm_method, x, topk_weights,
                         cold_experts, cold_topk_ids, cold_mask,
                         apply_router_weight_on_input, mc2_mask,
                         pertoken_scale):
        self._wait_and_prefetch_next(layer, cold_experts)
        dynamic_eplb = self._offload_dynamic_eplb(layer)
        cold_routing, cold_output = self._run_cold_experts(
            layer, moe_comm_method, x, topk_weights, cold_topk_ids, cold_mask,
            cold_experts, apply_router_weight_on_input, dynamic_eplb,
            mc2_mask, pertoken_scale)
        output = torch.zeros_like(x)
        if cold_routing is None:
            return output
        add_routing_output(output, cold_routing, cold_output)
        return output

    def _apply_split_offload(self, layer, moe_comm_method, x, topk_weights,
                             topk_ids, cold_experts, cold_topk_ids, cold_mask,
                             shared_experts, apply_router_weight_on_input,
                             mc2_mask, pertoken_scale):
        output = torch.zeros_like(x)
        num_rows = x.size(0)
        dynamic_eplb = self._offload_dynamic_eplb(layer)

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
                layer.local_num_experts,
                None, None, 0,
                shared_experts, apply_router_weight_on_input,
                dynamic_eplb,
                select_optional_rows(mc2_mask, hot_routing.row_indices,
                                     num_rows),
                select_optional_rows(pertoken_scale, hot_routing.row_indices,
                                     num_rows))
            if layer.runtime_core.collect_load:
                hot_output = self._record_and_unwrap(
                    layer, hot_output,
                    self._slot_to_global(layer, cold=False))
            else:
                hot_output = self._unwrap_result(hot_output)
            add_routing_output(output, hot_routing, hot_output)

        self._wait_and_prefetch_next(layer, cold_experts)
        cold_routing, cold_output = self._run_cold_experts(
            layer, moe_comm_method, x, topk_weights, cold_topk_ids, cold_mask,
            cold_experts, apply_router_weight_on_input, dynamic_eplb,
            mc2_mask, pertoken_scale)
        if cold_routing is not None:
            add_routing_output(output, cold_routing, cold_output)
        return output

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

        if not layer.runtime_core.uses_cold_buffer_for(layer):
            collect_load = (
                layer.runtime_core.config.runtime_mode == "profile"
                or layer.runtime_core.collect_load)
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

        prepared_cold_experts = layer.runtime_core.prepare_cold_experts(
            layer, topk_ids)
        if layer.local_num_experts == 0:
            return self._apply_cold_only(
                layer, moe_comm_method, x, topk_weights,
                prepared_cold_experts.experts, prepared_cold_experts.topk_ids,
                prepared_cold_experts.mask, apply_router_weight_on_input,
                mc2_mask, pertoken_scale)
        return self._apply_split_offload(
            layer, moe_comm_method, x, topk_weights, topk_ids,
            prepared_cold_experts.experts, prepared_cold_experts.topk_ids,
            prepared_cold_experts.mask, shared_experts,
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
                pertoken_scale=pertoken_scale,
                **eplb_kwargs)


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

    def _offload_dynamic_eplb(self, layer) -> bool:
        return layer.runtime_core.collect_load

    def process_weights_after_loading(self, layer):
        self.base_method.process_weights_after_loading(layer)
        if layer.runtime_core.config.runtime_mode == "balance":
            self._prepare_weight_lists(layer)
            layer.runtime_core.process_layer_after_loading(layer, self)
        elif layer.runtime_core.config.runtime_mode == "offload":
            layer.runtime_core.process_layer_after_loading(layer, self)
            self._prepare_weight_lists(layer)

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
        fused_mc2 = context.moe_comm_type == MoECommType.FUSED_MC2
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
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                global_num_experts=global_num_experts,
                use_int8_w8a8=True,
                expert_map=expert_map,
                shared_experts=shared_experts,
                dynamic_eplb=dynamic_eplb,
                mc2_mask=mc2_mask,
                **eplb_kwargs)

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

def _release_layer_weights(layer, names: tuple[str, ...]) -> None:
    for name in names:
        if hasattr(layer, name):
            delattr(layer, name)
    torch.npu.synchronize()
    torch.npu.empty_cache()


def wrap_quant_method(method):
    if (isinstance(method, AscendFusedMoEMethod)
            and isinstance(method.quant_method,
                           AscendW8A8DynamicFusedMoEMethod)):
        return RuntimeW8A8DynamicFusedMoEMethod(method)
    return method
