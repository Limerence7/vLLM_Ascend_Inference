from typing import Optional

import torch
import torch_npu
from torch.nn.functional import pad
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.moe_mlp import _custom_gmm_swiglu_enabled
from vllm_ascend.utils import dispose_tensor, get_weight_prefetch_method


def cumsum_group_list(group_list: torch.Tensor, src_list_type: int,
                      dst_list_type: int, active_num: int = 0,
                      expert_num: int = 0) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(
            f"group_list_type should be in [0, 1, 2], got {src_list_type}.")
    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        return torch.cat(
            [group_list[:1], group_list[1:] - group_list[:-1]])
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        result = torch.full((expert_num, ),
                            active_num,
                            dtype=group_list.dtype,
                            device=group_list.device)
        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                result[start:end] = tokens[i]
        return result
    raise NotImplementedError(
        f"Unsupported group_list conversion {src_list_type}->{dst_list_type}.")


def quant_apply_mlp(hidden_states: torch.Tensor,
                    w1: list[torch.Tensor],
                    w1_scale: list[torch.Tensor],
                    w2: list[torch.Tensor],
                    w2_scale: list[torch.Tensor],
                    group_list: torch.Tensor,
                    group_list_type: int = 1,
                    dynamic_scale: torch.Tensor = None,
                    w1_scale_bias: torch.Tensor = None,
                    w2_scale_bias: torch.Tensor = None,
                    w1_offset: Optional[torch.Tensor] = None,
                    w2_offset: Optional[torch.Tensor] = None,
                    fusion: bool = False,
                    dynamic_eplb: bool = False) -> torch.Tensor:
    if w1_offset is not None:
        raise NotImplementedError("Runtime offload does not support offsets.")
    if dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        pertoken_scale = dynamic_scale
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    output_dtype = w2_scale[0].dtype
    use_weight_list = len(w1) > 1

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(
            hidden_states)

    if get_forward_context().moe_comm_type == MoECommType.MC2:
        raise NotImplementedError("Runtime offload unified MLP excludes MC2.")

    if w1_scale_bias is not None:
        if group_list_type == 0:
            group_list = torch.cat(
                [group_list[:1], torch.diff(group_list, dim=0)])
            group_list_type = 1
        bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
        bias2 = [w2_scale_bias]
        output_dtype = torch.bfloat16

    if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
        hidden_states, swiglu_out_scale, _ = (
            torch.ops._C_ascend.
            grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                bias=bias1,
            ))
    elif fusion and not dynamic_eplb and not use_weight_list:
        hidden_states, swiglu_out_scale, _ = (
            torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                bias=bias1,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale))
        if quantized_hidden_states is not None:
            dispose_tensor(quantized_hidden_states)
    else:
        w1_scale = [scale.to(w2_scale[0].dtype) for scale in w1_scale]
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w1,
            scale=w1_scale,
            bias=bias1,
            per_token_scale=[pertoken_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype)[0]
        if quantized_hidden_states is not None:
            dispose_tensor(quantized_hidden_states)
        if HAS_TRITON:
            from vllm_ascend.ops.triton.activation.swiglu_quant import (
                swiglu_quant)
            hidden_states, swiglu_out_scale = swiglu_quant(
                hidden_states,
                group_list=group_list,
                group_list_type=group_list_type)
        else:
            hidden_states = torch_npu.npu_swiglu(hidden_states)
            hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
                hidden_states)

    return torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=w2,
        scale=w2_scale,
        bias=bias2,
        per_token_scale=[swiglu_out_scale],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
        output_dtype=output_dtype)[0]


def unified_apply_mlp(hidden_states: torch.Tensor,
                      w1: list[torch.Tensor],
                      w2: list[torch.Tensor],
                      group_list: torch.Tensor,
                      w1_scale: list[torch.Tensor],
                      w2_scale: list[torch.Tensor],
                      dynamic_scale: torch.Tensor = None,
                      group_list_type: int = 1,
                      w1_scale_bias: torch.Tensor = None,
                      w2_scale_bias: torch.Tensor = None,
                      topk_scales: Optional[torch.Tensor] = None,
                      with_quant: bool = False,
                      fusion: bool = False,
                      dynamic_eplb: bool = False) -> torch.Tensor:
    if not with_quant:
        raise NotImplementedError("Runtime offload unified MLP supports W8A8.")
    return quant_apply_mlp(
        hidden_states=hidden_states,
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        group_list=group_list,
        dynamic_scale=dynamic_scale,
        group_list_type=group_list_type,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        fusion=fusion,
        dynamic_eplb=dynamic_eplb)
