from __future__ import annotations

from types import MethodType
from typing import Optional

import torch
import torch_npu
from vllm.forward_context import get_forward_context
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl


class CompactAllGatherDispatchPatch:
    """
    Temporarily make AllGather dispatch emit compact-local expert groups.

    Ascend's default AllGather token dispatcher masks non-local experts with
    expert_map but still routes by the original contiguous global expert range.
    Compact expert-wise offload shrinks the local weight tensor to resident
    experts plus cache slots, so grouped matmul needs group_list length to match
    compact weight dim0.
    """

    def __init__(self, *, num_compact_slots: int):
        self.num_compact_slots = int(num_compact_slots)
        self._dispatcher = None

    def __enter__(self):
        self._dispatcher = activate_compact_allgather_dispatch(self.num_compact_slots)
        return self

    def __exit__(self, exc_type, exc, tb):
        deactivate_compact_allgather_dispatch(self._dispatcher)
        return False


def activate_compact_allgather_dispatch(*, num_compact_slots: int):
    moe_comm_method = get_forward_context().moe_comm_method
    if not isinstance(moe_comm_method, AllGatherCommImpl):
        raise RuntimeError(
            "compact_npu_cache shrink currently supports only the "
            "AllGather MoE communication backend."
        )

    dispatcher = moe_comm_method.token_dispatcher
    if not hasattr(dispatcher, "_expert_wise_original_token_dispatch"):
        dispatcher._expert_wise_original_token_dispatch = dispatcher.token_dispatch
        dispatcher.token_dispatch = MethodType(
            _compact_allgather_token_dispatch,
            dispatcher,
        )
    dispatcher._expert_wise_compact_slots = int(num_compact_slots)
    return dispatcher


def deactivate_compact_allgather_dispatch(dispatcher) -> None:
    if dispatcher is not None and hasattr(dispatcher, "_expert_wise_compact_slots"):
        delattr(dispatcher, "_expert_wise_compact_slots")


def _compact_allgather_token_dispatch(
    self,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: Optional[torch.Tensor] = None,
    log2phy: Optional[torch.Tensor] = None,
    global_redundant_expert_num: int = 0,
    shared_experts=None,
    quantized_x_for_share=None,
    dynamic_scale_for_share=None,
    mc2_mask: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    with_quant: bool = False,
    dynamic_eplb: bool = False,
    pertoken_scale: Optional[torch.Tensor] = None,
):
    num_compact_slots = getattr(self, "_expert_wise_compact_slots", None)
    if num_compact_slots is None:
        return self._expert_wise_original_token_dispatch(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            mc2_mask=mc2_mask,
            apply_router_weight_on_input=apply_router_weight_on_input,
            with_quant=with_quant,
            dynamic_eplb=dynamic_eplb,
            pertoken_scale=pertoken_scale,
        )

    if expert_map is None:
        raise RuntimeError("compact AllGather dispatch requires expert_map.")
    if log2phy is not None:
        raise RuntimeError("compact AllGather dispatch does not support log2phy yet.")
    if shared_experts is not None:
        raise RuntimeError("compact AllGather dispatch does not support shared experts yet.")
    if dynamic_eplb:
        raise RuntimeError("compact AllGather dispatch does not support dynamic EPLB.")

    self.with_quant = with_quant
    self.original_shape = hidden_states.shape

    num_tokens = hidden_states.shape[:-1].numel()
    self.apply_router_weight_on_input = apply_router_weight_on_input
    if self.apply_router_weight_on_input:
        assert topk_weights.dim() == 2
        _, topk = topk_weights.shape
        assert topk == 1
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

    num_compact_slots = int(num_compact_slots)
    compact_topk_ids = expert_map[topk_ids]
    local_mask = compact_topk_ids != -1
    topk_weights = topk_weights * local_mask

    routing_expert_num = max(int(expert_map.numel()), num_compact_slots + 1)
    inactive_expert = torch.full_like(compact_topk_ids, num_compact_slots)
    compact_topk_ids = torch.where(local_mask, compact_topk_ids, inactive_expert)
    compact_topk_ids = compact_topk_ids.to(torch.int32)

    sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
        torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            compact_topk_ids,
            scale=pertoken_scale,
            active_num=num_tokens * self.top_k,
            expert_num=routing_expert_num,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_compact_slots],
            quant_mode=1 if self.with_quant and pertoken_scale is None else -1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)
    context_metadata = {
        "topk_weights": topk_weights,
        "expanded_row_idx": expanded_row_idx,
    }
    return {
        "group_list_type": 1,
        "hidden_states": sorted_hidden_states,
        "group_list": expert_tokens,
        "dynamic_scale": pertoken_scale if self.with_quant else None,
        "context_metadata": context_metadata,
    }
