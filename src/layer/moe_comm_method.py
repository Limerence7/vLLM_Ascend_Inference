from __future__ import annotations

from typing import Any

import torch
from vllm.forward_context import get_forward_context
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import moe_comm_method as native_comm
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalizeWithAllGather, QuantType)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    TokenDispatcherWithAllGather)


class RuntimeAllGatherCommImpl(native_comm.MoECommMethod):
    """AllGather MoE path with explicit dispatch, MLP, and combine stages."""

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)

    def fused_experts(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor | list[torch.Tensor],
        w2: torch.Tensor | list[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_int8_w8a8: bool = False,
        use_int4_w4a8: bool = False,
        use_int4_w4a16: bool = False,
        global_num_experts: int | None = None,
        expert_map: torch.Tensor | None = None,
        w1_scale: list[torch.Tensor] | None = None,
        w2_scale: list[torch.Tensor] | None = None,
        w1_scale_bias: torch.Tensor | None = None,
        w2_scale_bias: torch.Tensor | None = None,
        w1_offset: torch.Tensor | None = None,
        w2_offset: torch.Tensor | None = None,
        shared_experts: Any | None = None,
        quantized_x_for_share: Any | None = None,
        dynamic_scale_for_share: Any | None = None,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        need_trans: bool = False,
        dynamic_eplb: bool = False,
        mc2_mask: torch.Tensor | None = None,
        pertoken_scale: torch.Tensor | None = None,
    ):
        assert hidden_states.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
        )
        assert get_forward_context().moe_comm_method is not None

        dispatched = self._dispatch(
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
            with_quant=use_int8_w8a8 or use_int4_w4a8,
            dynamic_eplb=dynamic_eplb,
            pertoken_scale=pertoken_scale,
        )
        output = self._combine(
            self._apply_mlp(
                dispatched,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                w1_offset=w1_offset,
                w2_offset=w2_offset,
                use_quant=use_int8_w8a8 or use_int4_w4a8 or use_int4_w4a16,
                fusion=use_int8_w8a8,
                need_trans=need_trans,
                dynamic_eplb=dynamic_eplb,
            ),
            dispatched["context_metadata"],
        )

        if dynamic_eplb:
            return output, dispatched["group_list_type"], dispatched["group_list"]
        return output

    def _dispatch(self, **kwargs):
        return self.token_dispatcher.token_dispatch(**kwargs)

    @staticmethod
    def _apply_mlp(dispatched: dict[str, Any], **kwargs):
        return unified_apply_mlp(
            hidden_states=dispatched["hidden_states"],
            group_list=dispatched["group_list"],
            dynamic_scale=dispatched.get("dynamic_scale"),
            group_list_type=dispatched["group_list_type"],
            topk_scales=dispatched.get("topk_scales"),
            **kwargs,
        )

    def _combine(self, hidden_states: torch.Tensor,
                 context_metadata: dict[str, Any]) -> torch.Tensor:
        return self.token_dispatcher.token_combine(
            hidden_states=hidden_states,
            context_metadata=context_metadata,
        )


def setup_moe_comm_method(moe_config) -> None:
    native_comm.setup_moe_comm_method(moe_config)
    native_comm._MoECommMethods[MoECommType.ALLGATHER] = (
        RuntimeAllGatherCommImpl(moe_config))


__all__ = [
    "QuantType",
    "RuntimeAllGatherCommImpl",
    "setup_moe_comm_method",
]
