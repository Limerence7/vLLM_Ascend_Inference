from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch_npu
from vllm.model_executor.models.utils import PPMissingLayer

from ..config import OffloadConfig
from ..utils import ExpertBuffer, ExpertBufferPool, ExpertOffloadSlot

from .offload import LayerWiseOffloadBuffers
from .prefetch import LayerWisePrefetcher
from .scheduler import LayerWiseScheduler


class LayerWiseManager:
    def __init__(self, model: nn.Module, config: OffloadConfig):
        self.model = model
        self.config = config
        self.scheduler = LayerWiseScheduler(config)

        self.moe_layer_indices: Set[int] = set()
        self.offload_layer_indices: List[int] = []
        self.expert_slots: Dict[int, ExpertOffloadSlot] = {}

        self.offload_buffers = LayerWiseOffloadBuffers(config)
        self.expert_buffer_pool: Optional[ExpertBufferPool] = None
        self.prefetcher: Optional[LayerWisePrefetcher] = None

        self.shared_npu_experts: Optional[nn.Module] = None
        self.shared_npu_owner_layer: Optional[int] = None
        self.initialized = False
        self.forward_patched = False

    def setup_after_weight_loading(self) -> None:
        if not self.initialized:
            self._init_expert_offload_slots()
            self.initialized = True

        if not self.forward_patched:
            self._patch_moe_forwards()
            self.forward_patched = True

    def _available_moe_layer_indices(self) -> List[int]:
        return [
            layer_idx
            for layer_idx, layer in enumerate(self.model.model.layers)
            if self._is_moe_layer(layer)
        ]

    @staticmethod
    def _is_moe_layer(layer: nn.Module) -> bool:
        if isinstance(layer, PPMissingLayer):
            return False
        return hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")

    def _iter_offload_layers(self) -> Iterator[Tuple[int, nn.Module]]:
        selected = set(self.offload_layer_indices)
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in selected and self._is_moe_layer(layer):
                yield layer_idx, layer

    def _init_expert_offload_slots(self) -> None:
        available_layers = self._available_moe_layer_indices()
        self.offload_layer_indices = self.scheduler.select_layers(available_layers)
        offload_layers = list(self._iter_offload_layers())

        if not offload_layers:
            print(
                "[Plugin] No valid MoE layers selected for layer-wise offload; "
                f"available_layers={available_layers}, interval="
                f"{self.config.offload_interval}"
            )
            return

        owner_layer_idx, owner_layer = offload_layers[0]
        buffer0_experts = owner_layer.mlp.experts
        self.expert_buffer_pool = self.offload_buffers.create_pool(buffer0_experts)
        self.expert_buffer_pool.mark_loaded(owner_layer_idx, buf_idx=0)
        self.prefetcher = LayerWisePrefetcher(
            self.expert_buffer_pool,
            self.expert_slots,
        )

        self.shared_npu_experts = buffer0_experts
        self.shared_npu_owner_layer = owner_layer_idx

        for layer_idx, layer in offload_layers:
            self._register_offloaded_layer(layer_idx, layer, buffer0_experts)

        torch_npu.npu.empty_cache()
        print(
            "[Plugin] Layer-wise offload initialized: "
            f"layers={self.offload_layer_indices}, "
            f"buffers={len(self.expert_buffer_pool)}"
        )

    def _register_offloaded_layer(
        self,
        layer_idx: int,
        layer: nn.Module,
        buffer0_experts: nn.Module,
    ) -> None:
        experts = layer.mlp.experts
        self.moe_layer_indices.add(layer_idx)
        self.expert_slots[layer_idx] = ExpertOffloadSlot(
            layer_idx=layer_idx,
            experts_module=experts,
            target_module=buffer0_experts,
            pin_cpu_memory=True,
        )

        layer.mlp.experts = self.shared_npu_experts
        if layer_idx == self.shared_npu_owner_layer:
            print(f"[Plugin] Layer {layer_idx} experts kept as shared NPU buffer0.")
            return

        self.offload_buffers.move_to_cpu_safely(experts)
        print(
            f"[Plugin] Layer {layer_idx} experts offloaded to CPU; "
            "forward uses shared NPU expert buffers."
        )

    def _patch_moe_forwards(self) -> None:
        for layer_idx, layer in self._iter_offload_layers():
            if layer_idx in self.moe_layer_indices:
                layer.mlp.forward = self._make_offloading_forward(layer_idx, layer.mlp)

    def _make_offloading_forward(self, layer_idx: int, mlp_module: nn.Module):
        if not hasattr(mlp_module, "gate"):
            raise RuntimeError(f"Layer {layer_idx} mlp has no gate module.")
        if not hasattr(mlp_module, "experts"):
            raise RuntimeError(f"Layer {layer_idx} mlp has no experts module.")

        def forward(hidden_states: torch.Tensor) -> torch.Tensor:
            expert_buffer = self._acquire_expert_buffer(layer_idx)
            self._prefetch_next_layer(layer_idx, avoid_buffer=expert_buffer)

            try:
                return self._run_mlp_with_buffer(
                    hidden_states=hidden_states,
                    mlp_module=mlp_module,
                    expert_buffer=expert_buffer,
                )
            finally:
                self._release_expert_buffer(expert_buffer)

        return forward

    def _prefetch_next_layer(
        self,
        layer_idx: int,
        avoid_buffer: Optional[ExpertBuffer],
    ) -> None:
        if self.prefetcher is None:
            return

        next_layer_idx = self.scheduler.next_prefetch_layer(
            self.offload_layer_indices,
            layer_idx,
        )
        self.prefetcher.prefetch(next_layer_idx, avoid=avoid_buffer)

    @staticmethod
    def _run_mlp_with_buffer(
        hidden_states: torch.Tensor,
        mlp_module: nn.Module,
        expert_buffer: ExpertBuffer,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = mlp_module.gate(flat_hidden_states)

        final_hidden_states = expert_buffer.experts(
            hidden_states=flat_hidden_states,
            router_logits=router_logits,
        )
        if mlp_module.ep_size > 1:
            final_hidden_states = (
                expert_buffer.experts
                .maybe_all_reduce_tensor_model_parallel(final_hidden_states)
            )

        return final_hidden_states.view(orig_shape)

    def _acquire_expert_buffer(self, layer_idx: int) -> ExpertBuffer:
        if layer_idx not in self.moe_layer_indices:
            raise RuntimeError(f"Layer {layer_idx} is not configured for offload.")
        if self.expert_buffer_pool is None:
            raise RuntimeError("Layer-wise expert buffer pool is not initialized.")

        slot = self.expert_slots.get(layer_idx)
        if slot is None:
            raise RuntimeError(f"Missing expert offload slot for layer {layer_idx}.")

        return self.expert_buffer_pool.acquire_for_compute(layer_idx, slot)

    def _release_expert_buffer(self, expert_buffer: ExpertBuffer) -> None:
        if self.expert_buffer_pool is not None:
            self.expert_buffer_pool.release_compute(expert_buffer)

    def summary(self) -> dict:
        return {
            "mode": "layer_wise",
            "offload_layers": list(self.offload_layer_indices),
            "num_buffers": 0 if self.expert_buffer_pool is None else len(self.expert_buffer_pool),
            "cpu_store_bytes": sum(slot.nbytes() for slot in self.expert_slots.values()),
            "prefetch_distance": self.config.prefetch_distance,
            "offload_interval": self.config.offload_interval,
        }
