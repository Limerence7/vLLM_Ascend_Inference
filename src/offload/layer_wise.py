import copy
from bisect import bisect_right
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch_npu
from vllm.model_executor.models.utils import PPMissingLayer

from .config import ExpertOffloadConfig
from .memory import ExpertBuffer, ExpertBufferPool, ExpertOffloadSlot


class LayerWiseOffloadController:
    """
    Layer-wise MoE expert offload controller for one Qwen3 model instance.

    The model wrapper owns registration and lifecycle. This controller owns
    layer selection, CPU expert slots, shared NPU expert buffers, prefetch, and
    patched MLP forwards.
    """

    def __init__(self, model: nn.Module, config: ExpertOffloadConfig):
        self.model = model
        self.config = config

        self.moe_layer_indices: Set[int] = set()
        self.offload_layer_indices: List[int] = []
        self.expert_slots: Dict[int, ExpertOffloadSlot] = {}

        self.prefetch_stream = torch_npu.npu.Stream()
        self.expert_buffer_modules = nn.ModuleList()
        self.expert_buffer_pool: Optional[ExpertBufferPool] = None

        self.shared_npu_experts: Optional[nn.Module] = None
        self.shared_npu_owner_layer: Optional[int] = None
        self.initialized = False
        self.forward_patched = False

    def setup_after_weight_loading(self) -> None:
        if not self.config.enabled:
            print("[Plugin] Qwen3 layer-wise expert offload disabled by config.")
            return

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

    def _build_offload_layer_indices(self, available_layers: Sequence[int]) -> List[int]:
        if not available_layers:
            return []

        selected = self._select_layer_indices(available_layers)
        return sorted(set(selected))

    def _select_layer_indices(self, available_layers: Sequence[int]) -> List[int]:
        cfg = self.config
        available_set = set(available_layers)

        if cfg.policy == "explicit":
            return [idx for idx in cfg.explicit_layers or [] if idx in available_set]

        if cfg.policy == "ratio":
            target_count = max(1, int(round(len(available_layers) * cfg.ratio)))
            if target_count >= len(available_layers):
                return list(available_layers)

            step = len(available_layers) / target_count
            return [
                available_layers[
                    min(len(available_layers) - 1, int(round((i + 1) * step)) - 1)
                ]
                for i in range(target_count)
            ]

        if cfg.policy == "tail":
            return list(available_layers[-cfg.tail_n:])

        return [idx for idx in available_layers if (idx + 1) % cfg.every_n == 0]

    def _iter_offload_layers(self) -> Iterator[Tuple[int, nn.Module]]:
        selected = set(self.offload_layer_indices)
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in selected and self._is_moe_layer(layer):
                yield layer_idx, layer

    def _next_offload_layer(self, layer_idx: int) -> Optional[int]:
        if not self.offload_layer_indices:
            return None

        pos = bisect_right(self.offload_layer_indices, layer_idx)
        if pos >= len(self.offload_layer_indices):
            pos = 0
        return self.offload_layer_indices[pos]

    def _init_expert_offload_slots(self) -> None:
        available_layers = self._available_moe_layer_indices()
        self.offload_layer_indices = self._build_offload_layer_indices(available_layers)
        offload_layers = list(self._iter_offload_layers())

        if not offload_layers:
            print(
                "[Plugin] No valid MoE layers selected for layer-wise offload; "
                f"available_layers={available_layers}, config={self.config}"
            )
            return

        owner_layer_idx, owner_layer = offload_layers[0]
        buffer0_experts = owner_layer.mlp.experts
        self._ensure_npu_module(buffer0_experts)
        self._create_expert_buffer_pool(buffer0_experts)

        self.shared_npu_experts = buffer0_experts
        self.shared_npu_owner_layer = owner_layer_idx

        for layer_idx, layer in offload_layers:
            self._register_offloaded_layer(layer_idx, layer, buffer0_experts)

        torch_npu.npu.empty_cache()

    def _create_expert_buffer_pool(self, buffer0_experts: nn.Module) -> None:
        self.expert_buffer_modules.append(buffer0_experts)
        buffers = [ExpertBuffer(idx=0, experts=buffer0_experts)]

        for buffer_idx in range(1, self.config.num_buffers):
            cloned = self._clone_module_for_npu_buffer(buffer0_experts)
            self._ensure_npu_module(cloned)
            self.expert_buffer_modules.append(cloned)
            buffers.append(ExpertBuffer(idx=buffer_idx, experts=cloned))

        self.expert_buffer_pool = ExpertBufferPool(
            buffers=buffers,
            prefetch_stream=self.prefetch_stream,
            enable_prefetch=self.config.prefetch,
        )
        print(
            "[Plugin] Created layer-wise expert buffer pool: "
            f"num_buffers={len(buffers)}, selected_layers={self.offload_layer_indices}"
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
            pin_cpu_memory=self.config.pin_cpu_memory,
        )

        layer.mlp.experts = self.shared_npu_experts
        if (
            layer_idx == self.shared_npu_owner_layer
            and self.config.keep_owner_on_npu
        ):
            print(f"[Plugin] Layer {layer_idx} experts kept as shared NPU buffer0.")
            return

        self._move_to_cpu_safely(experts)
        print(
            f"[Plugin] Layer {layer_idx} experts offloaded to CPU; "
            "forward uses shared NPU expert buffers."
        )

    @classmethod
    def _clone_module_for_npu_buffer(cls, module: nn.Module) -> nn.Module:
        try:
            cloned = copy.copy(module)
        except Exception as exc:
            raise RuntimeError(
                "Failed to shallow-copy experts module for expert buffer."
            ) from exc

        cloned._parameters = OrderedDict()
        cloned._buffers = OrderedDict()
        cloned._modules = OrderedDict()

        cls._clone_parameters(module, cloned)
        cls._clone_buffers(module, cloned)
        cls._clone_children(module, cloned)
        cls._clear_runtime_hooks(cloned)

        if hasattr(module, "_non_persistent_buffers_set"):
            cloned._non_persistent_buffers_set = set(module._non_persistent_buffers_set)

        return cloned

    @staticmethod
    def _clone_parameters(source: nn.Module, target: nn.Module) -> None:
        for name, param in source._parameters.items():
            if param is None:
                target._parameters[name] = None
            else:
                target._parameters[name] = nn.Parameter(
                    torch.empty_like(param, device=param.device),
                    requires_grad=False,
                )

    @staticmethod
    def _clone_buffers(source: nn.Module, target: nn.Module) -> None:
        for name, buffer in source._buffers.items():
            target._buffers[name] = (
                None
                if buffer is None
                else torch.empty_like(buffer, device=buffer.device)
            )

    @classmethod
    def _clone_children(cls, source: nn.Module, target: nn.Module) -> None:
        for name, child in source._modules.items():
            target._modules[name] = (
                None if child is None else cls._clone_module_for_npu_buffer(child)
            )

    @staticmethod
    def _clear_runtime_hooks(module: nn.Module) -> None:
        module._backward_pre_hooks = OrderedDict()
        module._backward_hooks = OrderedDict()
        module._forward_hooks = OrderedDict()
        module._forward_hooks_with_kwargs = OrderedDict()
        module._forward_hooks_always_called = OrderedDict()
        module._forward_pre_hooks = OrderedDict()
        module._forward_pre_hooks_with_kwargs = OrderedDict()
        module._state_dict_hooks = OrderedDict()
        module._state_dict_pre_hooks = OrderedDict()
        module._load_state_dict_pre_hooks = OrderedDict()
        module._load_state_dict_post_hooks = OrderedDict()

    @staticmethod
    def _ensure_npu_module(module: nn.Module) -> None:
        try:
            first_param = next(module.parameters())
            if first_param.device.type != "npu":
                module.to("npu")
        except StopIteration:
            module.to("npu")

        for param in module.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _move_to_cpu_safely(module: nn.Module) -> None:
        try:
            module.to("cpu")
        except Exception:
            pass

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
        next_layer_idx = self._next_offload_layer(layer_idx)
        if next_layer_idx is not None and next_layer_idx != layer_idx:
            self._signal_prefetch_expert(next_layer_idx, avoid_buffer=avoid_buffer)

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
            raise RuntimeError(f"Layer {layer_idx} is not configured for expert offload.")
        if self.expert_buffer_pool is None:
            raise RuntimeError("Expert offload buffer pool is not initialized.")

        slot = self.expert_slots.get(layer_idx)
        if slot is None:
            raise RuntimeError(f"Missing expert offload slot for layer {layer_idx}.")

        return self.expert_buffer_pool.acquire_for_compute(layer_idx, slot)

    def _release_expert_buffer(self, expert_buffer: ExpertBuffer) -> None:
        if self.expert_buffer_pool is not None:
            self.expert_buffer_pool.release_compute(expert_buffer)

    def _signal_prefetch_expert(
        self,
        layer_idx: int,
        avoid_buffer: Optional[ExpertBuffer] = None,
    ) -> None:
        if layer_idx not in self.moe_layer_indices or self.expert_buffer_pool is None:
            return

        slot = self.expert_slots.get(layer_idx)
        if slot is not None:
            self.expert_buffer_pool.prefetch(layer_idx, slot, avoid=avoid_buffer)
