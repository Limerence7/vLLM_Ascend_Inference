import copy
import threading
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch_npu

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer


try:
    from vllm.model_executor.models.utils import sequence_parallel_chunk
except ImportError:
    def sequence_parallel_chunk(hidden_states):
        raise RuntimeError("sequence_parallel_chunk not available.")


@dataclass
class ExpertOffloadConfig:
    """
    Small-option config for Qwen3 MoE expert offloading.

    常用配置示例：
    - Qwen3-30B-A3B 默认：policy="every_n", every_n=16, num_buffers=2
    - Qwen3-235B-A22B 保守：policy="every_n", every_n=16, num_buffers=1, prefetch=False
    - Qwen3-235B-A22B 性能优先：policy="every_n", every_n=8, num_buffers=2, prefetch=True

    policy:
    - "every_n": 每 N 层 offload 一层；zero-based 选择结果为 N-1, 2N-1, ...
    - "explicit": 使用 explicit_layers，层号为 zero-based
    - "ratio": 按 available MoE layer 数量的比例均匀选择
    - "tail": 只选择最后 tail_n 个 MoE layer

    """

    enabled: bool = True
    policy: str = "every_n"
    every_n: int = 16
    explicit_layers: Optional[List[int]] = None
    ratio: float = 0.0
    tail_n: int = 0
    num_buffers: int = 2
    pin_cpu_memory: bool = True
    prefetch: bool = True
    keep_owner_on_npu: bool = True

    def validate(self) -> None:
        self.policy = self.policy.strip().lower()
        if self.policy not in {"every_n", "explicit", "ratio", "tail"}:
            raise ValueError(
                "ExpertOffloadConfig.policy must be one of: "
                "every_n, explicit, ratio, tail."
            )

        if self.every_n <= 0:
            raise ValueError("ExpertOffloadConfig.every_n must be > 0.")

        if self.num_buffers <= 0:
            raise ValueError("ExpertOffloadConfig.num_buffers must be > 0.")

        if self.policy == "explicit" and not self.explicit_layers:
            raise ValueError(
                "ExpertOffloadConfig.policy='explicit' requires explicit_layers."
            )

        if self.policy == "ratio" and not (0.0 < self.ratio <= 1.0):
            raise ValueError(
                "ExpertOffloadConfig.policy='ratio' requires 0 < ratio <= 1."
            )

        if self.policy == "tail" and self.tail_n <= 0:
            raise ValueError(
                "ExpertOffloadConfig.policy='tail' requires tail_n > 0."
            )

    def normalized_copy(self) -> "ExpertOffloadConfig":
        cfg = copy.copy(self)
        if cfg.explicit_layers is not None:
            cfg.explicit_layers = sorted(set(int(x) for x in cfg.explicit_layers))
        cfg.validate()
        return cfg


# =========================
# 内置小选项：直接修改这里
# =========================
# 默认更接近原始 Qwen3-30B-A3B 实现：每 16 层 offload 一层，双 buffer，开启预取。
# DEFAULT_EXPERT_OFFLOAD_CONFIG = ExpertOffloadConfig(
#     enabled=True,
#     policy="every_n",
#     every_n=16,
#     explicit_layers=None,
#     ratio=0.0,
#     tail_n=0,
#     num_buffers=2,
#     pin_cpu_memory=True,
#     prefetch=True,
#     keep_owner_on_npu=True,
# )

# 如果要跑 Qwen3-235B-A22B 且显存比较紧，可以改成类似：
# DEFAULT_EXPERT_OFFLOAD_CONFIG = ExpertOffloadConfig(
#     enabled=True,
#     policy="every_n",
#     every_n=16,
#     num_buffers=1,
#     pin_cpu_memory=True,
#     prefetch=False,
#     keep_owner_on_npu=True,
# )

# 如果要跑 Qwen3-235B-A22B 且希望更积极 offload / 预取，可以改成类似：
DEFAULT_EXPERT_OFFLOAD_CONFIG = ExpertOffloadConfig(
    enabled=True,
    policy="every_n",
    every_n=24,
    num_buffers=2,
    pin_cpu_memory=True,
    prefetch=True,
    keep_owner_on_npu=True,
)


class ExpertOffloadSlot:
    """
    One CPU pinned expert-weight slot for one offloaded MoE layer.
    CPU -> NPU copy is performed by ExpertBufferPool.
    """

    def __init__(
        self,
        layer_idx: int,
        experts_module: nn.Module,
        target_module: nn.Module,
        pin_cpu_memory: bool = True,
    ):
        self.layer_idx = layer_idx
        self.cpu_state = self._make_cpu_state(
            experts_module=experts_module,
            target_module=target_module,
            pin_cpu_memory=pin_cpu_memory,
        )

    @classmethod
    @torch.no_grad()
    def _make_cpu_state(
        cls,
        experts_module: nn.Module,
        target_module: nn.Module,
        pin_cpu_memory: bool,
    ) -> Dict[str, torch.Tensor]:
        cpu_state: Dict[str, torch.Tensor] = {}
        target_state = target_module.state_dict()

        for name, tensor in experts_module.state_dict().items():
            cpu_tensor = tensor.detach().cpu().contiguous()
            target_tensor = target_state.get(name)
            if target_tensor is None:
                raise RuntimeError(
                    f"Target expert buffer does not have state key {name}."
                )
            # Here we need to transpose the tensor in dim 1 and 2
            if cpu_tensor.ndim == 3:
                cpu_tensor = cpu_tensor.transpose(1, 2).contiguous()
            if pin_cpu_memory:
                cpu_tensor = cpu_tensor.pin_memory()

            cpu_state[name] = cpu_tensor
        return cpu_state


@dataclass
class ExpertBuffer:
    idx: int
    experts: nn.Module
    loaded_layer: Optional[int] = None
    loading_layer: Optional[int] = None
    ready_event: Optional[torch_npu.npu.Event] = None
    in_use: bool = False

    @property
    def loading(self) -> bool:
        return self.loading_layer is not None

    def reset(self) -> None:
        self.loaded_layer = None
        self.loading_layer = None
        self.ready_event = None
        self.in_use = False


class ExpertBufferPool:
    """
    Shared NPU expert buffer pool.

    State semantics:
    - loaded_layer: layer logically resident in the buffer.
    - loading_layer: CPU -> NPU copy has been issued but not finalized by compute stream.
    - ready_event: event recorded on the copy/prefetch stream.
    - in_use: buffer is currently used by one forward.
    """

    def __init__(
        self,
        buffers: List[ExpertBuffer],
        prefetch_stream: torch_npu.npu.Stream,
        enable_prefetch: bool = True,
    ):
        assert len(buffers) >= 1, "Length of buffers must be at least 1."
        self.buffers = buffers
        self.prefetch_stream = prefetch_stream
        self.enable_prefetch = enable_prefetch
        self.lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.buffers)

    def first_module(self) -> nn.Module:
        return self.buffers[0].experts

    def get_loaded_or_loading(self, layer_idx: int) -> Optional[ExpertBuffer]:
        for buf in self.buffers:
            if buf.loaded_layer == layer_idx or buf.loading_layer == layer_idx:
                return buf
        return None

    def get_idle_buffer(
        self,
        avoid: Optional[ExpertBuffer] = None,
    ) -> Optional[ExpertBuffer]:
        # Prefer completely empty buffers to preserve cache entries.
        for buf in self.buffers:
            if avoid is not None and buf.idx == avoid.idx:
                continue
            if not buf.in_use and not buf.loading and buf.loaded_layer is None:
                return buf

        # If no empty buffer exists, overwrite an old cached buffer.
        for buf in self.buffers:
            if avoid is not None and buf.idx == avoid.idx:
                continue
            if not buf.in_use and not buf.loading:
                return buf

        return None

    @torch.no_grad()
    def _copy_slot_to_buffer(
        self,
        slot: ExpertOffloadSlot,
        buf: ExpertBuffer,
        stream: torch_npu.npu.Stream,
    ) -> None:
        with torch_npu.npu.stream(stream):
            shared_state = buf.experts.state_dict()

            for name, cpu_tensor in slot.cpu_state.items():
                if name not in shared_state:
                    raise RuntimeError(
                        f"Expert buffer missing state key {name} for layer {slot.layer_idx}."
                    )
                target_tensor = shared_state[name]
                if tuple(target_tensor.shape) != tuple(cpu_tensor.shape):
                    raise RuntimeError(
                        f"Expert tensor shape mismatch for {name}, layer {slot.layer_idx}: "
                        f"cpu={tuple(cpu_tensor.shape)}, npu={tuple(target_tensor.shape)}."
                    )
                target_tensor.copy_(cpu_tensor, non_blocking=True)

            event = torch_npu.npu.Event()
            event.record(stream)

        buf.loaded_layer = slot.layer_idx
        buf.loading_layer = slot.layer_idx
        buf.ready_event = event

    def _wait_and_finalize(self, buf: ExpertBuffer) -> None:
        event = buf.ready_event
        loading_layer = buf.loading_layer

        if event is not None:
            torch_npu.npu.current_stream().wait_event(event)

        if loading_layer is not None:
            with self.lock:
                if buf.loading_layer == loading_layer:
                    buf.loading_layer = None

    def acquire_for_compute(
        self,
        layer_idx: int,
        slot: ExpertOffloadSlot,
    ) -> ExpertBuffer:
        """
        Return an NPU expert buffer loaded with the current layer.

        - If already prefetched, wait for the event.
        - If absent, choose an idle buffer and copy synchronously on the compute stream.
        """

        with self.lock:
            buf = self.get_loaded_or_loading(layer_idx)
            buf = self.get_idle_buffer()
            if buf is None:
                raise RuntimeError(
                    f"No free expert buffer for layer {layer_idx}. "
                    "Increase ExpertOffloadConfig.num_buffers or check "
                    "overlapping MoE forwards."
                )

            buf.reset()
            buf.loading_layer = layer_idx
            self._copy_slot_to_buffer(
                slot=slot,
                buf=buf,
                stream=torch_npu.npu.current_stream(),
            )
            buf.in_use = True

        self._wait_and_finalize(buf)
        return buf

    def release_compute(self, buf: ExpertBuffer) -> None:
        with self.lock:
            buf.in_use = False

    def prefetch(
        self,
        layer_idx: int,
        slot: ExpertOffloadSlot,
        avoid: Optional[ExpertBuffer] = None,
    ) -> bool:
        """
        Asynchronously prefetch layer_idx into a buffer different from the
        currently computing one when possible.
        """

        if not self.enable_prefetch or len(self.buffers) < 2:
            return False

        with self.lock:
            existing = self.get_loaded_or_loading(layer_idx)
            if existing is not None:
                return False

            buf = self.get_idle_buffer(avoid=avoid)
            if buf is None:
                return False

            buf.reset()
            buf.loading_layer = layer_idx

            try:
                self._copy_slot_to_buffer(
                    slot=slot,
                    buf=buf,
                    stream=self.prefetch_stream,
                )
            except Exception:
                buf.reset()

            return True


class MyQwen3Model(Qwen3MoeForCausalLM):
    """
    Qwen3 MoE shared expert offloading patch for Qwen3-30B-A3B and
    Qwen3-235B-A22B.

    Design points:
    - Do not replace layer.mlp.experts before vLLM load_weights.
    - Build CPU expert slots and shared NPU buffers after load_weights.
    - Do not call module.to(...) inside forward.
    - Do not deepcopy FusedMoE.
    - Additional buffers are shallow skeletons with independent NPU tensor storage.
    - CPU -> NPU only uses copy_ into existing NPU storage.
    - Model-specific differences are inferred from hf_text_config.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.config = vllm_config.model_config.hf_text_config
        self.num_layers = self.config.num_hidden_layers
        self.offload_config = DEFAULT_EXPERT_OFFLOAD_CONFIG.normalized_copy()

        self.moe_layer_indices: Set[int] = set()
        self.offload_layer_indices: List[int] = []
        self.expert_slots: Dict[int, ExpertOffloadSlot] = {}

        self.prefetch_stream = torch_npu.npu.Stream()

        self.expert_buffer_modules = nn.ModuleList()
        self.expert_buffer_pool: Optional[ExpertBufferPool] = None

        # Backward-compatible field names.
        self.shared_npu_experts: Optional[nn.Module] = None
        self.shared_npu_owner_layer: Optional[int] = None

        self._expert_offload_initialized = False
        self._moe_forward_patched = False

        self._print_model_summary("initializing")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded_weights = super().load_weights(weights)

        if self.offload_config.enabled:
            if not self._expert_offload_initialized:
                self._init_expert_offload_slots()
                self._expert_offload_initialized = True

            if not self._moe_forward_patched:
                self._patch_moe_forwards()
                self._moe_forward_patched = True
        else:
            print("[Plugin] Qwen3 expert offload disabled by config.")

        return loaded_weights

    def _print_model_summary(self, phase: str) -> None:
        cfg = self.config
        attrs = {
            "num_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "num_experts": cfg.num_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "moe_intermediate_size": cfg.moe_intermediate_size,
        }
        print(
            "[Plugin] MyQwen3Model "
            f"{phase}; model={attrs}; offload_config={self.offload_config}"
        )

    def _available_moe_layer_indices(self) -> List[int]:
        layers: List[int] = []
        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer, PPMissingLayer):
                continue
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                layers.append(layer_idx)
        return layers

    def _build_offload_layer_indices(self, available_layers: Sequence[int]) -> List[int]:
        cfg = self.offload_config
        available_set = set(available_layers)

        if not available_layers:
            return []

        if cfg.policy == "explicit":
            requested = cfg.explicit_layers or []
            selected = [idx for idx in requested if idx in available_set]

        elif cfg.policy == "ratio":
            target_count = max(1, int(round(len(available_layers) * cfg.ratio)))
            if target_count >= len(available_layers):
                selected = list(available_layers)
            else:
                step = len(available_layers) / target_count
                selected = []
                for i in range(target_count):
                    pos = min(len(available_layers) - 1, int(round((i + 1) * step)) - 1)
                    selected.append(available_layers[pos])

        elif cfg.policy == "tail":
            selected = list(available_layers[-cfg.tail_n:])

        else:  # every_n
            # Preserve the original zero-based behavior: every 16 selects
            # 15, 31, 47, ... rather than 16, 32, 48, ...
            selected = [idx for idx in available_layers if (idx + 1) % cfg.every_n == 0]

        selected = sorted(set(selected))
        return selected

    def _iter_offload_layers(self):
        selected = set(self.offload_layer_indices)
        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer, PPMissingLayer):
                continue
            if layer_idx in selected:
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

        if not self.offload_layer_indices:
            print(
                "[Plugin] No MoE layers selected for expert offloading. "
                f"available_layers={available_layers}, config={self.offload_config}"
            )
            return

        offload_layers = []
        for layer_idx in self.offload_layer_indices:
            layer = self.model.layers[layer_idx]
            if isinstance(layer, PPMissingLayer):
                continue
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
                continue
            offload_layers.append((layer_idx, layer))

        if not offload_layers:
            print("[Plugin] Selected offload layers do not contain valid MoE experts.")
            return

        owner_layer_idx, owner_layer = offload_layers[0]
        buffer0_experts = owner_layer.mlp.experts
        self._ensure_npu_module(buffer0_experts)

        self.shared_npu_experts = buffer0_experts
        self.shared_npu_owner_layer = owner_layer_idx

        self.expert_buffer_modules.append(buffer0_experts)
        buffers: List[ExpertBuffer] = [ExpertBuffer(idx=0, experts=buffer0_experts)]

        for buffer_idx in range(1, self.offload_config.num_buffers):
            cloned = self._clone_module_for_npu_buffer(buffer0_experts)
            # cloned = copy.copy(buffer0_experts)
            self._ensure_npu_module(cloned)
            self.expert_buffer_modules.append(cloned)
            buffers.append(ExpertBuffer(idx=buffer_idx, experts=cloned))

        self.expert_buffer_pool = ExpertBufferPool(
            buffers=buffers,
            prefetch_stream=self.prefetch_stream,
            enable_prefetch=self.offload_config.prefetch,
        )

        print(
            "[Plugin] Created shared NPU expert buffer pool: "
            f"num_buffers={len(buffers)}, selected_layers={self.offload_layer_indices}"
        )

        for layer_idx, layer in offload_layers:
            experts = layer.mlp.experts

            self.moe_layer_indices.add(layer_idx)
            self.expert_slots[layer_idx] = ExpertOffloadSlot(
                layer_idx=layer_idx,
                experts_module=experts,
                target_module=buffer0_experts,
                pin_cpu_memory=self.offload_config.pin_cpu_memory,
            )

            # load_weights has completed, so replacing modules here does not
            # interfere with vLLM weight loading / parameter enumeration.
            layer.mlp.experts = self.shared_npu_experts

            if layer_idx == owner_layer_idx and self.offload_config.keep_owner_on_npu:
                print(f"[Plugin] Layer {layer_idx} experts kept as shared NPU buffer0.")
            else:
                self._move_to_cpu_safely(experts)
                print(
                    f"[Plugin] Layer {layer_idx} experts offloaded to CPU; "
                    "forward uses shared NPU expert buffers."
                )
        torch_npu.npu.empty_cache()

    @classmethod
    def _clone_module_for_npu_buffer(cls, module: nn.Module) -> nn.Module:
        """
        Clone a structurally identical module with independent tensor storage.

        Do not use deepcopy:
        - FusedMoE / distributed modules may hold ProcessGroup objects.
        - deepcopy may try to pickle distributed runtime objects.

        This uses shallow copy:
        - Keep non-tensor Python attributes by reference, such as process groups,
          quant methods and parallel groups.
        - Recursively replace _parameters / _buffers / _modules.
        - Every parameter / buffer receives new NPU storage.
        """

        try:
            cloned = copy.copy(module)
        except Exception as exc:
            raise RuntimeError(
                "Failed to shallow-copy experts module for expert buffer. "
            ) from exc
        
        cloned._parameters = OrderedDict()
        cloned._buffers = OrderedDict()
        cloned._modules = OrderedDict()

        for name, param in module._parameters.items():
            if param is None:
                cloned._parameters[name] = None
                continue

            new_param = nn.Parameter(
                torch.empty_like(param, device=param.device),
                requires_grad=False,
            )
            cloned._parameters[name] = new_param

        for name, buf in module._buffers.items():
            if buf is None:
                cloned._buffers[name] = None
                continue

            cloned._buffers[name] = torch.empty_like(buf, device=buf.device)

        for name, child in module._modules.items():
            if child is None:
                cloned._modules[name] = None
            else:
                cloned._modules[name] = cls._clone_module_for_npu_buffer(child)

        if hasattr(module, "_non_persistent_buffers_set"):
            cloned._non_persistent_buffers_set = set(module._non_persistent_buffers_set)

        # Do not copy runtime hooks/state hooks.
        cloned._backward_pre_hooks = OrderedDict()
        cloned._backward_hooks = OrderedDict()
        cloned._forward_hooks = OrderedDict()
        cloned._forward_hooks_with_kwargs = OrderedDict()
        cloned._forward_hooks_always_called = OrderedDict()
        cloned._forward_pre_hooks = OrderedDict()
        cloned._forward_pre_hooks_with_kwargs = OrderedDict()
        cloned._state_dict_hooks = OrderedDict()
        cloned._state_dict_pre_hooks = OrderedDict()
        cloned._load_state_dict_pre_hooks = OrderedDict()
        cloned._load_state_dict_post_hooks = OrderedDict()

        return cloned

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
            if layer_idx not in self.moe_layer_indices:
                continue
            layer.mlp.forward = self._create_offloading_forward(layer_idx, layer.mlp)

    def _create_offloading_forward(self, layer_idx: int, mlp_module: nn.Module):
        model_self = self

        if not hasattr(mlp_module, "gate"):
            raise RuntimeError(f"Layer {layer_idx} mlp has no gate module.")
        if not hasattr(mlp_module, "experts"):
            raise RuntimeError(f"Layer {layer_idx} mlp has no experts module.")

        def forward(hidden_states: torch.Tensor) -> torch.Tensor:
            buf = model_self._acquire_expert_buffer(layer_idx)

            next_layer_idx = model_self._next_offload_layer(layer_idx)
            if next_layer_idx is not None and next_layer_idx != layer_idx:
                model_self._signal_prefetch_expert(
                    layer_idx=next_layer_idx,
                    avoid_buffer=buf,
                )

            try:
                orig_shape = hidden_states.shape
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                router_logits, _ = mlp_module.gate(hidden_states)

                final_hidden_states = buf.experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )
                if mlp_module.ep_size > 1:
                    final_hidden_states = (
                        buf.experts
                        .maybe_all_reduce_tensor_model_parallel(final_hidden_states)
                    )

                return final_hidden_states.view(orig_shape)

            finally:
                model_self._release_expert_buffer(buf)

        return forward

    def _acquire_expert_buffer(self, layer_idx: int) -> ExpertBuffer:
        if layer_idx not in self.moe_layer_indices:
            raise RuntimeError(f"Layer {layer_idx} is not configured for expert offload.")

        if self.expert_buffer_pool is None:
            raise RuntimeError("Expert offload buffer pool is not initialized.")

        slot = self.expert_slots.get(layer_idx)
        if slot is None:
            raise RuntimeError(f"Missing expert offload slot for layer {layer_idx}.")

        return self.expert_buffer_pool.acquire_for_compute(
            layer_idx=layer_idx,
            slot=slot,
        )

    def _release_expert_buffer(self, buf: ExpertBuffer) -> None:
        if self.expert_buffer_pool is not None:
            self.expert_buffer_pool.release_compute(buf)

    def _signal_prefetch_expert(
        self,
        layer_idx: int,
        avoid_buffer: Optional[ExpertBuffer] = None,
    ) -> None:
        if layer_idx not in self.moe_layer_indices or self.expert_buffer_pool is None:
            return

        slot = self.expert_slots.get(layer_idx)
        if slot is None:
            return

        self.expert_buffer_pool.prefetch(
            layer_idx=layer_idx,
            slot=slot,
            avoid=avoid_buffer,
        )
