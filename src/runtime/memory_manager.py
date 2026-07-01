from typing import NamedTuple

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


class ColdExperts(nn.Module):
    """Reusable NPU buffer for one layer's cold expert weights."""

    def __init__(self, templates: dict[str, torch.Tensor], num_experts: int,
                 device: torch.device, use_w8a8: bool):
        super().__init__()
        self.num_experts = num_experts
        for name, template in templates.items():
            if use_w8a8 and name in ("w13_weight", "w2_weight"):
                self._register_w8a8_weight_list(name, template, num_experts,
                                                device)
            elif use_w8a8 and name == "w13_weight_scale":
                self._register_w8a8_scale_lists(name, template, num_experts,
                                                device, keep_original=False)
            elif use_w8a8 and name == "w2_weight_scale":
                self._register_w8a8_scale_lists(name, template, num_experts,
                                                device)
            else:
                self.register_parameter(
                    name,
                    nn.Parameter(
                        torch.empty((num_experts, *template.shape[1:]),
                                    dtype=template.dtype,
                                    device=device),
                        requires_grad=False,
                    ))
        self.load_stream: torch.npu.Stream | None = None

    def load_from_cpu(self, weights: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, source in weights.items():
                tensor_list = getattr(self, f"{name}_list", None)
                if tensor_list is None and hasattr(self, name):
                    getattr(self, name)[:source.size(0)].copy_(
                        source, non_blocking=True)
                elif tensor_list is not None:
                    for target, expert_source in zip(tensor_list, source):
                        target.copy_(expert_source, non_blocking=True)

                if name in ("w13_weight_scale", "w2_weight_scale"):
                    fp32_list = getattr(self, f"{name}_fp32_list", None)
                    if fp32_list is None:
                        getattr(self, f"{name}_fp32")[:source.size(0)].copy_(
                            source, non_blocking=True)
                    else:
                        for target, expert_source in zip(fp32_list, source):
                            target.copy_(expert_source, non_blocking=True)

    def wait(self) -> None:
        if self.load_stream is not None:
            torch.npu.current_stream().wait_stream(self.load_stream)

    def _register_w8a8_weight_list(self, name: str, template: torch.Tensor,
                                   num_experts: int,
                                   device: torch.device) -> None:
        tensors = []
        for _ in range(num_experts):
            tensor = torch.empty(template.shape[1:],
                                 dtype=template.dtype,
                                 device=device)
            tensor = torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_NZ)
            tensors.append(nn.Parameter(tensor, requires_grad=False))
        setattr(self, f"{name}_list", nn.ParameterList(tensors))

    def _register_w8a8_scale_lists(self, name: str, template: torch.Tensor,
                                   num_experts: int,
                                   device: torch.device,
                                   keep_original: bool = True) -> None:
        if keep_original:
            tensors = [
                nn.Parameter(torch.empty(template.shape[1:],
                                         dtype=template.dtype,
                                         device=device),
                             requires_grad=False)
                for _ in range(num_experts)
            ]
            setattr(self, f"{name}_list", nn.ParameterList(tensors))

        tensors = [
            nn.Parameter(torch.empty(template.shape[1:],
                                     dtype=torch.float32,
                                     device=device),
                         requires_grad=False)
            for _ in range(num_experts)
        ]
        setattr(self, f"{name}_fp32_list", nn.ParameterList(tensors))


class PreparedColdExperts(NamedTuple):
    experts: ColdExperts
    topk_ids: torch.Tensor | None
    mask: torch.Tensor | None


class SharedTensorFactory:
    """Create CPU tensors backed by process-shared file storage."""

    def __init__(self, root_dir: str, namespace: str | None):
        self.root_dir = Path(root_dir)
        self.namespace = namespace or self._default_namespace()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def empty(self, key: str, shape: tuple[int, ...],
              dtype: torch.dtype) -> torch.Tensor:
        path = self.root_dir / self.namespace / f"{key}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)

        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()

        lock_path = path.with_suffix(path.suffix + ".lock")
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                storage = torch.UntypedStorage.from_file(
                    str(path), True, nbytes)
            finally:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

        stride = self._contiguous_stride(shape)
        return torch.empty(0, dtype=dtype,
                           device="cpu").set_(storage, 0, shape, stride)

    @staticmethod
    def tensor_key(layer_id: int, tp_rank: int, name: str,
                   shape: tuple[int, ...], dtype: torch.dtype) -> str:
        raw_key = f"layer{layer_id}.tp{tp_rank}.{name}.{dtype}.{shape}"
        return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
        stride = []
        value = 1
        for dim in reversed(shape):
            stride.append(value)
            value *= int(dim)
        return tuple(reversed(stride))

    @staticmethod
    def _default_namespace() -> str:
        explicit = os.environ.get("VLLM_ASCEND_CPU_EXPERT_SHM_NAME")
        if explicit:
            return explicit
        cwd_hash = hashlib.sha1(os.getcwd().encode("utf-8")).hexdigest()[:12]
        return f"uid{os.getuid()}_{cwd_hash}"


class CpuExpertWeights:
    """CPU copy of one MoE layer's offloaded expert weights."""

    PARAM_NAMES = (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w13_weight_offset",
        "w2_weight_scale",
        "w2_weight_offset",
    )
    TENSOR_LISTS = {
        "w13_weight": "w13_weight_list",
        "w2_weight": "w2_weight_list",
        "w13_weight_scale": "w13_weight_scale_fp32_list",
        "w2_weight_scale": "w2_weight_scale_list",
    }

    def __init__(self, layer, expert_ids: list[int], cpu_pin_memory: bool,
                 shared_factory: SharedTensorFactory | None = None):
        self.layer_id = int(layer.moe_instance_id)
        self.tp_rank = int(layer.tp_rank)
        self.shared_factory = shared_factory
        self.expert_id_to_slot = {
            expert_id: slot
            for slot, expert_id in enumerate(expert_ids)
        }
        self.tensors = {
            name: self._empty_cpu_like(getattr(layer, name),
                                       len(expert_ids),
                                       cpu_pin_memory,
                                       name)
            for name in self.PARAM_NAMES
            if hasattr(layer, name)
        }

    def load_shard(self, param_name: str, local_expert_id: int,
                   shard_id: str, loaded_weight: torch.Tensor,
                   tp_rank: int) -> bool:
        if param_name not in self.tensors:
            return False

        slot = self.expert_id_to_slot[local_expert_id]
        expert_data = self.tensors[param_name][slot]

        if param_name.startswith("w2_") and shard_id == "w2":
            if param_name != "w2_weight":
                expert_data.copy_(loaded_weight)
                return True
            shard_dim = 1
            shard_size = expert_data.shape[shard_dim]
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size)
            expert_data.copy_(loaded_weight)
            return True

        if param_name.startswith("w13_") and shard_id in ("w1", "w3"):
            shard_dim = 0
            shard_size = expert_data.shape[shard_dim] // 2
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size)
            offset = 0 if shard_id == "w1" else shard_size
            expert_data.narrow(shard_dim, offset,
                               shard_size).copy_(loaded_weight)
            return True

        return False

    def process_after_loading(self, quant_method) -> None:
        quant_method.process_offloaded_weights(self.tensors)

    def get_weights(self,
                    expert_ids: list[int] | None = None
                    ) -> dict[str, torch.Tensor]:
        if expert_ids is None:
            return self.tensors

        slots = [self.expert_id_to_slot[expert_id]
                 for expert_id in expert_ids]
        slot_tensor = torch.tensor(slots, dtype=torch.long, device="cpu")
        return {
            name: tensor.index_select(0, slot_tensor)
            for name, tensor in self.tensors.items()
        }

    def copy_to_module(self, expert_ids: list[int], module,
                       target_slots: list[int]) -> None:
        for expert_id, target_slot in zip(expert_ids, target_slots):
            source_slot = self.expert_id_to_slot[expert_id]
            for name, tensor in self.tensors.items():
                self._copy_tensor(module, name, target_slot,
                                  tensor[source_slot])
                if name in ("w13_weight_scale", "w2_weight_scale"):
                    fp32_target = getattr(module, f"{name}_fp32", None)
                    if fp32_target is not None:
                        fp32_target[target_slot].copy_(
                            tensor[source_slot], non_blocking=True)

    @staticmethod
    def _copy_tensor(module, name: str, target_slot: int,
                     source: torch.Tensor) -> None:
        target = getattr(module, name, None)
        if target is not None:
            target[target_slot].copy_(source, non_blocking=True)
            return

        list_name = CpuExpertWeights.TENSOR_LISTS.get(name)
        if list_name is None or not hasattr(module, list_name):
            return
        getattr(module, list_name)[target_slot].copy_(source,
                                                      non_blocking=True)

    def _empty_cpu_like(
        self,
        template_tensor: torch.Tensor,
        num_experts: int,
        cpu_pin_memory: bool,
        name: str,
    ) -> torch.Tensor:
        shape = (num_experts, *template_tensor.shape[1:])
        if self.shared_factory is not None:
            key = self.shared_factory.tensor_key(
                self.layer_id, self.tp_rank, name, shape,
                template_tensor.dtype)
            return self.shared_factory.empty(key, shape,
                                             template_tensor.dtype).contiguous()

        cpu_tensor = torch.empty(shape,
                                 dtype=template_tensor.dtype,
                                 device="cpu")
        if cpu_pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.contiguous()


class ExpertMemoryManager(nn.Module):
    """Stores expert weights as contiguous per-layer CPU tensors."""

    def __init__(
        self,
        cpu_pin_memory: bool = True,
        share_cpu_experts: bool = False,
        shared_cpu_expert_dir: str = "/dev/shm/vllm_ascend_runtime",
        shared_cpu_expert_name: str | None = None,
    ):
        super().__init__()
        self.cpu_pin_memory = cpu_pin_memory
        self.shared_factory = (
            SharedTensorFactory(shared_cpu_expert_dir,
                                shared_cpu_expert_name)
            if share_cpu_experts else None)
        self.layers: dict[int, CpuExpertWeights] = {}

    def register_layer(self, layer_id: int, layer,
                       expert_ids: list[int]) -> None:
        self.layers[layer_id] = CpuExpertWeights(
            layer=layer,
            expert_ids=expert_ids,
            cpu_pin_memory=self.cpu_pin_memory,
            shared_factory=self.shared_factory,
        )

    def load_weight_shard(self, layer_id: int, param_name: str,
                          local_expert_id: int, shard_id: str,
                          loaded_weight: torch.Tensor, tp_rank: int) -> bool:
        return self.layers[layer_id].load_shard(param_name, local_expert_id,
                                                shard_id, loaded_weight,
                                                tp_rank)

    def process_layer_after_loading(self, layer_id: int,
                                    quant_method) -> None:
        self.layers[layer_id].process_after_loading(quant_method)

    def get_expert_weights(
        self,
        layer_id: int,
        expert_ids: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.layers[layer_id].get_weights(expert_ids)

    def copy_experts_to_module(
        self,
        layer_id: int,
        expert_ids: list[int],
        module,
        target_slots: list[int],
    ) -> None:
        self.layers[layer_id].copy_to_module(expert_ids, module,
                                             target_slots)
