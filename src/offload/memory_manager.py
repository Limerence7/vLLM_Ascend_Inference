import torch


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

    def __init__(self, layer, expert_ids: list[int], cpu_pin_memory: bool):
        expert_ids = list(expert_ids)
        self.expert_id_to_slot = {
            expert_id: slot
            for slot, expert_id in enumerate(expert_ids)
        }
        self.tensors = {
            name: self._empty_cpu_like(getattr(layer, name),
                                       len(expert_ids),
                                       cpu_pin_memory)
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
                       target_slots: list[int] | None = None) -> None:
        if target_slots is None:
            target_slots = range(len(expert_ids))

        for expert_id, target_slot in zip(expert_ids, target_slots):
            source_slot = self.expert_id_to_slot[expert_id]
            for name, tensor in self.tensors.items():
                target = getattr(module, name)
                target[target_slot].copy_(tensor[source_slot],
                                          non_blocking=True)
                if name in ("w13_weight_scale", "w2_weight_scale"):
                    fp32_target = getattr(module, f"{name}_fp32", None)
                    if fp32_target is not None:
                        fp32_target[target_slot].copy_(
                            tensor[source_slot], non_blocking=True)

    @staticmethod
    def _empty_cpu_like(
        template_tensor: torch.Tensor,
        num_experts: int,
        cpu_pin_memory: bool
    ) -> torch.Tensor:
        cpu_tensor = torch.empty(
            (num_experts, *template_tensor.shape[1:]),
            dtype=template_tensor.dtype,
            device="cpu"
        )
        if cpu_pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.contiguous()


class ExpertMemoryManager:
    """Stores expert weights as contiguous per-layer CPU tensors."""

    def __init__(self, cpu_pin_memory: bool = True):
        self.cpu_pin_memory = cpu_pin_memory
        self.layers: dict[int, CpuExpertWeights] = {}

    def register_layer(self, layer_id: int, layer,
                       expert_ids: list[int]) -> None:
        self.layers[layer_id] = CpuExpertWeights(
            layer=layer,
            expert_ids=expert_ids,
            cpu_pin_memory=self.cpu_pin_memory,
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
        target_slots: list[int] | None = None,
    ) -> None:
        self.layers[layer_id].copy_to_module(expert_ids, module,
                                             target_slots)
