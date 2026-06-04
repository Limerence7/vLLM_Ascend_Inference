import torch


class CpuExpertWeights:
    """CPU copy of one MoE layer's cold expert weights."""

    def __init__(self, w13_weight: torch.Tensor, w2_weight: torch.Tensor,
                 cold_expert_ids: list[int], cpu_pin_memory: bool):
        self.processed = False
        self.cold_expert_ids = [int(expert_id) for expert_id in cold_expert_ids]
        self.expert_id_to_slot = {
            expert_id: slot
            for slot, expert_id in enumerate(self.cold_expert_ids)
        }
        self.w13_weight = self._empty_cpu_like(
            w13_weight, len(self.cold_expert_ids), cpu_pin_memory)
        self.w2_weight = self._empty_cpu_like(
            w2_weight, len(self.cold_expert_ids), cpu_pin_memory)

    def load_shard(self, weight_name: str, local_expert_id: int,
                   shard_id: str, loaded_weight: torch.Tensor,
                   tp_rank: int) -> bool:
        slot = self.expert_id_to_slot[int(local_expert_id)]
        loaded_weight = loaded_weight.detach().cpu()

        if weight_name == "w2_weight" and shard_id == "w2":
            expert_data = self.w2_weight[slot]
            shard_dim = 1
            shard_size = expert_data.shape[shard_dim]
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size)
            expert_data.copy_(loaded_weight)
            return True

        if weight_name == "w13_weight" and shard_id in ("w1", "w3"):
            expert_data = self.w13_weight[slot]
            shard_dim = 0
            shard_size = expert_data.shape[shard_dim] // 2
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size)
            offset = 0 if shard_id == "w1" else shard_size
            expert_data.narrow(shard_dim, offset,
                               shard_size).copy_(loaded_weight)
            return True

        return False

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.w13_weight, self.w2_weight

    def process_after_loading(self, quant_method) -> None:
        if self.processed:
            return
        self.w13_weight = quant_method._maybe_pad_weight(
            self.w13_weight).transpose(1, 2).contiguous()
        self.w2_weight = quant_method._maybe_pad_weight(
            self.w2_weight).transpose(1, 2).contiguous()
        self.processed = True

    def _empty_cpu_like(self, tensor: torch.Tensor, num_experts: int,
                        cpu_pin_memory: bool) -> torch.Tensor:
        cpu_tensor = torch.empty((num_experts, *tensor.shape[1:]),
                                 dtype=tensor.dtype,
                                 device="cpu")
        if cpu_pin_memory:
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.contiguous()


class ExpertMemoryManager:
    """Stores cold expert weights as contiguous per-layer CPU tensors."""

    def __init__(self, cpu_pin_memory: bool = True):
        self.cpu_pin_memory = cpu_pin_memory
        self.layers: dict[int, CpuExpertWeights] = {}

    def register_layer(self, layer_id: int, w13_weight: torch.Tensor,
                       w2_weight: torch.Tensor,
                       cold_expert_ids: list[int]) -> CpuExpertWeights | None:
        if not cold_expert_ids:
            return None
        self.layers[layer_id] = CpuExpertWeights(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            cold_expert_ids=cold_expert_ids,
            cpu_pin_memory=self.cpu_pin_memory,
        )
        return self.layers[layer_id]

    def load_weight_shard(self, layer_id: int, weight_name: str,
                          local_expert_id: int, shard_id: str,
                          loaded_weight: torch.Tensor, tp_rank: int) -> bool:
        return self.layers[layer_id].load_shard(weight_name, local_expert_id,
                                                shard_id, loaded_weight,
                                                tp_rank)

    def process_layer_after_loading(self, layer_id: int,
                                    quant_method) -> None:
        self.layers[layer_id].process_after_loading(quant_method)

    def get_expert_weights(self, layer_id: int) -> tuple[
            torch.Tensor, torch.Tensor]:
        return self.layers[layer_id].weights()

    def summary(self) -> dict[str, object]:
        return {
            "num_layers": len(self.layers),
        }
