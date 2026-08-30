from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class HcclCopyTask:
    source_rank: int
    source_slot: int
    target_rank: int
    target_slot: int
    expert_id: int


@dataclass(frozen=True)
class ExpertUpdateTask:
    target_slots: list[list[int]]
    copies: list[HcclCopyTask]


class ExpertUpdator:
    """Execute prepared expert weight update tasks with HCCL."""

    def transfer(self, layer, task: ExpertUpdateTask) -> None:
        self.transfer_many([(layer, task)])

    def transfer_many(
        self,
        layer_tasks: list[tuple[object, ExpertUpdateTask]],
    ) -> None:
        """Launch all selected layers in one batched P2P operation."""
        ops: list[dist.P2POp] = []
        send_tensors: list[torch.Tensor] = []
        local_bundles: list[
            tuple[object, int, dict[str, torch.Tensor]]] = []
        recv_bundles: list[
            tuple[object, int, dict[str, torch.Tensor]]] = []
        for layer, task in layer_tasks:
            for copy_task in task.copies:
                if (copy_task.source_rank == layer.ep_rank
                        and copy_task.target_rank == layer.ep_rank):
                    local_bundles.append((
                        layer,
                        copy_task.target_slot,
                        {
                            name: tensor.clone()
                            for name, tensor in self._expert_slot(
                                layer, copy_task.source_slot).items()
                        },
                    ))
                    continue
                if copy_task.target_rank == layer.ep_rank:
                    bundle = self._empty_expert_slot_like(layer)
                    recv_bundles.append(
                        (layer, copy_task.target_slot, bundle))
                    source_rank = self._global_rank(
                        layer, copy_task.source_rank)
                    ops.extend(
                        dist.P2POp(dist.irecv, tensor, source_rank)
                        for tensor in bundle.values())
                if copy_task.source_rank == layer.ep_rank:
                    target_rank = self._global_rank(
                        layer, copy_task.target_rank)
                    for tensor in self._expert_slot(
                            layer, copy_task.source_slot).values():
                        tensor = tensor.clone()
                        send_tensors.append(tensor)
                        ops.append(
                            dist.P2POp(dist.isend, tensor, target_rank))

        if ops:
            for request in dist.batch_isend_irecv(ops):
                request.wait()
        with torch.no_grad():
            for layer, target_slot, tensors in local_bundles:
                self._write_expert_slot(layer, target_slot, tensors)
            for layer, target_slot, tensors in recv_bundles:
                self._write_expert_slot(layer, target_slot, tensors)

    @classmethod
    def _empty_expert_slot_like(cls, layer) -> dict[str, torch.Tensor]:
        return {
            name: torch.empty_like(source)
            for name, source in cls._expert_slot(layer, 0).items()
        }

    @staticmethod
    def _expert_slot(layer, slot: int) -> dict[str, torch.Tensor]:
        names = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w13_weight_scale_fp32",
            "w13_weight_offset",
            "w2_weight_scale",
            "w2_weight_scale_fp32",
            "w2_weight_offset",
        )
        tensors: dict[str, torch.Tensor] = {}
        for name in names:
            tensor = getattr(layer, name, None)
            if tensor is not None:
                tensors[name] = tensor[slot]
            tensor_list = getattr(layer, f"{name}_list", None)
            if tensor_list is not None:
                tensors[f"{name}_list"] = tensor_list[slot]
        return tensors

    @staticmethod
    def _write_expert_slot(
        layer,
        target_slot: int,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        for name, source in tensors.items():
            if name.endswith("_list"):
                base_name = name[:-5]
                getattr(layer, name)[target_slot].copy_(source)
                fp32_tensor = getattr(layer, f"{base_name}_fp32", None)
                if fp32_tensor is not None:
                    fp32_tensor[target_slot].copy_(source)
            else:
                getattr(layer, name)[target_slot].copy_(source)

    @staticmethod
    def _global_rank(layer, ep_rank: int) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return int(ep_rank)
        group = getattr(layer.moe_config.ep_group, "device_group", None)
        if group is None:
            return int(ep_rank)
        return dist.get_process_group_ranks(group)[int(ep_rank)]
