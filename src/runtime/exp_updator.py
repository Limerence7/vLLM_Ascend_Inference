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
        ops: list[dist.P2POp] = []
        send_tensors: list[torch.Tensor] = []
        recv_bundles: list[tuple[int, dict[str, torch.Tensor]]] = []
        for copy_task in task.copies:
            if copy_task.target_rank == layer.ep_rank:
                bundle = self._empty_expert_slot_like(layer)
                recv_bundles.append((copy_task.target_slot, bundle))
                ops.extend(
                    dist.P2POp(dist.irecv, tensor, copy_task.source_rank)
                    for tensor in bundle.values())
            if copy_task.source_rank == layer.ep_rank:
                for tensor in self._expert_slot(
                        layer, copy_task.source_slot).values():
                    tensor = tensor.clone()
                    send_tensors.append(tensor)
                    ops.append(
                        dist.P2POp(dist.isend, tensor,
                                   copy_task.target_rank))

        if ops:
            for request in dist.batch_isend_irecv(ops):
                request.wait()
        with torch.no_grad():
            for target_slot, tensors in recv_bundles:
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
