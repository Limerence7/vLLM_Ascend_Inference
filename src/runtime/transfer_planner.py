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
class TransferPlan:
    target_slots: list[list[int]]
    copies: list[HcclCopyTask]


class TransferPlanner:
    """Keep stable local slots and transfer incoming experts with HCCL."""

    def plan(
        self,
        current_slots: list[list[int]],
        target_slots: list[list[int]],
    ) -> TransferPlan:
        aligned = [
            self._align_slots(current, target)
            for current, target in zip(current_slots, target_slots)
        ]
        locations = {
            expert_id: (rank, slot)
            for rank, rank_slots in enumerate(current_slots)
            for slot, expert_id in enumerate(rank_slots)
        }
        copies = [
            HcclCopyTask(*locations[expert_id], rank, slot, expert_id)
            for rank, rank_slots in enumerate(aligned)
            for slot, expert_id in enumerate(rank_slots)
            if current_slots[rank][slot] != expert_id
        ]
        return TransferPlan(aligned, copies)

    def transfer(self, layer, plan: TransferPlan) -> None:
        ops: list[dist.P2POp] = []
        send_tensors: list[torch.Tensor] = []
        recv_bundles: list[tuple[int, dict[str, torch.Tensor]]] = []
        for task in plan.copies:
            if task.target_rank == layer.ep_rank:
                bundle = self._empty_expert_slot_like(layer)
                recv_bundles.append((task.target_slot, bundle))
                ops.extend(
                    dist.P2POp(dist.irecv, tensor, task.source_rank)
                    for tensor in bundle.values())
            if task.source_rank == layer.ep_rank:
                for tensor in self._expert_slot(layer,
                                                task.source_slot).values():
                    tensor = tensor.clone()
                    send_tensors.append(tensor)
                    ops.append(
                        dist.P2POp(dist.isend, tensor, task.target_rank))

        for request in dist.batch_isend_irecv(ops):
            request.wait()
        with torch.no_grad():
            for target_slot, tensors in recv_bundles:
                self._write_expert_slot(layer, target_slot, tensors)

    @staticmethod
    def _align_slots(current: list[int], target: list[int]) -> list[int]:
        target_set = set(target)
        incoming = iter(expert_id for expert_id in target
                        if expert_id not in current)
        return [
            expert_id if expert_id in target_set else next(incoming)
            for expert_id in current
        ]

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
