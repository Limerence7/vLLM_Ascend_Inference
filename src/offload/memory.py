import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch_npu


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
    - loading_layer: CPU -> NPU copy has been issued but not finalized by
      compute stream.
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
        for buf in self.buffers:
            if avoid is not None and buf.idx == avoid.idx:
                continue
            if not buf.in_use and not buf.loading and buf.loaded_layer is None:
                return buf

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
                        f"Expert buffer missing state key {name} for layer "
                        f"{slot.layer_idx}."
                    )
                target_tensor = shared_state[name]
                if tuple(target_tensor.shape) != tuple(cpu_tensor.shape):
                    raise RuntimeError(
                        f"Expert tensor shape mismatch for {name}, layer "
                        f"{slot.layer_idx}: cpu={tuple(cpu_tensor.shape)}, "
                        f"npu={tuple(target_tensor.shape)}."
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

        If prefetch already loaded the layer, reuse it and wait for its event.
        Otherwise copy the slot into an idle buffer on the current stream.
        """

        with self.lock:
            buf = self.get_loaded_or_loading(layer_idx)
            if buf is None:
                buf = self.get_idle_buffer()
                if buf is not None:
                    buf.reset()
                    buf.loading_layer = layer_idx
                    self._copy_slot_to_buffer(
                        slot=slot,
                        buf=buf,
                        stream=torch_npu.npu.current_stream(),
                    )

            if buf is None:
                raise RuntimeError(
                    f"No free expert buffer for layer {layer_idx}. "
                    "Increase ExpertOffloadConfig.num_buffers or check "
                    "overlapping MoE forwards."
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
