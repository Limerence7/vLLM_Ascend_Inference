import copy
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch_npu

from ..config import OffloadConfig
from ..utils import (
    ExpertBuffer,
    ExpertBufferPool,
    clear_runtime_hooks,
)


class LayerWiseOffloadBuffers:
    def __init__(self, config: OffloadConfig):
        self.config = config
        self.prefetch_stream = torch_npu.npu.Stream()
        self.buffer_modules = nn.ModuleList()
        self.pool = None

    def create_pool(self, template_experts: nn.Module) -> ExpertBufferPool:
        self._ensure_npu_module(template_experts)
        self.buffer_modules.append(template_experts)
        buffers: List[ExpertBuffer] = [ExpertBuffer(idx=0, experts=template_experts)]

        for buffer_idx in range(1, self.config.layer_wise.cpu_cache_size):
            cloned = self.clone_module_for_npu_buffer(template_experts)
            self._ensure_npu_module(cloned)
            self.buffer_modules.append(cloned)
            buffers.append(ExpertBuffer(idx=buffer_idx, experts=cloned))

        self.pool = ExpertBufferPool(
            buffers=buffers,
            prefetch_stream=self.prefetch_stream,
            enable_prefetch=self.config.layer_wise.async_prefetch,
        )
        return self.pool

    @classmethod
    def clone_module_for_npu_buffer(cls, module: nn.Module) -> nn.Module:
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
        clear_runtime_hooks(cloned)

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
                None if child is None else cls.clone_module_for_npu_buffer(child)
            )

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
    def move_to_cpu_safely(module: nn.Module) -> None:
        try:
            module.to("cpu")
        except Exception:
            pass
