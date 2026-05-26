import torch_npu


def create_npu_stream():
    return torch_npu.npu.Stream()
