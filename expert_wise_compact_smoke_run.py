import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src as inference_plugin
from src.offload.config import (
    ExpertOffloadConfig,
    ExpertWiseOffloadConfig,
    OffloadConfig,
)


MODEL_PATH = "/workspace/models/Qwen3-30B-A3B"

OFFLOAD_CONFIG = OffloadConfig(
    mode="expert_wise",
    layer_wise=ExpertOffloadConfig(enabled=False),
    expert_wise=ExpertWiseOffloadConfig(
        offloaded_experts={0: None},
        npu_cache_capacity=48,
        pin_cpu_memory=True,
        prefetch=False,
        overlap=False,
        num_copy_streams=1,
        keep_loaded_on_npu=True,
        compact_npu_cache=True,
        log_transfers=True,
        max_transfer_logs=96,
    ),
)


def main():
    inference_plugin.register_plugin(OFFLOAD_CONFIG)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=64,
        dtype="bfloat16",
        enforce_eager=True,
    )

    outputs = llm.generate(
        ["User: Write one short sentence about offloading.\nAssistant:"],
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
