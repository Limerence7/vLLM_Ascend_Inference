import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src as inference_plugin
from src.config import ExpertWiseConfig, LayerWiseConfig, OffloadConfig
from src.utils import print_offload_summary


MODEL_PATH = "/workspace/models/Qwen3-30B-A3B"

OFFLOAD_CONFIG = OffloadConfig(
    mode="expert_wise",
    overlap=True,
    prefetch_distance=1,
    offload_interval=1,
    layer_wise=LayerWiseConfig(cpu_cache_size=1, async_prefetch=False),
    expert_wise=ExpertWiseConfig(
        resident_experts=32,
        offload_multiple=4,
        enable_prediction=True,
        on_demand_load=True,
        npu_cache_capacity=0,
        pin_cpu_memory=True,
        keep_loaded_on_npu=True,
        compact_npu_cache=False,
        large_batch_active_ratio=0.8,
        log_transfers=True,
        max_transfer_logs=64,
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
        ["User: Write one short sentence about expert offloading.\nAssistant:"],
        SamplingParams(temperature=0.0, max_tokens=1),
    )
    print(outputs[0].outputs[0].text)
    print_offload_summary(llm)


if __name__ == "__main__":
    main()
