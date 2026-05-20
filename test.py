import os
import sys
import json
import random

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
        npu_cache_capacity=64,
        pin_cpu_memory=True,
        prefetch=True,
        overlap=False,
        num_copy_streams=1,
        keep_loaded_on_npu=True,
        compact_npu_cache=True,
        log_transfers=False,
        max_transfer_logs=96,
    ),
)

def load_contents_from_jsonl(jsonl_path):
    contents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            conversation = data.get("conversation")
            if conversation is None:
                continue

            text = "".join(
                item.get("human", "") + item.get("assistant", "")
                for item in conversation
            )
            if len(text) >= 512:
                contents.append(text)
    return contents


def build_prompts(jsonl_path, batch_size, max_length):
    samples = load_contents_from_jsonl(jsonl_path)[:batch_size]
    random.shuffle(samples)
    return [f"User: {text[:max_length]}\nAssistant:" for text in samples]

def main():
    inference_plugin.register_plugin(OFFLOAD_CONFIG)

    from vllm import LLM, SamplingParams
    
    MODEL_PATH = "/workspace/models/Qwen3-30B-A3B"
    BATCH_SIZE = 1024
    MAX_LENGTH = 256
    MAX_NEW_TOKENS = 256
    WORLD_SIZE = 2
    UTILIZATION = 0.85
    DATASET_PATH = "/workspace/Huawei/datasets/computer_en_26k.jsonl"
    max_model_len = MAX_LENGTH + MAX_NEW_TOKENS

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=WORLD_SIZE,
        enable_expert_parallel=True,
        trust_remote_code=True,
        gpu_memory_utilization=UTILIZATION,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enforce_eager=True,
    )

    outputs = llm.generate(
        build_prompts(DATASET_PATH, BATCH_SIZE, MAX_LENGTH),
        sampling_params,
    )
    print(f"Response is:\n {outputs[0].outputs[0].text}\n")


if __name__ == "__main__":
    main()
