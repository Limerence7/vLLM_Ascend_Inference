import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src as inference_plugin
from src.config import ExpertWiseConfig, LayerWiseConfig, OffloadConfig
from src.utils import print_offload_summary


# =========================
# 手动修改推理参数
# =========================

# MODEL_PATH = "/workspace/models/Qwen3-235B-A22B"
# BATCH_SIZE = 128
# MAX_LENGTH = 64
# MAX_NEW_TOKENS = 64
# MAX_MODEL_LEN = MAX_LENGTH + MAX_NEW_TOKENS
# WORLD_SIZE = 8
# UTILIZATION = 0.98

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Qwen3-30B-A3B")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "2"))
UTILIZATION = float(os.getenv("UTILIZATION", "0.85"))
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    "/workspace/Huawei/datasets/computer_en_26k.jsonl",
)
OFFLOAD_MODE = os.getenv("OFFLOAD_MODE", "layer_wise")
PREFETCH_DISTANCE = int(os.getenv("PREFETCH_DISTANCE", "16"))
OFFLOAD_INTERVAL = int(os.getenv("OFFLOAD_INTERVAL", "16"))


def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# OFFLOAD_CONFIG = OffloadConfig(
#     mode="expert_wise",
#     overlap=True,
#     prefetch_distance=16,
#     offload_interval=16,
#     layer_wise=LayerWiseConfig(cpu_cache_size=1, async_prefetch=False),
#     expert_wise=ExpertWiseConfig(
#         resident_experts=124,
#         offload_multiple=4,
#         enable_prediction=True,
#         on_demand_load=True,
#         npu_cache_capacity=4,
#         pin_cpu_memory=True,
#         keep_loaded_on_npu=True,
#         compact_npu_cache=True,
#         large_batch_active_ratio=0.8,
#         log_transfers=False,
#         max_transfer_logs=96,
#     ),
# )

OFFLOAD_CONFIG = OffloadConfig(
    mode=OFFLOAD_MODE,
    overlap=env_bool("OVERLAP", True),
    prefetch_distance=PREFETCH_DISTANCE,
    offload_interval=OFFLOAD_INTERVAL,
    layer_wise=LayerWiseConfig(
        cpu_cache_size=int(os.getenv("LAYER_CPU_CACHE_SIZE", "2")),
        async_prefetch=env_bool("LAYER_ASYNC_PREFETCH", True),
    ),
    expert_wise=ExpertWiseConfig(
        resident_experts=int(os.getenv("RESIDENT_EXPERTS", "124")),
        partition_scope=os.getenv("PARTITION_SCOPE", "global"),
        offload_multiple=int(os.getenv("OFFLOAD_MULTIPLE", "4")),
        enable_prediction=env_bool("ENABLE_PREDICTION", True),
        on_demand_load=env_bool("ON_DEMAND_LOAD", True),
        npu_cache_capacity=int(os.getenv("NPU_CACHE_CAPACITY", "4")),
        pin_cpu_memory=env_bool("PIN_CPU_MEMORY", True),
        keep_loaded_on_npu=env_bool("KEEP_LOADED_ON_NPU", True),
        compact_npu_cache=env_bool("COMPACT_NPU_CACHE", True),
        skip_no_shrink_compact=env_bool("SKIP_NO_SHRINK_COMPACT", True),
        enable_large_batch_fast_path=env_bool("ENABLE_LARGE_BATCH_FAST_PATH", True),
        enable_single_select_forward=env_bool("ENABLE_SINGLE_SELECT_FORWARD", True),
        large_batch_active_ratio=float(os.getenv("LARGE_BATCH_ACTIVE_RATIO", "0.8")),
        log_transfers=env_bool("LOG_TRANSFERS", False),
        max_transfer_logs=int(os.getenv("MAX_TRANSFER_LOGS", "96")),
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
    return [f"User: {text[:max_length]}\nAssistant:" for text in samples]


def main():
    inference_plugin.register_plugin(OFFLOAD_CONFIG)

    from vllm import LLM, SamplingParams

    max_model_len = MAX_LENGTH + MAX_NEW_TOKENS
    print(f"Loading model with offload_config={OFFLOAD_CONFIG}...")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
    )

    load_start = time.perf_counter()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=WORLD_SIZE,
        enable_expert_parallel=True,
        trust_remote_code=True,
        gpu_memory_utilization=UTILIZATION,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enforce_eager=env_bool("ENFORCE_EAGER", True),
    )
    load_elapsed = time.perf_counter() - load_start

    prompts = build_prompts(DATASET_PATH, BATCH_SIZE, MAX_LENGTH)
    generate_start = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params,
    )
    generate_elapsed = time.perf_counter() - generate_start
    output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

    print(f"Response is:\n {outputs[0].outputs[0].text}\n")
    print(
        "[Benchmark] "
        f"prompts={len(prompts)}, "
        f"output_tokens={output_tokens}, "
        f"load_seconds={load_elapsed:.3f}, "
        f"generate_seconds={generate_elapsed:.3f}, "
        f"output_tokens_per_second={output_tokens / generate_elapsed:.3f}"
    )
    print_offload_summary(llm)


if __name__ == "__main__":
    main()
