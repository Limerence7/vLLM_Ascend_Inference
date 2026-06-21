import src
import json
import random
from pathlib import Path

from src.offload_config import OffloadConfig
from vllm import LLM, SamplingParams


# LOAD_STATS_PATH = (
#     Path(__file__).resolve().parents[1]
#     / "load_records"
#     / "only_run_load_stats"
# )
LOAD_STATS_PATH = (
    Path(__file__).resolve().parents[1]
    / "load_records"
    / "qwen235"
)

# TEST_CONFIG = {
#     "model_path": "/workspace/models/Qwen3-30B-A3B",
#     "batch_size": 1024,
#     "max_length": 256,
#     "max_new_tokens": 256,
#     "world_size": 2,
#     "utilization": 0.85,
# }
TEST_CONFIG = {
    "model_path": "/workspace/models/Qwen3-235B-A22B",
    "batch_size": 512,
    "max_length": 64,
    "max_new_tokens": 64,
    "world_size": 8,
    "utilization": 0.98,
}

# OFFLOAD_CONFIG = OffloadConfig(
#     mode="manual",
#     interval=1,
#     num_buffers=2,
#     num_hot_experts=62,
#     cpu_pin_memory=True,
#     offloaded_layer_ids=[0, 24],
#     load_stats_path=str(LOAD_STATS_PATH),
#     load_balance_mode="dynamic",
#     dynamic_update_interval=64,
#     dynamic_max_swaps=1,
#     dynamic_min_swap_gain=32,
#     dynamic_cooldown_interval=1,
# )

OFFLOAD_CONFIG = OffloadConfig(
    mode="manual",
    interval=1,
    num_buffers=2,
    num_hot_experts=12,
    cpu_pin_memory=True,
    offloaded_layer_ids=[0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88],
    load_stats_path=str(LOAD_STATS_PATH),
    load_balance_mode="dynamic",
    dynamic_update_interval=32,
    dynamic_max_swaps=1,
    dynamic_min_swap_gain=32,
    dynamic_cooldown_interval=1,
)

def load_contents_from_jsonl(jsonl_path):
    contents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            conversation = data.get('conversation', None)
            if conversation is None:
                continue
            text = ""
            for conver in conversation:
                human = conver.get("human", "")
                assistant = conver.get("assistant", "")
                text += (human + assistant)
            if len(text) >= 1024:
                contents.append(text)
    random.shuffle(contents)
    return contents


def build_prompts(batch_size: int, max_length: int) -> list[str]:
    jsonl_path = '/workspace/Huawei/datasets/computer_en_26k.jsonl'
    combined_list = load_contents_from_jsonl(jsonl_path)
    batch_user_inputs = combined_list[:batch_size]
    batch_user_inputs = [text[:max_length] for text in batch_user_inputs]
    return batch_user_inputs


def save_load_stats(llm: LLM) -> None:
    if (OFFLOAD_CONFIG.load_balance_mode != "none"
            or not OFFLOAD_CONFIG.load_stats_path):
        print("Skip load stats save.")
        return

    save_results = llm.collective_rpc(
        "save_load_stats",
        timeout=120,
    )
    if not all(result.get("saved") for result in save_results):
        raise RuntimeError(f"Failed to save load stats: {save_results}")
    print(f"Load stats worker results: {save_results}")
    print(f"Load stats saved under: {LOAD_STATS_PATH}")


def shutdown_llm(llm: LLM) -> None:
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    src.register_plugin(OFFLOAD_CONFIG)

    print("Loading model...")
    print(f"Offload config: {OFFLOAD_CONFIG}")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=TEST_CONFIG["max_new_tokens"],
    )

    llm = LLM(
        model=TEST_CONFIG["model_path"],
        tensor_parallel_size=TEST_CONFIG["world_size"],
        enable_expert_parallel=True,
        trust_remote_code=True,
        gpu_memory_utilization=TEST_CONFIG["utilization"],
        max_model_len=TEST_CONFIG["max_length"] + TEST_CONFIG["max_new_tokens"],
        dtype="bfloat16",
        # quantization='ascend',
        enforce_eager=True,
        worker_extension_cls="src.utils.OffloadWorkerExtension",
    )

    outputs = llm.generate(
        build_prompts(TEST_CONFIG["batch_size"], TEST_CONFIG["max_length"]),
        sampling_params,
    )
    save_load_stats(llm)
    shutdown_llm(llm)

    for output in outputs:
        print(f"Prompt:\n{output.prompt}\n")
        print(f"Response:\n{output.outputs[0].text}\n")
        break
