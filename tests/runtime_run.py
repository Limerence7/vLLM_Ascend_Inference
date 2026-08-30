import src
import json
import random
from pathlib import Path

from src.runtime_config import RuntimeConfig
from vllm import LLM, SamplingParams


LOAD_HISTORY_PATH = (
    Path(__file__).resolve().parents[1]
    / "load_records"
    / "only_run_load_history"
)
TEST_CONFIG = {
    "model_path": "/workspace/models/Qwen3-30B-A3B",
    "batch_size": 1024,
    "max_length": 2048,
    "max_new_tokens": 128,
    "world_size": 4,
    "utilization": 0.85,
}

# LOAD_HISTORY_PATH = (
#     Path(__file__).resolve().parents[1]
#     / "load_records"
#     / "only_run_235b_w8a8"
# )
# TEST_CONFIG = {
#     "model_path": "/workspace/models/Qwen3-235B-A22B-W8A8",
#     "batch_size": 512,
#     "max_length": 1024,
#     "max_new_tokens": 512,
#     "world_size": 4,
#     "utilization": 0.98,
# }

# LOAD_HISTORY_PATH = (
#     Path(__file__).resolve().parents[1]
#     / "load_records"
#     / "only_run_235b"
# )
# TEST_CONFIG = {
#     "model_path": "/workspace/models/Qwen3-235B-A22B",
#     "batch_size": 512,
#     "max_length": 1024,
#     "max_new_tokens": 32,
#     "world_size": 8,
#     "utilization": 0.98,
# }

RUNTIME_CONFIG = RuntimeConfig(
    runtime_mode="balance",
    interval=12,
    num_buffers=2,
    num_runtime_experts=0,
    cpu_pin_memory=True,
    runtime_layer_ids=[],
    # load_history_path=str(LOAD_HISTORY_PATH),
    load_history_path=None,
    enable_load_collection=True,
    enable_offline_scheduler=False,
    enable_history_mapping=False,
    scheduler_policy="expert",
    scheduler_min_step_tokens=8192,
    scheduler_reorder_window=64,
    rebalance_min_step_tokens=16384,
)

def load_contents_from_jsonl(jsonl_path, tokenizer, batch_size, max_length):
    text = ""
    num_seqs = 0
    contents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            conversation = data.get('conversation', None)
            if conversation is None:
                continue
            for conver in conversation:
                human = conver.get("human", "")
                assistant = conver.get("assistant", "")
                text += (human + assistant)
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
            )
            if len(encoded["input_ids"]) >= max_length:
                contents.append(text)
                text = ""
                num_seqs += 1
            if num_seqs >= batch_size:
                break

    return contents


def build_prompts(batch_size: int, max_length: int, tokenizer) -> list[str]:
    jsonl_path = "/workspace/Huawei/datasets/computer_en_26k.jsonl"
    combined_list = load_contents_from_jsonl(
        jsonl_path,
        tokenizer,
        batch_size,
        max_length
    )

    encoded = tokenizer(
        combined_list,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=False,
    )

    batch_user_inputs = tokenizer.batch_decode(
        encoded["input_ids"],
        skip_special_tokens=True,
    )

    random.shuffle(batch_user_inputs)
    return batch_user_inputs


def save_load_history(llm: LLM) -> None:
    if not RUNTIME_CONFIG.load_history_path:
        print("Skip load stats save.")
        return

    save_results = llm.collective_rpc(
        "save_load_history",
        timeout=120,
    )
    if not all(result.get("saved") for result in save_results):
        raise RuntimeError(f"Failed to save load history: {save_results}")
    print(f"Load history worker results: {save_results}")
    print(f"Load history saved under: {LOAD_HISTORY_PATH}")


def shutdown_llm(llm: LLM) -> None:
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    src.register_plugin(RUNTIME_CONFIG)

    print("Loading model...")
    print(f"Runtime config: {RUNTIME_CONFIG}")

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
        worker_extension_cls="src.utils.RuntimeWorkerExtension",
    )

    outputs = llm.generate(
        build_prompts(
            TEST_CONFIG["batch_size"], 
            TEST_CONFIG["max_length"],
            llm.get_tokenizer(),
        ),
        sampling_params,
    )
    save_load_history(llm)
    shutdown_llm(llm)

    for output in outputs:
        print(f"Response:\n{output.outputs[0].text}\n")
        break
