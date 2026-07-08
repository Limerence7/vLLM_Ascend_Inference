import argparse
import json
import src
from src.offline_scheduler import is_offline_scheduler_patch_enabled
from src.runtime_config import RuntimeConfig


def get_scheduler_class():
    try:
        from vllm.core.scheduler import Scheduler
    except ModuleNotFoundError:
        from vllm.v1.core.sched.scheduler import Scheduler
    return Scheduler


TEST_CONFIG = {
    "model_path": "/workspace/models/Qwen3-30B-A3B",
    "batch_size": 1024,
    "max_length": 1024,
    "max_new_tokens": 512,
    "world_size": 2,
    "utilization": 0.80,
}

RUNTIME_CONFIG = RuntimeConfig(
    runtime_mode="balance",
    interval=24,
    num_buffers=2,
    num_runtime_experts=0,
    num_experts_per_update = 2,
    runtime_layer_ids=[],
    enable_offline_scheduler=True,
    min_step_tokens=4000,
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
            if len(text) >= 2048:
                contents.append(text)
    return contents

def build_prompts(batch_size: int, max_length: int, tokenizer) -> list[str]:
    jsonl_path = "/workspace/Huawei/datasets/computer_en_26k.jsonl"
    combined_list = load_contents_from_jsonl(jsonl_path)
    batch_user_inputs = combined_list[:batch_size]

    encoded = tokenizer(
        batch_user_inputs,
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

    return batch_user_inputs


def check_patch_installed() -> None:
    src.register_plugin(RUNTIME_CONFIG)

    Scheduler = get_scheduler_class()
    if not is_offline_scheduler_patch_enabled():
        raise RuntimeError("Offline scheduler patch was not enabled.")
    if (getattr(Scheduler, "_vllm_ascend_offline_scheduler_mode", "")
            != "v1-marker" and not hasattr(Scheduler,
                                           "_schedule_prefills_offline")):
        raise RuntimeError("Scheduler._schedule_prefills_offline is missing.")
    if not getattr(Scheduler, "_vllm_ascend_offline_scheduler_enabled", False):
        raise RuntimeError("Scheduler patch marker is missing.")

    print("Offline scheduler patch is installed.")
    print(f"min_step_tokens: {RUNTIME_CONFIG.min_step_tokens}")


def run_model_smoke() -> None:
    from vllm import LLM, SamplingParams

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
    llm.llm_engine.engine_core.shutdown()
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="Load the model and run a tiny scheduler smoke test.",
    )
    args = parser.parse_args()

    check_patch_installed()
    if args.run_model:
        run_model_smoke()
