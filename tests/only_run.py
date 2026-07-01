import src
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
    "batch_size": 1,
    "max_length": 32,
    "max_new_tokens": 8,
    "max_num_batched_tokens": 64,
    "max_num_seqs": 1,
    "world_size": 2,
    "utilization": 0.85,
}

# LOAD_HISTORY_PATH = (
#     Path(__file__).resolve().parents[1]
#     / "load_records"
#     / "only_run_235b_w8a8"
# )
# TEST_CONFIG = {
#     "model_path": "/workspace/models/Qwen3-235B-A22B-W8A8",
#     "batch_size": 1,
#     "max_length": 32,
#     "max_new_tokens": 8,
#     "max_num_batched_tokens": 64,
#     "max_num_seqs": 1,
#     "world_size": 8,
#     "utilization": 0.80,
# }

RUNTIME_CONFIG = RuntimeConfig(
    runtime_mode="balance",
    interval=16,
    num_buffers=2,
    num_runtime_experts=0,
    cpu_pin_memory=True,
    runtime_layer_ids=[],
    load_history_path=str(LOAD_HISTORY_PATH),
    enable_history_mapping=False,
)


def build_prompts(batch_size: int, max_length: int, tokenizer) -> list[str]:
    prompt = "User: Explain what expert offloading is in one sentence.\nAssistant:"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt = tokenizer.decode(token_ids[:max_length])
    return [prompt for _ in range(batch_size)]


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
        max_num_batched_tokens=TEST_CONFIG["max_num_batched_tokens"],
        max_num_seqs=TEST_CONFIG["max_num_seqs"],
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
        print(f"Prompt:\n{output.prompt}\n")
        print(f"Response:\n{output.outputs[0].text}\n")
        break
