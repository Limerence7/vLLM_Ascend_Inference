import src
from src.offload_config import OffloadConfig
from vllm import LLM, SamplingParams


TEST_CONFIG = {
    "model_path": "/workspace/models/Qwen3-30B-A3B",
    "batch_size": 1,
    "max_length": 32,
    "max_new_tokens": 4,
    "max_num_batched_tokens": 64,
    "max_num_seqs": 1,
    "world_size": 2,
    "utilization": 0.85,
}

OFFLOAD_CONFIG = OffloadConfig(
    mode="expert_wise",
    interval=8,
    num_buffers=2,
    num_hot_experts=0,
    cpu_pin_memory=True,
    offloaded_layer_ids=[],
)


def build_prompts(batch_size: int, max_length: int) -> list[str]:
    prompt = "User: Explain what expert offloading is in one sentence.\nAssistant:"
    prompt = prompt[:max_length]
    return [prompt for _ in range(batch_size)]


if __name__ == "__main__":
    src.register_plugin(OFFLOAD_CONFIG)

    print("Loading model...")
    print(f"Offload config: {OFFLOAD_CONFIG}")

    sampling_params = SamplingParams(
        temperature=0.0,
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
        enforce_eager=True,
    )

    outputs = llm.generate(
        build_prompts(TEST_CONFIG["batch_size"], TEST_CONFIG["max_length"]),
        sampling_params,
    )

    for output in outputs:
        print(f"Prompt:\n{output.prompt}\n")
        print(f"Response:\n{output.outputs[0].text}\n")
        break
