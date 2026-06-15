import src
import json
import random
from src.offload_config import OffloadConfig
from vllm import LLM, SamplingParams


TEST_CONFIG = {
    "model_path": "/workspace/models/Qwen3-30B-A3B",
    "batch_size": 256,
    "max_length": 256,
    "max_new_tokens": 256,
    "world_size": 2,
    "utilization": 0.85,
}

OFFLOAD_CONFIG = OffloadConfig(
    mode="manual",
    interval=16,
    num_buffers=2,
    num_hot_experts=56,
    cpu_pin_memory=True,
    offloaded_layer_ids=[],
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
    batch_user_inputs = combined_list[:TEST_CONFIG["batch_size"]]
    batch_user_inputs = [text[:TEST_CONFIG["max_length"]] for text in batch_user_inputs]
    
    return batch_user_inputs


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
