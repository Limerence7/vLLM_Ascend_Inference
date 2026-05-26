import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import pandas as pd
import csv
import random
import math

from vllm import LLM, SamplingParams

# MODEL_PATH = "/workspace/models/Qwen3-235B-A22B"
# batch_size = 128
# max_length = 64
# max_new_tokens = 64
# max_model_len = max_length + max_new_tokens
# world_size = 8
# utilization = 0.98

MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Qwen3-30B-A3B")
batch_size = int(os.getenv("BATCH_SIZE", "1024"))
max_length = int(os.getenv("MAX_LENGTH", "256"))
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
max_model_len = max_length + max_new_tokens
world_size = int(os.getenv("WORLD_SIZE", "2"))
utilization = float(os.getenv("UTILIZATION", "0.85"))
jsonl_path = os.getenv(
    "DATASET_PATH",
    "/workspace/Huawei/datasets/computer_en_26k.jsonl",
)

# --------------------------
# 初始化引擎
# --------------------------
print("Loading model...")

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=max_new_tokens  # 不建议设成 32768，太大了，占内存，MoE可能爆
)

load_start = time.perf_counter()
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=world_size,
    enable_expert_parallel=True,
    trust_remote_code=True,
    gpu_memory_utilization=utilization,
    max_model_len=max_model_len,
    dtype="bfloat16", # 或 "float16"
    enforce_eager=True, # 建议开启，避免图编译带来的额外复杂性，方便调试
)
load_elapsed = time.perf_counter() - load_start

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
            if len(text) >= 512:
                contents.append(text)
    return contents


combined_list = load_contents_from_jsonl(jsonl_path)
batch_user_inputs = combined_list[:batch_size]
batch_user_inputs = [text[:max_length] for text in batch_user_inputs]

prompts = []
for prompt in batch_user_inputs:
    # 示例模板，根据模型实际格式调整
    chat_text = f"User: {prompt}\nAssistant:"
    prompts.append(chat_text)

generate_start = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
generate_elapsed = time.perf_counter() - generate_start
output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)

for output in outputs:
    prompt = output.prompt
    response = output.outputs[0].text
    print(f"Response is:\n {response}\n")
    break

print(
    "[Benchmark] "
    f"prompts={len(prompts)}, "
    f"output_tokens={output_tokens}, "
    f"load_seconds={load_elapsed:.3f}, "
    f"generate_seconds={generate_elapsed:.3f}, "
    f"output_tokens_per_second={output_tokens / generate_elapsed:.3f}"
)
