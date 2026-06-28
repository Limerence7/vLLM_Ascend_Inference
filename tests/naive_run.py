import sys
import os
import json
import pandas as pd
import csv
import random
import math

from vllm import LLM, SamplingParams

Inference_Config = {
    "Qwen3-30B-A3B": {
        "model_path": "/workspace/models/Qwen3-30B-A3B",
        "batch_size": 1024,
        "max_length": 256,
        "max_new_tokens": 256,
        "world_size": 2,
        "utilization": 0.85,
    },
    "Qwen3-235B-A22B": {
        "model_path": "/workspace/models/Qwen3-235B-A22B",
        "batch_size": 512,
        "max_length": 1024,
        "max_new_tokens": 32,
        "world_size": 8,
        "utilization": 0.98,
    },
    "Qwen3-235B-A22B-W8A8": {
        "model_path": "/workspace/models/Qwen3-235B-A22B-W8A8",
        "batch_size": 512,
        "max_length": 2560,
        "max_new_tokens": 2560,
        "world_size": 8,
        "utilization": 0.80,
    },
}

current_config = Inference_Config["Qwen3-235B-A22B-W8A8"]

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
    # random.shuffle(contents)
    return contents

if __name__ == "__main__":
    
    print("Loading model...")

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=current_config["max_new_tokens"]
    )

    llm = LLM(
        model=current_config["model_path"],
        tensor_parallel_size=current_config["world_size"],
        enable_expert_parallel=True,
        trust_remote_code=True,
        gpu_memory_utilization=current_config["utilization"],
        max_model_len=current_config["max_length"] + current_config["max_new_tokens"],
        # dtype="bfloat16",
        quantization='ascend',
        enforce_eager=True,
        
    )

    jsonl_path = '/workspace/Huawei/datasets/computer_en_26k.jsonl'
    combined_list = load_contents_from_jsonl(jsonl_path)
    batch_user_inputs = combined_list[:current_config["batch_size"]]
    batch_user_inputs = [text[:current_config["max_length"]] for text in batch_user_inputs]

    prompts = []
    for prompt in batch_user_inputs:
        chat_text = f"User: {prompt}\nAssistant:"
        prompts.append(chat_text)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        print(f"Response is:\n {response}\n")
        break

