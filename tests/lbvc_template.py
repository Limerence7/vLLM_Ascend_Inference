import time
import src
import json
import torch
import torch_npu

from pathlib import Path
from src.runtime_config import RuntimeConfig
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

RUNTIME_CONFIG = RuntimeConfig(
    runtime_mode="balance",
    interval=4,
    num_buffers=2,
    num_runtime_experts=-4,
    cpu_pin_memory=True,
    runtime_layer_ids=[],
    load_history_path=None,
    enable_scheduler=False,
    enable_history_mapping=False,
)


def main():
    src.register_plugin(RUNTIME_CONFIG)
    print("Loading model...")
    print(f"Runtime config: {RUNTIME_CONFIG}")
    
    # Create an LLM.
    llm = LLM(
        model="/workspace/models/Qwen3-30B-A3B",
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "/workspace/Huawei/vLLM_Ascend_Inference/load_records/vllm_profile",
            "torch_profiler_with_stack": False,
        },
        enforce_eager=True,
    )

    llm.start_profile()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    llm.stop_profile()
    from torch_npu.profiler.profiler import analyse
    analyse("/workspace/Huawei/vLLM_Ascend_Inference/load_records/vllm_profile")

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)
        break

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()
