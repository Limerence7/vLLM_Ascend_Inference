from setuptools import find_packages, setup

setup(
    name="vLLM-Ascend-Inference-Plugin",
    version="1.0",
    packages=find_packages(),
    entry_points={"vllm.plugins": ["inference_plugin = src:register_plugin"]},
)
