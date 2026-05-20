from setuptools import setup, find_packages

setup(
    name="vLLM-Ascend-Inference-Plugin",
    version="1.0",
    packages=find_packages(),
    # 注册 vllm 插件入口点
    entry_points={
        "vllm.plugins": [
            "inference_plugin = src:register_plugin"
        ]
    }
)
