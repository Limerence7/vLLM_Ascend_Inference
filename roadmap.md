# vLLM-Ascend Expert Offload 项目架构

## 1. 项目目标

本项目通过 vLLM plugin 为 vLLM-Ascend 的 Qwen3 MoE 模型提供专家卸载能力，不修改 vLLM 和 vLLM-Ascend 主仓库代码。

支持以下运行模式：

* `none`：使用原生 vLLM-Ascend MoE，不进行卸载。
* `manual`：根据配置的层编号或间隔执行专家卸载。
* `auto`：自动卸载模式的扩展入口，当前层选择逻辑与 `manual` 相同。

卸载路径由 `num_hot_experts` 决定：

* `num_hot_experts=0`：使用整层卸载路径。
* `num_hot_experts` 达到单卡专家数：不进行专家卸载。
* `num_hot_experts` 超过单卡专家数：自动限制为单卡专家数。

## 2. 项目结构

```text
vLLM_Ascend_Inference/
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── offload_config.py
│   ├── utils.py
│   ├── layer/
│   │   ├── fused_moe.py
│   │   ├── moe_mlp.py
│   │   ├── quant_method.py
│   │   ├── routing.py
│   │   └── token_dispatcher.py
│   ├── offload/
│   │   ├── __init__.py
│   │   ├── executor.py
│   │   ├── memory_manager.py
│   │   └── routing.py
│   ├── roofline/
│   │   ├── calculate.py
│   │   └── strategy.py
│   └── loadbalance/
│       └── __init__.py
└── tests/
    ├── only_run.py
    ├── naive_run.py
    └── offloading_run.py
```

## 3. 插件与模型入口

### `src/__init__.py`

* 暴露插件注册入口。
* 接收并设置全局 `OffloadConfig`。
* 向 vLLM 注册自定义 Qwen3 MoE 模型。

### `src/model.py`

* 定义自定义 `OffloadQwen3MoeForCausalLM`。
* 根据模型规模和并行配置规范化卸载参数。
* 仅为需要卸载的层安装 `OffloadAscendFusedMoE`。
* 无需卸载的层继续使用原生 `AscendFusedMoE`。

### `src/offload_config.py`

* 定义和校验卸载配置。
* 计算需要卸载的层。
* 规范化单卡热专家数量。
* 判断当前配置是否使用整层卸载路径。

## 4. MoE 计算层

### `src/layer/fused_moe.py`

* 定义卸载版本的 Ascend Fused MoE。
* 在权重加载阶段拦截并保存冷专家权重。
* 初始化专家映射、通信后端和量化方法。

### `src/layer/quant_method.py`

* 实现非量化与 W8A8 MoE 卸载计算。
* 复用 vLLM-Ascend 原生路由和计算接口。
* 执行整层卸载或冷热专家拆分计算。

### `src/layer/routing.py`

* 构造热专家和冷专家的 token routing view。
* 同步筛选 token 维度的辅助张量。
* 将局部 MoE 计算结果合并回完整输出。

### `src/layer/token_dispatcher.py`

* 临时设置 dispatcher 使用的本地专家数量。
* 保证复用原生 dispatcher 时与当前专家 buffer 数量一致。

### `src/layer/moe_mlp.py`

* 保留自定义 MoE MLP 扩展入口。

## 5. 卸载管理

### `src/offload/executor.py`

* 注册需要卸载的 MoE 层。
* 管理可复用的 NPU 冷专家 buffer。
* 使用独立 NPU stream 执行 CPU 到 NPU 的异步预取。
* 使用循环预取衔接连续推理步骤。
* 为整层卸载和专家卸载准备对应的计算输入。

### `src/offload/memory_manager.py`

* 为每个卸载层保存连续的 CPU 冷专家权重。
* 在模型加载阶段接收专家权重分片。
* 将 CPU 权重转换为 Ascend Fused MoE 所需布局。
* 向 executor 提供待加载的冷专家权重。

### `src/offload/routing.py`

* 缓存全局专家到本地专家的映射。
* 缓存本地专家到冷专家 buffer slot 的映射。
* 在设备侧生成冷专家 `topk_ids` 和路由 mask。

## 6. 辅助模块

### `src/roofline/`

* 提供卸载策略的性能建模与计算扩展入口。

### `src/loadbalance/`

* 提供专家热度统计和负载均衡扩展入口。

### `src/utils.py`

* 提供跨模块通用工具扩展入口。

## 7. 测试入口

### `tests/only_run.py`

* 验证插件注册、卸载配置、模型加载和推理生成。

### `tests/naive_run.py`

* 运行未启用卸载的原生模型基准。

### `tests/offloading_run.py`

* 运行卸载场景的功能和性能测试。
