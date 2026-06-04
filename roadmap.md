# vLLM-Ascend Expert Offload 项目结构

## 1. 项目目标

本项目目标是在 **vLLM-Ascend** 上实现 MoE 模型专家卸载能力。

实现方式：

* 不修改 vLLM 主仓库核心代码
* 通过 vLLM plugin 注册自定义模型
* 使用 `ModelRegistry.register_model(...)` 替换 `Qwen3MoeForCausalLM`
* 在自定义模型中替换原始 `FusedMoE`
* 支持两种卸载模式：

  * `layer_wise`
  * `expert_wise`

---

## 2. 文件结构

```text
vllm_ascend_inference/
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── offload_config.py
│   ├── utils.py
│   │
│   ├── layer/
│   │   ├── fused_moe.py
│   │   ├── moe_mlp.py
│   │   └── routing.py
│   │
│   ├── offload/
│   │   ├── __init__.py
│   │   ├── executor.py
│   │   ├── memory_manager.py
│   │   └── routing.py
│   │
│   ├── roofline/
│   │   ├── calculate.py
│   │   └── strategy.py
│   │
│   └── loadbalance/
│       └── __init__.py
│
└── tests/
    ├── only_run.py
    ├── naive_run.py
    └── offloading_run.py
```

---

## 3. 顶层模块

### 3.1 `src/__init__.py`

职责：

* 暴露 `register_plugin`
* 设置全局 `OffloadConfig`
* 使用 `ModelRegistry.register_model(...)` 注册 `OffloadQwen3MoeForCausalLM`

---

### 3.2 `src/model.py`

模型封装入口。

职责：

* 定义 `OffloadQwen3MoeForCausalLM`
* 继承 vLLM 原生 `Qwen3MoeForCausalLM`
* 初始化模型前 patch `qwen3_moe.FusedMoE`
* 使用 `OffloadAscendFusedMoE` 替换原始 MoE layer

---

### 3.3 `src/offload_config.py`

专家卸载配置。

当前配置：

```python
@dataclass
class OffloadConfig:
    mode: str = "layer_wise"
    interval: int = 2
    num_buffers: int = 2
    num_hot_experts: int = 0
    cpu_pin_memory: bool = True
    offloaded_layer_ids: list[int] = field(default_factory=list)
```

说明：

* `mode` 支持 `none` / `layer_wise` / `expert_wise`
* `interval` 控制预取间隔
* `num_buffers` 至少为 2，用于 double buffer
* `num_hot_experts` 控制 `expert_wise` 中常驻 NPU 的 local experts 数量
* `offloaded_layer_ids` 用于指定 `layer_wise` 卸载层

---

## 4. `layer/`

`layer/` 负责封装 vLLM-Ascend 的 MoE 计算层。

### 4.1 `layer/fused_moe.py`

核心文件。

职责：

* 定义 `OffloadAscendFusedMoE`
* 定义 `OffloadUnquantizedFusedMoEMethod`
* 复用 vLLM-Ascend 原始 fused MoE 初始化逻辑
* 在 `weight_loader` 阶段将 cold expert 权重加载到 CPU
* 在 `process_weights_after_loading` 阶段处理 CPU cold weights 布局
* 在 forward 中执行冷热专家两阶段计算

当前 forward 流程：

1. router 计算 `topk_ids` / `topk_weights`
2. executor 准备当前层 cold experts
3. 构造 cold routing mask 和 cold-local `topk_ids`
4. 按 token 行拆分 hot / cold routing view
5. hot experts 只计算命中 hot expert 的 token 行
6. 等待当前 cold experts 加载完成
7. 预取后续层 cold experts
8. cold experts 只计算命中 cold expert 的 token 行
9. 将 hot / cold 输出 scatter 回完整输出

当前优化：

* hot / cold 不再对完整 `x` 重复执行 `fused_experts`
* 保留原始 `top_k` 维度，兼容 vLLM-Ascend dispatcher 输入约束
* `mc2_mask` 若为 token 维度，会随 token 行同步拆分

约束：

* 当前主要支持非量化路径
* split offload 路径暂不统计 dynamic EPLB load
* cold expert ids 默认连续
* cold NPU buffer 当前保持 local expert 数量，用于兼容 Ascend grouped matmul 的 group list

---

### 4.2 `layer/routing.py`

MoE token 行拆分与合并工具。

职责：

* 定义 `MoERoutingView`
* 根据 token 行 mask 构造局部 `hidden_states` / `topk_ids` / `topk_weights`
* 将局部 MoE 输出 scatter 回完整输出
* 对 token 维度的可选 tensor 做同步选行

---

### 4.3 `layer/moe_mlp.py`

当前保留。

后续若需要自定义冷专家 MLP 计算，再在此扩展。

---

## 5. `offload/`

`offload/` 负责专家权重管理、预取调度和 cold routing。

### 5.1 `offload/memory_manager.py`

CPU cold expert 权重管理。

职责：

* 每层只保存 cold experts 权重
* 在模型加载阶段接收 cold expert shard
* 按 vLLM-Ascend fused MoE 需要的布局处理 CPU 权重
* 向 executor 提供待预取的 cold expert weights

当前策略：

* `expert_wise` 只保存 `[num_hot_experts, local_num_experts)` 区间
* `layer_wise` 保存整层 local experts
* cold expert ids 默认连续
* CPU cold weights 已按 cold expert 顺序连续存储，预取时直接返回整层 cold weights

---

### 5.2 `offload/executor.py`

专家卸载执行器。

职责：

* 注册 MoE layer
* 判断 expert 是否 offload
* 在加载阶段拦截 cold expert 权重
* 管理 NPU cold expert buffers
* 使用 NPU stream 预取 cold experts
* 构造 cold routing 信息

当前预取策略：

* 当前层无预取结果时按需加载
* 当前 cold experts 加载完成后，预取 `layer_id + interval`
* 使用 double buffer 避免加载覆盖正在计算的 buffer
* 每层缓存 cold expert ids，避免 forward 中重复构造
* cold buffer 加载时不再整块 `zero_()`

---

### 5.3 `offload/routing.py`

设备侧 cold routing 映射。

职责：

* 定义 `LayerRoutingMap`
* 缓存 global expert id 到 local expert id 的映射
* 缓存 local expert id 到 cold buffer slot 的映射
* 在 `topk_ids.device` 上构造 `cold_topk_ids` 和 `cold_mask`

当前策略：

* forward 路径不再将 `topk_ids` 拉回 CPU
* 无效或非 cold top-k entry 指向已加载的 fallback cold slot
* `cold_mask` 控制真实 cold 权重参与计算

---

## 6. 推理测试

### 6.1 `tests/only_run.py`

唯一当前测试入口。

用途：

* 注册 plugin
* 设置 `OffloadConfig`
* 启动 vLLM 推理
* 验证模型加载、profile、generate 是否跑通

当前默认配置：

* `mode="expert_wise"`
* `interval=8`
* `num_buffers=2`
* `num_hot_experts=60`
* 每个 EP rank 只 offload 少量 cold experts
* 小 batch / 短序列，用于优先验证 correctness

当前验证状态：

* `tests/only_run.py` 已完成模型加载和 `generate`
* 推理输出已打印
* 非沙箱环境下验证通过

### 6.2 `tests/naive_run.py`

原始模型基准脚本。

当前不作为主要测试入口。

### 6.3 `tests/offloading_run.py`

早期 offload 测试脚本。

当前不作为主要测试入口。

---

## 7. 保留模块

### 7.1 `roofline/`

用于后续性能建模。

当前不影响 correctness。

### 7.2 `loadbalance/`

用于后续 expert 热度统计和重排。

当前保留空模块。

### 7.3 `utils.py`

通用工具文件。

根据后续需要补充。

---

## 8. 当前实现约束

* 优先保证小配置推理跑通
* 只使用 `tests/only_run.py` 做当前测试入口
* cold expert ids 默认连续
* cold NPU buffer 暂时保持 local expert 数量
* 暂不处理量化 MoE 路径
* 暂不做 roofline 自动策略选择
* 暂不做 expert 热度动态调整

---

## 9. 后续 fused_moe 优化计划

目标：参考 vLLM-Ascend 原生 `fused_moe` 实现，将当前外层 hot / cold split 逐步下沉到 dispatcher / MLP 路径，减少无效 token-expert dispatch 和 grouped matmul 计算。

### 9.1 阶段一：梳理原生调用链

范围：

* `vllm_ascend/ops/fused_moe/fused_moe.py`
* `vllm_ascend/ops/fused_moe/moe_comm_method.py`
* `vllm_ascend/ops/fused_moe/token_dispatcher.py`
* `vllm_ascend/ops/fused_moe/moe_mlp.py`
* `vllm_ascend/ops/fused_moe/prepare_finalize.py`

输出：

* 明确 `fused_experts` 输入输出约束
* 明确 `topk_ids` / `topk_weights` 是否支持 invalid entry
* 明确 `group_list` / `expanded_row_idx` / combine metadata 生成方式

---

### 9.2 阶段二：实现 offload 专用 compact dispatcher

目标：

* 在项目内新增 offload 专用 dispatcher helper
* 不直接修改 vLLM-Ascend 安装目录文件
* 只让 `OffloadAscendFusedMoE` 使用 compact dispatch

优化点：

* 按 hot / cold mask 过滤有效 token-expert pair
* 减少 `active_num = num_tokens * top_k`
* 生成 compact `group_list`
* 保留完整输出的反向 scatter metadata

---

### 9.3 阶段三：适配 cold experts MLP

目标：

* 让 cold experts 只计算 compact 后的 expert group
* 减少空 expert / masked expert 的 grouped matmul 开销
* 保持当前 cold NPU buffer 布局兼容

约束：

* 先支持非量化路径
* 先支持连续 cold expert ids
* 不影响原生 hot expert 正常路径

---

### 9.4 阶段四：合并 prepare / finalize 元数据

目标：

* 明确 split 后 token 行与原始 token 行的映射
* 保证 all-to-all / MC2 / allgather 路径下输出可正确还原
* 处理 `mc2_mask`、SP、shared experts 等少见路径

策略：

* 默认路径先走当前已验证 split offload
* compact dispatcher 只在满足约束时启用
* 不满足约束时 fallback 到当前实现

---

### 9.5 阶段五：验证与性能对比

验证项：

* `tests/only_run.py` correctness
* naive / offload 输出一致性
* hot-only / cold-only / mixed routing case
* layer-wise / expert-wise case

性能指标：

* prefill latency
* decode latency
* tokens/s
* H2D copy 时间
* fused MoE dispatch / MLP 时间
* NPU memory 占用
