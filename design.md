# vLLM-Ascend FusedMoE Runtime 新架构

## 1. 项目目标

本项目通过 vLLM plugin 为 vLLM-Ascend 的 Qwen3 MoE 模型提供可插拔的 FusedMoE Runtime，不修改 vLLM 和 vLLM-Ascend 主仓库代码。

Runtime 面向三类互斥运行模式：

* `profile`：专家负载统计模式。保持原生专家计算路径，仅记录专家访问频率、激活 token 数、计算耗时和短期热点变化。
* `offload`：专家卸载模式。NPU 常驻 Hot Experts 和少量 Cold Buffers，CPU 保存 Cold Experts，用专家权重换入换出换取更大的 KV Cache 容量。
* `balance`：负载均衡模式。NPU 保留 Local Experts 和 Redundant Experts，CPU 保存 Pair 内对端专家副本，通过冗余专家和 token 分流降低尾延迟。

核心原则：

* 三种模式共享模型接管、配置解析、专家负载统计和 CPU expert 存储能力。
* `offload` 和 `balance` 的调度策略相互隔离，避免把专家卸载和负载均衡耦合成一条复杂路径。
* 专家访问统一通过 `logical_expert_id -> physical_slot_id` 映射完成。
* NPU 专家显存布局在初始化和 `create_weights` 阶段确定，运行时只更新 slot 内容和映射快照，不动态修改模型结构。
* 专家负载统计、历史读取和初始 expert map 生成参考 vLLM-Ascend EPLB；专家参数传输以 CPU 到 NPU 为主，不复用 EPLB 的卡间参数迁移路径。

## 2. 项目结构

```text
vLLM_Ascend_Inference/
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── runtime_config.py
│   ├── utils.py
│   ├── layer/
│   │   ├── fused_moe.py
│   │   ├── quant_method.py
│   │   ├── moe_comm_method.py
│   │   └── routing.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── exo_executor.py
│   │   ├── lbvc_adaptor.py
│   │   └── memory_manager.py
│   ├── moeload/
│   │   ├── __init__.py
│   │   ├── profiler.py
│   │   └── history_mapping.py
│   └── roofline/
│       ├── calculate.py
│       └── strategy.py
└── tests/
    ├── only_run.py
    ├── naive_run.py
    └── runtime_run.py
```

目录职责：

* `src/__init__.py`：插件注册入口，接收 Runtime 配置并注册自定义 Qwen3 MoE 模型。
* `src/model.py`：模型接管入口，按配置选择需要替换的 MoE 层，非目标层保持原生 `AscendFusedMoE`。
* `src/runtime_config.py`：统一配置定义和校验，覆盖模式选择、接管层、CPU memory、offload、balance 和 profiler 参数。
* `src/layer/`：Runtime FusedMoE 计算层，负责权重创建、通信、routing view 构造和 MoE MLP 计算。
* `src/runtime/`：运行时执行层，封装 CPU expert 管理、offload 换入换出和 balance 冗余专家加载。
* `src/moeload/`：专家负载统计和历史映射模块，负责实时 profiler、历史记录读写和初始 expert map 生成。
* `src/roofline/`：性能建模与策略评估预留模块，不参与初版主运行路径。
* `tests/`：插件注册、原生基准和 Runtime 模式的统一测试入口。

## 3. 顶层接管流程

1. vLLM 加载插件，`src/__init__.py` 解析外部参数并生成全局 `RuntimeConfig`。
2. 插件注册 `RuntimeQwen3MoeForCausalLM`。
3. `src/model.py` 根据模型层数、专家数、EP/TP 并行信息和 rank 信息规范化配置。
4. 根据 `runtime_layer_ids` 或 `interval` 计算 Runtime 接管层。
5. 目标 MoE 层替换为 Runtime FusedMoE，其他层继续使用原生 vLLM-Ascend 实现。
6. 初始化共享对象：`RuntimeConfig`、profiler、CPU expert memory manager，以及当前模式需要的 executor/adaptor。

## 4. 配置模型

`runtime_config.py` 统一管理所有 Runtime 参数，避免不同模式各自解析配置。

基础参数：

* `runtime_mode`：只能为 `profile`、`offload` 或 `balance`。
* `runtime_layer_ids`：显式指定接管的 MoE 层。
* `interval`：按间隔自动选择接管层。
* `cpu_pin_memory`：CPU experts 是否使用 pinned memory。

Profiler 参数：

* `load_history_path`：专家负载记录路径。
* `enable_history_mapping`：是否读取历史负载生成专家分布。

Offload 参数：

* `num_hot_experts`：每卡常驻 NPU 的热点专家数量。
* `num_buffers`：每层可复用的 NPU 冷专家 buffer 数量。

Balance 参数：

* `pair_topology`：参与负载均衡的 rank pair 关系，默认NPU两两一组。
* `num_redundant_experts`：每卡冗余专家 slot 数量。
* `imbalance_threshold`：触发分流或冗余加载的负载不均衡阈值。
* `scheduler_interval`：scheduler 更新周期。

## 5. 核心抽象

### Profiler

`moeload/profiler.py` 是三种模式共享的专家负载统计模块。

统计内容：

* 该rank在该layer每个专家所处理的tokens数目
* 参考eplb的处理，在dispatch后直接通过元数据记录

`profile` 模式只记录和输出统计；`offload` 和 `balance` 可以读取 profiler 或历史记录作为初始专家放置依据。

### CPU Expert Memory

`runtime/memory_manager.py` 统一管理 CPU 上的专家权重。

职责：

* `offload` 模式保存 Cold Experts。
* `balance` 模式保存 Pair 内对端 NPU 专家副本。
* 维护连续 CPU 内存布局和 pinned memory 选项。
* 保存 shape、dtype、量化 scale/offset 等元数据。
* 将 CPU 权重转换为 Ascend FusedMoE 可加载的布局。

## 6. 计算层职责

### `layer/fused_moe.py`

定义 Runtime 版本的 Ascend FusedMoE，是单层 MoE 的接管入口。

职责：

* 根据 `runtime_mode` 决定 `create_weights` 使用的 NPU expert slot 数量。
* 在权重加载阶段拦截专家权重，并写入 NPU 常驻参数或 CPU expert memory manager。
* Forward 阶段调用 profiler、通信层、routing 层和 quant method。

不同模式的权重布局：

* `profile`：使用原生本地专家数量和原生权重布局。
* `offload`：NPU 参数只创建 Hot Expert slots 和 Cold Buffer slots，Cold Expert 权重保存到 CPU。
* `balance`：NPU 参数创建 Local Expert slots 和 Redundant Expert slots，Pair 内对端专家副本保存到 CPU。

### `layer/moe_comm_method.py`

封装 MoE 计算所需通信。

职责：

* 处理 expert parallel 下的 dispatch、all-to-all、combine 或 reduce。
* 保持 `profile` 模式的原生通信路径。
* 为 `offload` 提供 dispatch 后冷热 token 拆分入口。
* 为 `balance` 提供 Pair 内 token 分流入口。

通信层只移动 token 和张量，不决定专家替换策略。

### `layer/routing.py`


### `layer/quant_method.py`

实现 Runtime FusedMoE 的非量化和 W8A8 计算路径。

职责：

* 复用 vLLM-Ascend 原生 MoE MLP / Grouped GEMM 能力。
* 只接收 expert weight buffer、`physical_slot_id` 和 routing view。
* 不感知 offload 或 balance 的策略细节。

## 7. Runtime 执行层

### `runtime/exo_executor.py`

负责 expert offload 的运行时执行。

职责：
* 使用独立 NPU stream 执行 CPU 到 NPU 的异步加载。
* 在计算流使用 Cold Buffer 前插入 event 同步。
* 支持单缓冲串行加载和双缓冲流水线加载。

### `runtime/lbvc_adaptor.py`

负责 load balance via CPU 的运行时适配。

职责：

* 判断 Pair 内 rank 是否出现负载不均。
* 选择需要加载到 Redundant Slot 的对端热点专家。
* 调用 memory manager 和加载执行逻辑更新 Redundant Slot。
* 生成 token 分流计划，并通知 routing 层更新映射快照。
* 控制更新频率，避免冗余专家频繁替换造成抖动。
* 可参考eplb

## 8. 三种模式流程

### Profile

1. Runtime 接管目标 MoE 层。
2. `create_weights` 和权重加载保持原生本地专家布局。
3. Forward 使用原生专家计算路径。
4. 按配置保存负载记录。

Profile 不加载 CPU experts，不更新专家映射，不执行 token 分流。

### Offload

1. 初始化阶段读取配置和历史负载，确定 Hot Experts。
2. NPU 创建 Hot Expert slots 和 Cold Buffer slots。
3. CPU memory manager 保存 Cold Expert 权重。
4. Forward dispatch 后识别当前 step 命中的 Cold Experts。
5. `exo_executor.py` 将所需 Cold Experts 异步加载到 Cold Buffers。
6. routing 层把 cold token 指向对应 Cold Buffer slots。
7. quant method 对 Hot Experts 和 Cold Buffer slots 执行统一 MoE 计算。
8. profiler 更新本次专家负载记录。

### Balance

1. 初始化阶段建立 Pair 拓扑和初始 expert map。
2. NPU 创建 Local Expert slots + Redundant Expert slots（同一tensor创建）。
3. CPU memory manager 保存 Pair 内对端专家副本。
4. `lbvc_adaptor.py` 周期性判断负载不均，并选择对端热点专家加载到 Redundant Slot。
5. routing 层根据分流计划更新下一 step 的映射快照。
6. quant method 按统一 `physical_slot_id` 执行计算。

## 9. Roofline 与策略评估

`roofline/` 只作为建模和策略实验入口，不做任何实现。

`calculate.py` 估算：

`strategy.py` 评估：

* 不同 batch、专家热度、DMA 带宽和计算负载下的模式收益。
* Offload 与 Balance 策略的适用边界。

## 10. 测试入口

* `tests/only_run.py`：验证插件注册、配置解析、模型加载和基础推理，覆盖 `profile` smoke test。
* `tests/naive_run.py`：运行原生 vLLM-Ascend 推理基准，作为正确性和性能对照。
* `tests/runtime_run.py`：统一验证 Runtime 模式，覆盖 Offload 的 Cold Buffer 加载和 Balance 的 profiler/adaptor/routing 流程。
