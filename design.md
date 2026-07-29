# vLLM-Ascend FusedMoE Runtime 新架构

## 1. 项目目标

本项目通过 vLLM plugin 为 vLLM-Ascend 的 Qwen3 MoE 模型提供可插拔的 FusedMoE Runtime，不修改 vLLM 和 vLLM-Ascend 主仓库代码。

Runtime 面向三类互斥运行模式：

* `profile`：专家负载统计模式。模型保持原生 vLLM-Ascend MoE 计算路径，仅记录专家访问频率、激活 token 数和负载变化，不调整专家映射，不执行 CPU/NPU 参数传输。
* `offload`：专家部分卸载模式。NPU 只保留一部分运行时专家和 cold buffer，CPU 保存本 rank cold experts，通过运行时换入换出减少 NPU 常驻专家显存。
* `balance`：专家负载均衡模式。Runtime 根据全局负载动态调整 NPU 上的常驻专家分布，可使用冗余专家实现全局负载均衡。

核心原则：

* `runtime_mode` 只允许为 `profile`、`offload`、`balance`。
* `num_hot_experts` 统一改名为 `num_runtime_experts`，该参数决定 Runtime 专家布局和执行语义。
* `num_runtime_experts < 0` 表示每卡减少的 NPU 常驻专家数量，并创建对应 cold buffer。
* `num_runtime_experts = 0` 表示不对专家进行卸载或冗余，等价于保持原生专家常驻布局。
* `num_runtime_experts > 0` 表示使用冗余专家，Runtime 可调整 NPU 常驻专家参数。
* `OffloadFusedMoE` 只负责模型层接管、权重创建和调用 Runtime Core，不直接持有 executor、memory manager 或 balance adaptor 的策略逻辑。
* `runtime_core` 是 Runtime 执行中枢，负责创建并协调 `exo_executor`、`memory_manager` 和 `lbvc_adaptor`。
* `transfer_planner` 负责 balance 专家替换规划和 HCCL 卡间互传。
* `moeload/profiler.py` 只负责专家负载统计，`moeload/policy.py` 只负责根据负载计算专家映射策略。
* 专家访问统一通过 `logical_expert_id -> physical_slot_id` 映射完成。
* NPU 专家显存布局在初始化和 `create_weights` 阶段确定，运行时只更新 slot 内容和映射快照，不动态修改模型结构。

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
│   │   ├── runtime_core.py
│   │   ├── exo_executor.py
│   │   ├── memory_manager.py
│   │   ├── transfer_planner.py
│   │   └── lbvc_adaptor.py
│   ├── moeload/
│   │   ├── __init__.py
│   │   ├── profiler.py
│   │   └── policy.py
│   └── roofline/
│       ├── calculate.py
│       └── strategy.py
└── tests/
    ├── only_run.py
    ├── naive_run.py
    ├── runtime_run.py
    ├── test_profiler.py
    ├── test_policy.py
    ├── test_lbvc_adaptor.py
    ├── test_exo_executor.py
    ├── test_transfer_planner.py
    └── test_runtime_config.py
```

目录职责：

* `src/__init__.py`：插件注册入口，接收 Runtime 配置并注册自定义 Qwen3 MoE 模型。
* `src/model.py`：模型接管入口，按配置选择需要替换的 MoE 层；`profile` 模式下保持原生专家计算路径并接入 profiler。
* `src/runtime_config.py`：统一配置定义和校验，覆盖模式选择、接管层、CPU memory、profiler、policy 和 `num_runtime_experts` 参数。
* `src/layer/`：Runtime FusedMoE 计算层，负责权重创建、通信、routing view 构造和 MoE MLP 计算。
* `src/runtime/`：运行时执行层，包含 `runtime_core`、`exo_executor`、`memory_manager`、`lbvc_adaptor` 和 `transfer_planner`。
* `src/moeload/`：专家负载统计和策略模块，包含 profiler 与 policy。
* `src/roofline/`：性能建模与策略评估预留模块，不参与初版主运行路径。
* `tests/`：插件注册、原生基准和 Runtime 模式的统一测试入口。

## 3. 顶层接管流程

1. vLLM 加载插件，`src/__init__.py` 解析外部参数并生成全局 `RuntimeConfig`。
2. 插件注册自定义 Qwen3 MoE 模型。
3. `src/model.py` 根据模型层数、专家数、EP/TP 并行信息和 rank 信息规范化配置。
4. 如果 `runtime_mode = profile`，目标层保持原生专家计算路径，但接入 profiler 记录专家负载。
5. 如果 `runtime_mode = offload` 或 `runtime_mode = balance`，根据 `runtime_layer_ids` 或 `interval` 计算 Runtime 接管层。
6. 目标 MoE 层替换为 `OffloadFusedMoE`，其他层继续使用原生 vLLM-Ascend 实现。
7. `OffloadFusedMoE` 在初始化时只建立对应层的 `RuntimeCore` 实例。
8. `RuntimeCore` 内部创建并协调 `MemoryManager`、`ExoExecutor` 和 `LBVCAdaptor`。

## 4. 配置模型

`runtime_config.py` 统一管理所有 Runtime 参数，避免不同模式各自解析配置。

基础参数：

* `runtime_mode`：只能为 `profile`、`offload` 或 `balance`。
* `runtime_layer_ids`：显式指定接管的 MoE 层。
* `interval`：按间隔自动选择接管层。
* `cpu_pin_memory`：CPU experts 是否使用 pinned memory。
* `num_runtime_experts`：统一控制专家卸载、无调整和冗余专家布局。

`num_runtime_experts` 语义：

* `< 0`：每卡减少 `abs(num_runtime_experts)` 个 NPU 常驻 experts，并创建对应 cold buffer。
* `= 0`：不进行专家卸载或冗余，NPU 专家布局与原生实现一致。
* `> 0`：冗余专家数量。NPU 在原生常驻专家 slot 之外创建冗余 slot，用于全局负载均衡。

Profiler 与策略参数：

* `load_history_path`：专家负载记录路径。
* `enable_history_mapping`：是否读取历史负载并生成初始化专家映射。
* `policy_interval`：策略更新周期。
* `imbalance_threshold`：触发专家重映射或冗余加载的负载不均衡阈值。

Balance 参数：

Balance 专家权重只通过 HCCL 在 rank 间互传；CPU-NPU 仅用于固定 cold experts。

## 5. 核心抽象

### `RuntimeCore`

`runtime/runtime_core.py` 是 `OffloadFusedMoE` 与 Runtime 执行层之间的唯一入口。

职责：

* 根据 `runtime_mode` 和 `num_runtime_experts` 初始化运行时布局。
* 创建 `MemoryManager`、`ExoExecutor` 和 `LBVCAdaptor`。
* 在权重加载阶段接收 `OffloadFusedMoE` 拦截到的专家权重，并转交给 `MemoryManager` 管理。
* 在 forward 阶段接收 profiler 统计结果、当前 token routing 信息和 layer/rank 信息。
* 调用 `LBVCAdaptor` 生成或更新专家映射任务。
* 调用 `TransferPlanner` 执行 balance 的 HCCL 卡间互传，或调用 `ExoExecutor` 加载固定 cold experts。
* 向 routing 层提供当前稳定的 `logical_expert_id -> physical_slot_id` 映射快照。

`OffloadFusedMoE` 不直接调度专家换入换出，也不直接制定负载均衡策略。

### `MemoryManager`

`runtime/memory_manager.py` 统一管理 CPU expert 和 NPU cold/redundant/runtime slot。

职责：

* 保存 CPU 专家权重、shape、dtype、量化 scale/offset 等元数据。
* 管理 CPU pinned memory 选项。
* 管理 NPU 上由 `create_weights` 创建出的可更新 expert slots。
* 在 `offload` 且 `num_runtime_experts < 0` 时，CPU 只保存本 rank cold experts，NPU 只调整 cold buffer。
* CPU 只保存固定 cold experts，不参与 balance 专家重分布。
* 提供从 CPU expert 到 NPU physical slot 的加载描述，不直接执行传输。

### `ExoExecutor`

`runtime/exo_executor.py` 只负责固定 cold experts 的 CPU-NPU 传输。

职责：

* 执行 CPU-NPU expert slot 参数传输。
* 支持独立 NPU stream、event 同步和必要的流水线加载。

### `TransferPlanner`

`runtime/transfer_planner.py` 负责专家替换路径选择。

职责：

* 保持仍驻留本 rank 的专家物理 slot 不变。
* 为迁入专家生成 HCCL P2P 任务并执行卡间互传。
* 权重传输完成后由 `LBVCAdaptor` 发布新的 expert map。

### `LBVCAdaptor`

`runtime/lbvc_adaptor.py` 负责专家映射制定、修改和执行任务整合。

职责：

* 从 profiler 或历史记录获得专家负载信息。
* 调用 `moeload/policy.py` 计算专家映射。
* 把 expert map 转换为 NPU slot 更新任务。
* 将 balance 目标映射交给 `TransferPlanner` 执行 HCCL 互传。
* 更新 routing 所需的映射快照。
* 在 `enable_history_mapping = True` 时主动发起初始化映射任务，调用 policy 生成 expert map，并驱动 Runtime 初始化专家分布。
* `balance < 0` 时只调整 resident 热专家，cold experts 始终固定。
* `balance >= 0` 时对全部 NPU experts 做全局负载均衡。

### `Profiler`

`moeload/profiler.py` 是 Runtime 共享的专家负载统计模块。

统计内容：

* 每个 rank、layer、expert 处理的 token 数。
* dispatch 后可直接记录的专家访问元数据。
* 短期和历史负载变化。

Profiler 只记录和输出负载，不决定专家映射。

Profiler 统计在设备侧累加，避免 forward 每步同步到 CPU；保存历史或读取 delta 时才同步。

### `Policy`

`moeload/policy.py` 是专家映射策略模块。

职责：

* 根据当前负载或历史负载计算 expert map。
* 为 `offload` 模式提供 cold buffer 调整策略。
* 为 `balance` 模式提供全局专家重分布或冗余专家加载策略。
* 只输出策略结果，不执行参数传输，不修改 routing 映射。

## 6. 计算层职责

### `layer/fused_moe.py`

定义 Runtime 版本的 Ascend FusedMoE，是单层 MoE 的接管入口。

职责：

* 根据 `runtime_mode` 和 `num_runtime_experts` 决定 `create_weights` 使用的 NPU expert slot 数量。
* 初始化该层对应的 `RuntimeCore`。
* 在权重加载阶段拦截专家权重，并交给 `RuntimeCore`。
* Forward 阶段调用 profiler、通信层、routing 层、quant method 和 `RuntimeCore`。

不同配置下的权重布局：

* `runtime_mode = profile`：保持原生专家布局和原生计算路径，只记录专家负载。
* `num_runtime_experts < 0`：NPU 创建减少后的常驻专家 slots 和 cold buffer slots。
* `num_runtime_experts = 0`：NPU 创建原生本地专家 slots，不启用专家迁移。
* `num_runtime_experts > 0`：NPU 创建原生本地专家 slots 和 redundant slots，CPU 保存该层全部 experts。

### `layer/moe_comm_method.py`

封装 MoE 计算所需通信。

职责：

* 处理 expert parallel 下的 dispatch、all-to-all、combine 或 reduce。
* 为 `offload` 提供 dispatch 后 cold expert token 识别入口。
* 为 `balance` 提供全局或跨 rank token 分流入口。
* 通信层只移动 token 和张量，不决定专家替换策略。

### `layer/routing.py`

负责构造 Runtime 计算所需的 routing view。

职责：

* 接收 `RuntimeCore` 提供的 expert map 快照。
* 将 `logical_expert_id` 转换为 `physical_slot_id`。
* 为 cold buffer、原生常驻 slot 和 redundant slot 提供统一 routing 表示。
* 保证 quant method 不感知 offload 或 balance 的策略差异。
* 原来的token_dispatcher文件部分内容，直接迁移到这来，删除该文件。

### `layer/quant_method.py`

实现 Runtime FusedMoE 的非量化和 W8A8 计算路径。

职责：

* 复用 vLLM-Ascend 原生 MoE MLP / Grouped GEMM 能力。
* 只接收 expert weight buffer、`physical_slot_id` 和 routing view。
* 不感知 offload 或 balance 的策略细节。

## 7. Runtime 执行流程

### Profile

1. `runtime_mode = profile`。
2. Runtime 接管目标 MoE 层，但 `create_weights` 和 forward 计算保持原生本地专家布局。
3. Forward 过程中通过 profiler 在设备侧记录每层、每个 expert 的 token 负载。
4. 不创建 CPU expert storage。
5. 不调整专家映射，不执行专家参数传输。

### Offload

适用条件：

* `runtime_mode = offload`。
* 主要使用 `num_runtime_experts < 0` 表示每卡卸载专家数量。

流程：

1. 初始化阶段根据 `num_runtime_experts` 计算每卡需要卸载的 cold experts。
2. `create_weights` 创建 NPU 常驻 experts 和 cold buffer slots。
3. `RuntimeCore` 将专家权重交给 `MemoryManager`，CPU 只保存本 rank cold experts。
4. 如果 `enable_history_mapping = True`，`LBVCAdaptor` 读取历史负载并调用 `Policy` 生成初始 expert map。
5. Forward dispatch 后，profiler 统计当前 expert token 负载。
6. `LBVCAdaptor` 根据当前负载或历史策略计算 cold buffer 更新任务。
7. `ExoExecutor` 将需要命中的 cold experts 从 CPU 加载到 NPU cold buffer。
8. `RuntimeCore` 更新 routing 映射快照。
9. quant method 对常驻 experts 和 cold buffer slots 执行统一 MoE 计算。

Offload 模式只调整 cold buffer，不调整全部 NPU 常驻专家布局。

### Balance

适用条件：

* `runtime_mode = balance`。
* `num_runtime_experts < 0`、`= 0` 或 `> 0` 分别表示 cold buffer、原生 slot 数和 redundant slots。

流程：

1. 初始化阶段建立全局 rank/layer/expert 视图。
2. CPU 侧仅在存在 cold buffer 时保存固定 cold experts。
3. `create_weights` 根据 `num_runtime_experts` 创建 resident、cold buffer 或 redundant slots。
4. 如果 `enable_history_mapping = True`，`LBVCAdaptor` 主动发起初始化映射任务。
5. `LBVCAdaptor` 将历史负载或当前全局负载传给 `Policy`。
6. `Policy` 计算全局 expert map，决定哪些专家应进入 resident、cold buffer 或 redundant slots。
7. `LBVCAdaptor` 将 expert map 交给 `TransferPlanner`，通过 HCCL 完成迁入专家互传。
8. `ExoExecutor` 不参与 balance 权重迁移。
9. `RuntimeCore` 更新 routing 映射快照，后续 token 根据新的全局映射进行计算。

Balance 模式不再局限于 pair 内对端专家副本，而是以全局负载为输入，直接调整 NPU 上由 `create_weights` 创建出的常驻专家参数和冗余专家参数，实现真正的全局负载均衡。

不同 `num_runtime_experts` 的传输语义：

* `< 0`：resident 热专家调换走 HCCL，cold buffer 专家保持固定并从 CPU 加载。
* `= 0`：原生数量的 slots 参与全局均衡，迁入专家走 HCCL。
* `> 0`：redundant slots 用于热点专家副本，迁入专家走 HCCL。

## 8. 历史映射初始化

当 `enable_history_mapping = True`：

1. `RuntimeCore` 初始化后通知 `LBVCAdaptor`。
2. `LBVCAdaptor` 读取 `load_history_path` 中的历史负载记录。
3. `LBVCAdaptor` 将历史负载、当前 rank 信息、layer 信息、NPU slot 信息和 `num_runtime_experts` 传给 `Policy`。
4. `Policy` 返回初始化 expert map。
5. `LBVCAdaptor` 将 expert map 转换为加载任务，并交给 `ExoExecutor`。
6. `TransferPlanner` 完成 balance 的 HCCL 互传；`ExoExecutor` 仅完成固定 cold experts 的 CPU-NPU 加载。
7. `RuntimeCore` 发布初始化后的 routing 映射快照。

该流程适用于 `offload` 和 `balance`，但语义不同：

* `offload`：初始化 cold buffer 的专家分布。
* `balance`：初始化全局 resident、cold buffer 或 redundant expert 分布。

## 9. Roofline 与策略评估

`roofline/` 只作为建模和策略实验入口，不做主运行路径实现。

`calculate.py` 估算：

* CPU 到 NPU 专家参数传输成本。
* cold buffer 数量和冗余专家数量对显存占用的影响。
* 不同 token 分布下的专家计算开销。

`strategy.py` 评估：

* 不同 batch、专家热度、DMA 带宽和计算负载下的模式收益。
* `offload` 与 `balance` 策略的适用边界。
* `num_runtime_experts` 不同取值对性能和显存的影响。

## 10. 测试入口

* `tests/only_run.py`：验证插件注册、配置解析、模型加载和 `profile` smoke test。
* `tests/naive_run.py`：运行原生 vLLM-Ascend 推理基准，作为正确性和性能对照。
* `tests/runtime_run.py`：统一验证 Runtime 模式，覆盖 `offload` 的 cold buffer 加载和 `balance` 的全局 profiler/policy/adaptor/routing 流程。
* `tests/test_profiler.py`：验证 profiler 设备侧统计和历史保存行为。
* `tests/test_policy.py`：验证全局专家分布策略。
* `tests/test_lbvc_adaptor.py`：验证 balance/offload 的 slot 更新策略。
* `tests/test_exo_executor.py`：验证固定 cold experts 的 CPU-NPU 加载语义。
* `tests/test_transfer_planner.py`：验证 CPU/HCCL/local NPU 传输任务规划。
* `tests/test_runtime_config.py`：验证 Runtime 配置校验。
