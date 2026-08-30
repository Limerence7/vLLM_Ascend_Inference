# vLLM-Ascend FusedMoE Runtime 设计

## 1. 项目目标

本项目通过 vLLM plugin 为 vLLM-Ascend 的 Qwen3 MoE 模型提供可插拔的 FusedMoE Runtime，不修改 vLLM 和 vLLM-Ascend 主仓库代码。

Runtime 面向三类互斥运行模式：

* `profile`：专家负载统计模式。模型保持原生 vLLM-Ascend MoE 计算路径，仅记录专家访问频率、激活 token 数和负载变化，不调整专家映射，不执行 CPU/NPU 参数传输。
* `offload`：专家卸载模式。NPU 保留运行时 hot experts 和 cold buffer，CPU 保存本 rank cold experts；支持单卡专家全卸载，forward 时把 hot experts 和已换入的 cold experts 作为统一专家列表计算。
* `balance`：专家负载均衡模式。Runtime 根据全局负载动态调整 NPU 上的常驻专家分布，可使用冗余专家实现全局负载均衡。

核心原则：

* `runtime_mode` 只允许为 `profile`、`offload`、`balance`。
* `num_runtime_experts` 决定 Runtime 专家布局和执行语义。
* `num_runtime_experts < 0` 表示每卡减少的 NPU 常驻专家数量，并创建对应 cold buffer；可减少到 0 个常驻专家。
* `num_runtime_experts = 0` 表示不对专家进行卸载或冗余，等价于保持原生专家常驻布局。
* `num_runtime_experts > 0` 只允许在 `balance` 模式下使用，表示创建冗余专家 slot。
* offload 模式不再提供冷热专家 split compute，也不额外修改 `topk_ids`、mask 或 token routing；专家排布自初始化后保持一致，通过 hot/cold weight list 拼接成完整本地专家视图；全卸载时 hot list 为空。
* `RuntimeAscendFusedMoE` 只负责模型层接管、权重创建和调用 Runtime Core，不直接持有 executor、memory manager 或 balance adaptor 的策略逻辑。
* `runtime_core` 是 Runtime 执行中枢，负责创建并协调 `exo_executor`、`memory_manager` 和 `lbvc_adaptor`。
* `lbvc_adaptor` 负责 balance 专家替换任务生成，`exp_updator` 负责接收任务并执行 HCCL 卡间互传。
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
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   └── patch.py
│   ├── layer/
│   │   ├── fused_moe.py
│   │   ├── moe_mlp.py
│   │   ├── quant_method.py
│   │   └── routing.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── runtime_core.py
│   │   ├── exo_executor.py
│   │   ├── memory_manager.py
│   │   ├── exp_updator.py
│   │   ├── balance_coordinator.py
│   │   └── lbvc_adaptor.py
│   ├── moeload/
│   │   ├── __init__.py
│   │   ├── profiler.py
│   │   └── policy.py
│   └── roofline/
│       ├── calculate.py
│       └── strategy.py
└── tests/
    ├── lbvc_template.py
    ├── naive_dp_test.py
    ├── only_run.py
    ├── naive_run.py
    ├── runtime_dp_test.py
    ├── runtime_run.py
    ├── test_scheduler.py
    ├── test_balance_policy.py
    └── template.py
```

目录职责：

* `src/__init__.py`：插件注册入口，接收 Runtime 配置并注册自定义 Qwen3 MoE 模型。
* `src/model.py`：模型接管入口，按配置选择需要替换的 MoE 层；`profile` 模式下保持原生专家计算路径并接入 profiler。
* `src/runtime_config.py`：统一配置定义和校验，覆盖模式选择、接管层、CPU memory、profiler、policy 和 `num_runtime_experts` 参数。
* `src/scheduler/`：vLLM V1 请求调度子系统；`core.py` 集中实现自动参数推导、请求级 MoE 激活画像和 prefill/decode 策略，`patch.py` 负责接入 V1 Scheduler，`__init__.py` 提供公开接口。
* `src/layer/fused_moe.py`：Runtime FusedMoE 层入口，负责专家 slot 创建、权重拦截、forward 接管和 profiler 写入。
* `src/layer/quant_method.py`：Runtime MoE 计算适配层，接入 vLLM-Ascend token dispatcher，并在 cold-buffer 场景下调用统一专家 MLP。
* `src/layer/moe_mlp.py`：从 vLLM-Ascend MLP 逻辑中抽取出的本地 MoE MLP，支持非量化和 W8A8 expert weight list。
* `src/layer/routing.py`：只提供 `dispatch_with_local_experts`，用于在一次 grouped matmul 调用期间临时匹配 token dispatcher 的本地专家数量。
* `src/runtime/`：运行时执行层，包含 `runtime_core`、`exo_executor`、`memory_manager`、`balance_coordinator`、`lbvc_adaptor` 和 `exp_updator`。
* `src/moeload/`：专家负载统计和策略模块。
* `src/roofline/`：性能建模与策略评估预留模块，不参与初版主运行路径。
* `tests/`：插件注册、原生基准和 Runtime 模式的统一测试入口。

## 3. 顶层接管流程

1. vLLM 加载插件，`src/__init__.py` 解析外部参数并生成全局 `RuntimeConfig`。
2. 插件注册自定义 Qwen3 MoE 模型。
3. `src/model.py` 根据模型层数、专家数、EP/TP 并行信息和 rank 信息规范化配置。
4. 如果 `runtime_mode = profile`，目标层保持原生专家计算路径，但接入 profiler 记录专家负载。
5. 如果 `runtime_mode = offload` 或 `runtime_mode = balance`，根据 `runtime_layer_ids` 或 `interval` 计算 Runtime 接管层。
6. 目标 MoE 层替换为 `RuntimeAscendFusedMoE`，其他层继续使用原生 vLLM-Ascend 实现。
7. `RuntimeAscendFusedMoE` 在初始化时只建立对应层的 `RuntimeCore` 实例。
8. `RuntimeCore` 内部创建并协调 `MemoryManager`、`ExoExecutor` 和 `LBVCAdaptor`。

## 4. 配置模型

`runtime_config.py` 统一管理所有 Runtime 参数，避免不同模式各自解析配置。

基础参数：

* `runtime_mode`：只能为 `profile`、`offload` 或 `balance`。
* `runtime_layer_ids`：显式指定接管的 MoE 层。
* `interval`：按间隔自动选择接管层。
* `cpu_pin_memory`：CPU experts 是否使用 pinned memory。
* `num_runtime_experts`：统一控制专家卸载、无调整和冗余专家布局。
* `num_buffers`：cold buffer 的流水 buffer 数量。

`num_runtime_experts` 语义：

* `< 0`：每卡减少 `abs(num_runtime_experts)` 个 NPU 常驻 experts，并创建对应 cold buffer；可减少到 0 个常驻 expert。
* `= 0`：不进行专家卸载或冗余，NPU 专家布局与原生实现一致。
* `> 0`：冗余专家数量。NPU 在原生常驻专家 slot 之外创建冗余 slot，用于全局负载均衡。

Profiler 与策略参数：

* `load_history_path`：专家负载记录路径。
* `enable_history_mapping`：是否读取历史负载并生成初始化专家映射。
* `enable_load_collection`：非 `profile`、非 `balance` 模式下是否显式采样专家负载；用于 offload 生成历史记录，默认关闭以避免影响吞吐。
* `load_collect_interval`：每隔多少次推理统计一次专家负载，balance 模式只在采样步或 rebalance 步开启统计。
* `rebalance_interval`：每隔多少次推理尝试一次专家分布调整。
* `policy_interval`：旧策略更新周期参数，作为 `rebalance_interval` 的兼容默认值。
* `imbalance_threshold`：触发专家重映射或冗余加载的 rank 负载 peak/average ratio（PAR）阈值。
* `rebalance_max_layers`：每个 rebalance 周期最多实际更新的层数；`0` 表示不限制，默认 `1`。
* `rebalance_min_improvement`：候选布局相对当前布局至少需要降低的 peak/average 负载比，避免低收益迁移。
* `enable_scheduler`：是否启用请求调度 patch。
* `scheduler_reorder_window`：请求调度只在队列前若干请求内做局部稳定重排；留空时自动根据最大并发序列数和是否使用 cold buffer 推导。
* `scheduler_policy`：请求调度策略，可选 `auto`、`fifo`、`throughput`、`expert` 或 `offload`；默认 `auto`。
* `scheduler_decode_reserve_ratio`：running 队列中 decode 请求簇的前置保留比例，避免 decode 延迟和长 prefill 饥饿互相失控；留空时自动推导。
* `scheduler_activation_similarity_threshold`：请求 MoE 激活画像聚类阈值；留空时根据是否卸载自动推导。
* `scheduler_max_profile_experts`：每个请求激活画像保留的热点专家上限；留空时根据卸载专家数自动推导。
* `scheduler_feedback_interval`：实际 MoE 激活画像的周期采样间隔；请求首次进入模型时始终采样，后续间隔留空时根据卸载专家数自动推导。
* `scheduler_feedback_max_layers`：单次请求画像最多采样的 Runtime MoE 层数；留空时均匀选择最多 4 层。
* `rebalance_min_step_tokens`：balance 调整前的最小统计 token 数，窗口内 token 太少时跳过重排。

Balance 参数：

Balance 专家权重只通过 HCCL 在 rank 间互传；CPU-NPU 仅用于固定 cold experts。
`runtime_mode = balance` 时动态负载均衡始终开启；Runtime 通过 `load_collect_interval` 降低观测成本，通过 `rebalance_interval`、`rebalance_min_step_tokens` 和 `imbalance_threshold` 控制实际专家迁移频率。

请求调度参数：

* `fifo`：保持 vLLM waiting 队列原始顺序，只保留原有 chunked prefill 填充逻辑。
* `throughput`：在 `scheduler_reorder_window` 内优先调度 prompt token 数较多的请求，用于离线大 batch 场景下放大单 step token 数，摊薄 offload cold buffer 加载成本。
* `expert`：优先读取请求上的轻量专家代价 hint，如 `vllm_ascend_expert_cost`、`expert_cost`、`scheduler_cost` 或 `priority`，代价低的请求先调度；没有 hint 时回退到 `throughput`。
* `offload`：把 CPU-NPU 权重传输视为每层固定成本，扩大局部候选窗口，并在 prefill/decode 各阶段内按 MoE 激活相似度形成较大的请求 cohort。
* `auto`：存在 cold buffer 时选择 `offload`，否则选择 `expert`；所有未显式配置的调度参数由运行布局和 vLLM capacity 自动计算。

请求调度通过 `RuntimeConfig` 消费 offload count、buffer 数和接管层数，不直接持有 `RuntimeCore`、`MemoryManager`、`ExoExecutor` 或 `LBVCAdaptor`。请求级激活画像可从 `sampling_params.extra_args`/metadata 读取，也可通过 `record_request_activation()` 在线发布，从而为后续 profiler 请求追踪保留稳定接口。

## 5. 核心抽象

### `RuntimeCore`

`runtime/runtime_core.py` 是 `RuntimeAscendFusedMoE` 与 Runtime 执行层之间的唯一入口。

职责：

* 根据 `runtime_mode` 和 `num_runtime_experts` 初始化运行时布局。
* 创建 `MemoryManager`、`ExoExecutor` 和 `LBVCAdaptor`。
* 在权重加载阶段接收 `RuntimeAscendFusedMoE` 拦截到的专家权重，并转交给 `MemoryManager` 管理。
* 在 forward 阶段接收 profiler 统计结果、当前 token dispatch 信息和 layer/rank 信息。
* 调用 `LBVCAdaptor` 生成或更新 balance 专家映射任务。
* 调用 `LBVCAdaptor` 生成 balance 专家更新任务，再由 `ExpertUpdator` 执行 HCCL 卡间互传。
* cold-buffer forward 前通过 `prepare_combined_experts()` 获得完整本地专家视图、full expert map 和 profiler 所需 slot-to-global 映射。

`RuntimeAscendFusedMoE` 不直接调度专家换入换出，也不直接制定负载均衡策略。

### `MemoryManager`

`runtime/memory_manager.py` 统一管理 CPU expert、NPU cold buffer 和组合专家视图。

职责：

* 保存 CPU 专家权重、shape、dtype、量化 scale/offset 等元数据。
* 管理 CPU pinned memory 选项。
* 在 `offload` 或 `balance` 且 `num_runtime_experts < 0` 时，CPU 只保存本 rank 固定 cold experts，NPU 创建固定数量 cold buffer。
* `ColdExperts` 维护可复用的 NPU cold buffer 参数；W8A8 权重以 per-expert list 形式保存，非量化权重以 tensor 保存。
* `CombinedExperts` 是 forward-only 视图，把 layer 上的 hot expert weights 和 cold buffer weights 拼成一个完整 expert list，不复制权重主体；全卸载时 hot expert list 为空。
* CPU 只保存固定 cold experts，不参与 balance 专家重分布；无 cold buffer 的 balance 不创建 CPU expert storage。

### `ExoExecutor`

`runtime/exo_executor.py` 只负责固定 cold experts 的 CPU-NPU 传输和 unified cold-buffer 准备。

职责：

* 执行固定 cold experts 的 CPU-NPU expert slot 参数传输。
* 支持独立 NPU stream、event 同步和必要的流水线加载。
* 管理每层 `expert_map`，其中前半部分为 hot experts，末尾为固定 cold experts。
* 通过 `_full_expert_map()` 构建 `logical_expert_id -> physical_slot_id` 的完整映射。
* `prepare_combined_experts()` 加载当前层 cold buffer，并返回 `PreparedCombinedExperts(CombinedExperts, expert_map, slot_to_global)`；offload 和带 cold buffer 的 balance 共用该路径，`expert_map` 和 `slot_to_global` 按层缓存并在 layout 更新后失效。
* 不再维护 hot/cold routing maps，也不再生成 cold-only topk 或 mask。

### `ExpertUpdator`

`runtime/exp_updator.py` 负责执行专家更新任务。

职责：

* 接收 `LBVCAdaptor` 生成的专家更新任务。
* 为迁入专家执行 HCCL P2P 卡间互传。
* 权重传输完成后由 `LBVCAdaptor` 发布新的 expert map。

### `LBVCAdaptor`

`runtime/lbvc_adaptor.py` 负责 balance 专家更新任务生成和映射发布。

职责：

* 从 profiler delta 或历史记录获得专家负载信息。
* 调用 `moeload/policy.py` 计算全局专家分布。
* 把 expert map 转换为 NPU slot 更新任务，并保持仍驻留本 rank 的专家物理 slot 不变。
* 将 balance 目标任务交给 `ExpertUpdator` 执行 HCCL 互传。
* 更新 Runtime 执行所需的映射快照。
* 在 `enable_history_mapping = True` 时读取历史负载，调用 policy 生成初始化 expert map，并驱动 Runtime 初始化专家分布。
* 按 `load_collect_interval` 控制负载统计频率，按 `rebalance_interval` 控制专家分布调整频率。
* 每次调整前把 expert 负载按当前 slot 分布投影到 rank 负载；冗余副本按副本数均摊负载，均衡指标统一使用 peak/average ratio（PAR）。
* 策略生成冗余副本时使用相同的副本均摊模型，按目标 rank 负载比、负载差和最大负载选择副本位置。
* 比较当前 rank 的 PAR，超过 `imbalance_threshold` 后生成全局专家重排候选。
* `num_runtime_experts < 0` 时只调整每张卡前面的 resident 热专家，后面的 cold experts 始终固定。
* `num_runtime_experts >= 0` 时对全部 NPU slots 做全局负载均衡。
* 同一周期先收集所有目标层的负载 delta，按 expert 数分组后合并执行 collective；各层只生成候选计划，不立即通信。
* `BalanceCoordinator` 按预测 PAR 收益对候选层排序，并应用 `rebalance_max_layers` 和 `rebalance_min_improvement` 预算。
* 选中层的 HCCL P2P 操作由 `ExpertUpdator.transfer_many()` 合并为一次 batch，完成后只执行一次 barrier，再发布全部映射。

### `BalanceCoordinator`

`runtime/balance_coordinator.py` 是跨层 rebalance 预算控制器。

职责：

* 收集同一 rebalance 周期内各层的候选布局、当前/目标 PAR 和迁移 expert 数。
* 丢弃收益小于 `rebalance_min_improvement` 的候选。
* 按预测收益、目标 PAR、迁移数和 layer id 做确定性排序。
* 每周期最多选择 `rebalance_max_layers` 层，确保所有 EP rank 得到一致选择结果。

### `Profiler`

`moeload/profiler.py` 是 Runtime 共享的专家负载统计模块。

统计内容：

* 每个 rank、layer、expert 处理的 token 数。
* dispatch 后可直接记录的专家访问元数据。
* 短期和历史负载变化。
* cold-buffer unified compute 下通过 slot-to-global 映射把 physical slot token 数还原到 global expert 维度。

Profiler 只记录和输出负载，不决定专家映射。

Profiler 统计在设备侧累加，避免 forward 每步同步到 CPU；保存历史或读取 delta 时才同步。Profiler 缓存 layer 的 slot-to-global key 和设备侧映射，减少采样步重复转换。

### `Policy`

`moeload/policy.py` 是专家映射策略模块。

职责：

* 根据当前负载或历史负载计算 expert map。
* 为 `offload` 模式提供 hot-first、cold-fixed 的初始 expert map。
* 为 `balance` 模式提供全局专家重分布或冗余专家加载策略，并保证目标分布的负载评估与运行期 rebalance 判断一致。
* 只输出策略结果，不执行参数传输，不修改 Runtime 对象。
* 带 cold buffer 时先把每张卡固定 cold experts 的负载作为不可迁移基线，再分配 resident experts。
* resident experts 先使用容量约束的 LPT 放置；冗余场景为热点副本预留各 rank slot，并按副本数重新计算负载份额。
* 动态调整采用两阶段规划：所有层先生成低成本 LPT/副本候选；`BalanceCoordinator` 选中有限层后，再从最高负载 rank 开始执行有界 pair-swap 局部搜索，按最大负载、负载差和方差的字典序目标继续优化。
* 同时评估从当前布局开始的低迁移候选；均衡目标相同时优先保留更多原有 rank/slot。

## 6. 计算层职责

### `layer/fused_moe.py`

定义 Runtime 版本的 Ascend FusedMoE，是单层 MoE 的接管入口。

职责：

* 根据 `runtime_mode` 和 `num_runtime_experts` 决定 `create_weights` 使用的 NPU expert slot 数量。
* 初始化该层对应的 `RuntimeCore`。
* 在权重加载阶段拦截专家权重，并交给 `RuntimeCore`。
* Forward 阶段调用 profiler、通信层、quant method 和 `RuntimeCore`。

不同配置下的权重布局：

* `runtime_mode = profile`：保持原生专家布局和原生计算路径，只记录专家负载。
* `num_runtime_experts < 0`：NPU 创建减少后的常驻专家 slots 和 cold buffer slots；常驻专家数可以为 0。
* `num_runtime_experts = 0`：NPU 创建原生本地专家 slots，不启用专家迁移。
* `num_runtime_experts > 0`：NPU 创建原生本地专家 slots 和 redundant slots；该模式不创建 CPU expert storage，冗余专家权重通过初始化加载或后续 HCCL 更新进入对应 slot。

### `layer/quant_method.py`

实现 Runtime FusedMoE 的非量化和 W8A8 计算路径。

职责：

* 复用 vLLM-Ascend 的 expert selection 和 token dispatcher。
* 无 cold buffer 时继续走 vLLM-Ascend 原生 `moe_comm_method.fused_experts()`，并可携带 `log2phy` 和冗余专家信息。
* `offload` 或 `balance` 使用 cold buffer 时，调用 `RuntimeCore.prepare_combined_experts()` 得到 hot/cold 组合专家视图。
* 对组合专家视图调用本地 `_runtime_fused_experts()`，保留原始 `topk_ids`、`topk_weights`、`mc2_mask`、`pertoken_scale`，不再构造冷热拆分 routing。
* W8A8 路径使用 `w13_weight_list`、`w2_weight_list`、`w13_weight_scale_fp32_list`、`w2_weight_scale_list`；非量化路径使用 `w13_weight_list`、`w2_weight_list`。
* 全卸载时跳过 0 expert 的原生权重后处理，直接处理 CPU cold weights，并向 combined experts 提供空 hot list。
* combined experts 下，即使当前通信类型为 `FUSED_MC2`，也复用 MC2 token dispatch/combine，加本地 grouped matmul MLP，不调用 FUSED_MC2 的单 tensor FFN 融合算子。

### `layer/moe_mlp.py`

本地 MoE MLP 计算实现，从 vLLM-Ascend `moe_mlp.py` 抽取并收敛为 Runtime 所需接口。

职责：

* `unified_apply_mlp()` 同时支持非量化和 W8A8。
* 非量化路径使用 `torch_npu.npu_grouped_matmul` 接收多个 weight tensor，完成 gate/up、SwiGLU 和 down。
* W8A8 非 MC2 路径优先使用 list-capable custom op；当无法使用 custom op 或单 tensor fusion 不适用时，回退到 `npu_grouped_matmul` list 路径。
* W8A8 `MC2/FUSED_MC2` 路径支持多个 weight list；多 list 情况下避免误走只接受单个 weight tensor 的 `npu_grouped_matmul_swiglu_quant`。
* 不支持 W8A8 offset 路径；`w1_offset` 存在时直接报错。

### `layer/routing.py`

当前只保留 `dispatch_with_local_experts()`。

职责：

* 在一次 Runtime MLP 调用范围内，临时把 vLLM-Ascend token dispatcher 的本地专家数量改为当前 weight list 的长度。
* 调用结束后恢复 dispatcher 原值。
* 不再构造 routing view，不再提供 hot/cold mask、cold-only topk 或输出 scatter。

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
* `num_runtime_experts <= 0`，其中 `< 0` 表示每卡卸载专家数量。

初始化流程：

1. 初始化阶段根据 `num_runtime_experts` 计算每卡需要卸载的 cold experts。
2. `create_weights` 只为 hot experts 创建常驻 NPU 参数；全卸载时 hot expert 数为 0。
3. `RuntimeCore` 将专家权重交给 `MemoryManager`，CPU 只保存本 rank cold experts。
4. `ExoExecutor` 创建 NPU cold buffer，数量由 `offload_count` 和 `num_buffers` 决定。
5. 如果 `enable_history_mapping = True`，`Policy` 根据历史负载生成 hot-first expert map；否则使用默认本地专家顺序。

Forward 流程：

1. 原生 expert selection 生成 `topk_ids` 和 `topk_weights`。
2. `RuntimeCore.prepare_combined_experts()` 触发当前层 cold buffer 加载，并构建 `CombinedExperts`。
3. `CombinedExperts` 将 hot expert list 和 cold buffer list 拼接为一个完整本地 expert list；全卸载时直接使用 cold expert list。
4. `ExoExecutor` 提供完整 `expert_map`，token dispatcher 按原始 `topk_ids` 映射到 physical slot。
5. `quant_method._runtime_fused_experts()` 调用 token dispatch。
6. `moe_mlp.unified_apply_mlp()` 对完整 expert list 执行一次 grouped matmul MLP。
7. token dispatcher 执行 combine。
8. 如需统计负载，profiler 使用 `slot_to_global` 将 slot token 数映射回 global expert。
9. `ExoExecutor` 预取下一层 cold buffer。

Offload 模式只更新 cold buffer 内容，不拆分冷热计算，不对 `topk_ids`、mask 或 token 行做额外改写。

带 cold buffer 的 balance 模式复用同一条 unified cold-buffer compute 路径；差异在于前面的 resident hot experts 可由 `LBVCAdaptor` 动态重排，末尾 cold experts 始终保持固定。

### Balance

适用条件：

* `runtime_mode = balance`。
* `num_runtime_experts < 0`、`= 0` 或 `> 0` 分别表示 cold buffer、原生 slot 数和 redundant slots。

流程：

1. 初始化阶段建立全局 rank/layer/expert 视图。
2. CPU 侧仅在存在 cold buffer 时保存固定 cold experts。
3. `create_weights` 根据 `num_runtime_experts` 创建 resident、cold buffer 或 redundant slots。
4. 如果 `enable_history_mapping = True`，初始化阶段读取历史负载并由 `Policy` 生成初始 expert map 和 `log2phy`。
5. Forward 阶段按 `load_collect_interval` 将专家 token 统计写入 profiler。
6. 按 `rebalance_interval` 读取所有目标层的 profiler delta，把多层 tensor 堆叠后跨 EP rank 合并聚合。
7. `LBVCAdaptor` 基于当前完整 slot 分布计算每个 rank 的 PAR；存在 cold buffer 时先计入固定 cold experts 的负载。
8. 如果窗口 token 数小于 `rebalance_min_step_tokens`，或 PAR 不超过 `imbalance_threshold`，该层不生成候选。
9. 超过阈值后，`Policy` 在固定 cold 负载基线上快速计算 resident/redundant slots，形成第一阶段候选。
10. `LBVCAdaptor` 将目标 slot 分布对齐到当前物理 slot，生成包含预测收益和迁移数的候选 `ExpertUpdateTask`。
11. `BalanceCoordinator` 对所有层候选排序，按配置限制本周期实际替换层数；只对选中层执行 pair-swap 精调并重建迁移任务。
12. `ExpertUpdator` 将选中层的 P2P 任务合并成一次 HCCL batch；传输完成后统一 barrier。
13. `LBVCAdaptor` 原子发布选中层的新 expert map、`log2phy` 和 profiler slot-to-global 映射。

Balance 模式不再局限于 pair 内对端专家副本，而是以全局负载为输入，直接调整 NPU 上由 `create_weights` 创建出的常驻专家参数和冗余专家参数，实现真正的全局负载均衡。

不同 `num_runtime_experts` 的传输语义：

* `< 0`：每张卡前面的 resident 热专家参与全局均衡并走 HCCL，末尾 cold buffer 专家保持固定并从 CPU 加载；resident 数为 0 时不执行 HCCL 热专家迁移。
* `= 0`：原生数量的 slots 参与全局均衡，迁入专家走 HCCL，计算继续使用原生 fused experts 路径。
* `> 0`：原生 slots 加 redundant slots 共同参与全局均衡，热点专家可拥有副本，迁入专家走 HCCL。

### Request Scheduler

适用条件：

* `enable_scheduler = True`。
* 当前使用 vLLM V1 Scheduler。

初始化流程：

1. 插件注册阶段调用 `apply_scheduler_patch()`。
2. Runtime 把完整 `RuntimeConfig` 传给 scheduler patch，调度器在 vLLM Scheduler 实例建立后结合其 token/sequence capacity 解析最终参数。
3. Patch 在原生 `Scheduler.schedule()` 前局部重排 running 和 FCFS waiting 队列，再把 token budget、KV cache 分配、抢占和输出构造完整委托给 vLLM；priority queue 保持原生优先级语义。

调度流程：

1. 自动模式根据 `max_num_seqs`、cold expert 数量、Runtime 层数和 buffer 数推导 reorder window、decode reserve、画像阈值和画像宽度。
2. 调度器识别 running 请求当前处于 chunked prefill 还是 decode；先安排达到 `scheduler_decode_reserve_ratio` 的完整 decode 激活簇，再安排 prefill 簇，最后安排剩余 decode 簇。
3. waiting 队列和各阶段 running 请求都在局部窗口内按 MoE 激活画像余弦相似度聚类；同簇请求保持相邻，簇内 prefill 优先长请求，decode 保持到达顺序。
4. 请求画像优先读取实际推理产生的在线画像 registry；首次运行前可读取 `vllm_ascend_moe_activation`、`moe_activation`、`expert_activation`、`expert_histogram` 或 expert id 列表作为先验，再回退到 expert cost/throughput。
5. Patch 保留原生 token budget、KV cache allocation、preemption、structured output 和 speculative decode 逻辑，只改变进入原生算法前的有限窗口顺序。
6. 无法调度的请求仍按原 Scheduler 规则保留或放回队列。

在线激活反馈流程：

1. `NPUModelRunner._prepare_inputs()` 完成后，Runtime 读取本 step 的 `req_ids` 和每请求 token 数，建立扁平 token 到 request 的区间映射。
2. Runtime MoE 完成 `select_experts()` 后，把逻辑 `topk_ids` 交给 worker 画像收集器。
3. 收集器在设备侧使用 `index_add_` 聚合 request/layer/expert 计数；ACLGraph 末尾 padding 不计入画像，无法可靠匹配请求区间的短 token 布局直接跳过。
4. 每个请求首次执行时采样；后续按 `scheduler_feedback_interval` 周期采样，并从均匀选取的有限层中只保留 top experts。
5. `NPUModelRunner.sample_tokens()` 把纯 Python 字典 payload 附加到 `ModelRunnerOutput`，复用 vLLM 现有 pickle IPC 返回 Scheduler，不新增 RPC。
6. `Scheduler.update_from_output()` 在下一调度步前合并请求画像；请求结束或 abort 后同时清理 scheduler registry 和 worker seen state。

当前反馈边界：

* eager 模式可逐 step 获得真实 `topk_ids`；ACLGraph replay 不执行 Python 路由采集时不会生成错误画像，但只能等待后续 graph-aware hook。
* sequence parallel 导致本 rank token 数小于请求区间总数时跳过该批，避免把局部 token 归属到错误请求。
* 第一版以非 PP 或 Runtime MoE 位于输出 PP stage 为目标；跨 PP stage 的画像归并需要后续 collective/side channel。
* 当前 patch 面向 vLLM-Ascend V1 `NPUModelRunner`，V2 model runner 尚未接入反馈 hook。

该调度策略与专家卸载、负载均衡的关系：

* 对 offload 模式，自动重排窗口随 fixed cold-buffer transfer pressure 增长，使 MoE 激活相似请求形成更大的连续 cohort，并依靠 V1 原生 token budget 填充摊薄 CPU-NPU 加载成本。
* 对 balance 模式，激活画像或专家代价 hint 可把相似请求组成稳定批次，为后续结合当前 expert map 评估 rank 热点成本预留接口，但不直接触发专家迁移。
* scheduler 不读取 Runtime expert map，不执行 CPU-NPU 传输，也不生成 HCCL 更新任务；专家布局仍由 `RuntimeCore`、`ExoExecutor` 和 `LBVCAdaptor` 管理。

## 8. 历史映射初始化

当 `enable_history_mapping = True`：

1. `RuntimeCore` 初始化后通知 `LBVCAdaptor` 或直接通过 `Policy` 读取历史负载。
2. Runtime 读取 `load_history_path` 中的历史负载记录。
3. Runtime 将历史负载、当前 rank 信息、layer 信息、NPU slot 信息和 `num_runtime_experts` 传给 `Policy`。
4. `Policy` 返回初始化 expert map。
5. offload 模式使用 hot-first expert map 固定 hot/cold 排布；cold 部分后续只通过 `ExoExecutor` 加载到 cold buffer。
6. balance 模式在权重创建前使用历史负载生成初始 local expert map、`log2phy` 和 slot-to-global 映射。
7. Runtime 发布初始化后的映射快照；后续动态调整才通过 `ExpertUpdator` 执行 HCCL 互传。

该流程适用于 `offload` 和 `balance`，但语义不同：

* `offload`：初始化固定 hot/cold expert 排布，forward 时拼接 hot/cold weights 统一计算。
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

* `tests/only_run.py`：验证插件注册、配置解析、模型加载和单次 Runtime 推理。
* `tests/naive_run.py`：运行原生 vLLM-Ascend 推理基准，作为正确性和性能对照。
* `tests/runtime_run.py`：统一验证 Runtime 模式。
* `tests/template.py`：通用运行模板。
* `tests/lbvc_template.py`：LBVC/balance 运行模板。
* `tests/naive_dp_test.py`：原生 DP 场景测试入口。
* `tests/runtime_dp_test.py`：Runtime DP 场景测试入口。
