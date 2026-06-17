# MoE 专家负载均衡实现计划

## 1. 目标与原则

负载均衡按三个阶段推进，每个阶段都应能独立运行和验证：

1. 统计每个卸载 MoE 层的专家负载，并在推理结束后保存。
2. 根据历史负载选择常驻热专家，替代当前“前 `num_hot_experts` 个专家常驻”的策略。
3. 根据实时负载动态调整常驻热专家和冷专家。

实现代码主要放在 `src/loadbalance/`。`src/offload/`、`src/layer/` 和测试脚本只保留必要接入点。

负载统计和映射更新可以参考 vLLM-Ascend EPLB 的思路，但本项目保持 rank 内独立调度，不做跨 rank 专家交换。

## 2. 当前配置

`OffloadConfig` 已增加以下字段：

```python
load_stats_path: str | None = None
load_balance_mode: str = "none"  # none、history、dynamic
dynamic_update_interval: int = 32
dynamic_max_swaps: int = 2
```

配置语义：

* `load_stats_path`：负载统计目录。每个 rank 读写自己的 JSON 文件。
* `load_balance_mode`：负载均衡模式。`none` 只统计和保存；`history` 使用历史负载初始化热专家；`dynamic` 在历史初始化后进行实时调整。
* `dynamic_update_interval`：动态调度检查间隔。
* `dynamic_max_swaps`：单次动态调度最多交换的专家数。

当前 rank 文件命名规则：

```text
{load_stats_path}/{basename(load_stats_path)}_rank{rank}.json
```

例如：

```text
load_records/only_run_load_stats/only_run_load_stats_rank0.json
load_records/only_run_load_stats/only_run_load_stats_rank1.json
```

## 3. 模块设计

```text
src/loadbalance/
├── __init__.py
├── load_stats.py
├── policy.py      # 后续新增
└── scheduler.py   # 后续新增
```

### `load_stats.py`

已实现运行时负载统计：

* 接收 `select_experts` 后的全局 `topk_ids`。
* 使用 `full_expert_map` 过滤当前 rank 本地专家，避免重复统计其他 rank 的专家。
* 统计结果按全局专家编号保存。
* 保存和读取都只处理当前 rank 文件。
* 不做 `all_reduce`，不生成全局合并文件。
* 写文件使用临时文件、`flush`、`fsync` 和 `os.replace`，保证写入完成后再替换目标文件。

文件结构：

```json
{
  "version": 1,
  "metadata": {
    "rank": 0,
    "offload_mode": "manual",
    "load_balance_mode": "none",
    "offloaded_layer_ids": [0, 16],
    "layers": {
      "0": {
        "global_num_experts": 128,
        "local_num_experts": 64,
        "local_global_expert_ids": [0, 1]
      }
    }
  },
  "layers": {
    "0": [0, 3, 1]
  }
}
```

### `policy.py`

后续负责根据历史负载和实时负载生成专家驻留策略：

* `HistoryLoadPolicy`：根据当前 rank 的历史负载选择热专家。
* `DynamicLoadPolicy`：根据实时负载生成换入、换出计划。
* 稳定排序规则：负载高优先，负载相同按专家编号升序。
* 维护每层 `full_local_expert_id <-> resident_slot` 和 `full_local_expert_id <-> cold_slot` 映射。

### `scheduler.py`

后续负责动态调度流程：

* 在安全的 forward 边界检查是否达到调度间隔。
* 调用 `DynamicLoadPolicy` 生成交换计划。
* 同步版本先保证正确，再优化异步传输。
* 动态层在 CPU 保存全部本地专家权重，作为换入 resident slot 的来源。

## 4. 当前已完成内容

### 4.1 配置接入

已修改 `src/offload_config.py`：

* 增加 `load_stats_path`、`load_balance_mode`、`dynamic_update_interval`、`dynamic_max_swaps`。
* 校验 `load_balance_mode` 只能为 `none`、`history`、`dynamic`。
* 校验动态调度参数的基本边界。

### 4.2 统计对象

已新增 `src/loadbalance/load_stats.py`：

* 为每个注册层维护全局专家计数。
* 支持从当前 rank 文件读取历史统计。
* 支持更新统计、查询单层负载、保存当前 rank 文件。
* 保存路径由 `load_stats_path` 目录和当前分布式 rank 决定。

### 4.3 推理路径接入

已修改 `src/layer/quant_method.py`：

* 在 `select_experts` 得到最终 `topk_ids` 后调用统计接口。
* 如果启用了 `enable_force_load_balance`，统计的是替换后的实际 `topk_ids`。

已修改 `src/offload/executor.py`：

* 创建并持有 `ExpertLoadStats`。
* 在卸载层注册时同步注册统计层信息。
* 提供 `record_load()` 和 `save_load_stats()`。

### 4.4 保存入口

已在 `src/utils.py` 增加 `OffloadWorkerExtension`：

* 通过 vLLM `worker_extension_cls` 注入 worker。
* `only_run.py` 在 `generate` 后调用 `llm.collective_rpc("save_load_stats")`。
* 每个 worker 保存自己的 rank 文件。
* 保存完成后再关闭 `llm.llm_engine.engine_core`。

`tests/only_run.py` 已配置：

* `load_stats_path=load_records/only_run_load_stats`
* `worker_extension_cls="src.utils.OffloadWorkerExtension"`
* `generate -> save_load_stats -> shutdown_llm`

阶段一当前完成条件：

* `only_run.py` 能在推理结束后为每个 rank 生成独立负载文件。
* 负载统计按全局专家编号保存。
* 多 rank 不做通信合并，后续策略按 rank 独立读取。

## 5. 后续计划

### 阶段二：根据历史负载初始化热专家

#### 1. 历史策略

新增 `src/loadbalance/policy.py`：

* 实现 `HistoryLoadPolicy`。
* 读取当前 rank 的负载文件。
* 从当前 rank 本地专家中按负载选择 `num_hot_experts` 个热专家。
* 无历史文件、层缺失或专家数不匹配时回退到当前前缀热专家策略。
* 维护 resident/cold 显式映射。

#### 2. 模型加载前初始化

修改 `src/layer/fused_moe.py` 和 `src/offload/executor.py`：

* 在创建 resident 参数前完成历史热专家选择。
* resident 参数数量仍为 `num_hot_experts`。
* resident slot 不再默认等于前 `num_hot_experts` 个本地专家。
* 权重加载根据 policy 映射写入 resident slot 或 CPU cold storage。

修改 `src/offload/routing.py`：

* 使用 policy 提供的 resident/cold 映射构建路由。
* 删除对“前缀热专家、后缀冷专家”的隐式依赖。

#### 3. 验证

* 构造偏斜历史负载，验证高负载专家进入 resident slots。
* 验证权重加载到正确的 resident/cold slot。
* 验证无历史文件时保持当前行为。
* 使用 `only_run.py` 对比推理结果，确认历史调度不破坏输出。

阶段二完成条件：

* 任意本地专家都可以成为热专家。
* resident/cold 路由不依赖连续专家编号。
* 使用历史热专家后推理结果正确。

### 阶段三：实时动态调度

阶段三先实现同步正确版本，再优化异步传输。

#### 1. 动态策略

扩展 `src/loadbalance/policy.py`：

* 实现 `DynamicLoadPolicy`。
* 根据最近统计窗口选择新的热专家。
* 每次最多交换 `dynamic_max_swaps` 个专家。
* 只有调度完成后才提交新映射版本。

修改 `src/offload/memory_manager.py`：

* dynamic 模式下保存动态层的全部本地专家 CPU 权重。
* CPU 权重按完整本地专家编号索引。

#### 2. 同步动态交换

新增 `src/loadbalance/scheduler.py`：

1. 在层 forward 开始前检查调度间隔。
2. policy 生成换入专家、换出专家和目标 slot。
3. 等待该层相关计算和预取 stream 完成。
4. 从 CPU 将换入专家复制到 resident slot。
5. 将换出专家从 resident slot 复制到当前 cold buffer slot。
6. 等待复制完成。
7. 提交 resident/cold 映射并增加版本号。

#### 3. 异步优化

同步版本正确后再优化：

* 使用独立 NPU stream 执行专家交换。
* 将下一层调度与当前层计算重叠。
* 只更新发生变化的专家权重和映射项。

#### 4. 验证

* 人工改变实时负载，验证热专家按预期交换。
* 验证换出专家复制到 cold buffer 后权重一致。
* 验证调度前后输出与固定布局基线一致。
* 多次连续调度压力测试。
* W8A8 权重、scale、offset 同步交换测试。

阶段三完成条件：

* 动态交换期间权重和映射版本一致。
* 调度后推理结果正确。
* 热专家命中率相对固定前缀策略提升。
* 性能收益大于交换和同步开销。

## 6. 建议提交顺序

1. `loadbalance: add per-rank load statistics persistence`
2. `loadbalance: select resident experts from historical rank load`
3. `loadbalance: introduce explicit resident and cold mappings`
4. `loadbalance: add synchronous dynamic expert scheduler`
5. `loadbalance: overlap dynamic expert transfers with execution`

每个提交只实现一个可验证能力，并保留 `load_balance_mode="none"` 的原有路径作为回归基线。
