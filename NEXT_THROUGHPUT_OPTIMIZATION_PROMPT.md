# 下一轮吞吐优化提示词：Qwen3-235B-A22B Expert-wise Offload

仓库：`/workspace/Huawei/vLLM_Ascend_Inference`

目标模型：`/workspace/models/Qwen3-235B-A22B`

当前结论：对 Qwen3-235B-A22B 的吞吐优化，后续重点转向 `expert_wise`。`layer_wise` 可以释放更多 KV cache，但在当前实现下每个 offload 层搬整层 expert，decode 阶段表现为 worker 满 CPU、NPU AICore 接近 0，不适合这轮“尽快提升整体吞吐”的目标。

## 已测基线

naive 运行命令：

```bash
MODEL_PATH=/workspace/models/Qwen3-235B-A22B \
BATCH_SIZE=128 MAX_LENGTH=64 MAX_NEW_TOKENS=64 \
WORLD_SIZE=8 UTILIZATION=0.98 \
python naive_run.py
```

naive 结果：

- `output_tokens=8164`
- `load_seconds=51.109`
- `generate_seconds=24.139`
- `output_tokens_per_second=338.213`
- KV cache：约 `42,880` tokens

layer-wise 观察：

- 修复后可完成启动和 profile。
- KV cache 可提升到约 `87k-93k` tokens。
- decode 阶段长期无 token 产出，worker 接近 100% CPU，NPU AICore 接近 0。
- 初步判断瓶颈是整层 expert CPU->NPU 搬运和 Python 调度开销，当前不作为 235B 吞吐主线。

## 之前已经做过的改进

1. 运行脚本参数化
   - `naive_run.py`、`offloading_run.py` 支持通过环境变量切换 `MODEL_PATH`、batch、长度、world size、utilization 等参数。
   - 可直接跑 235B，不再手改脚本。

2. expert-wise 快路径
   - 增加大 batch fast path，避免插件额外做一次 routing select。
   - 增加 small batch single-select 方向的基础逻辑，减少重复路由开销。

3. expert-wise compact/offload 稳定性
   - 增加 `partition_scope=local_rank`，避免 global suffix 只让部分 rank 发生 offload。
   - 增加 no-shrink / capacity-unsafe skip，避免 compact 后没有 HBM 收益却引入调度和 copy 开销。
   - `wait_for_prefetch()` 对同一 event 去重等待。
   - offload summary 增加 skip、copy、prefetch、routing 等聚合计数。

4. compact dispatcher 优化
   - AllGather compact dispatch patch 改为 activate/deactivate 方式，避免每个 forward 反复 monkey patch。
   - 保留原始 dispatch，便于 fallback。

5. layer-wise 兼容修复
   - 修复 Qwen3-235B-A22B expert 权重布局不一致导致的 `w13_weight` shape mismatch。
   - owner buffer 标记为已加载，避免 layer 0 首次 forward 不必要拷贝。
   - 修复 layer-wise prefetch distance 语义，使 `prefetch_distance=16` 表示提前 16 个模型层。
   - `offloading_run.py` 中 layer-wise 默认开启 async prefetch。

6. 验证状态
   - `tests/test_expert_wise_scheduler.py` 和 `tests/test_layer_wise_scheduler.py` 当前通过。
   - 关键 Python 文件已做过 `py_compile` 验证。

## 接下来边推理边优化的主计划

### P0：先建立 expert-wise 235B 可运行基线

先确认环境干净：

```bash
ps -ef | rg 'offloading_run.py|naive_run.py|EngineCore|VLLM::Worker' | rg -v rg
npu-smi info
```

先用小输出快速验证启动和 summary：

```bash
MODEL_PATH=/workspace/models/Qwen3-235B-A22B \
OFFLOAD_MODE=expert_wise \
BATCH_SIZE=128 MAX_LENGTH=64 MAX_NEW_TOKENS=8 \
WORLD_SIZE=8 UTILIZATION=0.98 \
PREFETCH_DISTANCE=1 OFFLOAD_INTERVAL=1 \
PARTITION_SCOPE=local_rank \
RESIDENT_EXPERTS=96 NPU_CACHE_CAPACITY=4 \
COMPACT_NPU_CACHE=1 SKIP_NO_SHRINK_COMPACT=1 \
ENABLE_LARGE_BATCH_FAST_PATH=1 ENABLE_SINGLE_SELECT_FORWARD=1 \
LOG_TRANSFERS=0 \
python offloading_run.py
```

观察重点：

- 是否能完成 generate。
- `output_tokens_per_second` 是否接近或超过 naive 的 `338.213 tok/s`。
- summary 中是否有实际 `offloaded_experts`，且不是 no-shrink skip。
- `copy_count`、`prefetch_count`、`prefetch_wait_count` 是否过高。
- NPU AICore 是否有计算利用率，避免重现 layer-wise 的 CPU 忙等。

如果短输出可用，再跑与 naive 相同参数：

```bash
MODEL_PATH=/workspace/models/Qwen3-235B-A22B \
OFFLOAD_MODE=expert_wise \
BATCH_SIZE=128 MAX_LENGTH=64 MAX_NEW_TOKENS=64 \
WORLD_SIZE=8 UTILIZATION=0.98 \
PREFETCH_DISTANCE=1 OFFLOAD_INTERVAL=1 \
PARTITION_SCOPE=local_rank \
RESIDENT_EXPERTS=96 NPU_CACHE_CAPACITY=4 \
COMPACT_NPU_CACHE=1 SKIP_NO_SHRINK_COMPACT=1 \
ENABLE_LARGE_BATCH_FAST_PATH=1 ENABLE_SINGLE_SELECT_FORWARD=1 \
LOG_TRANSFERS=0 \
python offloading_run.py
```

### P1：根据日志调 expert-wise 配置

按吞吐和 summary 做网格小测，每次只改一个关键变量：

1. `RESIDENT_EXPERTS`
   - 候选：`112`、`104`、`96`、`88`
   - 目标：找到 HBM 释放和 CPU->NPU copy 开销的平衡点。

2. `NPU_CACHE_CAPACITY`
   - 候选：`2`、`4`、`8`
   - 如果 routed offloaded experts 经常超过 capacity，增大 cache。
   - 如果 copy 开销高且 HBM 紧张，优先降低 offloaded 数量，而不是盲目增大 cache。

3. `OFFLOAD_INTERVAL`
   - 候选：`1`、`2`、`4`
   - 如果每层都 offload 导致 copy 过多，拉大 interval。
   - 对 235B 首先以吞吐为目标，不追求最大 KV cache。

4. `PREFETCH_DISTANCE`
   - 候选：`1`、`2`
   - 观察 `prefetch_wait_count` 和吞吐。等待多说明预取太近或 copy 太慢。

### P2：低风险代码优化

优先改动范围小、可用真实短测快速验证的点：

1. 减少热路径 Python 分配
   - 缓存每层常用 set/list。
   - 避免每次 forward 构造重复 closure/context。

2. 继续收敛 single-select
   - 小 batch 下插件和 vLLM 原始 MoE 可能重复 select。
   - 目标是在 `ExpertWiseAscendFusedMoE.forward_impl` 中只 select 一次，同时复用 `topk_ids/topk_weights` 做调度和 fused experts。

3. 优化 CPU->NPU copy
   - 当前 expert-wise 仍可能逐 expert、逐 tensor copy。
   - 下一步做 packed CPU store 或连续 slot 批量 copy。
   - 先只支持 unquantized BF16，保证 235B 路径正确。

4. summary 更精细
   - 增加每层 copy/prefetch/select 计数。
   - 让下一次调参能快速定位是哪些层拖慢。

### P3：中高风险优化

这些先不作为第一轮实现，除非 P1/P2 后仍无法接近 naive：

1. resident/offloaded compute split
   - resident experts 先算，offloaded experts 异步加载后再算，最后合并。
   - 这是理论收益最大的重构，但涉及通信、dispatch、grouped matmul 和输出合并，风险较高。

2. MC2/FusedMC2 compact dispatch
   - 当前 compact dispatch 主要覆盖 AllGather。
   - 如果 vLLM-Ascend 的 MC2/FusedMC2 在 235B batch=128 更快，需要补对应 patch。

3. 非 eager / 编译路径
   - 可测试 `ENFORCE_EAGER=0`。
   - 之前 30B 上非 eager 有过长时间卡住，需独立验证，不和主要优化混在一起。

## 推荐下一步执行顺序

1. 跑 `expert_wise` 235B 短输出基线，确认能生成。
2. 跑同参数 `MAX_NEW_TOKENS=64`，和 naive `338.213 tok/s` 对比。
3. 如果吞吐低，先调 `RESIDENT_EXPERTS`、`NPU_CACHE_CAPACITY`、`OFFLOAD_INTERVAL`。
4. 如果 copy/prefetch 计数明显过高，做 packed copy 或减少 offload 层。
5. 如果 routing/select 计数高，优先完成 single-select forward。
6. 每次改动都先跑短输出，再跑完整 64 tokens。

## 交接提示词

从这里继续：

> 继续在 `/workspace/Huawei/vLLM_Ascend_Inference` 优化 Qwen3-235B-A22B 的 expert-wise offload 吞吐。先确认没有后台 vLLM 进程，然后用 `MODEL_PATH=/workspace/models/Qwen3-235B-A22B OFFLOAD_MODE=expert_wise BATCH_SIZE=128 MAX_LENGTH=64 MAX_NEW_TOKENS=8 WORLD_SIZE=8 UTILIZATION=0.98 PARTITION_SCOPE=local_rank RESIDENT_EXPERTS=96 NPU_CACHE_CAPACITY=4` 跑短输出基线。根据吞吐、summary 和 NPU 利用率调参，再跑 `MAX_NEW_TOKENS=64` 对比 naive 的 `338.213 tok/s`。优先做 expert-wise 的低风险吞吐优化，不再把 layer-wise 作为 235B 吞吐主线。
