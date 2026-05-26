# 下一轮吞吐优化提示词：Qwen3-235B-A22B Expert-wise Offload

仓库：`/workspace/Huawei/vLLM_Ascend_Inference`

目标模型：`/workspace/models/Qwen3-235B-A22B`

当前目标：提高 **expert-wise offload 后的吞吐量**，尽量接近原始 vLLM-Ascend/native 路径。当前不把 FlashInfer 作为讨论重点，因为用户关心的是卸载后的吞吐表现，而不是 FlashInfer/native 开关本身。

## 已知基线

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

- 可释放更多 KV cache，约 `87k-93k` tokens。
- decode 阶段长期无 token 产出，worker 接近满 CPU，NPU AICore 接近 0。
- 判断瓶颈是整层 expert CPU->NPU 搬运和 Python 调度，当前不作为 235B 吞吐主线。

## 本轮已经完成

### 代码改动

1. compact sizing 和 summary 诊断
   - 在 `src/fused_moe/fused_moe.py` 增加 `expert_wise_compact_sizing`。
   - summary 增加：
     - `compact_sizing_examples`
     - `total_chunked_compact_forward_count`
     - `total_chunked_compact_piece_count`
     - `total_chunked_compact_token_count`
     - `total_chunked_compact_full_token_count`
   - 目的：区分真实 compact offload、no-shrink skip、capacity unsafe skip。

2. chunked compact forward
   - 支持 compact cache 下 `offloaded_local > NPU_CACHE_CAPACITY`。
   - 之前这种配置会因为 capacity unsafe 跳过真实卸载；现在可以将 offloaded experts 分 chunk 加载和计算。
   - 相关开关：
     - `ENABLE_CHUNKED_COMPACT_FORWARD=1`
     - `ExpertWiseConfig.enable_chunked_compact_forward=True`

3. profile/warmup compact dispatch 修复
   - 修复 profile run 中 compact AllGather dispatch 未激活导致的维度错误。
   - 之前真实 compact 配置会遇到类似：
     - `groupList size 16 should equal weight dim0 14`
   - 现在 profile/warmup 会在 compact 权重 slot 数下激活 compact dispatch。

4. chunked dispatch 次数优化
   - 将 resident experts 合并到第一个 offloaded chunk。
   - 例如 `resident + [offloaded chunk 1]` 作为第一段，后续只跑剩余 offloaded chunk。
   - 这样 `NPU_CACHE_CAPACITY=3`、本地 offloaded=4 时，常见 piece 数从 3 降到 2。

5. token 子集筛选实验
   - 实现了只对当前 chunk 命中 token 做 fused_experts，再 scatter 回全量输出。
   - 实测变慢，因此默认关闭：
     - `ENABLE_COMPACT_CHUNK_TOKEN_FILTER=0`
     - `ExpertWiseConfig.enable_compact_chunk_token_filter=False`
   - 保留为实验开关，不建议作为默认吞吐路径。

### 验证

已通过：

```bash
pytest -q tests/test_expert_wise_scheduler.py
python -m py_compile src/config/expertwise_config.py src/fused_moe/fused_moe.py src/expert_wise/manager.py src/utils/summary.py offloading_run.py
```

当前结果：

- `16 passed`
- `py_compile` 通过

## 已测配置和结果

### 1. no-shrink 配置：`RESIDENT_EXPERTS=96 / NPU_CACHE_CAPACITY=4`

命令要点：

```bash
RESIDENT_EXPERTS=96
NPU_CACHE_CAPACITY=4
PARTITION_SCOPE=local_rank
COMPACT_NPU_CACHE=1
SKIP_NO_SHRINK_COMPACT=1
```

结果：

- 短输出：约 `385.598 tok/s`
- 64 tokens：约 `460.527 tok/s`
- 但这不是有效卸载结果：
  - `total_no_shrink_skipped_layers=94`
  - `compact_enabled_layers=0`
  - `offloaded_layers=0`
- 原因：
  - 每 rank resident local=12、offloaded local=4、cache=4。
  - compact slots = 16，等于原始 local slots，没有 HBM shrink。

结论：这个配置看起来快，但没有真实 expert offload，不用于后续卸载吞吐对比。

### 2. 真实 compact：`RESIDENT_EXPERTS=96 / NPU_CACHE_CAPACITY=2`

命令要点：

```bash
RESIDENT_EXPERTS=96
NPU_CACHE_CAPACITY=2
PREFETCH_DISTANCE=1
ENABLE_LARGE_BATCH_FAST_PATH=0
ENABLE_SINGLE_SELECT_FORWARD=1
ENABLE_CHUNKED_COMPACT_FORWARD=1
```

结果：

- KV cache：约 `39,424` tokens
- `output_tokens_per_second=41.438`
- 真实卸载：
  - `compact_enabled_layers=94`
  - `offloaded_layers=94`
  - `total_cpu_store_bytes=14193524736` per worker
- copy/chunked 代价很高：
  - copy count 约 `2491-2911` per worker
  - chunked piece count 约 `1248-1776` per worker

结论：真实卸载成立，但吞吐太低。

### 3. 真实 compact：`RESIDENT_EXPERTS=96 / NPU_CACHE_CAPACITY=3`

推荐对比命令：

```bash
MODEL_PATH=/workspace/models/Qwen3-235B-A22B \
OFFLOAD_MODE=expert_wise \
BATCH_SIZE=128 MAX_LENGTH=64 MAX_NEW_TOKENS=8 \
WORLD_SIZE=8 UTILIZATION=0.98 \
PREFETCH_DISTANCE=0 OFFLOAD_INTERVAL=1 \
PARTITION_SCOPE=local_rank \
RESIDENT_EXPERTS=96 NPU_CACHE_CAPACITY=3 \
COMPACT_NPU_CACHE=1 SKIP_NO_SHRINK_COMPACT=1 \
ENABLE_LARGE_BATCH_FAST_PATH=0 ENABLE_SINGLE_SELECT_FORWARD=1 \
ENABLE_CHUNKED_COMPACT_FORWARD=1 ENABLE_COMPACT_CHUNK_TOKEN_FILTER=0 \
LOG_TRANSFERS=0 \
python offloading_run.py
```

结果：

- 当前真实 compact 最佳：约 `53 tok/s`
- 具体已测：
  - `PREFETCH_DISTANCE=1`，合并 resident+first chunk 后：`53.513 tok/s`
  - `PREFETCH_DISTANCE=0`：`53.222 tok/s`
- KV cache：约 `39,168` tokens
- 真实卸载：
  - `compact_enabled_layers=94`
  - `offloaded_layers=94`
  - `total_cpu_store_bytes=14193524736` per worker
- `PREFETCH_DISTANCE=0` 后：
  - `prefetch_count=0`
  - `prefetch_wait_count=0`
  - copy count 仍约 `737-930` per worker

结论：prefetch/wait 不是主瓶颈。核心瓶颈是 on-demand CPU->NPU copy 和每层多次 chunked fused dispatch。

### 4. 真实 compact：`RESIDENT_EXPERTS=112 / NPU_CACHE_CAPACITY=1`

命令要点：

```bash
RESIDENT_EXPERTS=112
NPU_CACHE_CAPACITY=1
PREFETCH_DISTANCE=0
ENABLE_LARGE_BATCH_FAST_PATH=0
ENABLE_CHUNKED_COMPACT_FORWARD=1
```

结果：

- `output_tokens_per_second=50.182`
- KV cache：约 `36,096` tokens
- CPU store 下降到约 `7096762368` bytes per worker
- 但 chunked forward 次数更高：
  - chunked forward count 约 `366-485` per worker
  - chunked piece count 约 `732-970` per worker

结论：提高 resident、减少 offloaded experts 并没有提高吞吐。`cache=1` 导致 chunked dispatch 更频繁，整体更慢。

### 5. token 子集筛选实验

命令要点：

```bash
RESIDENT_EXPERTS=96
NPU_CACHE_CAPACITY=3
PREFETCH_DISTANCE=0
ENABLE_COMPACT_CHUNK_TOKEN_FILTER=1
```

结果：

- `output_tokens_per_second=45.667`
- token 筛选确实生效：
  - 每 worker `chunked_compact_token_count / full_token_count` 约 `52%-54%`
- 但吞吐下降。

结论：小 batch/子 batch fused MoE 调用、切片和 scatter 的额外开销大于省下的计算。该开关保留但默认关闭。

### 6. `RESIDENT_EXPERTS=104 / NPU_CACHE_CAPACITY=2`

状态：

- 启动后卡在 EngineCore/HCCL 初始化附近，没有进入 Worker_TP 加载阶段。
- 已终止，无遗留进程。

结论：本轮没有得到有效吞吐数据。若后续重测，先确认 NPU/HCCL 状态干净。

## 当前判断

当前真实卸载吞吐和 naive 的差距很大：

- naive：`338.213 tok/s`
- 当前最佳真实 compact offload：约 `53 tok/s`

主要瓶颈不是 FlashInfer，也不是 prefetch wait，而是：

1. compact cache 比本地 offloaded expert 少 1 个 slot 时，路由常命中全部 offloaded experts。
2. 每层 decode 经常需要至少一次 CPU->NPU 换入。
3. chunked compact 会对同一层 MoE 做多次 fused dispatch。
4. token 子集化会降低计算量，但小 batch dispatch/切片/scatter 成本更高。

所以后续要接近原生，重点不应是继续微调 prefetch，也不应只靠提高 resident；应优先减少：

- CPU->NPU copy 次数
- chunked fused dispatch 次数
- 每次 dispatch 的 Python/通信开销

## 下一步计划

### P0：保持可复现实验基线

先确认没有后台进程：

```bash
ps -ef | rg 'offloading_run.py|naive_run.py|EngineCore|Worker_TP|VLLM::Worker' | rg -v rg
npu-smi info
```

当前建议基线命令：

```bash
MODEL_PATH=/workspace/models/Qwen3-235B-A22B \
OFFLOAD_MODE=expert_wise \
BATCH_SIZE=128 MAX_LENGTH=64 MAX_NEW_TOKENS=8 \
WORLD_SIZE=8 UTILIZATION=0.98 \
PREFETCH_DISTANCE=0 OFFLOAD_INTERVAL=1 \
PARTITION_SCOPE=local_rank \
RESIDENT_EXPERTS=96 NPU_CACHE_CAPACITY=3 \
COMPACT_NPU_CACHE=1 SKIP_NO_SHRINK_COMPACT=1 \
ENABLE_LARGE_BATCH_FAST_PATH=0 ENABLE_SINGLE_SELECT_FORWARD=1 \
ENABLE_CHUNKED_COMPACT_FORWARD=1 ENABLE_COMPACT_CHUNK_TOKEN_FILTER=0 \
LOG_TRANSFERS=0 \
python offloading_run.py
```

判断标准：

- 应该真实 offload：
  - `compact_enabled_layers=94`
  - `offloaded_layers=94`
- 吞吐应在 `53 tok/s` 左右。
- 如果显著低于该值，先排查环境、NPU 状态、是否误开 `ENABLE_COMPACT_CHUNK_TOKEN_FILTER=1`。

### P1：减少 chunked dispatch 次数

优先考虑结构：

1. 对 `offloaded_count = cache_capacity + 1` 的情况做专门路径
   - 当前 `96/3` 正是每 rank 4 个 offloaded、本地 cache 3 个。
   - 常见场景只差 1 个 slot。
   - 可以尝试在一个 forward 内复用 resident+cache 结果，只对缺失的 1 个 expert 做补算。

2. 避免第二个 chunk 再跑完整 fused MoE
   - token 子集筛选已验证直接切 token 会变慢。
   - 但仍可探索 expert-level 补算，避免全 top-k dispatch。
   - 方向：只对缺失 expert 的 token 做更轻量的 matmul/MLP，而不是再次调用完整 fused experts。

3. 将 chunked 路径做成“resident/cache 主路径 + missing expert 补偿”
   - 主路径尽量接近原始 fused MoE。
   - 缺失 expert 的输出单独计算后加回。
   - 这比当前多次 masked fused MoE 更接近原生吞吐。

### P2：减少 CPU->NPU copy 次数

1. cache 策略从 LRU 改为 decode-step 友好
   - 当前 cache_slots 是 LRU。
   - 对 MoE decode 来说，最近一步的 LRU 未必是下一步最优。
   - 可利用上一层/上一 token 的 routed experts 做预测，减少抖动。

2. 优先保留高频 offloaded experts
   - 统计每层 offloaded expert 命中频率。
   - 对高频 expert 做 sticky cache，低频 expert 才被 evict。
   - summary 需要增加 per expert hit/miss，先只在 debug 开关下记录，避免热路径开销。

3. 批量 copy 或 packed CPU store
   - 当前按 expert、按 tensor copy。
   - 可以尝试 packed CPU tensor，连续 expert 一次 copy。
   - 先只支持 unquantized BF16 路径，覆盖 235B 当前实验。

### P3：重新评估配置网格，但只测有意义点

不要再测 `NPU_CACHE_CAPACITY=4` 当作真实卸载配置，因为它 no-shrink。

建议候选：

1. `RESIDENT_EXPERTS=96 / NPU_CACHE_CAPACITY=3`
   - 当前真实 offload 最佳基线。

2. `RESIDENT_EXPERTS=88 / NPU_CACHE_CAPACITY=4 或 5`
   - 需要先算是否真实 shrink。
   - 目标是判断更多 offloaded experts + 较大 cache 是否能减少 chunked piece。

3. `OFFLOAD_INTERVAL=2`
   - 如果全层 offload copy 过多，可只 offload 一半层。
   - 目标是提升吞吐，同时保留一部分 KV cache 收益。

4. `MAX_NEW_TOKENS=64`
   - 只有短输出稳定后再跑。
   - 用于和 naive `338.213 tok/s` 做正式对比。

### P4：保留但不要默认启用的实验

1. `ENABLE_COMPACT_CHUNK_TOKEN_FILTER=1`
   - 已证明在 `96/3` 下变慢。
   - 除非底层 fused MoE 对小 batch 有优化，否则不建议继续主推。

2. `PREFETCH_DISTANCE=1/2`
   - `PREFETCH_DISTANCE=0` 和 `1` 吞吐基本持平。
   - 因为主瓶颈是 copy 和 dispatch，不是等待 event。

3. layer-wise
   - 当前不是 235B 吞吐主线。

## 推荐交接提示词

从这里继续：

> 继续在 `/workspace/Huawei/vLLM_Ascend_Inference` 优化 Qwen3-235B-A22B 的 expert-wise offload 吞吐。用户当前关心的是卸载后的吞吐接近原生，不要把 FlashInfer 作为主线。当前真实 compact offload 最佳基线是 `RESIDENT_EXPERTS=96 NPU_CACHE_CAPACITY=3 PREFETCH_DISTANCE=0 ENABLE_LARGE_BATCH_FAST_PATH=0 ENABLE_SINGLE_SELECT_FORWARD=1 ENABLE_CHUNKED_COMPACT_FORWARD=1 ENABLE_COMPACT_CHUNK_TOKEN_FILTER=0`，短输出约 `53 tok/s`，真实卸载 94 层，CPU store 约 `14.19GB/worker`，KV cache 约 `39,168 tokens`。`96/4` 是 no-shrink，不算真实卸载；`112/1` 约 `50 tok/s`；token filter 约 `45.7 tok/s`，默认关闭。下一步优先减少 chunked fused dispatch 和 CPU->NPU copy 次数，探索 resident/cache 主路径 + missing expert 补偿，而不是继续微调 prefetch 或单纯增加 resident。
