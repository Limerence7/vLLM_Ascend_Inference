# vLLM Ascend MoE Offload 阶段总结与后续计划

## 项目目标

本项目面向 vLLM-Ascend 的 Qwen3 MoE 推理，目标是在插件层实现 MoE expert offload 能力。当前代码围绕两种卸载粒度展开：

- `layer_wise`：以 MoE 层为单位，将被选中的层专家权重放到 CPU，推理时拷贝到共享 NPU expert buffer。
- `expert_wise`：以 expert 为单位，将被选中的专家权重放到 CPU，推理时根据 router 结果按需恢复到 NPU。

当前主模型入口是 `src/model.py`，配置入口是 `src/offload/config.py`，插件注册入口是 `src/__init__.py`。

## 当前目录结构

```text
src/
  __init__.py
  model.py
  fused_moe/
    __init__.py
    compact_dispatch.py
    expert_wise.py
  offload/
    __init__.py
    config.py
    expert_wise_memory.py
    layer_wise.py
    memory.py

offloading_run.py
expert_wise_smoke_run.py
expert_wise_compact_smoke_run.py
naive_run.py
setup.py
```

## 已完成阶段

### 阶段 1：插件入口与配置体系

已完成内容：

- `setup.py` 提供 vLLM plugin entry point。
- `src/__init__.py` 提供 `register_plugin(config)` 和 `configure(config)`。
- `src/offload/config.py` 定义统一配置对象：
  - `OffloadConfig`
  - `ExpertOffloadConfig`
  - `ExpertWiseOffloadConfig`
- 支持的 `mode` 已统一为：
  - `none`
  - `layer_wise`
  - `expert_wise`
  - `auto`
- 配置由 Python 代码手动构造并传入，不依赖 argparse 或环境变量。

阶段结果：

- 推理脚本可以直接通过 `OFFLOAD_CONFIG` 控制卸载模式。
- 配置校验和归一化逻辑已经集中在 `src/offload/config.py`。

### 阶段 2：模型总入口整理

已完成内容：

- `src/model.py` 定义 `AscendQwen3MoeModel`，作为对外的总模型类。
- 该模型类负责：
  - 注册到 vLLM `ModelRegistry`。
  - 替换 Qwen3 内部 `FusedMoE` 后端为插件实现。
  - 在 `load_weights()` 后触发 offload 初始化。
  - 根据 `OffloadConfig.mode` 分发到不同 offload 路径。
- 具体 offload 细节不放在 `model.py` 中，而是下沉到 `src/offload/` 和 `src/fused_moe/`。

阶段结果：

- `src/model.py` 已成为模型级集成入口。
- layer-wise 和 expert-wise 的具体执行逻辑已经与模型入口分离。

### 阶段 3：Layer-wise Offload 框架

已完成内容：

- `src/offload/layer_wise.py` 提供 `LayerWiseOffloadController`。
- 支持按策略选择 MoE 层：
  - `every_n`
  - `explicit`
  - `ratio`
  - `tail`
- `src/offload/memory.py` 提供 layer-wise 内存组件：
  - `ExpertOffloadSlot`
  - `ExpertBuffer`
  - `ExpertBufferPool`
- 当前 layer-wise 路径会：
  - 将选中的层专家权重保存到 CPU。
  - 建立共享 NPU expert buffer pool。
  - patch 被选中 MoE 层的 MLP forward。
  - 在 forward 中按层加载专家权重。
  - 支持简单预取下一层专家权重。

阶段结果：

- layer-wise 的模块边界已经稳定。
- 当前实现是可执行框架，后续优化点主要在异步拷贝、内存统计、健壮性和性能验证。

### 阶段 4：Expert-wise FusedMoE 框架

已完成内容：

- `src/fused_moe/expert_wise.py` 提供 `ExpertWiseAscendFusedMoE`。
- `src/offload/expert_wise_memory.py` 提供 expert-wise CPU store 和 NPU cache 状态管理。
- 当前 expert-wise 路径会：
  - 在权重加载后把选中的 expert 权重复制到 CPU。
  - 根据 router logits 计算本次实际路由到的 expert。
  - 只恢复本次路由命中的 offloaded expert。
  - 记录 copy count、cache hit、cache miss、eviction 等基础状态。
- 支持两种 NPU 恢复方式：
  - dense slice restore：把 CPU expert 权重复制回原本 dense expert slice。
  - compact cache：用 resident expert slot 加有限 cache slot 替代完整本地 expert tensor。
- `src/fused_moe/compact_dispatch.py` 为 compact cache 提供 AllGather dispatch patch。

阶段结果：

- expert-wise 的执行链路已经打通到 FusedMoE 层。
- dense restore 路径更接近功能验证。
- compact cache 路径开始具备真实缩小 NPU expert weight 常驻布局的基础，但还需要更严格验证和边界处理。

### 阶段 5：运行脚本整理

已完成内容：

- `offloading_run.py`：面向完整 layer-wise/offload 配置的运行脚本。
- `expert_wise_smoke_run.py`：expert-wise dense restore 小规模验证脚本。
- `expert_wise_compact_smoke_run.py`：expert-wise compact cache 小规模验证脚本。
- `naive_run.py`：保留 native/baseline 路径对照。

阶段结果：

- 当前具备 baseline、layer-wise、expert-wise、compact expert-wise 的脚本入口。
- 后续性能和正确性验证可以围绕这些脚本继续扩展。

## 当前所处阶段

当前处于 **阶段 4 到阶段 5 之间：框架已成型，进入系统化验证和能力补齐阶段**。

当前状态可以概括为：

- 插件注册、配置、模型替换路径已经形成闭环。
- `src/model.py` 已经是总模型入口。
- `layer_wise` 和 `expert_wise` 已经拆成独立模块。
- expert-wise 已经接入 FusedMoE wrapper。
- compact cache 已有原型实现。
- `auto` 模式还没有真正策略生成逻辑，目前只回退到 layer-wise 配置。
- 统计、策略建模、通信计算重叠、多模型适配还没有完成。

当前最重要的下一步不是继续扩大功能面，而是先把现有框架验证扎实：

1. 确认 native、layer-wise、expert-wise dense、expert-wise compact 四条路径都能稳定运行。
2. 明确每条路径的 NPU 显存占用、CPU 存储量、拷贝次数和延迟表现。
3. 再基于数据推进 compact cache、overlap 和 auto 策略。

## 后续阶段计划

### 阶段 6：正确性与基础验证

目标：

- 建立稳定的 smoke test 和最小回归检查。
- 确认所有模式在 Qwen3-30B-A3B 上能跑通。

执行步骤：

1. 固定一组短 prompt，分别运行：
   - `naive_run.py`
   - `offloading_run.py`
   - `expert_wise_smoke_run.py`
   - `expert_wise_compact_smoke_run.py`
2. 记录每条路径是否能完成模型加载和一次 generate。
3. 对 expert-wise 路径打开 `log_transfers=True`，确认 router 命中的 expert 会触发 CPU 到 NPU 拷贝。
4. 对 compact cache 路径测试不同 `npu_cache_capacity`，确认容量不足时能明确报错，容量足够时能正常运行。

验收标准：

- 四条脚本路径都有明确结果。
- 失败场景能定位到配置、capacity、通信后端或模型结构问题。
- `python -m compileall src` 长期保持通过。

### 阶段 7：运行统计与可观测性

目标：

- 给 offload 运行过程补齐统一统计，方便后续做策略建模。

执行步骤：

1. 为 layer-wise 增加统计：
   - 被卸载层列表。
   - 每层 expert 权重大小。
   - CPU 存储量。
   - NPU buffer 数量和大小。
   - prefetch 命中情况。
2. 为 expert-wise 增加统计：
   - 每层 offloaded expert 列表。
   - resident/cache expert 数量。
   - CPU store 大小。
   - copy count、cache hit/miss、eviction。
   - 每层 routed expert 频率。
3. 提供统一 summary 输出接口，例如：
   - `model.offload_summary()`
   - 或独立 `src/offload/stats.py`
4. 在 smoke 脚本结束后打印 summary。

验收标准：

- 每次推理结束后能看到 layer-wise/expert-wise 的核心统计。
- 统计数据能解释当前策略为什么发生拷贝、命中或 eviction。
- 不影响原有推理路径。

### 阶段 8：Expert-wise Compact Cache 完善

目标：

- 将 compact cache 从原型推进到可靠的显存节省方案。

执行步骤：

1. 明确 global expert id、local expert id、compact slot 之间的映射关系。
2. 梳理 expert parallel 下 `expert_map` 的边界条件。
3. 完善 cache eviction：
   - eviction 后 `expert_map` 必须同步恢复为 `-1`。
   - 被 evict 的 expert placement 必须回到 CPU。
   - protected routed experts 不允许在同一次 forward 中被淘汰。
4. 验证 AllGather dispatch patch 的适用范围。
5. 对不支持的通信后端、shared experts、dynamic EPLB 给出清晰错误。

验收标准：

- compact cache 打开时，NPU resident expert tensor 的 dim0 明显小于原 dense 布局。
- cache capacity 改变会影响 copy、hit/miss 和 eviction 统计。
- 小 batch smoke run 稳定完成。

### 阶段 9：通信与计算重叠

目标：

- 降低 CPU 到 NPU expert 拷贝对 forward 延迟的影响。

执行步骤：

1. 将 expert-wise 的 CPU 到 NPU copy 抽象成 copy stream 管理模块。
2. 支持多个 copy stream，对应 `ExpertWiseOffloadConfig.num_copy_streams`。
3. 在已知 router 结果后尽早发起需要 expert 的拷贝。
4. 对 layer-wise prefetch 做更严格的事件同步和命中统计。
5. 用 profiler 或日志确认 copy stream 与 compute stream 是否重叠。

验收标准：

- `overlap=False` 和 `overlap=True` 有可对比结果。
- 打开 overlap 后不会改变输出正确性。
- 延迟相对同步拷贝版本下降，或至少能通过 profiler 证明 copy/compute 重叠发生。

### 阶段 10：自动策略建模

目标：

- 让 `mode="auto"` 根据模型结构、显存预算和运行统计生成 offload 策略。

执行步骤：

1. 建立 cost model 输入：
   - 每层 expert 权重大小。
   - NPU 显存预算。
   - CPU 到 NPU 带宽估计。
   - batch、seq length、top-k。
   - routed expert 频率。
2. 建立策略输出：
   - layer-wise selected layers。
   - expert-wise offloaded experts。
   - npu cache capacity。
3. 先实现确定性启发式：
   - 优先卸载低频 expert。
   - 优先保留高频 expert。
   - 显存不足时从 expert-wise 扩展到 layer-wise。
4. 将 `auto` 模式接入 `src/model.py` 的 setup 流程。

验收标准：

- `mode="auto"` 不再只是回退到 layer-wise。
- 同一输入下生成确定性策略。
- summary 中能解释 auto 策略的选择依据。

### 阶段 11：性能对比与多模型适配

目标：

- 形成 native、layer-wise、expert-wise、auto 的完整对比，并扩展到第二个模型。

执行步骤：

1. 固定评测输入：
   - prompt 数量。
   - max length。
   - max new tokens。
   - tensor parallel / expert parallel 配置。
2. 对比指标：
   - 首 token 延迟。
   - 总吞吐。
   - 峰值 NPU 显存。
   - CPU 存储量。
   - CPU 到 NPU 拷贝次数和总量。
3. 将 Qwen3-30B-A3B 作为第一模型基线。
4. 适配第二个 MoE 模型时，只在必要位置增加模型适配层，避免破坏通用 offload 模块。

验收标准：

- 至少两个模型能复用同一套 offload 配置和核心代码。
- 每种模式都有可解释的性能与显存数据。
- 能明确说明 layer-wise、expert-wise 和 auto 各自适合的场景。

## 当前风险与注意事项

- `expert_wise` dense restore 路径主要验证功能链路，不一定真正减少 NPU expert weight 常驻显存。
- compact cache 路径已经开始压缩 NPU expert tensor，但通信后端、expert_map 和 eviction 仍是主要风险点。
- `auto` 目前还没有真实策略生成。
- `prefetch`、`overlap`、`num_copy_streams` 等配置字段并不都已经有完整运行时实现。
- 当前运行脚本依赖实际 NPU、vLLM-Ascend 环境和本地模型路径，普通 CPU 环境只能做语法和导入检查。

## 建议的近期执行顺序

1. 先跑通 `expert_wise_smoke_run.py` 和 `expert_wise_compact_smoke_run.py`。
2. 补齐 summary 统计，确保每次运行后能看到 offload 行为。
3. 用统计结果验证 compact cache 是否真的降低 NPU expert tensor 常驻量。
4. 再做 overlap。
5. 最后接入 auto 策略。
