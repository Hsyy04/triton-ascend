# SplitIfByBlockIdPass

## Overview

### 1. 目标

当一个 `scf.if` 内部同时包含 CUBE 和 VECTOR 两组 op，且二者之间存在 CVC 时，拆分成一串 `scf.if` 链，使每个新 if 只含单一 block_id。

### 2. 规格

1. Module 必须不带 `fallback` 标注；否则整个 pass 直接跳过。
2. 仅对 main loop（外层 `scf.for`/`scf.while`）内的 `scf.if` 触发；非主循环位置跳过。
3. 三个内核被黑名单跳过：`parallel_deltaformer_fwd_kernel`、`chunkwise_fwd_kernel`、`parallel_nsa_fwd_kernel`。
4. 仅当 then-region 或 else-region 内**至少 2 个 block group**（按 block_id 分组后）时才拆分；单 group 直接成功返回。
5. 仅处理 `core_type == VECTOR_ONLY` 的跨组 op；CUBE 块通过其他 pass 处理。
6. Fallback：`materializeCandidate` 任意一步失败 → 全模块回滚到 pass 开始时的 IR（`FallbackHelper::restore`）。
7. 拆分后用 `type=` 为新 if 设 `core_type`（按 yield operand core_type 串拼接）；用 `kSplittedIf` 标注待后处理。
8. 后处理 `rearrangeIfOp` 按依赖把所有新 if 重排到最迟的依赖 op 之后，并移除 `kSplittedIf` 标注。

### 3. 算法流程

入口 `runOnOperation()`：

#### Step 0 — 准备与短路

- 检查 `hasFallbackAttr(module)`，是则直接 return。
- 建 `FallbackHelper{module}` 备份。
- 建 `CVPipeline::ComputeBlockIdManager bm(module)`。
- 调 `walkMainLoop` 找到 main loop；遇到黑名单 func 则 `success()` 跳过。
- 对 main loop 内每个 `scf.if` 走下列三阶段；若 `materializeCandidate` 失败，`WalkResult::interrupt`，最终 `restore()` 回滚。

#### Step 1 — Part1 分组 `getCandidate` + `preprocessScalarDependencies`

`getCandidate(ifOp)`：
- 收集 `CandidateIf`：自身 `selfBlockId`、`hasYield`、`thenGroups`、`elseGroups`、`yieldAug`。
- `groupOpsInBlock(block)`：在每个 region 内按 block_id 分组：
  - 有 block_id 且 ≠ -1 → 加入对应 group；
  - 嵌套 `scf::IfOp`：按自身 block_id 形成 group；若当前 `currentId != -1` 则归入当前 group；否则入 `pendingAmbient`；
  - 无 block_id（ambient）：归入 `currentId` 组并补打 `block_id` 属性；`currentId` 仍为 -1 则暂存；
  - 末尾 flush pending 到首个 real group。

`preprocessScalarDependencies`：对每个 group 调 `ScalarClosure::collect`，把外部标量依赖显式 `clone` 进该 group 头部并重写 operand，确保后续 SSA 重写可解析。

#### Step 2 — Part2 依赖分析 `analyzeDependencies`

1. 建 `opToThenGroup` / `opToElseGroup`：把每个 group 的 ops + nestedIfs 映射到 group index。
2. `scanRegion`：对每个 region 的每个 group 的每个 op（含 region 内 nested op）：
   - 每个 operand 调 `addCrossGroupDependency`：
     - `defOp == nullptr` 跳过；
     - `opToGroup` 中无 defOp 跳过；
     - `defOp` 与 `consumer` 同 group 跳过；
     - 否则登记 `crossValueMap[val] = {fromG, {consumer}}`。
   - 嵌套 `if` 的 result 也按 user 路径回溯到 `opToGroup` 跟踪点，跨组则登记。
3. `planYield`：基于 `crossValueMap` 推断"哪个 group 应携带哪些 yield slot"：
   - `splitThen` 判定：then 侧 group 数 ≥ 2 则 true；否则用 else。
   - 按 producer group 排序 `crossValues`（确定性）。
   - Case A（无原始 yield）：`planYieldCaseA`——每个非末尾 group yield 自己产出的跨组 value；末尾 group void。
   - Case B（有原始 yield）：`planYieldCaseB`——保留原 yield 槽；非末尾 group yield 自己产出的 cross-group value + 自己产出的原 yield 槽；末尾 group（last-if）携带全部原 yield 类型。
4. `dumpYieldAugmentation` 打 debug。

#### Step 3 — Part3 物化 `materializeCandidate`

按 `splitThen` 决定切哪一侧，并据此决定 condition 是否取反（else 拆分时 XOR 1 取反）。

- `collectOtherSideInfo`：另一侧有 op 则一并迁入 last-if 的 else 块（Scene 3/4）；同时收集另一侧 yield 值（用于 Case B）。
- `crossValueReplacement` 字典在创建每个新 if 后填入：把 group 产出的旧 value 映射到 `splitIf.getResult(idx)`，供下游 group 重写 operand。
- 非末尾 group（循环 `gi in [0, nGroups-1)`）：
  - Void group：建 `hasElse=false` 的 `scf.if`；迁入 group 内 ops + 嵌套 if；用 void `scf.yield` 收尾。
  - Result-bearing group：建带 per-group 输出类型的 `scf.if`；迁入 then 块 ops；`buildThenYieldForGroup` yield `output.outputValues`；else 块用 `buildElseYieldForGroup` 填 placeholder（Case B 对应原 yield 槽则用 `ensureLocalValue` 把原 else 值搬入，否则用 `createPlaceholderValue`）。
  - `updateCrossValueReplacementGroup`：把新 if 的结果替换掉 group 产出的旧 SSA 值的"组外" use（用 `thenBlocks` 集合排除本 then 块），同时填 `crossValueReplacement`。
- 末尾 group（Case A → 直接 `ifOp` 已被替换为新链上的最后一个 void if，源代码已迁；Case B）：
  - `buildThenYieldForLastIf`：遍历原 yield 槽，prodGroup 为末尾或 <0（block arg / 外部值）则直接 yield；否则通过 `crossValueReplacement` 跳引用前置 group 的 if 结果。
  - `buildElseYieldForLastIf`：吸收 other-side ops 到 else 块；用原 else yield 值（`ensureLocalValue`）或 placeholder 收尾。
- `postProcess`：把原 if 的 attrs 拷到新 if；按 yield operand core_type 拼接新 `core_type`；把 if + 两个 yield 都打上 `blockId`；打 `kSplittedIf` 标注等待重排。

#### Step 4 — 后处理

构造 `MemoryDependenceGraph` 后对所有带 `kSplittedIf` 的 if 调 `rearrangeIfOp`：

- 沿 if 内部所有 op 的 operand（排除 `ifOp` 自身定义的）找到 block 内最迟的 SSA 前驱；
- 加上 `memGraph.getExecBefore(ifOp)` 中能映射到该 block 的最迟内存前驱；
- 把整个 if `moveAfter(lastDependency)`。
- 移除 `kSplittedIf`。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `CandidateIf` | struct | 候选 if 上下文：`ifOp` / `selfBlockId` / `hasYield` / `thenGroups` / `elseGroups` / `yieldAug` |
| `BlockGroup` | struct | `blockId` + `ops` + `nestedIfs`，对应一个 block_id 在 region 内的全部内容 |
| `YieldAugmentation` | struct | 跨组 SSA + yield 计划：`crossValues` / `groupOutputs` / `numOriginalSlots` / `origYieldValues` / `origYieldProducerGroup` |
| `CrossGroupValue` | struct | 单个跨组 value：`value` / `fromGroupIdx` / `toGroupIndices` |
| `GroupOutputInfo` | struct | 单个 group 输出描述：`outputValues` / `origElseValues` / `outputTypes` / `isVoid()` |
| `OtherSideContext` | struct | 另一侧（不被拆分）的 ops + yieldValues（用于 Scene 3/4 吸收 / Case B else 收尾） |
| `crossValueReplacement` | `SmallDenseMap<Value,Value>` | 旧 SSA → 新 if 结果，跳引用重写 |
| `FallbackHelper` | 类 | 整模块备份与回滚（materializeCandidate 失败时调用 `restore`） |
| `opToGroup` | `SmallDenseMap<Op*,unsigned>` | op → group 索引（then / else 各一份） |
| `crossValueMap` | `SmallDenseMap<Value, pair<fromG, consumers>>` | value 级跨组依赖 |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `SplitIfByBlockIdPass::runOnOperation` | `SplitIfByBlockId.cpp:1664` | 入口：fallback 短路、walkMainLoop、对每个 if 三阶段，失败回滚 |
| `walkMainLoop` | `WalkMainLoop.cpp` | 在 module 中找到主循环（`scf.for`/`scf.while`），按内核名黑名单过滤 |
| `getCandidate` | `SplitIfByBlockId.cpp:290` | 构造 `CandidateIf`：`groupOpsInBlock` × then/else |
| `groupOpsInBlock` | `:206` | 按 block_id + 嵌套 if + ambient 把 region 内 op 分组 |
| `preprocessScalarDependencies` | `:312` | 用 `ScalarClosure::collect` 把 group 外部标量依赖克隆入 group 头部 |
| `ScalarClosure::collect` / `::capture` | `ScalarClosure.cpp` | 收集外部标量依赖并生成新 op 链 |
| `analyzeDependencies` | `:689` | Part2：建 `opToGroup` → `scanRegion` → `planYield` → `dump` |
| `addCrossGroupDependency` | `:88` | 把跨组 SSA 依赖登记到 `crossValueMap` |
| `scanRegion` | `:432` | 单 region 全 op + nested op + nested-if-result 的跨组依赖单次扫描 |
| `planYield` / `planYieldCaseA` / `planYieldCaseB` | `:648` / `:610` / `:501` | 按 Case A/B 推导每个非末尾 group 的 output & last-if 形状 |
| `materializeCandidate` | `:1400` | Part3：构造 if 链、迁 op、build yield、crossValueReplacement 重写、postProcess |
| `rearrangeIfOp` | `:1604` | 后处理：按 SSA + 内存依赖把新 if 移到最迟依赖 op 之后 |
| `getRootMemRef` / `getRootTensor` | `:750` / `:776` | 透过 reinterpret_cast / subview / split-if / scf.for 反查 placeholder 来源 |
| `createPlaceholderValue` / `ensureLocalValue` | （impl） | 创建 / 搬运占位 value，捕获 block_id 并克隆 ViewLikeOp |
| `createSplitIfByBlockIdPass` | `:1723` | 注册入口，pass 名 `"split-if-by-block-id"` |

### 6. 安全机制

- **Fallback 快照**：`FallbackHelper{module}` 在 runOnOperation 起点备份 IR，`materializeCandidate` 失败 → `restore()`。
- **黑名单内核**：三个已知会冲突的内核直接跳过。
- **hasFallbackAttr 短路**：模块带 fallback 标注时整个 pass 直接 return。
- **block_id 精确门控**：仅处理带 block_id 且 ≠ -1 的 op；ambient op 自动并入 active group 但会补打属性。
- **嵌套 if 单独成组**：嵌套 `scf.if` 视为一个整体 group，便于多轮拆分。
- **确定性排序**：`crossValues` 按 producer group + value ptr 排序，避免 yield 顺序非确定。
- **跨组 value 跳引用**：用 `splitIf.getResult` 替换下游 use 而非 passthrough 链，缩短 SSA 路径。
- **placeholder 复用**：同 type placeholder 在 `placeholderCache` 中只创建一次，避免大量重复 `arith.constant 0`。
- **依赖重排**：`rearrangeIfOp` 把新 if 推到最迟依赖 op 之后，确保 IR 顺序合法。
- **Cycle 间接防护**：Part3 通过 `crossValueReplacement` 与 yield 计划显式控制 value 流，避免在 if 链上形成不可达。

## 目标场景

1. 任意 main loop 内同时含 CUBE + VECTOR op 的 `scf.if`（如带条件 mask 的融合算子），按 block_id 拆分后便于后续 block 级调度。