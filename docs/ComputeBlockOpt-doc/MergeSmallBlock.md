# MergeSmallBlock

## Overview

### 1. 目标

将"小计算块"（VECTOR_ONLY 且 `isTensorComputeOp` 数量 ≤ `MIN_VF_SIZE`=3）合并到其上游（操作数定义所在块）或下游（结果使用所在块），消除冗余块边界、降低 UB 占用与跨块搬运。

### 2. 规格

1. 仅处理 **VECTOR_ONLY** 小块（core_type 非 VECTOR_ONLY 直接跳过）；计算 op 计数 `cntComputeOps > 3` 的大块不动。

2. 上下游必须均为 **VECTOR_ONLY**：上游/下游一旦出现 CUBE 链接（block_id=-1 或 core_type≠VECTOR_ONLY），该侧候选全部清空（不与 Cube 块合并）。

3. 上游候选要求 唯一 producer 块
4. 下游允许多个 consumer 块各自独立判环入候选。

5. 如果上游块是一个纯标量块, 则标量块不应该作为候选块.
6. 合并后不存在环.

7. 筛选候选者遵循如下原则:
   - 候选集合为1时直接选择.
   - 策略一: FaFWDPatternStrategy, 识别 Flash Attention Forward 的 softmax-like 3-op 小块（exp→mulf→addf）合并到下游 reduce/broadcast 块。
   - 策略二: UBOccupationStrategy, 选 UB 节省最大的候选
   - 策略三: IROrderStrategy, 选 IR 序最早的下游块, 出于不成环考虑
   - 策略四: DefaultUpStrategy, 兜底：优先上游块, 如果没有上游块, 则选择下游块.


8. 限制同步点合入的原则: `sitofp→subf→mulf` 模式中, 添加 subBlock_id 区分.

### 3. 算法流程

入口 `runOnOperation()`（`:768`）。

#### Step 1 — 准备

1. `hasFallbackAttr(module)` 为真直接返回。
2. `firstRun = !kMergeSmallBlockFirstRunDone`；随即把该 attr 置 true（本 pass 调度两次，第二次走 sitofp 模式合入分支）。
3. 构建 `MemoryDependenceGraph memGraph` 与 `ComputeBlockIdManager bm`。
4. `module.walk` 每个 `Block *block`：调 `getBlockIdsInProgramOrder` 得到按 IR 顺序的 `orderedBlockIds` 与 `id2order`（block_id→序号）；块内 block_id 种类 `< 2` 则跳过该 block。

#### Step 2 — 小块筛选

对 `orderedBlockIds` 中每个 `nowBlockId`：
- `ops = bm.getOpsByBlockId(nowBlockId)`；空或 `cntComputeOps(ops) > 3` 跳过。
- 首个 op 的 core_type ≠ VECTOR_ONLY 跳过。

#### Step 3 — 收集候选 `collectMergeCandidates`（`:666`）

`ops = bm.getOpsByBlockId(smallBlockId)`，分别调 upstream/downstream：

**`collectUpstream`（`:559`）** — 找操作数定义所在块：
- 对每个 op 的每个 operand，`getAncestorInBlock(defOp, block)` 取块内祖先。
- 祖先 block_id==-1 或 core_type≠VECTOR_ONLY → `haveCubeLink=true`（Cube 链接，禁上游）。
- 否则 bid≠smallBlockId 收入 `upBlockIds`，operand 收入 `allDepValues`。
- `haveCubeLink` → 清空 upBlockIds；`allDepValues` 全 `isScalarLike` → 清空 upBlockIds（标量依赖无合入收益）。
- `upBlockIds.size()==1` 且 `!willCreateCycle` → push 候选。

**`collectDownstream`（`:613`）** — 找结果使用所在块：
- 对每个 op 的每个 result 的每个 user，`getAncestorInBlock`；跳过 terminator。
- user block_id==-1 或 core_type≠VECTOR_ONLY → `haveCubeLink=true`（FIXME: 待支持 multi-region）。
- 否则 bid≠smallBlockId 收入 `downBlockIds`。
- `haveCubeLink` → 清空 downBlockIds；每个 downBlockId 各自 `!willCreateCycle` → push 候选。

#### Step 4 — 选目标 `selectMergeTarget`（`:680`）

- 候选恰 1 个 → 直接返回。
- 否则按序跑策略链，每跑完一个若 `candidates.size()==1` 即返回：

**① `FaFWDPatternStrategy`（`:525`）** — Flash Attention Forward 的 softmax-like 3-op小块：
- 条件：ops 恰 3 个、恰 1 up + 1 down。
- `findTargetOps` 找到 `math.exp` / `arith.mulf` / `arith.addf` 三者。
- `isValidExpOp`：exp 输入是 `arith.subf` 且其定义在 upBlock；exp 结果恰 2 user——mulf（块内）+ broadcast（`linalg`/`triton`，在 downBlock）。
- `isValidMulfOp`：mulf 一侧取 exp 结果、另一侧取 `BlockArgument`；mulf 结果仅被 addf 使用。
- `isValidAddfOp`：addf 一侧取 mulf 结果、另一侧取 `linalg.reduce`/`triton.reduce`（在 upBlock）；addf 结果单 user = `scf.yield`，且 yield 对应的 `regionIterArg` == mulf 的 BlockArgument（闭环更新累加器）。
- 全匹配 → 候选置为 `downBlockIds[0]`。

**② `UBOccupationStrategy`（`:281`）** — 选 UB 节省最大的候选：
- up 块用 `getOperandUB`：sum operand size，其中 defOp 在该 upBlock 且非 `tensor.empty`。
- down 块用 `getUserUB`：sum result size，仅当该 result **所有** user 都在该 downBlock（多 user 跨块不算节省）。
- `getValueSizeInBytes`（`:202`）：scalar=0；静态 tensor=元素数×elemBytes（≥1）；动态 tensor / 非 tensor非scalar = `INF`(1<<30)。
- 取 max UB（排除 INF），保留等于 max 的候选。

**③ `IROrderStrategy`（`:173`）** — down 块里选 IR 序最早的；候选 = 全部 up + 该 down。无 down 则不变（保留所有 up）。

**④ `DefaultUpStrategy`（`:324`）** — 有 up 取 `upBlockIds[0]`；否则有 down 取 `downBlockIds[0]`。

- 策略链跑完仍 >1 候选 → 返回 `nullopt`（不合入）。

#### Step 5 — 应用合入

`targetBlockId.has_value()` 时：
- `matchSIToFPSubMulPattern(ops)`（`:727`）：sitofp→subf→mulf 链。若命中且 `firstRun` → **defer**（continue，等第二次）。若第二次命中 → `markSubBlockOps` 对源块和目标块所有 op 打 `kSubBlock` 原始 block_id（保留下游 pass 的来源信息）。
- 合入：对 ops 中每个 op `bm.updateBlockId(op, targetBlockId.value())`。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `MIN_VF_SIZE` | `const int = 3` | 小块计算 op 数上限 |
| `orderedBlockIds` | `SmallVector<int>` | 块内按 IR 顺序的 block_id 序列 |
| `id2order` | `DenseMap<int,int>` | block_id → 程序序号（供上下行分类） |
| `MergeStrategy::Context` | `struct` | 策略上下文：block/bm/memGraph/id2order/smallBlockOps/smallBlockId |
| `UpDownClassification` | `struct` | 候选按序号分 up（<smallBlock）/down 两类 |
| `strategies` | `SmallVector<unique_ptr<MergeStrategy>>` | 策略链：FaFWD→UB→IROrder→DefaultUp |
| `haveCubeLink` | `bool` | 该侧存在 Cube 链接，禁合入标志 |
| `kMergeSmallBlockFirstRunDone` | `BoolAttr` | 标记首次运行完成，控制 sitofp 模式推迟 |
| `kSubBlock` | attr | 合入前记录 op 原始 block_id |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `isTensorComputeOp` / `...Legacy` | `:93` / `:79` | 判定是否计入"计算 op"（legacy 走全 tensor 判定） |
| `cntComputeOps` | `:124` | 统计块内计算 op 数（与 MIN_VF_SIZE 比） |
| `MergeStrategy::classifierUpDown` | `:153` | 候选按 id2order 分 up/down |
| `IROrderStrategy::filter` | `:173` | down 取最早 IR 序，保留全部 up |
| `UBOccupationStrategy::getValueSizeInBytes` | `:202` | Value 的 UB 字节数（scalar=0，动态=INF） |
| `UBOccupationStrategy::getOperandUB` | `:231` | 合入 up 块可省的 operand UB |
| `UBOccupationStrategy::getUserUB` | `:257` | 合入 down 块可省的 result UB（须全 user 同块） |
| `UBOccupationStrategy::filter` | `:281` | 保留 UB 节省最大的候选 |
| `DefaultUpStrategy::filter` | `:324` | 兜底：优先 up[0]，否则 down[0] |
| `FaFWDPatternStrategy::findTargetOps` | `:373` | 找 exp/mulf/addf 三元组 |
| `FaFWDPatternStrategy::isValidExpOp` | `:392` | 校验 exp（subf 来源、双 user：mulf+broadcast） |
| `FaFWDPatternStrategy::isValidMulfOp` | `:433` | 校验 mulf（exp + BlockArgument，单 user addf） |
| `FaFWDPatternStrategy::isValidAddfOp` | `:458` | 校验 addf（reduce 来源、yield 闭环 iterArg） |
| `FaFWDPatternStrategy::filter` | `:525` | 命中 FaFWD 则候选置为 downBlockIds[0] |
| `collectUpstream` | `:559` | 收集上游候选（Cube 链接/全标量清空，单块+无环入候选） |
| `collectDownstream` | `:613` | 收集下游候选（Cube 链接清空，逐块判环入候选） |
| `collectMergeCandidates` | `:666` | upstream + downstream 合并候选 |
| `selectMergeTarget` | `:680` | 候选==1 直接返回；否则跑策略链收敛到 1 |
| `getBlockIdsInProgramOrder` | `:707` | 块内按 IR 顺序收集 block_id 与序号映射 |
| `matchSIToFPSubMulPattern` | `:727` | 识别 sitofp→subf→mulf 模式（第二次才合入） |
| `markSubBlockOps` | `:757` | 合入前给源/目标块 op 打 kSubBlock 原始 id |
| `runOnOperation` | `:768` | 入口：建图→遍历块→筛选小块→候选→选目标→合入 |
| `createMergeSmallBlockPass` / `registerMergeSmallBlockPass` | `:850` / `:854` | 构造/注册 pass |

### 6. 安全机制

- **fallback 短路**：`hasFallbackAttr` 为真直接返回。
- **Cube 隔离**：上下游任一侧出现 Cube 链接（block_id=-1 或非 VECTOR_ONLY），该侧候选全清——避免把 Vector 小块并入 Cube 块。
- **唯一 producer**：上游仅当 `upBlockIds.size()==1` 才入候选，避免多源分叉。
- **全标量豁免**：纯标量依赖无 UB 收益，上游清空。
- **环检测**：每个候选合入前 `willCreateCycle` 判环，成环不入候选。
- **策略链收敛**：多候选必须经策略链收敛到唯一才合入，否则放弃，保证合入确定性。
- **sitofp 模式推迟**：`sitofp→subf→mulf` 首次运行 defer 到第二次，避免过早合入影响后续 pass；合入前 `markSubBlockOps` 保留原始 block_id 供下游使用。

## 存在问题

1. FaFWDPatternStrategy, 预计删除
2. 上游块不应该唯一
3. 合并顺序, 不能按照代码序, 不能依赖reorder, 需要自己找拓扑序.

## 目标场景

1. FaFWDPatternStrategy: swa_fwd/sdpa_fwd
2. matchSIToFPSubMulPattern:
2. DefaultUpStrategy: 
