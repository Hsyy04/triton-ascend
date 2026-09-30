# SinkI1ProducersIntoUsersPass

## Overview

### 1. 目标

把产生 `tensor<i1>` 的纯无副作用 op **下沉（复制）到每个 i1 消费块**，使各块各自持有副本，避免跨块 i1 传输。

### 2. 规格

1. 仅考虑"i1 producer"：单结果、所有结果均为 `TensorType` 且元素类型为 `i1` 的 op。
2. 仅考虑纯无副作用（pure、regionless）op：
   - 不带 `HasRecursiveMemoryEffects` trait；
   - 若实现 `MemoryEffectOpInterface`，`getEffects` 必须为空。
3. 排除 SCF dialect 内的 op（`scf.if`、`scf.for` 等控制流不能被克隆/移动）。
4. Producer 必须带有效 block_id（`getBlockIdByOp(op) != -1`）才纳入处理。
5. 至少存在一个能解析到 producer 所在 block 的 consumer（即 consumer 的某个 block 内祖先能取到 producer 同 block 的 op）。
6. Consumer block_id 为 -1（控制流打断）的消费者被忽略；首例触发 producer 进入 `seenBlockIds/blockId2Producer`。
7. 每个目标 block_id 最多一份 producer 副本：同 block_id 的后续 consumer 通过 `replaceUsesWithIf` 复用到已克隆/已存在的副本。
8. Consumer 使用 `topologicalSort(consumers)` 排序，确保处理顺序的稳定与无回环。

### 3. 算法流程

入口 `runOnOperation()` 在 ModuleOp 上构建 `ComputeBlockIdManager`，先收集 producer 列表，再**逆序**逐个下沉（先下沉下游依赖较浅的，最末处理源头）。

#### Step 1 — 收集 i1 producer `producers`

`moduleOp.walk`：对每个 op，满足 `isValidI1Producer(op) && isPureAndRegionless(op)` 且 `getBlockIdByOp(op) != -1` 才加入 `producers`。

- `isValidI1Producer`：`!isa<SCFDialect>`、单结果、所有结果为 `TensorType` 且元素类型 `isInteger(1)`。
- `isPureAndRegionless`：无递归内存效果 trait；若有 `MemoryEffectOpInterface` 则 effects 必须为空。

#### Step 2 — 收集当前 producer 的所有"块内可见 consumer"

对 producer 的每个 `OpOperand::use`：
- 取 consumer 的 block 内祖先 `consumerInblock = getAncestorInBlock(consumer, p->getBlock())`；
- 不为空才插入 `consumers`（`SetVector` 自动去重）。

空consumers 集合的 producer 直接跳过（不需要下沉）。

#### Step 3 — Topo 排序消费者

`orderedConsumuers = mlir::topologicalSort(consumers)`：保证处理顺序符合数据流方向。

#### Step 4 — 处理首个 consumer：迁移或登记

- `consumerBlockId = bm.getBlockIdByOp(orderedConsumuers[0])`：
  - 若为 -1：被控制流打断，将 producer 原 block 加入 `seenBlockIds`，producer 自身登记到 `blockId2Producer`。
  - 若 producer 与首个 consumer 已同块（`isSameBlock(p, orderedConsumuers[0])`）：登记到 `seenBlockIds/blockId2Producer`，**不移动**。
  - 否则：`p->moveBefore(orderedConsumuers[0])` + `bm.updateBlockId(p, consumerBlockId)`，登记到 `blockId2Producer`。

#### Step 5 — 处理其余 consumer：克隆并替换

遍历 `orderedConsumuers`：
- `consumerBlockId = bm.getBlockIdByOp(consumerInblock)`；为 -1 直接跳过（控制流打断的消费者不进入下沉）。
- 若 `consumerBlockId` 已登记（`!seenBlockIds.insert(...).second`）：对 producer 每个 result，用 `replaceUsesWithIf` 把属于 `consumerInblock` 的 use 切到已克隆/已存在的副本 `producer->getResult(id)`。
- 否则（首次见到）：`OpBuilder(consumerInblock).clone(*p)` 在该 consumer 前克隆一份；`bm.updateBlockId(cloned, consumerBlockId)`；登记 `blockId2Producer[consumerBlockId] = cloned`；再 `replaceUsesWithIf` 把属于该 consumer 的 use 切到克隆体。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `producers` | `SmallVector<Operation *>` | 满足 i1 producer + pure + regionless + 有 block_id 的 op 列表 |
| `consumers` | `SetVector<Operation *>` | producer 在其所在 block 内的可见 consumer（去重） |
| `orderedConsumuers` | topo-sort 结果 | 处理顺序，确保下沉时无数据流回退 |
| `seenBlockIds` | `DenseSet<int>` | 已处理过的 consumer block_id 集合 |
| `blockId2Producer` | `DenseMap<int, Operation *>` | 每个目标块对应的 producer 副本（首个 consumer 走 move / 其余走 clone） |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `SinkI1ProducersIntoUsersPass::runOnOperation` | `SinkI1ProducersIntoUsersPass.cpp:107` | 入口：walk 收集 producer + 逆序逐个下沉 |
| `isValidI1Producer` | `:72` | i1 producer 识别（`TensorType` + `i1` 元素） |
| `isPureAndRegionless` | `:90` | 无副作用 + 无 region 判断（`MemoryEffectOpInterface` 检查） |
| `CVPipeline::getAncestorInBlock` | `Common/Utils.h`（外部） | 在指定 Block 内找到 consumer 的最近祖先 op |
| `mlir::topologicalSort` | `mlir/Analysis/TopologicalSortUtils.h` | 对 consumer 集合做拓扑排序 |
| `createSinkI1ProducersIntoUsersPass` | `:195` | 注册入口，pass 名 `"sink-i1-producers-into-users"` |

### 6. 安全机制

- **Pure & regionless 强制**：`isPureAndRegionless` 阻止任何带内存效果或含 region 的 op 被克隆/下沉，副作用 op 下沉会破坏语义。
- **SCF 排除**：producer 不能是 SCF dialect op，避免把 `scf.if/for` 等控制流复制多份。
- **block 内可见性**：`getAncestorInBlock` 限定 producer 与 consumer 必须可解析到同一 MLIR Block，否则不视为 i1 下沉候选。
- **控制流打断豁免**：consumer block_id = -1 时直接跳过（避免在跨 `scf.for/yield` 边界下沉）。
- **去重副本**：每个目标 block_id 一份 producer，后续 consumer 通过 `replaceUsesWithIf` 复用，避免无限制克隆。
- **逆序处理**：producer 列表 `llvm::reverse` 处理，使下游依赖先于源头被下沉到位。
- **i1 类型精确门控**：只有 `TensorType` 且元素 `i1` 才下沉，避免无谓克隆其他 dtype 的 op。

## 目标场景

1. 包含 i1 mask 计算并被多个 VECTOR/CUBE 块分支消费的 kernel（如 attention mask、segment mask、动态 shape 条件），下沉后各块各自持有 mask 副本，省去跨块 i1 UB 传输。