# PosMaskPattern

## Overview

### 1. 目标

识别“位置mask模式”， 即， 循环体中使用了跨越了sync屏障的memref， 且该memref的声明（及其前向包）与使用位于不同同步区的情况。
为了解决i1依赖处理，引入uboverflow问题。

### 2. 规格

1.仅处理 linalg::BroadcastOp，且其 core type 为 VECTOR_ONLY。

2.模式必须严格匹配：

    broadcast 结果恰好有 2 个用户，且均为 arith::CmpIOp；

    两个 cmpi 的谓词分别为 eq 和 sle，不可重复或出现其他谓词；

    每个 cmpi 结果仅有一个用户，且均为 arith::ExtUIOp。

3.模式内所有操作（broadcast、eqcmp、slecmp、eqext、sleext）必须位于同一 Block 且具有相同的 block_id。

4.目标块由两个 extui 的用户决定：所有用户必须与 broadcast 同 Block，且具有相同的有效 block_id（≠ -1）。

5.移动前进行`CVPipeline::willCreateCycle`环检测，若会引入循环依赖则放弃。

6.仅更新 block_id，不改变 IR 中操作的物理位置。


### 3. 算法流程

入口 `runOnOperation()` 遍历 module 中所有 `linalg::BroadcastOp`，对每个符合条件的 broadcast 调用 `collectPosPattern` 收集模式，再 `findTargetBlock` 确定目标 block，最后检查环并应用变更。

#### Step 1 — 筛选 Pos Broadcast

遍历所有 `linalg::BroadcastOp`，通过 `isPosBroadcast` 检查其 core type 是否为 `VECTOR_ONLY`，收集候选。

#### Step 2 — 收集模式 `collectPosPattern`

对每个候选 broadcast：
1. 检查其结果是否有恰好 2 个用户，且都是 `arith::CmpIOp`。
2. 识别两个 cmp：一个 `eq`，一个 `sle`，各一个，不能重复。
3. 检查两个 cmp 的结果都只有一个用户，且都是 `arith::ExtUIOp`。
4. 检查这五个操作（`broadcast`, `eqcmp`, `slecmp`, `eqext`, `sleext`）是否在同一个 block，且通过 `ComputeBlockIdManager` 获取的 block_id 相同。
5. 满足则返回 `PosPattern`。

#### Step 3 — 查找目标块 `findTargetBlock`

对于收集到的模式：
1. 收集 `eqextOp` 和 `sleextOp` 的用户，但只保留与 `broadcastOp` 在同一个 block 的用户。
2. 如果用户列表为空，返回 `nullopt`。
3. 检查所有用户是否具有相同的 block_id（通过 `bm.getBlockIdByOp`），且 block_id 有效（≠ -1）。
4. 返回目标 block_id。

#### Step 4 — 环检测与应用变更

在主循环中：
1. 比较 `broadcastBlockId` 与 `targetBlockId`，若相同则跳过。
2. 获取模式的所有操作 `allOps`。
3. 调用 `CVPipeline::willCreateCycle(allOps, memGraph, targetBlockId, bm)` 检查是否会形成循环。
4. 若不会成环，遍历 `allOps`，调用 `bm.updateBlockId(op, targetBlockId)` 更新每个操作的 block_id。


### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `PosPattern` | struct | 保存 broadcast、eqcmp、slecmp、eqext、sleext 五个 op |
| `ComputeBlockIdManager` | 类 | 管理 op 与 block_id 的映射，提供 `getBlockIdByOp`、`updateBlockId` 等 |
| `MemoryDependenceGraph` | 类 | 内存依赖图，用于环检测 |
| `broadcasts` | `SmallVector<linalg::BroadcastOp>` | 收集到的候选 broadcast op |
| `pattern` | `PosPattern` | 当前处理的模式 |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `isPosBroadcast` | `:82` | 判断 broadcast 是否为 VECTOR_ONLY |
| `collectPosPattern` | `:91` | 收集并验证 PosPattern，检查同 block 同 block_id |
| `findTargetBlock` | `:161`| 根据 extui 用户确定目标 block_id |
| `runOnOperation` | `:200` | 入口：遍历 broadcast，收集模式，确定目标，检查环，更新 block_id |
| `PosPattern::getAllOps` | `:77`| 返回模式内所有 op 的列表 |
| `createPosMaskPatternPass` | `:245` | 创建 pass 实例 |

### 6. 安全机制

- **模式完整性检查**：`collectPosPattern` 严格验证 broadcast 的用户数量、类型、谓词，以及 cmp 的用户类型。
- **同 block 同 block_id 检查**：确保模式内所有 op 当前在同一个计算块内。
- **目标 block 一致性检查**：所有下游用户必须在同一 block，且 block_id 有效。
- **环检测**：移动前调用 `willCreateCycle`，避免引入循环依赖。
- **同块跳过**：若目标 block 与源 block 相同，不做移动。

## 存在问题

1. 
2. 

## 目标场景

1. 
2. 
