# BroadcastUBOptPass

## Overview

### 1. 目标

当 `linalg.broadcast` 的所有用户都在同一个 VECTOR 块时，把 broadcast 搬到该用户块，减少一次跨块 UB 传输。

### 2. 规格

1. 仅当 broadcast op 本身的 `core_type` 为 `VECTOR_ONLY` 时考虑。
2. 仅当 broadcast 至少有一个用户且第一个用户与 broadcast 位于**同一 MLIR Block**（说明在同一 region 内可见）时考虑。
3. 仅当第一个用户的 block_id 有效（≠-1）且为 `VECTOR_ONLY` 时考虑；控制流场景或非 VECTOR 用户直接跳过。
4. 仅当 broadcast 的**所有用户 block_id 都相同**时考虑：若任一用户在不同块，则放弃合并（避免把 broadcast 同时绑到多个块上引入新跨块传输）。
5. 仅当 broadcast 自身的 block_id 与目标用户块不同（同块已就位）时才执行；否则跳过。
6. 移动前必须通过 `willCreateCycle` 环检测，失败则放弃。

### 3. 算法流程

入口 `runOnOperation()` 在 ModuleOp 上构建 `MemoryDependenceGraph` 与 `ComputeBlockIdManager`，然后 `walk(linalg::BroadcastOp)` 遍历每个 broadcast：

#### Step 1 — 筛选候选

1. `getOpCoreType(op) != VECTOR_ONLY` 直接返回；
2. 无用户直接返回；
3. 取首个用户 `oneUser`；
4. `oneUser->getBlock() != op->getBlock()` 直接返回（不同 region，可见性不符）；
5. `firstUserBlockId = bm.getBlockIdByOp(oneUser)`；若为 -1 或 core_type 非 VECTOR_ONLY，返回。

#### Step 2 — 全用户同块判定

- `llvm::all_of` 检查 `op->getUsers()`：每个用户的 `getBlockIdByOp(user) == firstUserBlockId`；
- 任意用户不在该块 → 放弃。

#### Step 3 — 已就位短路

- `broadcastBlockId = bm.getBlockIdByOp(op)`；
- 若 `broadcastBlockId == firstUserBlockId`，跳过（避免无意义重写）。

#### Step 4 — 环检测并改 block_id

- 构造 `opsToCheck = {op}`；
- `CVPipeline::willCreateCycle(opsToCheck, memGraph, firstUserBlockId, bm)`：若返回 true 则跳过；
- 通过则 `bm.updateBlockId(op, firstUserBlockId)`：把 broadcast 的 block_id 改到用户块。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `AliasAnalysis` | MLIR Analysis | 构造 `MemoryDependenceGraph` 所需的别名分析 |
| `CVPipeline::MemoryDependenceGraph` | 类 | 整模块 op 间内存依赖（`getExecBefore` 等查询） |
| `CVPipeline::ComputeBlockIdManager` | 类 | op ↔ block_id 双向映射 + `updateBlockId` / `willCreateCycle` |
| `linalg::BroadcastOp` | MLIR Op | 待优化对象（`linalg.broadcast`） |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `BroadcastUBOptPass::runOnOperation` | `BroadcastUBOptPass.cpp:69` | 入口：walk 全部 `linalg::BroadcastOp`，按上述 4 步过滤-检测-改块 |
| `CVPipeline::willCreateCycle` | `Common/Utils.h`（外部） | 给定一组 op + 目标 block_id，模拟改块并检测是否成环 |
| `CVPipeline::ComputeBlockIdManager::updateBlockId` | 同上（外部） | 修改 op 的 `block_id` 属性 |
| `CVPipeline::getOpCoreType` | `ComputeBlockOpt/Common.h`（外部） | 读取 op 的 core_type 标注 |
| `createBroadcastUBOptPass` | `BroadcastUBOptPass.cpp:120` | 注册入口，pass 名 `"broadcast-ub-opt"` |

### 6. 安全机制

- **同 region 强制**：`oneUser->getBlock() != op->getBlock()` 直接返回，保证候选 broadcast 与用户在同一 MLIR Block 中可见。
- **全用户同块检查**：任一用户落在不同 block_id 上即放弃，避免"半搬半留"反而引入新跨块。
- **已就位短路**：bcast 已在目标块时跳过，避免重复改块造成属性噪声。
- **环检测**：`willCreateCycle` 在改块前先模拟，失败则跳过。
- **核心类型门控**：仅处理 `VECTOR_ONLY` 的 broadcast 与首个 VECTOR_ONLY 用户，CUBE/控制流场景被显式排除。

## 目标场景

1. 任意含 `linalg.broadcast` 且下游仅在单个 VECTOR 块内被消费的 kernel。