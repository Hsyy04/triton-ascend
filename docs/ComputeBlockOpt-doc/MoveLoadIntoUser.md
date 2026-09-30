# MoveLoadIntoUserPass

## Overview

### 1. 目标

把 `alloc → memref.copy(GM→UB) → to_tensor` 这一 load 模式（及其同块依赖）搬到第一个 VECTOR 计算用户所在块，使数据加载与首个消费者同块，减少跨块 UB 滞留。

### 2. 规格

1. Module 必须不带 `fallback` 标注；否则整个 pass 直接跳过。
2. 待匹配模式：`memref::CopyOp`，其 `source` 必须源自全局内存（由 `CVPipeline::collectViewOpsAndCheckGlobalMemory` 判定），`target` 必须**回溯到**一个 `memref::AllocOp`（允许中间是 `ViewLikeOpInterface`）；随后 `allocOp` 的 user 中必须能找到 `bufferization::ToTensorOp`。
3. 匹配成功后，三 op（`allocOp` / `copyOp` / `toTensorOp`）必须共享**同一 block_id**；否则放弃。
4. `toTensorOp` 的首个 user 必须是 `VECTOR_ONLY` 且有有效 block_id，且该 block_id 与 `toTensorOp` 不同；否则放弃。
5. 首个 user 通过 `traceCalculateUser` 沿 `tensor::CollapseShapeOp` 单链（每个中间 collapse 单 user）追溯到第一个 block_id 不同的 op；这些 collapse 同样要并入 `matchedOps` 一起搬。
6. 跨块标量副作用由 `cloneScalarOpsForCrossBlockUses` 预先克隆，避免搬动后出现悬空。
7. `collectAllDependencies` 把 `matchedOps` 各自的同块前驱递归收集到 `opsToMove`，仅保留与 `commonBlockId` 同块的 op。
8. 改块前必须 `willCreateCycle` 通过，否则放弃。
9. 通过则对 `opsToMove` 中每个 op 调 `bm.updateBlockIdWithInner(op, targetBlockId)`（含内层 region）。

### 3. 算法流程

入口 `runOnOperation()`：

#### Step 0 — 准备与短路

- `hasFallbackAttr(module)` → return。
- 建 `AliasAnalysis`、`MemoryDependenceGraph memGraph`、`ComputeBlockIdManager bm`。

#### Step 1 — 模式匹配阶段

`module.walk(memref::CopyOp copyOp)`：

- `matchLoadPattern(info)`：
  1. `collectViewOpsAndCheckGlobalMemory(copyOp.getSource(), sourceViewOps)`：source 源自 GM 则通过。
  2. `dest = copyOp.getTarget()`；`getOriginAlloc(dest)`：若 dest 直接是 `memref::AllocOp`，返回它；若为 `ViewLikeOpInterface` 则沿 `getViewSource` 递归向上查；找不到则失败。
  3. 遍历 `allocOp->getUsers()` 找首个 `bufferization::ToTensorOp`，找不到则失败。
  4. `matchedOps = {allocOp, copyOp, toTensorOp}`。
- `checkAllOpsInSameBlock(info, bm)`：3 op 任一 block_id=-1 或不全相等 → 失败；成功则记录 `info.commonBlockId`。
- 全部通过则 `validPatterns.push_back(info)`。

#### Step 2 — 决定首个 VECTOR user 与目标块

`getFirstUser(toTensorOp, bm, matchedOps)`：

1. 遍历 `toTensorOp->getUsers()`；对每个 user 调 `getAncestorInBlock(user, toTensorOp->getBlock())` 取块内可见祖先；为空则跳过。
2. `traceCalculateUser(userInBlock, bm, bm.getBlockIdByOp(toTensorOp), userChain)`：循环"当前 op 与 `toTensorOp` 同 block_id 且是 `tensor::CollapseShapeOp` 且 result 仅 1 user"，否则停，把链上 collapse push 进 `userChain`。
3. 选"IR 顺序最靠前"的 `firstUser`（`isBeforeInBlock` 判定），并把对应的 `fistUserChain` 记下。
4. `firstUser` 必须 `core_type == VECTOR_ONLY`、`block_id != -1` 且与 `toTensorOp` 不同块。
5. 把 `fistUserChain`（collapse 链）append 到 `matchedOps`。

#### Step 3 — 收集同块依赖

`opsToMove = SetVector`；对 `matchedOps` 每个 op 调 `collectAllDependencies(op, opsToMove, commonBlockId, bm, memGraph)`：

- 已存在 → 跳过。
- `blockId == -1` 或 `blockId != commonBlockId` → 不递归（跨块依赖不进 opsToMove）。
- 否则插入 `opsToMove`；用 `DependencyHelper.forEachSource(op, ...)` 取所有 SSA/内存前驱递归调用。

#### Step 4 — 跨块标量克隆与环检测

- `CVPipeline::cloneScalarOpsForCrossBlockUses(bm, opsToMove, targetBlockId)`：把跨块被用的标量 op 在目标块克隆副本，避免搬动后悬空。
- `CVPipeline::willCreateCycle(opsToMove.getArrayRef(), memGraph, targetBlockId, bm)`：返回 true 则跳过本次模式。

#### Step 5 — 改块

对 `opsToMove` 每个 op 调 `bm.updateBlockIdWithInner(op, targetBlockId)`：把 block_id（含内层 region）一起改成 `firstUser` 的 block_id。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `LoadPatternInfo` | struct | 一次匹配的上下文：`matchedOps` / `commonBlockId` / `toTensorOp` / `copyOp` / `allocOp` |
| `validPatterns` | `SmallVector<LoadPatternInfo>` | 所有通过 Step 1 的模式 |
| `firstUser` | `Operation*` | `toTensorOp` 后的首个 VECTOR_ONLY 不同块 op（可经过 collapse 链） |
| `userChain` | `SmallVector<Operation *>` | `firstUser` 之前的 `tensor::CollapseShapeOp` 链 |
| `opsToMove` | `SetVector<Operation *>` | matchedOps + 同块 SSA/内存前驱，最终一起改块 |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `MoveLoadIntoUserPass::runOnOperation` | `MoveLoadIntoUserPass.cpp:244` | 入口：fallback 短路 → walk copyOp → 匹配 → 决策 → 收集依赖 → 环检测 → updateBlockIdWithInner |
| `matchLoadPattern` | `:179` | 校验 source=GM、dest 源自 allocOp、allocOp 后有 toTensor，填充 `matchedOps` |
| `getOriginAlloc` | `:166` | 沿 `ViewLikeOpInterface::getViewSource` 递归回溯源 alloc |
| `checkAllOpsInSameBlock` | `:142` | 校验 `matchedOps` 三 op 同 block_id |
| `traceCalculateUser` | `:75` | 沿 collapse 单链延伸，记录中间 collapse 进 `userChain` |
| `getFirstUser` | `:95` | 在 toTensor user 中选 IR 序最前的 VECTOR_ONLY 不同块 op，并把 collapse 链 append 进 `matchedOps` |
| `collectAllDependencies` | `:222` | 递归收集同块 SSA/内存前驱到 `opsToMove` |
| `DependencyHelper` | `Common/MemoryEffectsTracker.h`（外部） | 提供 `forEachSource` 遍历 op 的 SSA/内存前驱 |
| `CVPipeline::collectViewOpsAndCheckGlobalMemory` | `Common/Utils.h`（外部） | 校验 source view 链源自 GM |
| `CVPipeline::cloneScalarOpsForCrossBlockUses` | 同上 | 跨块被用标量 op 的目标块克隆 |
| `CVPipeline::willCreateCycle` | 同上 | 模拟改块环检测 |
| `createMoveLoadIntoUserPass` | `:312` | 注册入口，pass 名 `"move-load-into-user"` |

### 6. 安全机制

- **Fallback 短路**：模块带 fallback 标注时整个 pass 直接 return。
- **source=GM 校验**：仅当 copy 的 source 源自全局内存时匹配，避免误把内部 UB↔UB 拷贝搬走。
- **alloc 回溯**：`getOriginAlloc` 容忍 `ViewLikeOpInterface` 中间层，但最终必须指向 `memref::AllocOp`。
- **同块先决**：`allocOp/copyOp/toTensorOp` 必须同 block_id 才能整体搬运，否则只搬部分会破坏一致性。
- **首个 VECTOR user 验证**：必须 `core_type == VECTOR_ONLY`、`block_id != -1`、与 `toTensorOp` 不同块。
- **collapse 单链限制**：`traceCalculateUser` 要求每个中间 collapse 仅 1 user 才延伸，防止多分叉崩塌被错误卷入。
- **跨块标量预克隆**：`cloneScalarOpsForCrossBlockUses` 在改块前把跨块被用标量先克隆，避免搬动后悬空。
- **环检测**：`willCreateCycle` 在最终改块前模拟。
- **同块依赖闭包**：`collectAllDependencies` 通过 `block_id == commonBlockId` 限定，确保搬走 opsToMove 后无跨块悬空。

## 目标场景

1. 任意含"GM→UB→tensor"加载且后续在另一个 VECTOR 块被消费的 kernel（如 embedding lookup、weight tile load），合并到首个 VECTOR 消费块后消除跨块 UB 滞留。