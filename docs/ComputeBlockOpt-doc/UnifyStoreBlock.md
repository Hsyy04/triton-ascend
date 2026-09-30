# UnifyStoreBlockPass

## Overview

### 1. 目标

将 store 语义操作（`bufferization::MaterializeInDestinationOp`、`hivm::StoreOp`）合并到其生产者所在的 VECTOR 计算块中，以减少跨块数据搬运。

### 2. 规格

1. 仅处理 core type 为 `VECTOR_ONLY` 的 store 操作。
2. 生产者必须是 VECTOR 类型计算 op，且与 store 在同一 block 内，两者都有有效的 block_id。
3. 追溯 store 的 source，跳过 view-like op（`ViewLikeOpInterface`、`tensor::ExtractSliceOp`）和 scalar-like 值，直到找到真正的生产者。
4. 收集需要统一的操作：store 本身、生产者、数据链和目的链上的 view op（要求与 store 同 block）、以及递归收集的 scalar 依赖（要求与 store 同 big block）。
5. 在应用前，克隆跨块使用的 scalar 操作以避免环。
6. 进行环检测，通过后将所有收集到的操作的 block_id 更新为生产者的 block_id。

## 3. 算法流程

入口 `runOnOperation()`：

1. 遍历 module，收集所有满足 `isStoreOp` 且 core type 为 `VECTOR_ONLY` 的 store 操作。
2. 对每个 store 操作调用 `matchStorePattern`：
   - 获取 store 的 block_id，若无效则跳过。
   - 调用 `traceProducerOp` 追溯生产者，同时收集数据链上的 view op。
   - 若生产者不存在、或 block_id 无效、或与 store 不在同一 block，则跳过。
   - 构建 `matchedOps`：首先插入生产者（必须位于第一个），然后插入 `collectOpsToUnify` 返回的 ops（包括 store、view ops、scalar deps）。
3. 若匹配成功：
   - 调用 `CVPipeline::cloneScalarOpsForCrossBlockUses` 克隆跨块使用的 scalar 操作。
   - 调用 `applyStoreUnify` 进行环检测，若通过则更新 block_id。

`traceProducerOp(storeOp, dataViewOps)`：

- 从 store 的 source 开始，循环获取定义 op：
  - 若是 view-like 或 `tensor::ExtractSliceOp`，加入 `dataViewOps`，继续向上追溯其 source。
  - 若是 scalar-like，返回 nullptr。
  - 否则是真正的计算 op，检查其 core type 是否与 store 相同，相同则返回该 op，否则返回 nullptr。
- 若到达 block argument，返回 nullptr。

`collectViewOpsToUnify(ops, storeOp, dataViewOps, seen)`：

- 将 `dataViewOps` 中与 store 同 block 的 op 加入 `ops`。
- 从 store 的 dest 开始，向上追溯 view-like op，若与 store 同 block 则加入 `ops`。

`collectScalarDeps(ops, storeOp, seen)`：

- 从 `ops` 和 store 的父级控制流 op（如 `scf.for`、`scf.if`）开始，递归收集所有 scalar-like 操作数对应的定义 op，仅当定义 op 与 store 在同一 block 时加入 `ops`。

`collectOpsToUnify(storeOp, dataViewOps, bm)`：

- 返回 `{store}` + view ops + scalar deps。

`matchStorePattern(storeOp, bm, matchedOps)`：

- 若 store block_id 无效，返回 false。
- 追溯生产者，若失败返回 false。
- 检查生产者 block_id 有效且与 store 同 block，否则返回 false。
- 将生产者插入 `matchedOps` 第一个，再将 `collectOpsToUnify` 的结果插入。
- 返回 true。

`applyStoreUnify(matchedOps, memGraph, bm)`：

- 取 `matchedOps[0]` 为生产者，获取其 block_id 为目标。
- 将 `matchedOps` 转为 `SmallVector`，调用 `willCreateCycle` 检测环，若成环返回 false。
- 否则遍历 `matchedOps`，调用 `bm.updateBlockId(op, targetBlockId)`。
- 返回 true。

## 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `matchedOps` | `SetVector<Operation *>` | 匹配到的操作集合，生产者位于第一个 |
| `dataViewOps` | `SmallVector<Operation *>` | 数据链上的 view 操作 |
| `ComputeBlockIdManager` | 类 | 管理 op 与 block_id 的映射 |
| `MemoryDependenceGraph` | 类 | 内存依赖图，用于环检测 |
| `seen` | `SmallPtrSet<Operation *, 16>` | 去重集合 |

## 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `isStoreOp` | :55 | 判断是否为 store 操作 |
| `getStoreSource` / `getStoreDest` | :60/:71 | 获取 store 的源/目的值 |
| `isViewOrExtractSliceOp` | :82 | 判断是否为 view-like 或 extract_slice |
| `getViewSourceValue` | :86 | 获取 view 操作的源值 |
| `traceProducerOp` | :96 | 追溯 store 的生产者，收集数据链 view op |
| `collectViewOpsToUnify` | :130 | 收集数据链和目的链上的 view op |
| `collectScalarDeps` | :165 | 递归收集同 block 的 scalar 依赖 |
| `collectOpsToUnify` | :209 | 组装需要统一的操作列表 |
| `matchStorePattern` | :236 | 匹配 store 模式，构建 matchedOps |
| `applyStoreUnify` | :276 | 环检测并更新 block_id |
| `runOnOperation` | :312 | 入口：收集 store，匹配，克隆 scalar，应用 |
| `createUnifyStoreBlockPass` | :365 | 创建 pass 实例 |
| `registerUnifyStoreBlockPass` | :369 | 注册 pass |

## 6. 安全机制

- **仅处理 VECTOR_ONLY 的 store**：过滤非 VECTOR 核的 store 操作。
- **block_id 有效性检查**：store 和生产者都必须有有效的 block_id（≠ -1）。
- **同 block 要求**：生产者必须与 store 在同一 block；收集 view op 和 scalar deps 时也要求与 store 同 block 或同 big block。
- **scalar 克隆**：在移动前调用 `cloneScalarOpsForCrossBlockUses` 克隆跨块使用的 scalar 操作，避免引入环。
- **环检测**：`applyStoreUnify` 中调用 `willCreateCycle`，若成环则放弃该模式的统一。

## 存在问题

1. 可选保留/移除
2. 待合并op：store op, data/dest链上的view op, scalar deps（Block过滤：只收与store同一MLIR Block的）

## 目标场景

1. 将store相关操作的`block_id`修改为producer的`block_id`，使store与数据生产者位于同于同一compute block。 只处理producer为VECTOR的情况
