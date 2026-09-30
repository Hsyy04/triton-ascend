# MergeSameSourceAxisPass

## Overview

### 1. 目标

当一个 tensor 产生 op（source）被 ≥2 个 VECTOR 块消费，且这些消费链在下游某个 op（convergence）汇聚时，把汇聚链上的 op 并入 source 所在块，使同一数据轴上的 op 归一到同一块。

### 2. 规格

1. 仅考虑 `core_type == VECTOR_ONLY`、有 block_id 且结果为单 `TensorType` 的 op 作为候选 source；`arith::ConstantOp` 显式跳过。
2. 仅收集"在"用户：与 source 同 region、`VECTOR_ONLY`、有有效 block_id，且 block_id ≠ source 的 block_id。
3. 仅当满足"同 region 跨越 source 所在块且 ≥2 个消费分支"时才触发；否则直接返回。
4. 跨分支必须能在某条 VECTOR_ONLY 同 region 下游 op 处**首次交汇**（nearest convergence）；找不到交汇点则放弃。
5. 交汇点（`convergenceOp`）的所有"非 source 上游 def op"必须满足以下三选一：① def 在 source 所在 block；② def 不与 source 同 region；③ def 已经在待移动链 `chainOps` 中；否则放弃（防止引入无法解析的第三块依赖）。
6. 通过 `willCreateCycle` 环检测后才真正改块；否则放弃本次合并。
7. Fallback：module 上若带 `fallback` 标注，整个 pass 直接跳过。

### 3. 算法流程

入口 `runOnOperation()` 在 ModuleOp 上构建 `MemoryDependenceGraph` 与 `ComputeBlockIdManager`，先收候选 source 列表，再逐个尝试合并。

#### Step 1 — 收集候选 source `candidates`

`module.walk`：遍历每个 op，满足下列全部才入列：
- `core_type == VECTOR_ONLY`；
- `getOpBlockId(op)` 有值（即带 block_id 属性）；
- `op->getUsers()` 非空；
- 结果数量恰为 1 且为 `TensorType`。

#### Step 2 — 单 source 处理 `tryMergeSource(source, memGraph, bm)`

1. 读 source 自身 `srcBlockId`，无效直接返回。
2. 跳过 `arith::ConstantOp`。
3. 跳过非单 `TensorType` 结果的 op。
4. 收集 `consumers`：所有 user 中同 region、VECTOR_ONLY、有有效 block_id 且 `block_id != srcBlockId` 的 op；不足 2 个则放弃。
5. 调用 `findNearestConvergence` 在这些 consumer 上做多源 BFS，寻找它们下游首次交汇的 VECTOR_ONLY 同 region op；找不到则放弃。
6. **交汇点上游守卫**：遍历 `convergenceOp` 的 operand，对每个有 defining op、且与 source 同 region、非 source 本身、非 chainOps 成员的 def op，要求其 block_id == `srcBlockId`（即"将被新边界覆盖"），否则放弃合并。
7. `willCreateCycle(chainOps, memGraph, srcBlockId, bm)` 模拟环检测；失败则跳过。
8. 通过则对链上每个 op `bm.updateBlockId(op, srcBlockId)`，统一到 source 块。

#### Step 3 — 多源 BFS 找最近汇聚点 `findNearestConvergence`

- 每个 consumer 起始 BFS 节点 `(consumer, consumerIdx)`，维护 `firstIdx[op]` 记录"该 op 首次被哪个 consumer 的搜索前沿到达"以及 `parent[op]` 反向链。
- 沿 `cur->getUsers()` 扩展，仅扩展 `VECTOR_ONLY` 且与 source 同 region 的 user。
- 命中一个已被"另一 consumer 前沿"标记的 user 时，说明两条搜索前沿在此交汇。**若 user 本身就是 consumer 起点**，则视为同一起点的另一路径，跳过（避免伪汇聚）。
- 交汇后 `chainOps` 由两部分组成：
  - `appendPath(convergenceOp, cons1, parent, inChain, ...)`：沿 parent 链从 convergenceOp 走到 cons1；
  - `appendPath(cur, cons2, parent, inChain, ...)`：从当前节点 cur 沿 parent 链走到 cons2（去重）。
  - `extendChainWithConvergenceTail`：在 convergenceOp 所在块、再追加其同块 VECTOR_ONLY 直接 user（"收敛尾巴"），防止搬走 convergence 后留半截尾巴。
- 找不到任何交汇点返回 false。

#### Step 4 — `appendPath` 反向链回溯

- 从 `start` 出发，沿 `parent` 链朝 `target` 走，使用 `inChain` 集合去重。
- 走到 `target` 即停（`pathOp == target` 跳出）。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `candidates` | `SmallVector<Operation *>` | 候选 source 列表（VECTOR_ONLY + tensor 结果 + 有 user） |
| `consumers` | `SmallVector<Operation *>` | 当前 source 在其他 VECTOR 块的下游 VECTOR_ONLY 用户 |
| `firstIdx` | `DenseMap<Operation *, int>` | BFS 中"op 首次到达时所属 consumer 编号" |
| `parent` | `DenseMap<Operation *, Operation *>` | BFS 父链，用于 `appendPath` 重放 consumer → convergence 路径 |
| `bfsQueue` | `SmallVector<pair<Op*,int>>` | BFS 队列 |
| `chainOps` | `SmallVector<Operation *>` | 待并入 source 块的 op 集合（含 consumer、路径、收敛尾） |
| `inChain` | `DenseSet<Operation *>` | `chainOps` 去重辅助集合 |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `MergeSameSourceAxisPass::runOnOperation` | `MergeSameSourceAxisPass.cpp:257` | 入口：fallback 检查、构图、收候选、逐个 `tryMergeSource` |
| `tryMergeSource` | `:156` | 单 source 全流程：collect consumers → find convergence → guard → cycle-check → updateBlockId |
| `isInSameRegion` | `:49` | 判定两 op 是否在同一 region（`getParentRegion()` 相等） |
| `appendPath` | `:55` | 沿 `parent` 链从 start 反向追到 target，去重加入 `chainOps` |
| `extendChainWithConvergenceTail` | `:71` | 把 convergenceOp 的同块 VECTOR_ONLY 直接下游补入 chainOps |
| `findNearestConvergence` | `:97` | 多源 BFS，找首个跨 consumer 交汇的 VECTOR_ONLY 同 region op，产出待搬入链 |
| `createMergeSameSourceAxisPass` | `:290` | 注册入口，pass 名 `"merge-same-source-axis"` |

### 6. 安全机制

- **核心类型门控**：source 与候选 consumer 都必须 `VECTOR_ONLY`；CUBE/控制流场景被显式排除。
- **同 region 门控**：消费链与 source 必须在同一 region，避免跨 region 操作出错。
- **≥2 消费者门槛**：单个消费者不构成"轴汇聚"语义，直接返回。
- **convergenceOp 上游守卫**：所有"非 source 上游 def op"必须归 source 块或同 chainOps，防止把块边界切到无法解析的第三块依赖上。
- **环检测**：改块前 `willCreateCycle(chainOps, memGraph, srcBlockId, bm)` 模拟。
- **Fallback 短路**：模块带 fallback 标注时整个 pass 直接 return。
- **ConstantOp 豁免**：`arith::ConstantOp` 直接跳过，避免常量被错误绑定到某一轴。

## 目标场景

1. 多分支 VECTOR 计算沿同一 source 收敛的 kernel（如 layernorm、softmax、reduce 类），合并后可避免在收敛点前的多块并行维护同一中间结果。