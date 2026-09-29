# UBUsageOpt

## Overview

### 1. 目标

在循环体（`scf.for` / `scf.while`）内，通过改变个别VECTOR类型op的block_id, 尽可能减小计算块之间的数据搬运。如下图, 不同颜色代表不同的计算块

![](img/UBUsageOpt-1.PNG)

### 2. 规格

1. 仅考虑for/while循环体中的VECTOR类型op，CUBE类op不切分（同一块内共享一个节点）。
2. 仅考虑统一region层级的op， 对于包含其他retion的op，其包含的op不参与优化， 将看作一个整体处理。（例如ifOp的子op不做任何调整）
3. 对于循环变量的处理：
    - 在不成环的情况下， 循环变量的使用者将被上提到循环变量的定义者所在的block中， 以减少跨块数据搬运。
4. 在满足上述条件的情况下， 仅考虑如下op的上提：
    - 上提前后仅涉及两个块， 即，srcOp{block_id = id_src}, dstOp{block_id = id_dst}, 其中id_src != id_dst,  dstOp依赖op均为id_dst或id_src
    - 上提后， dstOp的依赖也将全部上提，以确保不成环


### 3. 算法流程

入口 `runOnOperation()` 遍历 module 中所有 block，对每个 block 调用 `UBUsageOptimization`。
后者仅处理父 op 为 `scf.for`/`scf.while` 的 block，分四步：

#### Step 1 — 建图 `buildUBUsageGraph`

构建带权有向依赖图，输出若干并行数组（`linkOut/linkIn/linkSize/linkStart/linkEnd` + `nodeBlockId/nodeCoreType/nodeArgs`）。

| 要素 | 说明 |
|------|------|
| **节点** | block 内每个 op 一个节点。**Cube 块坍缩**：同一 block_id 的 CUBE_ONLY op 共享一个节点（`cubeBlockId2nodeId`），避免 Cube 块内部产生自环。Vector op 各自独立。 |
| **SSA 边** | 对每个 op（walk 进嵌套 region）的每个操作数，若定义 op 的 block 内祖先是别的 op，则加边 `src→dst`，权 = `getValueSizeInBytes(operand)`。 |
| **内存边** | 由 `memGraph.getExecBefore(op)` 得到的内存执行前驱，加权 0 的边。 |
| **循环携带边** | 若操作数是 BlockArgument 且对应 yield 槽（loop-carried），则回溯到 yield 值在 block 内的定义 op，标 `fromArgEdge=true`，**边权 ×2**（因为值跨边界两次：yield 出去 + 下轮 re-enter）。 |
| **自环规避** | ① 操作数定义 op 的 block 内祖先 == 当前 op 本身时跳过；② 同一 Cube 块内两 op 间的边跳过。 |

**边权函数 `getValueSizeInBytes`**（`UBUsageOptPass.cpp:114`）：

| Value 类型 | 边权 | 语义 |
|-----------|------|------|
| Tensor（静态 shape） | numElements × elemBytes | 直接用数据字节数 |
| Tensor（动态 shape） | `MAX_EDGE_SIZE` (1<<30) | 不可在此切分 |
| MemRef | `MAX_EDGE_SIZE` | memref 不应被切分 |
| Vector | numElements × elemBytes | 数据字节数 |
| Index | 0 | 不占 UB |
| 其他 | 0 | 不占 UB |

`nodeArgs[nodeId]`：记录该节点产出的是 yield 第几个参数（循环携带值追踪用）。

#### Step 2 — 收集候选 `collectNeedUbOpts`

遍历所有节点，筛选**源候选节点**：

1. 节点 coreType 必须是 `VECTOR_ONLY`，且 block_id ≥ 0。
2. 该节点的某条出边指向一个 **"活跃终点"（active end node）**——`isActiveEndNode` 返回 true。
3. 满足则把该节点加入 `needUbOpts[srcBlockId]`。

**`isActiveEndNode(srcNode, endNode)`**（`:352`）判定条件：

- endNode 与 srcNode 同核类型（都 VECTOR）；
- endNode 有有效 block_id（≠ -1，排除 yield/return）；
- endNode 与 srcNode 不同块；
- **后向依赖安全**：对 endNode 做后向 BFS（`findDependency`，排除 srcNode 路径），所有依赖节点要么在 endNode 块、要么在 srcNode 块；若在**第三块**则必须无入边（是根节点）——保证上提 endNode 不会拖入无法解析的第三块依赖。

#### Step 3 — 计算切点并记录变更 `collectRecordChange`

核心优化逻辑（`:467`）。对每个 `optBlockId` 的每个 `optNode`：

1. **收集活跃集** `activateSet`：optNode 出边指向的所有 active end node（去重）。

2. **对每个 activateNode**：
   - `originUBSize = sumIncomingLinkSize(activateNode)`：该节点的**跨块入边**权之和（当前块边界 UB 成本）。若含 `MAX_EDGE_SIZE` 边则整体为 MAX。
   - **沿单后继链前进**：从 activateNode 起，反复调用 `findUniqueDependentNode` 扩展 `chain`：
     - `findUniqueDependentNode`（`:442`）：curNode 必须恰有 1 条出边；目标节点与 curNode 同块；且目标的全部后向依赖（排除 curNode 路径）都在同块内。保证是一条块内线性单消费链。
   - **遍历 chain 逐点算切分代价**：对每个 `chain[i]`，`nowUBSize = Σ chain[i] 的出边权`（若含 MAX 则为 MAX）。这代表"若在 chain[i] 之后切断，新边界需跨块传输的 UB"。
   - 取 `min(nowUBSize)` 对应的位置，设 `bestCutPointIdx = i+1`。
   - **仅当 `bestCutPointIdx > 0`（即确实找到更优点）时记录变更**：
     - `recordChange[chain[0..bestCutPointIdx-1]] = optBlockId`；
     - 对每个被移动的 `chain[i]`，还收集其**跨块后向依赖**（`findDependency(chain[i], chainPreNode)`，其中 `chainPreNode` 前一个节点或 optNode），若依赖不在 optBlockId 则一并记录并入源块。

**直觉**：`originUBSize` 是当前块边界的数据搬运成本；把链上一段并入源块后，新边界移到该段末尾，`nowUBSize` 是新边界的成本。选最小 `nowUBSize`，且仅当它小于原始成本才动，从而保证 UB 占用不增反减。

#### Step 4 — 应用变更 `applyRecordChange`

- 把 `recordChange` 按 targetBlockId 分组（`blockWilladd`）。
- 对每组：`willCreateCycle` 环检测 → 通过则 `updateBlockIdWithInner`（含内层 region 一起改 block_id）；不通过则跳过该组（`hasError=true` 但不中断整体）。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `op2nodeId` / `nodeId2op` | `DenseMap` | op ↔ 节点编号双向映射 |
| `linkOut[node]` / `linkIn[node]` | `SmallVector<int>` | 节点的出/入边 ID 列表 |
| `linkSize[edge]` | `int64_t` | 边权（UB 字节数） |
| `linkStart[edge]` / `linkEnd[edge]` | `int` | 边的源/目标节点 |
| `nodeBlockId[node]` | `int` | 节点的 block_id（-1 表示无/控制流） |
| `nodeCoreType[node]` | `int` | 节点的 core_type |
| `nodeArgs[node]` | `int` | 节点产出的 yield 参数下标（-1 表示非 yield 产出） |
| `needUbOpts[blockId]` | `SmallVector<int>` | 每个 block_id 下的候选源节点列表 |
| `recordChange` | `DenseMap<int,int>` | nodeId → 目标 blockId 的待变更映射 |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `getValueSizeInBytes` | `:114` | 计算 Value 的 UB 边权 |
| `buildUBUsageGraph` | `:154` | 构建带权依赖图（Cube 块坍缩、SSA+内存边、循环携带边 ×2） |
| `findDependency` | `:325` | 后向 BFS，返回 targetNode 的全部依赖节点（排除 preNode 路径） |
| `isActiveEndNode` | `:352` | 判定下游节点是否可安全上提（同核、不同块、后向依赖安全） |
| `collectNeedUbOpts` | `:387` | 收集有活跃下游的 VECTOR 源候选节点 |
| `sumIncomingLinkSize` | `:425` | 求节点跨块入边权之和（当前边界 UB 成本） |
| `findUniqueDependentNode` | `:442` | 判定单后继链可否继续延伸（唯一出边、同块、依赖封闭） |
| `collectRecordChange` | `:467` | 核心：沿链找最小 UB 切点，记录待变更节点及依赖 |
| `applyRecordChange` | `:553` | 分组环检测后批量 `updateBlockIdWithInner` |
| `UBUsageOptimization` | `:590` | 单 block 编排：建图→候选→切点→应用 |
| `runOnOperation` | `:637` | 入口：遍历所有 block |

### 6. 安全机制

- **仅 for/while body**：`UBUsageOptimization` 首先检查 `isa<scf::ForOp/WhileOp>`，非循环体直接返回。
- **活跃性后向依赖检查**：`isActiveEndNode` 确保上提不会引入第三块无法解析的依赖。
- **单链封闭性**：`findUniqueDependentNode` 保证链上每步的后向依赖都在同块内。
- **环检测**：`applyRecordChange` 每组变更前 `willCreateCycle`，不通过则跳过。

## 存在问题
1. Q4预计删除规格2

> 仅考虑统一region层级的op， 对于包含其他retion的op，其包含的op不参与优化， 将看作一个整体处理.(例如ifOp的子op不做任何调整)

添加一条arg更新者(yield的结果)到 同层级使用者的边. 到了ComputeOpt应该保证不成环, 如果成环, 那么其中一定包括了不可跨越的CUBE计算, 那么应该删掉这条边. 

2. 预计删除环检测

## 目标场景

1. flash_attention_npu_v8_generalized_c9.py::fwd_kernel
2. nsa_bwd_dq (Q4)
