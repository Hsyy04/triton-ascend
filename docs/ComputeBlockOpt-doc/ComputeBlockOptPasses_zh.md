# ComputeBlockOptPass 流水线 Pass 文档

本文档说明 `ComputeBlockOptPass.cpp` 编排的计算块（Compute Block）优化流水线中各个 pass 的作用。
流水线入口为 `ComputeBlockOptPass::runOnOperation()`（`ComputeBlockOptPass.cpp:37`），它在 module 级别依次
注册一系列子 pass。每个子 pass 都围绕 **block_id** 和 **core_type** 这两个核心概念对 IR 上的计算块边界做优化。

> 本文档不包含 `createReorderOpsByBlockIdPass`（该 pass 仅按 block_id 对 op 做拓扑排序重排，逻辑简单）。

---

## 1. 背景与核心概念

### 1.1 block_id 与计算块

每个 op 上携带一个 `block_id`（整数）属性。**block_id 相同的 op 属于同一个"计算块"**，在 Ascend NPU 上会被
作为一个整体调度执行。block_id 是本流水线几乎所有 pass 改写的对象——通过将 op 的 block_id 从 A 改为 B，
即等价于把该 op 从计算块 A "搬到" 计算块 B。

- `ComputeBlockIdManager`（代码中常缩写为 `bm`）：管理 block_id 属性的读写，提供 `getBlockIdByOp`、
  `updateBlockId`、`updateBlockIdWithInner`（连内层 region 一起改）、`markOpBlockId` 等接口。
- `getOpsByBlockId(id)`：返回某个 block_id 下的全部 op。

### 1.2 core_type（核类型）

每个 op 还携带 `core_type`，取值：

| 取值 | 含义 |
|------|------|
| `CUBE_ONLY` | Cube 核算子（如 `linalg.matmul`），跑在 Cube 单元 |
| `VECTOR_ONLY` | Vector 核算子（elementwise、broadcast 等），跑在 Vector 单元 |
| `CUBE_AND_VECTOR` | 混合，一个 block_id 内同时含 Cube 和 Vector 算子 |
| `UNDETERMINED` | 未定（通常是控制流 op） |

很多 pass 只处理 `VECTOR_ONLY` 块，且**禁止把 CUBE 算子并入 VECTOR-only 的 block_id**，否则下游
（如 `DataDependencyAnalysis` 的 `collectBlockInfo`）会得到混合核类型的块而无法处理。

### 1.3 UB（Unified Buffer）

UB 是 NPU 上的片上快速缓存。计算块切分必须兼顾 UB 容量：块太大放不下，块太小则跨块数据搬运开销大。
多个 pass 的优化目标之一就是 **在不超 UB 容量的前提下尽量合并 block，减少跨块传输**。

### 1.4 MemoryDependenceGraph 与环检测

`MemoryDependenceGraph`（别名分析构建）记录 op 间的内存依赖（别名写后读等）。`willCreateCycle(ops, memGraph,
targetBlockId, bm)` 在把一组 op 并入 targetBlockId 之前检查是否会产生依赖环——**几乎所有合并类 pass 都会在
改写 block_id 之前调用它做安全检查**，避免破坏可调度性。

### 1.5 同步 op 与 SyncWall

`isSyncOp(op)` 判断 op 是否为同步/屏障 op。同步 op 拥有自己**独占的 block_id**，绝不能并入计算块，
否则 before/after 之间的栅栏语义会丢失。`SyncWall` 则用于判断两个 op 之间是否存在同步屏障。

### 1.6 标量闭包克隆

当把 op 从一个块搬到另一个块时，若它依赖一些标量产生 op，直接搬可能导致跨块依赖环。`ScalarClosure` /
`cloneScalarOpsForCrossBlockUses` 会把这些**标量依赖 op 克隆一份到目标块**，从而断开环。

### 1.7 Fallback 机制

`hasFallbackAttr(module)` / `setFallbackAttr`：若 module 已标记 fallback 或某 pass 失败，则直接跳过后续优化，
回退到非优化路径。流水线整体在 `runPipeline` 失败时会设置 `ERRCODE_FAILED`。

---

## 2. 流水线总览

`ComputeBlockOptPass::runOnOperation()` 按以下顺序添加子 pass（省略 `ReorderOpsByBlockIdPass`）：

```
阶段一：load 侧块统一
  1. UnifyAllocBlockPass
  2. MergeVectorIfBlockPass

阶段二：UB 占用优化与块切分
  3. UBUUsageOptPass
  4. BroadcastUBOptPass
  5. PosMaskPatternPass
  6. MergeSameSourceAxisPass
  7. MergeSmallBlockPass          (第一次)

阶段三：i1 生产者下沉
  8. SinkI1ProducersIntoUsersPass

阶段四：Fixpipe 与控制流拆分
  9. FixpipeOptPass
  10. SplitIfByBlockIdPass
  11. MoveLoadIntoUserPass
  12. UnifyStoreBlockPass
  13. ExpSubfPatternPass
  14. MergeSmallBlockPass         (第二次)

阶段五：大块合并
  15. MergeComputeBlockPass
  16. MergeCubeBlockPass
  17. MergeInputInitSharedCubeBlockPass
  18. RelocateMemrefDeclPass
```

下面逐个说明（编号对应上述顺序）。

---

## 3. 各 Pass 详解

### 3.1 UnifyAllocBlockPass (`unify-alloc-block`)

**文件**: `UnifyAllocBlockPass.cpp:319`

**目的**: 把 `memref.alloc`、其 `scf.if` 内的 `linalg.fill`、以及关联的 `memref.subview`/`memref.copy`
链统一到同一个 block_id，让 load 语义的 op 成为一个整体块。

**关键逻辑**:
1. 收集 alloc 的直接用户（排除 `linalg.fill`，因为 fill 用的是 outs BlockArgument 而非 SSA 依赖）。
2. 查找在 `scf.if`（仅 then 分支、无 else）内、以该 alloc 为 outs 的 `linalg.fill`（`findFillOpInSCFIf`）。
3. 检查所有直接用户是否落在同一个 block_id（通过穿透 ViewLike op 找到 `memref.copy`，取其 block_id；
   多个不同则放弃）。
4. 若 scf.if 内除了 fill 还有别的 op，放弃（避免污染）。
5. 汇总 alloc / fill / if / 直接用户 / 源 view 链上的全部 op，做环检测后统一 block_id。

**安全检查**: `willCreateCycle`；若 if 体内有其他 op 则跳过。

---

### 3.2 MergeVectorIfBlockPass (`merge-vector-if-block`)

**文件**: `MergeVectorIfBlockPass.cpp:267`

**目的**: 把"纯 Vector"的 `scf.if` 块与它的上游数据源块、以及一个下游消费块合并成一个大块，
减少跨块 UB 占用。

**关键逻辑**:
1. `isPureVectorIf`：判断 scf.if 两个 region 内所有非终止 op 都是 `VECTOR_ONLY` 且无嵌套控制流。
2. `getUpstreamBlockId`：找 if 的 condition 及 region 内所有外部数据源的**共同 block_id**（常量、if 内部
   产生的值忽略）；要求唯一。
3. `collectDownstreamBlockIds`：收集消费 if 结果的下游 block_id，按程序序排列。
4. 对每个下游块：若该块也是纯 Vector（`isPureVectorOpGroup`，且不含 sync op），且 `willCreateCycle` 通过，
   则把 if + 下游块统一到上游 block_id。若无合适下游，退化为只把 if 并入上游。
5. `applyMerge`：改写 if 本身、其内层非 sync op、以及下游 op 的 block_id。

**安全检查**: 下游块必须全是 VECTOR_ONLY；sync op 不参与合并；环检测。

---

### 3.3 UBUUsageOptPass (`ub-usage-opt`) 

要保留，但是迭代变量的部分要拆出来。

**文件**: `UBUsageOptPass.cpp:59`

**目的**: 在循环体（`scf.for`/`scf.while`）内，以 **UB 占用边权** 为依据，在边权最小的位置切分
VECTOR 计算块，使切分后跨块的 UB 占用最小。

**关键逻辑**:
1. `getValueSizeInBytes`：把每个 Value 的 UB 占用作为边权——Tensor/Vector 用数据字节数；Memref 设为
   `MAX_EDGE_SIZE`（不可在此切分）；Index 设为 0。
2. `buildUBUsageGraph`：为 block 内每个 op（Cube 块整体坍缩为一个节点以减少自环）建图，SSA 依赖和内存
   执行依赖都作为边，边权 = Value 字节数（yield 跨块边权 ×2）。
3. `collectNeedUbOpts`：找出有活跃下游节点的 VECTOR 源块节点（`isActiveEndNode`：下游同核类型、不同块、
   且其依赖只来自源块或无依赖）。
4. `collectRecordChange`：对每个候选，沿唯一后继链（`findUniqueDependentNode`）遍历，计算每个切分点的
   出边 UB 总和，选最小点；把切分点之前的链节点及其依赖并入源块 block_id。
5. `applyRecordChange`：分块批量做环检测后 `updateBlockIdWithInner`。

**特点**: 这是唯一以 UB 占用数学优化为目标的 pass；只对 VECTOR 块生效（Cube 块整体作为一个节点不参与切分）。

---

### 3.4 BroadcastUBOptPass (`broadcast-ub-opt`)

**文件**: `BroadcastUBOptPass.cpp:48`

**目的**: 当 `linalg.broadcast` 的所有用户都在同一个 VECTOR 块时，把 broadcast 搬到该用户块，
减少一次跨块 UB 传输。

**关键逻辑**:
1. 只处理 `VECTOR_ONLY` 的 broadcast，且用户非空。
2. 取第一个用户，要求它与 broadcast 在同一 IR block，且有有效 block_id 且 `VECTOR_ONLY`。
3. 所有用户必须都在这同一个 block_id（`allUsersSameBlock`）。
4. broadcast 当前不在该块时，环检测通过则 `updateBlockId`。

**安全检查**: `willCreateCycle`（只检查 broadcast 自身）。

---

### 3.5 PosMaskPatternPass (`pos-mask-pattern-opt`)

**文件**: `PosMaskPatternPass.cpp:50`

**目的**: 识别"位置 mask 模式"——`broadcast → cmpi(eq) → extui` 和 `broadcast → cmpi(sle) → extui`
两路——把整组 op 搬到用户所在块，减少 UB 占用。

**关键逻辑**:
1. `isPosBroadcast`：broadcast 是 VECTOR_ONLY。
2. `collectPosPattern`：broadcast 恰好 2 个用户且都是 `arith.cmpi`（一个 `eq` 一个 `sle`，各单用）；
   各 cmpi 的唯一用户是 `arith.extui`；整组 op 当前同一 block_id 且同一 IR block。
3. `findTargetBlock`：两组 extui 的用户在同一 IR block 且共享同一 block_id（即目标块）。
4. 环检测通过后，把 5 个 op 整体 `updateBlockId` 到目标块。

**安全检查**: 模式必须完整匹配；`willCreateCycle`。

---

### 3.6 MergeSameSourceAxisPass (`merge-same-source-axis`)

**文件**: `MergeSameSourceAxisPass.cpp:242`

**目的**: 当一个 tensor 产生 op（source）被 ≥2 个 VECTOR 块消费，且这些消费链在下游某个 op
（convergence）汇聚时，把汇聚链上的 op 并入 source 所在块，使同一数据轴上的 op 归一到同一块。

**关键逻辑**:
1. source 必须是单结果 tensor 型、非常量、有 block_id。
2. 收集 source 的 VECTOR 消费者（跨块的）。
3. `findNearestConvergence`：BFS 从各消费者出发沿用户链搜索，找到第一个被 ≥2 个消费者链路同时到达的
   op（convergenceOp），记录两条路径上的 op（`appendPath`）和汇聚点的同 block 下游尾 op
   （`extendChainWithConvergenceTail`）。
4. 守卫：convergenceOp 的上游操作数定义 op 要么在链中、要么已在本合并涉及的 source 块，否则放弃
   （避免引入未覆盖的外部依赖）。
5. 环检测后把链上所有 op `updateBlockId` 到 source 块。

**安全检查**: 上游操作数对齐检查；`willCreateCycle`。

---

### 3.7 MergeSmallBlockPass (`merge-small-block`)

**文件**: `MergeSmallBlockPass.cpp:60`

**目的**: 把"小计算块"（计算 op 数 ≤ `MIN_VF_SIZE=3`）合并到其上游或下游相邻 VECTOR 块，消除过小碎片块。

**关键逻辑**:
1. `isTensorComputeOp`：判断是否为有效计算 op（linalg 非 copy/broadcast、elementwise tensor op 等），
   用于计数；copy/broadcast/fill-const 不计入。
2. 对每个 block_id 统计计算 op 数，≤3 且 VECTOR_ONLY 才是候选。
3. `collectMergeCandidates`：收集上游候选（操作数来自某单一块、非纯标量依赖、无 Cube 链接）和
   下游候选（结果被某块消费、无 Cube 链接）；各自过环检测。
4. `selectMergeTarget`：候选 >1 时按策略链筛选（依次）：
   - `FaFWDPatternStrategy`：匹配 Flash Attention 前向的 `exp→mulf→addf` 模式（subf 来自上游、reduce
     来自上游、addf 回写给循环迭代变量），命中则选下游块。
   - `UBOccupationStrategy`：按并入后节省的 UB 字节数取最大（`getOperandUB`/`getUserUB`）。
   - `IROrderStrategy`：按程序序取最早下游块。
   - `DefaultUpStrategy`：优先上游第一个，否则下游第一个。
5. **两次运行**：本 pass 在流水线中执行两次。`matchSIToFPSubMulPattern`（`sitofp→subf→mulf`）只在
   **第二次运行**时合并（通过 `kMergeSmallBlockFirstRunDone` 标记区分），且合并前用 `setSubBlockId` 记录
   原块 id。

**安全检查**: Cube 链接则无候选；纯标量依赖则上游候选清空；`willCreateCycle`。

---

### 3.8 SinkI1ProducersIntoUsersPass (`sink-i1-producers-into-users`)

**文件**: `SinkI1ProducersIntoUsersPass.cpp:50`

**目的**: 把产生 i1（tensor<i1>）的纯无副作用 op **下沉（复制）到每个 i1 消费块**，使各块各自持有副本，
避免跨块 i1 传输。

**关键逻辑**:
1. `isValidI1Producer`：非 SCF、单结果、结果含 `tensor<i1>`。
2. `isPureAndRegionless`：无内存副作用、无递归副作用、无 region。
3. 按 IR 逆序处理每个 producer：
   - 收集所有消费块锚点（`getAncestorInBlock`），拓扑排序。
   - 第一个消费块：若 block_id 有效且与 producer 不同，则 `moveBefore` 第一个消费者并改 block_id；
     若被控制流打断（block_id=-1）则原样保留。
   - 后续每个新出现 block_id：`clone` 一份 producer 放到该块，改 block_id，并把对应消费者的使用替换为克隆副本。
   - 已见 block_id：复用已有副本（`replaceUsesWithIf`）。

**特点**: 这是少数会**克隆 IR op**（而非仅改 block_id 属性）的 pass；目标是在 SplitDataflow 后每个核
各自拥有 i1 mask 副本。

---

### 3.9 FixpipeOptPass (`fixpipe-opt`)

**文件**: `FixpipeOptPass.cpp:69`

**目的**: 识别 `matmul → cast/mul → store to GM` 的 fixpipe 模式，把整组 op 并入 matmul 块并将
core_type 改为 CUBE，使 matmul 结果可直接经 fixpipe 写回 GM，省去 UB 中转。

**关键逻辑**:
1. 从 `linalg.matmul` 出发，`getFirstResultAfterLoop` 沿单链 scf.for 穿出循环得到外层可见结果 op
   （要求 iter arg 单用、单链 yield），以此 op 的 block_id 为目标。
2. 两种模式：
   - **Cast 模式** `isFixpipeCastPattern`：`matmul → arith.truncf → extract_slice →
     materialize_in_destination(subview(GM))`，或 `matmul → truncf → matmul`（trunc 结果作为下游 matmul 输入）。
   - **Mul(量化) 模式** `isFixpipeMulPattern`：`matmul → arith.mulf/muli(scalar, 带 kInlinableQuantScale
     标注) → [extract_slice] → store to GM`。`isValidMul` 校验标量来源（含 `linalg.fill` 常量）。
3. `applyFixpipeOpt`：对匹配 op（含 SCF 内层）环检测后 `updateBlockId` 到 matmul 块并 `setAttr core_type=CUBE`。
4. 合并前用 `ScalarClosure` 收集标量依赖并 `cloneScalarOpsForCrossBlockUses` 克隆，避免跨块环。

**安全检查**: matmul 结果单用；环检测；标量克隆断环。

---

### 3.10 SplitIfByBlockIdPass (`split-if-by-block-id`)

**文件**: `SplitIfByBlockId/SplitIfByBlockId.cpp:1685`

**目的**: 当一个 `scf.if` 内部同时包含 CUBE 和 VECTOR 两组 op，且二者之间存在**双向跨核数据依赖**
（CUBE 喂 VECTOR 且 VECTOR 喂 CUBE）时，把该 if **拆分成一串 scf.if 链**，使每个新 if 只含单一 block_id。

**关键逻辑**:
1. `walkMainLoop`：只在 `CUBE_AND_VECTOR` 的主循环（scf.for/while）内处理；可跳过黑名单 kernel。
2. `groupOpsInBlock`：把 if 的 then/else block 内 op 按 block_id 分组（`BlockGroup`），嵌套 if 归到最近前组，
   环境无 block_id 的 op 归到当前组。
3. `hasBidirectionalCrossCoreDeps`：判定是否需要拆分——必须同时有 CUBE↔VECTOR 双向数据流（仅单向则靠排序即可，不拆）。
4. `preprocessScalarDependencies`：用 `ScalarClosure` 把跨组标量依赖克隆进各自组。
5. `analyzeDependencies` + `planYield`：扫描跨组 SSA 依赖（`scanRegion`），规划 yield 扩展：
   - **Case A**（无原始 yield）：非末组各自 yield 交叉值；末组 void。
   - **Case B**（有原始 yield）：每个非末组 yield 其产出的交叉值 + 原始 yield 槽；末组（last-if）承载
     全部原始结果类型，else 侧吸收另一侧的 op（Scene 3/4）或放占位值。
6. `materializeCandidate`：逐组创建新 scf.if（按组签名，无统一槽），`rewireAndMoveOps` 把组 op 搬入 then block
   并替换交叉值引用；`buildElseYieldForGroup`/`buildElseYieldForLastIf` 构造 else 占位（placeholder：tensor 用
   empty+fill zero，memref 用 alloca/alloc+reinterpret_cast，标量用 const 0）。
7. `postProcess`：复制原 if 属性，设置 core_type（按 then yield 值的核类型拼接）、block_id、`kSplittedIf` tag。
8. `rearrangeIfOp`：把拆分后的 if 移到其最后依赖之后，保证 dominance。

**安全检查**: 仅双向跨核依赖才拆；失败时 `FallbackHelper.restore()` 回滚。

---

### 3.11 MoveLoadIntoUserPass (`move-load-into-user`)

**文件**: `MoveLoadIntoUserPass.cpp:50`

**目的**: 把 `alloc → memref.copy(GM→UB) → to_tensor` 这一 load 模式（及其同块依赖）搬到第一个
VECTOR 计算用户所在块，使数据加载与首个消费者同块，减少跨块 UB 滞留。

**关键逻辑**:
1. `matchLoadPattern`：copy 源来自 GM（`collectViewOpsAndCheckGlobalMemory`）；目的来自 alloc；
   alloc 有对应 `to_tensor`。
2. `checkAllOpsInSameBlock`：alloc/copy/to_tensor 三者同 block_id。
3. `getFirstUser`：找 to_tensor 的第一个 IR 序 VECTOR 用户（穿透 `collapse_shape` 单链），其 block_id 为目标。
4. `collectAllDependencies`：递归收集与 load 同块的内存/SSA 依赖 op。
5. `cloneScalarOpsForCrossBlockUses` 克隆标量断环；`willCreateCycle` 后 `updateBlockIdWithInner` 搬入目标块。

**安全检查**: 首用户须 VECTOR_ONLY 且有 block_id；环检测；标量克隆。

---

### 3.12 UnifyStoreBlockPass (`unify-store-block`)

**文件**: `UnifyStoreBlockPass.cpp:298`

**目的**: 把 store 语义 op（`materialize_in_destination` / `hivm.store`）及其数据/目标 view 链和标量依赖
并入其 **VECTOR 生产者块**，使写回与最后计算同块。

**关键逻辑**:
1. 只处理 `VECTOR_ONLY` 的 store op 且有 block_id。
2. `traceProducerOp`：从 store.source 沿 view 链穿透，跳过标量，找到第一个同核类型的"真正计算" producer op。
3. `collectOpsToUnify`：= store + 数据 view 链 + 目标 memref view 链 + 递归标量依赖（`collectScalarDeps`，
   只收同 IR block 内的标量，避免搬动共享常量）。
4. `cloneScalarOpsForCrossBlockUses` 克隆标量断环（producer 在 matchedOps[0]）。
5. `applyStoreUnify`：环检测后统一到 producer 的 block_id。

**安全检查**: producer 须有 block_id 且与 store 同 block；环检测；标量克隆断环。

---

### 3.13 ExpSubfPatternPass (`exp-subf-pattern`)

**文件**: `ExpSubfPatternPass.cpp:149`

**目的**: 识别 `subf → exp`（及可选的 `f16→f32 extf`）模式，把 exp 和 extf 的 block_id 统一到 subf，
使该 softmax 子模式整体同块。

**关键逻辑**:
1. `matchExpFromSubf`：subf 单用、用户是 `math.exp`、同 IR block。
2. 若 exp 与 subf 不同块则 `updateBlockId(exp, subfBlockId)`。
3. `matchExtfFromSubf`：subf 两个操作数都来自 `arith.extf`（f16→f32），extf 单用且用户即 subf。
4. 满足则把两个 extf 也并入 subf 块。

**特点**: 不做环检测（仅做相邻同 block 的局部属性统一，风险低）。

---

### 3.14 MergeComputeBlockPass (`merge-compute-block`)

**文件**: `MergeComputeBlockPass.cpp:479`

**目的**: 在 CUBE 块之间/周围，把**相邻的 VECTOR 计算块**合并，减少跨块传输。仅对特定 kernel 启用
（如 `flex_attention_backward_dkdv_kernel` 等），且要求 intraBufCount≥3、interCoreBufCount≥2。

**关键逻辑**:
1. `groupAndBuildGraph`：按 block_id 分组，用 `DependencyHelper`（SSA + 内存依赖）建块级 DAG（succs/preds/边源 op）。
2. `collectVectorCandidates`：VECTOR_ONLY、有 tensor 结果、且与某 CUBE 块相邻（前或后）。
3. `findAdjacentVectorPair`：在候选中找一对相邻的 predV→succV。
4. `tryDirectMerge`：直接把 succV 并入 predV（环检测通过则成，`markSubBlock` 记录原 id）。
5. `tryCrossCubeCloneMerge`：若直接合产生环，则找 succV 的 CUBE 前驱 Cube、Cube 的 CUBE 前驱 CubePre，
   若 Cube 经 `to_tensor` 依赖 CubePre，则把 CubePre 中相关 op **克隆进 Cube** 断环，再合并。

**安全检查**: 仅特定 kernel；buffer 数门槛；环检测；跨 Cube 克隆断环；设 `kMergeComputeBlockApplied` 标记供
后续 ReorderOpsByBlockIdPass 参考。

---

### 3.15 MergeCubeBlockPass (`merge-cube-block`)

**文件**: `MergeCubeBlock.cpp`（实现类 `MergeCubeBlockPass`，声明于 `MergeCubeBlockPass.h`）

**目的**: 把**可合并的相邻 CUBE 块**合并成一个，减少 Cube 流水线段数。有 kernel 黑名单
（`_attn_bwd`、`kernel_sdpa_bwd_q`、`_sdpa_infer_kernel` 等）。

**关键逻辑**:
1. `walkMainLoop` 收集主循环 block。
2. `processBlock`：`BlockDependencyGraph` 建块依赖图（节点含 isCube/depth/ops，边为跨块依赖）。
3. `performMerging`：双层循环扫描所有 CUBE 块对 (target, source)：
   - `canMergeBlocks`：二者皆 Cube；`hasCommonInputOrOutput`（有共同异类型前驱或后继）；`checkNoCycle`
     （`willCreateCycle`）；`hasSameDepth`（深度相同且其后继最大深度相同）。
   - `mergeBlocks`：把 source 全部 op `updateBlockId` 到 target，`rebuildAfterMerge` 更新图。
4. 迭代直到无更多合并；无合并则设 `kMergeComputeBlockApplied=false`。

**安全检查**: 共同输入输出；环检测；同深度；kernel 黑名单。

---

### 3.16 MergeInputInitSharedCubeBlockPass (`merge-input-init-shared-cube-block`)

**文件**: `MergeInputInitSharedCubeBlockPass.cpp:86`

**目的**: 当一个 matmul（consumer）的 **init（输出初始化）和 input 都来自同一个上游 matmul（producer）**
（input 可经中间 op 穿透）时，把 consumer 块并入 producer 块，使两个 Cube 块共享数据时合并。

**关键逻辑**:
1. `getSharedInputInitProducer`：consumer 的 dps init 由某 matmul 定义，且某 dps input 经
   `getSourceThroughCIntermediateOps` 穿透中间 op 后也指向同一 matmul。
2. producer/consumer block_id 不同时，`tryMergeBlocks` 把 consumer 全块并入 producer。
3. 合并失败（环）则设 `ERRCODE_FAILED` fallback。

**安全检查**: `willCreateCycle`；失败即 fallback。

---

### 3.17 RelocateMemrefDeclPass (`relocate-memref-decl`)

**文件**: `RelocateMemrefDeclPass.cpp:257`

**目的**: 把**使用跨越了 sync 屏障**的 memref 声明（及其前向闭包）**搬迁到消费者所在 block 的屏障另一侧**，
使声明与其使用位于同一同步区，为后续 SplitDataflow 拆同步做准备。

**关键逻辑**:
1. 无 sync op 则直接返回。
2. 遍历每个 op 的 memref 型操作数，找到其定义 producer；用 `SyncWall.hasSyncBetween` 判断 producer 与 consumer
   之间是否有 sync。
3. `buildForwardClosure`：构建前向闭包——producer 及其不在目标 block 的使用（递归含使用的使用），
   以及这些 op 的 memref 操作数定义 op（向上回溯），直到不动点；遇终止 op 则放弃。
4. 要求最早跨 sync 使用（anchor）在 sync 之后；`allOperandsDominate` 校验闭包操作数在插入点之前 dominate。
5. 把闭包成员按原序 `moveBefore(anchor)` 并 `updateBlockId` 到消费者块、继承消费者 core_type。

**安全检查**: 必须有 sync 在 producer 与 consumer 之间；anchor 在 sync 之后；操作数 dominance；
不跨终止 op。

---

## 4. 设计模式小结

纵观全部 pass，可归纳出以下共性设计模式：

1. **"匹配模式 → 环检测 → 改 block_id"三段式**：几乎所有合并类 pass 都遵循"先只读收集候选模式，再
   `willCreateCycle` 安全检查，最后 `updateBlockId` 改写"的流程，且把改写放在最后以避免边改边收集的不一致。
2. **标量克隆断环**：`ScalarClosure` / `cloneScalarOpsForCrossBlockUses` 在合并前克隆标量依赖，是打破
   跨块环的标准手段（见 UnifyStoreBlock、FixpipeOpt、MoveLoadIntoUser、SplitIfByBlockId）。
3. **同步 op 独占 block_id**：任何合并都跳过 sync op（`isSyncOp`），保证栅栏语义。
4. **核类型不混用**：合并目标块与被合并 op 须同核类型，避免出现 `CUBE_AND_VECTOR` 的混合 block_id。
5. **Fallback 兜底**：`hasFallbackAttr` 早退、`setFallbackAttr(ERRCODE_FAILED)` 兜底贯穿全程，保证优化失败
   时安全回退。
6. **两次 MergeSmallBlock**：第二次才合并 `sitofp-subf-mulf` 模式，体现"先做通用合并、再做模式特化合并"
   的分阶段策略。
7. **属性标记驱动下游**：`kMergeComputeBlockApplied`、`kMergeSmallBlockFirstRunDone`、`kSplittedIf` 等
   属性用于 pass 间传递状态，使后续 pass 据此决定行为。
