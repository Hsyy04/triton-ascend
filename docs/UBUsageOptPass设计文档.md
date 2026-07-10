# UBUsageOptPass 设计文档

## 1. 背景与动机

### 1.1 问题背景

在异构计算架构中，Unified Buffer (UB) 是计算单元内部的快速存储区域，用于存储中间计算结果和数据。UB 具有以下特性：

- **容量限制**：UB 的大小有限，无法容纳所有中间数据
- **访问速度**：UB 访问速度远高于外部内存，但小于寄存器
- **生命周期管理**：需要合理管理数据的生命周期，避免不必要的占用

在 Triton 编译器中，Compute Block 是一组操作的集合，它们会被调度到同一个计算核心上执行。当 Compute Block 内的操作产生的中间数据过多时，会导致 UB 使用量超过限制，从而引发性能下降或编译失败。

### 1.2 优化目标

UBUsageOptPass 的核心目标是：

1. **减少 UB 使用峰值**：通过重新划分 Compute Block，将部分操作移动到其他 Block，减少 UB 的峰值使用量
2. **保持正确性**：确保操作重排不会引入循环依赖或破坏原有语义
3. **保持性能**：优化的同时不引入过多的数据移动开销

### 1.3 适用场景

本优化主要针对以下场景：

- **VECTOR_ONLY 类型的 Compute Block**：这类 Block 主要执行向量运算，UB 使用量较大
- **存在跨 Block 数据依赖**：某些操作的数据被多个后续 Block 使用，导致 UB 无法及时释放
- **Partition 机会**：存在可以将 Block 切分的点，使得部分数据可以被提前释放

### 1.4 适用场景图示

#### 场景 1：UB 容量限制

```
硬件架构视图：

┌─────────────────────────────────────────────┐
│         Compute Core (Vector Unit)          │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │      Unified Buffer (UB)            │    │
│  │      容量限制：例如 64KB             │    │
│  │                                     │    │
│  │  ┌──────────┐  ┌──────────┐       │    │
│  │  │  Tensor A │  │  Tensor B │       │    │
│  │  │  20KB     │  │  30KB     │       │    │
│  │  └──────────┘  └──────────┘       │    │
│  │                                     │    │
│  │  ┌──────────┐  ┌──────────┐       │    │
│  │  │  Tensor C │  │  Tensor D │       │    │
│  │  │  25KB     │  │  15KB     │       │    │
│  │  └──────────┘  └──────────┘       │    │
│  │                                     │    │
│  │  当前使用：90KB > 64KB ❌           │    │
│  │  溢出到外部内存，性能下降           │    │
│  └────────────────────────────────────┘    │
│                                              │
│  ┌──────────────────┐                      │
│  │  Compute Units    │                      │
│  └──────────────────┘                      │
└─────────────────────────────────────────────┘

问题：单个 Compute Block 内的中间数据总量超过 UB 容量
结果：数据溢出到外部内存，访问延迟增加 10-100 倍
```

#### 场景 2：跨 Block 数据依赖

```
依赖关系视图（优化前）：

Compute Block 0 (VECTOR_ONLY)
┌─────────────────────────────────────────────┐
│                                              │
│  op1: %t0 = tensor.generate (400 bytes)     │
│           │                                  │
│           v                                  │
│  op2: %t1 = process(%t0) (320 bytes)        │
│           │                                  │
│      ┌────┴────┐                            │
│      │         │                             │
│      v         v                             │
│  op3: %t2   op4: %t3                         │
│  (240 bytes) (240 bytes)                    │
│                                              │
│  UB 峰值使用：400 + 320 + 240 = 960 bytes   │
│                                              │
└─────────────────────────────────────────────┘
      │         │
      │         │  跨 Block 数据依赖
      │         │  %t2 和 %t3 必须保留到
      │         │  Block 1 和 Block 2 使用完
      v         v
 ┌─────────┐ ┌─────────┐
 │Block 1  │ │Block 2  │
 │%u0 =    │ │%v0 =    │
 │use(%t2) │ │use(%t3) │
 └─────────┘ └─────────┘

问题：
1. Block 0 内的 %t2 和 %t3 占用 UB 空间
2. 即使 Block 1 和 Block 2 可以独立执行，
   %t2 和 %t3 也无法提前释放
3. UB 使用峰值高
```

#### 场景 3：优化机会识别

```
数据流分析视图：

时间线：
t0: op1 执行，产生 %t0 (400 bytes)
t1: op2 执行，产生 %t1 (320 bytes)，%t0 被释放
t2: op3 执行，产生 %t2 (240 bytes)
t3: op4 执行，产生 %t3 (240 bytes)
t4: Block 1 使用 %t2
t5: Block 2 使用 %t3

关键观察：
- %t2 只被 Block 1 使用
- %t3 只被 Block 2 使用
- %t1 被 %t2 和 %t3 共同依赖

优化机会：
如果将 op3 (%t2) 移动到 Block 1：
  Block 0 的 UB 峰值：400 (t0) + 320 (t1) = 720 bytes
  Block 1 的 UB 峰值：240 (t2) bytes
  
如果将 op4 (%t3) 移动到 Block 2：
  Block 0 的 UB 峰值：400 (t0) + 320 (t1) = 720 bytes
  Block 2 的 UB 峰值：240 (t3) bytes

优化收益：
  原 UB 峰值：960 bytes
  新 UB 峰值：720 bytes
  减少：240 bytes (25%)
```

#### 场景 4：优化后的依赖关系

```
优化后的视图：

Compute Block 0 (VECTOR_ONLY)
┌─────────────────────────────────────────────┐
│                                              │
│  op1: %t0 = tensor.generate (400 bytes)     │
│           │                                  │
│           v                                  │
│  op2: %t1 = process(%t0) (320 bytes)        │
│                                              │
│  UB 峰值使用：400 + 320 = 720 bytes ✓       │
│                                              │
└─────────────────────────────────────────────┘
           │
           │  %t1 被多个 Block 使用
           │
    ┌──────┴──────┐
    │             │
    v             v
┌──────────┐  ┌──────────┐
│ Block 1  │  │ Block 2  │
│          │  │          │
│ op3:     │  │ op4:     │
│ %t2 =    │  │ %t3 =    │
│ proc(%t1)│  │ proc(%t1)│
│ (240B)   │  │ (240B)   │
│          │  │          │
│ op5:     │  │ op6:     │
│ use(%t2) │  │ use(%t3) │
│          │  │          │
│ UB: 240B │  │ UB: 240B │
└──────────┘  └──────────┘

关键变化：
1. op3 和 op4 从 Block 0 移出
2. Block 0 的 UB 使用峰值从 960 bytes 降至 720 bytes
3. Block 1 和 Block 2 各自承担 240 bytes 的 UB 使用
4. 整体 UB 峰值降低，避免溢出
```

#### 场景 5：不适用优化的情况

```
情况 A：CUBE_ONLY Block（不适用）

Compute Block 0 (CUBE_ONLY)
┌─────────────────────────────────────────────┐
│                                              │
│  op1 -> op2 -> op3 -> op4                   │
│                                              │
│  特点：                                      │
│  - 所有操作在 Cube 核心上执行                │
│  - 操作之间紧密耦合，不能切分                │
│  - Pass 会将这些操作合并为一个节点            │
│                                              │
│  结果：不进行优化 ✓                          │
└─────────────────────────────────────────────┘


情况 B：Memref 依赖（不适用）

Block 0:
  op1: %m0 = memref.alloc (memref<100xf32>)
           │
           v
  op2: use(%m0)
           │
           v
  Block 1: use(%m0)

特点：
- Memref 表示内存引用，涉及地址计算
- 边权重设置为 MAX_EDGE_SIZE (2^30)
- 不在此处切分，避免破坏内存语义

结果：不进行优化 ✓


情况 C：循环依赖风险（不适用）

原始依赖：
Block 0: op1 -> op2
              │
              v
Block 1: op3 (依赖 op2)
              │
              v
Block 2: op4 (依赖 op3，也依赖 Block 0 的其他 op)

如果将 op2 移动到 Block 1：
Block 0: op1 -> Block 1: op2 -> Block 2: op4
                      ↓
                  Block 1: op3

风险：
- op3 依赖 Block 0 的 op1
- op4 依赖 op3 和 Block 0
- 可能形成循环：Block 0 -> Block 1 -> Block 2 -> Block 0

结果：检测到循环依赖，拒绝优化 ✓
```

#### 适用场景总结

| 场景 | 特征 | 是否优化 | 原因 |
|------|------|---------|------|
| VECTOR_ONLY Block，跨 Block 数据依赖 | 多个 Block 使用同一数据源 | ✅ 优化 | 可以切分，减少峰值 UB |
| VECTOR_ONLY Block，单 Block 独占数据 | 数据只被一个 Block 使用 | ✅ 优化 | 可以移动操作到使用 Block |
| CUBE_ONLY Block | 紧密耦合的操作 | ❌ 不优化 | Cube 核心语义限制 |
| Memref 依赖 | 内存引用类型 | ❌ 不优化 | 避免破坏内存语义 |
| 循环依赖风险 | 切分后可能形成环 | ❌ 不优化 | 正确性优先 |

---

## 2. 核心概念

### 2.1 Compute Block

**定义**：Compute Block 是一组操作的集合，这些操作会被调度到同一个计算核心上执行。

**属性**：
- **BlockId**：每个 Compute Block 的唯一标识符
- **CoreType**：计算核心类型（见 2.2）
- **包含的操作**：一个 Block 可以包含多个 MLIR Operation

**关键约束**：
- 同一个 Compute Block 内的操作必须能够顺序执行
- Block 之间的依赖关系必须是有向无环图（DAG）

### 2.2 CoreType

**定义**：CoreType 标识操作应该调度到哪类计算核心上执行。

**类型**：
- **CUBE_ONLY**：操作调度到 Cube 核心执行
  - 特点：可以合并多个操作到同一个 Block（通过 `cubeBlockId2nodeId` 映射）
  - 优化策略：在构建依赖图时，同一个 CUBE_ONLY Block 内的操作会被合并为一个节点
  
- **VECTOR_ONLY**：操作调度到 Vector 核心执行
  - 特点：是 UBUsageOptPass 的主要优化目标
  - 优化策略：可以通过切分来减少 UB 使用量

- **MIXED**：操作需要在 Cube 和 Vector 核心上协同执行
  - 特点：不参与本优化

### 2.3 UB 使用量计算

**边权重（Edge Weight）**：在依赖图中，边的权重表示数据在 UB 中占用的字节数。

计算规则：

```cpp
// Tensor 类型
if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
    return numElements * elementSize;
}

// Memref 类型 - 设置为最大值，表示不应在此处切分
if (auto memRefType = dyn_cast<MemRefType>(type)) {
    return MAX_EDGE_SIZE;
}

// Vector 类型
if (auto vectorType = dyn_cast<VectorType>(type)) {
    return numElements * elementSize;
}

// Index 类型 - 不占用 UB
if (auto idxTy = dyn_cast<IndexType>(type)) {
    return 0;
}
```

**关键点**：
- **Tensor 和 Vector**：直接计算数据大小
- **Memref**：设置为最大值（2^30），防止在 Memref 依赖处切分 Block
- **Index**：不占用 UB，权重为 0
- **循环依赖参数**：如果数据来自循环迭代的 BlockArgument（`fromArgEdge = true`），权重加倍，表示该数据在多次迭代中都会占用 UB

### 2.4 MemoryDependenceGraph

**定义**：MemoryDependenceGraph 是一个内存依赖图，记录操作之间的内存访问依赖关系。

**作用**：
- 捕获 SSA 依赖图无法表达的内存依赖（如通过指针的间接依赖）
- 确保优化后的 Block 划分不会破坏内存依赖的正确性

**使用方式**：
```cpp
for (auto memDef : memGraph.getExecBefore(op)) {
    // memDef 必须在 op 之前执行
}
```

---

## 3. 算法概述

### 3.1 核心思想

UBUsageOptPass 采用**依赖图切分**策略：

1. **构建依赖图**：将操作和依赖关系建模为有向图
   - 节点：操作或合并的操作组（CUBE_ONLY Block）
   - 边：数据依赖或内存依赖
   - 边权重：数据在 UB 中的占用大小

2. **识别优化候选**：找到可能通过切分减少 UB 使用量的 Compute Block
   - 只考虑 VECTOR_ONLY 类型的 Block
   - 寻找"活跃的出口节点"（active end node）

3. **寻找最优切分点**：在候选 Block 的依赖链上寻找最佳切分位置
   - 最小化切分后的 UB 使用量
   - 避免创建循环依赖

4. **应用变换**：将选定的操作移动到新的 Block 中

### 3.2 关键策略

#### 3.2.1 活跃出口节点（Active End Node）

**定义**：某个操作 A 的"活跃出口节点" B，是指：
- B 是 A 的直接或间接后继节点
- B 不在 A 所在的 Compute Block 中
- B 的所有依赖都来自 A 所在的 Block 或不依赖任何其他 Block

**作用**：识别哪些数据可以从当前 Block 中"释放"出来，移动到其他 Block。

**判断逻辑**：
```cpp
bool isActiveEndNode(int srcNode, int endNode, ...) {
    // 1. CoreType 必须相同
    if (nodeCoreType[endNode] != nodeCoreType[srcNode]) {
        return false;
    }
    
    // 2. endNode 必须在某个 Compute Block 中
    if (nodeBlockId[endNode] == -1) {
        return false;
    }
    
    // 3. 不能在同一个 Block
    if (nodeBlockId[srcNode] == nodeBlockId[endNode]) {
        return false;
    }
    
    // 4. endNode 的所有依赖必须来自 srcNode 的 Block 或无依赖
    auto dependNodes = findDependency(endNode, srcNode, ...);
    for (int node : dependNodes) {
        if (nodeBlockId[node] != nodeBlockId[endNode] && 
            nodeBlockId[node] != nodeBlockId[srcNode]) {
            return false;
        }
    }
    
    return true;
}
```

#### 3.2.2 依赖链切分

**问题**：在找到活跃出口节点后，如何决定切分哪个操作？

**策略**：沿着依赖链向前查找，寻找 UB 使用量最小的切分点。

**示例**：
```
原始链路：
Block A: op1 -> op2 -> op3 -> op4
            |      |      |
            v      v      v
Block B:  use1   use2   use3

UB 使用量：
- use1 依赖 op1 的数据 (100 bytes)
- use2 依赖 op2 的数据 (200 bytes)
- use3 依赖 op3 的数据 (150 bytes)

如果不切分，Block A 的 UB 峰值 = 100 + 200 + 150 = 450 bytes

切分策略：
- 如果在 op2 之后切分，将 op3, op4 移动到 Block A'
- Block A 的 UB 峰值 = 100 + 200 = 300 bytes（减少 150 bytes）
- 但 op3 的数据仍需传递给 Block A'
```

#### 3.2.3 循环依赖检测

**问题**：移动操作后可能引入循环依赖。

**示例**：
```
原始依赖：
Block A: op1 -> op2 -> op3
         |             |
         v             v
Block B: op4  ------> op5

如果将 op3 移动到 Block A：
Block A: op1 -> op2
         |      |
         v      v
Block A': op3
         |
         v
Block B: op4 -> op5

但如果 op5 也依赖 op4，而 op4 依赖 op3：
Block A': op3
         |
         v
Block B: op4 -> op5 -> op3 (循环！)
```

**解决方案**：使用 DFS 检测循环依赖，在应用变换前验证。

---

## 4. 数据结构设计

### 4.1 依赖图表示

**节点（Node）**：
- `nodeId`：节点唯一标识符
- `op`：对应的 MLIR Operation
- `blockId`：所属的 Compute Block ID
- `coreType`：CoreType（CUBE_ONLY, VECTOR_ONLY, MIXED）

**边（Edge）**：
- `edgeId`：边的唯一标识符
- `src`：起始节点
- `dst`：终止节点
- `size`：权重（UB 占用字节数）

**数据结构**：
```cpp
// 节点映射
DenseMap<Operation *, int> op2nodeId;       // Operation -> NodeId
DenseMap<int, Operation *> nodeId2op;       // NodeId -> Operation

// 边信息
SmallVector<SmallVector<int>> linkOut;      // linkOut[nodeId] = [出边的 edgeId]
SmallVector<SmallVector<int>> linkIn;       // linkIn[nodeId] = [入边的 edgeId]
SmallVector<int> linkSize;                   // linkSize[edgeId] = 权重
SmallVector<int> linkStart;                  // linkStart[edgeId] = 起始节点
SmallVector<int> linkEnd;                    // linkEnd[edgeId] = 终止节点

// 节点信息
SmallVector<int> nodeBlockId;                // nodeBlockId[nodeId] = Compute Block ID
SmallVector<int> nodeCoreType;               // nodeCoreType[nodeId] = CoreType
SmallVector<int> nodeArgs;                   // nodeArgs[nodeId] = 对应的 BlockArgument 编号
```

### 4.2 图构建策略

**关键设计**：
1. **CUBE_ONLY Block 的合并**：同一个 CUBE_ONLY Block 内的所有操作合并为一个节点
   ```cpp
   if (canShrink) {
       auto it = cubeBlockId2nodeId.find(blockId);
       if (it != cubeBlockId2nodeId.end()) {
           op2nodeId[op] = it->second;
           return it->second;
       }
   }
   ```

2. **自环边的处理**：避免创建不必要的自环边
   - 同一个操作内部的数据依赖不创建边
   - 同一个 CUBE_ONLY Block 内的操作不创建边

3. **循环迭代参数的处理**：识别来自循环迭代的 BlockArgument，并加倍权重
   ```cpp
   if (fromArgEdge) {
       edgeSize *= 2;
   }
   ```

---

## 5. 算法流程

### 5.1 整体流程

```
UBUsageOptPass::runOnOperation()
    |
    v
遍历 Module 中的所有 Block
    |
    v
对每个 Block 执行 UBUsageOptimization()
    |
    +---> 1. buildUBUsageGraph()      // 构建依赖图
    +---> 2. collectNeedUbOpts()      // 识别优化候选
    +---> 3. collectRecordChange()   // 寻找最优切分点
    +---> 4. applyRecordChange()      // 应用变换
```

### 5.2 详细步骤

#### Step 1: buildUBUsageGraph

**输入**：Block, MemoryDependenceGraph, ComputeBlockIdManager

**输出**：依赖图的完整表示

**流程**：
```
1. 初始化数据结构
2. 遍历 Block 中的每个 Operation
   a. 获取或创建节点 ID（getOrCreateNodeId）
      - 如果是 CUBE_ONLY Block，检查是否已存在合并节点
      - 否则创建新节点
   b. 处理 BlockArgument
      - 识别来自循环迭代的参数
      - 标记节点对应的 BlockArgument 编号
   c. 处理 SSA 依赖
      - 遍历每个操作的所有操作数
      - 找到定义该操作数的操作
      - 创建边，计算权重
   d. 处理内存依赖
      - 遍历 MemoryDependenceGraph
      - 创建权重为 0 的依赖边
```

**关键点**：
- **祖先查找**：`getAncestorInBlock(op, block)` 确保只处理当前 Block 内的操作
- **权重加倍**：对于来自循环迭代的 BlockArgument，权重加倍

#### Step 2: collectNeedUbOpts

**输入**：依赖图

**输出**：每个 Compute Block 的优化候选节点列表

**流程**：
```
1. 遍历所有节点
   a. 只考虑 VECTOR_ONLY 类型的节点
   b. 检查是否存在"活跃出口节点"
   c. 如果存在，将该节点加入对应 Block 的候选列表
```

**筛选条件**：
- CoreType 必须是 VECTOR_ONLY
- 必须有至少一个活跃出口节点

#### Step 3: collectRecordChange

**输入**：依赖图、候选节点列表

**输出**：节点 -> 新 Block ID 的映射

**流程**：
```
对每个候选 Block：
  对每个候选节点 optNode：
    1. 找到所有活跃出口节点（activateSet）
    
    2. 对每个活跃出口节点 activateNode：
      a. 计算原始 UB 使用量（originUBSize）
         = sum(所有入边的权重)
      
      b. 沿着依赖链向前查找（findUniqueDependentNode）
         - 构建依赖链：activateNode -> node1 -> node2 -> ...
         - 直到遇到多分支节点或跨 Block 依赖
      
      c. 遍历依赖链，寻找最优切分点
         for i in chain:
           计算切分后的 UB 使用量 = sum(chain[i] 的所有出边权重)
           if 切分后 UB < 当前最小 UB:
             更新最优切分点
      
      d. 如果找到最优切分点：
         - 记录需要移动的节点（chain[0..bestCutPointIdx]）
         - 记录这些节点的依赖节点
         - 更新 recordChange
```

**核心逻辑**：
- **依赖链查找**：`findUniqueDependentNode` 查找单链依赖
- **UB 计算**：`sumIncomingLinkSize` 计算入边权重总和
- **切分点选择**：最小化 UB 使用量

#### Step 4: applyRecordChange

**输入**：节点 -> 新 Block ID 的映射

**输出**：更新后的 ComputeBlockIdManager

**流程**：
```
1. 按 Block ID 分组需要移动的节点
   blockWilladd[blockId] = [需要移动到该 Block 的节点列表]

2. 对每个目标 Block：
   a. 收集所有需要移动的操作（willaddOps）
   
   b. 检测是否会引入循环依赖（willCreateCycle）
      - 临时将操作移动到目标 Block
      - 使用 DFS 从新 Block 的所有操作出发，检查是否能回到 Block 内的操作
      - 如果检测到循环，回滚并跳过该 Block
      - 否则保留变换
   
   c. 更新 ComputeBlockIdManager
      - processOpsInblock：处理嵌套操作的 Block ID
      - updateBlockId：更新操作的 Block ID
```

**循环依赖检测算法**：
```cpp
bool willCreateCycle(willaddOps, block, memGraph, targetBlockId, bm) {
    1. 将 willaddOps 临时加入 targetBlockId 的 okSet
    2. 对 okSet 中的每个操作 op：
       a. 收集所有用户（SSA 用户 + 内存依赖用户）
       b. 对每个用户 user：
          - 如果 user 在 okSet 中，跳过
          - 如果 user 的 Block ID == -1（未分配 Block）：
            DFS 检查是否能回到 okSet
          - 如果 user 有 Block ID：
            从该 Block 的所有操作出发，DFS 检查是否能回到 okSet
    3. 如果检测到循环，回滚并返回 true
    4. 否则返回 false
}
```

---

## 6. 关键优化策略

### 6.1 CUBE_ONLY Block 的合并

**问题**：CUBE_ONLY Block 内的操作通常需要作为一个整体执行，不应被切分。

**解决方案**：
- 在构建依赖图时，将同一个 CUBE_ONLY Block 内的所有操作合并为一个节点
- 使用 `cubeBlockId2nodeId` 映射记录合并关系

**优势**：
- 减少图的规模
- 避免在 CUBE_ONLY Block 内部进行不必要的切分

### 6.2 Memref 依赖的特殊处理

**问题**：Memref 表示内存引用，切分可能导致内存访问冲突。

**解决方案**：
- 将 Memref 依赖的边权重设置为最大值（MAX_EDGE_SIZE = 2^30）
- 在计算 UB 使用量时，如果遇到 MAX_EDGE_SIZE，直接返回 MAX_EDGE_SIZE
- 这会导致切分算法避开这些位置

### 6.3 循环迭代参数的权重加倍

**问题**：来自循环迭代的 BlockArgument 在每次迭代中都会占用 UB。

**示例**：
```mlir
scf.for %i = 0 to 10 {
  %value = ... // 定义在循环体内
  // %value 在每次迭代中都占用 UB
}
```

**解决方案**：
- 识别来自循环迭代的 BlockArgument（`fromArgEdge = true`）
- 将对应的边权重加倍，表示多次迭代的 UB 占用

**识别逻辑**：
```cpp
if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
    if (blockArg.getOwner() == block && terminator) {
        // 计算 BlockArgument 对应的 yield operand
        int offset = numArgs - numYieldOperands;
        int argIdx = blockArg.getArgNumber() - offset;
        
        if (argIdx >= 0 && argIdx < numYieldOperands) {
            Value yielded = terminator->getOperand(argIdx);
            if (Operation *yieldDefOp = yielded.getDefiningOp()) {
                srcInBlock = getAncestorInBlock(yieldDefOp, block);
                fromArgEdge = true;  // 标记为来自循环迭代
            }
        }
    }
}
```

### 6.4 避免不必要的自环边

**问题**：同一个操作内部或同一个 CUBE_ONLY Block 内的依赖会形成自环边。

**解决方案**：
- 在创建边时，检查源节点和目标节点是否相同
- 如果相同，跳过边的创建

**检查逻辑**：
```cpp
// 同一个操作内部的依赖
if (srcInBlock && srcInBlock == &blockOp) {
    continue;  // 跳过自环边
}

// 同一个 CUBE_ONLY Block 内的依赖
if (coreType == cubeCoreType && srcBlockId == dstBlockId) {
    continue;  // 跳过自环边
}
```

---

## 7. 示例说明

### 7.1 场景描述

考虑以下 MLIR 代码片段：

```mlir
// Block 0 (VECTOR_ONLY)
%t0 = tensor.generate ... : tensor<100xf32>
%t1 = some.op %t0 : tensor<100xf32>
%t2 = another.op %t1 : tensor<50xf32>
%t3 = another.op %t1 : tensor<50xf32>

// Block 1 (VECTOR_ONLY)
%u0 = use.op %t2 : tensor<50xf32>

// Block 2 (VECTOR_ONLY)
%v0 = use.op %t3 : tensor<50xf32>
```

**问题**：
- Block 0 产生三个中间结果：%t0 (400 bytes), %t1 (400 bytes), %t2 (200 bytes), %t3 (200 bytes)
- %t1 被 %t2 和 %t3 使用，导致 %t1 必须保留到 %t2 和 %t3 计算完成
- Block 0 的 UB 峰值 = 400 + 400 + 200 = 1000 bytes（%t0 可能在 %t1 计算后被释放）

### 7.2 依赖图构建

**节点**：
- Node 0: %t0 (Block 0, VECTOR_ONLY)
- Node 1: %t1 (Block 0, VECTOR_ONLY)
- Node 2: %t2 (Block 0, VECTOR_ONLY)
- Node 3: %t3 (Block 0, VECTOR_ONLY)
- Node 4: %u0 (Block 1, VECTOR_ONLY)
- Node 5: %v0 (Block 2, VECTOR_ONLY)

**边**：
```
Node 0 --[400]--> Node 1
Node 1 --[200]--> Node 2
Node 1 --[200]--> Node 3
Node 2 --[200]--> Node 4
Node 3 --[200]--> Node 5
```

**依赖图可视化**：
```
Node 0 (Block 0)
  |
  v [400 bytes]
Node 1 (Block 0)
  | \
  |  \ [200 bytes]
  |   \
  |    v
  |   Node 3 (Block 0) --[200 bytes]--> Node 5 (Block 2)
  |
  | [200 bytes]
  v
Node 2 (Block 0) --[200 bytes]--> Node 4 (Block 1)
```

### 7.3 识别优化候选

**Step 1: 找到活跃出口节点**

- Node 2 的活跃出口节点：Node 4（Block 1）
  - Node 4 不在 Block 0 中 ✓
  - Node 4 的依赖只有 Node 2（来自 Block 0）✓
  - Node 4 是活跃出口节点 ✓

- Node 3 的活跃出口节点：Node 5（Block 2）
  - Node 5 不在 Block 0 中 ✓
  - Node 5 的依赖只有 Node 3（来自 Block 0）✓
  - Node 5 是活跃出口节点 ✓

**Step 2: 收集候选节点**

- Block 0 的候选节点：Node 2, Node 3

### 7.4 寻找最优切分点

**对 Node 2**：
- 活跃出口节点：Node 4
- 原始 UB 使用量（Node 4 的入边权重总和）：200 bytes
- 依赖链：Node 4 -> Node 2（单链）
- 切分点选择：
  - 在 Node 2 之后切分：Node 2 的出边权重 = 200 bytes
  - 切分后 UB 使用量 = 200 bytes（不变）

**对 Node 3**：
- 活跃出口节点：Node 5
- 原始 UB 使用量（Node 5 的入边权重总和）：200 bytes
- 依赖链：Node 5 -> Node 3（单链）
- 切分点选择：
  - 在 Node 3 之后切分：Node 3 的出边权重 = 200 bytes
  - 切分后 UB 使用量 = 200 bytes（不变）

**结论**：在这个简单示例中，切分并不能减少 UB 使用量。

### 7.5 更复杂的示例

考虑以下更复杂的场景：

```mlir
// Block 0 (VECTOR_ONLY)
%t0 = tensor.generate ... : tensor<100xf32>  // 400 bytes
%t1 = op.a %t0 : tensor<80xf32>              // 320 bytes
%t2 = op.b %t1 : tensor<60xf32>              // 240 bytes
%t3 = op.c %t1 : tensor<60xf32>              // 240 bytes

// Block 1 (VECTOR_ONLY)
%u0 = use.op %t2 : tensor<60xf32>

// Block 2 (VECTOR_ONLY)
%v0 = use.op %t3 : tensor<60xf32>
```

**问题**：
- Block 0 的 UB 峰值 = 400 (t0) + 320 (t1) + 240 (t2 或 t3) = 960 bytes

**优化思路**：
- 如果将 %t2 和 %t3 移动到新的 Block 0'：
  - Block 0 的 UB 峰值 = 400 + 320 = 720 bytes
  - Block 0' 的 UB 峰值 = 240 bytes
  - 总 UB 使用量减少 240 bytes

**切分过程**：
1. 找到活跃出口节点：
   - Node 2 (%t2) 的活跃出口节点：Node 4 (%u0, Block 1)
   - Node 3 (%t3) 的活跃出口节点：Node 5 (%v0, Block 2)

2. 寻找最优切分点：
   - 对 Node 4：原始 UB = 240 bytes
   - 依赖链：Node 4 -> Node 2 -> Node 1 -> Node 0
   - 切分选项：
     - 在 Node 2 之后切分：Node 2 的出边权重 = 240 bytes（不变）
     - 在 Node 1 之后切分：Node 1 的出边权重 = 240 + 240 = 480 bytes（增加）
   
   - 最优切分点：Node 2 之后

3. 移动操作：
   - 将 Node 2 (%t2) 移动到 Block 1
   - 将 Node 3 (%t3) 移动到 Block 2

**优化后**：
```
Block 0:
%t0 = tensor.generate ... : tensor<100xf32>  // 400 bytes
%t1 = op.a %t0 : tensor<80xf32>              // 320 bytes
UB 峰值 = 720 bytes

Block 1:
%t2 = op.b %t1 : tensor<60xf32>              // 240 bytes
%u0 = use.op %t2 : tensor<60xf32>
UB 峰值 = 240 bytes

Block 2:
%t3 = op.c %t1 : tensor<60xf32>              // 240 bytes
%v0 = use.op %t3 : tensor<60xf32>
UB 峰值 = 240 bytes
```

### 7.6 循环依赖检测示例

**危险场景**：

```mlir
// Block 0
%t0 = op.a ... : tensor<100xf32>
%t1 = op.b %t0 : tensor<80xf32>

// Block 1
%t2 = op.c %t1 : tensor<60xf32>

// Block 2
%t3 = op.d %t2 : tensor<50xf32>
```

**依赖关系**：
- Block 0: %t0 -> %t1
- Block 1: %t2 (依赖 %t1)
- Block 2: %t3 (依赖 %t2)

**错误的切分尝试**：
- 如果将 %t2 移动到 Block 0：
  - Block 0: %t0 -> %t1 -> %t2
  - Block 2: %t3 (依赖 %t2)
  
- 如果 %t3 还依赖 Block 1 的其他操作：
  - 循环依赖：Block 2 -> Block 0 -> Block 2

**检测过程**：
1. 临时将 %t2 加入 Block 0
2. 从 %t2 的用户（%t3）出发，DFS 遍历
3. 发现 %t3 在 Block 2，且 %t3 依赖 Block 1 的操作
4. 从 Block 1 的操作出发，检查是否能回到 Block 0
5. 如果能回到，说明存在循环依赖，拒绝该切分

---

## 8. 限制与权衡

### 8.1 当前限制

#### 8.1.1 只支持 scf::ForOp

**限制**：当前实现只对 `scf::ForOp` 内的 Block 进行优化。

**原因**：
- 循环体内的 UB 使用优化最为关键
- 其他控制流结构（如 `scf::IfOp`）的优化需求较少

**代码**：
```cpp
if (!isa<scf::ForOp>(block->getParentOp())) {
    return llvm::success();
}
```

#### 8.1.2 MAX_EDGE_SIZE 的处理

**限制**：当遇到 `MAX_EDGE_SIZE`（Memref 依赖）时，直接返回 `MAX_EDGE_SIZE`，不进行精确计算。

**影响**：
- 可能错过一些优化机会
- 但避免了不正确的切分

**改进方向**：
- 更精确地估计 Memref 的生命周期
- 考虑部分切分的可能性

#### 8.1.3 CUBE_ONLY Block 的整体移动

**限制**：CUBE_ONLY Block 内的所有操作会被合并为一个节点，无法部分移动。

**原因**：
- CUBE_ONLY Block 通常需要作为一个整体执行
- 部分移动可能破坏 Cube 核心的执行语义

**权衡**：
- 减少了优化空间
- 但保证了正确性

#### 8.1.4 BlockArgument 的特殊处理

**限制**：来自循环迭代的 BlockArgument 的权重加倍是一个启发式规则，可能不精确。

**改进方向**：
- 分析循环展开因子
- 考虑循环携带依赖的实际 UB 占用

### 8.2 性能权衡

#### 8.2.1 图构建开销

**开销**：
- 遍历所有操作和依赖关系：O(N * E)，其中 N 是操作数，E 是平均依赖数
- 构建 MemoryDependenceGraph：O(N^2)（最坏情况）

**优化**：
- 避免创建重复的边
- 使用哈希表加速查找

#### 8.2.2 循环依赖检测开销

**开销**：
- 对每个候选切分，执行 DFS：O(V + E)，其中 V 是节点数，E 是边数
- 可能需要多次尝试和回滚

**优化**：
- 使用已访问集合（`visited`）避免重复遍历
- 提前终止（early stop）

### 8.3 优化效果权衡

#### 8.3.1 切分粒度

**问题**：切分粒度过细可能导致：
- 调度开销增加
- 数据移动开销增加
- Block 间通信开销增加

**权衡**：
- 只在 UB 使用量显著减少时才切分
- 考虑切分后的数据移动成本

#### 8.3.2 正确性保证

**优先级**：正确性 > 性能优化

**保证机制**：
- 循环依赖检测
- MemoryDependenceGraph 考虑
- 回滚机制（`willCreateCycle` 中的临时修改）

---

## 9. 总结

UBUsageOptPass 是一个针对 Compute Block 的 UB 使用优化 Pass，核心思想是通过依赖图分析和切分策略，将部分操作移动到其他 Block，以减少 UB 的峰值使用量。

**核心贡献**：
1. **依赖图建模**：将操作、依赖关系和 UB 占用量建模为有向图
2. **智能切分策略**：通过活跃出口节点分析和依赖链查找，找到最优切分点
3. **正确性保证**：通过循环依赖检测和回滚机制，确保优化不破坏语义

**适用场景**：
- VECTOR_ONLY 类型的 Compute Block
- 存在跨 Block 数据依赖的场景
- UB 使用量接近或超过限制的情况

**未来改进方向**：
1. 更精确的 UB 占用估计（考虑 Memref、循环展开等）
2. 更智能的切分策略（考虑切分成本）
3. 支持其他控制流结构（`scf::IfOp` 等）
4. 性能优化（减少图构建和遍历开销）

---

## 附录 A：关键数据结构总结

| 数据结构 | 类型 | 说明 |
|---------|------|------|
| `op2nodeId` | `DenseMap<Operation *, int>` | 操作到节点 ID 的映射 |
| `nodeId2op` | `DenseMap<int, Operation *>` | 节点 ID 到操作的映射 |
| `linkOut` | `SmallVector<SmallVector<int>>` | 每个节点的出边列表 |
| `linkIn` | `SmallVector<SmallVector<int>>` | 每个节点的入边列表 |
| `linkSize` | `SmallVector<int>` | 边的权重（UB 占用字节数） |
| `linkStart` | `SmallVector<int>` | 边的起始节点 |
| `linkEnd` | `SmallVector<int>` | 边的终止节点 |
| `nodeBlockId` | `SmallVector<int>` | 节点所属的 Compute Block ID |
| `nodeCoreType` | `SmallVector<int>` | 节点的 CoreType |
| `nodeArgs` | `SmallVector<int>` | 节点对应的 BlockArgument 编号 |

## 附录 B：关键函数总结

| 函数名 | 功能 | 复杂度 |
|--------|------|--------|
| `buildUBUsageGraph` | 构建依赖图 | O(N * E) |
| `collectNeedUbOpts` | 识别优化候选 | O(N * E) |
| `collectRecordChange` | 寻找最优切分点 | O(N * E) |
| `applyRecordChange` | 应用变换 | O(V * (V + E)) |
| `willCreateCycle` | 循环依赖检测 | O(V + E) |
| `findDependency` | BFS 查找依赖节点 | O(V + E) |
| `isActiveEndNode` | 判断活跃出口节点 | O(V + E) |
| `sumIncomingLinkSize` | 计算入边权重总和 | O(E) |

---

**文档版本**：v1.0  
**最后更新**：2026-07-10  
**维护者**：Triton-Ascend 编译器团队