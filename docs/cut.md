# 2 方案设计

## 2.0 概述

**问题定义**

给定一个 Fuse Group（待融合的操作集合），将其划分为两个子集：
- **Kept 集合**：当前计算块需要融合的操作子集
- **Cut 集合**：不在当前计算块，转移到下次计算的操作子集

**核心术语**

| 术语 | 定义 |
|------|------|
| Fuse Group | 待融合的操作集合，由上游拓扑搜索框架提供，类型为 `Operation*` |
| Kept Set | 当前计算块需要融合的操作子集 |
| Cut Set | 不在当前计算块，转移到下次计算的操作子集 |
| CUBE相邻OP | 数据依赖上直接连接到 CUBE_ONLY 操作的 op |
| UB 缓冲区 | 用于存储中间结果的硬件缓冲区 |
| 迭代变量 | `scf::ForOp` 的 region arguments，通过 `scf::YieldOp` 更新 |

**模块架构**

```
┌─────────────────────┐
│ KeptSetInitializer  │──┐
└─────────────────────┘  │
                         ├──> ┌──────────────────────┐
┌─────────────────────┐  │    │ FuseGroupSplit │
│    UBPlanner        │──┼──> └──────────────────────┘
└─────────────────────┘  │
                         │
┌─────────────────────┐  │
│  SplitScorer    │──┘
└─────────────────────┘
```

**模块职责**

| 模块 | 职责 |
|------|------|
| KeptSetInitializer | 初始化 Kept 集合，确定必须保留的操作 |
| UBPlanner | 分析 UB 使用情况 |
| SplitScorer  | 对操作进行评分，指导分割决策 |
| FuseGroupSplit | 综合前三者的结果，执行最终的分割 |

<div> <img src="https://wiki.huawei.com/vision-file-storage/api/file/download/upload-v2/WIKI2026071811912805/47315481/03837d80a540423ea4b7dba71a8e4757.png" width="60%" /> </div>

## 2.1 Kept 集合初始化 (KeptSetInitializer)

**输入**
- Fuse Group：`SmallVector<Operation*> fuseGroup`
- CUBE_ONLY 操作列表（按 MLIR Block 中的出现顺序）

**输出**
- Kept 集合：`SmallPtrSet<Operation*> keptSet`

**算法流程**

```
1. 调用 findOpsAdjacentToCube(fuseGroup)
   - 在 fuse group 中找到所有数据依赖上直接连接到 CUBE_ONLY 操作的 op
   - 如果某个 op 相邻多个 CUBE_ONLY 操作：
     * 按 CUBE_ONLY 操作在 MLIR Block 中的出现顺序（与 block_id 顺序一致）
     * 只处理最后一个 CUBE_ONLY 操作相邻的 op
     * （注：搜索框架保证当前处理的是最后一个 CUBE 块）

2. 对每个相邻 op，调用 collectKeepOps(op, keptSet)
   - 将 op 的所有数据依赖（传递闭包）加入 keptSet
   - 如果 op 的结果直接被 store 类操作使用：
     * 将该 store 操作加入 keptSet
     * 将该 store 操作的所有数据依赖加入 keptSet
```

**数据依赖闭包定义**

对操作 `op`，其数据依赖闭包包含：
- `op` 的所有 operands
- 递归地，所有 operands 的 operands（直至达到操作定义或 Block argument）

**示例**

```
%fused = arith.addi %a, %b        // 在 keptSet 中（相邻 CUBE）
store %fused, %ptr                 // 在 keptSet 中（直接使用 %fused）
%unused = arith.muli %a, %c       // 不在 keptSet 中（无数据依赖）
```

## 2.2 UB 使用分析 (UBPlanner)

**设计目标**

通过最大流算法，分析操作分割对 UB 缓冲区的影响，确保分割决策满足 UB 容量约束。

**数据结构**

```cpp
struct UBGraph {
  DenseMap<Operation*, int> op2nodeId;        // 操作到节点ID的映射
  DenseMap<int, Operation*> nodeId2op;        // 节点ID到操作的映射
  SmallVector<SmallVector<int>> linkOut;      // 邻接表：每个节点的出边
  SmallVector<SmallVector<int>> linkIn;       // 邻接表：每个节点的入边
  SmallVector<int64_t> linkSize;              // 正向边容量（UB 大小）
  SmallVector<int64_t> linkSizeRev;           // 反向边容量（初始为 0）
  SmallVector<int> linkStart;                 // 边的起点节点
  SmallVector<int> linkEnd;                   // 边的终点节点
  SmallVector<int64_t> resUB;                 // 残余网络：每条边的残余容量
};

class UBPlanner {
private:
  UBGraph graph;
  const int64_t MAX_UB_SACRIFICE;             // 允许的最大 UB 浪费阈值
  
public:
  UBPlanner(int64_t maxUbSacrifice) : MAX_UB_SACRIFICE(maxUbSacrifice) {}
};
```

**边的语义**

- 方向：`A → B` 表示操作 A 的结果被操作 B 使用（数据流方向）
- 容量：边的 UB 大小（通过 `getValueSizeInBytes(Value)` 计算）

**算法流程**

### 2.2.1 构建流网络 (buildUBGraph)

```
输入：fuse group, kept set
输出：UBGraph

1. 创建节点映射
   - 为 fuse group 中的每个操作分配一个节点 ID
   
2. 建立数据依赖边
   - 对每个操作 op，遍历其所有 operands
   - 如果 operand 定义操作在 fuse group 中：
     * 添加边：operand_def → op
     * 边容量 = getValueSizeInBytes(operand)
   
3. 添加虚拟节点
   - 创建虚拟源节点（代表 kept 集合）
   - 创建虚拟汇节点（代表 cut 集合）
   
4. 连接虚拟源
   - 对所有入度为 0 的节点：
     * 添加边：virtual_source → node
     * 边容量 = 0
     
5. 连接虚拟汇
   - 对所有出度为 0 的节点：
     * 添加边：node → virtual_sink
     * 边容量 = 0
```

### 2.2.2 计算最小割 (findMinCut)

```
输入：UBGraph
输出：残余网络 resUB

1. 执行最大流算法（如 Edmonds-Karp 或 Dinic）
   - 源：virtual_source
   - 汇：virtual_sink
   - 目标：找到最小割
   
2. 保存残余网络
   - resUB[edge] 表示边 edge 的残余容量
   - resUB[edge] == 0 意味着边在最小割中
```

### 2.2.3 判断操作是否可分割

**canCut(Operation* op)**

判断操作 `op` 是否可以放入 Cut 集合。

```
输入：操作 op
输出：true/false

1. 初始化 sacrifice = MAX_UB_SACRIFICE
2. 从 op 开始，沿数据依赖方向（operand）向上 BFS
3. 对每条经过的边 edge：
   - 如果 resUB[edge] > 0：
     * sacrifice -= resUB[edge]  // 累计 UB 浪费
   - 如果 sacrifice < 0：
     * 停止搜索，返回 false
4. 如果搜索过程中遇到 kept 集合中的节点：
   - 返回 false（op 不能放入 cut）
5. 如果搜索结束仍未遇到 kept 节点：
   - 返回 true（op 可以放入 cut）
```

**canKeep(Operation* op)**

判断操作 `op` 是否可以保留在 Kept 集合。

```
输入：操作 op
输出：true/false

1. 初始化 sacrifice = MAX_UB_SACRIFICE
2. 从 op 开始，沿数据流方向（user）向下 BFS
3. 对每条经过的边 edge：
   - 如果 resUB[edge] > 0：
     * sacrifice -= resUB[edge]
   - 如果 sacrifice < 0：
     * 停止搜索，返回 false
4. 如果搜索过程中遇到 cut 集合中的节点：
   - 返回 false（op 不能保留在 kept）
5. 如果搜索结束仍未遇到 cut 节点：
   - 返回 true（op 可以保留在 kept）
```

**关键属性**

- `sacrifice` 初始化为 `MAX_UB_SACRIFICE`，随着路径上的 UB 浪费递减
- 只要 `sacrifice >= 0`，即使浪费较大也认为可接受
- 最大流保证找到满足约束的最小割

## 2.3 分割评分 (SplitScorer)

**设计目标**

为每个操作分配一个分值，正值表示更适合保留在 Kept 集合，负值表示更适合放入 Cut 集合。

**数据结构**

```cpp
class SplitScorer {
private:
  DenseMap<Operation*, double> scores;        // 每个操作的分值（浮点数，无上下界）
  SmallVector<Operation*> sortedOps;          // 按分值绝对值排序的操作列表
  
public:
  void computeScores(SmallVector<Operation*> fuseGroup, 
                      SmallPtrSet<Operation*> keptSet);
  double getScore(Operation* op);
  SmallVector<Operation*> getSortedOps();
};
```

**算法流程**

### 2.3.1 分值种子生成

对 fuse group 中的每个操作 `op`，根据以下规则初始化分值：

**规则 1：迭代变量约束**

迭代变量的使用和更新应尽可能放在同一个计算块中。

```
1. 找到所有使用迭代变量的操作（iter_var_users）
2. 对每个 iter_var_user：
   - 溯源其迭代变量的更新操作（yield 前的计算操作）
   - 如果更新操作在 kept 集合中：
     * score[iter_var_user] += 1.0
   - 如果更新操作在 fuse group 待选择操作中：
     * score[iter_var_user] += 1.0
     * score[update_op] += 1.0
   - 如果更新操作不在 fuse group 中：
     * score[iter_var_user] -= 1.0
```

**示例**

```mlir
scf.for %i = 0 to 10 step 1 iter_args(%x = %init) {
  %new_x = arith.addi %x, %c1    // 更新操作
  %result = arith.muli %new_x, %y  // 使用操作
  scf.yield %new_x
}
```
- 如果 `%new_x` 在 kept 中：`%result` 得分 +1
- 如果 `%new_x` 在待选择中：`%new_x` 和 `%result` 都得分 +1

**规则 2：Load/Store 操作优化**

Load/Store 类操作应尽可能早发射（early dispatch）。

```
对每个 load/store 类操作 op：
  score[op] += 1.0
```

**规则 3：i1 类型依赖优化**

i1 类型的数据依赖不应跨越计算块（性能优化）。

```
对每个产生 i1 类型结果的操作 op：
  score[op] = sum(score[user] for all users of op)
```

### 2.3.2 分值传播

```
输入：初始分值 scores
输出：传播后的分值

1. 构建数据依赖图
2. 负分向下传播（沿数据流方向：op → user）
   - 对每个 score[op] < 0 的操作：
     * 对其所有 user：
       score[user] += score[op]
3. 正分向上传播（沿数据依赖方向：op → operand）
   - 对每个 score[op] > 0 的操作：
     * 对其所有 operand 定义操作（如果在 fuse group 中）：
       score[operand_def] += score[op]
```

**传播策略**
- 无衰减系数，直接传播至边界
- 边界：操作不在 fuse group 中，或已访问

### 2.3.3 操作排序

```
sortedOps = 按 |score[op]| 降序排序 (fuse_group \ kept_set)
```

**决策语义**
- `score > 0`：操作倾向于保留在 Kept 集合
- `score < 0`：操作倾向于放入 Cut 集合
- `|score|` 越大：优先级越高

## 2.4 分割执行 (FuseGroupSplit)

**设计目标**

综合 KeptSetInitializer、UBPlanner 和 SplitScorer 的结果，执行最终的分割。

**输入**
- Fuse Group：`SmallVector<Operation*> fuseGroup`
- Kept 集合：`SmallPtrSet<Operation*> keptSet`（来自模块 1）
- UBPlanner 实例：提供 `canCut()` 和 `canKeep()` 方法
- 排序后的操作列表：`SmallVector<Operation*> sortedOps`（来自模块 3）
- 分值映射：`DenseMap<Operation*, double> scores`（来自模块 3）

**输出**
- 最终 Kept 集合：`SmallPtrSet<Operation*> keptSet`
- 最终 Cut 集合：`SmallPtrSet<Operation*> cutSet`

**数据结构**

```cpp
class FuseGroupSplit {
private:
  SmallPtrSet<Operation*> keptSet;
  SmallPtrSet<Operation*> cutSet;
  SmallPtrSet<Operation*> visited;
  
public:
  void split(SmallVector<Operation*> fuseGroup,
                 SmallPtrSet<Operation*> initialKept,
                 UBPlanner& ubPlanner,
                 SmallVector<Operation*> sortedOps,
                 DenseMap<Operation*, double> scores);
};
```

**算法流程**

```
1. 初始化
   - keptSet = initialKept（来自 KeptSetInitializer）
   - cutSet = {}（空集）
   - visited = {}（空集）
   
2. 遍历排序后的操作
   for (op in sortedOps) {
     if (op in visited) continue;
     
     if (scores[op] > 0 && ubPlanner.canKeep(op)) {
       updateKeptSet(op);
     } else if (scores[op] < 0 && ubPlanner.canCut(op)) {
       updateCutSet(op);
     }
   }
   
3. 确保所有操作都被访问
   assert(fuseGroup.size() == visited.size());
```

**更新集合**

### updateKeptSet(Operation* op)

将操作及其数据依赖收集到 Kept 集合。

```
1. 从 op 开始，沿数据依赖方向（operand）向上遍历
2. 收集所有依赖操作（传递闭包）
3. 将 op 及其所有依赖加入 keptSet
4. 将 op 及其所有依赖标记为 visited
```

### updateCutSet(Operation* op)

将操作及其用户收集到 Cut 集合。

```
1. 从 op 开始，沿数据流方向（user）向下遍历
2. 收集所有用户操作（传递闭包）
3. 将 op 及其所有用户加入 cutSet
4. 将 op 及其所有用户标记为 visited
```

**完整性保证**

- Fuse Group 无环：算法假设由上游保证 fuse group 中不存在循环依赖
- 所有操作被访问：算法保证每个操作最终都会被分配到 keptSet 或 cutSet
- visited 标记：防止重复处理已分配的操作

**决策优先级**

按操作分值绝对值降序处理，优先处理影响最大的操作。
