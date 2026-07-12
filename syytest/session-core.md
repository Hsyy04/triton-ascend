# CVPipeline pass分析

**Session ID:** ses_0aa0a995affekGg5eMkwENGKa5
**Created:** 7/12/2026, 6:52:53 PM
**Updated:** 7/12/2026, 11:37:24 PM

---

## User

我们分析一下CVPipeline这个pass

---

## Assistant

**Tool: grep**

**Tool: grep**

**Tool: grep**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: task**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: glob**

**Tool: read**

---

## Assistant

以下是 CVPipeline pass 的完整分析：

---

## CVPipeline 概述

CVPipeline（Cube-Vector Pipelining）是 **BiShengIR/HIVM 方言**中的一个 pass，它对包含 CUBE（矩阵乘法/Davinci）和VECTOR（向量）混合运算的 `scf.for` 循环进行**软件流水线化**。其目标是将循环展开为按核心类型分组的独立阶段，使 CUBE 和 VECTOR 的执行能够重叠进行。

### 核心文件

| 文件 | 角色 |
|---|---|
| `bishengir/lib/Dialect/HIVM/Transforms/CVPipelining.cpp` (1917行) | 主实现 |
| `bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.h` | `CVPipelineMode` 枚举和 pass 声明 |
| `bishengir/include/bishengir/Dialect/HIVM/Utils/WorklistBuilder.h` | 将操作按核心类型分区 |
| `bishengir/include/bishengir/Dialect/HIVM/Utils/WorkItem.h` | 每个核心分区的数据结构 |
| `bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td` | Pass 定义（TableGen） |
| `bishengir/lib/Dialect/HIVM/Utils/WorklistBuilder.cpp` | 分区引擎实现 |

---

## Pipeline 模式（`CVPipelineMode`）

在 `Passes.h:29` 中定义了四种模式：

| 模式 | 描述 |
|---|---|
| **`Off`** | 完全禁用流水线 |
| **`Unroll`** (默认) | 标准展开模式 — 将循环展开为外层展开循环 + 内层 `scf.for`（每个核心类型一个），通过 `iter_args` 转发缓冲区 |
| **`Skew`** | Skew/预取模式 — 使用 `scope::ScopeOp` 封装每个阶段，并扩展 `AllocWorkspaceOp` 缓冲区 |
| **`Dynamic`** | 动态模式（开发中） |

---

## Pass 入口（`CVPipeliningPass::runOnOperation()`，第 1875 行）

1. 如果 `pipelineDepth` 为 0 或 1，则跳过
2. 如果函数包含 BatchMatmul 循环，则跳过
3. 按**后序遍历**收集 `scf.for` 循环候选（先处理最内层）
4. 对于每个循环，构造一个 `CVPipelineImpl` 并运行 `.run()`
5. 成功时，将外层循环标记为已流水线化，以避免在已展开的嵌套循环上重复操作

---

## `CVPipelineImpl::run()` — 主流水线（第 1795 行）

这是核心方法。其运行步骤如下：

### 第一阶段：分析与分区
1. **`collectAtomicEffects()`** — 记录在 `SetAtomicOp` 活跃状态下写入的存储类操作
2. **`preprocessCounterAllocas()`** — 克隆包含计数器递增的区域操作（例如 matmul 后的 `scf.if`），以生成仅含 vector 的计数器递增副本
3. **`wlBuilder.build()`** — 调用 `WorklistBuilder` 通过依赖分析将 `pipelineLoop` 体中的操作分区为 CUBE / VECTOR 工作项。产出 `worklist`、`opToWorkItemMap`、`outputMemrefMap` 和 `resolvedMultibuffer`
4. **`absorbMergerOpsIntoWorkItems()`** — 将非核心“合并”操作（`arith.select`、`arith.cmpi`）吸收到依赖于其输出的工作项中，以避免悬空引用
5. **`collectWorkspaceAllocsForPreload()`**（仅 Skew 模式） — 查找用于预取流水线的 `AllocWorkspaceOp` 实例
6. **`markOutputs()`** — 将操作结果分类为：
   - `localOutputs` — 在阶段之间跨越但保持在同一迭代内的值 → 通过 `iter_args` 多缓冲地转发
   - `yieldedOutputs` — 跨越迭代边界的值 → 通过外层循环的 `scf.yield` 转发
   - 跳过延迟加载张量（按需克隆 `LoadOp`，不进行多缓冲）
7. **`checkWorkItemDependencies()`** — 在有问题的跨核心数据依赖造成静默错误之前，对其进行拒绝

### 第二阶段：Unroll 模式（默认）
8. **`createNewLoops()`** — 构建：
   - 一个外层 `scf.for`（`newLoop`），步长为 `step * numMultibuffer`
   - 每个工作项一个内层 `scf.for`，具有用于 `localOutputs`/`yieldedOutputs` 的 `iter_args`，以及一个重建的归纳变量
   - `expandOutputInits()` 将每个局部输出的初始张量/内存缓冲区扩展一个前导的 `numMultibuffer` 维度
9. **`migrateOps()`** — 将每个工作项的操作克隆到其对应的 `forOp` 中，重新连接 `tensor.extract_slice`/`insert_slice` 以索引正确的多缓冲区槽位，并通过 `scf.yield` 更新结果
10. 用 `newLoop` 的结果替换 `pipelineLoop` 的使用，删除原始循环，如果为 `trailingAtomicEffect`，则在流水线循环后恢复原子状态

### 第二阶段：Skew 模式（预取）
8. **`expandWorkspace()`** 和 **`markScopesForPreload()`** — 将每个工作项的操作克隆到一个 `scope::ScopeOp` 中，打上 `PreloadNumAttr` 和 `MaxPreloadNumAttr` 属性标签，并重新连接工作空间操作以使用带有预取提示的扩展缓冲区槽位

---

## 关键数据结构

### `WorkItem`（`WorkItem.h`）
```cpp
struct WorkItem {
  SmallVector<std::pair<Value, Value>> localOutputs;  // (原始, 扩展)
  SetVector<Operation *> ops;                          // 归属于此工作项的操作
  SmallVector<std::pair<Value, unsigned>> yieldedOutputs; // 向外层循环产出的值
  SmallVector<Operation *> workspaceOutputs;           // 预取存储操作
  TCoreType core;                                      // CUBE 或 VECTOR
  scf::ForOp forOp;                                    // 展开/迁移阶段创建
  IRMapping irMap;                                     // 用于克隆操作的重新映射
  Value reconstructedIV;                               // 从外部IV恢复的原始IV
  scope::ScopeOp scopeOp;                              // Skew模式包装器
};
```

### `WorklistBuilder`（`WorklistBuilder.h`）
- **依赖追踪**：为每个操作建立 `dependenceMap` 和 `loopCarriedDependenceMap`
- **延迟加载支持**：允许 `LoadOp` 克隆到多个消费工作项中，而不是通过多缓冲张量转发
- **跨核心检测**：识别在 CUBE 和 VECTOR 两个核心都有消费者的张量
- 分区通过贪心算法进行：提取准备就绪的操作并将其分组到工作项中

---

## 多缓冲区策略

Pass 在工作项之间展开 `numMultibuffer` 个槽位（通常由 `annotation::MarkOp` 多缓冲区属性确定）。每个局部输出得到一个扩展的缓冲区（`tensor.empty` 或 `memref.alloc`），通过 `tensor.extract_slice`/`insert_slice` 操作索引以选择正确的流水线阶段。

---

## 安全保护

- **`revert()`** — 在失败时通过从原始循环的克隆（`checkpoint`）恢复来撤销部分IR更改
- **`checkWorkItemDependencies()`** — 在IR更改前拒绝有问题的分区（跨核心循环依赖、未跟踪的交叉引用）
- **跨核心函数参数别名** — `markOutputs()` 会检测并拒绝存储和加载到相同函数参数但位于不同工作项中的情况
- **跨核心循环依赖** — 循环携带的张量 `iter_arg` 如果由 CUBE 产生并由 VECTOR 消费（或反之），则会被拒绝

---

## Option 选项

| 选项 | 默认值 | 描述 |
|---|---|---|
| `pipeline-depth` | `-1` | 流水线深度/多缓冲区槽位（-1 = 自动） |
| `pipeline-mode` | `unroll` | `unroll`、`skew` 或 `dynamic` |
| `enable-lazy-loading` | `false` | 允许 LoadOp 克隆到多个工作项中以减少缓冲区膨胀 |
| `enable-auto-balance` | `false` | 已弃用 |

## 下游影响

操作被赋予 `hivm.cv_pipelined_multi_buffer`（`HIVMAttrs.td:1113`）属性以标记多缓冲区扩展的分配，供后续 pass（如 ND2NZ 聚合分解）使用。

---

## User

illegalRegionedOp(Operation &op, bool isLoopMode) 这个函数是干什么的

---

## Assistant

**Tool: grep**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

**Tool: read**

**Tool: read**

---

## Assistant

`illegalRegionedOp`（`WorklistBuilder.cpp:225`）是 WorklistBuilder 的辅助函数，在遍历 `pipelineLoop` body 寻找分隔符操作时调用。其作用如下：

## 核心目的：对 `scf.for` / `scf.if` 等含 Region 的操作为 CUBE/VECTOR 分区打标

### 工作流

1. **遍历该 op 的所有 Region 体**，检查其中是否有 HIVM 方言的操作。对于每个 HIVM op，通过 `queryCoreTypeHelper` 查询其核心类型（CUBE / VECTOR）。

2. **如果 Region 内的 HIVM op 全是 CUBE** → 给该 Region op 打上 `pipeline.cubeonly` 属性
3. **如果全是 VECTOR** → 打上 `pipeline.veconly` 属性
4. **如果 CUBE 和 VECTOR 混在**：
   - **循环模式（CV pipelining）**：返回 `true`（= "illegal"），因为 CVPipeline 无法对一个混合核心的 Region op 做分离。WorklistBuilder 收到 `true` 后会返回 `failure()` 放弃对该循环的流水线化
   - **Block 模式（split-if）**：返回 `false`，不给该 op 打标，让 split-if 后续递归处理嵌套的混合 `scf.if`

5. **如果 Region 内完全没有 HIVM op**（边界情况）→ 额外检查是否为"提取的标量 gather"模式（带有 `ExtractedLoadOrStore` 属性的 `scf.for`，内部有 `memref.load` + `tensor.insert`），若是则视为 VECTOR 打标

### 下游用途

这些 `pipeline.cubeonly` / `pipeline.veconly` 属性被后续两步使用：
- **`extractAvailableOps`**：在每一轮分区时，将带有这些属性的 Region op 视为对应核心类型的"种子操作"拉入 WorkItem
- **`isCrossCoreLoad`**：判断一个 LoadOp 的 tensor 结果是否被跨核心消费时，优先读取这些属性来判断消费者核心类型（`WorklistBuilder.cpp:488-494`）

### 一句话总结

**`illegalRegionedOp` 的作用是在分区前对嵌套 Region op（`scf.for`/`scf.if`）按核心类型做预检和打标：纯 CUBE/VECTOR 就标注属性，混合核心则在循环模式下直接拒绝流水线化。**

---

## User

bool isCube = op->hasTrait<OpTrait::CoreTypeTrait<TCoreType::CUBE>::Impl>();
  bool isVec = op->hasTrait<OpTrait::CoreTypeTrait<TCoreType::VECTOR>::Impl>(); 
这两行是什么意思， 这个Trait是哪里来的，td里就定义好了每个op是CUBE还是VECTOR吗？还是哪个pass专门标记的？

---

## Assistant

**Tool: grep**

---

## Assistant

**Tool: read**

**Tool: read**

**Tool: read**

---

## Assistant

这两行代码查询的是**编译时静态附着在每个 Op 定义上的 Trait**，它的来源是 `.td` (TableGen) 文件。

## 定义链路

### 1. TableGen 定义 (`HIVMTraits.td:172-177`)

```td
class CoreTypeTrait<string CoreType>
    : ParamNativeOpTrait<"CoreTypeTrait", CoreType>;

def VectorCoreTypeTrait   : CoreTypeTrait<"TCoreType::VECTOR">;
def CubeCoreTypeTrait     : CoreTypeTrait<"TCoreType::CUBE">;
def CubeVectorCoreTypeTrait : CoreTypeTrait<"TCoreType::CUBE_OR_VECTOR">;
```

这三个 Trait 是 TableGen 层面的标记，编译 MLIR 时会由 `mlir-tblgen` 展开成 C++ Trait 类。

### 2. C++ Trait 实现 (`HIVMTraits.h:155-159`)

```cpp
template <hivm::TCoreType CoreType> class CoreTypeTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, CoreTypeTrait<CoreType>::Impl> {};
};
```

只是一个空的 Trait（无成员函数），纯粹用**类型标签**来标识核心类型。

### 3. 在 `.td` 中每个 Op 定义时直接声明

核心类型的归属是在**定义 Op 时就写死在 `.td` 里的**，不是某个 pass 动态标记的。例如：

- **DMA/Fixpipe/MacroOp** → `CubeCoreTypeTrait`（`HIVMDMAOps.td:251`, `HIVMMacroOps.td:52`）
- **Vector 操作** → 继承 `HIVM_VectorOp` 基类，该类自带 `VectorCoreTypeTrait`（`HIVMVectorOps.td:25`）
- **同步操作 (sync_block_lock 等)** → `VectorCoreTypeTrait`（`HIVMSynchronizationOps.td:203`）
- **通用辅助操作 (get_block_idx 等)** → `CubeVectorCoreTypeTrait`（`HIVMOps.td:29`）

### 4. 查询时用 `hasTrait` 检查

`queryCoreTypeHelper` (`HIVMInterfaces.cpp:40-61`) 的查询逻辑：

```
hasTrait<CoreTypeTrait<CUBE>::Impl>()   → isCube
hasTrait<CoreTypeTrait<VECTOR>::Impl>() → isVec
hasTrait<CoreTypeTrait<CUBE_OR_VECTOR>::Impl>() → isCubeVec
```

如果 Trait 查不到，则回退到 dynamic 接口 `InferCoreTypeInterface::inferCoreType()`。

## 总结

| 问题 | 答案 |
|---|---|
| Trait 哪来的？ | `.td` 文件，TableGen 编译阶段展开 |
| 每个 op 是 CUBE/VECTOR 是 td 就定义好的吗？ | **是**，Op 定义时的 Trait 列表中就已经写死了 |
| 有没有 pass 专门标记？ | **大部分没有** — 这是编译期静态标记。仅有 `illegalRegionedOp` 等少数场景会在 Region 级 `scf.for`/`scf.if` 上运行时打 `pipeline.cubeonly`/`pipeline.veconly` 属性，且这只影响 Region Op 整体，不影响内部具体 HIVM Op 的核心类型 |

---

## User

// Round-based extraction, alternating CUBE/VECTOR. Bound iterations by the
  // number of block-level ops (plus slack) — each round consumes at least one
  // op into a WorkItem, so more iterations than that indicates a bug.
  SmallVector<Operation *> independentOps;
  TCoreType core = hivm::TCoreType::CUBE_OR_VECTOR;
  const size_t maxRounds = targetBlock->getOperations().size() + 2;
  bool extractionDone = false;
  for (size_t round = 0; round < maxRounds && !extractionDone; ++round) {
    if (extractAvailableOps(independentOps, core).failed())
      return failure();
    if (independentOps.empty()) {
      extractionDone = true;
      continue;
    }
    if (core == hivm::TCoreType::CUBE_OR_VECTOR)
      return failure();
    if (populateWorkItem(independentOps, core).failed())
      return failure();

    for (auto &[op, dependant] : dependenceMap)
      for (Operation *processed : independentOps)
        dependant.erase(processed);
    independentOps.clear();

    if (core == TCoreType::VECTOR)
      core = TCoreType::CUBE;
    else if (core == TCoreType::CUBE)
      core = TCoreType::VECTOR;
    else
      return failure();
  }
  if (!extractionDone)
    return failure(); 
这一段在干什么？

---

## Assistant

这一段是 **WorklistBuilder 的核心分区算法**：用贪心轮换的方式将 `pipelineLoop` body 中的操作按 CUBE/VECTOR 核心类型分组为 WorkItem。

## 算法流程

```
for round in 0..maxRounds:
    extractAvailableOps(independentOps, core)   // 找出当前就绪的、匹配core类型的op
    if empty → extractionDone = true             // 没有更多了，退出
    
    populateWorkItem(independentOps, core)       // 打包成一个WorkItem
    从dependenceMap中清除已处理op的依赖             // 释放被阻塞的下游op
    
    core = 交替(VECTOR ↔ CUBE)                   // 下一轮换另一个核心
```

### 关键点

1. **交替轮换**：核心类型在 CUBE ↔ VECTOR 之间轮流。这样产生的 WorkItem 顺序天然是 CUBE-VECTOR-CUBE-VECTOR 交错的，符合"同一核心类型的操作尽量聚合在一个阶段"的软件流水线调度目标。

2. **`extractAvailableOps`**：扫描 block 中所有未分区的 op，通过 `dependenceMap` 检查依赖是否已全部就绪，并用 `queryCoreTypeHelper` / `illegalRegionedOp` 标注的属性判断该 op 的核心类型是否匹配当前轮次的 `core`。匹配的就抽出。

3. **`populateWorkItem`**：进一步追踪选中的 op 的依赖（需要放进去的辅助/胶水操作），创建一个 `WorkItem` 并加入 `worklist`。

4. **有界轮次**：`maxRounds = num_ops + 2`，每轮至少消耗一个 op，超过 ops 数量还没结束说明有死锁/循环依赖，报错退出。

5. **`CUBE_OR_VECTOR` 的 op**（如 `get_block_idx`）不代表特定核心，会被 `extractAvailableOps` 的匹配逻辑放过（或由 `traceDependentOps` 在不同 WorkItem 中分别拉入），不会单独形成一个 `core == CUBE_OR_VECTOR` 的 WorkItem 轮次。

### 一句话总结

**这是一个交替核心类型的贪心拓扑排序分区器：每轮收集所有依赖就绪且匹配当前核心的 op，打包成 WorkItem，然后释放依赖、换核心、继续下一轮，直到所有 op 分完。**

---

## User

maybeCore = queryCoreTypeHelper(&op).value_or(TCoreType::CUBE_OR_VECTOR);
      if (maybeCore != TCoreType::VECTOR && isCrossCoreCopy(&op))
        maybeCore = TCoreType::VECTOR;
      // Block-mode-only: CUBE_OR_VECTOR core ops (e.g. plain tensor-level
      // hivm.hir.copy) get pulled in as deps of consuming core ops via
      // traceDependentOps; silently skip here. Loop mode falls through to
      // the unexpected-core-type warning below — preserving cv-pipelining's
      // contract that any flexibly-typed core op left here is a bug.
      if (!isLoopMode && maybeCore == TCoreType::CUBE_OR_VECTOR)
        continue;
    } 
这里再说什么bug

---

## Assistant

**Tool: grep**

---

## Assistant

**Tool: read**

---

## Assistant

这段注释说的是 **loop 模式下出现 `CUBE_OR_VECTOR` 的 op 被视为 bug**，因为它在 pipeline 阶段不应该还存在。

## 背景

`extractAvailableOps` 遍历 block 中尚未分区的 op，判断它的核心类型：

```
1. 有 pipeline.cubeonly/veconly 属性？→ 直接用属性值
2. 否则 isCoreOp(op) && !isLoadLikeOp(op)？
   → 调用 queryCoreTypeHelper 查 Trait（静态核心类型）
   → 如果查到的是 CUBE_OR_VECTOR，
      且 block 模式 → continue（跳过，留给 traceDependentOps 拉入）
      且 loop 模式 → 不跳过，继续往下走
3. 走到 line 1013:
   if (maybeCore != VECTOR && maybeCore != CUBE)
     return emitWarning(...)  // ← 这里报错
```

## 为什么 loop 模式下这是 bug

CV pipelining（loop 模式）要求**每个 op 都必须能被明确分配到一个 WorkItem（CUBE 或 VECTOR）中**，因为最终要把循环体拆成按核心类型隔离的阶段。分阶段执行意味着 CUBE 的 op 只在 CUBE 循环里跑，VECTOR 的 op 只在 VECTOR 循环里跑。

如果一个 op 的核心类型是 `CUBE_OR_VECTOR`（比如普通的 `hivm.hir.copy`），它不知道应该进 CUBE 阶段还是 VECTOR 阶段。block 模式下没事，因为 block 不拆阶段——这些灵活的胶水 op 会被 `traceDependentOps` 借由依赖关系自动拉入消费方的 WorkItem。但 loop 模式下，分阶段执行意味着**你不能把一个 op 同时放进两个阶段**，而它又是 `CUBE_OR_VECTOR`（哪边都能跑、或者说哪边都不专属于），放在哪边都不对。

所以注释说"此时残留的 CUBE_OR_VECTOR op 是 bug"——它在前面应该已经被下游消费者通过依赖追踪拉走了，如果还残留着，说明分区逻辑有漏洞，这个 loop 就不能流水线化，直接报错退出。

### 一句话

**`CUBE_OR_VECTOR` 的 op 在 block 模式可以留给 `traceDependentOps` 按需分配，但 loop 模式必须提前被消费方拉走——残留到这个地方说明它孤立无主，pipeline 没法决定放 CUBE 还是 VECTOR 阶段，是分区逻辑的 bug。**

---

## User

详细介绍一下extractAvailableOps 

---

## Assistant

**Tool: read**

---

## Assistant

**Tool: read**

---

## Assistant

## `extractAvailableOps` 详细分析

这个函数是分区算法的**核心选择器**——每轮找出"依赖已就绪、核心类型匹配当前轮次"的操作。输入 `core` 是当前轮次期望的核心类型，输出 `extractedOps` 是本轮抽出的 op 列表。

函数分为 **4 个阶段**：

---

## 阶段 1：扫描候选（lines 991-1023）

遍历 `targetBlock` 中所有尚未分区的 op，筛选条件：

| 条件 | 处理 |
|---|---|
| 已在 `opToWorkItemMap` 中 | 跳过 |
| 有 `pipeline.cubeonly` / `pipeline.veconly` 属性 | 直接用属性值确定 `maybeCore` |
| 无属性但是 `isCoreOp` + 非 `isLoadLikeOp` | 调用 `queryCoreTypeHelper` 查 Trait。查不到 → block 模式跳过（留给 `traceDependentOps`），loop 模式报 error |
| 是跨核心 Copy（`isCrossCoreCopy`） | 视为 VECTOR |
| `maybeCore` 与当前轮的 `core` 不匹配 | 跳过（等对应的轮次） |
| `dependenceMap[op]` 非空（还有未满足的依赖） | 跳过 |

注意 `core` 是 **in/out 参数**：如果本轮没有任何匹配的 op，第一轮会从 `CUBE_OR_VECTOR` 开始，遇到一个合适的 op 后 `core` 被重设为该 op 的实际核心类型（line 1020）。

---

## 阶段 2：推迟循环携带依赖（lines 1025-1047）

```
遍历 potentiallyAvailable:
  检查 loopCarriedDependenceMap[op]
  如果 op 依赖的另一个 op 也同时在 potentiallyAvailable 中 → 通过
  否则 → deferredOps.insert(op)
```

然后做**传播闭包**：如果 `deferredOps` 中的某个 op 被另一个 `potentiallyAvailable` 的 op 使用，那个使用方也要一起推迟。这是为了避免：生产者本应被推迟但因不在 `loopCarriedDependenceMap` 中而漏掉，导致生产者本轮被拿走、被推迟的消费者下轮引用悬空。

**目的**：循环携带依赖（如 `tensor.insert_slice` → `scf.yield` → 下轮 `iter_arg`）表达的是一种跨迭代序关系。如果生产者和消费者本轮同时可用，但它们的正确执行需要按迭代顺序来，那么把它们分到同一个 WorkItem 中是不安全的——需要推迟。

---

## 阶段 3：合并 DPS-init 链（lines 1049-1124）

这是针对 **累加器链** 的优化。场景：

```
%a = mmad(%x, %y) inits(%acc0)     // CUBE, 产出 %a
%b = mmad(%x, %y) inits(%a)        // CUBE, 消费 %a 作为 init
```

`mmad` 是一个 DPS op，`%a` 作为第二个 `mmad` 的 init operand（不是普通输入，是目的地张量）。此时 `%a` 必须是一个可扩展的局部输出——但如果第二个 `mmad` 本轮还不可用（依赖未就绪），而第一个 `mmad` 本轮可用，那第一个 `mmad` 会被拿走，`%a` 变成跨 WorkItem 的 `localOutput`。问题：`expandOutputInits` 无法把这种**非 tensor.empty、非 to_tensor 的 init**扩展为多缓冲区。

**`findBlockedCoChainConsumer`** 检测这种模式：
- op 有唯一结果
- 结果的唯一使用者是同一个父层下的 DPS core-op
- 使用者是用 op 的结果作为 init
- 使用者核心类型与本轮相同
- 使用者本轮**不可用**（依赖未就绪或已被推迟）

如果检测到，就把生产者 `chainDeferred`，等下轮跟消费者一起打包进同一个 WorkItem。

**安全保护**：提交 `chainDeferred` 前检查如果全推迟了 `remaining == 0`，则放弃——否则当前轮变成空轮，分区算法误以为结束了。

---

## 阶段 4：提交（lines 1126-1128）

```
potentiallyAvailable.set_subtract(deferredOps);   // 删掉被推迟的
extractedOps.append(potentiallyAvailable.takeVector());  // 剩下的全拿出来
```

---

## 整体数据流

```
                dependenceMap
                     │
                     ▼
targetBlock ──▶ 核心类型匹配 + 依赖就绪 ──▶ potentiallyAvailable
                                                 │
                                    ┌────────────┼────────────┐
                                    ▼            ▼            ▼
                             loopCarried       DPS-init      提交到
                             推迟闭包          链推迟        extractedOps
```

每个 op 只会被这个函数看到一次（一旦被抽走，下一轮 `opToWorkItemMap.contains` 就命中了），所以分区保序且无遗漏。

---

## User

解释一下absorbMergerOpsIntoWorkItems函数

---

## Assistant

**Tool: read**

---

## Assistant

## `absorbMergerOpsIntoWorkItems` 解析

这个函数解决一个**残差归属**问题：`WorklistBuilder` 只分区 HIVM 核心操作（CUBE/VECTOR），但循环体中还有大量**非核心的"胶水"操作**——`arith.select`、`arith.cmpi`、`arith.addi` 等——它们既不是 CUBE 也不是 VECTOR，因此没有被任何 WorkItem 认领。如果不处理，`migrateOps` 克隆时引用了它们就会产生悬空引用，最终 `pipelineLoop->erase()` 时崩溃。

函数分两部分：

---

## 第一部分：scf.yield 链吸收（lines 516-561）

### 问题场景

```
%matmul_result = hivm.mmad ...          // CUBE, 已被认领进 WorkItem_CUBE
%cmp = arith.cmpi ... %matmul_result    // 无人认领
%sel = arith.select %cmp, %a, %b         // 无人认领
scf.yield %sel                            // yield 的是 %sel
```

`%sel` 是 `scf.yield` 的操作数，但 `migrateOps` 克隆 `scf.yield` 时用的是 `globalIRMap`，不会重新映射未认领的 `%sel`→ 克隆后的 `scf.yield` 仍引用老的 `%sel`，删除原循环体后悬空。

### 算法

```
对每个 scf.yield 操作数:
  如果它不是 WorkItem 直接产出的:
    沿着 SSA 向后 DFS，收集所有未认领的非核心 op → chain
    同时记录这条 chain 的叶子依赖哪些 WorkItem → producers
    
  如果 producers.size() == 1:
    把整条 chain 全部吸收进那个唯一的 WorkItem
  否则:
    报错退出 (无法判断归属)
```

关键约束是 `producers.size() == 1`：吞并链必须只依赖于**一个** WorkItem。如果 `%sel` 的输入同时来自 CUBE WorkItem 和 VECTOR WorkItem（比如跨核心的标量决策），无法判断该放哪边，直接放弃流水线化。

---

## 第二部分：计数器递增吸收（lines 564-596）

这是针对 **normalize-matmul 的计数器 alloca** 的特殊处理。

### 问题场景

```
%cnt_alloca = memref.alloca ... {normalize_matmul_counter}
// ... 
%v = memref.load %cnt_alloca     // 进入某 WorkItem 作为依赖
// ...
%inc = arith.addi %v, %c1        // 无人认领
memref.store %inc, %cnt_alloca   // 无人认领
```

这两个操作（`arith.addi` + `memref.store`）**不产生任何 SSA 结果**，因此永远不会出现在 `scf.yield` 链上，第一部分不可能发现它们。但如果它们不被克隆进 WorkItem，流水线化后计数器就不会递增——后续的 `memref.load` 永远读到 0。

### 算法

```
对每个 WorkItem:
  找出它内部引用了哪些计数器 alloca (通过 memref.load 匹配 kNormalizeMatmulCounterAttr)
  在循环体中寻找写回这些 alloca 的 memref.store (还未被认领的)
  把 store 和它上游的 arith.addi 一起吸收进本 WorkItem
```

注意注释提到：如果计数器递增发生在**regioned op** 内部（如 matmul 后的 `scf.if`），则由 `preprocessCounterAllocas` 单独处理——它会为 VECTOR 阶段克隆一个不含 CUBE 操作的 skeleton 版本。

---

## 总结

| 部分 | 吸收目标 | 吸收条件 |
|---|---|---|
| scf.yield 链 | `arith.select`、`arith.cmpi` 等胶水 op | 链的叶子只依赖一个 WorkItem |
| 计数器递增 | `arith.addi` + `memref.store` 写回计数器 | 该 WorkItem 内部有读该计数器的 load |

---

## User

createNewLoops详细介绍一下这个

---

## Assistant

**Tool: read**

---

## Assistant

## `createNewLoops` 详细分析

这个函数负责构建流水线化的**物理循环结构**。它不克隆任何操作体——那由 `migrateOps` 完成——它只搭建骨架：一个展开的外层循环 + 每个 WorkItem 一个内层 jam 循环。

---

## 整体 IR 变换

假设原始循环 `pipelineLoop` 是：

```
scf.for %iv = 0 to 10 step 1 {
  CUBE ops ...  VECTOR ops ...    // 混排
  scf.yield %y
}
```

`numMultibuffer = 3`，2 个 WorkItem（CUBE, VECTOR），变换后：

```
scf.for %outer = 0 to 10 step 3 {           // newLoop
  %capped = ceilDiv(10 - %outer, 1)
  %actual = min(%capped, 3)                  // min(剩余, 多缓冲区数)

  // 阶段 1: CUBE
  scf.for %c = 0 to %actual step 1 {
    %rec = %c * 1 + %outer                   // 重建原始 iv
    // migrateOps 将 CUBE ops 克隆到这里
  }

  // 阶段 2: VECTOR
  scf.for %v = 0 to %actual step 1 {
    %rec = %v * 1 + %outer
    // migrateOps 将 VECTOR ops 克隆到这里
  }

  scf.yield ...                              // 克隆自原 yield
}
```

核心语义：**原循环的每次迭代被拆成两个阶段（CUBE → VECTOR），多缓冲区数（3）个外层迭代打包在一起执行。**

---

## 逐段解析

### 1. 创建外层展开循环（lines 1088-1108）

```cpp
Value unrollVal = const(numMultibuffer);        // 3
Value newStep = originStep * unrollVal;         // step * 3
newLoop = scf::ForOp(lb, ub, newStep, inits);   // 步长变为原来的 numMultibuffer 倍
```

- `newLoop` 的 `iter_args` 直接继承 `pipelineLoop` 的初始值
- 打上 `kCVUnrolledLoopName` 属性标记，供下游 pass 识别
- `globalIRMap` 映射原 `iter_args` → 新 `iter_args`，作为基础映射

### 2. 计算内层的上界（lines 1110-1130）

```cpp
cappedUB = ceilDiv(ub - iv, originStep)     // 剩余迭代数
actualUB = min(cappedUB, numMultibuffer)    // 最多 numMultibuffer 个
```

`cappedUB` 防止内层超过剩余迭代数（尾部不足一次展开时）。`min(..., pipelineIters)` 确保在循环中间 body 正常运行（`numMultibuffer` 个迭代一批）。

### 3. 对每个 WorkItem 创建 jam 循环（lines 1132-1189）

每个 WorkItem 依次处理：

**a) 扩展局部输出缓冲区（`expandOutputInits`）**

对每个 `localOutput`，将原 tensor.alloc/memref 扩展为前导 `numMultibuffer` 维度的多缓冲区：

```
原: tensor<64x128xf16>
扩展: tensor<3x64x128xf16>     // 前导 numMultibuffer 个 slot
```

**b) 构建 init 列表（lines 1139-1148）**

```
[yieldedOutput 的 iter_arg, ..., localOutput 的扩展 init, ...]
```

`yieldedOutputs` 的 init 直接复⽤ `newLoop` 的对应 `iter_arg`（来自外层循环）。`localOutputs` 使用刚刚扩展的缓冲区。

**c) 创建内层 for 循环（lines 1151-1156）**

```cpp
item->forOp = scf::ForOp(0, actualUB, 1, inits);
```

打上两个属性：
- `kPipelinedLoopCoreTypeAttrName` → `item->core`（CUBE 或 VECTOR）
- `kMultibufferUnrollAttrName` → `numMultibuffer`

**d) 重建原始归纳变量（lines 1158-1160）**

```cpp
item->reconstructedIV = innerIV * originStep + outerIV
```

用 `AffineMap: d0 * s0 + s1` 将 `(innerIV, step, outerIV)` 映射回原始循环的 iv 值。后续 `migrateOps` 用此替换对 `%orig_iv` 的引用。

**e) 填充 IRMapping（lines 1162-1183）**

| 映射 Key | 映射 Value |
|---|---|
| `pipelineLoop` 的 `yieldedValues[pos]` | `item->forOp` 的 `result[pos]` |
| `pipelineLoop` 的 `origIV` | `item->reconstructedIV` |
| `pipelineLoop` 的 `iter_arg[yieldArg]` | `item->forOp` 的 `iter_arg[yieldArg]` |

`item->irMap` 以 `globalIRMap` 为基础，逐步叠加——每个 WorkItem 共享全局映射，但叠加自己的 IV 和 iter_arg 专属映射。

---

## 图示总结

```
                   pipelineLoop (%iv, step=1)
                   ┌──────────────────────┐
                   │  CUBE  │  VECTOR      │
                   │  ops   │  ops         │
                   └──────────────────────┘
                            │
                   createNewLoops
                            ▼
                   newLoop (%outer, step=3)
                   ┌──────────────────────────────────┐
                   │ for %c = 0..min(rem,3)            │
                   │   %rec = %c*1 + %outer             │
                   │   [CUBE ops clone]                 │
                   │                                    │
                   │ for %v = 0..min(rem,3)            │
                   │   %rec = %v*1 + %outer             │
                   │   [VECTOR ops clone]               │
                   │                                    │
                   │ scf.yield                          │
                   └──────────────────────────────────┘
```

这样 CUBE 和 VECTOR 的执行就在循环层面物理隔离了，硬件可以分别调度它们的多缓冲区流水线。每个 WorkItem 的 `forOp` 遍历 `actualUB` 个 slot（最多 `numMultibuffer`），`tensor.extract_slice`/`insert_slice` 按 `innerIV` 索引对应的多缓冲区 slot，实现软件流水线。

---

## User

migrateOps这个函数呢？

---

## Assistant

**Tool: read**

---

## Assistant

## `migrateOps` 详细分析

这个函数是 pipeline 的**执行阶段**——把循环体中的操作真正克隆到 `createNewLoops` 建好的各 WorkItem 的 `forOp` 中，并重连跨阶段数据流（`tensor.extract_slice`/`tensor.insert_slice`）。分三个阶段：

---

## 阶段 1：克隆操作体（lines 1232-1251）

```
对 pipelineLoop body 中的每个 op:
  如果 opToWorkItemMap 没有它的记录 → 跳过（胶水 op 已被 absorbMergerOps 处理掉）
  否则，对每个目标 WorkItem:
    如果该 op 有原子效应 → 插入 SetAtomicOp 包裹
    调用 builder.clone(op, target->irMap) 克隆
    关闭原子效应
```

`clone` 使用 `target->irMap` 自动映射所有操作数——之前 `createNewLoops` 已把 `origIV → reconstructedIV`、`iter_args` 等填入 map，所以克隆后的 op 会直接使用重映射后的值。

---

## 阶段 2：重连局部输出（lines 1256-1468）

这是最复杂的部分。对每个 WorkItem，遍历 `localOutputs`，用 `extract_slice`/`insert_slice` 把多缓冲区 slot 正确地索引进去。根据 `expanded`（扩展后缓冲区的类型）分三种情况：

### 2a. `scf::ForOp` — 标量 gather 模式（lines 1295-1341）

当 `localOutput` 来自一个内部 `scf.for`（如标量 gather 循环），它没有 DPS 接口。处理方式：

```
// extract_slice 从多缓冲区 iter_arg 中切出第 iv 个 slot 作为 gather 的 init
%slot = tensor.extract_slice %expanded_iter_arg[%iv, ...] 
scf.for init(%slot) {
  ...
}
// insert_slice 把 gather 结果写回相同的 slot
%updated = tensor.insert_slice %gather_result into %expanded_iter_arg[%iv, ...]
```

内部循环内的使用者用 `%gather_result` 替换 `%orig`，外部使用者（其他 WorkItem 的循环）用 `extract_slice(%newResult, %userIV)` 读取对应 slot。

### 2b. DPS op + tensor 类型 init（lines 1364-1376）

最常见的情况——输出由 DPS op（如 matmul）产生，init 是 `tensor.empty`：

```
// 切出第 iv 个 slot 作为 DPS init
%slot = tensor.extract_slice %expanded_tensor[%iv, ...]
matmul ins(%a, %b) inits(%slot)  // 替换掉原来的 tensor.empty
// 产出写回
%updated = tensor.insert_slice %matmul_result into %expanded_tensor[%iv, ...]
```

本 WorkItem 内的使用者看到 `%matmul_result`，其他 WorkItem 用 `extract_slice` 读取对应 slot。

### 2c. DPS op + memref 类型 init（lines 1377-1441）

当 `expanded` 是 `memref.alloc`（Fixpipe 写直接 alloc，或跨核心 `hivm.hir.copy`）。处理方式：

```
// 给 to_tensor (或 alloc) 创建 subview，索引到第 iv 个 slot
%subview = memref.subview %expanded_alloc[%iv, ...]
// 替换 to_tensor 的 memref 操作数 → 指向当前 slot
to_tensor %subview
// extern: 创建新的 to_tensor 包裹整个多缓冲区供外部使用
%result = to_tensor %expanded_alloc
```

额外的 `updateMaskingSubview` 处理：如果原始 DPS init 是通过 `memref.subview` 传递的（做 mask 裁剪的情况），需要先更新那个 subview 的源指向多缓冲区，再创建 slot 级别的 subview。

### 共同收尾：更新外部使用者（lines 1447-1467）

```
对本 WorkItem 和 pipelineLoop 之外的所有 orig 使用者:
  找到使用者所属的内层 forOp
  创建 extract_slice(%newResult, %userIV) 指向该循环的多缓冲区 slot
  替换该 use
```

---

## 阶段 3：创建 yield 和 terminator（lines 1470-1481）

```
对每个 WorkItem:
  在 forOp body 末尾创建 scf::YieldOp(yieldVals)
    yieldVals = [yieldedOutput 的映射值, ... , insert_slice 的 updated 值, ...]

在 newLoop body 末尾:
  如果存在 trailingAtomicEffect → 恢复原子状态
  克隆原始循环的 scf.yield（此时 globalIRMap 已完整映射所有 yield 操作数）
```

---

## 图示总结

```
创建前 (createNewLoops 后的空壳):
newLoop {
  for CUBE (%c = 0..min(rem,3), iter_args(%iter_arg0, %iter_arg1, ...)) {
    // 空
  }
  for VECTOR (%v = 0..min(rem,3), iter_args(...)) {
    // 空
  }
  scf.yield ...  // pending
}

migrateOps 后:
newLoop {
  for CUBE (%c=0..min(rem,3), iter_args(%y_outer, %buf_3x64)) {
    %rec = %c * step + %outer
    %slot = extract_slice %buf_3x64[%c]   // 选对 slot
    %matmul_res = matmul ... inits(%slot)   // CUBE ops 克隆
    %updated = insert_slice %matmul_res into %buf_3x64[%c]
    scf.yield %matmul_res, %updated
  }
  for VECTOR (%v=0..min(rem,3), iter_args(%y_from_cube, %vec_buf)) {
    %rec = %v * step + %outer
    %slot = extract_slice %vec_buf[%v]
    %matmul = extract_slice %y_from_cube[%v]   // 读 CUBE 产出
    %vec_res = vector_op %matmul, %slot          // VECTOR ops 克隆
    %updated = insert_slice %vec_res into %vec_buf[%v]
    scf.yield %vec_res, %updated
  }
  scf.yield %yield_from_vector  // 克隆自原始
}
```

核心语义：**`tensor.extract_slice[%iv]/insert_slice[%iv]` 按内层循环索引将操作分别指向多缓冲区的正确 slot，实现跨 CUBE/VECTOR 阶段的软件流水线数据转发。**

---

