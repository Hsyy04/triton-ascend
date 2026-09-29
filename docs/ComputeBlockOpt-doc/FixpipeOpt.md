# FixpipeOpt

## Overview

### 1. 目标

利用 fixpipe 随路量化能力， 将可以被fixpipe融合的算子与对应的matmul放入同一计算块中，在NPUIR的InlineFixpipePass中，这些算子会被inline到fixpipe语句中。

### 2. 规格
1. 目前仅进行以下两种的随路量化模式， 链中每个中间结果必须是 单User（`MarkOp` 不计入用户计数，最终必须是store到GM的语义）， 链中每一个op必须在统一mlir层级.
   - Cast 模式：`matmul → arith.truncf/trunci → tensor.extract_slice → materialize_in_destination(subview(gm))` 

   - Mul 模式：`matmul → arith.mulf/muli → (tensor.extract_slice)? → materialize_in_destination/store(subview(gm))`

2. 若 matmul 是循环携带累加器，沿 `scf.for` 外提得到"逃逸循环后的最终值", 该特性仅支持**单个for循环**。
3. 上述链中依赖的所有scalar将按照如下原则复制：
   - 若 scalar 的结果被链外的op使用，则克隆一份给链内使用，原op留给链外使用。
   - 若 scalar 的结果仅被链内使用，则不克隆。
4. block_id的处理：
   - 若 matmul 的结果是循环携带累加器，则将链中所有op的block_id设置为 matmul 所在的 block_id。
   - 若 matmul 的结果不是循环携带累加器，则将链中所有op的block_id设置为 matmul 外层的for一致的block_id， 如果外层for的block_id为-1，则为for设置一个全新的block_id。
   
   注意，block_id的处理是以op为粒度进行的, 也就是说, 如果for中驻留了多个matmul，则多条链上的所有op将均设置为for循环的block_id，不区分不同的matmul产生的value.

### 3. 算法流程

入口 `runOnOperation()`（`:499`）整体分四阶段。

#### Step 1 — 全局收集候选

1. `hasFallbackAttr(module)` 为真则直接返回（fallback 路径不走 fixpipe）。
2. 构建 `MemoryDependenceGraph memDepGraph`（别名分析）与 `ComputeBlockIdManager bm`。
3. `module.walk` 所有 `linalg::MatmulOp`，对每个调用 `matchFixpipePattern`；命中则把 `{matchedOps, targetBlockId}` 收入 `allMatchedPatterns`。

#### Step 2 — 单模式匹配 `matchFixpipePattern`（`:405`）

1. **单用户**：`matmulResult` 必须恰有一个 user。
2. **循环外提** `getFirstResultAfterLoop(matmulResult, *matmulOp.getDpsInits().begin())`（`:350`）：
   - matmul 的 DPS init 通常是循环携带的 block argument；本函数沿 `scf.for` 逐层向外走，要求"block arg 单用 → 定义 op → 结果单用 → yield 对应同一 arg 槽"，最终返回**逃逸最外层 for 的结果值**。
   - 这样累加型 matmul（每轮 yield、下轮再进）也能定位到循环外真正的消费链头。
3. **target block 解析**：`outerOutOp = outerOutValue.getDefiningOp()`；若 block_id == -1 调 `bm.markOpBlockId`，仍失败则不匹配；否则 `targetBlockId = bm.getBlockIdByOp(outerOutOp)`（注释保证 ≠ -1）。
4. **外层结果单用户**；若 `outerOutOp` 本身是 `linalg::MatmulOp` 则并入 `toMergeWithMatmul`（matmul→trunc→matmul 直连也允许）。
5. **分支匹配** `matmulUser`：
   - `getFixpipePreQuantMode(matmulUser).has_value()` → `isFixpipeCastPattern`。
   - 否则 `isValidMul(matmulUser, ...)` → `isFixpipeMulPattern`。
   - 都不满足则不匹配。

#### Step 2a — Cast 模式 `isFixpipeCastPattern`（`:216`）

```
matmul → arith.truncf/trunci → tensor.extract_slice → materialize_in_destination(subview(gm))
```
也可直连 `matmul → trunc → matmul`（trunc 结果是下游 matmul 的 DPS 输入即命中，只收 trunc）。
- 每步中间结果需单用户；`extract_slice` 后必须是 `MaterializeInDestinationOp`；最后 `isStoreToGM` 校验落 GM。

#### Step 2b — Mul 模式 `isFixpipeMulPattern`（`:294`）

```
matmul → arith.mulf/muli → (tensor.extract_slice)? → materialize_in_destination/store(subview(gm))
```
- `extract_slice` 可选；中间结果用 `getOneUserExceptMarkOp` 计数（忽略 `annotation.mark`）。
- `isValidMul`（`:144`）额外要求：mul 带 quant scale hint，且标量侧为 `isScalarLike` 或来自 `linalg.fill` 的标量初值；标量来源经 `transSource`（`:105`）回溯收集（遇 `tensor.extract` 停，仅收同 block op）。

#### Step 3 — 标量闭包收集与克隆

对每个 matched pattern：
1. `SplitIf::ScalarClosure closure{block, matchedOps, false}; closure.collect();` 收集同 block 内被 matched ops 依赖的标量产出 op，并入 `matchedOps`。
2. `cloneScalarOpsForCrossBlockUses(bm, matchedOps, targetBlockId)`（见 `Common.h:75`）：按逆拓扑遍历 matched 中的标量产出 op，若其结果被"pattern 外且不在 target 块"的 op 使用，则克隆一份把那些外部用重定向到 clone，原 op 留给 pattern 内部——避免统一 block_id 后产生跨块依赖环。

#### Step 4 — 应用 `applyFixpipeOpt`（`:453`）

1. 在 matchedOps 中定位 `linalg::MatmulOp`（找不到则用首个 op）作为基准。
2. `willCreateCycle(matchedOps, memGraph, targetBlockId, bm)` 环检测：若合入会成环则整组跳过（返回 false，外层仅打日志，不中断后续组）。
3. 对每个 matched op：
   - `scf` dialect：`walk` 内部所有 nestedOp，**跳过 `isSyncOp`**（Fence 必须保留独立 block_id），其余 `bm.updateBlockId(nestedOp, targetBlockId)` + `core_type = CUBE`。
   - 非 scf：直接 `bm.updateBlockId(op, targetBlockId)` + `core_type = CUBE`。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `allMatchedPatterns` | `SmallVector<pair<SetVector<Operation*>, int>>` | 全局收集的所有命中模式及其 targetBlockId |
| `matchedOps` / `toMergeWithMatmul` | `SetVector<Operation*>` | 单条命中链上的所有 op（含标量来源、view 链、mark 等） |
| `memDepGraph` | `MemoryDependenceGraph` | 内存依赖图，供 `willCreateCycle` 判环 |
| `bm` | `ComputeBlockIdManager` | block_id 管理/标记/更新 |
| `ScalarClosure` | `SplitIf::ScalarClosure` | 收集 matched ops 依赖的同 block 标量 op |
| `targetBlockId` | `int` | 命中链要合入的目标 block_id（matmul/外层 op 所在块） |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `transSource` | `:105` | 递归回溯标量来源，遇 `tensor.extract` 停，仅收同 block op |
| `hasQuantScaleCompileHint` | `:132` | 判定 mul 是否带 `enable_fast_tf32_mul` 的 mark，收 markOp |
| `isValidMul` | `:144` | 判定 mul op 是否构成量化模式（hint + 标量侧/fill 标量初值） |
| `isStoreToGM` | `:184` | 校验 store/materialize 的目标 view 源于 GM 并收 view 链 |
| `isFixpipeCastPattern` | `:216` | 匹配 matmul→trunc→extract_slice→materialize(GM) |
| `getOneUserExceptMarkOp` | `:267` | 计数非 MarkOp 的唯一 user，否则返回 nullopt |
| `isFixpipeMulPattern` | `:294` | 匹配 matmul→mulf/muli→(extract_slice)?→store(GM) |
| `getFirstResultAfterLoop` | `:350` | 沿 scf.for 外提，返回逃逸最外层循环的结果值 |
| `matchFixpipePattern` | `:405` | 主匹配：外提→block_id→分支 cast/mul |
| `applyFixpipeOpt` | `:453` | 环检测后批量设 block_id + core_type=CUBE |
| `runOnOperation` | `:499` | 入口：建图→遍历 matmul 收集→标量闭包+克隆→应用 |
| `createFixpipeOptPass` | `:553` | 构造 pass 实例 |

### 6. 安全机制

- **单用户约束**：matmul 结果、外提结果、trunc/extract_slice 结果均要求单用户（`getOneUserExceptMarkOp` 忽略 mark），避免融合后引入多消费分叉。
- **GM 校验**：`collectViewOpsAndCheckGlobalMemory` 保证落盘目标确实源自函数参数，否则不匹配。
- **循环外提安全性**：`getFirstResultAfterLoop` 每层都校验 block arg 单用、定义 op 即 arg user、结果单用、yield 槽对齐，任一不满足即停止外提。
- **环检测**：`applyFixpipeOpt` 合入前 `willCreateCycle` 判环，成环则整组跳过。
- **标量克隆解环**：`cloneScalarOpsForCrossBlockUses` 主动为跨块外部用克隆标量 op，从源头规避统一 block_id 后的跨块依赖环。
- **Fence 保留**：scf 块内 walk 时 `isSyncOp` 跳过，sync 保留独立 block_id 以保前后 Fence。


## 存在问题
1. 增强规格2

> 对于驻留在if中的matmul进行处理

2. 预计删除环检测
3. Fence 保留 即将删除

## 目标场景

1. kernel_sdpa_bwd_kv: 带有循环驻留的.
2. flash_attention_npu_v8_generalized.py::bwd_qkv_kernel
