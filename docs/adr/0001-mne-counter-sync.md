# ADR-0001: MNE Counter Sync — may-not-execute matmul correctness

2026-08-02 · Status: accepted

---

## 1. 问题描述

Ascend NPU 硬件乘法计算单元（Cube）执行 `L0A[a] * L0A[b] + L0C → L0C`。
指令完成步骤：

1. 初始化 L0C 地址（可选，由控制位决定）
2. L0A 上的两个数据相乘
3. 加上 L0C 上的数据
4. 结果放入 L0C

若这条乘法指令不执行，L0C 地址上的数据就是随机数。而从 L0C 写回 GM 的硬件指令
（fixpipe）只负责搬运，无法感知计算单元是否执行，依旧会搬运该地址——导致精度出错。

需要在编译器中正确变换 `linalg.matmul(a, b, c)` 实现正确且高效的指令序列。

### 场景分类

| 场景 | 数据流向 | 错误表现 | 方案 |
|------|---------|---------|------|
| **L0C→UB** | fixpipe → Vector to_tensor → Vector 后续 op | Vector 消费者读到垃圾数据 | Part A: scf.if 插在 matmul 结果和 addf 之间 |
| **L0C→GM** | fixpipe → bufferization.materialize_in_destination | 垃圾数据写入 HBM | Part B: 双 scf.if（无 else）守卫 store |
| **L0C→L0C** | Cube chain 直连（无 fixpipe） | 下游硬件指令选项可处理 | 无操作 |
| **L0C→L1** | fixpipe → L1 → 下个 matmul | — | 当前方案直接拆分，预留 Part C |

---

## 2. 架构设计

如果可能不执行的for循环在main_loop之内, 通过ssbuffer的条件控制同步
如果在main_loop之外, 通过pipe_s控制同步

### 2.1 Pass 编排

以下按 DynamicCVPipeline 执行顺序，仅列出本方案修改的 pass 及方案所处位置：

| 顺序 | Pass | 职责 | 修改 |
|------|------|------|------|
| 1 | **SplitMatmulPattern** (StandardizeOp 内) | 检测 mayNotExec + fixpipeDst；split 拆分 matmul+C；插入 counter + scf.if | **本方案核心** |
| 2 | PlanComputeBlock | 分类 ops 为 CUBE/VECTOR，分配 block_id | OpClassifier / PlanCubeBlock 适配 |
| 3 | InterCoreTransferAndSync / MarkMainLoop | 插入 fixpipe / copy / memory sync | 不修改，作为本方案同步插入的定位参考 |
| 4 | **AddMNECounterSync** | 识别 `ssbuffer.mne_counter`，在main_loop外插入 PIPE_S 同步 | **本方案新增** |
| 5 | `…` | 后续 pass（AllocMultiCache etc.） | 不修改 |
| 6 | **AddControlFlowCondition** | 识别可能不执行的依赖关系, 并添加对应的控制条件 | **本方案新增**  |

### 2.2 IR 变化过程（端到端）


#### Part A: L0C→UB

```mlir
// === Input ===
%for_result = scf.for %i = %lb to %ub step %c1 iter_args(%bias_arg = %bias) {
  %m = linalg.matmul ins(%a, %b) outs(%bias_arg)
  scf.yield %m
}
use(%for_result)    // Vector op consumes result (UB)


// === Output ===
// CUBE 侧:
%counter = memref.alloc() : memref<i32, SSBUF(11)> {ssbuffer.mne_counter}
memref.store %c0, %counter[]

%for_result = scf.for %i = %lb to %ub step %c1 iter_args(%bias_arg = %zeroFill) {
  %m = linalg.matmul ins(%a, %b) outs(%bias_arg)
  memref.store %c1, %counter[] {core_type = "CUBE"}
  scf.yield %m
}
SyncBlockSet(CUBE, PIPE_S, PIPE_S)


// VECTOR 侧:
%counter = memref.alloc() : memref<i32, SSBUF(11)> {ssbuffer.mne_counter}
SyncBlockWait(VECTOR, PIPE_S, PIPE_S)
%cnt = memref.load %counter[] {core_type = "VECTOR"}
%has_exec = arith.cmpi ne, %cnt, %c0

%guarded = scf.if %has_exec {
  scf.yield %for_result
} else {
  scf.yield %zero_fill
}
use(%guarded)
```

#### Part B: L0C→GM

```mlir
// === Input ===
%for_result = scf.for %i = %lb to %ub step %c1 iter_args(%c_arg = %c) {
  xxx
  %m = linalg.matmul ins(%a, %b) outs(%c_arg)
  scf.yield %m
}
bufferization.materialize_in_destination %for_result in %gm_buf


// === Output ===
// CUBE 侧:
%counter = memref.alloc() : memref<i32, SSBUF(11)> {ssbuffer.mne_counter}
memref.store %c0, %counter[]

%for_result = scf.for %i = %lb to %ub step %c1 iter_args(%c_arg = %c) {
  %m = linalg.matmul ins(%a, %b) outs(%c_arg)
  memref.store %c1, %counter[] {core_type = "CUBE"}
  scf.yield %m
}

SyncBlockwait(CUBE, mte3, fixpipe)
SyncBlockSet(CUBE, fixpipe, mte3)
%cnt_cube = memref.load %counter[] {core_type = "CUBE"}
%has_exec = arith.cmpi ne, %cnt_cube, %c0
scf.if %has_exec {
  bufferization.materialize_in_destination %for_result in %gm_buf
    {core_type = "CUBE"}
}

// VECTOR 侧:

for(){
  xxx
}

SyncBlockWait(VECTOR, fixpipe, mte3)
SyncBlockset(VECTOR, mte3, fixpipe)
%counter = memref.alloc() : memref<i32, SSBUF(11)> {ssbuffer.mne_counter}
%cnt_vec = memref.load %counter[] {core_type = "VECTOR"}
%not_exec = arith.cmpi eq, %cnt_vec, %c0
scf.if %not_exec {
  %zero_store = linalg.fill ...
  bufferization.materialize_in_destination %zero_store in %gm_buf
    {core_type = "VECTOR"}
}
```

#### in main_loop
以存储到gm为例
```mlir

// CUBE
%mne_code1 = 0
%mne_code2 = 0
main_loop{
  %mne_cond1 = %mne_code eq 0
  %cond = %origin_cond and %mne_cond1 and %mne_cond2
  if (%cond){
    %for_result = scf.for %i = %lb to %ub step %c1 iter_args(%c_arg = %c) {
      %m = linalg.matmul ins(%a, %b) outs(%c_arg)
      memref.store %c1, %counter[] {core_type = "CUBE"}
      scf.yield %m
    }
    memref.store 1 into mne_code1
    %cnt_cube = memref.load %counter[] {core_type = "CUBE"}
    %has_exec = arith.cmpi ne, %cnt_cube, %c0
    scf.if %has_exec {
      bufferization.materialize_in_destination %for_result in %gm_buf
        {core_type = "CUBE"}
    }
  }


}

//VECTOR
main_loop{
  ...
  %mne_res_cond = %mne_code2 eq 1
  %cond = %origin_cond and %mne_res_cond
  if (%cond){
    %cnt_cube = memref.load %counter[] {core_type = "vector"}
    %has_exec = arith.cmpi ne, %cnt_cube, %c0
    scf.if %has_exec {
      bufferization.materialize_in_destination %for_result in %gm_buf
        {core_type = "vector"}
    }
    memref.store 0 into mne_code2
  }
}

```


## 3. 更改设计

### 3.1 SplitMatmulPattern.cpp

#### 3.1.1 执行顺序

```
matchAndRewrite:
  ┌─ shouldSplit()
  │   ├─ searchInArgsChain() → mayNotExec, outerInValue, outerOutValue
  │   └─ getFixpipeDst(outerOutValue) → fixpipeDst
  │
  ├─ [Step 1] 先拆分：
  │   若 shouldSplit → splitMatmul(matmulOp, rewriter, splitInfo, forOp)
  │     ├─ 零偏置 + 新 matmul + addf + replaceUsesWithIf
  │     └─ splitInfo.fixpipeDst = fixpipeDst::UB   // addf 后强制
  │
  └─ [Step 2] 后插入 MNE guard：
      若 mayNotExec && forOp:
        ├─ fixpipeDst::GM → insertMNEGuardStore()  // Part B
        └─ fixpipeDst::UB → insertMNEGuardUB()     // Part A
           (L0C/L1 暂不 guard)
```

#### 3.1.2 匹配矩阵

| shouldSplit | splitMatmul 后 fixpipeDst | MNE guard |
|------------|--------------------------|-----------|
| ✓ | **UB**（addf 强制） | Part A |
| ✗ | UB | Part A |
| ✗ | GM | Part B |
| ✗ | L0C | 无 |
| ✗ | L1 | 预留 Part C |

#### 3.1.3 IR 变化

**Part A（L0C→UB）**

```
Input:
  %for_result = scf.for ... iter_args(%bias_arg = %bias) {
    %m = linalg.matmul ins(%a, %b) outs(%bias_arg)
    scf.yield %m
  }
  // use(%for_result)

Output:
  %counter = memref.alloc() : memref<i32, SSBUF(11)> {ssbuffer.mne_counter}
  memref.store %c0, %counter[]

  %for_result = scf.for ... iter_args(%bias_arg = %zeroFill) {
    %m = linalg.matmul ins(%a, %b) outs(%bias_arg)
    memref.store %c1, %counter[]
    scf.yield %m
  }

  %cnt = memref.load %counter[]
  %has_exec = arith.cmpi ne, %cnt, %c0

  %guarded = scf.if %has_exec {
    scf.yield %for_result
  } else {
    scf.yield %zero_fill
  }
  %addf = arith.addf %guarded, %bias
  // use(%addf)  ← splitMatmul 的 replaceUsesWithIf 完成
```

**Part B（L0C→GM）**

```mlir
Input:
  %for_result = scf.for ... iter_args(%c_arg = %c) {
    %m = linalg.matmul ins(%a, %b) outs(%c_arg)
    scf.yield %m
  }
  bufferization.materialize_in_destination %for_result in %gm_buf

Output:
  %counter = memref.alloc() : memref<i32, SSBUF(11)> {ssbuffer.mne_counter}
  memref.store %c0, %counter[]

  %for_result = scf.for ... iter_args(%c_arg = %c) {
    %m = linalg.matmul ins(%a, %b) outs(%c_arg)
    memref.store %c1, %counter[]
    scf.yield %m
  }

  %cnt_cube = memref.load %counter[]
  %has_exec = arith.cmpi ne, %cnt_cube, %c0
  scf.if %has_exec {
    bufferization.materialize_in_destination %for_result in %gm_buf
  }

  %cnt_vec = memref.load %counter[]
  %not_exec = arith.cmpi eq, %cnt_vec, %c0
  scf.if %not_exec {
    %zero_store = linalg.fill ...
    bufferization.materialize_in_destination %zero_store in %gm_buf
  }
```

#### 3.1.4 函数清单

| 函数 | 职责 |
|------|------|
| `getFixpipeDst(outValue)` → fixpipeDst | 遍历 traceChainUser 判断 UB/GM/L1/L0C |
| `splitMatmul(…, forOp)` → LogicalResult | 零偏置+新matmul+addf+替换 uses。若 mayNotExec 则 fixpipeDst=UB |
| `insertMNEGuardUB(…, forOp)` → void | Part A：counter + scf.if，修改 addf operand |
| `insertMNEGuardStore(…, forOp)` → void | Part B：counter + 双 memref.load(Cube/Vector) + 双 scf.if 守卫 store |

#### 3.1.5 详细更改

**1) fixpipeDst 枚举 & SplitInfo 扩展**

```cpp
// 新增枚举
enum class fixpipeDst { UB, GM, L1, L0C, Unknown };

// SplitInfo 新增字段
struct SplitInfo {
  bool mayNotExec;
  Value outerInValue;
  Value outerOutValue;
  bool shouldSplit;
  fixpipeDst fixpipeDst;  // ← 新增
};
```

**2) `getFixpipeDst(outValue)` 函数**

通过 `traceChainUser` 依次匹配 store-to-GM(GM)、matmul-C(L0C)、matmul-AB(L1)，
默认返回 UB。消除 `shouldSplitByOutput` 及 `isOutputFilter` 中的三次独立 trace。

**3) addf 创建后 fixpipeDst 变为 UB**

```cpp
// splitMatmul 内部，addf 创建后：
splitInfo.fixpipeDst = fixpipeDst::UB;
```

无需判断 `mayNotExec`——任何 shouldSplit 场景 addf 都改变数据流向。

**4) `shouldSplitByOutput` 简化**

原三次 `traceChainUser`（storeToGM / usedByL0C / usedByL1）替换为读 `splitInfo.fixpipeDst`：

```cpp
static bool shouldSplitByOutput(…, fixpipeDst dst) {
  if (dst == fixpipeDst::GM && isOutputFilter(…)) return true;
  if (dst == fixpipeDst::L0C) return true;
  if (dst == fixpipeDst::L1) return true;
  return false;
}
```

**5) `handleMayNotExec` 不再强制 shouldSplit**

```diff
- return SplitInfo{true, initVal, result, true};
+ return SplitInfo{true, initVal, result, false};
```

**6) `splitMatmul` 新增 forOp 参数**

```cpp
static LogicalResult splitMatmul(linalg::MatmulOp matmulOp,
                                 PatternRewriter &rewriter,
                                 SplitInfo &splitInfo,
                                 scf::ForOp forOp = nullptr);
```

不再从 `splitInfo.outerOutValue.getDefiningOp()` 推导 forOp（因 outerOutValue 可能已被 insertMNEGuard 修改）。

**7) `insertMNEGuardUB` — Part A**

- 创建 counter memref（SSBUF addr=11, `ssbuffer.mne_counter`）
- 循环前 `memref.store %c0`（初始化）
- 循环内 matmul 后 `memref.store %c1`（标注 `core_type="CUBE"`）
- 循环后 `memref.load`（标注 `core_type="VECTOR"`）
- 创建 `arith.cmpi ne` → `%has_exec`
- 创建 `scf.if %has_exec { yield %for_result } else { yield %zeroFill }` → `%guarded`
- 找到 splitMatmul 创建的 addf（`kAddFromMatmul` 属性），将 operand 从 `%for_result` 改为 `%guarded`
- 设置 `forOp.kHIVMMatmulLimitedInCubeAttr`

**8) `insertMNEGuardStore` — Part B**

- 创建 counter memref（SSBUF addr=11, `ssbuffer.mne_counter`）
- 循环前 `memref.store %c0`（初始化）
- 循环内 matmul 后 `memref.store %c1`（标注 `core_type="CUBE"`）
- Cube 侧：`memref.load`（`core_type="CUBE"`）→ `arith.cmpi ne` → `scf.if` 守卫 CUBE store
- Vector 侧：`memref.load`（`core_type="VECTOR"`）→ `arith.cmpi eq` → `scf.if` 守卫 VECTOR zero-fill store
(Set/Wait 由 AddMNECounterSync 统一插入)
- 收集 `outerOutValue` 的所有 `bufferization::MaterializeInDestinationOp` 用户，分别包裹到对应的 if

**9) `matchAndRewrite` 最终流程**

```cpp
  auto forOp = dyn_cast_if_present<scf::ForOp>(
      splitInfo.outerOutValue.getDefiningOp());

  if (splitInfo.shouldSplit) {
    if (failed(splitMatmul(matmulOp, rewriter, splitInfo, forOp)))
      return failure();
  }

  if (splitInfo.mayNotExec && forOp) {
    if (splitInfo.fixpipeDst == fixpipeDst::GM) {
      insertMNEGuardStore(matmulOp, rewriter, splitInfo, forOp);
    } else if (splitInfo.fixpipeDst == fixpipeDst::UB) {
      insertMNEGuardUB(matmulOp, rewriter, splitInfo, forOp);
    }
  }
```

---

### 3.2 PlanComputeBlock

#### 3.2.1 OpClassifier.cpp

**问题**：matmul 结果现由 Part A 的 `scf.if` 包裹或 Part B 的 `scf.if` 守卫，
在找种子（seed）识别 store 链时，需要穿透 for 循环嵌套和 scf.if 追踪真实数据流。

**更改**：在 seed 追踪逻辑中增加对 `scf.if` then-branch yield 的穿透：

- matmul 结果 → for-loop result → scf.if then-branch → yield → store
- 当前 classifier 只能追踪 for-loop iter_args，需扩展为同时处理 scf.if 的 result → yield 关系

具体：在 `traceChainUser` 调用中捕获 scf.if 的 result → then-yield 映射，
保持 store 继承 matmul 的 CUBE 分类。

#### 3.2.2 PlanCubeBlock.cpp

**问题**：`matchSeed` 识别 matmul 的 store 消费者时存在相同的穿透需求。

**更改**：与 OpClassifier 保持一致——`matchSeed` 追踪 matmul 结果通过 scf.if 向 store 的路径。
确保 if-block 内的 store 和 if-block 外经由 result 连接的 store 都被正确识别为 CUBE 消费者。

---

### 3.3 AddMNECounterSync (区分一下main_loop内外)

**作用**：为 Part A / Part B 统一插入 PIPE_S 同步。Set 在 Cube 侧 op 前，Wait 在 Vector load 前。

#### 3.3.1 执行顺序

```
AddMNECounterSyncPass::runOnOperation:
  walk 所有 memref::AllocOp:
    若没有 sssbuffer.mne_counter 属性 → 跳过
    1. 找 Cube 侧 memref::StoreOp (core_type="CUBE")
    2. 找所有 memref::LoadOp (core_type="CUBE" 或 "VECTOR")
    3. 从 CubeStore 上溯 parent chain 找外围 scf::ForOp
    4. 插入同步（Part A / B 统一，无需区分）:
       SyncBlockSet(CUBE, PIPE_S, PIPE_S) — for 循环后
       SyncBlockWait(VECTOR, PIPE_S, PIPE_S) — VECTOR load 前
```

#### 3.3.2 IR 变化

```
Input (来自 InterCoreTransferAndSync 后):
  %counter = memref.alloc() ... {ssbuffer.mne_counter}
  memref.store %c0, %counter[]          // Vector init

  scf.for %i = %lb to %ub step %c1 {    // Cube main loop
    %m = linalg.matmul ins(%a, %b) outs(%bias_arg)
    memref.store %c1, %counter[] {core_type = "CUBE"}
    scf.yield %m
  }

  %cnt = memref.load %counter[] {core_type = "VECTOR"}
  %has_exec = arith.cmpi ne, %cnt, %c0
  ...

────────────────────────────────────────
AddMNECounterSync 处理后:
────────────────────────────────────────

  %counter = memref.alloc() ... {ssbuffer.mne_counter}
  memref.store %c0, %counter[]

  scf.for %i = %lb to %ub step %c1 {
    %m = linalg.matmul ...
    memref.store %c1, %counter[] {core_type = "CUBE"}
    scf.yield %m
  }
  SyncBlockSet(CUBE, PIPE_S, PIPE_S)    // ① 循环后：Cube 数据就绪

  SyncBlockWait(VECTOR, PIPE_S, PIPE_S) // ② load 前：Vector 等 Cube
  %cnt = memref.load %counter[] {core_type = "VECTOR"}
  %has_exec = arith.cmpi ne, %cnt, %c0
  ...
```

#### 3.3.3 详细更改

```cpp
void AddMNECounterSyncPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;

  auto cubeAttr = TCoreTypeAttr::get(ctx, TCoreType::CUBE);
  auto vecAttr  = TCoreTypeAttr::get(ctx, TCoreType::VECTOR);
  auto pipeSAttr = PipeAttr::get(ctx, PIPE::PIPE_S);

  module.walk([&](memref::AllocOp allocOp) {
    if (!allocOp->hasAttr(CVPipeline::kMNECounter)) return;

    // 1. 找 Cube store 和 Vector load
    memref::StoreOp cubeStore = findStoreOp(allocOp, "CUBE");
    memref::LoadOp  vecLoad  = findLoadOp(allocOp, "VECTOR");
    if (!cubeStore || !vecLoad) return;

    // 2. 找外围 scf::ForOp
    auto forOp = cubeStore->getParentOfType<scf::ForOp>();
    if (!forOp) return;

    // 3. 插入 PIPE_S 同步
    int flagId = flagIdCounter++;
    auto flag = Builder(ctx).getI64IntegerAttr(flagId);

    // ① for 后 Set
    builder.setInsertionPointAfter(forOp);
    builder.create<SyncBlockSetOp>(loc, cubeAttr, pipeSAttr, pipeSAttr, flag);

    // ② load 前 Wait
    builder.setInsertionPoint(vecLoad);
    builder.create<SyncBlockWaitOp>(loc, vecAttr, pipeSAttr, pipeSAttr, flag);
  });
}
```
