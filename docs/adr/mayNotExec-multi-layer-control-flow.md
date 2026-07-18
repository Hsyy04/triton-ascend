# ADR: 多层嵌套控制流的 mayNotExec 处理

## 背景

当前 `SplitMatmulPattern.cpp` 中的 `handleMayNotExec` 函数仅处理单层 `scf::ForOp` 的情况。当 matmul 嵌套在多层控制流（for/if）中时，需要新的处理方案。

## 设计目标

统一处理多层嵌套控制流的 `mayNotExec` 情况，包括：
- 单层或多层 `scf::ForOp`
- 单层或多层 `scf::IfOp`
- `scf::ForOp` 与 `scf::IfOp` 混合嵌套

## 核心概念

### 执行条件

对于嵌套的控制流操作，matmul 的执行条件是各层控制流执行条件的 AND：

- `scf::ForOp` 执行条件：`ub > lb`
- `scf::IfOp` 执行条件：`condition == true`

### 三步处理流程

1. **计算执行条件**：遍历 `controlFlowOps`，生成各层执行条件并 AND
2. **插入 `scf.if` wrapper**：then 块 yield 最外层控制流返回值，else 块 yield fill_zero
3. **替换使用者**：将最外层控制流返回值的使用者替换为 wrapper 结果

## 模块接口

### 输入

- `SmallVector<Operation*> controlFlowOps`：嵌套控制流 op（从外到内）
- `linalg::MatmulOp matmulOp`：目标 matmul
- `SplitInfo splitInfo`：分割信息
- `OpBuilder &builder`：构建器

### 输出

- `Value wrapped`：包装后的结果（`scf.if` 的结果）

### 插入位置

`%wrapped` 及相关 op 插入在 `%for1` 之后、原 `%for1` 使用者之前。

## 三种使用场景

### 场景 1：result 存回全局内存

#### 输入 IR

```mlir
%for1 = scf.for %i = %lb1 to %ub1 step %c1 iter_args(%arg1 = %bias) {
  %for2 = scf.for %j = %lb2 to %ub2 step %c1 iter_args(%arg2 = %arg1) {
    %if_result = scf.if %cond {
      %matmul = linalg.matmul ... outs(%arg2)
      scf.yield %matmul
    } else {
      scf.yield %arg2
    }
    scf.yield %if_result
  }
  scf.yield %for2
}
%store = bufferization.materialize_in_destination %for1 in %gm_buffer
```

#### 输出 IR

```mlir
%for1 = scf.for %i = %lb1 to %ub1 step %c1 iter_args(%arg1 = %bias) {
  %for2 = scf.for %j = %lb2 to %ub2 step %c1 iter_args(%arg2 = %arg1) {
    %if_result = scf.if %cond {
      %matmul = linalg.matmul ... outs(%arg2)
      scf.yield %matmul
    } else {
      scf.yield %arg2
    }
    scf.yield %if_result
  }
  scf.yield %for2
}

%exec_cond1 = arith.cmpi sgt, %ub1, %lb1 : index
%exec_cond2 = arith.cmpi sgt, %ub2, %lb2 : index
%and1 = arith.andi %exec_cond1, %exec_cond2 : i1
%final_exec_cond = arith.andi %and1, %cond : i1

%wrapped = scf.if %final_exec_cond {
  scf.yield %for1
} else {
  %fill_zero = linalg.fill ins(%zero) outs(%empty)
  scf.yield %fill_zero
}

%store = bufferization.materialize_in_destination %wrapped in %gm_buffer
```

#### 下游模块影响

下游模块需要在 `%wrapped` 之后插入 `fixpipe` → `MTE3` 同步点。

---

### 场景 2：result 被另一个 matmul 的 C 使用（L0C -> L0C）

#### 输入 IR

```mlir
%for1 = scf.for %i = %lb1 to %ub1 step %c1 iter_args(%arg1 = %bias) {
  %for2 = scf.for %j = %lb2 to %ub2 step %c1 iter_args(%arg2 = %arg1) {
    %if_result = scf.if %cond {
      %matmul = linalg.matmul ... outs(%arg2)
      scf.yield %matmul
    } else {
      scf.yield %arg2
    }
    scf.yield %if_result
  }
  scf.yield %for2
}
%matmul2 = linalg.matmul ... outs(%for1)
```

#### 输出 IR

```mlir
%for1 = scf.for %i = %lb1 to %ub1 step %c1 iter_args(%arg1 = %bias) {
  %for2 = scf.for %j = %lb2 to %ub2 step %c1 iter_args(%arg2 = %arg1) {
    %if_result = scf.if %cond {
      %matmul = linalg.matmul ... outs(%arg2)
      scf.yield %matmul
    } else {
      scf.yield %arg2
    }
    scf.yield %if_result
  }
  scf.yield %for2
}

%exec_cond1 = arith.cmpi sgt, %ub1, %lb1 : index
%exec_cond2 = arith.cmpi sgt, %ub2, %lb2 : index
%and1 = arith.andi %exec_cond1, %exec_cond2 : i1
%final_exec_cond = arith.andi %and1, %cond : i1

%wrapped = scf.if %final_exec_cond {
  scf.yield %for1
} else {
  %fill_zero = linalg.fill ins(%zero) outs(%empty)
  scf.yield %fill_zero
}

%matmul2 = linalg.matmul ... outs(%wrapped)
```

#### 下游模块影响

待定。

---

### 场景 3：result 被 vector 指令使用

#### 输入 IR

```mlir
%for1 = scf.for %i = %lb1 to %ub1 step %c1 iter_args(%arg1 = %bias) {
  %for2 = scf.for %j = %lb2 to %ub2 step %c1 iter_args(%arg2 = %arg1) {
    %if_result = scf.if %cond {
      %matmul = linalg.matmul ... outs(%arg2)
      scf.yield %matmul
    } else {
      scf.yield %arg2
    }
    scf.yield %if_result
  }
  scf.yield %for2
}
%result = arith.mulf %for1, %other_val
```

#### 输出 IR

```mlir
%for1 = scf.for %i = %lb1 to %ub1 step %c1 iter_args(%arg1 = %bias) {
  %for2 = scf.for %j = %lb2 to %ub2 step %c1 iter_args(%arg2 = %arg1) {
    %if_result = scf.if %cond {
      %matmul = linalg.matmul ... outs(%arg2)
      scf.yield %matmul
    } else {
      scf.yield %arg2
    }
    scf.yield %if_result
  }
  scf.yield %for2
}

%exec_cond1 = arith.cmpi sgt, %ub1, %lb1 : index
%exec_cond2 = arith.cmpi sgt, %ub2, %lb2 : index
%and1 = arith.andi %exec_cond1, %exec_cond2 : i1
%final_exec_cond = arith.andi %and1, %cond : i1

%wrapped = scf.if %final_exec_cond {
  scf.yield %for1
} else {
  %fill_zero = linalg.fill ins(%zero) outs(%empty)
  scf.yield %fill_zero
}

%result = arith.mulf %wrapped, %other_val
```

#### 下游模块影响

下游模块需要在 `%wrapped` 之后插入 `fixpipe` → `vector` 同步点。

---

## 实现细节

### 执行条件计算

```cpp
Value computeExecCondition(const SmallVector<Operation*> &controlFlowOps, 
                           OpBuilder &builder, Location loc) {
    Value finalCond;
    for (auto *op : controlFlowOps) {
        Value execCond;
        if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            execCond = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sgt, 
                forOp.getUpperBound(), forOp.getLowerBound());
        } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            execCond = ifOp.getCondition();
        }
        
        if (!finalCond) {
            finalCond = execCond;
        } else {
            finalCond = builder.create<arith::AndIOp>(loc, finalCond, execCond);
        }
    }
    return finalCond;
}
```

### tensor.empty 动态尺寸

从 `%for1` 结果类型推导：

```cpp
auto outputType = cast<RankedTensorType>(for1.getType());
SmallVector<Value> dynamicSizes;
for (int64_t i = 0; i < outputType.getRank(); ++i) {
    if (outputType.isDynamicDim(i)) {
        dynamicSizes.push_back(builder.create<tensor::DimOp>(loc, for1, i));
    }
}
auto emptyOp = builder.create<tensor::EmptyOp>(loc, outputType, dynamicSizes);
```

### 零值常量

从 `%for1` 结果类型的元素类型推导：

```cpp
auto elmType = cast<RankedTensorType>(for1.getType()).getElementType();
Value zeroValue;
if (auto floatType = dyn_cast<FloatType>(elmType)) {
    APFloat zeroAPFloat = APFloat::getZero(floatType.getFloatSemantics());
    zeroValue = builder.create<arith::ConstantFloatOp>(loc, zeroAPFloat, floatType);
} else if (auto intType = dyn_cast<IntegerType>(elmType)) {
    zeroValue = builder.create<arith::ConstantIntOp>(loc, 0, intType);
}
```

## 约束

- `controlFlowOps` 的条件由上游模块确保可静态确定
- `%for1` 只有单一使用者
- `%bias` 与 `%for1` 结果类型一致