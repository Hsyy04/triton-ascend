# MergeSmallBlockPass

## Overview

将算子数 ≤ `MIN_VF_SIZE`(3) 的小计算块合并到其上游(operand)或下游(user)计算块中，减少计算块碎片。

**Pass 参数**: `--merge-small-block`

**管线位置**: `IterVarOptPass` 之后，`FixpipeOptPass` 之前。

## Algorithm

```
for each Block in module:
    orderBlockIds = 按程序顺序收集当前Block中的计算块ID

    for each nowBlockId in orderedBlockIds:
        if (已合并) continue
        if ops.size() > MIN_VF_SIZE  continue

        upBlock = getUpBlock(nowBlockId)
        if upBlock存在:
            将nowBlockId中所有op的block_id改为upBlock
            continue

        downBlock = getDownBlock(nowBlockId)
        if downBlock存在:
            将downBlock中所有op的block_id改为nowBlockId
```

### getUpBlock(nowBlockId)

返回 `std::optional<int>`：

1. 遍历 nowBlock 中所有 op 的 operands
2. 过滤掉常量、block argument、无 block_id 的 operand
3. 若所有有效 operand 的 defining op 都来自**同一个**上游 block → 候选
4. 边界检查：若两个块之间的所有边界 op 都是 `isShapeChangeOp`，则不合并
5. 返回候选 block_id 或 `nullopt`

### getDownBlock(nowBlockId)

对称于 `getUpBlock`，通过遍历 op 的 users 而非 operands。

### 边界 Op 条件

收集 nowBlock 和候选 block 中所有存在跨块连接（operand 来自对方或 user 在对方块中）的 op。若这些边界 op 全部是 shape-change 操作，则跳过合并，避免形状变换操作与上下游计算混在一起。

### isShapeChangeOp(op)

```
return isa<linalg::BroadcastOp, linalg::ReduceOp>(op)
```

当前仅包含 `BroadcastOp` 和 `ReduceOp`，后续可按需扩展。

## Merge Direction

| 条件 | 行为 |
|---|---|
| upBlock 存在 | nowBlock → upBlock（向上游合并） |
| upBlock 不存在 && downBlock 存在 | downBlock → nowBlock（向下游合并） |

上游优先策略：小计算块优先吸收到其数据来源块中。

## Key Parameters

| 参数 | 值 | 含义 |
|---|---|---|
| `MIN_VF_SIZE` | 3 | 块中 op 数 ≤ 此阈值才考虑合并 |
| 迭代策略 | 单次遍历 | 按程序顺序动态合并，不重复迭代 |
| 循环检测 | 不需要 | 条件保证了无环 |

## Files

| 文件 | 说明 |
|---|---|
| `lib/DynamicCVPipeline/ComputeBlockOpt/MergeSmallBlockPass.cpp` | Pass 实现 |
| `include/DynamicCVPipeline/ComputeBlockOpt/Passes.h` | 工厂函数声明 |
| `lib/DynamicCVPipeline/ComputeBlockOptPass.cpp` | 管线注册 |
