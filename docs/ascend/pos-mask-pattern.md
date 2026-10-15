# PosMaskPattern：将注意力掩码计算归入分数使用者块

## 问题与目标

mask_in 和 mask_out_bitmask 的原始 ttadapter 经完整流程
`--ssbuf-standardize-op --plan-compute-block --compute-block-opt` 后，掩码生成和偏置 select 位于 VECTOR block 15，分数 addf 位于 VECTOR block 16。两者之间传递的是已展开的二维掩码偏置。

本设计扩展现有 `PosMaskPatternPass.cpp`，将独占、纯计算的掩码链划入分数使用者块。保留原来的位置局部模式，不引入新 annotation，不修改 Python kernel、后端开关或其他优化 pass。

## 现有行为必须保留

原模式为 VECTOR broadcast 恰有两个使用者，分别是 eq、sle，每个比较唯一使用者为 extui；五个操作同属一个计算块和 MLIR Block。若两个 extui 的直接使用者具有共同目标块，且归并不会产生环，则更新五个操作的 block_id。

这一规则原样执行，新增规则在其后执行。没有 attention 消费者的旧模式仍然可以归并；不满足旧规则的输入仍遵守旧规则的拒绝行为。

## 新增的消费锚点

识别 `arith.addf(scaled_matmul, bias)`，支持交换两侧：

- scaled_matmul 是 arith.mulf，其中一侧直接定义于 linalg.matmul。
- bias 是 arith.select，两个分支为浮点零和负的非 NaN 常量，支持交换极性及负无穷。
- 常量同时支持标量/稠密 splat 常量，以及由这些标量常量生成的 linalg.fill；非均匀常量不匹配。
- bias 与 addf 在同一 MLIR Block，分别有有效且不同的计算块编号；消费者必须是 VECTOR。

锚点说明该计算为分数的按谓词偏置，不依赖 SSA 名称、Python 变量名、文件名或固定 tile 大小。它是针对支持的 IR 结构的优化契约，不声称覆盖所有语义等价的 mask。

## 两种生成模式

### 比较组合

`((qpos <= kpos) & ((qtag == ktag) | (ktag == 0))) | (qpos == kpos)`

对应广播、cmpi、extui、andi/ori，再经整数非零判断和偏置 select。新规则允许 i1/i8/i32 等整型张量，不要求掩码全程是 i1。至少包含非 ne 比较与逻辑组合。

### 位掩码解码

紧凑字节加载 -> broadcast -> reshape -> shrui -> andi 1 -> 有效列提取并插入零张量 -> 非零判断 -> 偏置 select。

只有右移结果与常量 1 进行 andi 才作为位提取证据；普通位与不能单独构成该模式。

## 反向收集和安全边界

从 bias 出发，反向遍历同一 MLIR Block、同一源计算块内的 VECTOR 操作。允许的生成操作是 cmpi、extui、trunci、andi、ori、shrui、linalg.broadcast，以及 tensor expand/collapse/reshape/extract_slice/insert_slice。

加载、copy、to_tensor、alloc、循环、分支、任意 region、标量地址和尺寸计算均不搬动。它们仍作为输入保留在原块。常量和构造器也保留；因此本优化不保证消除掩码加载块或所有跨块传输，只消除满足条件的展开与偏置计算到消费端的传输。

对于候选操作，若其结果有候选集合外且不在目标块中的使用者，则从候选集合剔除该操作。重复剔除直到不再变化，保留共享输入，不克隆操作。如果 bias 自身被剔除，或剩余集合没有支持的生成模式证据，则拒绝此次归并。

最后调用已有 willCreateCycle：它检查同步分段及数据/内存/执行依赖下的块级环。通过后只用 ComputeBlockIdManager 更新候选 block_id，物理重排由完整 ComputeBlockOpt 后续 Reorder 完成。所有操作和数据计算保持不变，已有 MemoryDependenceGraph 因未增删或重排操作而继续有效。

第一版仅合并同一源块内可收集的计算，不跨多个来源块递归改写，不移动加载，不跨控制流边界，不以此修复 kernel 本身的边界正确性问题。

## UT 与验收

新增 `Conversion/General/DynamicCVPipeline/ComputeBlockOpt/pos_mask_pattern.mlir`，使用 split-input-file 隔离计算块编号。覆盖：

1. 原位置局部模式成功归并，无 attention 锚点。
2. 比较逻辑组合归并。
3. bitmask 取位及边界 slice 归并。
4. add 两侧交换、select 极性交换。
5. 共享比较输入保留，独占尾部归并。
6. 正偏置拒绝。
7. 非 matmul 分数拒绝。
8. 偏置在多个计算块使用时拒绝。
9. fill 常量偏置。
10. 跨外部同步拒绝。
11. 右移后与非 1 常量位与时拒绝。
12. 掩码的额外跨块依赖保留，不强制合并。
13. 原位置模式使用 sge 而非 sle 时不匹配。

真实样本验收使用已取得的两种实现 L128/L130 四份 ttadapter，统一运行完整三个 pass，检查生成逻辑、最终偏置 select 和消费 addf 的 block_id 相同。此检查与合成 UT 分开记录。

编译和 IR 验证不能代替重新编包后的 NPU 数值或性能验证。原 mask_in L130 在修复前已有数值失败，该问题不在本次计算块归并的修改范围内。

## 本次验证结果

- 分支：fix/pos-mask-consumer-fusion。
- triton-opt 编译成功。
- ComputeBlockOpt lit 目录 20/20 通过，其中新增文件包含 13 个独立 IR 场景。
- mask_in / mask_out_bitmask 的 L128 和 L130，共四份真实 IR，完整三个 pass 均成功；掩码代表性生成操作、偏置 select 与消费 addf 均归入 VECTOR block 16。
- 未重新编 TA 包或进行修复后的 NPU 数值及性能测试；本次完成的是编译器 IR 回归验证。
- 原仓库已有未提交改动保留；本次仅新增设计文档、UT，并修改 PosMaskPatternPass.cpp，未提交或推送。
