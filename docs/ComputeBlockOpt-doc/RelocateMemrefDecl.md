# RelocateMemrefDecl

## Overview

### 1. 目标

将使用跨同步操作（sync op）的 memref 声明重新定位到消费者所在的 block，并拉取依赖集群一起移动，以确保每个声明与其使用在同一个同步屏障侧，从而在 SplitDataflow 降低屏障之前完成重分区。
当用户写的dsl中包含同步语句时，如果存在跨计算块、且两个计算块间有同步op的memref依赖，这个pass将会把定义这个memref的op移动到下面的计算块。


### 2. 规格

1. 仅处理 memref 类型的操作数。
2. 当 memref 的生产者与消费者之间存在同步操作时，触发重定位。
3. 消费者必须已被分配 block_id（`targetId != -1`）。
4. 构建前向闭包：包括生产者及其所有不在目标 block 中的使用者（递归），以及必要的后向依赖（memref 生产者），确保移动后 SSA 支配关系保持。
5. 闭包中不能包含 terminator 操作。
6. 移动时要求所有操作数支配插入点（anchor），即最早在目标 block 中跨同步的使用。
7. 移动后更新这些操作的 block_id 为目标 block_id。
8. 若模块具有 fallback 属性或没有任何 sync 操作，则直接跳过。

### 3. 算法流程

入口 `runOnOperation()`：
- 若存在 fallback 属性，直接返回。
- 遍历模块，检查是否存在 sync 操作（`CVPipeline::isSyncOp`），若无则返回。
- 调用 `relocateMemrefDecls(module)`。

`relocateMemrefDecls(ModuleOp module)` 主流程：

1. 初始化 `ComputeBlockIdManager bm(module)`。
2. 为每个 block 创建 `SyncWall`，存储在 `DenseMap<Block *, CVPipeline::SyncWall> walls`。
3. 构建 `pos` 映射：按前序遍历记录每个 op 的索引，用于后续排序。
4. 构建 `DominanceInfo domInfo(module)`。
5. 遍历模块中所有 op：
   - 对每个 memref 类型的操作数：
     - 获取生产者 `producer`。
     - 确定生产者与消费者所在的共同 block，以及消费者在共同 block 中的祖先 `cAnchor`（若同块则为消费者本身）。
     - 获取该共同 block 的 `SyncWall`，检查生产者与消费者之间是否有 sync。若无则跳过。
     - 确定目标 block 为消费者所在 block，获取 `targetId = bm.getBlockIdByOp(op)`，若为 -1 则跳过。
     - 调用 `buildForwardClosure(producer, targetBlock, anchor, pos, wall, domInfo)` 构建闭包：
       - 从 producer 开始，使用 `SetVector` 收集集群。
       - 迭代至不动点：
         - 对当前 op 的每个用户：
           - 若用户在目标 block 中，则可能更新 anchor（最早在目标 block 中且与当前 op 之间有 sync 的用户），并根据 sync 关系决定是否将用户加入集群。
           - 若用户不在目标 block，且用户是 terminator，则返回空（失败）。
         - 对当前 op 的每个 memref 类型操作数：
           - 获取定义 op `def`。
           - 若 `def` 已支配目标 block 或已在集群中，则跳过。
           - 若 `def` 是 terminator，返回空。
           - 否则将 `def` 加入集群，并检查 `def` 的所有用户是否都在集群中，若不在则继续处理。
       - 若最终没有 anchor，返回空。
       - 将集群按 `pos` 排序后返回。
     - 若闭包为空，跳过。
     - 检查：若共同 block 等于目标 block，且 producer 与 anchor 之间没有 sync，则跳过。
     - 检查：`allOperandsDominate(cluster, anchor, domInfo)`，确保集群中所有操作数都支配 anchor。
     - 将 `Relocation{anchor, ordered, targetId, op->getAttr(CVPipeline::kCoreType)}` 加入 worklist。
6. 遍历 worklist，对每个 relocation：
   - 从后往前遍历集群成员。
   - 将成员移动到 `anchor` 之前（`moveBefore`），并更新 `insertPt`。
   - 调用 `bm.updateBlockId(member, reloc.targetId)` 更新 block_id。

### 4. 关键数据结构

| 结构 | 类型 | 含义 |
|------|------|------|
| `Relocation` | struct | 保存 anchor、cluster、targetId、coreType |
| `SyncWall` | 类 | 用于判断两个 op 之间是否存在同步操作 |
| `ComputeBlockIdManager` | 类 | 管理 op 与 block_id 的映射 |
| `DominanceInfo` | 类 | 支配信息 |
| `walls` | `DenseMap<Block *, CVPipeline::SyncWall>` | 每个 block 的 SyncWall |
| `pos` | `DenseMap<Operation *, unsigned>` | 操作在前序遍历中的索引 |
| `worklist` | `SmallVector<Relocation>` | 待执行的 relocation 列表 |

### 5. 关键函数索引

| 函数 | 位置 | 作用 |
|------|------|------|
| `allOperandsDominate` | 匿名 namespace | 检查集群中所有成员的操作数是否支配 anchor |
| `isAboveTargetBlock` | 匿名 namespace | 判断 def 的 block 是否严格包含 targetBlock |
| `buildForwardClosure` | 匿名 namespace | 递归构建需要一起下沉到目标 block 的闭包 |
| `relocateMemrefDecls` | 匿名 namespace | 主逻辑：扫描、构建闭包、记录并执行重定位 |
| `runOnOperation` | `RelocateMemrefDeclPass` | 入口：检查条件，调用 `relocateMemrefDecls` |
| `createRelocateMemrefDeclPass` | 文件末尾 | 创建 pass 实例 |

### 6. 安全机制

- **Fallback 检查**：若模块具有 fallback 属性，直接返回。
- **Sync 存在性检查**：若模块中没有任何 sync 操作，直接返回。
- **目标 block 有效性**：消费者必须已分配 block_id（≠ -1）。
- **同步屏障检查**：生产者与消费者之间必须存在 sync 操作。
- **Terminator 保护**：闭包构建过程中遇到 terminator 操作则失败返回。
- **支配关系检查**：移动前验证闭包内所有操作数都支配插入点（anchor）。
- **锚点同步检查**：确保 anchor 位于 sync 之后。
- **SSA 支配保持**：通过 `allOperandsDominate` 和闭包构建规则，保证移动后不破坏支配关系。

## 存在问题

无已知问题，pass保留

## 目标场景

1. 为了适配用户debug场景，当用户写的dsl中包含同步语句时生效。