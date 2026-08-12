/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <optional>
#include <utility>

static constexpr const char *DEBUG_TYPE = "merge-small-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {

class MergeSmallBlockPass
    : public PassWrapper<MergeSmallBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeSmallBlockPass)

  MergeSmallBlockPass() = default;

  StringRef getArgument() const override { return "merge-small-block"; }

  StringRef getDescription() const override {
    return "Merge small compute blocks (<= 3 ops) into their operand or user "
           "block";
  }
  void runOnOperation() override;

private:
  const int MIN_VF_SIZE = 3;
};

static int cntCalcuateOps(llvm::SmallVector<Operation *> ops) {
  int count = 0;
  for (Operation *op : ops) {
    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::EmptyOp>(
            op)) {
      continue;
    }
    bool allOperandsTensor = llvm::all_of(op->getOperands(), [](Value operand) {
      return isa<RankedTensorType>(operand.getType());
    });
    bool allResultsTensor = llvm::all_of(op->getResults(), [](Value result) {
      return isa<RankedTensorType>(result.getType());
    });
    if (!allOperandsTensor || !allResultsTensor) {
      continue;
    }
    count++;
  }
  return count;
}

class MergeStrategy {
public:
  struct Context {
    Block *block;
    CVPipeline::ComputeBlockIdManager &bm;
    CVPipeline::MemoryDependenceGraph &memGraph;
    const DenseMap<int, int> &id2order;
    const SmallVector<Operation *> &smallBlockOps;
    int smallBlockId;
  };

  struct UpDownClassification {
    SmallVector<int> upBlockIds;
    SmallVector<int> downBlockIds;
  };

  virtual ~MergeStrategy() = default;
  virtual void filter(SmallVector<int> &candidates, const Context &ctx) = 0;

  static UpDownClassification
  classifierUpDown(const SmallVector<int> &candidates, const Context &ctx) {
    UpDownClassification result;

    for (int blockId : candidates) {
      if (ctx.id2order.contains(blockId) &&
          ctx.id2order.lookup(blockId) <
              ctx.id2order.lookup(ctx.smallBlockId)) {
        result.upBlockIds.push_back(blockId);
      } else {
        result.downBlockIds.push_back(blockId);
      }
    }

    return result;
  }
};

class IROrderStrategy : public MergeStrategy {
public:
  void filter(SmallVector<int> &candidates, const Context &ctx) override {
    auto classification = classifierUpDown(candidates, ctx);

    if (classification.downBlockIds.empty()) {
      return;
    }

    int selectedDownBlock = -1;
    for (int id : classification.downBlockIds) {
      if (selectedDownBlock == -1 ||
          (ctx.id2order.contains(id) &&
           ctx.id2order.lookup(id) < ctx.id2order.lookup(selectedDownBlock))) {
        selectedDownBlock = id;
      }
    }

    candidates.clear();
    candidates.append(classification.upBlockIds.begin(),
                      classification.upBlockIds.end());
    if (selectedDownBlock != -1) {
      candidates.push_back(selectedDownBlock);
    }
  }
};

class UBOccupationStrategy : public MergeStrategy {
private:
  static constexpr int INF = (1 << 30);

  static int getValueSizeInBytes(Value value) {
    if (CVPipeline::isScalarLike(value)) {
      return 0;
    }
    Type type = value.getType();
    auto getElemBytes = [](Type elemType) -> int64_t {
      if (elemType.isIntOrFloat()) {
        unsigned bits = elemType.getIntOrFloatBitWidth();
        return std::max<int64_t>(1, bits / 8);
      }
      return 1;
    };
    if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
      if (!rankedTensorType.hasStaticShape()) {
        return INF;
      }
      int64_t numElements = 1;
      for (int64_t dim : rankedTensorType.getShape()) {
        if (dim < 0) {
          return 1;
        }
        numElements *= dim;
      }
      return static_cast<int>(std::max<int64_t>(
          1, numElements * getElemBytes(rankedTensorType.getElementType())));
    }
    return INF;
  }

  static int getOperandUB(const SmallVector<Operation *> &smallBlockOps,
                          int blockId, CVPipeline::ComputeBlockIdManager &bm) {
    int total = 0;
    for (Operation *op : smallBlockOps) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp) {
          continue;
        }
        if (bm.getBlockIdByOp(defOp) == blockId) {
          int valueSize = getValueSizeInBytes(operand);
          if (valueSize == INF) {
            return INF;
          }
          total += valueSize;
        }
      }
    }
    return total;
  }

  static int getUserUB(const SmallVector<Operation *> &smallBlockOps,
                       int blockId, CVPipeline::ComputeBlockIdManager &bm) {
    int total = 0;
    for (Operation *op : smallBlockOps) {
      for (Value result : op->getResults()) {
        if (llvm::all_of(result.getUsers(), [&](Operation *user) {
              return bm.getBlockIdByOp(user) == blockId;
            })) {
          int valueSize = getValueSizeInBytes(result);
          if (valueSize == INF) {
            return INF;
          }
          total += valueSize;
        }
      }
    }
    return total;
  }

public:
  void filter(SmallVector<int> &candidates, const Context &ctx) override {
    auto classification = classifierUpDown(candidates, ctx);

    DenseMap<int, int> blockIdToUB;

    for (int upBlockId : classification.upBlockIds) {
      int ub = getOperandUB(ctx.smallBlockOps, upBlockId, ctx.bm);
      blockIdToUB[upBlockId] = ub;
      LOG_DEBUG("Merge into up block " << upBlockId << " will save UB: " << ub
                                       << "B.");
    }

    for (int downBlockId : classification.downBlockIds) {
      int ub = getUserUB(ctx.smallBlockOps, downBlockId, ctx.bm);
      blockIdToUB[downBlockId] = ub;
      LOG_DEBUG("Merge into down block " << downBlockId
                                         << " will save UB: " << ub << "B.");
    }

    if (blockIdToUB.empty()) {
      return;
    }

    int maxUB = -1;
    for (auto &entry : blockIdToUB) {
      if (entry.second > maxUB && entry.second != INF) {
        maxUB = entry.second;
      }
    }

    SmallVector<int> filtered;
    for (int blockId : candidates) {
      if (blockIdToUB.contains(blockId) && blockIdToUB[blockId] == maxUB) {
        filtered.push_back(blockId);
      }
    }

    candidates = std::move(filtered);
  }
};

class DefaultUpStrategy : public MergeStrategy {
public:
  void filter(SmallVector<int> &candidates, const Context &ctx) override {
    auto classification = classifierUpDown(candidates, ctx);

    if (!classification.upBlockIds.empty()) {
      candidates.clear();
      candidates.push_back(classification.upBlockIds[0]);
      return;
    }

    if (!classification.downBlockIds.empty()) {
      candidates.clear();
      candidates.push_back(classification.downBlockIds[0]);
      return;
    }
  }
};

// This strategy matches a specific Flash Attention Forward (FaFWD) pattern in
// small compute blocks. The pattern consists of 3 operations: exp, mulf, and
// addf, forming a softmax-like computation.
//
// Pattern structure:
//   - exp: computes exp(subf), where subf is from an upstream block
//   (upBlockIds[0])
//   - mulf: computes exp_result * arg, where arg is a loop iteration argument
//   - addf: computes mulf_result + reduce, where reduce is from the same
//   upstream block
//
// Data flow:
//   1. exp takes subf as input, its result is used by:
//      - mulf (in the current small block)
//      - broadcast (in downstream block, downBlockIds[0])
//
//   2. mulf takes exp_result and arg (BlockArgument) as inputs, its result is
//   used by:
//      - addf (in the current small block, no other users)
//
//   3. addf takes mulf_result and reduce as inputs, its result is used by:
//      - yield (updates the same arg that mulf uses)
//
// This pattern typically appears in Flash Attention forward pass where:
//   - exp computes the exponential of (QK^T - max)
//   - mulf scales by a running sum
//   - addf accumulates the result
//
// When matched, the strategy recommends merging into the downstream block
// (downBlockIds[0]).
class FaFWDPatternStrategy : public MergeStrategy {
public:
  void filter(SmallVector<int> &candidates, const Context &ctx) override {
    auto classification = classifierUpDown(candidates, ctx);

    if (classification.upBlockIds.size() != 1 ||
        classification.downBlockIds.size() != 1) {
      return;
    }

    if (ctx.smallBlockOps.size() != 3) {
      return;
    }

    math::ExpOp expOp = nullptr;
    arith::MulFOp mulfOp = nullptr;
    arith::AddFOp addfOp = nullptr;

    for (Operation *op : ctx.smallBlockOps) {
      if (auto exp = dyn_cast<math::ExpOp>(op)) {
        expOp = exp;
      } else if (auto mulf = dyn_cast<arith::MulFOp>(op)) {
        mulfOp = mulf;
      } else if (auto addf = dyn_cast<arith::AddFOp>(op)) {
        addfOp = addf;
      } else {
        return;
      }
    }

    if (!expOp || !mulfOp || !addfOp) {
      return;
    }

    Operation *subfDef = expOp.getOperand().getDefiningOp();
    if (!subfDef || !isa<arith::SubFOp>(subfDef)) {
      return;
    }

    int subfBlockId = ctx.bm.getBlockIdByOp(subfDef);
    if (subfBlockId != classification.upBlockIds[0]) {
      return;
    }

    auto expResult = expOp.getResult();
    SmallVector<Operation *, 2> expUsers;
    for (Operation *user : expResult.getUsers()) {
      Operation *userInBlock = CVPipeline::getAncestorInBlock(user, ctx.block);
      if (userInBlock && userInBlock != ctx.block->getTerminator()) {
        expUsers.push_back(user);
      }
    }

    if (expUsers.size() != 2) {
      return;
    }

    bool hasMulfUser = false;
    bool hasBroadcastUser = false;
    Operation *broadcastUser = nullptr;

    for (Operation *user : expUsers) {
      if (user == mulfOp) {
        hasMulfUser = true;
      } else if (isa<linalg::BroadcastOp>(user) ||
                 isa<triton::BroadcastOp>(user)) {
        int broadcastBlockId = ctx.bm.getBlockIdByOp(user);
        if (broadcastBlockId == classification.downBlockIds[0]) {
          hasBroadcastUser = true;
          broadcastUser = user;
        }
      }
    }

    if (!hasMulfUser || !hasBroadcastUser) {
      return;
    }

    auto mulfLhs = mulfOp.getLhs();
    auto mulfRhs = mulfOp.getRhs();

    bool expIsOperand = (mulfLhs == expResult || mulfRhs == expResult);
    if (!expIsOperand) {
      return;
    }

    Value mulfArgOperand = (mulfLhs == expResult) ? mulfRhs : mulfLhs;
    BlockArgument mulfArg = dyn_cast<BlockArgument>(mulfArgOperand);
    if (!mulfArg) {
      return;
    }

    for (Operation *user : mulfOp.getResult().getUsers()) {
      if (user != addfOp) {
        return;
      }
    }

    auto addfLhs = addfOp.getLhs();
    auto addfRhs = addfOp.getRhs();

    bool mulfIsOperand =
        (addfLhs == mulfOp.getResult() || addfRhs == mulfOp.getResult());
    if (!mulfIsOperand) {
      return;
    }

    Value reduceOperand = (addfLhs == mulfOp.getResult()) ? addfRhs : addfLhs;
    Operation *reduceDef = reduceOperand.getDefiningOp();

    if (!reduceDef || !isa<linalg::ReduceOp, triton::ReduceOp>(reduceDef)) {
      return;
    }

    int reduceBlockId = ctx.bm.getBlockIdByOp(reduceDef);
    if (reduceBlockId != classification.upBlockIds[0]) {
      return;
    }

    auto addfResult = addfOp.getResult();
    SmallVector<Operation *, 1> addfUsers;
    for (Operation *user : addfResult.getUsers()) {
      addfUsers.push_back(user);
    }

    if (addfUsers.size() != 1) {
      return;
    }

    scf::YieldOp yieldOp = dyn_cast<scf::YieldOp>(addfUsers[0]);
    if (!yieldOp) {
      return;
    }

    Block *yieldBlock = yieldOp->getBlock();
    scf::ForOp forOp = dyn_cast<scf::ForOp>(yieldBlock->getParentOp());
    if (!forOp) {
      return;
    }

    int yieldOpIndex = -1;
    auto yieldOperands = yieldOp.getOperands();
    for (int i = 0; i < yieldOperands.size(); ++i) {
      if (yieldOperands[i] == addfResult) {
        yieldOpIndex = i;
        break;
      }
    }

    if (yieldOpIndex == -1) {
      return;
    }

    BlockArgument correspondingIterArg = forOp.getRegionIterArg(yieldOpIndex);
    if (correspondingIterArg != mulfArg) {
      return;
    }

    candidates.clear();
    candidates.push_back(classification.downBlockIds[0]);
  }
};

static SmallVector<int>
collectMergeCandidates(int smallBlockId, Block *block,
                       CVPipeline::ComputeBlockIdManager &bm,
                       CVPipeline::MemoryDependenceGraph &memGraph) {

  bool haveCubeLink = false;
  SmallVector<int> candidates;
  auto ops = bm.getOpsByBlockId(smallBlockId);

  // upstream candidates
  llvm::SmallDenseSet<int> upBlockIds;
  for (Operation *op : ops) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp) {
        continue;
      }
      Operation *defInBlock = CVPipeline::getAncestorInBlock(defOp, block);
      if (!defInBlock) {
        continue;
      }
      if (CVPipeline::getOpCoreType(defInBlock) !=
          CVPipeline::CoreType::VECTOR_ONLY) {
        haveCubeLink = true;
        break;
      }
      int bid = bm.getBlockIdByOp(defInBlock);
      if (bid != -1 && bid != smallBlockId) {
        upBlockIds.insert(bid);
      }
    }
    if (haveCubeLink) {
      break;
    }
  }
  if (haveCubeLink) {
    LOG_DEBUG("Block " << smallBlockId
                       << " has operand defined by CUBE, no up candidates.");
    upBlockIds.clear();
  }

  if (upBlockIds.size() == 1) {
    int upBlockId = *upBlockIds.begin();
    if (!CVPipeline::willCreateCycle(ops, memGraph, upBlockId, bm)) {
      candidates.push_back(upBlockId);
    }
  }

  // downstream candidates
  haveCubeLink = false;
  llvm::SmallDenseSet<int> downBlockIds;
  for (Operation *op : ops) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        Operation *userInBlock = CVPipeline::getAncestorInBlock(user, block);
        if (!userInBlock || (block->mightHaveTerminator() &&
                             userInBlock == block->getTerminator())) {
          continue;
        }
        if (CVPipeline::getOpCoreType(userInBlock) !=
            CVPipeline::CoreType::VECTOR_ONLY) {
          haveCubeLink = true;
          break;
        }
        int bid = bm.getBlockIdByOp(userInBlock);
        if (bid != -1 && bid != smallBlockId) {
          downBlockIds.insert(bid);
        }
      }
      if (haveCubeLink) {
        break;
      }
    }
    if (haveCubeLink) {
      break;
    }
  }
  if (haveCubeLink) {
    LOG_DEBUG("Block " << smallBlockId
                       << " has result used by CUBE, no down candidates.");
    downBlockIds.clear();
  }
  for (int downBlockId : downBlockIds) {
    if (!CVPipeline::willCreateCycle(ops, memGraph, downBlockId, bm)) {
      candidates.push_back(downBlockId);
    }
  }

  return candidates;
}

static std::optional<int>
selectMergeTarget(SmallVector<int> &candidates,
                  const SmallVector<Operation *> &ops, Block *block,
                  CVPipeline::ComputeBlockIdManager &bm,
                  CVPipeline::MemoryDependenceGraph &memGraph,
                  const DenseMap<int, int> &id2order, int smallBlockId) {
  if (candidates.size() == 1) {
    return candidates[0];
  }
  SmallVector<std::unique_ptr<MergeStrategy>> strategies;
  strategies.push_back(std::make_unique<FaFWDPatternStrategy>());
  strategies.push_back(std::make_unique<UBOccupationStrategy>());
  strategies.push_back(std::make_unique<IROrderStrategy>());
  strategies.push_back(std::make_unique<DefaultUpStrategy>());

  MergeStrategy::Context ctx{block, bm, memGraph, id2order, ops, smallBlockId};

  for (auto &strategy : strategies) {
    strategy->filter(candidates, ctx);
    if (candidates.size() == 1) {
      return candidates[0];
    }
  }

  return std::nullopt;
}

static void mergeBlock(int targetBlockId, const SmallVector<Operation *> &ops,
                       CVPipeline::ComputeBlockIdManager &bm) {
  for (Operation *op : ops) {
    bm.updateBlockId(op, targetBlockId);
  }
}

static SmallVector<int>
getBlockIdsInProgramOrder(Block *block, CVPipeline::ComputeBlockIdManager &bm,
                          SmallVector<int> &ordered,
                          DenseMap<int, int> &id2order) {
  ordered.clear();
  id2order.clear();
  llvm::SmallDenseSet<int, 4> seen;
  for (Operation &op : *block) {
    int bid = bm.getBlockIdByOp(&op);
    if (bid != -1 && seen.insert(bid).second) {
      id2order[bid] = ordered.size();
      ordered.push_back(bid);
    }
  }
  return ordered;
}

void MergeSmallBlockPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LOG_DEBUG("Before: " << *module);
  auto &aa = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph(module, aa);
  auto bm = CVPipeline::ComputeBlockIdManager(module);

  llvm::SmallVector<Block *> blocks;
  module.walk([&](Block *block) { blocks.push_back(block); });

  for (Block *block : blocks) {
    SmallVector<int> orderedBlockIds;
    DenseMap<int, int> id2order;
    getBlockIdsInProgramOrder(block, bm, orderedBlockIds, id2order);

    for (int nowBlockId : orderedBlockIds) {
      LOG_DEBUG("Processing block " << nowBlockId);
      auto ops = bm.getOpsByBlockId(nowBlockId);
      if (ops.empty() || cntCalcuateOps(ops) > MIN_VF_SIZE) {
        continue;
      }
      if (CVPipeline::getOpCoreType(*ops.begin()) !=
          CVPipeline::CoreType::VECTOR_ONLY) {
        continue;
      }

      LOG_DEBUG("Processing small block " << nowBlockId);
      for (auto op : ops) {
        LOG_DEBUG("op:" << *op);
      }

      SmallVector<int> candidates =
          collectMergeCandidates(nowBlockId, block, bm, memGraph);

      if (candidates.empty()) {
        LOG_DEBUG("No candidates for block " << nowBlockId);
        continue;
      }
      LOG_DEBUG("ready to merge block " << nowBlockId << " with candidates: ");
      for (int candidateId : candidates) {
        LOG_DEBUG("candidate block " << candidateId);
      }

      auto targetBlockId = selectMergeTarget(candidates, ops, block, bm,
                                             memGraph, id2order, nowBlockId);

      if (targetBlockId.has_value()) {
        LOG_DEBUG("Merging block " << nowBlockId << " into block "
                                   << targetBlockId.value());
        mergeBlock(targetBlockId.value(), ops, bm);
      }
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createMergeSmallBlockPass() {
  return std::make_unique<MergeSmallBlockPass>();
}

void registerMergeSmallBlockPass() {
  PassRegistration<MergeSmallBlockPass> reg;
}

} // namespace triton
} // namespace mlir
