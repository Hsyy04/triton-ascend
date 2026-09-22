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

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/HoistIfCondition.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "HoistIfCondition";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

/// Attribute placed on first-pass loops to prevent re-transformation.
constexpr llvm::StringLiteral kHoistedIfCondAttr = "ssbuffer.hoisted_if_cond";

/// Collect all operations in the backward (def-use) slice of `roots`, keeping
/// only those for which `isInScope` returns true.  The SetVector is used for
/// membership testing only — callers should iterate the source block in order
/// to guarantee correct topological cloning.
SetVector<Operation *>
collectBackwardSlice(ArrayRef<Value> roots,
                     function_ref<bool(Operation *)> isInScope) {
  SetVector<Operation *> result;
  SmallVector<Value> worklist(roots.begin(), roots.end());
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (isa<BlockArgument>(v))
      continue;
    Operation *defOp = v.getDefiningOp();
    if (!defOp || !isInScope(defOp))
      continue;
    if (!result.contains(defOp)) {
      result.insert(defOp);
      for (Value operand : defOp->getOperands())
        worklist.push_back(operand);
    }
  }
  return result;
}

/// Return true if `v` transitively depends on any of `args`, traversing only
/// operations whose parent block is `body`.
bool dependsOnArg(Value v, ArrayRef<BlockArgument> args, Block *body) {
  DenseSet<Value> argSet;
  for (BlockArgument arg : args)
    argSet.insert(arg);

  SetVector<Value> visited;
  SmallVector<Value> worklist{v};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (visited.contains(current))
      continue;
    visited.insert(current);

    if (argSet.contains(current))
      return true;

    if (isa<BlockArgument>(current))
      continue;

    Operation *defOp = current.getDefiningOp();
    if (!defOp || defOp->getBlock() != body)
      continue;

    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

/// Find an scf.if directly inside `forOp`'s body that matches the pattern:
///   1. if results are exactly the loop yield operands
///   2. else branch yields the loop iter_args (pass-through)
///   3. condition depends on the loop IV
///   4. condition does NOT depend on any iter_arg
/// Returns null IfOp if no match.
scf::IfOp findMatchingIfOp(scf::ForOp forOp) {
  Block *body = forOp.getBody();
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());

  for (Operation &op : *body) {
    auto ifOp = dyn_cast<scf::IfOp>(&op);
    if (!ifOp)
      continue;

    // 1. if results == yield operands
    if (ifOp.getNumResults() != yieldOp.getNumOperands())
      continue;
    bool resultsMatch = true;
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      if (yieldOp.getOperand(i) != ifOp.getResult(i)) {
        resultsMatch = false;
        break;
      }
    }
    if (!resultsMatch)
      continue;

    // 2. else branch yields iter_args (pass-through)
    if (!ifOp.getElseRegion().hasOneBlock())
      continue;
    auto elseYield =
        cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    auto iterArgs = forOp.getRegionIterArgs();
    if (elseYield.getNumOperands() != iterArgs.size())
      continue;
    bool passThrough = true;
    for (unsigned i = 0; i < iterArgs.size(); ++i) {
      if (elseYield.getOperand(i) != iterArgs[i]) {
        passThrough = false;
        break;
      }
    }
    if (!passThrough)
      continue;

    // 3. condition depends on IV
    SmallVector<BlockArgument> ivArgs{
        cast<BlockArgument>(forOp.getInductionVar())};
    if (!dependsOnArg(ifOp.getCondition(), ivArgs, body))
      continue;

    // 4. condition does NOT depend on iter_args
    if (!iterArgs.empty()) {
      SmallVector<BlockArgument> iterArgVec;
      for (Value arg : iterArgs)
        iterArgVec.push_back(cast<BlockArgument>(arg));
      if (dependsOnArg(ifOp.getCondition(), iterArgVec, body))
        continue;
    }

    return ifOp;
  }
  return scf::IfOp();
}

/// Transform a matching scf.for into the two-pass index-collection pattern.
LogicalResult transformLoop(scf::ForOp forOp, scf::IfOp ifOp) {
  Location loc = forOp.getLoc();
  OpBuilder builder(forOp);

  Value lb = forOp.getLowerBound();
  Value ub = forOp.getUpperBound();
  Value step = forOp.getStep();
  Type ivType = forOp.getInductionVar().getType();
  Block *body = forOp.getBody();
  Block *thenBlock = ifOp.thenBlock();

  // -- index constants --
  Value c0Index = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1Index = builder.create<arith::ConstantIndexOp>(loc, 1);

  // -- trip count = max(0, ceildiv(ub - lb, step)) --
  Value diff = builder.create<arith::SubIOp>(loc, ub, lb);
  Value rawTrip = builder.create<arith::CeilDivSIOp>(loc, diff, step);
  Value zeroIv = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(ivType, 0));
  Value trip = builder.create<arith::MaxSIOp>(loc, rawTrip, zeroIv);
  if (!isa<IndexType>(trip.getType())) {
    trip = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                               trip);
  }

  // -- create indices tensor: tensor<?x ivType> --
  auto indicesTensorType = RankedTensorType::get({ShapedType::kDynamic}, ivType);
  Value indicesTensor = builder.create<tensor::EmptyOp>(
      loc, indicesTensorType, ValueRange{trip});

  // ========================================================================
  // First pass: collect valid indices
  // ========================================================================
  // The loop carries two iter_args: count (index) and indices tensor.
  auto firstFor = builder.create<scf::ForOp>(loc, lb, ub, step,
                                             ValueRange{c0Index, indicesTensor});
  firstFor->setAttr(kHoistedIfCondAttr, builder.getUnitAttr());

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(firstFor.getBody());

    // Clone condition computation (backward slice of condition, scoped to
    // pre-if ops in the loop body).
    IRMapping firstMapping;
    firstMapping.map(forOp.getInductionVar(), firstFor.getInductionVar());

    auto isPreIf = [&](Operation *op) {
      return op->getBlock() == body && op->isBeforeInBlock(ifOp);
    };
    SetVector<Operation *> condSlice =
        collectBackwardSlice({ifOp.getCondition()}, isPreIf);

    for (Operation &op : body->without_terminator()) {
      if (&op == ifOp)
        break;
      if (condSlice.contains(&op))
        builder.clone(op, firstMapping);
    }

    Value cond = firstMapping.lookupOrDefault(ifOp.getCondition());

    // Create index-collection scf.if (results: index count, tensor<?xivType>)
    auto collectIf = builder.create<scf::IfOp>(
        loc, TypeRange{builder.getIndexType(), indicesTensorType},
        cond, /*withElseRegion=*/true);

    // then: insert IV into indices tensor, increment count
    {
      OpBuilder::InsertionGuard guardIf(builder);
      builder.setInsertionPointToStart(collectIf.thenBlock());
      Value countIter = firstFor.getRegionIterArgs()[0];
      Value indicesIter = firstFor.getRegionIterArgs()[1];
      Value inserted = builder.create<tensor::InsertOp>(
          loc, firstFor.getInductionVar(), indicesIter, ValueRange{countIter});
      Value next = builder.create<arith::AddIOp>(loc, countIter, c1Index);
      builder.create<scf::YieldOp>(loc, ValueRange{next, inserted});
    }
    // else: yield count and indices unchanged
    {
      OpBuilder::InsertionGuard guardElse(builder);
      builder.setInsertionPointToStart(collectIf.elseBlock());
      builder.create<scf::YieldOp>(loc, ValueRange{
          firstFor.getRegionIterArgs()[0], firstFor.getRegionIterArgs()[1]});
    }

    builder.create<scf::YieldOp>(loc, ValueRange{
        collectIf.getResult(0), collectIf.getResult(1)});
  }

  Value countFinal = firstFor.getResult(0);
  Value indicesTensorFinal = firstFor.getResult(1);

  // ========================================================================
  // Second pass: execute body for valid indices
  // ========================================================================
  // Use i32 for the second-pass loop IV; cast count_final to i32.
  Type i32Type = builder.getIntegerType(32);
  Value countFinalI32 = builder.create<arith::IndexCastOp>(loc, i32Type,
                                                             countFinal);
  Value c0I32 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(i32Type, 0));
  Value c1I32 = builder.create<arith::ConstantOp>(
      loc, builder.getIntegerAttr(i32Type, 1));

  auto initArgs = forOp.getInitArgs();
  auto secondFor = builder.create<scf::ForOp>(loc, c0I32, countFinalI32,
                                              c1I32, initArgs);

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(secondFor.getBody());

    // Cast loop IV (i32) to index for tensor indexing.
    Value loopIVAsIndex = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), secondFor.getInductionVar());

    // Extract original IV from indices tensor
    Value loadedIV = builder.create<tensor::ExtractOp>(
        loc, indicesTensorFinal, ValueRange{loopIVAsIndex});

    // Mapping: original IV → loaded IV, original iter_args → new iter_args
    IRMapping secondMapping;
    secondMapping.map(forOp.getInductionVar(), loadedIV);
    auto origIterArgs = forOp.getRegionIterArgs();
    auto newIterArgs = secondFor.getRegionIterArgs();
    for (unsigned i = 0; i < origIterArgs.size(); ++i)
      secondMapping.map(origIterArgs[i], newIterArgs[i]);

    // Collect roots: operands of then-branch body ops that are defined in
    // the loop body block before the scf.if.
    SmallVector<Value> roots;
    ifOp.getThenRegion().walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (defOp->getBlock() == body && defOp->isBeforeInBlock(ifOp))
            roots.push_back(operand);
        }
      }
    });

    auto isPreIf = [&](Operation *op) {
      return op->getBlock() == body && op->isBeforeInBlock(ifOp);
    };
    SetVector<Operation *> preIfSlice = collectBackwardSlice(roots, isPreIf);

    // Clone pre-if ops (in block order for correct topological cloning)
    for (Operation &op : body->without_terminator()) {
      if (&op == ifOp)
        break;
      if (preIfSlice.contains(&op))
        builder.clone(op, secondMapping);
    }

    // Clone then-branch body ops (all except yield)
    auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
    for (Operation &op : thenBlock->without_terminator())
      builder.clone(op, secondMapping);

    // Build yield operands
    SmallVector<Value> yieldOperands;
    for (Value operand : thenYield.getOperands())
      yieldOperands.push_back(secondMapping.lookupOrDefault(operand));

    // Handle existing terminator (default yield when no iter_args)
    Block *secondBody = secondFor.getBody();
    if (!secondBody->empty() && isa<scf::YieldOp>(secondBody->back()))
      secondBody->back().erase();
    builder.setInsertionPointToEnd(secondBody);
    builder.create<scf::YieldOp>(loc, yieldOperands);
  }

  // Replace original loop results
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(secondFor.getResult(i));

  forOp.erase();
  return success();
}

} // anonymous namespace

namespace mlir::triton::CVSplit {

void HoistIfConditionPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<scf::SCFDialect, arith::ArithDialect,
                  memref::MemRefDialect, tensor::TensorDialect>();
}

void HoistIfConditionPass::runOnOperation() {
  auto moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp))
    return;

  LOG_DEBUG("Input mlir:\n" << moduleOp);

  bool changed = true;
  while (changed) {
    changed = false;
    moduleOp.walk([&](scf::ForOp forOp) -> WalkResult {
      if (forOp->hasAttr(kHoistedIfCondAttr))
        return WalkResult::advance();

      scf::IfOp ifOp = findMatchingIfOp(forOp);
      if (!ifOp)
        return WalkResult::advance();

      if (failed(transformLoop(forOp, ifOp))) {
        LOG_DEBUG("Failed to transform loop at " << forOp.getLoc());
        CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
        return WalkResult::interrupt();
      }

      changed = true;
      return WalkResult::interrupt();
    });

    if (CVPipeline::hasFallbackAttr(moduleOp))
      return;
  }

  LOG_DEBUG("Output mlir:\n" << moduleOp);
}

std::unique_ptr<OperationPass<ModuleOp>> createHoistIfConditionPass() {
  return std::make_unique<HoistIfConditionPass>();
}

} // namespace mlir::triton::CVSplit
