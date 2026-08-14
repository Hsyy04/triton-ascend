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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/Passes.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "CopyPosOnehot";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace mlir::triton {

/**
 * Pass to duplicate fill and and operations for the posOnehot pattern.
 * 
 * This pass matches the following pattern:
 * 
 *   %posOnehot = linalg.generic {...} {tt.from_make_range} -> tensor<Nxi32>
 *   for (arg) {
 *     %fill = linalg.fill(arg) -> tensor<Nxi32>
 *     %and = arith.andi %fill, %posOnehot
 *     %bcast = linalg.broadcast(%and)  // One user is broadcast
 *     %other = someOp(%and)             // Another user is different shape
 *   }
 * 
 * And transforms it to:
 * 
 *   %posOnehot = linalg.generic {...} {tt.from_make_range} -> tensor<Nxi32>
 *   for (arg) {
 *     %fill = linalg.fill(arg) -> tensor<Nxi32>
 *     %and1 = arith.andi %fill, %posOnehot
 *     %fill_copy = linalg.fill(arg) -> tensor<Nxi32>  // Duplicated
 *     %and2 = arith.andi %fill_copy, %posOnehot        // Duplicated
 *     %bcast = linalg.broadcast(%and1)  // Uses original
 *     %other = someOp(%and2)             // Uses duplicated version
 *   }
 * 
 * This transformation prevents the posOnehot value from being used by
 * operations with different output shapes, which can cause issues in
 * downstream optimization passes.
 */
class CopyPosOnehotPass
    : public PassWrapper<CopyPosOnehotPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyPosOnehotPass)

  CopyPosOnehotPass() = default;

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

private:
  void processAndOp(arith::AddIOp addOp, Value posValue);
};

std::unique_ptr<OperationPass<ModuleOp>> createCopyPosOnehotPass() {
  return std::make_unique<CopyPosOnehotPass>();
}



void CopyPosOnehotPass::processAndOp(arith::AddIOp addOp, Value posValue) {
  auto lhs = addOp.getLhs();
  auto rhs = addOp.getRhs();

  linalg::FillOp fillOp = nullptr;
  Value fillValue;

  if (auto fill = lhs.getDefiningOp<linalg::FillOp>()) {
    fillOp = fill;
    fillValue = lhs;
  } else if (auto fill = rhs.getDefiningOp<linalg::FillOp>()) {
    fillOp = fill;
    fillValue = rhs;
  }

  if (!fillOp) {
    return;
  }
  if(!fillValue.hasOneUse()) {
    return;
  }

  auto users = llvm::to_vector<2>(addOp.getResult().getUsers());
  if (users.size() != 2) {
    return;
  }

  Operation *broadcastUser = nullptr;
  Operation *otherUser = nullptr;

  for (auto *user : users) {
    if (isa<linalg::BroadcastOp>(user)) {
      broadcastUser = user;
    } else {
      otherUser = user;
    }
  }

  if (!broadcastUser || !otherUser) {
    return;
  }

  OpBuilder builder(addOp);
  builder.setInsertionPoint(addOp);

  auto newFillOp = builder.create<linalg::FillOp>(
      fillOp.getLoc(), fillOp.getInputs(), fillOp.getOutputs());

  Value newFillResult = newFillOp.getResult(0);

  Value lhsForNewAnd = (lhs == fillValue) ? newFillResult : lhs;
  Value rhsForNewAnd = (rhs == fillValue) ? newFillResult : rhs;
  if (lhsForNewAnd != newFillResult && rhsForNewAnd != newFillResult) {
    return;
  }

  auto newAndOp =
      builder.create<arith::AddIOp>(addOp.getLoc(), lhsForNewAnd, rhsForNewAnd);
  unsigned operandIdx = 0;
  for (auto &operand : otherUser->getOpOperands()) {
    if (operand.get() == addOp.getResult()) {
      operandIdx = operand.getOperandNumber();
      break;
    }
  }

  otherUser->setOperand(operandIdx, newAndOp.getResult());
}


void CopyPosOnehotPass::runOnOperation() {
  auto moduleOp = getOperation();
  LOG_DEBUG("input mlir: \n" << moduleOp );

  moduleOp.walk([&](linalg::GenericOp genericOp) {
    if (!genericOp->hasAttr("tt.from_make_range")) {
      return WalkResult::advance();
    }

    if (genericOp.getNumResults() != 1) {
      return WalkResult::advance();
    }

    Value posValue = genericOp.getResult(0);

    SmallVector<Operation *, 4> andUsers;
    for (auto *user : posValue.getUsers()) {
      if (auto addOp = dyn_cast<arith::AddIOp>(user)) {
        andUsers.push_back(addOp);
      }
    }

    for (auto *op : andUsers) {
      auto addOp = cast<arith::AddIOp>(op);
      processAndOp(addOp, posValue);
    }

    return WalkResult::advance();
  });
}

llvm::StringRef CopyPosOnehotPass::getArgument() const {
  return "copy-pos-onehot";
}

llvm::StringRef CopyPosOnehotPass::getDescription() const {
  return "Copy fill and and operations for posOnehot pattern to avoid "
         "different shape usage";
}

} // namespace mlir::triton