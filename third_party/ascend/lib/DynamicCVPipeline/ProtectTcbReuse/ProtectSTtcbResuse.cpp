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

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ProtectTcbReuse.h"
#include "ascend/include/DynamicCVPipeline/ProtectTcbReuse/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
static bool isGM(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type)
    return false;
  // An explicit address space takes precedence over the origin heuristic.
  if (auto space =
          dyn_cast_or_null<hivm::AddressSpaceAttr>(type.getMemorySpace()))
    return space.getAddressSpace() == hivm::AddressSpace::GM;
  // Without an explicit HIVM address space, recognize function arguments
  // through view chains. A local allocation alone does not establish GM.
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  auto arg = dyn_cast<BlockArgument>(value);
  return arg && isa<func::FuncOp>(arg.getOwner()->getParentOp());
}

// Restrict detection to writeback forms supported at this pipeline stage;
// unknown store-like operations do not establish an output-tail candidate.
static bool isOutputStore(Operation *op) {
  if (auto store = dyn_cast<bufferization::MaterializeInDestinationOp>(op))
    return isGM(store.getDest());
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return isGM(copy.getTarget());
  return false;
}

/// Check only direct body operations, excluding stores inside main loops.
/// Candidates are enclosing loops discovered from a main_loop operation.
static bool hasOutputTail(scf::ForOp outerLoop) {
  return llvm::any_of(*outerLoop.getBody(), [](Operation &bodyOperation) {
    return !CVPipeline::isMainLoopOp(&bodyOperation) &&
           isOutputStore(&bodyOperation);
  });
}

class ProtectSTtcbResusePass
    : public PassWrapper<ProtectSTtcbResusePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProtectSTtcbResusePass)
  StringRef getArgument() const override { return "protect-st-tcb-reuse"; }
  StringRef getDescription() const override {
    return "Protect Vector output stores before TCB reuse";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect>();
  }
  void runOnOperation() override {
    tcbReuse::protectTcbReuse(getOperation(), "store tcb reuse",
                              hivm::PIPE::PIPE_MTE3, hasOutputTail);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createProtectSTtcbResusePass() {
  return std::make_unique<ProtectSTtcbResusePass>();
}
