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

#include "ascend/include/DynamicCVPipeline/ProtectOutputStore.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "protect-output-store"
#define LDBG(...)                                                              \
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {
constexpr StringLiteral kProtected = "hivm.output_store_protected";

struct LoopPair {
  scf::ForOp vector;
  scf::ForOp cube;
};

static std::optional<hivm::TCoreType> getCore(Operation *op) {
  auto scope = op->getParentOfType<scope::ScopeOp>();
  if (!scope)
    return std::nullopt;
  auto attr = scope->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
  if (!attr)
    return std::nullopt;
  return attr.getTcoretype();
}

static llvm::SmallSet<int64_t, 4> getMainLoops(scf::ForOp loop) {
  llvm::SmallSet<int64_t, 4> ids;
  loop.walk([&](Operation *op) {
    if (auto id = op->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop))
      ids.insert(id.getInt());
  });
  return ids;
}

static bool isGM(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type)
    return false;
  if (auto space =
          dyn_cast_or_null<hivm::AddressSpaceAttr>(type.getMemorySpace()))
    return space.getAddressSpace() == hivm::AddressSpace::GM;
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  auto arg = dyn_cast<BlockArgument>(value);
  return arg && isa<func::FuncOp>(arg.getOwner()->getParentOp());
}

static bool isOutputStore(Operation *op) {
  if (auto store = dyn_cast<bufferization::MaterializeInDestinationOp>(op))
    return isGM(store.getDest());
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return isGM(copy.getTarget());
  return false;
}

static bool hasOutputTail(scf::ForOp loop) {
  Operation *lastMain = nullptr;
  loop.walk([&](Operation *op) {
    if (!CVPipeline::isMainLoopOp(op))
      return;
    auto *anchor = loop.getBody()->findAncestorOpInBlock(*op);
    if (anchor && (!lastMain || lastMain->isBeforeInBlock(anchor)))
      lastMain = anchor;
  });
  if (!lastMain)
    return false;
  bool found = false;
  loop.walk([&](Operation *op) {
    if (!isOutputStore(op))
      return;
    auto *anchor = loop.getBody()->findAncestorOpInBlock(*op);
    if (anchor && lastMain != anchor && lastMain->isBeforeInBlock(anchor))
      found = true;
  });
  return found;
}

static bool sameDomain(scf::ForOp a, scf::ForOp b) {
  // Only accept shared SSA bounds; equal-looking loads need a separate proof.
  return a.getLowerBound() == b.getLowerBound() &&
         a.getUpperBound() == b.getUpperBound() && a.getStep() == b.getStep();
}

static void fallback(ModuleOp module, StringRef reason) {
  LDBG(reason);
  module.emitRemark() << "output store protection: " << reason;
  CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
}

class ProtectOutputStorePass
    : public PassWrapper<ProtectOutputStorePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProtectOutputStorePass)
  StringRef getArgument() const override { return "protect-output-store"; }
  StringRef getDescription() const override {
    return "Protect Vector output stores before cross-core buffer reuse";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect>();
  }
  void runOnOperation() override {
    auto module = getOperation();
    if (CVPipeline::hasFallbackAttr(module))
      return;
    SmallVector<LoopPair> plans;
    for (auto func : module.getOps<func::FuncOp>()) {
      SmallVector<scf::ForOp> vectors, cubes;
      func.walk([&](scf::ForOp loop) {
        if (CVPipeline::isMainLoopOp(loop) || loop->hasAttr(kProtected))
          return;
        // Direct scope children have no divergent enclosing branch predicates.
        if (!isa<scope::ScopeOp>(loop->getParentOp()))
          return;
        auto core = getCore(loop);
        if (core == hivm::TCoreType::VECTOR && hasOutputTail(loop))
          vectors.push_back(loop);
        if (core == hivm::TCoreType::CUBE)
          cubes.push_back(loop);
      });
      for (auto vector : vectors) {
        bool unsupported = false;
        vector.walk([&](scf::WhileOp op) {
          if (CVPipeline::isMainLoopOp(op))
            unsupported = true;
        });
        if (unsupported) {
          fallback(module, "while main loop requires a separate pairing proof");
          return;
        }
        auto ids = getMainLoops(vector);
        SmallVector<scf::ForOp> matches;
        for (auto cube : cubes) {
          auto cubeIds = getMainLoops(cube);
          if (ids == cubeIds && sameDomain(vector, cube) &&
              vector->getParentOp()->getBlock() ==
                  cube->getParentOp()->getBlock())
            matches.push_back(cube);
        }
        if (matches.size() != 1 || llvm::any_of(plans, [&](const LoopPair &p) {
              return p.cube == matches.front();
            })) {
          fallback(module, "cannot uniquely pair output loop");
          return;
        }
        bool fixpipe = false;
        matches.front().walk([&](hivm::FixpipeOp) { fixpipe = true; });
        if (!fixpipe)
          continue;
        plans.push_back({vector, matches.front()});
      }
    }
    if (plans.empty())
      return;
    // Boundary flags have no transfer_id: AllocMultiCache scans their ids
    // for its budget but does not duplicate their synchronization chain.
    llvm::SmallSet<int64_t, 16> used;
    bool dynamicFlag = false;
    module.walk([&](Operation *op) {
      if (isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(op) &&
          !op->getAttrOfType<IntegerAttr>("static_flag_id") &&
          !op->getAttrOfType<IntegerAttr>("flag"))
        dynamicFlag = true;
      if (auto attr = op->getAttrOfType<IntegerAttr>("static_flag_id"))
        used.insert(attr.getInt());
      if (auto attr = op->getAttrOfType<IntegerAttr>("flag"))
        used.insert(attr.getInt());
      if (auto sync = dyn_cast<hivm::SyncBlockOp>(op))
        if (auto id = sync.getFlagId())
          used.insert(id->getInt());
    });
    SmallVector<int64_t> flags;
    for (int64_t id = 0; id <= FlagIdManager::MAX_FLAG_ID; ++id)
      if (!used.contains(id))
        flags.push_back(id);
    if (dynamicFlag || flags.size() < plans.size()) {
      fallback(module, "no free flag for output store protection");
      return;
    }
    OpBuilder builder(module.getContext());
    auto src = hivm::PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_MTE3);
    auto dst = hivm::PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_FIX);
    auto vector =
        hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
    auto cube =
        hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
    for (auto [pair, flag] : llvm::zip(plans, flags)) {
      auto id = builder.getI64IntegerAttr(flag);
      builder.setInsertionPoint(pair.vector);
      builder.create<hivm::SyncBlockSetOp>(pair.vector.getLoc(), vector, src,
                                           dst, id);
      builder.setInsertionPoint(pair.vector.getBody()->getTerminator());
      builder.create<hivm::SyncBlockSetOp>(pair.vector.getLoc(), vector, src,
                                           dst, id);
      builder.setInsertionPointToStart(pair.cube.getBody());
      builder.create<hivm::SyncBlockWaitOp>(pair.cube.getLoc(), cube, src, dst,
                                            id);
      builder.setInsertionPointAfter(pair.cube);
      builder.create<hivm::SyncBlockWaitOp>(pair.cube.getLoc(), cube, src, dst,
                                            id);
      for (auto loop : {pair.vector, pair.cube}) {
        loop->setAttr(kProtected, builder.getUnitAttr());
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createProtectOutputStorePass() {
  return std::make_unique<ProtectOutputStorePass>();
}
void mlir::triton::registerProtectOutputStorePasses() {
  registerPass(createProtectOutputStorePass);
}
