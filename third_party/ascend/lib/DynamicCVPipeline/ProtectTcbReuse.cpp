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

#include "ascend/include/DynamicCVPipeline/ProtectTcbReuse.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "protect-tcb-reuse"
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {
// Policy  Trigger outside main_loop          Source -> destination
// [VF]    Vector compute before first loop   PIPE_V -> PIPE_FIX
// [ST]    Direct GM writeback in outer body  PIPE_MTE3 -> PIPE_FIX
enum class ProtectionKind {
  /// Order pre-loop Vector computation before Cube FIX writes.
  VF,
  /// Order output writeback from earlier work before Cube FIX writes.
  ST,
};

static StringRef getProtectionName(ProtectionKind kind) {
  return kind == ProtectionKind::VF ? "vf tcb reuse" : "store tcb reuse";
}

// Match loops by corresponding loop nests, logical main-loop IDs, and a
// common enclosing block for their scopes. This structural pairing does not
// depend on kernel names or prove physical UB aliasing after backend
// allocation.
struct LoopPair {
  scf::ForOp vector;
  scf::ForOp cube;
  Operation *vectorMainLoop;
  Operation *cubeMainLoop;
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

// Compare logical IDs rather than traversal order. Several serial main loops
// can share one outer-loop synchronization lifecycle.
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

// Use a direct child so synchronization stays outside the main loop and runs
// once per outer iteration, including when the main loop has zero iterations.
static Operation *getFirstMainLoop(scf::ForOp outerLoop) {
  auto &body = *outerLoop.getBody();
  auto firstMain = llvm::find_if(body, [](Operation &bodyOperation) {
    return CVPipeline::isMainLoopOp(&bodyOperation);
  });
  return firstMain == body.end() ? nullptr : &*firstMain;
}

// Recognize computation that runs on the Vector pipe at this TA stage.
// Scalar preparation, tensor constants, memory views, and DMA copies do not
// establish VF work. Lowered HIVM operations expose their pipe explicitly.
static bool isVFOperation(Operation &computation) {
  if (CVPipeline::isSyncOp(&computation))
    return false;
  if (auto pipeOperation = dyn_cast<hivm::OpPipeInterface>(&computation))
    return pipeOperation.getPipe() == hivm::PIPE::PIPE_V;
  if (isa<linalg::LinalgOp>(&computation))
    return !isa<linalg::MatmulOp, linalg::CopyOp>(&computation);
  if (auto call = dyn_cast<func::CallOp>(&computation)) {
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    return callee && callee->hasAttr("hivm.vector_function");
  }
  if (!isa<arith::ArithDialect, math::MathDialect>(computation.getDialect()) ||
      isa<arith::ConstantOp>(&computation))
    return false;
  return llvm::any_of(computation.getResultTypes(), [](Type resultType) {
    return isa<TensorType, VectorType>(resultType);
  });
}

static bool hasVFBeforeMainLoop(scf::ForOp outerLoop) {
  auto *firstMain = getFirstMainLoop(outerLoop);
  if (!firstMain)
    return false;
  return llvm::any_of(
      llvm::make_range(outerLoop.getBody()->begin(), firstMain->getIterator()),
      [](Operation &bodyOperation) { return isVFOperation(bodyOperation); });
}

static bool sameDomain(scf::ForOp a, scf::ForOp b) {
  // Only accept shared SSA bounds; equal-looking loads need a separate proof.
  return a.getLowerBound() == b.getLowerBound() &&
         a.getUpperBound() == b.getUpperBound() && a.getStep() == b.getStep();
}

// SeparateCVScope clones nested loop bounds, so their SSA values can differ.
// Match those loops by preserved control-flow block IDs, then require matching
// enclosing for-loops up to scopes in one block with shared outermost bounds.
// Branch-enclosed loops are excluded because their predicates need a proof.
static bool sameLoopNest(scf::ForOp vectorLoop, scf::ForOp cubeLoop) {
  while (true) {
    auto vectorParent = dyn_cast<scf::ForOp>(vectorLoop->getParentOp());
    auto cubeParent = dyn_cast<scf::ForOp>(cubeLoop->getParentOp());
    if (!vectorParent || !cubeParent) {
      return !vectorParent && !cubeParent &&
             isa<scope::ScopeOp>(vectorLoop->getParentOp()) &&
             isa<scope::ScopeOp>(cubeLoop->getParentOp()) &&
             vectorLoop->getParentOp()->getBlock() ==
                 cubeLoop->getParentOp()->getBlock() &&
             sameDomain(vectorLoop, cubeLoop);
    }
    auto vectorId =
        vectorLoop->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    auto cubeId = cubeLoop->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (!sameDomain(vectorLoop, cubeLoop) && (!vectorId || vectorId != cubeId))
      return false;
    vectorLoop = vectorParent;
    cubeLoop = cubeParent;
  }
}

// This protection is optional: leave IR unchanged and let dynamic CV proceed.
static void skipProtection(ModuleOp module, ProtectionKind kind,
                           StringRef reason) {
  for (auto function : module.getOps<func::FuncOp>())
    LOG_DEBUG("kernel '" << function.getSymName() << "' is not protected by "
                         << getProtectionName(kind)
                         << " synchronization: " << reason);
}

static void protectTcbReuse(ModuleOp module, ProtectionKind kind) {
  if (CVPipeline::hasFallbackAttr(module))
    return;
  // Validate every pair and reserve enough flags before changing IR. A
  // later failure must not leave a partially emitted synchronization plan.
  SmallVector<LoopPair> plans;
  for (auto func : module.getOps<func::FuncOp>()) {
    SmallVector<scf::ForOp> vectors, cubes;
    func.walk([&](Operation *mainLoop) {
      if (!CVPipeline::isMainLoopOp(mainLoop))
        return;
      auto core = getCore(mainLoop);
      // Walk outward from each main loop. Serial main loops may discover
      // the same enclosing loop, which must receive only one lifecycle.
      for (auto outerLoop =
               dyn_cast_or_null<scf::ForOp>(mainLoop->getParentOp());
           outerLoop;
           outerLoop = dyn_cast<scf::ForOp>(outerLoop->getParentOp())) {
        if (CVPipeline::isMainLoopOp(outerLoop))
          continue;
        if (core == hivm::TCoreType::VECTOR &&
            (kind == ProtectionKind::VF ? hasVFBeforeMainLoop(outerLoop)
                                        : hasOutputTail(outerLoop)) &&
            !llvm::is_contained(vectors, outerLoop))
          vectors.push_back(outerLoop);
        if (core == hivm::TCoreType::CUBE &&
            !llvm::is_contained(cubes, outerLoop))
          cubes.push_back(outerLoop);
      }
    });
    for (auto vector : vectors) {
      bool unsupported = false;
      vector.walk([&](scf::WhileOp op) {
        if (CVPipeline::isMainLoopOp(op))
          unsupported = true;
      });
      if (unsupported) {
        skipProtection(module, kind,
                       "while main loop requires a separate pairing proof");
        return;
      }
      auto ids = getMainLoops(vector);
      SmallVector<scf::ForOp> matches;
      for (auto cube : cubes) {
        auto cubeIds = getMainLoops(cube);
        if (ids == cubeIds && sameLoopNest(vector, cube))
          matches.push_back(cube);
      }
      // Require a unique match and prohibit two Vector loops from sharing
      // one Cube receiver, preserving one sender/receiver pair per flag.
      if (matches.size() != 1 || llvm::any_of(plans, [&](const LoopPair &p) {
            return p.cube == matches.front();
          })) {
        skipProtection(module, kind, "cannot uniquely pair output loop");
        return;
      }
      // A matching Cube loop without FIX has no target write to protect.
      bool fixpipe = false;
      matches.front().walk([&](hivm::FixpipeOp) { fixpipe = true; });
      if (!fixpipe)
        continue;
      auto *vectorMainLoop = getFirstMainLoop(vector);
      auto *cubeMainLoop = getFirstMainLoop(matches.front());
      if (!vectorMainLoop || !cubeMainLoop) {
        skipProtection(module, kind,
                       "no direct main loop synchronization anchor");
        return;
      }
      plans.push_back({vector, matches.front(), vectorMainLoop, cubeMainLoop});
    }
  }
  if (plans.empty())
    return;
  // Boundary syncs have no transfer_id, so AllocMultiCache accounts for
  // their IDs without duplicating the handshake for each buffer slot.
  // Reserve static IDs across the module. A set/wait without a known static
  // ID may alias any proposed flag, so reject the entire plan in that case.
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
  // Fill unused IDs instead of incrementing the maximum observed ID, which
  // may already be reserved flag 15. MAX_FLAG_ID excludes that reserved ID;
  // each boundary lifecycle uses one flag regardless of buffer slot count.
  SmallVector<int64_t> flags;
  for (int64_t id = 0; id <= FlagIdManager::MAX_FLAG_ID; ++id)
    if (!used.contains(id))
      flags.push_back(id);
  if (dynamicFlag || flags.size() < plans.size()) {
    skipProtection(module, kind, "no free flag for tcb reuse protection");
    return;
  }
  OpBuilder builder(module.getContext());
  auto srcPipe =
      kind == ProtectionKind::VF ? hivm::PIPE::PIPE_V : hivm::PIPE::PIPE_MTE3;
  auto src = hivm::PipeAttr::get(module.getContext(), srcPipe);
  auto dst = hivm::PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_FIX);
  auto vector =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto cube =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  // Place one source-pipe -> FIX handshake immediately before the first main
  // loop on each core. Serial main loops share it, preserving one set/wait per
  // outer iteration without delaying the scalar preparation before it.
  for (auto [pair, flag] : llvm::zip(plans, flags)) {
    auto id = builder.getI64IntegerAttr(flag);
    builder.setInsertionPoint(pair.vectorMainLoop);
    builder.create<hivm::SyncBlockSetOp>(pair.vector.getLoc(), vector, src, dst,
                                         id);
    builder.setInsertionPoint(pair.cubeMainLoop);
    builder.create<hivm::SyncBlockWaitOp>(pair.cube.getLoc(), cube, src, dst,
                                          id);
  }
}

class ProtectVFtcbResusePass
    : public PassWrapper<ProtectVFtcbResusePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProtectVFtcbResusePass)
  StringRef getArgument() const override { return "protect-vf-tcb-reuse"; }
  StringRef getDescription() const override {
    return "Protect pre-loop Vector computation before TCB reuse";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect>();
  }
  void runOnOperation() override {
    protectTcbReuse(getOperation(), ProtectionKind::VF);
  }
};

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
    protectTcbReuse(getOperation(), ProtectionKind::ST);
  }
};

class ProtectTcbReusePass
    : public PassWrapper<ProtectTcbReusePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProtectTcbReusePass)
  StringRef getArgument() const override { return "protect-tcb-reuse"; }
  StringRef getDescription() const override {
    return "Protect TCB reuse after Vector computation and output stores";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect>();
  }
  void runOnOperation() override {
    if (CVPipeline::hasFallbackAttr(getOperation()))
      return;
    OpPassManager pipeline(ModuleOp::getOperationName());
    pipeline.addPass(createProtectVFtcbResusePass());
    pipeline.addPass(createProtectSTtcbResusePass());
    if (failed(runPipeline(pipeline, getOperation())))
      LOG_DEBUG("tcb reuse protection pipeline did not complete");
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createProtectVFtcbResusePass() {
  return std::make_unique<ProtectVFtcbResusePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createProtectSTtcbResusePass() {
  return std::make_unique<ProtectSTtcbResusePass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createProtectTcbReusePass() {
  return std::make_unique<ProtectTcbReusePass>();
}

void mlir::triton::registerProtectTcbReusePasses() {
  registerPass(createProtectTcbReusePass);
  registerPass(createProtectVFtcbResusePass);
  registerPass(createProtectSTtcbResusePass);
}
