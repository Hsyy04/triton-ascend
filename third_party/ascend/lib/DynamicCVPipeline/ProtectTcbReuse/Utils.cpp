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

#include "ascend/include/DynamicCVPipeline/ProtectTcbReuse/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "protect-tcb-reuse"
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace mlir::triton;

namespace {
struct LoopPair {
  scf::ForOp vectorLoop;
  scf::ForOp cubeLoop;
  Operation *vectorMainLoop;
  Operation *cubeMainLoop;
};

struct LoopCandidates {
  SmallVector<scf::ForOp> vectorLoops;
  SmallVector<scf::ForOp> cubeLoops;
};

// Compute subtree facts once per candidate. Candidate vectors retain discovery
// order; membership sets are never used to decide pairing or flag order.
struct LoopInfo {
  scf::ForOp loop;
  Operation *firstMainLoop;
  llvm::SmallSet<int64_t, 4> mainLoopIds;
  unsigned mainLoopCount = 0;
  bool hasWhileMainLoop = false;
  bool hasFixpipe = false;
};

static std::optional<hivm::TCoreType> getCore(Operation *mainLoop) {
  auto scope = mainLoop->getParentOfType<scope::ScopeOp>();
  if (!scope)
    return std::nullopt;
  auto coreType =
      scope->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
  if (!coreType)
    return std::nullopt;
  return coreType.getTcoretype();
}

static LoopCandidates
collectLoopCandidates(func::FuncOp function,
                      tcbReuse::LoopPredicate matchesCandidate) {
  LoopCandidates candidates;
  llvm::SmallPtrSet<Operation *, 16> seenVectorLoops, seenCubeLoops;
  function.walk([&](Operation *mainLoop) {
    if (!CVPipeline::isMainLoopOp(mainLoop))
      return;
    auto core = getCore(mainLoop);
    // Several serial main loops can lead to the same enclosing loop.
    for (auto outerLoop = dyn_cast_or_null<scf::ForOp>(mainLoop->getParentOp());
         outerLoop;
         outerLoop = dyn_cast<scf::ForOp>(outerLoop->getParentOp())) {
      if (CVPipeline::isMainLoopOp(outerLoop))
        continue;
      if (core == hivm::TCoreType::VECTOR &&
          seenVectorLoops.insert(outerLoop.getOperation()).second &&
          matchesCandidate(outerLoop))
        candidates.vectorLoops.push_back(outerLoop);
      if (core == hivm::TCoreType::CUBE &&
          seenCubeLoops.insert(outerLoop.getOperation()).second)
        candidates.cubeLoops.push_back(outerLoop);
    }
  });
  return candidates;
}

static LoopInfo analyzeLoop(scf::ForOp loop) {
  LoopInfo info{loop, tcbReuse::getFirstMainLoop(loop), {}};
  loop.walk([&](Operation *bodyOperation) {
    if (CVPipeline::isMainLoopOp(bodyOperation))
      ++info.mainLoopCount;
    if (auto mainLoopId =
            bodyOperation->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop))
      info.mainLoopIds.insert(mainLoopId.getInt());
    info.hasWhileMainLoop |= isa<scf::WhileOp>(bodyOperation) &&
                             CVPipeline::isMainLoopOp(bodyOperation);
    info.hasFixpipe |= isa<hivm::FixpipeOp>(bodyOperation);
  });
  return info;
}

static bool sameDomain(scf::ForOp vectorLoop, scf::ForOp cubeLoop) {
  // Only accept shared SSA bounds; equal-looking loads need a separate proof.
  return vectorLoop.getLowerBound() == cubeLoop.getLowerBound() &&
         vectorLoop.getUpperBound() == cubeLoop.getUpperBound() &&
         vectorLoop.getStep() == cubeLoop.getStep();
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
static void skipProtection(ModuleOp module, StringRef protectionName,
                           StringRef reason) {
  for (auto function : module.getOps<func::FuncOp>())
    LOG_DEBUG("kernel '" << function.getSymName() << "' is not protected by "
                         << protectionName << " synchronization: " << reason);
}

static FailureOr<const LoopInfo *>
findUniqueCubeLoop(const LoopInfo &vectorInfo, ArrayRef<LoopInfo> cubeLoops,
                   const llvm::SmallPtrSetImpl<Operation *> &plannedCubeLoops) {
  const LoopInfo *matchedCube = nullptr;
  for (const auto &cubeInfo : cubeLoops) {
    if (cubeInfo.mainLoopCount != 2 ||
        vectorInfo.mainLoopIds != cubeInfo.mainLoopIds ||
        !sameLoopNest(vectorInfo.loop, cubeInfo.loop))
      continue;
    if (matchedCube)
      return failure();
    matchedCube = &cubeInfo;
  }
  if (!matchedCube || plannedCubeLoops.contains(matchedCube->loop))
    return failure();
  return matchedCube;
}

static FailureOr<SmallVector<LoopPair>>
collectProtectionPlans(ModuleOp module, StringRef protectionName,
                       tcbReuse::LoopPredicate matchesCandidate) {
  SmallVector<LoopPair> plans;
  llvm::SmallPtrSet<Operation *, 16> plannedCubeLoops;
  for (auto function : module.getOps<func::FuncOp>()) {
    auto candidates = collectLoopCandidates(function, matchesCandidate);
    if (candidates.vectorLoops.empty())
      continue;
    SmallVector<LoopInfo> cubeLoops;
    for (auto cubeLoop : candidates.cubeLoops)
      cubeLoops.push_back(analyzeLoop(cubeLoop));
    for (auto vectorLoop : candidates.vectorLoops) {
      auto vectorInfo = analyzeLoop(vectorLoop);
      // Count actual main-loop operations, not distinct logical IDs.
      if (vectorInfo.mainLoopCount != 2)
        continue;
      if (vectorInfo.hasWhileMainLoop) {
        skipProtection(module, protectionName,
                       "while main loop requires a separate pairing proof");
        return failure();
      }
      auto cubeMatch =
          findUniqueCubeLoop(vectorInfo, cubeLoops, plannedCubeLoops);
      if (failed(cubeMatch)) {
        skipProtection(module, protectionName,
                       "cannot uniquely pair enclosing loop");
        return failure();
      }
      const auto &cubeInfo = **cubeMatch;
      if (!cubeInfo.hasFixpipe)
        continue;
      if (!vectorInfo.firstMainLoop || !cubeInfo.firstMainLoop) {
        skipProtection(module, protectionName,
                       "no direct main loop synchronization anchor");
        return failure();
      }
      plans.push_back({vectorLoop, cubeInfo.loop, vectorInfo.firstMainLoop,
                       cubeInfo.firstMainLoop});
      plannedCubeLoops.insert(cubeInfo.loop);
    }
  }
  return plans;
}

static FailureOr<SmallVector<int64_t>>
findAvailableFlags(ModuleOp module, size_t requiredCount) {
  llvm::SmallSet<int64_t, 16> usedFlags;
  bool hasDynamicFlag = false;
  module.walk([&](Operation *operation) {
    if (isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(operation) &&
        !operation->getAttrOfType<IntegerAttr>("static_flag_id") &&
        !operation->getAttrOfType<IntegerAttr>("flag"))
      hasDynamicFlag = true;
    if (auto flagId = operation->getAttrOfType<IntegerAttr>("static_flag_id"))
      usedFlags.insert(flagId.getInt());
    if (auto flagId = operation->getAttrOfType<IntegerAttr>("flag"))
      usedFlags.insert(flagId.getInt());
    if (auto sync = dyn_cast<hivm::SyncBlockOp>(operation))
      if (auto flagId = sync.getFlagId())
        usedFlags.insert(flagId->getInt());
  });
  if (hasDynamicFlag)
    return failure();
  // Preserve ascending allocation, including holes below reserved flag 15.
  // Boundary syncs have no transfer_id and are not copied per buffer slot.
  SmallVector<int64_t> flags;
  for (int64_t flagId = 0;
       flagId <= FlagIdManager::MAX_FLAG_ID && flags.size() < requiredCount;
       ++flagId)
    if (!usedFlags.contains(flagId))
      flags.push_back(flagId);
  if (flags.size() != requiredCount)
    return failure();
  return flags;
}

static void insertSynchronization(ModuleOp module, ArrayRef<LoopPair> plans,
                                  ArrayRef<int64_t> flags,
                                  hivm::PIPE sourcePipe) {
  OpBuilder builder(module.getContext());
  auto source = hivm::PipeAttr::get(module.getContext(), sourcePipe);
  auto destination =
      hivm::PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_FIX);
  auto vectorCore =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto cubeCore =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  // Serial main loops share one handshake per outer iteration.
  for (auto [pair, flag] : llvm::zip(plans, flags)) {
    auto flagId = builder.getI64IntegerAttr(flag);
    builder.setInsertionPoint(pair.vectorMainLoop);
    builder.create<hivm::SyncBlockSetOp>(pair.vectorLoop->getLoc(), vectorCore,
                                         source, destination, flagId);
    builder.setInsertionPoint(pair.cubeMainLoop);
    builder.create<hivm::SyncBlockWaitOp>(pair.cubeLoop->getLoc(), cubeCore,
                                          source, destination, flagId);
  }
}
} // namespace

Operation *mlir::triton::tcbReuse::getFirstMainLoop(scf::ForOp outerLoop) {
  auto &body = *outerLoop.getBody();
  auto firstMainLoop = llvm::find_if(body, [](Operation &bodyOperation) {
    return CVPipeline::isMainLoopOp(&bodyOperation);
  });
  return firstMainLoop == body.end() ? nullptr : &*firstMainLoop;
}

void mlir::triton::tcbReuse::protectTcbReuse(ModuleOp module,
                                             StringRef protectionName,
                                             hivm::PIPE sourcePipe,
                                             LoopPredicate matchesCandidate) {
  if (CVPipeline::hasFallbackAttr(module))
    return;
  auto plans = collectProtectionPlans(module, protectionName, matchesCandidate);
  if (failed(plans) || plans->empty())
    return;
  auto flags = findAvailableFlags(module, plans->size());
  if (failed(flags)) {
    skipProtection(module, protectionName,
                   "no free flag for tcb reuse protection");
    return;
  }
  // Commit only after every function and resource has been validated.
  insertSynchronization(module, *plans, *flags, sourcePipe);
}
