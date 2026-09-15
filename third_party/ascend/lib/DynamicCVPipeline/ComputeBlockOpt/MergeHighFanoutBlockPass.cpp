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
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

static constexpr const char *DEBUG_TYPE = "merge-high-fanout-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

static constexpr int MIN_CROSS_BLOCK_EDGE_COUNT = 16;

using namespace mlir;

namespace mlir {
namespace triton {

namespace {

/// Rewrite of MergeHighFanoutBlockPass.
///
/// For every MLIR Block that carries compute-block ids, build the inter-compute
/// dependency graph, topologically sort it from indegree-0 (producers first),
/// and try at most ONE merge per Block: for each producer `cur` in topo order,
/// find its biggest VECTOR_ONLY consumer `c` (the consumer fed by the most
/// distinct source ops in `cur`). If that count exceeds
/// `MIN_CROSS_BLOCK_EDGE_COUNT` and merging `cur` into `c` would not create a
/// cycle, re-id all of `cur`'s ops (in this Block) to `c` and move on to the
/// next MLIR Block.
///
/// Merge direction: producer `cur` -> consumer `c` (cur's ops take c's id).
/// The depth-based grouping ("same depth, >16 sources") is intentionally
/// deferred; the trigger currently uses a plain distinct-source count.

/// Build the producer->consumer adjacency used for topological sorting.
///
/// Fills `succ`: producer block id -> set of consumer block ids it feeds. Both
/// endpoints must carry a compute-block id (queried through `bm`) and live as
/// direct ops of `block`. Edges are SSA use-def only (operand's defining op).
static void buildBlockEdge(Block *block, CVPipeline::ComputeBlockIdManager &bm,
                           DenseMap<int, DenseSet<int>> &succ) {
  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    auto cIdOpt = bm.getBlockIdByOpOpt(&op);
    if (!cIdOpt.has_value()) {
      continue;
    }
    int c = *cIdOpt;
    for (Value operand : op.getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp == nullptr || defOp->getBlock() != block) {
        continue;
      }
      auto pIdOpt = bm.getBlockIdByOpOpt(defOp);
      if (!pIdOpt.has_value()) {
        continue;
      }
      int p = *pIdOpt;
      if (p == c) {
        continue;
      }
      succ[p].insert(c);
    }
  }
}

/// Kahn topological sort from indegree-0 nodes (producers first).
///
/// Returns true on success and fills `ordered` with a producers-first
/// permutation of `ids`. Returns false if the block-level graph contains a
/// cycle (some nodes never reach indegree 0); the caller must abort the Block.
static bool topoSort(const SmallVectorImpl<int> &ids,
                     const DenseMap<int, DenseSet<int>> &succ,
                     DenseMap<int, int> &indegree,
                     SmallVectorImpl<int> &ordered) {
  SmallVector<int> queue;
  for (int id : ids) {
    if (indegree[id] == 0) {
      queue.push_back(id);
    }
  }
  for (size_t i = 0; i < queue.size(); ++i) {
    int cur = queue[i];
    ordered.push_back(cur);
    auto it = succ.find(cur);
    if (it == succ.end()) {
      continue;
    }
    for (int c : it->second) {
      if (--indegree[c] == 0) {
        queue.push_back(c);
      }
    }
  }
  return ordered.size() == static_cast<size_t>(ids.size());
}

/// Collect the distinct compute-block ids carried by direct ops of `block`
/// (terminators skipped), in first-seen order. Ids are queried through `bm`.
static SmallVector<int>
collectComputeBlockIds(Block *block, CVPipeline::ComputeBlockIdManager &bm) {
  DenseSet<int> seen;
  SmallVector<int> ids;
  for (Operation &op : *block) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    auto idOpt = bm.getBlockIdByOpOpt(&op);
    if (!idOpt.has_value()) {
      continue;
    }
    if (seen.insert(*idOpt).second) {
      ids.push_back(*idOpt);
    }
  }
  return ids;
}

/// User-walk: build crossBlockSources[c] = the distinct source ops in `cur`
/// (from `curOps`) that feed consumer block `c`. Only VECTOR_ONLY producers and
/// VECTOR_ONLY direct users living in `block` are considered; self-edges
/// (c == cur) are dropped.
static DenseMap<int, DenseSet<Operation *>>
buildCrossBlockSources(Block *block, CVPipeline::ComputeBlockIdManager &bm,
                       int cur, const SmallVectorImpl<Operation *> &curOps) {
  DenseMap<int, DenseSet<Operation *>> crossBlockSources;
  for (Operation *op : curOps) {
    if (CVPipeline::getOpCoreType(op) != CVPipeline::CoreType::VECTOR_ONLY) {
      continue;
    }
    for (Operation *u : op->getUsers()) {
      if (u->getBlock() != block) {
        continue;
      }
      if (CVPipeline::getOpCoreType(u) != CVPipeline::CoreType::VECTOR_ONLY) {
        continue;
      }
      if (u->hasTrait<OpTrait::IsTerminator>()) {
        continue;
      }
      auto cIdOpt = bm.getBlockIdByOpOpt(u);
      if (!cIdOpt.has_value()) {
        continue;
      }
      int c = *cIdOpt;
      if (c == cur) {
        continue;
      }
      crossBlockSources[c].insert(op);
    }
  }
  return crossBlockSources;
}

class MergeHighFanoutBlockPass
    : public PassWrapper<MergeHighFanoutBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeHighFanoutBlockPass)

  MergeHighFanoutBlockPass() = default;

  StringRef getArgument() const override { return "merge-high-fanout-block"; }

  StringRef getDescription() const override {
    return "Merge a producer compute block into its biggest VECTOR_ONLY "
           "consumer "
           "block when the producer feeds the consumer through more than 16 "
           "distinct ops, without creating cycles.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    LOG_DEBUG("Before: " << *module);
    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);

    module.walk([&](Block *block) {
      // Step 1: collect the distinct compute-block ids present in this Block.
      SmallVector<int> ids = collectComputeBlockIds(block, bm);
      if (ids.size() < 2) {
        return;
      }

      // Step 2: build the producer->consumer blockEdge.
      DenseMap<int, DenseSet<int>> succ;
      buildBlockEdge(block, bm, succ);

      // Step 3: topologically sort from indegree-0 (producers first). Abort on
      // any block-level cycle.
      DenseMap<int, int> indegree;
      for (int id : ids) {
        indegree[id] = 0;
      }
      for (const auto &entry : succ) {
        for (int c : entry.second) {
          indegree[c]++;
        }
      }
      SmallVector<int> ordered;
      if (!topoSort(ids, succ, indegree, ordered)) {
        LOG_DEBUG("Cyclic blockEdge detected, skipping Block");
        return;
      }

      // Step 4: for each producer `cur` in topo order, find its biggest
      // VECTOR_ONLY consumer `c`; if cur feeds c through more than
      // MIN_CROSS_BLOCK_EDGE_COUNT distinct ops, merge cur into c (guarded by
      // willCreateCycle). One merge per Block, then return.
      for (int cur : ordered) {
        SmallVector<Operation *> curOps = bm.getOpsByBlockId(cur);

        // User-walk: crossBlockSources[c] = distinct source ops in cur feeding
        // c.
        DenseMap<int, DenseSet<Operation *>> crossBlockSources =
            buildCrossBlockSources(block, bm, cur, curOps);
        if (crossBlockSources.empty()) {
          continue;
        }

        int bestConsumer = -1;
        size_t bestCount = 0;
        for (const auto &entry : crossBlockSources) {
          int c = entry.first;
          size_t cnt = entry.second.size();
          if (bestConsumer == -1 || cnt > bestCount ||
              (cnt == bestCount && c < bestConsumer)) {
            bestConsumer = c;
            bestCount = cnt;
          }
        }
        if (bestConsumer == -1) {
          continue;
        }
        if (bestCount <= MIN_CROSS_BLOCK_EDGE_COUNT) {
          continue;
        }

        LOG_DEBUG("Trying merge producer block "
                  << cur << " into consumer block " << bestConsumer
                  << " (edge count: " << bestCount << ")");
        if (CVPipeline::willCreateCycle(curOps, memGraph, bestConsumer, bm)) {
          LOG_DEBUG("Merge would create cycle, skipping");
          continue;
        }
        for (Operation *op : curOps) {
          bm.updateBlockId(op, bestConsumer);
        }
        LOG_DEBUG("Merged block " << cur << " into block " << bestConsumer);
        return;
      }
    });

    LOG_DEBUG("After: " << *module);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMergeHighFanoutBlockPass() {
  return std::make_unique<MergeHighFanoutBlockPass>();
}

void registerMergeHighFanoutBlockPass() {
  PassRegistration<MergeHighFanoutBlockPass> reg;
}

} // namespace triton
} // namespace mlir
