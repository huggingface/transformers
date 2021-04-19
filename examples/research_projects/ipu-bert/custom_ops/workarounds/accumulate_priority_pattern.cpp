// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/accumulate.hpp>

// This Pattern set the scheduling priority on AccumulateOps involved in gradient accumulation to 'max'.
// This is due to the usage of implicit recomputation when pipelining. The scheduler cannot reason about the liveness
// of recomputed operations in the backwards pass so it will make a sub-optimial schedule when there
// are more than 1 recompute stages per IPU. By increasing the priority of these Ops we ensure they are
// scheduled as early as possible. Changing the schedule from:
//
//  recomp, bwd, recomp, bwd... accum, accum...
// to:
//  recomp, bwd, accum, recomp, bwd, accum...
//
// (T29047) Remove this once Pipeline configs support using explicit recompute.

class AccumulatePriorityPattern : public popart::PreAliasPattern {
public:
    bool matches(popart::Op *op) const override {
        auto &ir = op->getIr();
        // Don't run in inference
        if (!ir.canTrain()) {
            return false;
        }
        if (!ir.hasDecomposedOptimizers()) {
            return false;
        }

        auto accum = dynamic_cast<popart::AccumulateOp *>(op);
        if (accum) {
            return accum->settings.executionContext == popart::ExecutionContext::Normal && accum->settings.schedulePriority != std::numeric_limits<double>::max();
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        op->settings.schedulePriority = std::numeric_limits<double>::max();

        return true;
    }
};

static popart::PatternCreator<AccumulatePriorityPattern> AccumulatePriorityPatternCreator("AccumulatePriorityPattern", true);
