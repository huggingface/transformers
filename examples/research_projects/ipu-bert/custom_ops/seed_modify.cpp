// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <limits>

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/graph.hpp>
#include <popart/error.hpp>
#include <popart/util.hpp>
#include <popart/logging.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier SeedModify = {"ai.graphcore", "SeedModify", 1};
} // namespace CustomOperators

class SeedModifyOp;
class SeedModifyOpx;

class SeedModifyOp : public popart::Op {
public:
    uint32_t seed_modifier;

    SeedModifyOp(const uint32_t seed_modifier,
                 const Op::Settings &opSettings)
        : popart::Op(CustomOperators::SeedModify, opSettings),
          seed_modifier(seed_modifier) {
        // Schedule the Op as early as possible to make sure it doesn't breakup the
        // sequence of outlinable operations
        getSettings().schedulePriority = std::numeric_limits<double>::max();
    }

    bool isOutlineable() const final { return false; }

    void setup() final { outInfo(0) = inInfo(0); }

    static popart::InIndex getInputIndex() { return 0; }
    static popart::OutIndex getOutputIndex() { return 0; }
    float getSubgraphValue() const final { return getLowSubgraphValue(); }

    std::unique_ptr<popart::Op> clone() const {
        return std::make_unique<SeedModifyOp>(*this);
    }

    void appendOutlineAttributes(popart::OpSerialiserBase &os) const {
        popart::Op::appendOutlineAttributes(os);

        os.appendAttribute("seedModifier", seed_modifier);
    }
};

static constexpr uint32_t MODIFIER_SHIFT = 1000;

class SeedModifyOpx : public popart::popx::Opx {
public:
    SeedModifyOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<SeedModifyOp>(op, CustomOperators::SeedModify);
    }

    void grow(poplar::program::Sequence &prog) const {
        auto op = getOp<SeedModifyOp>();
        auto seed = getInTensor(SeedModifyOp::getInputIndex());
        auto seed_modifier = op.seed_modifier;

        auto modified = popops::add(graph(), seed, seed_modifier*MODIFIER_SHIFT, prog, debugContext());

        setOutTensor(SeedModifyOp::getOutputIndex(), modified);
    }
};

static popart::popx::OpxCreator<SeedModifyOpx> SeedModifyOpxCreator(CustomOperators::SeedModify);