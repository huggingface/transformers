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

#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/opmanager.hpp>
#include <popart/op.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Encoding.hpp>

namespace CustomOperators {
    const popart::OperatorIdentifier OneHot = {"ai.graphcore", "OneHot", 1};
} // namespace CustomOperators

class OneHotOp : public popart::Op {
    int64_t num_classes;
public:
    OneHotOp(int64_t num_classes_, const popart::Op::Settings &settings_)
        : popart::Op(CustomOperators::OneHot, settings_), num_classes(num_classes_) {}

    void setup() {
        // auto outputShape = inInfo(0).shape();
        // outputShape.push_back(num_classes);
        // outInfo(0) = popart::TensorInfo(popart::DataType::FLOAT16, outputShape);
        outInfo(0) = inInfo(1);
    }

    float getSubgraphValue() const final { return getLowSubgraphValue(); }
    std::unique_ptr<Op> clone() const override {
        return std::make_unique<OneHotOp>(*this);
    }
};

class OneHotOpx : public popart::popx::Opx {
public:
    OneHotOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<OneHotOp>(op, CustomOperators::OneHot);
    }

    popart::popx::InputCreatorType getInputCreatorType(popart::InIndex idx) const {
        if (idx == 1)
            return popart::popx::InputCreatorType::CanUnwind;
        return popart::popx::Opx::getInputCreatorType(idx);
    }

    poplar::Tensor unwindTensorLayout(poplar::Tensor tensor, popart::InIndex, popart::OutIndex) const {
        return tensor;
    }

    popart::view::RegMap unwindRegion(popart::InIndex, popart::OutIndex) const {
        return [this](const popart::view::Region &r) {
            return popart::view::Regions(1, r);
        };
    }

    void grow(poplar::program::Sequence &prog) const final {
        auto indices = getInTensor(0);
        auto ref     = getInTensor(1);
        auto output  = graph().clone(ref, debugContext("output"));

        popops::encodeOneHot(
            graph(),
            indices.flatten(),
            output.reshapePartial(0, output.rank() - 1, {indices.numElements()}),
            prog,
            debugContext("onehot"));
        
        setOutTensor(0, output);
    }
};

static popart::popx::OpxCreator<OneHotOpx>
    OneHotOpxCreator(CustomOperators::OneHot);

static popart::OpDefinition OneHotOpDef({});

static popart::OpCreator<OneHotOp> OneHotOpCreator(
    popart::OpDefinitions({{CustomOperators::OneHot,
                            OneHotOpDef}}),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
        int64_t num_classes =
            oci.attributes.getAttribute<popart::Attributes::Int>("num_classes", -1);
        return std::unique_ptr<OneHotOp>(new OneHotOp(num_classes, oci.settings));
    },
    true);
