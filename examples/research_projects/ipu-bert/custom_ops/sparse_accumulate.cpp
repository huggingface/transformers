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

#include <algorithm>
#include <string>
#include <vector>
#include <memory>

#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/ir.hpp>
#include <popart/graph.hpp>
#include <popart/error.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/util.hpp>
#include <popart/logging.hpp>

#include <popart/popx/devicex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Cast.hpp>
#include <poputil/TileMapping.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier SparseAccumulate = {"ai.graphcore", "SparseAccumulate", 1};
} // namespace CustomOperators

class SparseAccumulateOp;
class SparseAccumulateOpx;

// Cannot subclass popart::SGD1AccumulateOp as the constructor hard codes the OperatorIdentifier
class SparseAccumulateOp : public popart::VarUpdateWithUpdaterOp {
public:
    unsigned axis;
    const popart::OptimizerValue initDpsf1;

    SparseAccumulateOp(const popart::TensorId &varToUpdate,
                           popart::OptimizerValue dpsf1,
                           const unsigned axis,
                           const Op::Settings &opSettings)
        : popart::VarUpdateWithUpdaterOp(CustomOperators::SparseAccumulate,
                                         opSettings),
          initDpsf1(dpsf1),
          axis(axis) {}

    static popart::InIndex getDpsf1InIndex() { return 2; }
    static popart::InIndex getIndicesInIndex() { return 3; }
    // Optional input of the original var. If present the accumulator will be created as a clone
    static popart::InIndex getOriginalVarInIndex() { return 4; }
    float getSubgraphValue() const final { return getLowSubgraphValue(); }

    std::unique_ptr<popart::Op>
    cloneWithNewName(const popart::TensorId &x) const {
        return std::make_unique<SparseAccumulateOp>(x, initDpsf1, axis, settings);
    }

    std::unique_ptr<popart::Op> clone() const {
        return std::make_unique<SparseAccumulateOp>(*this);
    }

    std::map<popart::InIndex, popart::TensorId> optimizerInputs() const {
        throw popart::error("CustomOps Error: Sparse SGD1 optimizer inputs not implemented");
    }

    void appendOutlineAttributes(popart::OpSerialiserBase &os) const {
        popart::Op::appendOutlineAttributes(os);

        if (initDpsf1.isConst()) {
            os.appendAttribute("const dampening scale factor", initDpsf1.val());
        }
        os.appendAttribute("axis", axis);
    }
};

class SparseAccumulateOpx : public popart::popx::Opx {
public:
    unsigned axis;
    SparseAccumulateOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex) {
        verifyOp<SparseAccumulateOp>(op, CustomOperators::SparseAccumulate);
        inputCreatorPriority = std::numeric_limits<double>::max();

        auto _op = getOp<SparseAccumulateOp>();
        axis = _op.axis;
    }

    popart::popx::InputCreatorType getInputCreatorType(int index0) const final {
        return index0 == SparseAccumulateOp::getVarToUpdateInIndex() ? 
            popart::popx::InputCreatorType::CanCreate : popart::popx::Opx::getInputCreatorType(index0);
    }

    std::set<popart::TensorId> mustExistBeforeCreate(int index0) const final {
        if (index0 == SparseAccumulateOp::getVarToUpdateInIndex() && hasInput(SparseAccumulateOp::getOriginalVarInIndex()))
            return {inId(SparseAccumulateOp::getOriginalVarInIndex())};
        return {}; 
    }

    poplar::Tensor createInput(popart::InIndex index,
                               const poplar::DebugNameAndId &dnai) const final {
        if (index != SparseAccumulateOp::getVarToUpdateInIndex()) {
            throw popart::error("CustomOps Error: SparseAccumulateOpx::createInput Cannot create input {}", index);
        }

        if (hasInput(SparseAccumulateOp::getOriginalVarInIndex())) {
            auto w = getInTensor(SparseAccumulateOp::getOriginalVarInIndex());
            return graph().clone(w, dnai);
        }  else {
            auto info = inInfo(SparseAccumulateOp::getVarToUpdateInIndex());
            const auto shape = info.shape_szt();

            // Perhaps should be a clone of the original weight tensor
            return popops::createGatherInput(graph(),
                                             popart::popx::popType(info),
                                             shape,
                                             static_cast<unsigned>(axis),
                                             popops::GatherParams{},
                                             dnai);
        }
    }

    void grow(poplar::program::Sequence &prog) const final {
        // If using tied weights, we want the dampening scale factor from the optimiser
        auto op = getOp<SparseAccumulateOp>();

        auto isConst = op.initDpsf1.isConst();

        auto accl = getInTensor(SparseAccumulateOp::getVarToUpdateInIndex());
        auto grad = getInTensor(SparseAccumulateOp::getUpdaterInIndex());
	grad = grad.reshape({grad.shape()[0] * grad.shape()[1], grad.shape()[2]});
        auto indices = getInTensor(SparseAccumulateOp::getIndicesInIndex());
	indices = indices.reshape({indices.shape()[0] * indices.shape()[1]});
        auto dpsf = isConst ? 
            getConst(accl.elementType(), {}, op.initDpsf1.val(), "ConstSparseDPSF") :
            getInTensor(SparseAccumulateOp::getDpsf1InIndex());

        if (dpsf.elementType() != accl.elementType()) {
            dpsf = popops::cast(graph(), dpsf, accl.elementType(), prog, debugContext("dpsf_cast"));
        }

        if (isConst && op.initDpsf1.val() == 0.0f) {
            throw popart::internal_error(
                "dpsf1 of 0 is not allowed, should have been caught in "
                "the Ir, dpsf1 of 0 could be caused by dampening of 1, which "
                "means the gradient is multiplied by 0 (no learning)");
        }

        grad = grad.expand({1 - axis});
        indices = indices.expand({1 - axis});

	if (grad.elementType() != accl.elementType()) {
          grad = popops::cast(graph(), grad, accl.elementType(), prog, debugContext("grad_cast"));
	}

        // Accumulate the updates into the target
        popops::multiUpdateAdd(graph(),
                               accl,
                               grad,
                               popops::cast(graph(), indices, poplar::UNSIGNED_INT, prog),
                               dpsf,
                               {axis},
                               {1},
                               prog,
                               popops::SlicePlan(),
                               poplar::OptionFlags(),
                               debugContext("nonConstSparseSGD1Accl"));

        // reference accl returned
        setOutTensor(SparseAccumulateOp::getUpdatedVarOutIndex(), accl);
    }
};

static popart::popx::OpxCreator<SparseAccumulateOpx> SparseAccumulateOpxCreator(CustomOperators::SparseAccumulate);
