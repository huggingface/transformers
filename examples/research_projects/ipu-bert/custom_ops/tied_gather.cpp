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
#include <popart/region.hpp>
#include <popart/op.hpp>
#include <popart/op/gather.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/op/gatherx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/util.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Cast.hpp>

namespace CustomOperators {
  const popart::OperatorIdentifier TiedGather = {"ai.graphcore", "TiedGather", 1};
} // namespace CustomOperators

class TiedGatherOp;
class TiedGatherGradOp;

class TiedGatherGradOp : public popart::GatherGradOp {
public:
  TiedGatherGradOp(const popart::GatherOp &op, int64_t axis_)
      : popart::GatherGradOp(op, axis_),
        fwd_op(&op) {}
  const popart::GatherOp *fwd_op;
};

class TiedGatherOp : public popart::GatherOp {
public:
  TiedGatherOp(int64_t axis_, const popart::Op::Settings &settings_)
      : popart::GatherOp(CustomOperators::TiedGather, axis_, settings_) {}
  bool check_indices = true;

  std::vector<std::unique_ptr<Op>> getGradOps() {
    std::vector<std::unique_ptr<Op>> result;
    result.push_back(std::make_unique<TiedGatherGradOp>(*this, getAxis()));
    result[0]->pruneable = false;
    return result;
  }
};

class TiedGatherOpx : public popart::popx::Opx {
public:
  TiedGatherOpx(popart::Op *op, popart::popx::Devicex *devicex) : popart::popx::Opx(op, devicex) {
    verifyOp<TiedGatherOp>(op, CustomOperators::TiedGather);
    // We always want this to layout its inputs
    inputCreatorPriority = std::numeric_limits<double>::max();
  }

  bool createsEquiv(int, const popart::popx::Opx *, int) const final { return false; }

  std::set<popart::TensorId> mustExistBeforeCreate(int) const final { return {}; }

  popart::popx::InputCreatorType getInputCreatorType(int index0) const final {
    return index0 == TiedGatherOp::dataInIndex() ? popart::popx::InputCreatorType::CanCreate
                                                 : popart::popx::Opx::getInputCreatorType(index0);
  }

  poplar::Tensor createInput(popart::InIndex index,
                             const poplar::DebugNameAndId &dnai) const final {
    popart::logging::debug("TiedGather asked to create index {}: name {}", index, dnai);
    if (index != TiedGatherOp::dataInIndex()) {
      throw popart::error("CustomOps Error: GatherOpx::createInput Cannot create input {}", index);
    }

    auto inputInfo = inInfo(TiedGatherOp::indicesInIndex());
    auto weightInfo = inInfo(TiedGatherOp::dataInIndex());

    unsigned inputSize = inputInfo.nelms();
    unsigned inChannels = weightInfo.dim(getOp<TiedGatherOp>().getAxis());
    unsigned outChannels = weightInfo.nelms() / inChannels;

    std::vector<std::size_t> lhsShape = {inputSize, inChannels};
    std::vector<std::size_t> rhsShape = {inChannels, outChannels};

    return poplin::createMatMulInputRHS(graph(),
                                        popart::popx::popType(weightInfo),
                                        lhsShape,
                                        rhsShape,
                                        dnai,
                                        {},
                                        &dv_p->matmulCache);
  }

  // Identical to popart::opx::GatherOpx::grow however:
  //    1) uses popops::gather instead of popops::multislice
  //    2) range checks the indices and masks those out of range
  void grow(poplar::program::Sequence &prog) const final {
    const auto indicesShape = inShape(TiedGatherOp::indicesInIndex());
    const auto outputShape =
        popart::vXtoY<int64_t, std::size_t>(outShape(TiedGatherOp::outIndex()));

    auto op       = getOp<TiedGatherOp>();
    unsigned axis = op.getAxis();
    auto indices  = getInTensor(TiedGatherOp::indicesInIndex());
    auto data     = getInTensor(TiedGatherOp::dataInIndex());

    // If there are no indices, return an empty tensor of the appropriate
    // shape
    if (indices.numElements() == 0) {
      auto result = graph().addVariable(
          data.elementType(), outputShape, debugContext("result"));

      setOutTensor(TiedGatherOp::outIndex(), result);
    } else {
      // Flatten the scalar indices.
      auto offsets = indices.flatten();
      // reinterpret the indices as unsigned int. This assumes negative indices.
      // are impossible.
      offsets = offsets.reinterpret(poplar::UNSIGNED_INT);

      // Place the gather axis at the front.
      data = data.dimShufflePartial({0}, {axis});
      // Store the shape for later.
      auto tmp_shape = data.shape();
      // Flatten the other dimensions.
      data = data.flatten(1, data.rank());

      // Change (2)
      poplar::Tensor mask;
      if (op.check_indices) {
        auto gather_size = data.shape()[0];
        mask = popops::lt(graph(), offsets, static_cast<unsigned>(gather_size), prog, debugContext("mask<size"));
        auto indices_mask = popops::cast(graph(), mask, offsets.elementType(), prog, debugContext("mask_castInt"));
        offsets = popops::mul(graph(), offsets, indices_mask, prog, debugContext("masked_indices"));
      }

      // Change (1)
      auto result = popops::gather(graph(),
                                   data,
                                   offsets,
                                   0,
                                   prog,
                                   popops::GatherParams(),
                                   debugContext());
      
      // Change (2)
      if (op.check_indices) {
        auto out_mask = popops::cast(graph(), mask, data.elementType(), prog, debugContext("mask_cast"));
        popops::mulInPlace(graph(), result, out_mask.expand({1}), prog, debugContext("masked_result"));
      }

      // Reshape the result to "unflatten" the other dimensions.
      tmp_shape.front() = result.dim(0);
      result            = result.reshape(tmp_shape);
      // Put the gather axis dimension back in the right place.
      result = result.dimShufflePartial({axis}, {0});

      // Reshape into the expected ONNX shape.
      result = result.reshape(outputShape);

      setOutTensor(TiedGatherOp::outIndex(), result);
    }
  }
};

static popart::popx::OpxCreator<TiedGatherOpx>
    tiedGatherOpxCreator(CustomOperators::TiedGather);
