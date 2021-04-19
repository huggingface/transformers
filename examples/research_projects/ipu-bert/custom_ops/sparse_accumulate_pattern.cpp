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

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/topocons.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/accumulate.hpp>
#include <iostream>

#include "sparse_accumulate.cpp"

// This pattern replaces:
//  GatherGradOp -> SGD1Accumulate 
//          with
//  SparseAccumulate

class SparseAccumulatePattern : public popart::PreAliasPattern {
public:
  bool matches(popart::Op *op) const override {
    if (op->isConvertibleTo<popart::GatherGradOp>()) {
        popart::Tensor *gradient = op->outTensor(popart::GatherGradOp::gradOutIndex());
        for (popart::Op *consumer : gradient->consumers.getOps()) {
            if (consumer->isConvertibleTo<popart::AccumulateOp>()) {
                return true;
            }
        }
    }
    return false;
  }

  std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

  bool apply(popart::Op *op) const override {
    auto &ir = op->getIr();
    auto &graph = op->getGraph();

    popart::GatherGradOp *gather_grad = dynamic_cast<popart::GatherGradOp *>(op);
    popart::AccumulateOp *dense_accl;

    popart::Tensor *gradient = op->outTensor(popart::GatherGradOp::gradOutIndex());
    for (popart::Op *consumer : gradient->consumers.getOps()) {
        if (consumer->isConvertibleTo<popart::AccumulateOp>() &&
            !consumer->isConvertibleTo<SparseAccumulateOp>()) {
            dense_accl = dynamic_cast<popart::AccumulateOp *>(consumer);
            break;
        }
    }

    auto accl_type = dense_accl->getAccumulationType();
    if (accl_type != popart::AccumulationType::Add ||
        accl_type != popart::AccumulationType::DampenedAdd) {
        return false;
    }

    popart::TensorId accl_id = dense_accl->inId(popart::AccumulateOp::getVarToUpdateInIndex());

    auto sparse_accl_up = std::make_unique<SparseAccumulateOp>(
        accl_id,
        dense_accl->getFactor(),
        gather_grad->getAxis(),
        popart::Op::Settings(graph, dense_accl->name() + "_accumulate"));

    auto sparse_accl = sparse_accl_up.get();
    transferBaseProperties(dense_accl, sparse_accl);
    graph.moveIntoGraph(std::move(sparse_accl_up));

    // Inputs
    // Accumulator
    sparse_accl->connectInTensor(SparseAccumulateOp::getVarToUpdateInIndex(),
                                 accl_id);
    // Gradients
    sparse_accl->connectInTensor(SparseAccumulateOp::getUpdaterInIndex(),
                                 gather_grad->inId(popart::GatherGradOp::gradInIndex()));
    // Scale
    if (!dense_accl->getFactor().isConst()) {
        sparse_accl->connectInTensor(
            // the index at which the dampening scale factor is received,
            SparseAccumulateOp::getDpsf1InIndex(),
            // the name of the dampening scale factor
            dense_accl->inId(popart::AccumulateOp::getFactorInIndex()));
    }
    // Indices
    sparse_accl->connectInTensor(SparseAccumulateOp::getIndicesInIndex(),
                                 gather_grad->inId(popart::GatherGradOp::indicesInIndex()));

    auto outId = dense_accl->outId(popart::AccumulateOp::getUpdatedVarOutIndex());
    auto gradId = gather_grad->outId(popart::GatherGradOp::gradOutIndex());

    // Transfer TopoCons
    graph.topoCons->transfer(gather_grad, sparse_accl);
    graph.topoCons->transfer(dense_accl, sparse_accl);

    // Delete the replaced ops
    dense_accl->disconnectAllInputs();
    dense_accl->disconnectAllOutputs();
    graph.eraseOp(dense_accl->id);
    
    gather_grad->disconnectAllInputs();
    gather_grad->disconnectAllOutputs();
    graph.eraseOp(gather_grad->id);

    // Outputs
    // Connect the updated accl
    sparse_accl->connectOutTensor(SparseAccumulateOp::getUpdatedVarOutIndex(),
                                  outId);
    // remove the gatherGrad output
    graph.getTensors().remove(gradId);

    // Finalise sparse op
    sparse_accl->setup();

    return true;
  }
};

static popart::PatternCreator<SparseAccumulatePattern> sparsesgd1PatternCreator("SparseAccumulatePattern", true);
