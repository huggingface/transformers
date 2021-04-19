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
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensornames.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/add.hpp>
#include <popart/op/sum.hpp>
#include <popart/op/lamb.hpp>
#include <popart/op/adamvarupdate.hpp>

#include "utils.cpp"

// This Pattern finds Weights that have been serialised and are being
// updated in the Lamb Optimizer in slices. Transforming:
//
//   Slice(W)        U_sliced    }
//     | (R1)           | (R2)   }
//   LambSquare     LambSquare   }   x N
//           \      /            }
//         AdamVarUpdate         }
// Into:
//
//   Slice(W)        U_sliced    }
//     |                |        }
//   LambSquare     LambSquare   }   x N
//     |                |        }
//     Sum             Sum
//        \            /         
//        AdamVarUpdate          }   x N
// 
// A key property of LambSquare is that the output has not been sqrt yet, so it is valid
// to just Sum the outputs.

namespace {

bool produced_by_slice(popart::Tensor *t) {
    return t->hasProducer() && t->getProducer()->isConvertibleTo<popart::BaseSliceOp>();
}

bool consumed_by_add(popart::Tensor *t) {
    for (auto cons : t->consumers.getOps()) {
        if (cons->isConvertibleTo<popart::SumOp>()) {
            return true;
        }
    }
    return false;
}

}

class LambSerialisedWeightPattern : public popart::PreAliasPattern {
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

        auto lambsq = dynamic_cast<popart::LambSquareOp *>(op);
        if (lambsq) {
            bool only_a_slice = produced_by_slice(lambsq->inTensor(popart::LambSquareOp::getInIndex()));
            // Check we haven't already applied the pattern.
            return only_a_slice && !consumed_by_add(lambsq->outTensor(popart::LambSquareOp::getOutIndex()));
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    popart::SumOp *insert_sum_op(popart::Graph &graph,
                                 std::vector<popart::TensorId> &in_ids,
                                 popart::TensorId out_id,
                                 popart::Op *ref_op,
                                 std::string debug_name) const {
        auto sum_up = std::make_unique<popart::SumOp>(
            popart::Onnx::Operators::Sum_8,
            popart::Op::Settings(graph, debug_name));
        auto sum = sum_up.get();
        transferBaseProperties(ref_op, sum);
        graph.moveIntoGraph(std::move(sum_up));
        
        for (unsigned i = 0; i < in_ids.size(); i++) {
            sum->connectInTensor(i, in_ids[i]);
        }

        sum->createAndConnectOutTensor(popart::SumOp::getOutIndex(), out_id);
        sum->setup();

        sum->settings.executionContext = popart::ExecutionContext::AccumulateOuterFragment;
        sum->settings.excludePatterns.insert("SumToAdd");
        return sum;
    }

    bool apply(popart::Op *op) const override {
        auto &graph = op->getGraph();

        // To fix R1:
        //  1. Find root weight
        //  2. Find all LambSquareOp consumers
        //  3. Insert a SumOp between the consumers of LambSquareOp->outTensor(0) and their consumers

        // (1)
        auto root_weight = get_variable(op->inTensor(popart::LambSquareOp::getInIndex()));
        // (2)
        auto r1_lamb_ops = find_all_consumers<popart::LambSquareOp, popart::ExecutionContext::AccumulateOuterFragment>(root_weight);

        // Only Sum if there is more than one Op
        if (r1_lamb_ops.size() <= 1) {
            return true;
        }

        std::vector<popart::TensorId> r1_inputs(r1_lamb_ops.size());
        std::transform(r1_lamb_ops.begin(), r1_lamb_ops.end(), r1_inputs.begin(),
                       [](auto lambsq) { return lambsq->outId(popart::LambSquareOp::getOutIndex()); });

        popart::TensorId r1_output = popart::reservedLambR1SqPrefix() + root_weight->id;

        // (3)
        auto r1_sum = insert_sum_op(
            graph,
            r1_inputs,
            r1_output,
            op,
            "R1SerialisedSum");

        std::vector<popart::Op *> var_updates_to_search_for_r2;

        for (auto lamb_op : r1_lamb_ops) {
            auto tensor_to_replace = lamb_op->outTensor(popart::LambSquareOp::getOutIndex());
            for (auto cons : tensor_to_replace->consumers.getOps()) {
                if (cons->id != r1_sum->id) {
                    for (auto in_index : cons->input->indices(tensor_to_replace)) {
                        cons->disconnectInTensor(tensor_to_replace);
                        cons->connectInTensor(in_index, r1_output);
                    }
                    if (cons->isConvertibleTo<popart::AdamVarUpdateOp>()) {
                        var_updates_to_search_for_r2.push_back(cons);
                    }
                }
            }
        }

        // To fix R2:
        //  1. Start from the AdamVarUpdateOps from fixing R1
        //  2. Find all LambSquareOps in their producers on AdamVarUpdate::getR2InIndex()
        //  3. Insert a SumOp between the consumers of LambSquareOp->outTensor(0) and their consumers

        std::vector<popart::Op *> r2_lamb_ops;
        // (1)
        for (auto var_update : var_updates_to_search_for_r2) {
            // (2)
            auto r2_tensor = var_update->inTensor(popart::AdamVarUpdateOp::getLambR2SqInIndex());
            auto r2_op = search_producers_for<popart::LambSquareOp, popart::ExecutionContext::AccumulateOuterFragment>(r2_tensor, 3);
            if (r2_op == nullptr) {
                throw popart::error("CustomOps Error: Could not find the R2 LambSquareOp for AdamVarUpdate {}", var_update->debugName());
            }
            r2_lamb_ops.push_back(r2_op);
        }

        // (3)
        std::vector<popart::TensorId> r2_inputs(r2_lamb_ops.size());
        std::transform(r2_lamb_ops.begin(), r2_lamb_ops.end(), r2_inputs.begin(),
                       [](auto lambsq) { return lambsq->outId(popart::LambSquareOp::getOutIndex()); });

        popart::TensorId r2_output = popart::reservedLambR2SqPrefix() + root_weight->id;

        // (3)
        auto r2_sum = insert_sum_op(
            graph,
            r2_inputs,
            r2_output,
            op,
            "R2SerialisedSum");

        for (auto lamb_op : r2_lamb_ops) {
            auto tensor_to_replace = lamb_op->outTensor(popart::LambSquareOp::getOutIndex());
            for (auto cons : tensor_to_replace->consumers.getOps()) {
                if (cons->id != r2_sum->id) {
                    for (auto in_index : cons->input->indices(tensor_to_replace)) {
                        cons->disconnectInTensor(tensor_to_replace);
                        cons->connectInTensor(in_index, r2_output);
                    }
                }
            }
        }

        return true;
    }
};

static popart::PatternCreator<LambSerialisedWeightPattern> LambSerialisedWeightPatternCreator("LambSerialisedWeightPattern", true);