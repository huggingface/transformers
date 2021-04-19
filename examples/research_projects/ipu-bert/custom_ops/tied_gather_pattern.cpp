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
#include <popart/op.hpp>
#include <popart/opidentifier.hpp>
#include <popart/topocons.hpp>
#include <popart/optimizer.hpp>
#include <popart/adam.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/logging.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/add.hpp>
#include <popart/op/subtract.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/transpose.hpp>
#include <popart/op/sgd1varupdate.hpp>
#include <popart/op/detach.hpp>

#include <map>

#include "sparse_accumulate.cpp"
#include "tied_gather.cpp"
#include "utils.cpp"
#include "params.hpp"

using SerialiseSettings = popart::MatMulBaseOp::SerialiseSettings;

// This pattern matches for graphs of the shape.
//
//              Weight
//             /     \     
//        Transpose   MatMul
//            |
// Indices --Gather
//
// And performs the following transformations:
//    1) Disable FullyConnectedPass on MatMul
//    2) Add Detach between the Gather and the Weight so no SGD ops are created (they will be added later by TiedGatherAccumulatePattern)
//    3) Replace Gather with TiedGather
// Resulting in:
//              Weight
//             /     \     
//        Transpose   MatMul
//            |
//          Detach
//            |
// Indices --TiedGather
//
// Conditionally, if MatMul is annotated with serialisation it will:
//    4) Replace Gather with N x TiedGather to match the serialisation on the MatMul
// Resulting in:
//    For serialisation factor: 2
//
//              Weight
//             /     \     
//        Transpose  MatMul
//            |
// Indices  Detach
//  |   |    |  |
//  |   |    | Slice--\ 
//  |   Sub -|------TiedGather
//  |        |              |
//  |       Slice--\        |
//  Sub ---------TiedGather |
//                        \ |
//                        Add
//
namespace {
bool produced_by_transpose(popart::Tensor *t) {
    return t->hasProducer() && t->getProducer()->isConvertibleTo<popart::TransposeBaseOp>();
}

/**
 * Change from PopART BERT:
 * This is hacky, but for now should identify the word embedding separately from the others.
 * Have the user set the vocab size from the Python model (see below), then check that dim0 == v.
 *
 *   ops = ctypes.cdll.LoadLibrary(so_path)
 *   # ...
 *   ops.setvocabSize(config.vocab_length)
 */
bool dim0_is_vocab_length(popart::Op *op) {

    if (getVocabSize() < 0) {
        throw popart::error("TiedGatherPattern: Vocab size has not been set from Python model `ops.setvocabSize(v)`.");
    }

    auto dim0Input = op->inInfo(0).shape()[0];
    return dim0Input == getVocabSize();
}
}

class TiedGatherPattern : public popart::PreAliasPattern {
    mutable std::map<popart::Op *, popart::MatMulBaseOp *> tied_op_map;
public:
    bool matches(popart::Op *op) const override {
        auto &ir = op->getIr();
        // Only run in the fwd pass
        if (op->getIr().hasConstructedBackwards()) {
            return false;
        }
        if (op->getIr().isTraining() && !op->getIr().getSessionOptions().enableGradientAccumulation) {
            return false;
        }
        if (op->isConvertibleTo<popart::GatherOp>() && !op->isConvertibleTo<TiedGatherOp>()) {
            // Change from PopART BERT: We don't have the benefit of the transposed embedding at the start, so we can't use
	    // it to match the pattern. Instead, we look for the tensor with vocab size as its first dim.
	    // if (produced_by_transpose(op->input->tensor(popart::GatherOp::dataInIndex()))) {
	    if (dim0_is_vocab_length(op)) {
                auto matmul = weight_consumed_by<popart::MatMulBaseOp>(op->input->tensor(popart::GatherOp::dataInIndex()));
                if (matmul) {
                    tied_op_map.insert({op, matmul});
                    return true;
                }
            }
        }
        return false;
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto &graph = op->getGraph();

        auto gather = dynamic_cast<popart::GatherOp *>(op);
        auto matmul = tied_op_map[gather];

        // (1)
        matmul->setUseFullyConnectedPass(false);

        auto axis = gather->getAxis();
        auto serialisation = matmul->getSerialiseSettings();

        auto data    = gather->input->tensor(popart::GatherOp::dataInIndex());
        auto indices = gather->input->tensor(popart::GatherOp::indicesInIndex());
        auto out     = gather->output->tensor(popart::GatherOp::outIndex());

        // Disconnect "out" so it can be connected to the replacing ops.
        gather->disconnectAllOutputs();

        // (2)
        auto detach_up = std::make_unique<popart::DetachOp>(
            popart::Onnx::CustomOperators::Detach_1,
            popart::Op::Settings(graph, "TiedGatherDetach")
        );
        auto detach = detach_up.get();
        transferBaseProperties(gather, detach);
        graph.moveIntoGraph(std::move(detach_up));
        detach->connectInTensor(0, data->id);
        auto detached_data_id = data->id + "/detached";
        detach->createAndConnectOutTensor(0, detached_data_id);
        detach->setup();
        data = graph.getTensors().get(detached_data_id);

        std::string name = gather->name();
        if (name.empty()) {
            name = std::to_string(gather->id);
        }

        auto replace_with_tied_gather = [&](popart::TensorId dict, popart::TensorId ind, int64_t i, const std::string &debugContext) {
            auto tied_gather_up = std::make_unique<TiedGatherOp>(
                axis,
                popart::Op::Settings(graph, debugContext));
            auto tied_gather = tied_gather_up.get();
            transferBaseProperties(gather, tied_gather);
            graph.moveIntoGraph(std::move(tied_gather_up));

            tied_gather->connectInTensor(TiedGatherOp::dataInIndex(), dict);
            tied_gather->connectInTensor(TiedGatherOp::indicesInIndex(), ind);

            auto out_id = out->id;
            if (i >= 0) {
                out_id = debugContext + ":0";
                tied_gather->createAndConnectOutTensor(TiedGatherOp::outIndex(), out_id);
            } else {
                tied_gather->connectOutTensor(TiedGatherOp::outIndex(), out_id);
            }

            graph.topoCons->transfer(gather, tied_gather);

            tied_gather->setup();

            return out_id;
        };

        if (serialisation.factor <= 1 || serialisation.mode == SerialiseSettings::Mode::None) {
            // (3)
            replace_with_tied_gather(data->id, indices->id, -1, name);
        } else {
            // (4)
            if (serialisation.mode != SerialiseSettings::Mode::OutputChannels) {
                throw popart::error("CustomOps Error: Tied Gather Pattern only supports Serialisation::Mode::OutputChannels");
            }

            auto slice_op = [&](int64_t starts, int64_t ends, const std::string &debugContext) {
                auto slice_up = std::make_unique<popart::SliceOp>(
                    popart::Onnx::AiOnnx::OpSet9::Slice,
                    std::vector<int64_t>({starts}),
                    std::vector<int64_t>({ends}),
                    std::vector<int64_t>({axis}),
                    popart::Op::Settings(graph, debugContext + "/slice"));
                auto slice = slice_up.get();
                transferBaseProperties(gather, slice);
                graph.moveIntoGraph(std::move(slice_up));
                slice->connectInTensor(popart::SliceOp::getInIndex(), data->id);
                auto data_slice = debugContext + "/slice:0";
                slice->createAndConnectOutTensor(popart::SliceOp::getOutIndex(), data_slice);
                slice->setup();
                return data_slice;
            };

            auto subtract_with_constant = [&](popart::Tensor *a, int64_t c, const std::string &debugContext) {
                auto sub_up = std::make_unique<popart::SubtractOp>(
                    popart::Onnx::Operators::Sub_7,
                    popart::Op::Settings(graph, debugContext + "/sub"));
                auto sub = sub_up.get();
                transferBaseProperties(gather, sub);
                graph.moveIntoGraph(std::move(sub_up));
                sub->connectInTensor(popart::SubtractOp::getArg0InIndex(), a->id);
                // Create constant to subtract from
                static unsigned i = 0;
                auto sub_const_id = a->id + "_sub_const_" + std::to_string(i++);
                popart::TensorInfo subInfo(a->info.dataType(), {1});
                std::vector<unsigned> d(1, c);
                graph.getTensors().addConstInit(sub_const_id, subInfo, d.data());
                sub->connectInTensor(popart::SubtractOp::getArg1InIndex(), sub_const_id);
                auto indices_sub = debugContext + "/sub:0";
                sub->createAndConnectOutTensor(popart::SubtractOp::getOutIndex(), indices_sub);
                sub->setup();
                return indices_sub;
            };

            auto add_op = [&](popart::TensorId a, popart::TensorId b, popart::TensorId out, const std::string &debugContext) {
                auto add_up = std::make_unique<popart::AddOp>(
                    popart::Onnx::Operators::Add_6,
                    popart::Op::Settings(graph, debugContext + "/add"));
                auto add = add_up.get();
                transferBaseProperties(gather, add);
                graph.moveIntoGraph(std::move(add_up));
                add->connectInTensor(popart::AddOp::getArg0InIndex(), a);
                add->connectInTensor(popart::AddOp::getArg1InIndex(), b);
                if (graph.getTensors().contains(out)) {
                    add->connectOutTensor(popart::AddOp::getOutIndex(), out);
                } else {
                    add->createAndConnectOutTensor(popart::AddOp::getOutIndex(), out);
                }
                add->setup();
                return out;
            };

            popart::TensorId tmp_id;
            for (int64_t i = 0; i < serialisation.factor; i++) {
                int64_t slice_size = data->info.dim(axis) / serialisation.factor;
                auto serial_name = name + "/" + std::to_string(i);
                // Slice the Dictionary
                auto data_slice = slice_op(i * slice_size, (i + 1) * slice_size, serial_name);
                // Subtract the indicies
                auto indices_sub = subtract_with_constant(indices, i * slice_size, serial_name);
                // Add the tied gather to the graph
                auto next_id = replace_with_tied_gather(data_slice, indices_sub, i, serial_name);

                // Add the results
                if (i == 0) {
                    tmp_id = next_id;
                } else {
                    auto out_id = out->id;
                    if (i < serialisation.factor - 1) {
                        out_id += "_tmp" + std::to_string(i);   
                    }
                    tmp_id = add_op(tmp_id, next_id, out_id, serial_name);

                    // Tie the add to happen directly after the gather
                    graph.topoCons->insert(
                        graph.getTensors().get(next_id)->getProducer(),
                        graph.getTensors().get(tmp_id)->getProducer(),
                        true);
                }
            }
        }

        gather->disconnectAllInputs();
        graph.eraseOp(gather->id);

        return true;
    }
};

// This pattern matches for graphs of the shape.
//
//    Weight
//    |              \     
// TiedGatherGrad   MatMul
//                    |
//         Accl  -  Accumulate
//
// And will perform the following transformation
//   1) Replace TiedGatherGrad with SparseAccumulate
//
// Resulting in:
//
//    Weight
//    |              \     
//    |             MatMul
//    |               |
//    |    Accl  -  Accumulate
//    |     |          |         
// SparseAccumulate  - Optimizer
//
// (--> is a topocon)

class TiedGatherAccumulatePattern : public popart::PreAliasPattern { 
public:
    bool matches(popart::Op *op) const override {
        // Only works with gradient accumulation
        if (!op->getIr().getSessionOptions().enableGradientAccumulation) {
            return false;
        }
        // Only run after the optimizers have been created
        if (!op->getIr().hasDecomposedOptimizers()) {
            return false;
        }
        return op->isConvertibleTo<TiedGatherGradOp>();
    }

    std::vector<const popart::Tensor *> touches(popart::Op *) const override { return {}; }

    bool apply(popart::Op *op) const override {
        auto gather_grad = dynamic_cast<TiedGatherGradOp *>(op);
        auto gather = gather_grad->fwd_op;
        auto root_weight = get_variable(gather->input->tensor(popart::GatherOp::dataInIndex()));

        auto gather_ops = find_all_consumers<TiedGatherOp>(root_weight);

        auto &ir = op->getIr(); 

        // Get all the Accumulate ops in the normal context
        std::vector<popart::AccumulateOp *> accumulate_ops;

        auto update_ops = find_all_consumers<popart::VarUpdateWithUpdaterOp, popart::ExecutionContext::AccumulateOuterFragment>(root_weight);
        if (update_ops.size() < 1) {
            // SGD1DecomposePattern has not run.
            throw popart::error("CustomOps Error: Could not find update ops for weight {}", root_weight->id);
        }

        for (size_t i = 0; i < update_ops.size(); i++) {
            auto var_update = update_ops[i];
            auto accum = var_update->input->tensor(popart::VarUpdateWithUpdaterOp::getUpdaterInIndex());

            // Accumulate Ops in the normal fragment are Gradient Accumulation.
            auto accl_op = search_producers_for<popart::AccumulateOp, popart::ExecutionContext::Normal>(accum, 10);

            if (accl_op) {
                auto exists = std::find_if(accumulate_ops.begin(), accumulate_ops.end(), [&accl_op](popart::Op* op){ return op->id == accl_op->id; });
                if (exists == accumulate_ops.end()) {
                    accumulate_ops.push_back(accl_op);
                }
            }
        }

	// Change from PopART BERT:
        // If recompute is enabled, we'll get a clone of the TiedGather op, which will then mean we have more TiedGathers
        // than accumulators. We want to throw away the original (non-recomputed) version.
        if (accumulate_ops.size() != gather_ops.size()) {
            gather_ops.erase(std::remove_if(gather_ops.begin(),
                                      gather_ops.end(),
                                      [](TiedGatherOp *g){return g->settings.recomputeType == popart::RecomputeType::Recomputed;}),
                                      gather_ops.end());
        }

        if (accumulate_ops.size() != gather_ops.size()) {
            throw popart::error("CustomOps Error: The number of gather ops ({}) does not match the number of accumulate ops ({}).", gather_ops.size(), accumulate_ops.size());
        }

        // Match up gather serial index to SGD1Accumulator's matmul index.
        std::sort(accumulate_ops.begin(), accumulate_ops.end(),
                  [](const popart::Op *l, const popart::Op *r) {
                      return l->input->tensor(popart::AccumulateOp::getVarToUpdateInIndex())->id.compare(
                          r->input->tensor(popart::AccumulateOp::getVarToUpdateInIndex())->id) < 0;
                  });
        std::sort(gather_ops.begin(), gather_ops.end(), 
            [](const popart::Op *l, const popart::Op *r) {
            return l->name().compare(r->name()) < 0;
        });

        auto itr = std::find(gather_ops.begin(), gather_ops.end(), gather);
        if (itr == gather_ops.end()) {
            throw popart::error("CustomOps Error: Could not find {} in the consumers of {}.", gather->name(), root_weight->id);
        }

        unsigned serial_index = std::distance(gather_ops.begin(), itr);

        auto dense_accl = accumulate_ops[serial_index];

        auto accl_id = dense_accl->inId(popart::AccumulateOp::getVarToUpdateInIndex());
        auto weight_id = gather->inId(popart::GatherOp::dataInIndex());
        popart::logging::pattern::info("Using tied accumulator {} for {}", accl_id, gather->name());

        // Transpose must be inplace so the accumulator is actually updated
        // Transpose must be inplace so the accumulator is actually updated
        // Change from PopART BERT: This transpose isn't needed for the HF model as we don't have the transposed emb
        // accl_id    = transpose_inplace(accl_id, gather_grad);
        // weight_id  = transpose_inplace(weight_id, gather_grad);

        auto &graph = op->getGraph();

        // Add sparseSGD1AccumulateOp.
        auto sparse_accl_up = std::make_unique<SparseAccumulateOp>(
            accl_id,
            dense_accl->getFactor(),
            gather_grad->getAxis(),
            popart::Op::Settings(graph, "_tiedAccumulate/" + std::to_string(serial_index)));

        auto sparse_accl = sparse_accl_up.get();
        transferBaseProperties(gather_grad, sparse_accl);
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

        // Original weight to be cloned
        sparse_accl->connectInTensor(SparseAccumulateOp::getOriginalVarInIndex(),
                                     weight_id);

        // Transfer TopoCons
        graph.topoCons->transfer(gather_grad, sparse_accl);

        // gatherGrad output that will be isolated
        auto grad_Id = gather_grad->outId(TiedGatherGradOp::gradOutIndex());

        // Remove TiedGatherGrad
        gather_grad->disconnectAllInputs();
        gather_grad->disconnectAllOutputs();
        graph.eraseOp(gather_grad->id);

        // Outputs
        sparse_accl->createAndConnectOutTensor(SparseAccumulateOp::getUpdatedVarOutIndex(), sparse_accl->name() + ":0");
        
        // remove the gatherGrad output
        graph.getTensors().remove(grad_Id);

        // Finalise sparse op
        sparse_accl->setup();

        return true;
    }

    popart::TensorId transpose_inplace(popart::TensorId tid, popart::Op *op) const {
        auto &graph = op->getGraph();

        // TransposeInplaceOp's constructor requires a transposeOp
        auto outplace_up = std::make_unique<popart::TransposeOp>(
            popart::Onnx::AiOnnx::OpSet9::Transpose,
            std::vector<int64_t>{1, 0},
            popart::Op::Settings(graph, tid + "_Transpose"));
        auto transpose_up = outplace_up->getInplaceVariant(popart::Onnx::CustomOperators::TransposeInplace);

        auto transpose = transpose_up.get();
        transferBaseProperties(op, transpose);
        graph.moveIntoGraph(std::move(transpose_up));

        transpose->connectInTensor(popart::TransposeOp::getInIndex(), tid);
        popart::TensorId out_id = tid + "/transposed";
        transpose->createAndConnectOutTensor(popart::TransposeOp::getOutIndex(), out_id);

        transpose->setup();
        return out_id;
    }
};

static popart::PatternCreator<TiedGatherPattern> TiedGatherPatternCreator("TiedGatherPattern", true);
static popart::PatternCreator<TiedGatherAccumulatePattern> TiedGatherAccumulatePatternCreator("TiedGatherAccumulatePattern", true);
