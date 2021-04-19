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

#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/tensorindex.hpp>
#include <popart/op.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/logging.hpp>
#include <queue>

#include "compile_time_version.h"


template <class T, popart::ExecutionContext Ctx = popart::ExecutionContext::Normal>
static T *search_producers_for(popart::Tensor *t, int max_depth=-1) {

    // Searched as far as we can without success
    if (t->tensorType() == popart::TensorType::Variable || !t->hasProducer()) {
        return nullptr;
    }
    auto op = t->getProducer();
    if (op->isConvertibleTo<T>() && op->settings.executionContext == Ctx) {
        return dynamic_cast<T *>(op);
    }

    if (op->input->n() < 1) {
        return nullptr;
    }

    unsigned producer_index = 0;
    if (op->input->n() > 1) {
        if (op->isConvertibleTo<popart::AdamUpdaterOp>()) {
            producer_index = popart::AdamUpdaterOp::getAccl1InIndex();
        } else if (op->isConvertibleTo<popart::AdamVarUpdateOp>()) {
            producer_index = popart::AdamVarUpdateOp::getUpdaterInIndex();
        } else if (op->isConvertibleTo<popart::AccumulateOp>()) {
            // Accumulates for M/V in Adam-based optimizers
            producer_index = popart::AccumulateOp::getUpdaterInIndex();
        } else if (op->isConvertibleTo<popart::DropoutGradOp>()) {
            producer_index = popart::DropoutGradOp::getGradInIndex();
        } else if (op->isConvertibleTo<popart::MulOp>()) {
            // Grad Unscaling for Adam-based optimizers
            producer_index = popart::MulOp::getArg0InIndex();
        } else if (op->isConvertibleTo<popart::ReplicatedReduceScatterOp>()) {
            // Replicated Tensor Sharding
            producer_index = popart::ReplicatedReduceScatterOp::getInIndex();
        } else if (op->isConvertibleTo<popart::ReplicatedAllGatherOp>()) {
            // Replicated Tensor Sharding
            producer_index = popart::ReplicatedAllGatherOp::getInIndex();
        } else {
            return nullptr;
        }
    }

    // Providing a max-search depth of -1 will remove the depth limit at the cost of potentially
    // unnecessary checks.
    if (max_depth > 0) {
        max_depth -= 1;
        if (max_depth == 0) {
            return nullptr;
        }
    }

    return search_producers_for<T, Ctx>(op->input->tensor(producer_index), max_depth);
}

// Finds the underlying variable by searching through producers.
static popart::Tensor *get_variable(popart::Tensor *t) {
    if (t->tensorType() == popart::TensorType::Variable || t->tensorType() == popart::TensorType::Const) {
        return t;
    } else if (!t->hasProducer()) {
        return nullptr;
    }
    auto op = t->getProducer();
    if (op->input->n() != 1) {
        return nullptr;
    }
    return get_variable(op->input->tensors().front());
}

// Attempts to find T by searching through consumers.
template <class T, popart::ExecutionContext Ctx = popart::ExecutionContext::Normal>
static T *search_consumers_for(popart::Tensor *w, std::queue<popart::Tensor *> &q) {
    for (auto consumer : w->consumers.getOps()) {
        if (consumer->isConvertibleTo<T>() && consumer->settings.executionContext == Ctx) {
            return reinterpret_cast<T *>(consumer);
        }

        if (consumer->isConvertibleTo<popart::DropoutGradOp>()) {
            q.push(consumer->output->tensor(popart::DropoutGradOp::getGradInIndex()));
        }

        if (consumer->input->n() == 1 && consumer->output->n() == 1) {
            q.push(consumer->output->tensor(0));
        }
    }
    if (q.size() < 1) {
        return nullptr;
    }
    w = q.front();
    q.pop();
    return search_consumers_for<T>(w, q);
}
template <class T, popart::ExecutionContext Ctx = popart::ExecutionContext::Normal>
static T *search_consumers_for(popart::Tensor *w) {
    std::queue<popart::Tensor *> q;
    return search_consumers_for<T, Ctx>(w, q);
}

template <class T>
static T *weight_consumed_by(popart::Tensor *w) {
    w = get_variable(w);
    if (w) {
        return search_consumers_for<T>(w);
    }
    return nullptr;
}

template <class T, popart::ExecutionContext Ctx>
static void find_all_consumers(popart::Tensor *w,std::queue<popart::Tensor *> &q, std::vector<T *> &result) {
    for (auto consumer : w->consumers.getOps()) {
        if (std::find(result.begin(), result.end(), consumer) == result.end()) {
            if (consumer->isConvertibleTo<T>() && consumer->settings.executionContext == Ctx) {
                T *op = reinterpret_cast<T *>(consumer);
                result.push_back(op);
            }
            if (consumer->isConvertibleTo<popart::MatMulOp>()) {
                q.push(consumer->output->tensor(popart::MatMulOp::getOutIndex()));
            }
            // Most ops that have one input and one output are view changing.
            if (consumer->input->n() == 1 && consumer->output->n() == 1) {
                q.push(consumer->output->tensor(0));
            }
        }
    }
    if (q.size() < 1) {
        return;
    }
    w = q.front();
    q.pop();
    return find_all_consumers<T, Ctx>(w, q, result);
}
template <class T, popart::ExecutionContext Ctx = popart::ExecutionContext::Normal>
static std::vector<T *> find_all_consumers(popart::Tensor *w) {
    std::queue<popart::Tensor *> q;
    std::vector<T *> result;
    find_all_consumers<T, Ctx>(w, q, result);
    return result;
}
