# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import poptorch
import transformers

from bert_fused_attention import BertFusedSelfAttention
from utils import logger


def gather_indices(sequence, positions):
    """
    Gather the vectors at the specific positions over a batch.
    """
    num_classes = int(sequence.shape[1])
    if poptorch.isRunningOnIpu():
        # Use custom-op workaround for one_hot until F.one_hot is supported
        reference = torch.zeros(
            (*positions.shape, num_classes), dtype=sequence.dtype)
        # Passing in reference as an input
        # is a workaround for attributes not being implemented yet.
        one_hot_positions = poptorch.custom_op([positions, reference],
                                               "OneHot",
                                               "ai.graphcore",
                                               1,
                                               example_outputs=[reference.float()])[0].detach()
    else:
        one_hot_positions = F.one_hot(positions, num_classes).float().detach()
    return torch.matmul(one_hot_positions, sequence)


class SerializedLinear(nn.Linear):
    def __init__(self, in_features, out_features, factor, bias=False,
                 mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


class RecomputationCheckpoint(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *x):
        return tuple(poptorch.recomputationCheckpoint(y) for y in self.layer(*x))


def accuracy(out, targ):
    return (out.argmax(dim=-1) == targ).float().mean()


def accuracy_masked(out, targ, mask_val):
    mask = (targ != mask_val).float()
    num_unmasked = mask.sum(1).unsqueeze(1)
    return (out.argmax(dim=-1) == targ).float().mul(mask).div(num_unmasked).sum(1).mean()


class PipelinedBertWithLoss(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model = transformers.BertForPreTraining(config)

        for layer in self.model.bert.encoder.layer:
            layer.attention.self = BertFusedSelfAttention(config)

        if not self.config.pred_head_transform:
            # Disable prediction head transform
            self.model.cls.predictions.transform = nn.Identity()

        if self.config.embedding_serialization_factor > 1:
            self.model.cls.predictions.decoder = SerializedLinear(self.config.hidden_size,
                                                                  self.config.vocab_size,
                                                                  self.config.embedding_serialization_factor,
                                                                  mode=poptorch.MatMulSerializationMode.OutputChannels)
            self.model.tie_weights()

        layer_ipu = _get_layer_ipu(config.layers_per_ipu)

        logger("-------------------- Device Allocation --------------------")
        logger("Embedding  --> IPU 0")
        self.model.bert.embeddings = poptorch.BeginBlock(self.model.bert.embeddings, "Embedding", ipu_id=0)

        for index, layer in enumerate(self.model.bert.encoder.layer):
            ipu = layer_ipu[index]
            layer = RecomputationCheckpoint(layer) if config.recompute_checkpoint_every_layer else layer
            self.model.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Encoder {index:<2} --> IPU {ipu}")

        logger("Pooler     --> IPU 0")
        self.model.bert.pooler = poptorch.BeginBlock(self.model.bert.pooler, "Pooler", ipu_id=0)

        logger("Classifier --> IPU 0")
        self.model.cls = poptorch.BeginBlock(self.model.cls, "Classifier", ipu_id=0)
        logger("-----------------------------------------------------------")

    def forward(self, input_ids, attention_mask, token_type_ids, masked_lm_positions, masked_lm_labels=None, next_sentence_label=None):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        outputs = self.model.bert(**inputs)
        sequence_output, pooled_output = outputs[:2]

        # Select only the masked tokens for the classifier
        masked_output = gather_indices(sequence_output, masked_lm_positions)

        prediction_scores, sequential_relationship_score = self.model.cls(masked_output, pooled_output)
        outputs = (prediction_scores, sequential_relationship_score,) + outputs[2:]

        if masked_lm_labels is not None and next_sentence_label is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=0).float()
            next_sentence_loss = F.cross_entropy(sequential_relationship_score.view(-1, 2), next_sentence_label.view(-1)).float()
            total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")

            next_sentence_acc = accuracy(sequential_relationship_score.view([-1, 2]), next_sentence_label.view(-1))
            # masked_lm_labels: 0 if corresponding token not masked, original value otherwise
            masked_lm_acc = accuracy_masked(prediction_scores.view([-1, self.config.mask_tokens, self.config.vocab_size]), masked_lm_labels, 0)
            outputs = (total_loss, masked_lm_loss, next_sentence_loss, masked_lm_acc, next_sentence_acc)

        return outputs
