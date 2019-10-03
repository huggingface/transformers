# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert RoBERTa checkpoint."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np
import torch

from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from transformers import (BertConfig, BertEncoder,
                                                BertIntermediate, BertLayer,
                                                BertModel, BertOutput,
                                                BertSelfAttention,
                                                BertSelfOutput)
from transformers import (RobertaEmbeddings,
                                                   RobertaForMaskedLM,
                                                   RobertaForSequenceClassification,
                                                   RobertaModel)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = 'Hello world! cécé herlolip'


def convert_roberta_checkpoint_to_pytorch(roberta_checkpoint_path, pytorch_dump_folder_path, classification_head):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path)
    roberta.eval()  # disable dropout
    config = BertConfig(
        vocab_size_or_config_json_file=50265,
        hidden_size=roberta.args.encoder_embed_dim,
        num_hidden_layers=roberta.args.encoder_layers,
        num_attention_heads=roberta.args.encoder_attention_heads,
        intermediate_size=roberta.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5, # PyTorch default used in fairseq
    )
    if classification_head:
        config.num_labels = roberta.args.num_classes
    print("Our BERT config:", config)

    model = RobertaForSequenceClassification(config) if classification_head else RobertaForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    roberta_sent_encoder = roberta.model.decoder.sentence_encoder
    model.roberta.embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(model.roberta.embeddings.token_type_embeddings.weight)  # just zero them out b/c RoBERTa doesn't use them.
    model.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.emb_layer_norm.weight
    model.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.emb_layer_norm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        ### self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert(
            roberta_layer.self_attn.in_proj_weight.shape == torch.Size((3 * config.hidden_size, config.hidden_size))
        )
        # we use three distinct linear layers so we split the source layer here.
        self_attn.query.weight.data = roberta_layer.self_attn.in_proj_weight[:config.hidden_size, :]
        self_attn.query.bias.data = roberta_layer.self_attn.in_proj_bias[:config.hidden_size]
        self_attn.key.weight.data = roberta_layer.self_attn.in_proj_weight[config.hidden_size:2*config.hidden_size, :]
        self_attn.key.bias.data = roberta_layer.self_attn.in_proj_bias[config.hidden_size:2*config.hidden_size]
        self_attn.value.weight.data = roberta_layer.self_attn.in_proj_weight[2*config.hidden_size:, :]
        self_attn.value.bias.data = roberta_layer.self_attn.in_proj_bias[2*config.hidden_size:]

        ### self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert(
            self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        )
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias

        ### intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert(
            intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        )
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        ### output
        bert_output: BertOutput = layer.output
        assert(
            bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        )
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
        #### end of layer
    
    if classification_head:
        model.classifier.dense.weight = roberta.model.classification_heads['mnli'].dense.weight
        model.classifier.dense.bias = roberta.model.classification_heads['mnli'].dense.bias
        model.classifier.out_proj.weight = roberta.model.classification_heads['mnli'].out_proj.weight
        model.classifier.out_proj.bias = roberta.model.classification_heads['mnli'].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = roberta.model.decoder.lm_head.dense.weight
        model.lm_head.dense.bias = roberta.model.decoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = roberta.model.decoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = roberta.model.decoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = roberta.model.decoder.lm_head.weight
        model.lm_head.bias = roberta.model.decoder.lm_head.bias

    # Let's check that we get the same results.
    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0) # batch of size 1

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = roberta.model.classification_heads['mnli'](roberta.extract_features(input_ids))
    else:
        their_output = roberta.model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}") # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print(
        "Do both models output the same tensors?",
        "🔥" if success else "💩"
    )
    if not success:
        raise Exception("Something went wRoNg")

    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--roberta_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the official PyTorch dump.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    parser.add_argument("--classification_head",
                        action = "store_true",
                        help = "Whether to convert a final classification head.")
    args = parser.parse_args()
    convert_roberta_checkpoint_to_pytorch(
        args.roberta_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.classification_head
    )

