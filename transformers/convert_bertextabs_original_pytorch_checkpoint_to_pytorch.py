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
""" Convert BertExtAbs's checkpoints """

import argparse
from collections import namedtuple
import logging

import torch

from models.model_builder import AbsSummarizer  # The authors' implementation

from transformers import BertConfig, Model2Model, BertModel, BertForMaskedLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BertExtAbsConfig = namedtuple(
    "BertExtAbsConfig",
    ["temp_dir", "large", "finetune_bert", "encoder", "share_emb", "max_pos", "enc_layers", "enc_hidden_size", "enc_heads", "enc_ff_size", "enc_dropout", "dec_layers", "dec_hidden_size", "dec_heads", "dec_ff_size", "dec_dropout"],
)


def convert_bertextabs_checkpoints(path_to_checkpoints, dump_path):
    """ Copy/paste and tweak the pre-trained weights provided by the creators
    of BertExtAbs for the internal architecture.
    """

    # Load checkpoints in memory
    checkpoints = torch.load(path_to_checkpoints, lambda storage, loc: storage)

    # Instantiate the authors' model with the pre-trained weights
    config = BertExtAbsConfig(
        temp_dir=".",
        finetune_bert=False,
        large=False,
        share_emb=True,
        encoder="bert",
        max_pos=512,
        enc_layers=6,
        enc_hidden_size=512,
        enc_heads=8,
        enc_ff_size=512,
        enc_dropout=0.2,
        dec_layers=6,
        dec_hidden_size=768,
        dec_heads=8,
        dec_ff_size=2048,
        dec_dropout=0.2,
    )
    bertextabs = AbsSummarizer(config, torch.device("cpu"), checkpoints)
    bertextabs.eval()

    # Instantiate our version of the model
    decoder_config = BertConfig(
        hidden_size=config.dec_hidden_size,
        num_hidden_layers=config.dec_layers,
        num_attention_heads=config.dec_heads,
        intermediate_size=config.dec_ff_size,
        hidden_dropout_prob=config.dec_dropout,
        attention_probs_dropout_prob=config.dec_dropout,
        is_decoder=True,
    )

    decoder_model = BertForMaskedLM(decoder_config)
    model = Model2Model.from_pretrained('bert-base-uncased', decoder_model=decoder_model)
    model.eval()

    # Let us now start the weight copying process
    model.encoder.load_state_dict(bertextabs.bert.model.state_dict())

    # Decoder

    # Embeddings. The positional embeddings are equal to the word embedding plus a modulation
    # that is computed at each forward pass. This may be a source of discrepancy.
    model.decoder.bert.embeddings.word_embeddings.weight = bertextabs.decoder.embeddings.weight
    model.decoder.bert.embeddings.position_embeddings.weight = bertextabs.decoder.embeddings.weight
    model.decoder.bert.embeddings.token_type_embeddings.weight.data = torch.zeros_like(bertextabs.decoder.embeddings.weight)  # not defined for BertExtAbs decoder

    # In the original code the LayerNorms are applied twice in the layers, at the beginning and between the
    # attention layers.
    model.decoder.bert.embeddings.LayerNorm.weight = bertextabs.decoder.transformer_layers[0].layer_norm_1.weight

    for i in range(config.dec_layers):

        # self attention
        model.decoder.bert.encoder.layer[i].attention.self.query.weight = bertextabs.decoder.transformer_layers[i].self_attn.linear_query.weight
        model.decoder.bert.encoder.layer[i].attention.self.key.weight = bertextabs.decoder.transformer_layers[i].self_attn.linear_keys.weight
        model.decoder.bert.encoder.layer[i].attention.self.value.weight = bertextabs.decoder.transformer_layers[i].self_attn.linear_values.weight
        model.decoder.bert.encoder.layer[i].attention.output.dense.weight = bertextabs.decoder.transformer_layers[i].self_attn.final_linear.weight
        model.decoder.bert.encoder.layer[i].attention.output.LayerNorm.weight = bertextabs.decoder.transformer_layers[i].layer_norm_2.weight

        # attention
        model.decoder.bert.encoder.layer[i].crossattention.self.query.weight = bertextabs.decoder.transformer_layers[i].context_attn.linear_query.weight
        model.decoder.bert.encoder.layer[i].crossattention.self.key.weight = bertextabs.decoder.transformer_layers[i].context_attn.linear_keys.weight
        model.decoder.bert.encoder.layer[i].crossattention.self.value.weight = bertextabs.decoder.transformer_layers[i].context_attn.linear_values.weight
        model.decoder.bert.encoder.layer[i].crossattention.output.dense.weight = bertextabs.decoder.transformer_layers[i].context_attn.final_linear.weight
        model.decoder.bert.encoder.layer[i].crossattention.output.LayerNorm.weight = bertextabs.decoder.transformer_layers[i].feed_forward.layer_norm.weight

        # intermediate
        model.decoder.bert.encoder.layer[i].intermediate.dense.weight = bertextabs.decoder.transformer_layers[i].feed_forward.w_1.weight

        # output
        model.decoder.bert.encoder.layer[i].output.dense.weight = bertextabs.decoder.transformer_layers[i].feed_forward.w_2.weight

        try:
            model.decoder.bert.encoder.layer[i].output.LayerNorm.weight = bertextabs.decoder.transformer_layers[i + 1].layer_norm_1.weight
        except IndexError:
            model.decoder.bert.encoder.layer[i].output.LayerNorm.weight = bertextabs.decoder.layer_norm.weight

    # LM Head
    """
    model.decoder.cls.predictions.transform.dense.weight
    model.decoder.cls.predictions.transform.dense.biais
    model.decoder.cls.predictions.transform.LayerNorm.weight
    model.decoder.cls.predictions.transform.LayerNorm.biais
    model.decoder.cls.predictions.decoder.weight
    model.decoder.cls.predictions.decoder.biais
    model.decoder.cls.predictions.biais.data
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bertextabs_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch dump.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )
    args = parser.parse_args()

    convert_bertextabs_checkpoints(
        args.bertextabs_checkpoint_path,
        args.pytorch_dump_folder_path,
    )
