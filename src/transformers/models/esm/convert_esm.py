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
"""Convert ESM checkpoint."""


import argparse
import pathlib

# import fairseq
import torch

import esm as esm_module
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.models.esm.configuration_esm import ESMConfig
from transformers.models.esm.modeling_esm import ESMForMaskedLM, ESMForSequenceClassification
from transformers.utils import logging


# if version.parse(fairseq.__version__) < version.parse("0.9.0"):
#     raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_DATA = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
]


def convert_esm_checkpoint_to_pytorch(pytorch_dump_folder_path: str, classification_head: bool):
    """
    Copy/paste/tweak esm's weights to our BERT structure.
    """
    # esm = FairseqESMModel.from_pretrained(esm_checkpoint_path)
    esm, alphabet = esm_module.pretrained.esm1b_t33_650M_UR50S()
    esm.eval()  # disable dropout
    esm_sent_encoder = esm
    config = ESMConfig(
        vocab_size=esm_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=esm.args.embed_dim,
        num_hidden_layers=esm.args.layers,
        num_attention_heads=esm.args.attention_heads,
        intermediate_size=esm.args.ffn_embed_dim,
        max_position_embeddings=1026,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0,
        encoder_keep_prob=0.88,  # TODO: temporary hack, explained in modeling_esm
        pad_token_id=esm.padding_idx
    )
    if classification_head:
        config.num_labels = esm.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our BERT config:", config)

    model = ESMForSequenceClassification(config) if classification_head else ESMForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    model.esm.embeddings.word_embeddings.weight = esm_sent_encoder.embed_tokens.weight
    model.esm.embeddings.position_embeddings.weight = esm_sent_encoder.embed_positions.weight
    model.esm.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.esm.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c ESM doesn't use them.

    model.esm.embeddings.LayerNorm.weight = esm_sent_encoder.emb_layer_norm_before.weight
    model.esm.embeddings.LayerNorm.bias = esm_sent_encoder.emb_layer_norm_before.bias

    model.esm.encoder.emb_layer_norm_after.weight = esm_sent_encoder.emb_layer_norm_after.weight
    model.esm.encoder.emb_layer_norm_after.bias = esm_sent_encoder.emb_layer_norm_after.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.esm.encoder.layer[i]
        # esm_layer: TransformerSentenceEncoderLayer = esm_sent_encoder.layers[i]
        esm_layer = esm_sent_encoder.layers[i]

        # self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert (
            esm_layer.self_attn.k_proj.weight.data.shape
            == esm_layer.self_attn.q_proj.weight.data.shape
            == esm_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = esm_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = esm_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = esm_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = esm_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = esm_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = esm_layer.self_attn.v_proj.bias

        # LayerNorm changes for pre-activation
        layer.attention.LayerNorm.weight = esm_layer.self_attn_layer_norm.weight
        layer.attention.LayerNorm.bias = esm_layer.self_attn_layer_norm.bias
        layer.LayerNorm.weight = esm_layer.final_layer_norm.weight
        layer.LayerNorm.bias = esm_layer.final_layer_norm.bias

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == esm_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = esm_layer.self_attn.out_proj.weight
        self_output.dense.bias = esm_layer.self_attn.out_proj.bias

        # intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == esm_layer.fc1.weight.shape
        intermediate.dense.weight = esm_layer.fc1.weight
        intermediate.dense.bias = esm_layer.fc1.bias

        # output
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == esm_layer.fc2.weight.shape
        bert_output.dense.weight = esm_layer.fc2.weight
        bert_output.dense.bias = esm_layer.fc2.bias
        # end of layer

    if classification_head:
        model.classifier.dense.weight = esm.esm.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = esm.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = esm.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = esm.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = esm.lm_head.dense.weight
        model.lm_head.dense.bias = esm.lm_head.dense.bias
        model.lm_head.layer_norm.weight = esm.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = esm.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = esm.lm_head.weight
        model.lm_head.decoder.bias = esm.lm_head.bias

    # Let's check that we get the same results.
    batch_converter = alphabet.get_batch_converter()

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)

    batch_labels, batch_strs, batch_tokens = batch_converter(SAMPLE_DATA)
    input_ids = batch_tokens

    # TODO Models have different padding tokens in their embedding layers, and we're not passing an
    #      attention mask so this might be the cause of the differences.

    with torch.no_grad():
        our_output = model(input_ids, output_hidden_states=True)
        our_hiddens = our_output["hidden_states"]
        our_output = our_output["logits"]
        if classification_head:
            their_output = esm.model.classification_heads["mnli"](esm.extract_features(input_ids))
        else:
            their_output = esm(input_ids, repr_layers=list(range(999)))
            their_hiddens = their_output["representations"]
            their_output = their_output["logits"]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    breakpoint()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-5
    success = torch.allclose(our_output, their_output, atol=3e-4)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    args = parser.parse_args()
    convert_esm_checkpoint_to_pytorch(args.pytorch_dump_folder_path, args.classification_head)
