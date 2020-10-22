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


import argparse
import pathlib

import fairseq
import torch
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version

from transformers.modeling_bert import BertIntermediate, BertLayer, BertOutput, BertSelfAttention, BertSelfOutput
from transformers.modeling_roberta import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
from transformers.utils import logging


if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"


def map_roberta_embedding(
    dictionary: fairseq.data.Dictionary, emb: torch.Tensor, shape: tuple
):
    """
    Convert RoBERTa embedding weights using dict.txt to remove need for gpt2 bpe vocab
    exercises further down the pipeline.

    This enables direct use of e.g. fill-mask pipeline without modification.
    """

    # New embedding matrix potentially has a larger shape
    # than the old one with missing gpt2-bpe tokens mapped to unk
    new_emb = torch.zeros(shape)

    hf_mask_idx = dictionary.nspecial

    # Special tokens are not mapped based on dict.txt
    for i in range(hf_mask_idx):
        new_emb[i] = emb[i]

    # The mask is not special in the original gpt2 dict
    mask_idx = dictionary.add_symbol("<mask>")
    new_emb[hf_mask_idx] = emb[mask_idx]

    symb_count = len(dictionary.symbols)
    # Simple hack since vocab is always biggest

    vocab_size = max(shape)
    
    for i in range(hf_mask_idx + 1, vocab_size):
        if i < symb_count and dictionary.symbols[i].isnumeric():
            new_emb[int(dictionary.symbols[i])] = emb[i]
        else:
            new_emb[i] = emb[dictionary.unk()]

    return torch.nn.Parameter(new_emb)


def convert_roberta_checkpoint_to_pytorch(
    roberta_checkpoint_path: str, pytorch_dump_folder_path: str, roberta_dict: bool, classification_head: bool
):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """

    roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path)
    roberta.eval()  # disable dropout
    roberta_sent_encoder = roberta.model.encoder.sentence_encoder

    if roberta_dict:
        roberta_dict = fairseq.data.Dictionary.load(roberta_checkpoint_path + "/dict.txt")
        # We inject unk instead of the missing symbols in the extended embeddings
        vocab_size = max([int(symbol) for symbol in roberta_dict.symbols if symbol.isnumeric()]) + 1
    else:
        print("Note: Dictionary is not loaded (--roberta_dict), this may make inference more tedious down the line.")
        vocab_size = roberta_sent_encoder.embed_tokens.num_embeddings

    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=roberta.args.encoder_embed_dim,
        num_hidden_layers=roberta.args.encoder_layers,
        num_attention_heads=roberta.args.encoder_attention_heads,
        intermediate_size=roberta.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our BERT config:", config)

    model = RobertaForSequenceClassification(config) if classification_head else RobertaForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    if roberta_dict:
        word_emb = map_roberta_embedding(
            roberta_dict,
            roberta_sent_encoder.embed_tokens.weight,
            (vocab_size, roberta.args.encoder_embed_dim)
        )
    else:
        word_emb = roberta_sent_encoder.embed_tokens.weight

    model.roberta.embeddings.word_embeddings.weight = word_emb
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c RoBERTa doesn't use them.
    model.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.emb_layer_norm.weight
    model.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.emb_layer_norm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        # self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias

        # intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        # output
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
        # end of layer

    if classification_head:
        model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias

        if roberta_dict:
            model.classifier.out_proj.weight = map_roberta_embedding(
                roberta_dict,
                roberta.model.classification_heads["mnli"].out_proj.weight,
                (vocab_size, config.num_labels)
            )
        else:
            model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        
        model.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias

        if roberta_dict:
            model.lm_head.decoder.weight = map_roberta_embedding(
                roberta_dict,
                roberta.model.encoder.lm_head.weight,
                (vocab_size, roberta.args.encoder_embed_dim)
            )
            model.lm_head.decoder.bias = map_roberta_embedding(
                roberta_dict,
                roberta.model.encoder.lm_head.bias,
                (vocab_size, )
            )
        else:
            model.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
            model.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias

    # Let's check that we get the same results.
    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    else:
        their_output = roberta.model(input_ids)[0]
    print(our_output.shape, their_output.shape)

    if not roberta_dict:
        max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
        print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
        success = torch.allclose(our_output, their_output, atol=1e-3)
        print("Do both models output the same tensors?", "🔥" if success else "💩")
        if not success:
            raise Exception("Something went wRoNg")
    else:
        print("Model output not tested since embeddings have been mapped with dict.txt")

    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--roberta_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--roberta_dict", action="store_true", help="Whether to use fairseq preprocessing dict."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    args = parser.parse_args()
    convert_roberta_checkpoint_to_pytorch(
        args.roberta_checkpoint_path, args.pytorch_dump_folder_path, args.roberta_dict, args.classification_head
    )



