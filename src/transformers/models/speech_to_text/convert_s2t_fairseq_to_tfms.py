# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import argparse

import torch
from torch import nn

from transformers import Speech2TextConfig, Speech2TextForConditionalGeneration


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        if "transformer_layers" in key:
            s_dict[key.replace("transformer_layers", "layers")] = s_dict.pop(key)
        elif "subsample" in key:
            s_dict[key.replace("subsample", "conv")] = s_dict.pop(key)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def convert_fairseq_s2t_checkpoint_to_tfms(checkpoint_path, pytorch_dump_folder_path):
    m2m_100 = torch.load(checkpoint_path, map_location="cpu")
    args = m2m_100["args"]
    state_dict = m2m_100["model"]
    lm_head_weights = state_dict["decoder.output_projection.weight"]

    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)

    vocab_size = state_dict["decoder.embed_tokens.weight"].shape[0]

    tie_embeds = args.share_decoder_input_output_embed

    conv_kernel_sizes = [int(i) for i in args.conv_kernel_sizes.split(",")]
    config = Speech2TextConfig(
        vocab_size=vocab_size,
        max_source_positions=args.max_source_positions,
        max_target_positions=args.max_target_positions,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_embed_dim,
        decoder_ffn_dim=args.decoder_ffn_embed_dim,
        d_model=args.encoder_embed_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function="relu",
        num_conv_layers=len(conv_kernel_sizes),
        conv_channels=args.conv_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        input_feat_per_channel=args.input_feat_per_channel,
        input_channels=args.input_channels,
        tie_word_embeddings=tie_embeds,
        num_beams=5,
        max_length=200,
        use_cache=True,
        decoder_start_token_id=2,
        early_stopping=True,
    )

    model = Speech2TextForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 and not set(missing) <= set(
        [
            "encoder.embed_positions.weights",
            "decoder.embed_positions.weights",
        ]
    ):
        raise ValueError(
            f"Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing, but all the following weights are missing {missing}"
        )

    if tie_embeds:
        model.lm_head = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        model.lm_head.weight.data = lm_head_weights

    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--fairseq_path", type=str, help="Path to the fairseq model (.pt) file.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_fairseq_s2t_checkpoint_to_tfms(args.fairseq_path, args.pytorch_dump_folder_path)
