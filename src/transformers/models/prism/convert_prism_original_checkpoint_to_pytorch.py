# Copyright 2024 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

from transformers import PrismConfig, PrismForConditionalGeneration


# Copied from transformers.models.m2m_100.convert_m2m100_original_checkpoint_to_pytorch.remove_ignore_keys_
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


def convert_prism_checkpoint_from_disk(checkpoint_path):
    prism = torch.load(checkpoint_path, map_location="cpu")
    args = prism["args"] or prism["cfg"]["model"]
    state_dict = prism["model"]
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]

    config = PrismConfig(
        vocab_size=vocab_size,
        max_position_embeddings=1024,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_embed_dim,
        decoder_ffn_dim=args.decoder_ffn_embed_dim,
        d_model=args.encoder_embed_dim,
        encoder_layerdrop=args.encoder_layerdrop,
        decoder_layerdrop=args.decoder_layerdrop,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function="relu",
        layernorm_embedding=args.layernorm_embedding,
        tie_word_embeddings=True,
    )

    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    model = PrismForConditionalGeneration(config)
    model.model.load_state_dict(state_dict, strict=True)
    model._tie_weights()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("fairseq_path", type=str, help="path to a model.pt on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    model = convert_prism_checkpoint_from_disk(args.fairseq_path)
    model.save_pretrained(args.pytorch_dump_folder_path)
