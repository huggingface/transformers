# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

"""Convert CodeReviewerX checkpoints from the original repository to JAX/FLAX model."""

import argparse

from codereviewerx import checkpoints

from transformers import FlaxCodeReviewerForConditionalGeneration, CodeReviewerConfig


def convert_codereviewerx_checkpoint_to_flax(codereviewerx_checkpoint_path, config_name, flax_dump_folder_path):
    config = CodeReviewerConfig.from_pretrained(config_name)
    flax_model = FlaxCodeReviewerForConditionalGeneration(config=config)
    codereviewerx_model = checkpoints.load_codereviewerx_checkpoint(codereviewerx_checkpoint_path)

    split_mlp_wi = "wi_0" in codereviewerx_model["target"]["encoder"]["layers_0"]["mlp"]

    # Encoder
    for layer_index in range(config.num_layers):
        layer_name = f"layers_{str(layer_index)}"

        # Self-Attention
        codereviewerx_attention_key = codereviewerx_model["target"]["encoder"][layer_name]["attention"]["key"]["kernel"]
        codereviewerx_attention_out = codereviewerx_model["target"]["encoder"][layer_name]["attention"]["out"]["kernel"]
        codereviewerx_attention_query = codereviewerx_model["target"]["encoder"][layer_name]["attention"]["query"]["kernel"]
        codereviewerx_attention_value = codereviewerx_model["target"]["encoder"][layer_name]["attention"]["value"]["kernel"]

        # Layer Normalization
        codereviewerx_attention_layer_norm = codereviewerx_model["target"]["encoder"][layer_name]["pre_attention_layer_norm"]["scale"]

        if split_mlp_wi:
            codereviewerx_mlp_wi_0 = codereviewerx_model["target"]["encoder"][layer_name]["mlp"]["wi_0"]["kernel"]
            codereviewerx_mlp_wi_1 = codereviewerx_model["target"]["encoder"][layer_name]["mlp"]["wi_1"]["kernel"]
        else:
            codereviewerx_mlp_wi = codereviewerx_model["target"]["encoder"][layer_name]["mlp"]["wi"]["kernel"]

        codereviewerx_mlp_wo = codereviewerx_model["target"]["encoder"][layer_name]["mlp"]["wo"]["kernel"]

        # Layer Normalization
        codereviewerx_mlp_layer_norm = codereviewerx_model["target"]["encoder"][layer_name]["pre_mlp_layer_norm"]["scale"]

        # Assigning
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"][
            "kernel"
        ] = codereviewerx_attention_key
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"][
            "kernel"
        ] = codereviewerx_attention_out
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"][
            "kernel"
        ] = codereviewerx_attention_query
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"][
            "kernel"
        ] = codereviewerx_attention_value

        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"][
            "weight"
        ] = codereviewerx_attention_layer_norm

        if split_mlp_wi:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_0"][
                "kernel"
            ] = codereviewerx_mlp_wi_0
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_1"][
                "kernel"
            ] = codereviewerx_mlp_wi_1
        else:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi"][
                "kernel"
            ] = codereviewerx_mlp_wi

        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wo"][
            "kernel"
        ] = codereviewerx_mlp_wo
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"][
            "weight"
        ] = codereviewerx_mlp_layer_norm

    # Only for layer 0:
    codereviewerx_encoder_rel_embedding = codereviewerx_model["target"]["encoder"]["relpos_bias"]["rel_embedding"].T
    flax_model.params["encoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = codereviewerx_encoder_rel_embedding

    # Assigning
    codereviewerx_encoder_norm = codereviewerx_model["target"]["encoder"]["encoder_norm"]["scale"]
    flax_model.params["encoder"]["final_layer_norm"]["weight"] = codereviewerx_encoder_norm

    # Decoder
    for layer_index in range(config.num_decoder_layers):
        layer_name = f"layers_{str(layer_index)}"

        # Self-Attention
        codereviewerx_attention_key = codereviewerx_model["target"]["decoder"][layer_name]["self_attention"]["key"]["kernel"]
        codereviewerx_attention_out = codereviewerx_model["target"]["decoder"][layer_name]["self_attention"]["out"]["kernel"]
        codereviewerx_attention_query = codereviewerx_model["target"]["decoder"][layer_name]["self_attention"]["query"]["kernel"]
        codereviewerx_attention_value = codereviewerx_model["target"]["decoder"][layer_name]["self_attention"]["value"]["kernel"]

        # Layer Normalization
        codereviewerx_pre_attention_layer_norm = codereviewerx_model["target"]["decoder"][layer_name]["pre_self_attention_layer_norm"][
            "scale"
        ]

        # Encoder-Decoder-Attention
        codereviewerx_enc_dec_attention_key = codereviewerx_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["key"][
            "kernel"
        ]
        codereviewerx_enc_dec_attention_out = codereviewerx_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["out"][
            "kernel"
        ]
        codereviewerx_enc_dec_attention_query = codereviewerx_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["query"][
            "kernel"
        ]
        codereviewerx_enc_dec_attention_value = codereviewerx_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["value"][
            "kernel"
        ]

        # Layer Normalization
        codereviewerx_cross_layer_norm = codereviewerx_model["target"]["decoder"][layer_name]["pre_cross_attention_layer_norm"]["scale"]

        # MLP
        if split_mlp_wi:
            codereviewerx_mlp_wi_0 = codereviewerx_model["target"]["decoder"][layer_name]["mlp"]["wi_0"]["kernel"]
            codereviewerx_mlp_wi_1 = codereviewerx_model["target"]["decoder"][layer_name]["mlp"]["wi_1"]["kernel"]
        else:
            codereviewerx_mlp_wi = codereviewerx_model["target"]["decoder"][layer_name]["mlp"]["wi"]["kernel"]

        codereviewerx_mlp_wo = codereviewerx_model["target"]["decoder"][layer_name]["mlp"]["wo"]["kernel"]

        # Layer Normalization
        tx5_mlp_layer_norm = codereviewerx_model["target"]["decoder"][layer_name]["pre_mlp_layer_norm"]["scale"]

        # Assigning
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"][
            "kernel"
        ] = codereviewerx_attention_key
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"][
            "kernel"
        ] = codereviewerx_attention_out
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"][
            "kernel"
        ] = codereviewerx_attention_query
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"][
            "kernel"
        ] = codereviewerx_attention_value

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"][
            "weight"
        ] = codereviewerx_pre_attention_layer_norm

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["k"][
            "kernel"
        ] = codereviewerx_enc_dec_attention_key
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["o"][
            "kernel"
        ] = codereviewerx_enc_dec_attention_out
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["q"][
            "kernel"
        ] = codereviewerx_enc_dec_attention_query
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["v"][
            "kernel"
        ] = codereviewerx_enc_dec_attention_value

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"][
            "weight"
        ] = codereviewerx_cross_layer_norm

        if split_mlp_wi:
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi_0"][
                "kernel"
            ] = codereviewerx_mlp_wi_0
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi_1"][
                "kernel"
            ] = codereviewerx_mlp_wi_1
        else:
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi"][
                "kernel"
            ] = codereviewerx_mlp_wi

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wo"][
            "kernel"
        ] = codereviewerx_mlp_wo

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["layer_norm"][
            "weight"
        ] = tx5_mlp_layer_norm

    # Decoder Normalization
    tx5_decoder_norm = codereviewerx_model["target"]["decoder"]["decoder_norm"]["scale"]
    flax_model.params["decoder"]["final_layer_norm"]["weight"] = tx5_decoder_norm

    # Only for layer 0:
    codereviewerx_decoder_rel_embedding = codereviewerx_model["target"]["decoder"]["relpos_bias"]["rel_embedding"].T
    flax_model.params["decoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = codereviewerx_decoder_rel_embedding

    # Token Embeddings
    tx5_token_embeddings = codereviewerx_model["target"]["token_embedder"]["embedding"]
    flax_model.params["shared"]["embedding"] = tx5_token_embeddings

    # LM Head (only in v1.1 checkpoints)
    if "logits_dense" in codereviewerx_model["target"]["decoder"]:
        flax_model.params["lm_head"]["kernel"] = codereviewerx_model["target"]["decoder"]["logits_dense"]["kernel"]

    flax_model.save_pretrained(flax_dump_folder_path)
    print("CodeReviewerX Model was sucessfully converted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--codereviewerx_checkpoint_path", default=None, type=str, required=True, help="Path the TX5 checkpoint."
    )
    parser.add_argument("--config_name", default=None, type=str, required=True, help="Config name of CodeReviewer model.")
    parser.add_argument(
        "--flax_dump_folder_path", default=None, type=str, required=True, help="Path to the output FLAX model."
    )
    args = parser.parse_args()
    convert_codereviewerx_checkpoint_to_flax(args.codereviewerx_checkpoint_path, args.config_name, args.flax_dump_folder_path)
