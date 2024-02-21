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

"""Convert FusionInDecoderX checkpoints from the original repository to JAX/FLAX model."""

import argparse

from fusionindecoderx import checkpoints

from transformers import FlaxFusionInDecoderForConditionalGeneration, FusionInDecoderConfig


def convert_fusionindecoderx_checkpoint_to_flax(fusionindecoderx_checkpoint_path, config_name, flax_dump_folder_path):
    config = FusionInDecoderConfig.from_pretrained(config_name)
    flax_model = FlaxFusionInDecoderForConditionalGeneration(config=config)
    fusionindecoderx_model = checkpoints.load_fusionindecoderx_checkpoint(fusionindecoderx_checkpoint_path)

    split_mlp_wi = "wi_0" in fusionindecoderx_model["target"]["encoder"]["layers_0"]["mlp"]

    # Encoder
    for layer_index in range(config.num_layers):
        layer_name = f"layers_{str(layer_index)}"

        # Self-Attention
        fusionindecoderx_attention_key = fusionindecoderx_model["target"]["encoder"][layer_name]["attention"]["key"][
            "kernel"
        ]
        fusionindecoderx_attention_out = fusionindecoderx_model["target"]["encoder"][layer_name]["attention"]["out"][
            "kernel"
        ]
        fusionindecoderx_attention_query = fusionindecoderx_model["target"]["encoder"][layer_name]["attention"][
            "query"
        ]["kernel"]
        fusionindecoderx_attention_value = fusionindecoderx_model["target"]["encoder"][layer_name]["attention"][
            "value"
        ]["kernel"]

        # Layer Normalization
        fusionindecoderx_attention_layer_norm = fusionindecoderx_model["target"]["encoder"][layer_name][
            "pre_attention_layer_norm"
        ]["scale"]

        if split_mlp_wi:
            fusionindecoderx_mlp_wi_0 = fusionindecoderx_model["target"]["encoder"][layer_name]["mlp"]["wi_0"][
                "kernel"
            ]
            fusionindecoderx_mlp_wi_1 = fusionindecoderx_model["target"]["encoder"][layer_name]["mlp"]["wi_1"][
                "kernel"
            ]
        else:
            fusionindecoderx_mlp_wi = fusionindecoderx_model["target"]["encoder"][layer_name]["mlp"]["wi"]["kernel"]

        fusionindecoderx_mlp_wo = fusionindecoderx_model["target"]["encoder"][layer_name]["mlp"]["wo"]["kernel"]

        # Layer Normalization
        fusionindecoderx_mlp_layer_norm = fusionindecoderx_model["target"]["encoder"][layer_name][
            "pre_mlp_layer_norm"
        ]["scale"]

        # Assigning
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"][
            "kernel"
        ] = fusionindecoderx_attention_key
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"][
            "kernel"
        ] = fusionindecoderx_attention_out
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"][
            "kernel"
        ] = fusionindecoderx_attention_query
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"][
            "kernel"
        ] = fusionindecoderx_attention_value

        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"][
            "weight"
        ] = fusionindecoderx_attention_layer_norm

        if split_mlp_wi:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_0"][
                "kernel"
            ] = fusionindecoderx_mlp_wi_0
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_1"][
                "kernel"
            ] = fusionindecoderx_mlp_wi_1
        else:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi"][
                "kernel"
            ] = fusionindecoderx_mlp_wi

        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wo"][
            "kernel"
        ] = fusionindecoderx_mlp_wo
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"][
            "weight"
        ] = fusionindecoderx_mlp_layer_norm

    # Only for layer 0:
    fusionindecoderx_encoder_rel_embedding = fusionindecoderx_model["target"]["encoder"]["relpos_bias"][
        "rel_embedding"
    ].T
    flax_model.params["encoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = fusionindecoderx_encoder_rel_embedding

    # Assigning
    fusionindecoderx_encoder_norm = fusionindecoderx_model["target"]["encoder"]["encoder_norm"]["scale"]
    flax_model.params["encoder"]["final_layer_norm"]["weight"] = fusionindecoderx_encoder_norm

    # Decoder
    for layer_index in range(config.num_decoder_layers):
        layer_name = f"layers_{str(layer_index)}"

        # Self-Attention
        fusionindecoderx_attention_key = fusionindecoderx_model["target"]["decoder"][layer_name]["self_attention"][
            "key"
        ]["kernel"]
        fusionindecoderx_attention_out = fusionindecoderx_model["target"]["decoder"][layer_name]["self_attention"][
            "out"
        ]["kernel"]
        fusionindecoderx_attention_query = fusionindecoderx_model["target"]["decoder"][layer_name]["self_attention"][
            "query"
        ]["kernel"]
        fusionindecoderx_attention_value = fusionindecoderx_model["target"]["decoder"][layer_name]["self_attention"][
            "value"
        ]["kernel"]

        # Layer Normalization
        fusionindecoderx_pre_attention_layer_norm = fusionindecoderx_model["target"]["decoder"][layer_name][
            "pre_self_attention_layer_norm"
        ]["scale"]

        # Encoder-Decoder-Attention
        fusionindecoderx_enc_dec_attention_key = fusionindecoderx_model["target"]["decoder"][layer_name][
            "encoder_decoder_attention"
        ]["key"]["kernel"]
        fusionindecoderx_enc_dec_attention_out = fusionindecoderx_model["target"]["decoder"][layer_name][
            "encoder_decoder_attention"
        ]["out"]["kernel"]
        fusionindecoderx_enc_dec_attention_query = fusionindecoderx_model["target"]["decoder"][layer_name][
            "encoder_decoder_attention"
        ]["query"]["kernel"]
        fusionindecoderx_enc_dec_attention_value = fusionindecoderx_model["target"]["decoder"][layer_name][
            "encoder_decoder_attention"
        ]["value"]["kernel"]

        # Layer Normalization
        fusionindecoderx_cross_layer_norm = fusionindecoderx_model["target"]["decoder"][layer_name][
            "pre_cross_attention_layer_norm"
        ]["scale"]

        # MLP
        if split_mlp_wi:
            fusionindecoderx_mlp_wi_0 = fusionindecoderx_model["target"]["decoder"][layer_name]["mlp"]["wi_0"][
                "kernel"
            ]
            fusionindecoderx_mlp_wi_1 = fusionindecoderx_model["target"]["decoder"][layer_name]["mlp"]["wi_1"][
                "kernel"
            ]
        else:
            fusionindecoderx_mlp_wi = fusionindecoderx_model["target"]["decoder"][layer_name]["mlp"]["wi"]["kernel"]

        fusionindecoderx_mlp_wo = fusionindecoderx_model["target"]["decoder"][layer_name]["mlp"]["wo"]["kernel"]

        # Layer Normalization
        tx5_mlp_layer_norm = fusionindecoderx_model["target"]["decoder"][layer_name]["pre_mlp_layer_norm"]["scale"]

        # Assigning
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"][
            "kernel"
        ] = fusionindecoderx_attention_key
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"][
            "kernel"
        ] = fusionindecoderx_attention_out
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"][
            "kernel"
        ] = fusionindecoderx_attention_query
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"][
            "kernel"
        ] = fusionindecoderx_attention_value

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"][
            "weight"
        ] = fusionindecoderx_pre_attention_layer_norm

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["k"][
            "kernel"
        ] = fusionindecoderx_enc_dec_attention_key
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["o"][
            "kernel"
        ] = fusionindecoderx_enc_dec_attention_out
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["q"][
            "kernel"
        ] = fusionindecoderx_enc_dec_attention_query
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["v"][
            "kernel"
        ] = fusionindecoderx_enc_dec_attention_value

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"][
            "weight"
        ] = fusionindecoderx_cross_layer_norm

        if split_mlp_wi:
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi_0"][
                "kernel"
            ] = fusionindecoderx_mlp_wi_0
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi_1"][
                "kernel"
            ] = fusionindecoderx_mlp_wi_1
        else:
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi"][
                "kernel"
            ] = fusionindecoderx_mlp_wi

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wo"][
            "kernel"
        ] = fusionindecoderx_mlp_wo

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["layer_norm"][
            "weight"
        ] = tx5_mlp_layer_norm

    # Decoder Normalization
    tx5_decoder_norm = fusionindecoderx_model["target"]["decoder"]["decoder_norm"]["scale"]
    flax_model.params["decoder"]["final_layer_norm"]["weight"] = tx5_decoder_norm

    # Only for layer 0:
    fusionindecoderx_decoder_rel_embedding = fusionindecoderx_model["target"]["decoder"]["relpos_bias"][
        "rel_embedding"
    ].T
    flax_model.params["decoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = fusionindecoderx_decoder_rel_embedding

    # Token Embeddings
    tx5_token_embeddings = fusionindecoderx_model["target"]["token_embedder"]["embedding"]
    flax_model.params["shared"]["embedding"] = tx5_token_embeddings

    # LM Head (only in v1.1 checkpoints)
    if "logits_dense" in fusionindecoderx_model["target"]["decoder"]:
        flax_model.params["lm_head"]["kernel"] = fusionindecoderx_model["target"]["decoder"]["logits_dense"]["kernel"]

    flax_model.save_pretrained(flax_dump_folder_path)
    print("FusionInDecoderX Model was sucessfully converted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fusionindecoderx_checkpoint_path", default=None, type=str, required=True, help="Path the TX5 checkpoint."
    )
    parser.add_argument(
        "--config_name", default=None, type=str, required=True, help="Config name of FusionInDecoder model."
    )
    parser.add_argument(
        "--flax_dump_folder_path", default=None, type=str, required=True, help="Path to the output FLAX model."
    )
    args = parser.parse_args()
    convert_fusionindecoderx_checkpoint_to_flax(
        args.fusionindecoderx_checkpoint_path, args.config_name, args.flax_dump_folder_path
    )
