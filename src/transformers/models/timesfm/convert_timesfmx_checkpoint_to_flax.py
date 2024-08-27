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

"""Convert TimesFMX checkpoints from the original repository to JAX/FLAX model."""

import argparse

from timesfmx import checkpoints

from transformers import FlaxTimesFMForConditionalGeneration, TimesFMConfig


def convert_timesfmx_checkpoint_to_flax(timesfmx_checkpoint_path, config_name, flax_dump_folder_path):
    config = TimesFMConfig.from_pretrained(config_name)
    flax_model = FlaxTimesFMForConditionalGeneration(config=config)
    timesfmx_model = checkpoints.load_timesfmx_checkpoint(timesfmx_checkpoint_path)

    split_mlp_wi = "wi_0" in timesfmx_model["target"]["encoder"]["layers_0"]["mlp"]

    # Encoder
    for layer_index in range(config.num_layers):
        layer_name = f"layers_{str(layer_index)}"

        # Self-Attention
        timesfmx_attention_key = timesfmx_model["target"]["encoder"][layer_name]["attention"]["key"]["kernel"]
        timesfmx_attention_out = timesfmx_model["target"]["encoder"][layer_name]["attention"]["out"]["kernel"]
        timesfmx_attention_query = timesfmx_model["target"]["encoder"][layer_name]["attention"]["query"]["kernel"]
        timesfmx_attention_value = timesfmx_model["target"]["encoder"][layer_name]["attention"]["value"]["kernel"]

        # Layer Normalization
        timesfmx_attention_layer_norm = timesfmx_model["target"]["encoder"][layer_name]["pre_attention_layer_norm"]["scale"]

        if split_mlp_wi:
            timesfmx_mlp_wi_0 = timesfmx_model["target"]["encoder"][layer_name]["mlp"]["wi_0"]["kernel"]
            timesfmx_mlp_wi_1 = timesfmx_model["target"]["encoder"][layer_name]["mlp"]["wi_1"]["kernel"]
        else:
            timesfmx_mlp_wi = timesfmx_model["target"]["encoder"][layer_name]["mlp"]["wi"]["kernel"]

        timesfmx_mlp_wo = timesfmx_model["target"]["encoder"][layer_name]["mlp"]["wo"]["kernel"]

        # Layer Normalization
        timesfmx_mlp_layer_norm = timesfmx_model["target"]["encoder"][layer_name]["pre_mlp_layer_norm"]["scale"]

        # Assigning
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"]["kernel"] = (
            timesfmx_attention_key
        )
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"]["kernel"] = (
            timesfmx_attention_out
        )
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"]["kernel"] = (
            timesfmx_attention_query
        )
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"]["kernel"] = (
            timesfmx_attention_value
        )

        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"]["weight"] = (
            timesfmx_attention_layer_norm
        )

        if split_mlp_wi:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_0"][
                "kernel"
            ] = timesfmx_mlp_wi_0
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi_1"][
                "kernel"
            ] = timesfmx_mlp_wi_1
        else:
            flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wi"]["kernel"] = (
                timesfmx_mlp_wi
            )

        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["DenseReluDense"]["wo"]["kernel"] = (
            timesfmx_mlp_wo
        )
        flax_model.params["encoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"]["weight"] = (
            timesfmx_mlp_layer_norm
        )

    # Only for layer 0:
    timesfmx_encoder_rel_embedding = timesfmx_model["target"]["encoder"]["relpos_bias"]["rel_embedding"].T
    flax_model.params["encoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = timesfmx_encoder_rel_embedding

    # Assigning
    timesfmx_encoder_norm = timesfmx_model["target"]["encoder"]["encoder_norm"]["scale"]
    flax_model.params["encoder"]["final_layer_norm"]["weight"] = timesfmx_encoder_norm

    # Decoder
    for layer_index in range(config.num_decoder_layers):
        layer_name = f"layers_{str(layer_index)}"

        # Self-Attention
        timesfmx_attention_key = timesfmx_model["target"]["decoder"][layer_name]["self_attention"]["key"]["kernel"]
        timesfmx_attention_out = timesfmx_model["target"]["decoder"][layer_name]["self_attention"]["out"]["kernel"]
        timesfmx_attention_query = timesfmx_model["target"]["decoder"][layer_name]["self_attention"]["query"]["kernel"]
        timesfmx_attention_value = timesfmx_model["target"]["decoder"][layer_name]["self_attention"]["value"]["kernel"]

        # Layer Normalization
        timesfmx_pre_attention_layer_norm = timesfmx_model["target"]["decoder"][layer_name]["pre_self_attention_layer_norm"][
            "scale"
        ]

        # Encoder-Decoder-Attention
        timesfmx_enc_dec_attention_key = timesfmx_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["key"][
            "kernel"
        ]
        timesfmx_enc_dec_attention_out = timesfmx_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["out"][
            "kernel"
        ]
        timesfmx_enc_dec_attention_query = timesfmx_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["query"][
            "kernel"
        ]
        timesfmx_enc_dec_attention_value = timesfmx_model["target"]["decoder"][layer_name]["encoder_decoder_attention"]["value"][
            "kernel"
        ]

        # Layer Normalization
        timesfmx_cross_layer_norm = timesfmx_model["target"]["decoder"][layer_name]["pre_cross_attention_layer_norm"]["scale"]

        # MLP
        if split_mlp_wi:
            timesfmx_mlp_wi_0 = timesfmx_model["target"]["decoder"][layer_name]["mlp"]["wi_0"]["kernel"]
            timesfmx_mlp_wi_1 = timesfmx_model["target"]["decoder"][layer_name]["mlp"]["wi_1"]["kernel"]
        else:
            timesfmx_mlp_wi = timesfmx_model["target"]["decoder"][layer_name]["mlp"]["wi"]["kernel"]

        timesfmx_mlp_wo = timesfmx_model["target"]["decoder"][layer_name]["mlp"]["wo"]["kernel"]

        # Layer Normalization
        tx5_mlp_layer_norm = timesfmx_model["target"]["decoder"][layer_name]["pre_mlp_layer_norm"]["scale"]

        # Assigning
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["k"]["kernel"] = (
            timesfmx_attention_key
        )
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["o"]["kernel"] = (
            timesfmx_attention_out
        )
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["q"]["kernel"] = (
            timesfmx_attention_query
        )
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["SelfAttention"]["v"]["kernel"] = (
            timesfmx_attention_value
        )

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["0"]["layer_norm"]["weight"] = (
            timesfmx_pre_attention_layer_norm
        )

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["k"]["kernel"] = (
            timesfmx_enc_dec_attention_key
        )
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["o"]["kernel"] = (
            timesfmx_enc_dec_attention_out
        )
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["q"]["kernel"] = (
            timesfmx_enc_dec_attention_query
        )
        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["EncDecAttention"]["v"]["kernel"] = (
            timesfmx_enc_dec_attention_value
        )

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["1"]["layer_norm"]["weight"] = (
            timesfmx_cross_layer_norm
        )

        if split_mlp_wi:
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi_0"][
                "kernel"
            ] = timesfmx_mlp_wi_0
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi_1"][
                "kernel"
            ] = timesfmx_mlp_wi_1
        else:
            flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wi"]["kernel"] = (
                timesfmx_mlp_wi
            )

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["DenseReluDense"]["wo"]["kernel"] = (
            timesfmx_mlp_wo
        )

        flax_model.params["decoder"]["block"][str(layer_index)]["layer"]["2"]["layer_norm"]["weight"] = (
            tx5_mlp_layer_norm
        )

    # Decoder Normalization
    tx5_decoder_norm = timesfmx_model["target"]["decoder"]["decoder_norm"]["scale"]
    flax_model.params["decoder"]["final_layer_norm"]["weight"] = tx5_decoder_norm

    # Only for layer 0:
    timesfmx_decoder_rel_embedding = timesfmx_model["target"]["decoder"]["relpos_bias"]["rel_embedding"].T
    flax_model.params["decoder"]["block"]["0"]["layer"]["0"]["SelfAttention"]["relative_attention_bias"][
        "embedding"
    ] = timesfmx_decoder_rel_embedding

    # Token Embeddings
    tx5_token_embeddings = timesfmx_model["target"]["token_embedder"]["embedding"]
    flax_model.params["shared"]["embedding"] = tx5_token_embeddings

    # LM Head (only in v1.1 checkpoints)
    if "logits_dense" in timesfmx_model["target"]["decoder"]:
        flax_model.params["lm_head"]["kernel"] = timesfmx_model["target"]["decoder"]["logits_dense"]["kernel"]

    flax_model.save_pretrained(flax_dump_folder_path)
    print("TimesFMX Model was sucessfully converted!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--timesfmx_checkpoint_path", default=None, type=str, required=True, help="Path the TX5 checkpoint."
    )
    parser.add_argument("--config_name", default=None, type=str, required=True, help="Config name of TimesFM model.")
    parser.add_argument(
        "--flax_dump_folder_path", default=None, type=str, required=True, help="Path to the output FLAX model."
    )
    args = parser.parse_args()
    convert_timesfmx_checkpoint_to_flax(args.timesfmx_checkpoint_path, args.config_name, args.flax_dump_folder_path)
