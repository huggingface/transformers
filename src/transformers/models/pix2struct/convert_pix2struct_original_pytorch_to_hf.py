# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import os
import re

import torch
from flax.traverse_util import flatten_dict
from t5x import checkpoints

from transformers import (
    AutoTokenizer,
    Pix2StructConfig,
    Pix2StructForConditionalGeneration,
    Pix2StructImageProcessor,
    Pix2StructProcessor,
    Pix2StructTextConfig,
    Pix2StructVisionConfig,
)


def get_flax_param(t5x_checkpoint_path):
    flax_params = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    flax_params = flatten_dict(flax_params)
    return flax_params


def rename_and_convert_flax_params(flax_dict):
    converted_dict = {}

    CONVERSION_MAPPING = {
        "token_embedder": "embeddings",
        "encoder_norm": "layernorm",
        "kernel": "weight",
        ".out": ".output",
        "scale": "weight",
        "embedders_0.pos_embedding": "row_embedder.weight",
        "embedders_1.pos_embedding": "column_embedder.weight",
    }

    DECODER_CONVERSION_MAPPING = {
        "query": "attention.query",
        "key": "attention.key",
        "value": "attention.value",
        "output.dense": "output",
        "encoder_decoder_attention.o": "encoder_decoder_attention.attention.o",
        "pre_self_attention_layer_norm": "self_attention.layer_norm",
        "pre_cross_attention_layer_norm": "encoder_decoder_attention.layer_norm",
        "mlp.": "mlp.DenseReluDense.",
        "pre_mlp_layer_norm": "mlp.layer_norm",
        "self_attention.o": "self_attention.attention.o",
        "decoder.embeddings.embedding": "decoder.embed_tokens.weight",
        "decoder.relpos_bias.rel_embedding": "decoder.layer.0.self_attention.attention.relative_attention_bias.weight",
        "decoder.decoder_norm.weight": "decoder.final_layer_norm.weight",
        "decoder.logits_dense.weight": "decoder.lm_head.weight",
    }

    for key in flax_dict.keys():
        if "target" in key:
            # remove the first prefix from the key
            new_key = ".".join(key[1:])

            # rename the key
            for old, new in CONVERSION_MAPPING.items():
                new_key = new_key.replace(old, new)

            if "decoder" in new_key:
                for old, new in DECODER_CONVERSION_MAPPING.items():
                    new_key = new_key.replace(old, new)

            if "layers" in new_key and "decoder" not in new_key:
                # use regex to replace the layer number
                new_key = re.sub(r"layers_(\d+)", r"layer.\1", new_key)
                new_key = new_key.replace("encoder", "encoder.encoder")

            elif "layers" in new_key and "decoder" in new_key:
                # use regex to replace the layer number
                new_key = re.sub(r"layers_(\d+)", r"layer.\1", new_key)

            converted_dict[new_key] = flax_dict[key]

    converted_torch_dict = {}
    # convert converted_dict into torch format
    for key in converted_dict.keys():
        if ("embed_tokens" not in key) and ("embedder" not in key):
            converted_torch_dict[key] = torch.from_numpy(converted_dict[key].T)
        else:
            converted_torch_dict[key] = torch.from_numpy(converted_dict[key])

    return converted_torch_dict


def convert_pix2struct_original_pytorch_checkpoint_to_hf(
    t5x_checkpoint_path, pytorch_dump_folder_path, use_large=False, is_vqa=False
):
    flax_params = get_flax_param(t5x_checkpoint_path)

    if not use_large:
        encoder_config = Pix2StructVisionConfig()
        decoder_config = Pix2StructTextConfig()
    else:
        encoder_config = Pix2StructVisionConfig(
            hidden_size=1536, d_ff=3968, num_attention_heads=24, num_hidden_layers=18
        )
        decoder_config = Pix2StructTextConfig(hidden_size=1536, d_ff=3968, num_heads=24, num_layers=18)
    config = Pix2StructConfig(
        vision_config=encoder_config.to_dict(), text_config=decoder_config.to_dict(), is_vqa=is_vqa
    )

    model = Pix2StructForConditionalGeneration(config)

    torch_params = rename_and_convert_flax_params(flax_params)
    model.load_state_dict(torch_params)

    tok = AutoTokenizer.from_pretrained("ybelkada/test-pix2struct-tokenizer")
    image_processor = Pix2StructImageProcessor()
    processor = Pix2StructProcessor(image_processor=image_processor, tokenizer=tok)

    if use_large:
        processor.image_processor.max_patches = 4096

    processor.image_processor.is_vqa = True

    # mkdir if needed
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    model.save_pretrained(pytorch_dump_folder_path)
    processor.save_pretrained(pytorch_dump_folder_path)

    print("Model saved in {}".format(pytorch_dump_folder_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5x_checkpoint_path", default=None, type=str, help="Path to the original T5x checkpoint.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--use_large", action="store_true", help="Use large model.")
    parser.add_argument("--is_vqa", action="store_true", help="Use large model.")
    args = parser.parse_args()

    convert_pix2struct_original_pytorch_checkpoint_to_hf(
        args.t5x_checkpoint_path, args.pytorch_dump_folder_path, args.use_large
    )
