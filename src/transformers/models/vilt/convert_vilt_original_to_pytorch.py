# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert ViLT checkpoints from the original Github repository."""


import argparse
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import BertTokenizer, ViltConfig, ViltForVisualQuestionAnswering, ViltModel, ViTFeatureExtractor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"transformer.blocks.{i}.norm1.weight", f"vilt.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"transformer.blocks.{i}.norm1.bias", f"vilt.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.weight", f"vilt.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"transformer.blocks.{i}.attn.proj.bias", f"vilt.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"transformer.blocks.{i}.norm2.weight", f"vilt.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"transformer.blocks.{i}.norm2.bias", f"vilt.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"transformer.blocks.{i}.mlp.fc1.weight", f"vilt.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc1.bias", f"vilt.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.weight", f"vilt.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"transformer.blocks.{i}.mlp.fc2.bias", f"vilt.encoder.layer.{i}.output.dense.bias"))

    # embeddings
    rename_keys.extend(
        [
            # text embeddings
            ("text_embeddings.word_embeddings.weight", "vilt.embeddings.text_embeddings.word_embeddings.weight"),
            ("text_embeddings.position_embeddings.weight", "vilt.embeddings.text_embeddings.position_embeddings.weight"),
            ("text_embeddings.position_ids", "vilt.embeddings.text_embeddings.position_ids"),
            ("text_embeddings.token_type_embeddings.weight", "vilt.embeddings.text_embeddings.token_type_embeddings.weight"),
            ("text_embeddings.LayerNorm.weight", "vilt.embeddings.text_embeddings.LayerNorm.weight"),
            ("text_embeddings.LayerNorm.bias", "vilt.embeddings.text_embeddings.LayerNorm.bias"),
            # patch embeddings
            ("transformer.cls_token", "vilt.embeddings.cls_token"),
            ("transformer.patch_embed.proj.weight", "vilt.embeddings.patch_embeddings.projection.weight"),
            ("transformer.patch_embed.proj.bias", "vilt.embeddings.patch_embeddings.projection.bias"),
            ("transformer.pos_embed", "vilt.embeddings.position_embeddings"),
            # token type embeddings
            ("token_type_embeddings.weight", "vilt.embeddings.token_type_embeddings.weight"),
        ]
    )

    # final layernorm + pooler
    rename_keys.extend(
        [
            ("transformer.norm.weight", "vilt.layernorm.weight"),
            ("transformer.norm.bias", "vilt.layernorm.bias"),
            ("pooler.dense.weight", "vilt.pooler.dense.weight"),
            ("pooler.dense.bias", "vilt.pooler.dense.bias")
        ]       
    )

    # classifier head(s)
    if base_model:
        rename_keys.extend(
            [
                # TODO
                # ("classifier.0.weight", ""),
                # ("classifier.0.bias", ""),
                # ("classifier.1.weight", ""),
                # ("classifier.1.bias", ""),
                # ("classifier.3.weight", ""),
                # ("classifier.3.bias", ""),
            ]
        )

        # if just the base model, we should remove "vilt" from all keys that start with "vilt"
        rename_keys = [(pair[0], pair[1][5:]) if pair[1].startswith("vilt") else pair for pair in rename_keys]
    else:
        # classification head
        rename_keys.extend(
            [
                
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vilt."
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"transformer.blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "mlm_score.bias",
        "mlm_score.transform.dense.weight",
        "mlm_score.transform.dense.bias",
        "mlm_score.transform.LayerNorm.weight",
        "mlm_score.transform.LayerNorm.bias",
        "mlm_score.decoder.weight", 
        "itm_score.fc.weight", 
        "itm_score.fc.bias",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vilt_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ViLT structure.
    """

    # define default ViT configuration
    config = ViltConfig(image_size=384, patch_size=32)
    base_model = False
    if "mlm" in checkpoint_url:
        base_model = True
    # load state_dict of original model, remove and rename some keys
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")['state_dict']
    remove_ignore_keys_(state_dict)
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)

    # load HuggingFace model
    if base_model:
        model = ViltModel(config).eval()
    else:
        model = ViltForVisualQuestionAnswering(config).eval()
    model.load_state_dict(state_dict)

    # Prepare text + image
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "hello world"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    image = prepare_img()
    feature_extractor = ViTFeatureExtractor(size=384)
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    # Forward pass
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    print(outputs.last_hidden_state.shape)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vilt_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
