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
"""Convert TrOCR checkpoints from the unilm repository."""


import argparse
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import (
    RobertaConfig,
    RobertaModel,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTFeatureExtractor,
    ViTModel,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"deit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"deit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"deit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"deit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"deit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"deit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"deit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"deit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"deit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"deit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("cls_token", "deit.embeddings.cls_token"),
            ("patch_embed.proj.weight", "deit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "deit.embeddings.patch_embeddings.projection.bias"),
        ]
    )


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        prefix = "deit."
        # queries, keys and values
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"blocks.{i}.attn.v_bias")

        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # gamma_1 and gamma_2
        # we call them lambda because otherwise they are renamed when using .from_pretrained
        gamma_1 = state_dict.pop(f"blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"blocks.{i}.gamma_2")

        state_dict[f"{prefix}encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"{prefix}encoder.layer.{i}.lambda_2"] = gamma_2


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_tr_ocr_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our VisionEncoderDecoderModel structure.
    """
    # define encoder and decoder configs based on checkpoint_url
    encoder_config = ViTConfig()
    decoder_config = RobertaConfig.from_pretrained("roberta-large")

    # size of the architecture
    if "base" in checkpoint_url:
        pass
    elif "large" in checkpoint_url:
        encoder_config.hidden_size = 1024
        encoder_config.intermediate_size = 4096
        encoder_config.num_hidden_layers = 24
        encoder_config.num_attention_heads = 16
    else:
        raise ValueError("Should either find 'base' or 'large' in checkpoint URL")

    # load HuggingFace model
    encoder = ViTModel(encoder_config)
    decoder = RobertaModel(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)["model"]
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    # load state dict
    model.load_state_dict(state_dict)

    # Check outputs on an image
    feature_extractor = ViTFeatureExtractor(
        size=encoder_config.image_size, resample=Image.BILINEAR, do_center_crop=False
    )
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    outputs = model(pixel_values)
    logits = outputs.logits

    # TODO verify logits
    expected_shape = torch.Size([1, 1000])
    assert logits.shape == expected_shape, "Shape of logits not as expected"

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/trocr/model_zoo/fairseq/trocr-base-handwritten.pt",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_tr_ocr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
