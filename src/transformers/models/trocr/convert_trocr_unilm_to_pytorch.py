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

import requests
import torch
from PIL import Image

from transformers import (
    RobertaTokenizer,
    TrOCRConfig,
    TrOCRForCausalLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(encoder_config, decoder_config):
    rename_keys = []
    for i in range(encoder_config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm1.weight", f"encoder.encoder.layer.{i}.layernorm_before.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm1.bias", f"encoder.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.weight", f"encoder.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.attn.proj.bias", f"encoder.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.norm2.weight", f"encoder.encoder.layer.{i}.layernorm_after.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.norm2.bias", f"encoder.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.weight", f"encoder.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc1.bias", f"encoder.encoder.layer.{i}.intermediate.dense.bias")
        )
        rename_keys.append(
            (f"encoder.deit.blocks.{i}.mlp.fc2.weight", f"encoder.encoder.layer.{i}.output.dense.weight")
        )
        rename_keys.append((f"encoder.deit.blocks.{i}.mlp.fc2.bias", f"encoder.encoder.layer.{i}.output.dense.bias"))

    # cls token, position embeddings and patch embeddings of encoder
    rename_keys.extend(
        [
            ("encoder.deit.cls_token", "encoder.embeddings.cls_token"),
            ("encoder.deit.pos_embed", "encoder.embeddings.position_embeddings"),
            ("encoder.deit.patch_embed.proj.weight", "encoder.embeddings.patch_embeddings.projection.weight"),
            ("encoder.deit.patch_embed.proj.bias", "encoder.embeddings.patch_embeddings.projection.bias"),
            ("encoder.deit.norm.weight", "encoder.layernorm.weight"),
            ("encoder.deit.norm.bias", "encoder.layernorm.bias"),
        ]
    )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, encoder_config):
    for i in range(encoder_config.num_hidden_layers):
        # queries, keys and values (only weights, no biases)
        in_proj_weight = state_dict.pop(f"encoder.deit.blocks.{i}.attn.qkv.weight")

        state_dict[f"encoder.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : encoder_config.hidden_size, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            encoder_config.hidden_size : encoder_config.hidden_size * 2, :
        ]
        state_dict[f"encoder.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -encoder_config.hidden_size :, :
        ]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of the IAM Handwriting Database
def prepare_img(checkpoint_url):
    if "handwritten" in checkpoint_url:
        url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg"  # industry
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-12.jpg" # have
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02-10.jpg" # let
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"  #
        # url = "https://fki.tic.heia-fr.ch/static/img/a01-122.jpg"
    elif "printed" in checkpoint_url or "stage1" in checkpoint_url:
        url = "https://www.researchgate.net/profile/Dinh-Sang/publication/338099565/figure/fig8/AS:840413229350922@1577381536857/An-receipt-example-in-the-SROIE-2019-dataset_Q640.jpg"
    im = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return im


@torch.no_grad()
def convert_tr_ocr_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our VisionEncoderDecoderModel structure.
    """
    # define encoder and decoder configs based on checkpoint_url
    encoder_config = ViTConfig(image_size=384, qkv_bias=False)
    decoder_config = TrOCRConfig()

    # size of the architecture
    if "base" in checkpoint_url:
        decoder_config.encoder_hidden_size = 768
    elif "large" in checkpoint_url:
        # use ViT-large encoder
        encoder_config.hidden_size = 1024
        encoder_config.intermediate_size = 4096
        encoder_config.num_hidden_layers = 24
        encoder_config.num_attention_heads = 16
        decoder_config.encoder_hidden_size = 1024
    else:
        raise ValueError("Should either find 'base' or 'large' in checkpoint URL")

    # the large-printed + stage1 checkpoints uses sinusoidal position embeddings, no layernorm afterwards
    if "large-printed" in checkpoint_url or "stage1" in checkpoint_url:
        decoder_config.tie_word_embeddings = False
        decoder_config.activation_function = "relu"
        decoder_config.max_position_embeddings = 1024
        decoder_config.scale_embedding = True
        decoder_config.use_learned_position_embeddings = False
        decoder_config.layernorm_embedding = False

    # load HuggingFace model
    encoder = ViTModel(encoder_config, add_pooling_layer=False)
    decoder = TrOCRForCausalLM(decoder_config)
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.eval()

    # load state_dict of original model, rename some keys
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)["model"]

    rename_keys = create_rename_keys(encoder_config, decoder_config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, encoder_config)

    # remove parameters we don't need
    del state_dict["encoder.deit.head.weight"]
    del state_dict["encoder.deit.head.bias"]
    del state_dict["decoder.version"]

    # add prefix to decoder keys
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith("decoder") and "output_projection" not in key:
            state_dict["decoder.model." + key] = val
        else:
            state_dict[key] = val

    # load state dict
    model.load_state_dict(state_dict)

    # Check outputs on an image
    image_processor = ViTImageProcessor(size=encoder_config.image_size)
    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-large")
    processor = TrOCRProcessor(image_processor, tokenizer)

    pixel_values = processor(images=prepare_img(checkpoint_url), return_tensors="pt").pixel_values

    # verify logits
    decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]])
    outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits

    expected_shape = torch.Size([1, 1, 50265])
    if "trocr-base-handwritten" in checkpoint_url:
        expected_slice = torch.tensor(
            [-1.4502, -4.6683, -0.5347, -2.9291, 9.1435, -3.0571, 8.9764, 1.7560, 8.7358, -1.5311]
        )
    elif "trocr-large-handwritten" in checkpoint_url:
        expected_slice = torch.tensor(
            [-2.6437, -1.3129, -2.2596, -5.3455, 6.3539, 1.7604, 5.4991, 1.4702, 5.6113, 2.0170]
        )
    elif "trocr-base-printed" in checkpoint_url:
        expected_slice = torch.tensor(
            [-5.6816, -5.8388, 1.1398, -6.9034, 6.8505, -2.4393, 1.2284, -1.0232, -1.9661, -3.9210]
        )
    elif "trocr-large-printed" in checkpoint_url:
        expected_slice = torch.tensor(
            [-6.0162, -7.0959, 4.4155, -5.1063, 7.0468, -3.1631, 2.6466, -0.3081, -0.8106, -1.7535]
        )

    if "stage1" not in checkpoint_url:
        assert logits.shape == expected_shape, "Shape of logits not as expected"
        assert torch.allclose(logits[0, 0, :10], expected_slice, atol=1e-3), "First elements of logits not as expected"

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving processor to {pytorch_dump_folder_path}")
    processor.save_pretrained(pytorch_dump_folder_path)


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
