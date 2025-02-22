# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert DPT checkpoints from the original repository. URL: https://github.com/isl-org/DPT"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import DPTConfig, DPTForDepthEstimation, DPTForSemanticSegmentation, DPTImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dpt_config(checkpoint_url):
    config = DPTConfig(embedding_type="hybrid")

    if "large" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.backbone_out_indices = [5, 11, 17, 23]
        config.neck_hidden_sizes = [256, 512, 1024, 1024]
        expected_shape = (1, 384, 384)

    if "nyu" in checkpoint_url or "midas" in checkpoint_url:
        config.hidden_size = 768
        config.reassemble_factors = [1, 1, 1, 0.5]
        config.neck_hidden_sizes = [256, 512, 768, 768]
        config.num_labels = 150
        config.patch_size = 16
        expected_shape = (1, 384, 384)
        config.use_batch_norm_in_fusion_residual = False
        config.readout_type = "project"

    if "ade" in checkpoint_url:
        config.use_batch_norm_in_fusion_residual = True
        config.hidden_size = 768
        config.reassemble_stage = [1, 1, 1, 0.5]
        config.num_labels = 150
        config.patch_size = 16
        repo_id = "huggingface/label-files"
        filename = "ade20k-id2label.json"
        id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        expected_shape = [1, 150, 480, 480]

    return config, expected_shape


def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(name):
    if (
        "pretrained.model" in name
        and "cls_token" not in name
        and "pos_embed" not in name
        and "patch_embed" not in name
    ):
        name = name.replace("pretrained.model", "dpt.encoder")
    if "pretrained.model" in name:
        name = name.replace("pretrained.model", "dpt.embeddings")
    if "patch_embed" in name:
        name = name.replace("patch_embed", "")
    if "pos_embed" in name:
        name = name.replace("pos_embed", "position_embeddings")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "proj" in name and "project" not in name:
        name = name.replace("proj", "projection")
    if "blocks" in name:
        name = name.replace("blocks", "layer")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "norm1" in name and "backbone" not in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name and "backbone" not in name:
        name = name.replace("norm2", "layernorm_after")
    if "scratch.output_conv" in name:
        name = name.replace("scratch.output_conv", "head")
    if "scratch" in name:
        name = name.replace("scratch", "neck")
    if "layer1_rn" in name:
        name = name.replace("layer1_rn", "convs.0")
    if "layer2_rn" in name:
        name = name.replace("layer2_rn", "convs.1")
    if "layer3_rn" in name:
        name = name.replace("layer3_rn", "convs.2")
    if "layer4_rn" in name:
        name = name.replace("layer4_rn", "convs.3")
    if "refinenet" in name:
        layer_idx = int(name[len("neck.refinenet") : len("neck.refinenet") + 1])
        # tricky here: we need to map 4 to 0, 3 to 1, 2 to 2 and 1 to 3
        name = name.replace(f"refinenet{layer_idx}", f"fusion_stage.layers.{abs(layer_idx - 4)}")
    if "out_conv" in name:
        name = name.replace("out_conv", "projection")
    if "resConfUnit1" in name:
        name = name.replace("resConfUnit1", "residual_layer1")
    if "resConfUnit2" in name:
        name = name.replace("resConfUnit2", "residual_layer2")
    if "conv1" in name:
        name = name.replace("conv1", "convolution1")
    if "conv2" in name:
        name = name.replace("conv2", "convolution2")
    # readout blocks
    if "pretrained.act_postprocess1.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess1.0.project.0", "neck.reassemble_stage.readout_projects.0.0")
    if "pretrained.act_postprocess2.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess2.0.project.0", "neck.reassemble_stage.readout_projects.1.0")
    if "pretrained.act_postprocess3.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess3.0.project.0", "neck.reassemble_stage.readout_projects.2.0")
    if "pretrained.act_postprocess4.0.project.0" in name:
        name = name.replace("pretrained.act_postprocess4.0.project.0", "neck.reassemble_stage.readout_projects.3.0")

    # resize blocks
    if "pretrained.act_postprocess1.3" in name:
        name = name.replace("pretrained.act_postprocess1.3", "neck.reassemble_stage.layers.0.projection")
    if "pretrained.act_postprocess1.4" in name:
        name = name.replace("pretrained.act_postprocess1.4", "neck.reassemble_stage.layers.0.resize")
    if "pretrained.act_postprocess2.3" in name:
        name = name.replace("pretrained.act_postprocess2.3", "neck.reassemble_stage.layers.1.projection")
    if "pretrained.act_postprocess2.4" in name:
        name = name.replace("pretrained.act_postprocess2.4", "neck.reassemble_stage.layers.1.resize")
    if "pretrained.act_postprocess3.3" in name:
        name = name.replace("pretrained.act_postprocess3.3", "neck.reassemble_stage.layers.2.projection")
    if "pretrained.act_postprocess4.3" in name:
        name = name.replace("pretrained.act_postprocess4.3", "neck.reassemble_stage.layers.3.projection")
    if "pretrained.act_postprocess4.4" in name:
        name = name.replace("pretrained.act_postprocess4.4", "neck.reassemble_stage.layers.3.resize")
    if "pretrained" in name:
        name = name.replace("pretrained", "dpt")
    if "bn" in name:
        name = name.replace("bn", "batch_norm")
    if "head" in name:
        name = name.replace("head", "head.head")
    if "encoder.norm" in name:
        name = name.replace("encoder.norm", "layernorm")
    if "auxlayer" in name:
        name = name.replace("auxlayer", "auxiliary_head.head")
    if "backbone" in name:
        name = name.replace("backbone", "backbone.bit.encoder")

    if ".." in name:
        name = name.replace("..", ".")

    if "stem.conv" in name:
        name = name.replace("stem.conv", "bit.embedder.convolution")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "convolution" in name and "backbone" in name:
        name = name.replace("convolution", "conv")
    if "layer" in name and "backbone" in name:
        name = name.replace("layer", "layers")
    if "backbone.bit.encoder.bit" in name:
        name = name.replace("backbone.bit.encoder.bit", "backbone.bit")
    if "embedder.conv" in name:
        name = name.replace("embedder.conv", "embedder.convolution")
    if "backbone.bit.encoder.stem.norm" in name:
        name = name.replace("backbone.bit.encoder.stem.norm", "backbone.bit.embedder.norm")
    return name


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"dpt.encoder.layer.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"dpt.encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_dpt_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub, model_name, show_prediction):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """

    # define DPT configuration based on URL
    config, expected_shape = get_dpt_config(checkpoint_url)
    # load original state_dict from URL
    # state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    state_dict = torch.load(checkpoint_url, map_location="cpu")
    # remove certain keys
    remove_ignore_keys_(state_dict)
    # rename keys
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # read in qkv matrices
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = DPTForSemanticSegmentation(config) if "ade" in checkpoint_url else DPTForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Check outputs on an image
    size = 480 if "ade" in checkpoint_url else 384
    image_processor = DPTImageProcessor(size=size)

    image = prepare_img()
    encoding = image_processor(image, return_tensors="pt")

    # forward pass
    outputs = model(**encoding).logits if "ade" in checkpoint_url else model(**encoding).predicted_depth

    if show_prediction:
        prediction = (
            torch.nn.functional.interpolate(
                outputs.unsqueeze(1),
                size=(image.size[1], image.size[0]),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        Image.fromarray((prediction / prediction.max()) * 255).show()

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub("ybelkada/dpt-hybrid-midas")
        image_processor.push_to_hub("ybelkada/dpt-hybrid-midas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        type=str,
        help="URL of the original DPT checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    parser.add_argument(
        "--model_name",
        default="dpt-large",
        type=str,
        help="Name of the model, in case you're pushing to the hub.",
    )
    parser.add_argument(
        "--show_prediction",
        action="store_true",
    )

    args = parser.parse_args()
    convert_dpt_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub, args.model_name, args.show_prediction
    )
