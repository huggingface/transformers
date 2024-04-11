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
"""Convert ZoeDepth checkpoints from the original repository. URL: https://github.com/isl-org/ZoeDepth.

Original logits where obtained by running the following code:
!git clone -b understanding_zoedepth https://github.com/NielsRogge/ZoeDepth
!python inference.py
"""


import argparse
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import BeitConfig, ZoeDepthConfig, ZoeDepthForDepthEstimation, ZoeDepthImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_zoedepth_config(model_name):
    image_size = 384
    backbone_config = BeitConfig(
        image_size=image_size,
        num_hidden_layers=24,
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        use_relative_position_bias=True,
        reshape_hidden_states=False,
        out_features=["stage6", "stage12", "stage18", "stage24"],  # beit-large-512 uses [5, 11, 17, 23],
    )

    neck_hidden_sizes = [256, 512, 1024, 1024]
    bin_centers_type = "softplus" if model_name in ["ZoeD_N", "ZoeD_NK"] else "normed"
    use_multiple_heads = model_name == "ZoeD_NK"
    bin_configurations = [
        {"name": "nyu", "n_bins": 64, "min_depth": 1e-3, "max_depth": 10.0},
        {"name": "kitti", "n_bins": 64, "min_depth": 1e-3, "max_depth": 80.0},
    ]
    config = ZoeDepthConfig(
        backbone_config=backbone_config,
        neck_hidden_sizes=neck_hidden_sizes,
        bin_centers_type=bin_centers_type,
        use_multiple_heads=use_multiple_heads,
        bin_configurations=bin_configurations,
    )

    return config, image_size


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, model_name):
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("core.core.pretrained.model.cls_token", "backbone.embeddings.cls_token"))
    rename_keys.append(("core.core.pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("core.core.pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))

    # Transfomer encoder
    for i in range(config.backbone_config.num_hidden_layers):
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.gamma_1", f"backbone.encoder.layer.{i}.lambda_1"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.gamma_2", f"backbone.encoder.layer.{i}.lambda_2"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.output.dense.bias"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.attn.proj.weight", f"backbone.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.attn.proj.bias", f"backbone.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.attn.relative_position_bias_table", f"backbone.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"))
        rename_keys.append((f"core.core.pretrained.model.blocks.{i}.attn.relative_position_index", f"backbone.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"))

    # activation postprocessing (readout projections + resize blocks)
    for i in range(4):
        rename_keys.append((f"core.core.pretrained.act_postprocess{i+1}.0.project.0.weight", f"neck.reassemble_stage.readout_projects.{i}.0.weight"))
        rename_keys.append((f"core.core.pretrained.act_postprocess{i+1}.0.project.0.bias", f"neck.reassemble_stage.readout_projects.{i}.0.bias"))

        rename_keys.append((f"core.core.pretrained.act_postprocess{i+1}.3.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"))
        rename_keys.append((f"core.core.pretrained.act_postprocess{i+1}.3.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"))

        if i != 2:
            rename_keys.append((f"core.core.pretrained.act_postprocess{i+1}.4.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"))
            rename_keys.append((f"core.core.pretrained.act_postprocess{i+1}.4.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"))

    # refinenet (tricky here)
    mapping = {1:3, 2:2, 3:1, 4:0}

    for i in range(1, 5):
        j = mapping[i]
        rename_keys.append((f"core.core.scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.resConfUnit1.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.resConfUnit1.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.resConfUnit1.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.resConfUnit1.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.resConfUnit2.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.resConfUnit2.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.resConfUnit2.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight"))
        rename_keys.append((f"core.core.scratch.refinenet{i}.resConfUnit2.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias"))

    # scratch convolutions
    for i in range(4):
        rename_keys.append((f"core.core.scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    # relative depth estimation head
    rename_keys.append(("core.core.scratch.output_conv.0.weight", "relative_head.conv1.weight"))
    rename_keys.append(("core.core.scratch.output_conv.0.bias", "relative_head.conv1.bias"))
    rename_keys.append(("core.core.scratch.output_conv.2.weight", "relative_head.conv2.weight"))
    rename_keys.append(("core.core.scratch.output_conv.2.bias", "relative_head.conv2.bias"))
    rename_keys.append(("core.core.scratch.output_conv.4.weight", "relative_head.conv3.weight"))
    rename_keys.append(("core.core.scratch.output_conv.4.bias", "relative_head.conv3.bias"))

    # metric depth estimation head
    rename_keys.append(("conv2.weight", "metric_head.conv2.weight"))
    rename_keys.append(("conv2.bias", "metric_head.conv2.bias"))

    if model_name == "ZoeD_NK":
        # patch transformer
        for i in range(4):
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.self_attn.in_proj_weight", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.self_attn.in_proj_weight"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.self_attn.in_proj_bias", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.self_attn.in_proj_bias"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.self_attn.out_proj.weight", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.self_attn.out_proj.weight"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.self_attn.out_proj.bias", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.self_attn.out_proj.bias"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.linear1.weight", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.linear1.weight"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.linear1.bias", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.linear1.bias"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.linear2.weight", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.linear2.weight"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.linear2.bias", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.linear2.bias"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.norm1.weight", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.norm1.weight"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.norm1.bias", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.norm1.bias"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.norm2.weight", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.norm2.weight"))
            rename_keys.append((f"patch_transformer.transformer_encoder.layers.{i}.norm2.bias", f"metric_head.patch_transformer.transformer_encoder.layers.{i}.norm2.bias"))

        rename_keys.append(("patch_transformer.embedding_convPxP.weight", "metric_head.patch_transformer.embedding_convPxP.weight"))
        rename_keys.append(("patch_transformer.embedding_convPxP.bias", "metric_head.patch_transformer.embedding_convPxP.bias"))

        # MLP classifier
        rename_keys.append(("mlp_classifier.0.weight", "metric_head.mlp_classifier.0.weight"))
        rename_keys.append(("mlp_classifier.0.bias", "metric_head.mlp_classifier.0.bias"))
        rename_keys.append(("mlp_classifier.2.weight", "metric_head.mlp_classifier.2.weight"))
        rename_keys.append(("mlp_classifier.2.bias", "metric_head.mlp_classifier.2.bias"))

        # NYU and KITTI seed regressors
        for i in ["nyu", "kitti"]:
            rename_keys.append((f"seed_bin_regressors.{i}._net.0.weight", f"metric_head.seed_bin_regressors.{i}.conv1.weight"))
            rename_keys.append((f"seed_bin_regressors.{i}._net.0.bias", f"metric_head.seed_bin_regressors.{i}.conv1.bias"))
            rename_keys.append((f"seed_bin_regressors.{i}._net.2.weight", f"metric_head.seed_bin_regressors.{i}.conv2.weight"))
            rename_keys.append((f"seed_bin_regressors.{i}._net.2.bias", f"metric_head.seed_bin_regressors.{i}.conv2.bias"))

        # seed projector
        rename_keys.append(("seed_projector._net.0.weight", "metric_head.seed_projector.conv1.weight"))
        rename_keys.append(("seed_projector._net.0.bias", "metric_head.seed_projector.conv1.bias"))
        rename_keys.append(("seed_projector._net.2.weight", "metric_head.seed_projector.conv2.weight"))
        rename_keys.append(("seed_projector._net.2.bias", "metric_head.seed_projector.conv2.bias"))

        # projectors
        rename_keys.append(("projectors.0._net.0.weight", "metric_head.projectors.0.conv1.weight"))
        rename_keys.append(("projectors.0._net.0.bias", "metric_head.projectors.0.conv1.bias"))
        rename_keys.append(("projectors.0._net.2.weight", "metric_head.projectors.0.conv2.weight"))
        rename_keys.append(("projectors.0._net.2.bias", "metric_head.projectors.0.conv2.bias"))
        rename_keys.append(("projectors.1._net.0.weight", "metric_head.projectors.1.conv1.weight"))
        rename_keys.append(("projectors.1._net.0.bias", "metric_head.projectors.1.conv1.bias"))
        rename_keys.append(("projectors.1._net.2.weight", "metric_head.projectors.1.conv2.weight"))
        rename_keys.append(("projectors.1._net.2.bias", "metric_head.projectors.1.conv2.bias"))
        rename_keys.append(("projectors.2._net.0.weight", "metric_head.projectors.2.conv1.weight"))
        rename_keys.append(("projectors.2._net.0.bias", "metric_head.projectors.2.conv1.bias"))
        rename_keys.append(("projectors.2._net.2.weight", "metric_head.projectors.2.conv2.weight"))
        rename_keys.append(("projectors.2._net.2.bias", "metric_head.projectors.2.conv2.bias"))
        rename_keys.append(("projectors.3._net.0.weight", "metric_head.projectors.3.conv1.weight"))
        rename_keys.append(("projectors.3._net.0.bias", "metric_head.projectors.3.conv1.bias"))
        rename_keys.append(("projectors.3._net.2.weight", "metric_head.projectors.3.conv2.weight"))
        rename_keys.append(("projectors.3._net.2.bias", "metric_head.projectors.3.conv2.bias"))

        # NYU and kitti attractors
        for i in ["nyu", "kitti"]:
            for j in range(4):
                rename_keys.append((f"attractors.{i}.{j}._net.0.weight", f"metric_head.attractors.{i}.{j}.conv1.weight"))
                rename_keys.append((f"attractors.{i}.{j}._net.0.bias", f"metric_head.attractors.{i}.{j}.conv1.bias"))
                rename_keys.append((f"attractors.{i}.{j}._net.2.weight", f"metric_head.attractors.{i}.{j}.conv2.weight"))
                rename_keys.append((f"attractors.{i}.{j}._net.2.bias", f"metric_head.attractors.{i}.{j}.conv2.bias"))


        # conditional log binomial
        for i in ["nyu", "kitti"]:
            rename_keys.append((f"conditional_log_binomial.{i}.mlp.0.weight", f"metric_head.conditional_log_binomial.{i}.mlp.0.weight"))
            rename_keys.append((f"conditional_log_binomial.{i}.mlp.0.bias", f"metric_head.conditional_log_binomial.{i}.mlp.0.bias"))
            rename_keys.append((f"conditional_log_binomial.{i}.mlp.2.weight", f"metric_head.conditional_log_binomial.{i}.mlp.2.weight"))
            rename_keys.append((f"conditional_log_binomial.{i}.mlp.2.bias", f"metric_head.conditional_log_binomial.{i}.mlp.2.bias"))
            rename_keys.append((f"conditional_log_binomial.{i}.log_binomial_transform.k_idx", f"metric_head.conditional_log_binomial.{i}.log_binomial_transform.k_idx"))
            rename_keys.append((f"conditional_log_binomial.{i}.log_binomial_transform.K_minus_1", f"metric_head.conditional_log_binomial.{i}.log_binomial_transform.K_minus_1"))

    else:
        # seed regressor and projector
        rename_keys.append(("seed_bin_regressor._net.0.weight", "metric_head.seed_bin_regressor.conv1.weight"))
        rename_keys.append(("seed_bin_regressor._net.0.bias", "metric_head.seed_bin_regressor.conv1.bias"))
        rename_keys.append(("seed_bin_regressor._net.2.weight", "metric_head.seed_bin_regressor.conv2.weight"))
        rename_keys.append(("seed_bin_regressor._net.2.bias", "metric_head.seed_bin_regressor.conv2.bias"))
        rename_keys.append(("seed_projector._net.0.weight", "metric_head.seed_projector.conv1.weight"))
        rename_keys.append(("seed_projector._net.0.bias", "metric_head.seed_projector.conv1.bias"))
        rename_keys.append(("seed_projector._net.2.weight", "metric_head.seed_projector.conv2.weight"))
        rename_keys.append(("seed_projector._net.2.bias", "metric_head.seed_projector.conv2.bias"))

        rename_keys.append(("projectors.0._net.0.weight", "metric_head.projectors.0.conv1.weight"))
        rename_keys.append(("projectors.0._net.0.bias", "metric_head.projectors.0.conv1.bias"))
        rename_keys.append(("projectors.0._net.2.weight", "metric_head.projectors.0.conv2.weight"))
        rename_keys.append(("projectors.0._net.2.bias", "metric_head.projectors.0.conv2.bias"))
        rename_keys.append(("projectors.1._net.0.weight", "metric_head.projectors.1.conv1.weight"))
        rename_keys.append(("projectors.1._net.0.bias", "metric_head.projectors.1.conv1.bias"))
        rename_keys.append(("projectors.1._net.2.weight", "metric_head.projectors.1.conv2.weight"))
        rename_keys.append(("projectors.1._net.2.bias", "metric_head.projectors.1.conv2.bias"))
        rename_keys.append(("projectors.2._net.0.weight", "metric_head.projectors.2.conv1.weight"))
        rename_keys.append(("projectors.2._net.0.bias", "metric_head.projectors.2.conv1.bias"))
        rename_keys.append(("projectors.2._net.2.weight", "metric_head.projectors.2.conv2.weight"))
        rename_keys.append(("projectors.2._net.2.bias", "metric_head.projectors.2.conv2.bias"))
        rename_keys.append(("projectors.3._net.0.weight", "metric_head.projectors.3.conv1.weight"))
        rename_keys.append(("projectors.3._net.0.bias", "metric_head.projectors.3.conv1.bias"))
        rename_keys.append(("projectors.3._net.2.weight", "metric_head.projectors.3.conv2.weight"))
        rename_keys.append(("projectors.3._net.2.bias", "metric_head.projectors.3.conv2.bias"))

        rename_keys.append(("attractors.0._net.0.weight", "metric_head.attractors.0.conv1.weight"))
        rename_keys.append(("attractors.0._net.0.bias", "metric_head.attractors.0.conv1.bias"))
        rename_keys.append(("attractors.0._net.2.weight", "metric_head.attractors.0.conv2.weight"))
        rename_keys.append(("attractors.0._net.2.bias", "metric_head.attractors.0.conv2.bias"))
        rename_keys.append(("attractors.1._net.0.weight", "metric_head.attractors.1.conv1.weight"))
        rename_keys.append(("attractors.1._net.0.bias", "metric_head.attractors.1.conv1.bias"))
        rename_keys.append(("attractors.1._net.2.weight", "metric_head.attractors.1.conv2.weight"))
        rename_keys.append(("attractors.1._net.2.bias", "metric_head.attractors.1.conv2.bias"))
        rename_keys.append(("attractors.2._net.0.weight", "metric_head.attractors.2.conv1.weight"))
        rename_keys.append(("attractors.2._net.0.bias", "metric_head.attractors.2.conv1.bias"))
        rename_keys.append(("attractors.2._net.2.weight", "metric_head.attractors.2.conv2.weight"))
        rename_keys.append(("attractors.2._net.2.bias", "metric_head.attractors.2.conv2.bias"))
        rename_keys.append(("attractors.3._net.0.weight", "metric_head.attractors.3.conv1.weight"))
        rename_keys.append(("attractors.3._net.0.bias", "metric_head.attractors.3.conv1.bias"))
        rename_keys.append(("attractors.3._net.2.weight", "metric_head.attractors.3.conv2.weight"))
        rename_keys.append(("attractors.3._net.2.bias", "metric_head.attractors.3.conv2.bias"))

        # conditional log binomial
        rename_keys.append(("conditional_log_binomial.mlp.0.weight", "metric_head.conditional_log_binomial.mlp.0.weight"))
        rename_keys.append(("conditional_log_binomial.mlp.0.bias", "metric_head.conditional_log_binomial.mlp.0.bias"))
        rename_keys.append(("conditional_log_binomial.mlp.2.weight", "metric_head.conditional_log_binomial.mlp.2.weight"))
        rename_keys.append(("conditional_log_binomial.mlp.2.bias", "metric_head.conditional_log_binomial.mlp.2.bias"))

    return rename_keys


def remove_ignore_keys_(state_dict):
    ignore_keys = ["core.core.pretrained.model.head.weight", "core.core.pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    hidden_size = config.backbone_config.hidden_size
    for i in range(config.backbone_config.num_hidden_layers):
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"core.core.pretrained.model.blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"core.core.pretrained.model.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"core.core.pretrained.model.blocks.{i}.attn.v_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = v_bias


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_zoedepth_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ZoeDepth structure.
    """

    # define ZoeDepth configuration based on URL
    config, image_size = get_zoedepth_config(model_name)

    # load original model
    original_model = torch.hub.load(
        "NielsRogge/ZoeDepth:understanding_zoedepth", model_name, pretrained=True, force_reload=True
    )
    original_model.eval()
    state_dict = original_model.state_dict()

    # for name, param in original_model.named_parameters():
    #     print(name, param.shape)

    # remove certain keys
    remove_ignore_keys_(state_dict)
    # rename keys
    rename_keys = create_rename_keys(config, model_name)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # read in qkv matrices
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = ZoeDepthForDepthEstimation(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.eval()

    # Check outputs on an image
    image = prepare_img()

    # TODO we should default to this:
    # image_processor = ZoeDepthImageProcessor()
    # and verify pixel values on this large image:
    # filepath = hf_hub_download(repo_id="shariqfarooq/ZoeDepth", filename="examples/person_1.jpeg", repo_type="space")
    # image = Image.open(filepath).convert("RGB")

    # for now we'll default to this (for some reason this passes with bicubic):
    image_processor = ZoeDepthImageProcessor(
        size={"height": image_size, "width": image_size}, keep_aspect_ratio=False, do_pad=False, resample=Image.BICUBIC
    )
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    # verify pixel values
    filepath = hf_hub_download(
        repo_id="nielsr/test-image",
        filename="zoedepth_pixel_values.pt",
        repo_type="dataset",
        revision="1865dbb81984f01c89e83eec10f8d07efd10743d",
    )
    original_pixel_values = torch.load(filepath, map_location="cpu")
    assert torch.allclose(pixel_values, original_pixel_values)

    # forward pass HF model
    depth = model(pixel_values).predicted_depth

    # Verify logits
    # These were obtained by inserting the pixel_values at the patch embeddings of BEiT
    if model_name == "ZoeD_N":
        expected_shape = torch.Size([1, 384, 384])
        expected_slice = torch.tensor([[1.0328, 1.0604, 1.0747], [1.0816, 1.1293, 1.1456], [1.1117, 1.1629, 1.1766]])
    elif model_name == "ZoeD_K":
        expected_shape = torch.Size([1, 384, 384])
        expected_slice = torch.tensor([[1.6567, 1.6852, 1.7065], [1.6707, 1.6764, 1.6713], [1.7195, 1.7166, 1.7118]])
    elif model_name == "ZoeD_NK":
        expected_shape = torch.Size([1, 384, 384])
        expected_slice = torch.tensor([[1.1228, 1.1079, 1.1382], [1.1807, 1.1658, 1.1891], [1.2344, 1.2094, 1.2317]])

    print("Shape of depth:", depth.shape)
    print("First 3x3 slice of depth:", depth[0, :3, :3])

    assert depth.shape == torch.Size(expected_shape)
    assert torch.allclose(depth[0, :3, :3], expected_slice, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_name_to_repo_id = {
            "ZoeD_N": "zoedepth-nyu",
            "ZoeD_K": "zoedepth-kitti",
            "ZoeD_NK": "zoedepth-nyu-kitti",
        }

        print("Pushing model and processor to the hub...")
        repo_id = model_name_to_repo_id[model_name]
        model.push_to_hub(f"nielsr/{repo_id}")
        image_processor.push_to_hub(f"nielsr/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="ZoeD_N",
        choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK"],
        type=str,
        help="Name of the original ZoeDepth checkpoint you'd like to convert.",
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

    args = parser.parse_args()
    convert_zoedepth_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
