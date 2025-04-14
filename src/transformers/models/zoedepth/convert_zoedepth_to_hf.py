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
    if model_name == "ZoeD_NK":
        bin_configurations = [
            {"name": "nyu", "n_bins": 64, "min_depth": 1e-3, "max_depth": 10.0},
            {"name": "kitti", "n_bins": 64, "min_depth": 1e-3, "max_depth": 80.0},
        ]
    elif model_name in ["ZoeD_N", "ZoeD_K"]:
        bin_configurations = [
            {"name": "nyu", "n_bins": 64, "min_depth": 1e-3, "max_depth": 10.0},
        ]
    config = ZoeDepthConfig(
        backbone_config=backbone_config,
        neck_hidden_sizes=neck_hidden_sizes,
        bin_centers_type=bin_centers_type,
        bin_configurations=bin_configurations,
        num_patch_transformer_layers=4 if model_name == "ZoeD_NK" else None,
        patch_transformer_hidden_size=128 if model_name == "ZoeD_NK" else None,
        patch_transformer_intermediate_size=1024 if model_name == "ZoeD_NK" else None,
        patch_transformer_num_attention_heads=4 if model_name == "ZoeD_NK" else None,
    )

    return config, image_size


def rename_key(name):
    # Transformer backbone
    if "core.core.pretrained.model.blocks" in name:
        name = name.replace("core.core.pretrained.model.blocks", "backbone.encoder.layer")
    if "core.core.pretrained.model.patch_embed.proj" in name:
        name = name.replace(
            "core.core.pretrained.model.patch_embed.proj", "backbone.embeddings.patch_embeddings.projection"
        )
    if "core.core.pretrained.model.cls_token" in name:
        name = name.replace("core.core.pretrained.model.cls_token", "backbone.embeddings.cls_token")
    if "norm1" in name and "patch_transformer" not in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name and "patch_transformer" not in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    if "gamma_1" in name:
        name = name.replace("gamma_1", "lambda_1")
    if "gamma_2" in name:
        name = name.replace("gamma_2", "lambda_2")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn.relative_position_bias_table" in name:
        name = name.replace(
            "attn.relative_position_bias_table",
            "attention.attention.relative_position_bias.relative_position_bias_table",
        )
    if "attn.relative_position_index" in name:
        name = name.replace(
            "attn.relative_position_index", "attention.attention.relative_position_bias.relative_position_index"
        )

    # activation postprocessing (readout projections + resize blocks)
    if "core.core.pretrained.act_postprocess1.0.project" in name:
        name = name.replace(
            "core.core.pretrained.act_postprocess1.0.project", "neck.reassemble_stage.readout_projects.0"
        )
    if "core.core.pretrained.act_postprocess2.0.project" in name:
        name = name.replace(
            "core.core.pretrained.act_postprocess2.0.project", "neck.reassemble_stage.readout_projects.1"
        )
    if "core.core.pretrained.act_postprocess3.0.project" in name:
        name = name.replace(
            "core.core.pretrained.act_postprocess3.0.project", "neck.reassemble_stage.readout_projects.2"
        )
    if "core.core.pretrained.act_postprocess4.0.project" in name:
        name = name.replace(
            "core.core.pretrained.act_postprocess4.0.project", "neck.reassemble_stage.readout_projects.3"
        )

    if "core.core.pretrained.act_postprocess1.3" in name:
        name = name.replace("core.core.pretrained.act_postprocess1.3", "neck.reassemble_stage.layers.0.projection")
    if "core.core.pretrained.act_postprocess2.3" in name:
        name = name.replace("core.core.pretrained.act_postprocess2.3", "neck.reassemble_stage.layers.1.projection")
    if "core.core.pretrained.act_postprocess3.3" in name:
        name = name.replace("core.core.pretrained.act_postprocess3.3", "neck.reassemble_stage.layers.2.projection")
    if "core.core.pretrained.act_postprocess4.3" in name:
        name = name.replace("core.core.pretrained.act_postprocess4.3", "neck.reassemble_stage.layers.3.projection")

    if "core.core.pretrained.act_postprocess1.4" in name:
        name = name.replace("core.core.pretrained.act_postprocess1.4", "neck.reassemble_stage.layers.0.resize")
    if "core.core.pretrained.act_postprocess2.4" in name:
        name = name.replace("core.core.pretrained.act_postprocess2.4", "neck.reassemble_stage.layers.1.resize")
    if "core.core.pretrained.act_postprocess4.4" in name:
        name = name.replace("core.core.pretrained.act_postprocess4.4", "neck.reassemble_stage.layers.3.resize")

    # scratch convolutions
    if "core.core.scratch.layer1_rn.weight" in name:
        name = name.replace("core.core.scratch.layer1_rn.weight", "neck.convs.0.weight")
    if "core.core.scratch.layer2_rn.weight" in name:
        name = name.replace("core.core.scratch.layer2_rn.weight", "neck.convs.1.weight")
    if "core.core.scratch.layer3_rn.weight" in name:
        name = name.replace("core.core.scratch.layer3_rn.weight", "neck.convs.2.weight")
    if "core.core.scratch.layer4_rn.weight" in name:
        name = name.replace("core.core.scratch.layer4_rn.weight", "neck.convs.3.weight")

    # fusion layers
    # tricky here: mapping = {1:3, 2:2, 3:1, 4:0}
    if "core.core.scratch.refinenet1" in name:
        name = name.replace("core.core.scratch.refinenet1", "neck.fusion_stage.layers.3")
    if "core.core.scratch.refinenet2" in name:
        name = name.replace("core.core.scratch.refinenet2", "neck.fusion_stage.layers.2")
    if "core.core.scratch.refinenet3" in name:
        name = name.replace("core.core.scratch.refinenet3", "neck.fusion_stage.layers.1")
    if "core.core.scratch.refinenet4" in name:
        name = name.replace("core.core.scratch.refinenet4", "neck.fusion_stage.layers.0")

    if "resConfUnit1" in name:
        name = name.replace("resConfUnit1", "residual_layer1")

    if "resConfUnit2" in name:
        name = name.replace("resConfUnit2", "residual_layer2")

    if "conv1" in name:
        name = name.replace("conv1", "convolution1")

    if "conv2" in name and "residual_layer" in name:
        name = name.replace("conv2", "convolution2")

    if "out_conv" in name:
        name = name.replace("out_conv", "projection")

    # relative depth estimation head
    if "core.core.scratch.output_conv.0" in name:
        name = name.replace("core.core.scratch.output_conv.0", "relative_head.conv1")

    if "core.core.scratch.output_conv.2" in name:
        name = name.replace("core.core.scratch.output_conv.2", "relative_head.conv2")

    if "core.core.scratch.output_conv.4" in name:
        name = name.replace("core.core.scratch.output_conv.4", "relative_head.conv3")

    # patch transformer
    if "patch_transformer" in name:
        name = name.replace("patch_transformer", "metric_head.patch_transformer")

    if "mlp_classifier.0" in name:
        name = name.replace("mlp_classifier.0", "metric_head.mlp_classifier.linear1")
    if "mlp_classifier.2" in name:
        name = name.replace("mlp_classifier.2", "metric_head.mlp_classifier.linear2")

    if "projectors" in name:
        name = name.replace("projectors", "metric_head.projectors")

    if "seed_bin_regressors" in name:
        name = name.replace("seed_bin_regressors", "metric_head.seed_bin_regressors")

    if "seed_bin_regressor" in name and "seed_bin_regressors" not in name:
        name = name.replace("seed_bin_regressor", "metric_head.seed_bin_regressor")

    if "seed_projector" in name:
        name = name.replace("seed_projector", "metric_head.seed_projector")

    if "_net.0" in name:
        name = name.replace("_net.0", "conv1")

    if "_net.2" in name:
        name = name.replace("_net.2", "conv2")

    if "attractors" in name:
        name = name.replace("attractors", "metric_head.attractors")

    if "conditional_log_binomial" in name:
        name = name.replace("conditional_log_binomial", "metric_head.conditional_log_binomial")

    # metric depth estimation head
    if "conv2" in name and "metric_head" not in name and "attractors" not in name and "relative_head" not in name:
        name = name.replace("conv2", "metric_head.conv2")

    if "transformer_encoder.layers" in name:
        name = name.replace("transformer_encoder.layers", "transformer_encoder")

    return name


def read_in_q_k_v_metric_head(state_dict):
    hidden_size = 128
    for i in range(4):
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"patch_transformer.transformer_encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"patch_transformer.transformer_encoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"patch_transformer.transformer_encoder.{i}.self_attn.query.weight"] = in_proj_weight[
            :hidden_size, :
        ]
        state_dict[f"patch_transformer.transformer_encoder.{i}.self_attn.query.bias"] = in_proj_bias[:hidden_size]

        state_dict[f"patch_transformer.transformer_encoder.{i}.self_attn.key.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"patch_transformer.transformer_encoder.{i}.self_attn.key.bias"] = in_proj_bias[
            hidden_size : hidden_size * 2
        ]

        state_dict[f"patch_transformer.transformer_encoder.{i}.self_attn.value.weight"] = in_proj_weight[
            -hidden_size:, :
        ]
        state_dict[f"patch_transformer.transformer_encoder.{i}.self_attn.value.bias"] = in_proj_bias[-hidden_size:]


def convert_state_dict(orig_state_dict):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        # rename key
        new_name = rename_key(key)
        orig_state_dict[new_name] = val

    return orig_state_dict


def remove_ignore_keys(state_dict):
    for key, _ in state_dict.copy().items():
        if (
            "fc_norm" in key
            or "relative_position_index" in key
            or "k_idx" in key
            or "K_minus_1" in key
            or "core.core.pretrained.model.head" in key
        ):
            state_dict.pop(key, None)


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


# We will verify our results on an image
def prepare_img():
    filepath = hf_hub_download(repo_id="shariqfarooq/ZoeDepth", filename="examples/person_1.jpeg", repo_type="space")
    image = Image.open(filepath).convert("RGB")
    return image


@torch.no_grad()
def convert_zoedepth_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ZoeDepth structure.
    """

    # define ZoeDepth configuration based on URL
    config, _ = get_zoedepth_config(model_name)

    # load original model
    original_model = torch.hub.load(
        "NielsRogge/ZoeDepth:understanding_zoedepth", model_name, pretrained=True, force_reload=True
    )
    original_model.eval()
    state_dict = original_model.state_dict()

    print("Original state dict:")
    for name, param in state_dict.items():
        print(name, param.shape)

    # read in qkv matrices
    read_in_q_k_v(state_dict, config)
    if model_name == "ZoeD_NK":
        read_in_q_k_v_metric_head(state_dict)

    # rename keys
    state_dict = convert_state_dict(state_dict)
    # remove certain keys
    remove_ignore_keys(state_dict)

    # load HuggingFace model
    model = ZoeDepthForDepthEstimation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # verify image processor
    image = prepare_img()

    image_processor = ZoeDepthImageProcessor()
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    filepath = hf_hub_download(
        repo_id="nielsr/test-image",
        filename="zoedepth_pixel_values.pt",
        repo_type="dataset",
    )
    original_pixel_values = torch.load(filepath, map_location="cpu", weights_only=True)
    assert torch.allclose(pixel_values, original_pixel_values)

    # verify logits
    # this was done on a resized version of the cats image (384x384)
    filepath = hf_hub_download(
        repo_id="nielsr/test-image",
        filename="zoedepth_pixel_values.pt",
        repo_type="dataset",
        revision="1865dbb81984f01c89e83eec10f8d07efd10743d",
    )
    cats_pixel_values = torch.load(filepath, map_location="cpu", weights_only=True)
    depth = model(cats_pixel_values).predicted_depth

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
        model.push_to_hub(f"Intel/{repo_id}")
        image_processor = ZoeDepthImageProcessor()
        image_processor.push_to_hub(f"Intel/{repo_id}")


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
