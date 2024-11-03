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
"""Convert VitPose checkpoints from the original repository.

URL: https://github.com/vitae-transformer/vitpose
"""

import argparse
import os
import re

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation, VitPoseImageProcessor


KEYS_TO_MODIFY_MAPPING = {
    r"patch_embed.proj": "embeddings.patch_embeddings.projection",
    r"pos_embed": "embeddings.position_embeddings",
    r"blocks": "encoder.layer",
    r"attn.proj": "attention.output.dense",
    r"attn": "attention.self",
    r"norm1": "layernorm_before",
    r"norm2": "layernorm_after",
    r"last_norm": "layernorm",
}

MODEL_TO_FILE_NAME_MAPPING = {
    "vitpose-base-simple": "vitpose-b-simple.pth",
    "vitpose-base": "vitpose-b.pth",
    "vitpose-base-coco-aic-mpii": "vitpose_base_coco_aic_mpii.pth",
    "vitpose+-base": "vitpose+_base.pth",
}


def get_config(model_name):
    num_experts = 6 if "+" in model_name else 1
    part_features = 192 if "+" in model_name else 0

    backbone_config = VitPoseBackboneConfig(out_indices=[12], num_experts=num_experts, part_features=part_features)
    # size of the architecture
    if "small" in model_name:
        backbone_config.hidden_size = 768
        backbone_config.intermediate_size = 2304
        backbone_config.num_hidden_layers = 8
        backbone_config.num_attention_heads = 8
    elif "large" in model_name:
        backbone_config.hidden_size = 1024
        backbone_config.intermediate_size = 4096
        backbone_config.num_hidden_layers = 24
        backbone_config.num_attention_heads = 16
    elif "huge" in model_name:
        backbone_config.hidden_size = 1280
        backbone_config.intermediate_size = 5120
        backbone_config.num_hidden_layers = 32
        backbone_config.num_attention_heads = 16

    use_simple_decoder = "simple" in model_name

    keypoint_edges = (
        [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],
            [5, 11],
            [6, 12],
            [5, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
        ],
    )
    keypoint_labels = (
        [
            "Nose",
            "L_Eye",
            "R_Eye",
            "L_Ear",
            "R_Ear",
            "L_Shoulder",
            "R_Shoulder",
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",
            "L_Hip",
            "R_Hip",
            "L_Knee",
            "R_Knee",
            "L_Ankle",
            "R_Ankle",
        ],
    )

    config = VitPoseConfig(
        backbone_config=backbone_config,
        num_labels=17,
        use_simple_decoder=use_simple_decoder,
        keypoint_edges=keypoint_edges,
        keypoint_labels=keypoint_labels,
    )

    return config


def convert_old_keys_to_new_keys(state_dict, config):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    model_state_dict = {}

    output_hypernetworks_qkv_pattern = r".*.qkv.*"
    output_hypernetworks_head_pattern = r"keypoint_head.*"

    dim = config.backbone_config.hidden_size

    for key in state_dict.copy().keys():
        value = state_dict.pop(key)
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(output_hypernetworks_qkv_pattern, key):
            layer_num = int(key.split(".")[3])
            if "weight" in key:
                model_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.query.weight"] = value[
                    :dim, :
                ]
                model_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.key.weight"] = value[
                    dim : dim * 2, :
                ]
                model_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.value.weight"] = value[
                    -dim:, :
                ]
            else:
                model_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.query.bias"] = value[:dim]
                model_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.key.bias"] = value[
                    dim : dim * 2
                ]
                model_state_dict[f"backbone.encoder.layer.{layer_num}.attention.attention.value.bias"] = value[-dim:]

        if re.match(output_hypernetworks_head_pattern, key):
            if config.use_simple_decoder:
                key = key.replace("keypoint_head.final_layer", "head.conv")
            else:
                key = key.replace("keypoint_head", "head")
                key = key.replace("deconv_layers.0.weight", "deconv1.weight")
                key = key.replace("deconv_layers.1.weight", "batchnorm1.weight")
                key = key.replace("deconv_layers.1.bias", "batchnorm1.bias")
                key = key.replace("deconv_layers.1.running_mean", "batchnorm1.running_mean")
                key = key.replace("deconv_layers.1.running_var", "batchnorm1.running_var")
                key = key.replace("deconv_layers.1.num_batches_tracked", "batchnorm1.num_batches_tracked")
                key = key.replace("deconv_layers.3.weight", "deconv2.weight")
                key = key.replace("deconv_layers.4.weight", "batchnorm2.weight")
                key = key.replace("deconv_layers.4.bias", "batchnorm2.bias")
                key = key.replace("deconv_layers.4.running_mean", "batchnorm2.running_mean")
                key = key.replace("deconv_layers.4.running_var", "batchnorm2.running_var")
                key = key.replace("deconv_layers.4.num_batches_tracked", "batchnorm2.num_batches_tracked")
                key = key.replace("final_layer.weight", "conv.weight")
                key = key.replace("final_layer.bias", "conv.bias")
        model_state_dict[key] = value

    return model_state_dict


# We will verify our results on a COCO image
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000000139.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@torch.no_grad()
def write_model(model_path, model_name, push_to_hub):
    os.makedirs(model_path, exist_ok=True)

    # ------------------------------------------------------------
    # Vision model params and config
    # ------------------------------------------------------------

    # params from config
    config = get_config(model_name)

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    # load original state_dict
    filename = MODEL_TO_FILE_NAME_MAPPING[model_name]
    print(f"Fetching all parameters from the checkpoint at {filename}...")

    checkpoint_path = hf_hub_download(
        repo_id="nielsr/vitpose-original-checkpoints", filename=filename, repo_type="model"
    )

    print("Converting model...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    new_state_dict = convert_old_keys_to_new_keys(state_dict, config)

    print("Loading the checkpoint in a Vitpose model.")
    model = VitPoseForPoseEstimation(config)
    model.eval()
    model.load_state_dict(new_state_dict, strict=False)
    print("Checkpoint loaded successfully.")

    # create image processor
    image_processor = VitPoseImageProcessor()

    # verify image processor
    image = prepare_img()
    boxes = [[[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]]]
    pixel_values = image_processor(images=image, boxes=boxes, return_tensors="pt").pixel_values

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="vitpose_batch_data.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath, map_location="cpu")["img"]
    assert torch.allclose(pixel_values, original_pixel_values, atol=1e-1)

    dataset_index = torch.tensor([0])

    with torch.no_grad():
        # first forward pass
        outputs = model(pixel_values, dataset_index=dataset_index)
        output_heatmap = outputs.heatmaps

        # second forward pass (flipped)
        # this is done since the model uses `flip_test=True` in its test config
        pixel_values_flipped = torch.flip(pixel_values, [3])
        outputs_flipped = model(
            pixel_values_flipped,
            dataset_index=dataset_index,
            flip_pairs=torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]),
        )
        output_flipped_heatmap = outputs_flipped.heatmaps

    outputs.heatmaps = (output_heatmap + output_flipped_heatmap) * 0.5

    # Verify pose_results
    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)[0]

    if model_name == "vitpose-base-simple":
        assert torch.allclose(
            pose_results[1]["keypoints"][0, :3],
            torch.tensor([3.98180511e02, 1.81808380e02, 8.66642594e-01]),
            atol=5e-2,
        )
    elif model_name == "vitpose-base":
        assert torch.allclose(
            pose_results[1]["keypoints"][0, :3],
            torch.tensor([3.9807913e02, 1.8182812e02, 8.8235235e-01]),
            atol=5e-2,
        )
    elif model_name == "vitpose-base-coco-aic-mpii":
        assert torch.allclose(
            pose_results[1]["keypoints"][0, :3],
            torch.tensor([3.98305542e02, 1.81741592e02, 8.69966745e-01]),
            atol=5e-2,
        )
    elif model_name == "vitpose+-base":
        assert torch.allclose(
            pose_results[1]["keypoints"][0, :3],
            torch.tensor([3.98201294e02, 1.81728302e02, 8.75046968e-01]),
            atol=5e-2,
        )
    else:
        raise ValueError("Model not supported")
    print("Conversion successfully done.")

    if push_to_hub:
        print(f"Pushing model and image processor for {model_name} to hub")
        model.push_to_hub(f"nielsr/{model_name}")
        image_processor.push_to_hub(f"nielsr/{model_name}")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vitpose-base-simple",
        choices=MODEL_TO_FILE_NAME_MAPPING.keys(),
        type=str,
        help="Name of the VitPose model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    write_model(model_path=args.pytorch_dump_folder_path, model_name=args.model_name, push_to_hub=args.push_to_hub)


if __name__ == "__main__":
    main()
