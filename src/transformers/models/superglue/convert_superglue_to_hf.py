# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import gc
import os
import re
from typing import List

import torch
from datasets import load_dataset
from torch import nn

from transformers import (
    AutoModelForKeypointDetection,
    SuperGlueConfig,
    SuperGlueForKeypointMatching,
    SuperGlueImageProcessor,
)


def prepare_imgs_for_image_processor():
    dataset = load_dataset("stevenbucaille/image_matching_fixtures", split="train")
    return [[dataset[0]["image"], dataset[1]["image"]], [dataset[2]["image"], dataset[1]["image"]]]


def verify_model_outputs(model, model_name):
    from tests.models.superglue.test_modeling_superglue import prepare_imgs

    images = prepare_imgs()
    preprocessor = SuperGlueImageProcessor()
    inputs = preprocessor(images=images, return_tensors="pt").to("cuda")
    model.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    predicted_matches_values = outputs.matches[0, 0, :10]
    predicted_matching_scores_values = outputs.matching_scores[0, 0, :10]

    predicted_number_of_matches = torch.sum(outputs.matches[0][0] != -1).item()

    if "outdoor" in model_name:
        expected_max_number_keypoints = 866
        expected_matches_shape = torch.Size((len(images), 2, expected_max_number_keypoints))
        expected_matching_scores_shape = torch.Size((len(images), 2, expected_max_number_keypoints))

        expected_matches_values = torch.tensor(
            [125, -1, 137, 138, 19, -1, 135, -1, 160, 153], dtype=torch.int64, device=predicted_matches_values.device
        )
        expected_matching_scores_values = torch.tensor(
            [0.2406, 0, 0.8879, 0.7491, 0.3161, 0, 0.6232, 0, 0.2723, 0.9559],
            device=predicted_matches_values.device,
        )

        expected_number_of_matches = 162
    elif "indoor" in model_name:
        expected_max_number_keypoints = 866
        expected_matches_shape = torch.Size((len(images), 2, expected_max_number_keypoints))
        expected_matching_scores_shape = torch.Size((len(images), 2, expected_max_number_keypoints))

        expected_matches_values = torch.tensor(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int64, device=predicted_matches_values.device
        )
        expected_matching_scores_values = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            device=predicted_matches_values.device,
        )

        expected_number_of_matches = 0

    assert outputs.matches.shape == expected_matches_shape
    assert outputs.matching_scores.shape == expected_matching_scores_shape

    assert torch.allclose(predicted_matches_values, expected_matches_values, atol=1e-4)
    assert torch.allclose(predicted_matching_scores_values, expected_matching_scores_values, atol=1e-4)

    assert predicted_number_of_matches == expected_number_of_matches


def conv1d_to_linear(conv1d_layer):
    conv1d_weights = conv1d_layer.weight.data
    linear_weights = conv1d_weights.squeeze(-1)

    linear_layer = nn.Linear(conv1d_layer.in_channels, conv1d_layer.out_channels, bias=conv1d_layer.bias is not None)
    linear_layer.weight.data = linear_weights

    if conv1d_layer.bias is not None:
        linear_layer.bias.data = conv1d_layer.bias.data

    return linear_layer


class SuperGlueMultiLayerPerceptronConversionModel(nn.Module):
    def __init__(self, config: SuperGlueConfig, channels: List[int]) -> None:
        super().__init__()
        num_layers = len(channels)
        layers = []
        for i in range(1, num_layers):
            layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1))
            if i < (num_layers - 1):
                layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def convert(self):
        new_layers = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                layer = conv1d_to_linear(layer)
            new_layers.append(layer)
        self.layers = nn.ModuleList(new_layers)


class SuperGlueKeypointEncoderConversionModel(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        layer_sizes = config.keypoint_encoder_sizes
        feature_dim = config.descriptor_dim
        encoder_channels = [3] + layer_sizes + [feature_dim]
        self.encoder = SuperGlueMultiLayerPerceptronConversionModel(config, channels=encoder_channels)

    def convert(self):
        self.encoder.convert()


class SuperGlueMultiHeadAttentionConversionModel(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        feature_dim = config.descriptor_dim
        self.num_heads = config.num_heads
        self.head_dim = feature_dim // self.num_heads

        self.q_proj = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.k_proj = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

    def convert(self):
        self.q_proj = conv1d_to_linear(self.q_proj)
        self.k_proj = conv1d_to_linear(self.k_proj)
        self.v_proj = conv1d_to_linear(self.v_proj)
        self.out_proj = conv1d_to_linear(self.out_proj)


class SuperGlueAttentionalPropagationConversionModel(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        descriptor_dim = config.descriptor_dim
        self.attention = SuperGlueMultiHeadAttentionConversionModel(config)
        mlp_channels = [descriptor_dim * 2, descriptor_dim * 2, descriptor_dim]
        self.mlp = SuperGlueMultiLayerPerceptronConversionModel(config, channels=mlp_channels)

    def convert(self):
        self.attention.convert()
        self.mlp.convert()


class SuperGlueAttentionalGNNConversionModel(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        self.descriptor_dim = config.descriptor_dim
        self.layers_types = config.gnn_layers_types
        self.layers = nn.ModuleList(
            [SuperGlueAttentionalPropagationConversionModel(config) for _ in range(len(self.layers_types))]
        )

    def convert(self):
        [layer.convert() for layer in self.layers]


class SuperGlueFinalProjectionConversionModel(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        descriptor_dim = config.descriptor_dim
        self.final_proj = nn.Conv1d(descriptor_dim, descriptor_dim, kernel_size=1, bias=True)

    def convert(self):
        self.final_proj = conv1d_to_linear(self.final_proj)


class SuperGlueConversionModel(nn.Module):
    def __init__(self, config: SuperGlueConfig):
        super().__init__()
        self.keypoint_encoder = SuperGlueKeypointEncoderConversionModel(config)
        self.gnn = SuperGlueAttentionalGNNConversionModel(config)
        self.final_projection = SuperGlueFinalProjectionConversionModel(config)
        self.register_parameter("bin_score", nn.Parameter(torch.tensor(1.0)))

    def convert(self):
        self.keypoint_encoder.convert()
        self.gnn.convert()
        self.final_projection.convert()


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"kenc.encoder.(\d+).weight": r"keypoint_encoder.encoder.layers.\1.weight",
    r"kenc.encoder.(\d+).bias": r"keypoint_encoder.encoder.layers.\1.bias",
    r"kenc.encoder.(\d+).running_mean": r"keypoint_encoder.encoder.layers.\1.running_mean",
    r"kenc.encoder.(\d+).running_var": r"keypoint_encoder.encoder.layers.\1.running_var",
    r"kenc.encoder.(\d+).num_batches_tracked": r"keypoint_encoder.encoder.layers.\1.num_batches_tracked",
    r"gnn.layers.(\d+).attn.proj.0.weight": r"gnn.layers.\1.attention.q_proj.weight",
    r"gnn.layers.(\d+).attn.proj.0.bias": r"gnn.layers.\1.attention.q_proj.bias",
    r"gnn.layers.(\d+).attn.proj.1.weight": r"gnn.layers.\1.attention.k_proj.weight",
    r"gnn.layers.(\d+).attn.proj.1.bias": r"gnn.layers.\1.attention.k_proj.bias",
    r"gnn.layers.(\d+).attn.proj.2.weight": r"gnn.layers.\1.attention.v_proj.weight",
    r"gnn.layers.(\d+).attn.proj.2.bias": r"gnn.layers.\1.attention.v_proj.bias",
    r"gnn.layers.(\d+).attn.merge.weight": r"gnn.layers.\1.attention.out_proj.weight",
    r"gnn.layers.(\d+).attn.merge.bias": r"gnn.layers.\1.attention.out_proj.bias",
    r"gnn.layers.(\d+).mlp.(\d+).weight": r"gnn.layers.\1.mlp.layers.\2.weight",
    r"gnn.layers.(\d+).mlp.(\d+).bias": r"gnn.layers.\1.mlp.layers.\2.bias",
    r"gnn.layers.(\d+).mlp.(\d+).running_mean": r"gnn.layers.\1.mlp.layers.\2.running_mean",
    r"gnn.layers.(\d+).mlp.(\d+).running_var": r"gnn.layers.\1.mlp.layers.\2.running_var",
    r"gnn.layers.(\d+).mlp.(\d+).num_batches_tracked": r"gnn.layers.\1.mlp.layers.\2.num_batches_tracked",
    r"final_proj.weight": r"final_projection.final_proj.weight",
    r"final_proj.bias": r"final_projection.final_proj.bias",
}


def convert_old_keys_to_new_keys(state_dict_keys: List[str]):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def add_keypoint_detector_state_dict(superglue_state_dict):
    keypoint_detector = AutoModelForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    keypoint_detector_state_dict = keypoint_detector.state_dict()
    for k, v in keypoint_detector_state_dict.items():
        superglue_state_dict[f"keypoint_detector.{k}"] = v
    return superglue_state_dict


@torch.no_grad()
def write_model(
    model_path,
    checkpoint_url,
    safe_serialization=True,
    push_to_hub=False,
):
    os.makedirs(model_path, exist_ok=True)

    # ------------------------------------------------------------
    # SuperGlue config
    # ------------------------------------------------------------

    config = SuperGlueConfig(
        descriptor_dim=256,
        keypoint_encoder_sizes=[32, 64, 128, 256],
        gnn_layers_types=["self", "cross"] * 9,
        sinkhorn_iterations=100,
        matching_threshold=0.2,
    )
    config.architectures = ["SuperGlueForKeypointMatching"]
    config.save_pretrained(model_path, push_to_hub=push_to_hub)
    print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    print(f"Fetching all parameters from the checkpoint at {checkpoint_url}...")
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)

    print("Converting model...")
    all_keys = list(original_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        state_dict[new_key] = original_state_dict.pop(key).contiguous().clone()

    del original_state_dict
    gc.collect()

    conversion_model = SuperGlueConversionModel(config)
    conversion_model.load_state_dict(state_dict, strict=True, assign=True)
    conversion_model.convert()
    state_dict = conversion_model.state_dict()

    state_dict = add_keypoint_detector_state_dict(state_dict)

    print("Loading the checkpoint in a SuperGlue model...")
    with torch.device("cuda"):
        model = SuperGlueForKeypointMatching(config)
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully...")
    del model.config._name_or_path

    print("Saving the model...")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    model = SuperGlueForKeypointMatching.from_pretrained(model_path)
    print("Model reloaded successfully.")

    model_name = "superglue"
    if "superglue_outdoor.pth" in checkpoint_url:
        model_name += "_outdoor"
    elif "superglue_indoor.pth" in checkpoint_url:
        model_name += "_indoor"

    print("Checking the model outputs...")
    verify_model_outputs(model, model_name)
    print("Model outputs verified successfully.")

    organization = "stevenbucaille"
    if push_to_hub:
        print("Pushing model to the hub...")
        model.push_to_hub(
            repo_id=f"{organization}/{model_name}",
            commit_message="Add model",
        )

    write_image_processor(model_path, model_name, organization, push_to_hub=push_to_hub)


def write_image_processor(save_dir, model_name, organization, push_to_hub=False):
    image_processor = SuperGlueImageProcessor()
    image_processor.save_pretrained(save_dir)

    if push_to_hub:
        print("Pushing image processor to the hub...")
        image_processor.push_to_hub(
            repo_id=f"{organization}/{model_name}",
            commit_message="Add image processor",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_indoor.pth",
        type=str,
        help="URL of the original SuperGlue checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model and image preprocessor to the hub",
    )

    args = parser.parse_args()
    write_model(
        args.pytorch_dump_folder_path, args.checkpoint_url, safe_serialization=True, push_to_hub=args.push_to_hub
    )
    # convert_superglue_checkpoint(
    #     args.checkpoint_url,
    #     args.pytorch_dump_folder_path,
    #     args.save_model,
    #     args.push_to_hub,
    # )
