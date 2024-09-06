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
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForKeypointDetection,
    SuperGlueConfig,
    SuperGlueForKeypointMatching,
    SuperGlueImageProcessor,
)
from transformers.models.superpoint.modeling_superpoint import SuperPointKeypointDescriptionOutput


def get_superglue_config():
    config = SuperGlueConfig(
        descriptor_dim=256,
        keypoint_encoder_sizes=[32, 64, 128, 256],
        gnn_layers_types=["self", "cross"] * 9,
        sinkhorn_iterations=100,
        matching_threshold=0.2,
    )

    return config


def conv1d_to_linear(conv1d_layer):
    conv1d_weights = conv1d_layer.weight.data
    linear_weights = conv1d_weights.squeeze(-1)

    linear_layer = nn.Linear(conv1d_layer.in_channels, conv1d_layer.out_channels, bias=conv1d_layer.bias is not None)
    linear_layer.weight.data = linear_weights

    if conv1d_layer.bias is not None:
        linear_layer.bias.data = conv1d_layer.bias.data

    return linear_layer


def fuse_batchnorm_into_linear(linear_layer, bn_layer):
    # Convert BatchNorm1d weights and biases
    bn_weight = bn_layer.weight.data
    bn_bias = bn_layer.bias.data
    bn_running_mean = bn_layer.running_mean.data
    bn_running_var = bn_layer.running_var.data

    # Fuse BatchNorm into Linear layer
    scale = bn_weight / torch.sqrt(bn_running_var + 1e-8)
    bias = bn_bias - bn_running_mean * scale

    linear_layer.weight.data = linear_layer.weight.data * scale.unsqueeze(1)
    if linear_layer.bias is not None:
        linear_layer.bias.data += bias
    else:
        linear_layer.bias = bias


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
            if isinstance(layer, nn.ReLU):
                new_layers.append(layer)
            if isinstance(layer, nn.BatchNorm1d):
                last_new_layer = new_layers[-1]
                # Convert BatchNorm1d weights and biases
                fuse_batchnorm_into_linear(last_new_layer, layer)
        self.layers = nn.Sequential(*new_layers)


class SuperGlueKeypointEncoderConversionModel(nn.Module):
    def __init__(self, config: SuperGlueConfig) -> None:
        super().__init__()
        layer_sizes = config.keypoint_encoder_sizes
        feature_dim = config.descriptor_dim
        # 3 here consists of 2 for the (x, y) coordinates and 1 for the score of the keypoint
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

        self.query = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.key = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.value = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)

    def convert(self):
        self.query = conv1d_to_linear(self.query)
        self.key = conv1d_to_linear(self.key)
        self.value = conv1d_to_linear(self.value)
        self.merge = conv1d_to_linear(self.merge)


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


def create_rename_keys(config):
    rename_keys = []

    # keypoint encoder
    n = len([3] + config.keypoint_encoder_sizes + [config.descriptor_dim])
    for i in range(n * 2 + 1):
        if ((i + 1) % 3) != 0:
            rename_keys.append(
                (
                    f"kenc.encoder.{i}.weight",
                    f"keypoint_encoder.encoder.layers.{i}.weight",
                )
            )
            rename_keys.append((f"kenc.encoder.{i}.bias", f"keypoint_encoder.encoder.layers.{i}.bias"))
            if ((i % 3) - 1) == 0:
                rename_keys.append(
                    (
                        f"kenc.encoder.{i}.running_mean",
                        f"keypoint_encoder.encoder.layers.{i}.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"kenc.encoder.{i}.running_var",
                        f"keypoint_encoder.encoder.layers.{i}.running_var",
                    )
                )
                rename_keys.append(
                    (
                        f"kenc.encoder.{i}.num_batches_tracked",
                        f"keypoint_encoder.encoder.layers.{i}.num_batches_tracked",
                    )
                )

    # gnn
    for i in range(len(config.gnn_layers_types)):
        rename_keys.append(
            (
                f"gnn.layers.{i}.attn.merge.weight",
                f"gnn.layers.{i}.attention.merge.weight",
            )
        )
        rename_keys.append((f"gnn.layers.{i}.attn.merge.bias", f"gnn.layers.{i}.attention.merge.bias"))
        for j in range(3):
            if j == 0:
                rename_keys.append(
                    (
                        f"gnn.layers.{i}.attn.proj.{j}.weight",
                        f"gnn.layers.{i}.attention.query.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"gnn.layers.{i}.attn.proj.{j}.bias",
                        f"gnn.layers.{i}.attention.query.bias",
                    )
                )
            elif j == 1:
                rename_keys.append(
                    (
                        f"gnn.layers.{i}.attn.proj.{j}.weight",
                        f"gnn.layers.{i}.attention.key.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"gnn.layers.{i}.attn.proj.{j}.bias",
                        f"gnn.layers.{i}.attention.key.bias",
                    )
                )
            elif j == 2:
                rename_keys.append(
                    (
                        f"gnn.layers.{i}.attn.proj.{j}.weight",
                        f"gnn.layers.{i}.attention.value.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"gnn.layers.{i}.attn.proj.{j}.bias",
                        f"gnn.layers.{i}.attention.value.bias",
                    )
                )
        for j in range(
            len(
                [
                    config.descriptor_dim * 2,
                    config.descriptor_dim * 2,
                    config.descriptor_dim,
                ]
            )
            + 1
        ):
            if j != 2:
                rename_keys.append(
                    (
                        f"gnn.layers.{i}.mlp.{j}.weight",
                        f"gnn.layers.{i}.mlp.layers.{j}.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"gnn.layers.{i}.mlp.{j}.bias",
                        f"gnn.layers.{i}.mlp.layers.{j}.bias",
                    )
                )
                if j == 1:
                    rename_keys.append(
                        (
                            f"gnn.layers.{i}.mlp.{j}.running_mean",
                            f"gnn.layers.{i}.mlp.layers.{j}.running_mean",
                        )
                    )
                    rename_keys.append(
                        (
                            f"gnn.layers.{i}.mlp.{j}.running_var",
                            f"gnn.layers.{i}.mlp.layers.{j}.running_var",
                        )
                    )
                    rename_keys.append(
                        (
                            f"gnn.layers.{i}.mlp.{j}.num_batches_tracked",
                            f"gnn.layers.{i}.mlp.layers.{j}.num_batches_tracked",
                        )
                    )

    # final projection
    rename_keys.append(("final_proj.weight", "final_projection.final_proj.weight"))
    rename_keys.append(("final_proj.bias", "final_projection.final_proj.bias"))

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def add_keypoint_detector_state_dict(superglue_state_dict, keypoint_detector_state_dict):
    for k, v in keypoint_detector_state_dict.items():
        superglue_state_dict[f"keypoint_detector.{k}"] = v
    return superglue_state_dict


def prepare_imgs_for_image_processor():
    dataset = load_dataset("stevenbucaille/image_matching_fixtures", split="train")
    return [[dataset[0]["image"], dataset[1]["image"]], [dataset[2]["image"], dataset[1]["image"]]]


def extract_keypoint_information_from_image_point_description_output(
    output: SuperPointKeypointDescriptionOutput, i: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.nonzero(output.mask[i]).squeeze()
    keypoints = torch.unsqueeze(output.keypoints[i][indices], dim=0)
    descriptors = torch.unsqueeze(output.descriptors[i][indices], dim=0)
    scores = torch.unsqueeze(output.scores[i][indices], dim=0)
    return keypoints, descriptors, scores


@torch.no_grad()
def convert_superglue_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our SuperPoint structure.
    Also test the model with the image processor and other methods of reading the images.
    """

    keypoint_detector_config = AutoConfig.from_pretrained("magic-leap-community/superpoint")
    superglue_config = get_superglue_config()
    superglue_config.keypoint_detector_config = keypoint_detector_config

    keypoint_detector = AutoModelForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    keypoint_detector.to("cuda")
    keypoint_detector.eval()
    keypoint_detector_state_dict = keypoint_detector.state_dict()

    print("Downloading original model from checkpoint...")
    original_superglue_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)

    print("Converting model parameters...")
    rename_keys = create_rename_keys(superglue_config)
    new_superglue_state_dict = original_superglue_state_dict.copy()
    for src, dest in rename_keys:
        rename_key(new_superglue_state_dict, src, dest)

    conversion_model = SuperGlueConversionModel(superglue_config)
    conversion_model.load_state_dict(new_superglue_state_dict, strict=True, assign=True)
    conversion_model.convert()
    converted_superglue_state_dict = conversion_model.state_dict()
    print("Successfully converted model parameters...")

    converted_superglue_state_dict = add_keypoint_detector_state_dict(
        converted_superglue_state_dict, keypoint_detector_state_dict
    )
    model = SuperGlueForKeypointMatching(superglue_config)
    model.load_state_dict(converted_superglue_state_dict)
    model.to("cuda")
    model.eval()

    print("Successfully loaded weights in the model")
    images = prepare_imgs_for_image_processor()
    preprocessor = SuperGlueImageProcessor()
    inputs = preprocessor(images=images, return_tensors="pt")
    inputs.to("cuda")

    output = model(**inputs, return_dict=True)

    print("Number of matching keypoints")
    print(torch.sum(output.matches[0][0] != -1))
    if save_model:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

        if push_to_hub:
            print("Pushing model to the hub...")
            model_name = "superglue"
            if (
                checkpoint_url
                == "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_outdoor.pth"
            ):
                model_name += "_outdoor"
            elif (
                checkpoint_url
                == "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/master/models/weights/superglue_indoor.pth"
            ):
                model_name += "_indoor"

            model.push_to_hub(
                repo_id=model_name,
                organization="stevenbucaille",
                commit_message="Add model",
            )
            preprocessor.push_to_hub(model_name)


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
    convert_superglue_checkpoint(
        args.checkpoint_url,
        args.pytorch_dump_folder_path,
        args.save_model,
        args.push_to_hub,
    )
