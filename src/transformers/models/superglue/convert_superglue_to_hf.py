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


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"kenc.encoder.(\d+)": r"keypoint_encoder.encoder.\1.old",
    r"gnn.layers.(\d+).attn.proj.0": r"gnn.layers.\1.attention.self.query",
    r"gnn.layers.(\d+).attn.proj.1": r"gnn.layers.\1.attention.self.key",
    r"gnn.layers.(\d+).attn.proj.2": r"gnn.layers.\1.attention.self.value",
    r"gnn.layers.(\d+).attn.merge": r"gnn.layers.\1.attention.output.dense",
    r"gnn.layers.(\d+).mlp.0": r"gnn.layers.\1.mlp.0.linear",
    r"gnn.layers.(\d+).mlp.1": r"gnn.layers.\1.mlp.0.batch_norm",
    r"gnn.layers.(\d+).mlp.3": r"gnn.layers.\1.mlp.1",
    r"final_proj": r"final_projection.final_proj",
}


def convert_old_keys_to_new_keys(state_dict_keys: List[str], conversion_mapping=ORIGINAL_TO_CONVERTED_KEY_MAPPING):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in conversion_mapping.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def replace_state_dict_keys(all_keys, new_keys, original_state_dict):
    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        state_dict[new_key] = original_state_dict.pop(key).contiguous().clone()
    return state_dict


def convert_state_dict(state_dict, config):
    converted_to_final_key_mapping = {}

    def convert_conv_to_linear(keys):
        for key in keys:
            state_dict[key] = state_dict[key].squeeze(-1)

    conv_keys = []
    # Keypoint Encoder
    keypoint_encoder_key = "keypoint_encoder.encoder"
    for i in range(1, len(config.keypoint_encoder_sizes) + 2):
        old_conv_key = f"{keypoint_encoder_key}.{(i - 1) * 3}.old"
        new_index = i - 1
        new_conv_key = f"{keypoint_encoder_key}.{new_index}."
        if i < len(config.keypoint_encoder_sizes) + 1:
            new_conv_key = f"{new_conv_key}linear."
        converted_to_final_key_mapping[rf"{old_conv_key}\."] = new_conv_key
        if i < len(config.keypoint_encoder_sizes) + 1:
            old_batch_norm_key = f"{keypoint_encoder_key}.{(i - 1) * 3 + 1}.old"
            new_batch_norm_key = f"{keypoint_encoder_key}.{new_index}.batch_norm."
            converted_to_final_key_mapping[rf"{old_batch_norm_key}\."] = new_batch_norm_key

        conv_keys.append(f"{old_conv_key}.weight")

    # Attentional GNN
    for i in range(len(config.gnn_layers_types)):
        gnn_layer_key = f"gnn.layers.{i}"
        ## Attention
        attention_key = f"{gnn_layer_key}.attention"
        conv_keys.extend(
            [
                f"{attention_key}.self.query.weight",
                f"{attention_key}.self.key.weight",
                f"{attention_key}.self.value.weight",
                f"{attention_key}.output.dense.weight",
            ]
        )

        ## MLP
        conv_keys.extend([f"{gnn_layer_key}.mlp.0.linear.weight", f"{gnn_layer_key}.mlp.1.weight"])

    # Final Projection
    conv_keys.append("final_projection.final_proj.weight")

    convert_conv_to_linear(conv_keys)
    all_keys = list(state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys, converted_to_final_key_mapping)
    state_dict = replace_state_dict_keys(all_keys, new_keys, state_dict)
    return state_dict


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
        hidden_size=256,
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

    state_dict = replace_state_dict_keys(all_keys, new_keys, original_state_dict)
    state_dict = convert_state_dict(state_dict, config)

    del original_state_dict
    gc.collect()
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

    organization = "magic-leap-community"
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
