# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted weights from original model code at https://github.com/sczhou/ProPainter

import argparse
import os

import numpy as np
import torch
from datasets import load_dataset

from transformers import (
    ProPainterConfig,
    ProPainterModel,
    ProPainterVideoProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_optical_flow(old_key, network_mapping):
    new_key = ""
    for old_prefix, new_prefix in network_mapping.items():
        if old_prefix in old_key:
            new_key = old_key.replace(f"module.{old_prefix}", f"optical_flow_model.{new_prefix}")
            # Handle layer & block transformations
            if "layer" in new_key:
                parts = new_key.split(".")
                layer_x = int(parts[2][5])  # Extract the number after 'layer'
                layer_y = int(parts[3])  # The sub-layer number

                # Compute the corresponding resblock number
                # Example: layer1.0 -> resblock 0, layer1.1 -> resblock 1, etc.
                resblock_z = (layer_x - 1) * 2 + layer_y

                # Replace 'layerX.Y' with 'resblocks.Z'
                new_key = new_key.replace(f"layer{layer_x}.{layer_y}", f"resblocks.{resblock_z}")

            if "convc" in new_key:
                new_key = new_key.replace("convc", "conv_corr")
            if "convf" in new_key:
                new_key = new_key.replace("convf", "conv_flow")

    return new_key


def rename_flow_completion(old_key, network_mapping):
    new_key = ""
    for old_prefix, new_prefix in network_mapping.items():
        if old_prefix in old_key:
            new_key = old_key.replace(f"{old_prefix}", f"{new_prefix}")
            # Handle specific layer/block transformations
            if "mid_dilation" in new_key:
                new_key = new_key.replace("mid_dilation", "intermediate_dilation")
            if "feat_prop_module" in new_key:
                new_key = new_key.replace("feat_prop_module", "feature_propagation_module")
            if "edgeDetector.mid_layer" in new_key:
                new_key = new_key.replace("edgeDetector.mid_layer", "edgeDetector.intermediate_layer")

    return new_key


def rename_inpaint_generator(old_key, network_mapping):
    new_key = ""
    for old_prefix, new_prefix in network_mapping.items():
        if old_prefix in old_key:
            new_key = old_key.replace(f"{old_prefix}", f"{new_prefix}")
            if "norm" in new_key:
                new_key = new_key.replace("norm", "layer_norm")
    return new_key


def map_keys(old_keys, module):
    key_mapping = {}

    # Define network type and layer/block mappings
    network_mapping_optical_flow = {
        "fnet": "feature_network",
        "cnet": "context_network",
        "update_block": "update_block",
    }

    network_mapping_flow_completion = {
        "downsample": "flow_completion_net.downsample",
        "encoder1": "flow_completion_net.encoder1",
        "encoder2": "flow_completion_net.encoder2",
        "decoder1": "flow_completion_net.decoder1",
        "decoder2": "flow_completion_net.decoder2",
        "upsample": "flow_completion_net.upsample",
        "mid_dilation": "flow_completion_net.intermediate_dilation",
        "feat_prop_module.deform_align.backward_": "flow_completion_net.feature_propagation_module.deform_align.backward_",
        "feat_prop_module.deform_align.forward_": "flow_completion_net.feature_propagation_module.deform_align.forward_",
        "feat_prop_module.backbone.backward_": "flow_completion_net.feature_propagation_module.backbone.backward_",
        "feat_prop_module.backbone.forward_": "flow_completion_net.feature_propagation_module.backbone.forward_",
        "feat_prop_module.fusion": "flow_completion_net.feature_propagation_module.fusion",
        "edgeDetector.projection": "flow_completion_net.edgeDetector.projection",
        "edgeDetector.mid_layer": "flow_completion_net.edgeDetector.intermediate_layer",
        "edgeDetector.out_layer": "flow_completion_net.edgeDetector.out_layer",
    }

    network_mapping_inpaint_generator = {
        "encoder": "inpaint_generator.encoder",
        "decoder": "inpaint_generator.decoder",
        "ss": "inpaint_generator.soft_split",
        "sc": "inpaint_generator.soft_comp",
        "feat_prop_module.": "inpaint_generator.feature_propagation_module.",
        "transformers.transformer.": "inpaint_generator.transformers.transformer.",
    }

    if module == "optical_flow":
        network_mapping = network_mapping_optical_flow
        for old_key in old_keys:
            new_key = rename_optical_flow(old_key, network_mapping)
            key_mapping[new_key] = old_key

    elif module == "flow_completion":
        network_mapping = network_mapping_flow_completion
        for old_key in old_keys:
            new_key = rename_flow_completion(old_key, network_mapping)
            key_mapping[new_key] = old_key

    else:
        network_mapping = network_mapping_inpaint_generator
        for old_key in old_keys:
            new_key = rename_inpaint_generator(old_key, network_mapping)
            key_mapping[new_key] = old_key

    return key_mapping


def rename_key(state_dict, old_key, new_key):
    if old_key in state_dict:
        state_dict[new_key] = state_dict.pop(old_key)


def create_new_state_dict(combined_state_dict, original_state_dict, key_mapping):
    for new_key, old_key in key_mapping.items():
        rename_key(original_state_dict, old_key, new_key)
        combined_state_dict[new_key] = original_state_dict[new_key]


def prepare_input():
    ds = load_dataset("ruffy369/propainter-object-removal")
    ds_images = ds["train"]["image"]
    num_frames = len(ds_images) // 2
    video = [np.array(ds_images[i]) for i in range(num_frames)]
    # stack to convert H,W mask frame to compatible H,W,C frame
    masks = [np.stack([np.array(ds_images[i])], axis=-1) for i in range(num_frames, 2 * num_frames)]
    return video, masks


@torch.no_grad()
def convert_propainter_checkpoint(args):
    combined_state_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download the original checkpoint
    original_state_dict_optical_flow = torch.hub.load_state_dict_from_url(
        args.optical_flow_checkpoint_url, map_location="cpu"
    )
    original_state_dict_flow_completion = torch.hub.load_state_dict_from_url(
        args.flow_completion_checkpoint_url, map_location="cpu"
    )
    original_state_dict_inpaint_generator = torch.hub.load_state_dict_from_url(
        args.inpaint_generator_checkpoint_url, map_location="cpu"
    )

    key_mapping_optical_flow = map_keys(list(original_state_dict_optical_flow.keys()), "optical_flow")

    key_mapping_flow_completion = map_keys(list(original_state_dict_flow_completion.keys()), "flow_completion")

    key_mapping_inpaint_generator = map_keys(list(original_state_dict_inpaint_generator.keys()), "inpaint_generator")

    # Create new state dict with updated keys for optical flow model
    create_new_state_dict(combined_state_dict, original_state_dict_optical_flow, key_mapping_optical_flow)

    # Create new state dict with updated keys for flow completion network
    create_new_state_dict(
        combined_state_dict,
        original_state_dict_flow_completion,
        key_mapping_flow_completion,
    )

    # Create new state dict with updated keys for propainter inpaint generator
    create_new_state_dict(
        combined_state_dict,
        original_state_dict_inpaint_generator,
        key_mapping_inpaint_generator,
    )

    dummy_checkpoint_path = os.path.join(args.pytorch_dump_folder_path, "pytorch_model.bin")
    torch.save(combined_state_dict, dummy_checkpoint_path)

    # Load created new state dict after weights conversion (model.load_state_dict wasn't used because some parameters in the model are initialised
    # instead of being loaded from any pretrained model and error occurs with `load_state_dict`)
    model = (
        ProPainterModel(ProPainterConfig())
        .from_pretrained(f"{args.pytorch_dump_folder_path}/", local_files_only=True)
        .to(device)
    )
    model.eval()

    expected_output_reconstruction = torch.tensor([[36, 32, 21], [65, 52, 43], [84, 67, 57]], dtype=torch.uint8)
    if args.verify_logits:
        video, masks = prepare_input()
        image_processor = ProPainterVideoProcessor()
        inputs = image_processor(video, masks=masks, return_tensors="pt").to(device)
        outputs = model(**inputs)
        outputs_reconstruction = outputs.reconstruction

        assert torch.allclose(
            torch.tensor(outputs_reconstruction[0][0][-3:]),
            expected_output_reconstruction,
            atol=1e-4,
        )
        print("Looks good!")
    else:
        print("Converted without verifying logits")

    if os.path.exists(dummy_checkpoint_path):
        os.remove(dummy_checkpoint_path)

    if args.pytorch_dump_folder_path is not None:
        print(f"Saving model for {args.model_name} to {args.pytorch_dump_folder_path}")
        model.save_pretrained(args.pytorch_dump_folder_path)

    if args.push_to_hub:
        print(f"Pushing model for {args.model_name} to hub")
        model.push_to_hub(f"ruffy369/{args.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model-name",
        default="propainter-hf",
        type=str,
        choices=["propainter-hf"],
        help="Name of the ProPainter model you'd like to convert.",
    )
    parser.add_argument(
        "--optical-flow-checkpoint-url",
        default="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
        type=str,
        help="Url for the optical flow module weights.",
    )
    parser.add_argument(
        "--flow-completion-checkpoint-url",
        default="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        type=str,
        help="Url for the flow completion module weights.",
    )
    parser.add_argument(
        "--inpaint-generator-checkpoint-url",
        default="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
        type=str,
        help="Url for the inpaint generator module weights.",
    )
    parser.add_argument(
        "--pytorch-dump-folder-path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--verify-logits",
        action="store_true",
        help="Whether or not to verify the logits against the original implementation.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_propainter_checkpoint(args)
