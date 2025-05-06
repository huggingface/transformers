#!/usr/bin/env python
"""Converts a Dia model in OpenAI format to Hugging Face format."""
# Copyright 2025 The HuggingFace Inc. team and the OpenAI team. All rights reserved.
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

import torch

from transformers import AutoModel, DiaAudioProcessor, DiaProcessor, DiaTokenizer
from transformers.utils.import_utils import _is_package_available
from huggingface_hub import hf_hub_download

# Provide just the list of layer keys you want to fix
layer_keys_to_fix = [
    'encoder.layers.*.mlp.wi_fused.weight',
    'decoder.layers.*.cross_attention.k_proj.weight',
    'decoder.logits_dense.weight',
    'encoder.layers.*.self_attention.q_proj.weight',
    'encoder.layers.*.mlp.wo.weight',
    'decoder.layers.*.mlp.wo.weight',
    'decoder.layers.*.self_attention.v_proj.weight',
    'decoder.layers.*.self_attention.o_proj.weight',
    'encoder.embedding.weight',
    'encoder.layers.*.self_attention.k_proj.weight',
    'decoder.layers.*.cross_attention.q_proj.weight',
    'decoder.layers.*.self_attention.k_proj.weight',
    'decoder.layers.*.self_attention.q_proj.weight',
    'encoder.layers.*.self_attention.o_proj.weight',
    'decoder.layers.*.cross_attention.o_proj.weight',
    'encoder.layers.*.self_attention.v_proj.weight',
    'decoder.layers.*.mlp.wi_fused.weight',
    'decoder.layers.*.cross_attention.v_proj.weight',
]

def match_layer(pattern, layer_name):
    """Check if a wildcard pattern (with *) matches the layer name."""
    if '*' not in pattern:
        return pattern == layer_name
    prefix, suffix = pattern.split('*')
    return layer_name.startswith(prefix) and layer_name.endswith(suffix)

def reshape_or_transpose(tensor, target_tensor):
    """Try reshaping or transposing tensor to match the shape of target_tensor."""
    numel = tensor.numel()
    target_shape = target_tensor.shape
    target_numel = target_tensor.numel()

    if numel != target_numel:
        raise ValueError(f"Cannot fix tensor of {tensor.shape} to {target_shape} (element count mismatch)")

    # Direct reshape
    try:
        reshaped = tensor.view(target_shape)
        return reshaped, 'reshaped'
    except Exception:
        pass

    # Transpose if 2D and transpose shape fits
    if tensor.ndim == 2 and tensor.T.shape == target_shape:
        return tensor.T, 'transposed'

    # Flatten-reshape fallback
    reshaped = tensor.view(-1).reshape(target_shape)
    return reshaped, 'flattened_reshape'


def convert_dia_model_to_hf(checkpoint_path, pytorch_dump_folder_path):
    """
    Converts a Dia model in OpenAI format to Hugging Face format.
    Args:
        checkpoint_path (`str`):
            Path to the downloaded checkpoints.
        pytorch_dump_folder_path (`str`):
            Path to the output PyTorch model.
    """
    # Download from HF Hub if checkpoint_path is None
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename='*.safetensors')
        print(f"Downloaded checkpoint from Hugging Face Hub: {checkpoint_path}")

    with torch.device('meta'):
        model_class = DiaModel()

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    converted_state_dict = {}

    for key, tensor in state_dict.items():
        reshaped = False
        for pattern, (loaded_shape, target_shape) in shape_mappings.items():
            if match_layer(pattern, key):
                try:
                    new_tensor, method = reshape_or_transpose(tensor, target_shape)
                    print(f"{key}: {method} from {tensor.shape} to {target_shape}")
                    tensor = new_tensor
                    reshaped = True
                except Exception as e:
                    print(f"WARNING: Could not reshape {key}: {e}")
                break
        if not reshaped:
            print(f"Keeping {key} with shape {tensor.shape}")
        converted_state_dict[key] = tensor

    print(f"Saved converted checkpoint to {output_path}")
    model.load_state_dict(converted_state_dict)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the downloaded checkpoints")
    parser.add_argument("--pytorch_dump_folder_path", default="converted_dia_ckpt", type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=False,
        help="Whether or not the preprocessor (tokenizer + feature extractor) should be converted along with the model.",
    )
    args = parser.parse_args()

    model = convert_dia_model_to_hf(args.checkpoint_path, args.pytorch_dump_folder_path)

    if args.convert_preprocessor:
        try:
            if not _is_package_available("tiktoken"):
                raise ModuleNotFoundError(
                    """`tiktoken` is not installed, use `pip install tiktoken` to convert the tokenizer"""
                )
        except Exception as e:
            print(e)
        else:
            feature_extractor = DiaAudioProcessor(
                feature_size=model.config.num_mel_bins,
                # the rest of default parameters are the same as hardcoded in openai/dia
            )
            tokenizer = DiaTokenizer()
            processor = DiaProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
            processor.save_pretrained(args.pytorch_dump_folder_path)

    model.save_pretrained(args.pytorch_dump_folder_path)
