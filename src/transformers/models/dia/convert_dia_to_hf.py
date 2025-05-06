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
import re
import torch
import os
from transformers import DiaModel,  DiaProcessor, DiaTokenizer, DiaConfig
from transformers.utils.import_utils import _is_package_available
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Provide just the list of layer keys you want to fix
shape_mappings = [
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

def reshape_or_transpose(tensor, target_tensor):
    """Try reshaping or transposing tensor to match the shape of target_tensor."""
    numel = tensor.numel()
    target_shape = target_tensor
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
    checkpoint_path = snapshot_download(repo_id=checkpoint_path, allow_patterns="*.safetensors")
    print(f"Downloaded checkpoint from Hugging Face Hub: {checkpoint_path}")

    with torch.device('meta'):
        model_class = DiaModel(config=DiaConfig())
    model_dict = model_class.state_dict()
    model_class_keys = model_dict.keys()
    files = os.listdir(checkpoint_path)
    for file in files:
        if file.endswith(".safetensors"):
            load_function = load_file
        elif file.endswith(".pt"):
            load_function = torch.load
    checkpoint_path = os.path.join(checkpoint_path, files[0])
    state_dict = load_function(checkpoint_path, 'cpu')
    converted_state_dict = {}
    embeddings = {}
    for key, tensor in state_dict.items():
        reshaped = False

        if re.sub(r"\d+", "*",key) in shape_mappings:
            if key in model_class_keys:
                target_shape = model_dict[key].shape
                try:
                    new_tensor, method = reshape_or_transpose(tensor, target_shape)
                    print(f"{key}: {method} from {tensor.shape} to {target_shape}")
                    tensor = new_tensor
                    reshaped = True
                except Exception as e:
                    print(f"WARNING: Could not reshape {key}: {e}")
            else:
                print(f"WARNING: {key} not found in model class keys, skipping reshape.")
            converted_state_dict[key] = tensor
        elif "embeddings" in key:
            embeddings_key = key.rsplit(".",2)[0]+".embed.weight"
            if embeddings_key in embeddings:
                embeddings[embeddings_key] += [tensor]
            else:
                embeddings[embeddings_key] = [tensor]
        else:
            converted_state_dict[key] = tensor
    embeddings = {k: torch.cat(v,dim=-1) for k, v in embeddings.items()}
    converted_state_dict.update(embeddings)
    print(f"Saved converted checkpoint to {pytorch_dump_folder_path}")
    model_class.load_state_dict(converted_state_dict, assign=True)
    return model_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument("--checkpoint_path", type=str, default="nari-labs/Dia-1.6B", help="Path to the downloaded checkpoints")
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
