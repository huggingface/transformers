# Copyright 2025 The Nari Labs and HuggingFace Inc. team. All rights reserved.
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
"""Converts a Dia model in Nari Labs format to Hugging Face format."""

import argparse
import os
import re

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from transformers import (
    DacModel,
    DiaConfig,
    DiaFeatureExtractor,
    DiaForConditionalGeneration,
    DiaProcessor,
    DiaTokenizer,
    GenerationConfig,
)
from transformers.utils.import_utils import is_tiktoken_available


# Provide just the list of layer keys you want to fix
shape_mappings = [
    "encoder.layers.*.mlp.gate_up_proj.weight",
    "encoder.layers.*.mlp.down_proj.weight",
    "encoder.layers.*.self_attention.q_proj.weight",
    "encoder.layers.*.self_attention.k_proj.weight",
    "encoder.layers.*.self_attention.v_proj.weight",
    "encoder.layers.*.self_attention.o_proj.weight",
    "decoder.layers.*.mlp.gate_up_proj.weight",
    "decoder.layers.*.mlp.down_proj.weight",
    "decoder.layers.*.self_attention.q_proj.weight",
    "decoder.layers.*.self_attention.k_proj.weight",
    "decoder.layers.*.self_attention.v_proj.weight",
    "decoder.layers.*.self_attention.o_proj.weight",
    "decoder.layers.*.cross_attention.q_proj.weight",
    "decoder.layers.*.cross_attention.k_proj.weight",
    "decoder.layers.*.cross_attention.v_proj.weight",
    "decoder.layers.*.cross_attention.o_proj.weight",
    "decoder.logits_dense.weight",
]

# Provide renamings here
rename_mapping = {
    "mlp.wo": "mlp.down_proj",
    "mlp.wi_fused": "mlp.gate_up_proj",
}


def get_generation_config(config):
    model_generation_config = GenerationConfig.from_model_config(config)
    model_generation_config._from_model_config = False
    model_generation_config.do_sample = True
    model_generation_config.top_k = 45
    model_generation_config.top_p = 0.95
    model_generation_config.temperature = 1.2
    model_generation_config.guidance_scale = 3.0
    model_generation_config.max_length = 3072  # Decoder max length

    return model_generation_config


def convert_dia_model_to_hf(checkpoint_path, verbose=False):
    """
    Converts a Dia model in Nari Labs format to Hugging Face format.
    Args:
        checkpoint_path (`str`):
            Path to the downloaded checkpoints.
        verbose (`bool`, *optional*)
            Whether to print information during conversion.
    """
    # Download from HF Hub if checkpoint_path is None
    checkpoint_path = snapshot_download(repo_id=checkpoint_path, allow_patterns=["*.pth", "*.safetensors"])
    print(f"Downloaded checkpoint from Hugging Face Hub: {checkpoint_path}")

    # Initialize base model with default config == 1.6B model
    with torch.device("meta"):
        hf_model = DiaForConditionalGeneration(config=DiaConfig())
    hf_model_dict = hf_model.state_dict()
    hf_model_keys = hf_model_dict.keys()

    # Iterate through dir to catch all respective files - prefers safetensors but allows pt
    files = os.listdir(checkpoint_path)
    for file in files:
        if file.endswith(".safetensors"):
            load_function = load_file
        elif file.endswith(".pth"):
            load_function = torch.load
    checkpoint_path = os.path.join(checkpoint_path, files[0])
    nari_state_dict = load_function(checkpoint_path, "cpu")

    # Conversion starts here
    converted_state_dict = {}
    embeddings = {}
    for key, tensor in nari_state_dict.items():
        # add prefix
        key = "model." + key

        # rename some weights
        for original, rename in rename_mapping.items():
            if original in key:
                key = re.sub(original, rename, key)

        # decoder multi channel
        if "embeddings" in key:
            embeddings_key = key.rsplit(".", 2)[0] + ".embed.weight"
            if embeddings_key in embeddings:
                embeddings[embeddings_key] += [tensor]
            else:
                embeddings[embeddings_key] = [tensor]
            continue
        elif re.sub(r"\d+", "*", key).removeprefix("model.") in shape_mappings:
            # add exception to the head
            if "logits_dense" in key:
                key = re.sub("decoder.logits_dense", "logits_dense", key).removeprefix("model.")

            # dense general
            if key in hf_model_keys:
                tensor_shape = tensor.shape
                target_shape = hf_model_dict[key].shape
                try:
                    tensor = tensor.reshape(target_shape[1], target_shape[0]).T
                    if verbose:
                        print(f"{key}: transpose reshaped from {tensor_shape} to {target_shape}")
                except Exception as e:
                    print(f"WARNING: Could not reshape {key}: {e}")

        converted_state_dict[key] = tensor

    # Combining the embeddings as last step
    embeddings = {k: torch.cat(v, dim=0) for k, v in embeddings.items()}
    converted_state_dict.update(embeddings)

    # Load converted weights into HF model
    hf_model.load_state_dict(converted_state_dict, assign=True)

    # Overwrite generation config
    hf_model.generation_config = get_generation_config(DiaConfig())

    return hf_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument(
        "--checkpoint_path", type=str, default="nari-labs/Dia-1.6B", help="Path to the downloaded checkpoints"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="AntonV/Dia-1.6B", type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=True,
        help="Whether or not the preprocessor (tokenizer + feature extractor) should be converted along with the model.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether or not to log information during conversion.",
    )
    args = parser.parse_args()

    model = convert_dia_model_to_hf(args.checkpoint_path, args.verbose)
    if args.convert_preprocessor:
        try:
            if not is_tiktoken_available(with_blobfile=False):
                raise ModuleNotFoundError(
                    """`tiktoken` is not installed, use `pip install tiktoken` to convert the tokenizer"""
                )
        except Exception as e:
            print(e)
        else:
            processor = DiaProcessor(
                DiaFeatureExtractor(sampling_rate=44100, hop_length=512),
                DiaTokenizer(),
                DacModel.from_pretrained("descript/dac_44khz"),
            )
            processor.save_pretrained(args.pytorch_dump_folder_path)

    model.save_pretrained(args.pytorch_dump_folder_path)
    print(f"Saved converted checkpoint to {args.pytorch_dump_folder_path}")
