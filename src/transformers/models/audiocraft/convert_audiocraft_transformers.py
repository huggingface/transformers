# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Audiocraft checkpoints from the original repository."""
import argparse
import os
from pathlib import Path
from typing import OrderedDict

import torch
from audiocraft.models import MusicGen
from huggingface_hub import hf_hub_download

from transformers.models.audiocraft.configuration_audiocraft import AudiocraftConfig
from transformers.models.audiocraft.modeling_audiocraft import AudiocraftForConditionalGeneration
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


EXPECTED_MISSING_KEYS = ["model.embed_positions.weights"]
EXPECTED_UNEXPECTED_KEYS = [
    "condition_provider.conditioners.description.output_proj.weight",
    "condition_provider.conditioners.description.output_proj.bias",
]


def rename_key(name):
    if "emb" in name:
        name = name.replace("emb", "model.embed_tokens")
    if "transformer.layers" in name:
        name = name.replace("transformer.layers", "model.layers")
    if "cross_attention" in name:
        name = name.replace("cross_attention", "encoder_attn")
    if "linear1" in name:
        name = name.replace("linear1", "fc1")
    if "linear2" in name:
        name = name.replace("linear2", "fc2")
    if "norm1" in name:
        name = name.replace("norm1", "self_attn_layer_norm")
    if "norm_cross" in name:
        name = name.replace("norm_cross", "encoder_attn_layer_norm")
    if "norm2" in name:
        name = name.replace("norm2", "final_layer_norm")
    if "out_norm" in name:
        name = name.replace("out_norm", "model.layer_norm")
    if "linears" in name:
        name = name.replace("linears", "lm_heads")
    return name


def rename_state_dict(state_dict: OrderedDict, d_model: int) -> OrderedDict:
    keys = list(state_dict.keys())
    for key in keys:
        val = state_dict.pop(key)
        key = rename_key(key)
        if "in_proj_weight" in key:
            # split fused qkv proj
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:d_model, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[d_model : 2 * d_model, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-d_model:, :]
        else:
            state_dict[key] = val
    return state_dict


def config_from_checkpoint(checkpoint):
    if checkpoint == "dummy":
        d_model = 1024
        ffn_dim = d_model * 4
        num_layers = 2
        num_codebooks = 4
        config = AudiocraftConfig(
            d_model=d_model, intermediate_size=ffn_dim, num_hidden_layers=num_layers, num_codebooks=num_codebooks
        )
    elif checkpoint == "small":
        d_model = 1024
        ffn_dim = d_model * 4
        num_layers = 24
        num_codebooks = 4
        config = AudiocraftConfig(
            d_model=d_model, intermediate_size=ffn_dim, num_hidden_layers=num_layers, num_codebooks=num_codebooks
        )
    return config


def convert_audiocraft_checkpoint(
    checkpoint, pytorch_dump_folder=None, push_to_hub=False, device="cpu"
):
    if checkpoint == "dummy":
        hf_hub_download("music-gen-sprint/audiocraft-dummy", filename="model.pt", local_dir=pytorch_dump_folder)
        state_dict = torch.load(os.path.join(pytorch_dump_folder, "model.pt"))
        os.remove(os.path.join(pytorch_dump_folder, "model.pt"))
        config = config_from_checkpoint(checkpoint)
    else:
        fairseq_model = MusicGen.get_pretrained(checkpoint, device=device)
        state_dict = fairseq_model.lm.state_dict()
        config = config_from_checkpoint(checkpoint)

    state_dict = rename_state_dict(state_dict, d_model=config.d_model)

    model = AudiocraftForConditionalGeneration(config).eval()
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if not set(missing_keys) == set(EXPECTED_MISSING_KEYS):
        raise ValueError(f"Missing key(s) in state_dict: {list(set(missing_keys) - set(EXPECTED_MISSING_KEYS))}")

    if not set(unexpected_keys) == set(EXPECTED_UNEXPECTED_KEYS):
        raise ValueError(
            f"Unexpected key(s) in state_dict:  {list(set(unexpected_keys) - set(EXPECTED_UNEXPECTED_KEYS))}"
        )

    # Check we can do a forward pass
    input_values = torch.ones((2, 4, 1), dtype=torch.long)
    cross_attention_inputs = torch.ones((1, 2, 1, 1024))

    with torch.no_grad():
        model.forward(input_values, encoder_hidden_states=cross_attention_inputs)

    if pytorch_dump_folder is not None:
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        logger.info(f"Saving model {checkpoint} to {pytorch_dump_folder}")
        model.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        model.push_to_hub(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint",
        default="dummy",
        type=str,
        help="Checkpoint size of the Audiocraft model you'd like to convert. Can be one of: small, medium, large.",
    )
    parser.add_argument(
        "--pytorch_dump_folder", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Torch device to run the conversion, either cpu or cuda."
    )

    args = parser.parse_args()
    convert_audiocraft_checkpoint(args.checkpoint, args.pytorch_dump_folder, args.push_to_hub)
