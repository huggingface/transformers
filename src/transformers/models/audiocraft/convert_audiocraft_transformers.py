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
from typing import Dict, OrderedDict

import torch
from audiocraft.models import MusicGen
from huggingface_hub import hf_hub_download

from transformers import T5Config, T5EncoderModel
from transformers.models.audiocraft.configuration_audiocraft import AudiocraftConfig, AudiocraftDecoderConfig
from transformers.models.audiocraft.modeling_audiocraft import AudiocraftForConditionalGeneration
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


CHECKPOINT_TO_T5 = {
    "dummy": "t5-base",
    "small": "t5-base",
}

EXPECTED_MISSING_DECODER_KEYS = ["embed_positions.weights"]


def rename_key(name):
    if "emb" in name:
        name = name.replace("emb", "embed_tokens")
    if "transformer.layers" in name:
        name = name.replace("transformer.layers", "layers")
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
        name = name.replace("out_norm", "layer_norm")
    if "linears" in name:
        name = name.replace("linears", "lm_heads")
    if "condition_provider.conditioners.description.output_proj" in name:
        name = name.replace("condition_provider.conditioners.description.output_proj", "encoder_projection")
    return name


def rename_decoder_state_dict(state_dict: OrderedDict, d_model: int) -> tuple[Dict, Dict, Dict]:
    """Function that takes the fairseq Audiocraft state dict and renames it according to the HF
    module names. It further partitions the state dict into three: the decoder state dict (state_dict),
    the state dict for the LM head, and the state dict for the encoder projection."""
    keys = list(state_dict.keys())
    lm_heads = {}
    encoder_projection = {}
    for key in keys:
        val = state_dict.pop(key)
        key = rename_key(key)
        if "lm_heads" in key:
            lm_heads[key[len("lm_heads.") :]] = val

        elif "encoder_projection" in key:
            encoder_projection[key[len("encoder_projection.") :]] = val

        elif "in_proj_weight" in key:
            # split fused qkv proj
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:d_model, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[d_model : 2 * d_model, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-d_model:, :]

        else:
            state_dict[key] = val
    return state_dict, lm_heads, encoder_projection


def config_from_checkpoint(checkpoint: str) -> AudiocraftConfig:
    if checkpoint == "dummy":
        d_model = 1024
        ffn_dim = d_model * 4
        num_layers = 2
        num_codebooks = 4
    elif checkpoint == "small":
        d_model = 1024
        ffn_dim = d_model * 4
        num_layers = 24
        num_codebooks = 4
    lm_config = AudiocraftDecoderConfig(
        d_model=d_model, intermediate_size=ffn_dim, num_hidden_layers=num_layers, num_codebooks=num_codebooks
    )
    t5_config = T5Config.from_pretrained(CHECKPOINT_TO_T5[checkpoint])
    config = AudiocraftConfig.from_t5_lm_config(t5_config=t5_config, lm_config=lm_config)
    return config


def convert_audiocraft_checkpoint(checkpoint, pytorch_dump_folder=None, push_to_hub=False, device="cpu"):
    if checkpoint == "dummy":
        hf_hub_download("music-gen-sprint/audiocraft-dummy", filename="model.pt", local_dir=pytorch_dump_folder)
        state_dict = torch.load(os.path.join(pytorch_dump_folder, "model.pt"))
        os.remove(os.path.join(pytorch_dump_folder, "model.pt"))
        config = config_from_checkpoint(checkpoint)
    else:
        fairseq_model = MusicGen.get_pretrained(checkpoint, device=device)
        state_dict = fairseq_model.lm.state_dict()
        config = config_from_checkpoint(checkpoint)

    state_dict, lm_heads, encoder_projection = rename_decoder_state_dict(state_dict, d_model=config.lm_config.d_model)

    model = AudiocraftForConditionalGeneration(config).eval()
    model.model.encoder = T5EncoderModel.from_pretrained(CHECKPOINT_TO_T5[checkpoint])
    model.model.encoder_projection.load_state_dict(encoder_projection)
    missing_decoder_keys, unexpected_decoder_keys = model.model.decoder.load_state_dict(state_dict, strict=False)
    model.lm_heads.load_state_dict(
        lm_heads,
    )

    if set(missing_decoder_keys) != set(EXPECTED_MISSING_DECODER_KEYS):
        raise ValueError(
            f"Missing key(s) in state_dict: {list(set(missing_decoder_keys) - set(EXPECTED_MISSING_DECODER_KEYS))}"
        )
    if len(unexpected_decoder_keys) > 0:
        raise ValueError(f"Unexpected key(s) in state_dict: {unexpected_decoder_keys}")

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
        "--pytorch_dump_folder",
        default="/Users/sanchitgandhi/convert-audiocraft",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Torch device to run the conversion, either cpu or cuda."
    )

    args = parser.parse_args()
    convert_audiocraft_checkpoint(args.checkpoint, args.pytorch_dump_folder, args.push_to_hub)
