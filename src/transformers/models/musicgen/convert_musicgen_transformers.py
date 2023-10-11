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
"""Convert MusicGen checkpoints from the original repository."""
import argparse
from pathlib import Path
from typing import Dict, OrderedDict, Tuple

import torch
from audiocraft.models import MusicGen

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    EncodecModel,
    MusicgenDecoderConfig,
    MusicgenForConditionalGeneration,
    MusicgenProcessor,
    T5EncoderModel,
)
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


EXPECTED_MISSING_KEYS = ["model.decoder.embed_positions.weights"]


def rename_keys(name):
    if "emb" in name:
        name = name.replace("emb", "model.decoder.embed_tokens")
    if "transformer" in name:
        name = name.replace("transformer", "model.decoder")
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
        name = name.replace("out_norm", "model.decoder.layer_norm")
    if "linears" in name:
        name = name.replace("linears", "lm_heads")
    if "condition_provider.conditioners.description.output_proj" in name:
        name = name.replace("condition_provider.conditioners.description.output_proj", "enc_to_dec_proj")
    return name


def rename_state_dict(state_dict: OrderedDict, hidden_size: int) -> Tuple[Dict, Dict]:
    """Function that takes the fairseq Musicgen state dict and renames it according to the HF
    module names. It further partitions the state dict into the decoder (LM) state dict, and that for the
    encoder-decoder projection."""
    keys = list(state_dict.keys())
    enc_dec_proj_state_dict = {}
    for key in keys:
        val = state_dict.pop(key)
        key = rename_keys(key)
        if "in_proj_weight" in key:
            # split fused qkv proj
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:hidden_size, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[hidden_size : 2 * hidden_size, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-hidden_size:, :]
        elif "enc_to_dec_proj" in key:
            enc_dec_proj_state_dict[key[len("enc_to_dec_proj.") :]] = val
        else:
            state_dict[key] = val
    return state_dict, enc_dec_proj_state_dict


def decoder_config_from_checkpoint(checkpoint: str) -> MusicgenDecoderConfig:
    if checkpoint == "small":
        # default config values
        hidden_size = 1024
        num_hidden_layers = 24
        num_attention_heads = 16
    elif checkpoint == "medium":
        hidden_size = 1536
        num_hidden_layers = 48
        num_attention_heads = 24
    elif checkpoint == "large":
        hidden_size = 2048
        num_hidden_layers = 48
        num_attention_heads = 32
    else:
        raise ValueError(f"Checkpoint should be one of `['small', 'medium', 'large']`, got {checkpoint}.")
    config = MusicgenDecoderConfig(
        hidden_size=hidden_size,
        ffn_dim=hidden_size * 4,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )
    return config


@torch.no_grad()
def convert_musicgen_checkpoint(checkpoint, pytorch_dump_folder=None, repo_id=None, device="cpu"):
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=device)
    decoder_config = decoder_config_from_checkpoint(checkpoint)

    decoder_state_dict = fairseq_model.lm.state_dict()
    decoder_state_dict, enc_dec_proj_state_dict = rename_state_dict(
        decoder_state_dict, hidden_size=decoder_config.hidden_size
    )

    text_encoder = T5EncoderModel.from_pretrained("t5-base")
    audio_encoder = EncodecModel.from_pretrained("facebook/encodec_32khz")
    decoder = MusicgenForCausalLM(decoder_config).eval()

    # load all decoder weights - expect that we'll be missing embeddings and enc-dec projection
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)

    for key in missing_keys.copy():
        if key.startswith(("text_encoder", "audio_encoder")) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)

    if len(missing_keys) > 0:
        raise ValueError(f"Missing key(s) in state_dict: {missing_keys}")

    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected key(s) in state_dict: {unexpected_keys}")

    # init the composite model
    model = MusicgenForConditionalGeneration(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder)

    # load the pre-trained enc-dec projection (from the decoder state dict)
    model.enc_to_dec_proj.load_state_dict(enc_dec_proj_state_dict)

    # check we can do a forward pass
    input_ids = torch.arange(0, 8, dtype=torch.long).reshape(2, -1)
    decoder_input_ids = input_ids.reshape(2 * 4, -1)

    with torch.no_grad():
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    if logits.shape != (8, 1, 2048):
        raise ValueError("Incorrect shape for logits")

    # now construct the processor
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/encodec_32khz", padding_side="left")

    processor = MusicgenProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # set the appropriate bos/pad token ids
    model.generation_config.decoder_start_token_id = 2048
    model.generation_config.pad_token_id = 2048

    # set other default generation config params
    model.generation_config.max_length = int(30 * audio_encoder.config.frame_rate)
    model.generation_config.do_sample = True
    model.generation_config.guidance_scale = 3.0

    if pytorch_dump_folder is not None:
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        logger.info(f"Saving model {checkpoint} to {pytorch_dump_folder}")
        model.save_pretrained(pytorch_dump_folder)
        processor.save_pretrained(pytorch_dump_folder)

    if repo_id:
        logger.info(f"Pushing model {checkpoint} to {repo_id}")
        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint",
        default="small",
        type=str,
        help="Checkpoint size of the MusicGen model you'd like to convert. Can be one of: `['small', 'medium', 'large']`.",
    )
    parser.add_argument(
        "--pytorch_dump_folder",
        required=True,
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Torch device to run the conversion, either cpu or cuda."
    )

    args = parser.parse_args()
    convert_musicgen_checkpoint(args.checkpoint, args.pytorch_dump_folder, args.push_to_hub)
