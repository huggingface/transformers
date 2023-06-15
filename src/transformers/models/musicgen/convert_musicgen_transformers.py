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
"""Convert Musicgen checkpoints from the original repository."""
import argparse
from pathlib import Path
from typing import Dict, OrderedDict, Tuple

import torch
from audiocraft.models import MusicGen

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    MusicgenDecoderConfig,
    MusicgenForConditionalGeneration,
    MusicgenProcessor,
    T5EncoderModel,
)
from transformers.models.musicgen.modeling_encodec import EncodecModel
from transformers.models.musicgen.modeling_musicgen import MusicgenForCausalLM
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


CHECKPOINT_TO_T5 = {
    "small": "t5-base",
}
CHECKPOINT_TO_ENCODEC = {
    "small": "facebook/encodec_32khz",
}

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


def rename_state_dict(state_dict: OrderedDict, d_model: int) -> Tuple[Dict, Dict]:
    """Function that takes the fairseq Musicgen state dict and renames it according to the HF
    module names. It further partitions the state dict into the decoder (LM) state dict, and that
    for the encoder-decoder projection."""
    keys = list(state_dict.keys())
    enc_dec_proj_state_dict = {}
    for key in keys:
        val = state_dict.pop(key)
        key = rename_keys(key)
        if "in_proj_weight" in key:
            # split fused qkv proj
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:d_model, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[d_model : 2 * d_model, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-d_model:, :]
        elif "enc_to_dec_proj" in key:
            enc_dec_proj_state_dict[key[len("enc_to_dec_proj.") :]] = val
        else:
            state_dict[key] = val
    return state_dict, enc_dec_proj_state_dict


def decoder_config_from_checkpoint(checkpoint: str) -> MusicgenDecoderConfig:
    if checkpoint == "small":
        d_model = 1024
        ffn_dim = d_model * 4
        num_layers = 24
        num_codebooks = 4
    config = MusicgenDecoderConfig(
        d_model=d_model,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        num_codebooks=num_codebooks,
        tie_word_embeddings=False,
    )
    return config


@torch.no_grad()
def convert_musicgen_checkpoint(checkpoint, pytorch_dump_folder=None, push_to_hub=False, device="cpu"):
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=device)
    decoder_config = decoder_config_from_checkpoint(checkpoint)

    # TODO(SG): remove debugging statement
    fairseq_model.lm.transformer.layers = fairseq_model.lm.transformer.layers[:2]
    decoder_config.num_layers = 2

    decoder_state_dict = fairseq_model.lm.state_dict()
    decoder_state_dict, enc_dec_proj_state_dict = rename_state_dict(decoder_state_dict, d_model=decoder_config.d_model)

    text_encoder = T5EncoderModel.from_pretrained(CHECKPOINT_TO_T5[checkpoint])
    audio_encoder = EncodecModel.from_pretrained(CHECKPOINT_TO_ENCODEC[checkpoint])
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
    decoder_input_ids = input_ids.reshape(2, 4, -1)

    with torch.no_grad():
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    if logits.shape != (2, 4, 1, 2048):
        raise ValueError("Incorrect shape for logits")

    EXPECTED_SLICE = [-3.3180, -1.5341, -3.2796, -2.0692, -2.1036, -2.3339, 0.6623, -6.3549, 0.8477, -2.0866]

    if torch.max(torch.abs(logits[0, 0, 0, :10] - torch.tensor(EXPECTED_SLICE))) > 1e-4:
        raise ValueError("Logits exceed tolerance threshold")

    # now construct the processor
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_TO_T5[checkpoint])
    feature_extractor = AutoFeatureExtractor.from_pretrained(CHECKPOINT_TO_ENCODEC[checkpoint])

    processor = MusicgenProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    if pytorch_dump_folder is not None:
        Path(pytorch_dump_folder).mkdir(exist_ok=True)
        logger.info(f"Saving model {checkpoint} to {pytorch_dump_folder}")
        model.save_pretrained(pytorch_dump_folder)
        processor.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        model.push_to_hub(checkpoint)
        processor.push_to_hub(checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint",
        default="small",
        type=str,
        help="Checkpoint size of the Musicgen model you'd like to convert. Can be one of: small, medium, large.",
    )
    parser.add_argument(
        "--pytorch_dump_folder",
        default="/Users/sanchitgandhi/convert-musicgen",
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
    convert_musicgen_checkpoint(args.checkpoint, args.pytorch_dump_folder, args.push_to_hub)
