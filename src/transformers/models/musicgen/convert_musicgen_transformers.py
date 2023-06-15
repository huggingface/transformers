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
from typing import Dict, OrderedDict

import torch
from audiocraft.models import MusicGen

from transformers import (
    MusicgenConfig,
    MusicgenDecoderConfig,
    MusicgenForConditionalGeneration,
    T5Config,
    EncodecFeatureExtractor, T5EncoderModel,
)
from transformers.utils import logging

from transformers.models.musicgen.modeling_encodec import EncodecConfig, EncodecModel


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


CHECKPOINT_TO_T5 = {
    "small": "t5-base",
}
CHECKPOINT_TO_ENCODEC = {
    "small": "facebook/encodec_32khz",
}

EXPECTED_MISSING_KEYS = ["decoder.model.decoder.embed_positions.weights"]

# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.MAPPING_QUANTIZER
ENCODEC_MAPPING_QUANTIZER = {
    "quantizer.vq.layers.*._codebook.inited": "quantizer.layers.*.codebook.inited",
    "quantizer.vq.layers.*._codebook.cluster_size": "quantizer.layers.*.codebook.cluster_size",
    "quantizer.vq.layers.*._codebook.embed": "quantizer.layers.*.codebook.embed",
    "quantizer.vq.layers.*._codebook.embed_avg": "quantizer.layers.*.codebook.embed_avg",
}
# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.MAPPING_ENCODER
ENCODEC_MAPPING_ENCODER = {
    "encoder.model.0.conv.conv": "encoder.layers.0.conv",
    "encoder.model.1.block.1.conv.conv": "encoder.layers.1.block.1.conv",
    "encoder.model.1.block.3.conv.conv": "encoder.layers.1.block.3.conv",
    "encoder.model.1.shortcut.conv.conv": "encoder.layers.1.shortcut.conv",
    "encoder.model.3.conv.conv": "encoder.layers.3.conv",
    "encoder.model.4.block.1.conv.conv": "encoder.layers.4.block.1.conv",
    "encoder.model.4.block.3.conv.conv": "encoder.layers.4.block.3.conv",
    "encoder.model.4.shortcut.conv.conv": "encoder.layers.4.shortcut.conv",
    "encoder.model.6.conv.conv": "encoder.layers.6.conv",
    "encoder.model.7.block.1.conv.conv": "encoder.layers.7.block.1.conv",
    "encoder.model.7.block.3.conv.conv": "encoder.layers.7.block.3.conv",
    "encoder.model.7.shortcut.conv.conv": "encoder.layers.7.shortcut.conv",
    "encoder.model.9.conv.conv": "encoder.layers.9.conv",
    "encoder.model.10.block.1.conv.conv": "encoder.layers.10.block.1.conv",
    "encoder.model.10.block.3.conv.conv": "encoder.layers.10.block.3.conv",
    "encoder.model.10.shortcut.conv.conv": "encoder.layers.10.shortcut.conv",
    "encoder.model.12.conv.conv": "encoder.layers.12.conv",
    "encoder.model.13.lstm": "encoder.layers.13.lstm",
    "encoder.model.15.conv.conv": "encoder.layers.15.conv",
}
# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.MAPPING_ENCODER_48K
ENCODEC_MAPPING_ENCODER_48K = {
    "encoder.model.0.conv.norm": "encoder.layers.0.norm",
    "encoder.model.1.block.1.conv.norm": "encoder.layers.1.block.1.norm",
    "encoder.model.1.block.3.conv.norm": "encoder.layers.1.block.3.norm",
    "encoder.model.1.shortcut.conv.norm": "encoder.layers.1.shortcut.norm",
    "encoder.model.3.conv.norm": "encoder.layers.3.norm",
    "encoder.model.4.block.1.conv.norm": "encoder.layers.4.block.1.norm",
    "encoder.model.4.block.3.conv.norm": "encoder.layers.4.block.3.norm",
    "encoder.model.4.shortcut.conv.norm": "encoder.layers.4.shortcut.norm",
    "encoder.model.6.conv.norm": "encoder.layers.6.norm",
    "encoder.model.7.block.1.conv.norm": "encoder.layers.7.block.1.norm",
    "encoder.model.7.block.3.conv.norm": "encoder.layers.7.block.3.norm",
    "encoder.model.7.shortcut.conv.norm": "encoder.layers.7.shortcut.norm",
    "encoder.model.9.conv.norm": "encoder.layers.9.norm",
    "encoder.model.10.block.1.conv.norm": "encoder.layers.10.block.1.norm",
    "encoder.model.10.block.3.conv.norm": "encoder.layers.10.block.3.norm",
    "encoder.model.10.shortcut.conv.norm": "encoder.layers.10.shortcut.norm",
    "encoder.model.12.conv.norm": "encoder.layers.12.norm",
    "encoder.model.15.conv.norm": "encoder.layers.15.norm",
}
# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.MAPPING_DECODER
ENCODEC_MAPPING_DECODER = {
    "decoder.model.0.conv.conv": "decoder.layers.0.conv",
    "decoder.model.1.lstm": "decoder.layers.1.lstm",
    "decoder.model.3.convtr.convtr": "decoder.layers.3.conv",
    "decoder.model.4.block.1.conv.conv": "decoder.layers.4.block.1.conv",
    "decoder.model.4.block.3.conv.conv": "decoder.layers.4.block.3.conv",
    "decoder.model.4.shortcut.conv.conv": "decoder.layers.4.shortcut.conv",
    "decoder.model.6.convtr.convtr": "decoder.layers.6.conv",
    "decoder.model.7.block.1.conv.conv": "decoder.layers.7.block.1.conv",
    "decoder.model.7.block.3.conv.conv": "decoder.layers.7.block.3.conv",
    "decoder.model.7.shortcut.conv.conv": "decoder.layers.7.shortcut.conv",
    "decoder.model.9.convtr.convtr": "decoder.layers.9.conv",
    "decoder.model.10.block.1.conv.conv": "decoder.layers.10.block.1.conv",
    "decoder.model.10.block.3.conv.conv": "decoder.layers.10.block.3.conv",
    "decoder.model.10.shortcut.conv.conv": "decoder.layers.10.shortcut.conv",
    "decoder.model.12.convtr.convtr": "decoder.layers.12.conv",
    "decoder.model.13.block.1.conv.conv": "decoder.layers.13.block.1.conv",
    "decoder.model.13.block.3.conv.conv": "decoder.layers.13.block.3.conv",
    "decoder.model.13.shortcut.conv.conv": "decoder.layers.13.shortcut.conv",
    "decoder.model.15.conv.conv": "decoder.layers.15.conv",
}
# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.MAPPING_DECODER_48K
ENCODEC_MAPPING_DECODER_48K = {
    "decoder.model.0.conv.norm": "decoder.layers.0.norm",
    "decoder.model.3.convtr.norm": "decoder.layers.3.norm",
    "decoder.model.4.block.1.conv.norm": "decoder.layers.4.block.1.norm",
    "decoder.model.4.block.3.conv.norm": "decoder.layers.4.block.3.norm",
    "decoder.model.4.shortcut.conv.norm": "decoder.layers.4.shortcut.norm",
    "decoder.model.6.convtr.norm": "decoder.layers.6.norm",
    "decoder.model.7.block.1.conv.norm": "decoder.layers.7.block.1.norm",
    "decoder.model.7.block.3.conv.norm": "decoder.layers.7.block.3.norm",
    "decoder.model.7.shortcut.conv.norm": "decoder.layers.7.shortcut.norm",
    "decoder.model.9.convtr.norm": "decoder.layers.9.norm",
    "decoder.model.10.block.1.conv.norm": "decoder.layers.10.block.1.norm",
    "decoder.model.10.block.3.conv.norm": "decoder.layers.10.block.3.norm",
    "decoder.model.10.shortcut.conv.norm": "decoder.layers.10.shortcut.norm",
    "decoder.model.12.convtr.norm": "decoder.layers.12.norm",
    "decoder.model.13.block.1.conv.norm": "decoder.layers.13.block.1.norm",
    "decoder.model.13.block.3.conv.norm": "decoder.layers.13.block.3.norm",
    "decoder.model.13.shortcut.conv.norm": "decoder.layers.13.shortcut.norm",
    "decoder.model.15.conv.norm": "decoder.layers.15.norm",
}
# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.MAPPING_24K with MAPPING->ENCODEC_MAPPING
ENCODEC_MAPPING_24K = {
    **ENCODEC_MAPPING_QUANTIZER,
    **ENCODEC_MAPPING_ENCODER,
    **ENCODEC_MAPPING_DECODER,
}
# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.MAPPING_48K with MAPPING->ENCODEC_MAPPING
ENCODEC_MAPPING_48K = {
    **ENCODEC_MAPPING_QUANTIZER,
    **ENCODEC_MAPPING_ENCODER,
    **ENCODEC_MAPPING_ENCODER_48K,
    **ENCODEC_MAPPING_DECODER,
    **ENCODEC_MAPPING_DECODER_48K,
}
TOP_LEVEL_KEYS = []
IGNORE_KEYS = []

# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.set_recursively
def set_encodec_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    elif weight_type == "weight_ih_l0":
        hf_pointer.weight_ih_l0.data = value
    elif weight_type == "weight_hh_l0":
        hf_pointer.weight_hh_l0.data = value
    elif weight_type == "bias_ih_l0":
        hf_pointer.bias_ih_l0.data = value
    elif weight_type == "bias_hh_l0":
        hf_pointer.bias_hh_l0.data = value
    elif weight_type == "weight_ih_l1":
        hf_pointer.weight_ih_l1.data = value
    elif weight_type == "weight_hh_l1":
        hf_pointer.weight_hh_l1.data = value
    elif weight_type == "bias_ih_l1":
        hf_pointer.bias_ih_l1.data = value
    elif weight_type == "bias_hh_l1":
        hf_pointer.bias_hh_l1.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")

# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.should_ignore
def should_ignore_encodec(name, ignore_keys):
    for key in ignore_keys:
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        elif key in name:
            return True
    return False


# Copied from transformers.models.encodec.convert_encodec_checkpoint_to_pytorch.recursively_load_weights with set_recursively->set_encodec_recursively, should_ignore->should_ignore_encodec, MAPPING_->ENCODEC_MAPPING_
def recursively_load_encodec_weights(orig_dict, hf_model, model_name="encodec_32khz"):
    unused_weights = []

    if model_name == "encodec_24khz" or "encodec_32khz":
        MAPPING = ENCODEC_MAPPING_24K
    elif model_name == "encodec_48khz":
        MAPPING = ENCODEC_MAPPING_48K
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for name, value in orig_dict.items():
        if should_ignore_encodec(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        is_used = False
        for key, mapped_key in MAPPING.items():
            if "*" in key:
                prefix, suffix = key.split(".*.")
                if prefix in name and suffix in name:
                    key = suffix

            if key in name:
                # HACK otherwise .embed gets initialized with .embed_avg too
                if key.endswith("embed") and name.endswith("embed_avg"):
                    continue

                is_used = True
                if "*" in mapped_key:
                    layer_index = name.split(key)[0].split(".")[-2]
                    mapped_key = mapped_key.replace("*", layer_index)
                if "weight_g" in name:
                    weight_type = "weight_g"
                elif "weight_v" in name:
                    weight_type = "weight_v"
                elif "weight_ih_l0" in name:
                    weight_type = "weight_ih_l0"
                elif "weight_hh_l0" in name:
                    weight_type = "weight_hh_l0"
                elif "bias_ih_l0" in name:
                    weight_type = "bias_ih_l0"
                elif "bias_hh_l0" in name:
                    weight_type = "bias_hh_l0"
                elif "weight_ih_l1" in name:
                    weight_type = "weight_ih_l1"
                elif "weight_hh_l1" in name:
                    weight_type = "weight_hh_l1"
                elif "bias_ih_l1" in name:
                    weight_type = "bias_ih_l1"
                elif "bias_hh_l1" in name:
                    weight_type = "bias_hh_l1"
                elif "bias" in name:
                    weight_type = "bias"
                elif "weight" in name:
                    weight_type = "weight"
                elif "running_mean" in name:
                    weight_type = "running_mean"
                elif "running_var" in name:
                    weight_type = "running_var"
                elif "num_batches_tracked" in name:
                    weight_type = "num_batches_tracked"
                else:
                    weight_type = None
                set_encodec_recursively(hf_model, mapped_key, value, name, weight_type)
            continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")

def convert_encodec_checkpoint(
    original_checkpoint,
    config,
):
    """
    Copy/paste/tweak Encodec's weights to transformers design.
    """

    model = EncodecModel(config)

    feature_extractor = EncodecFeatureExtractor(
        feature_size=config.audio_channels,
        sampling_rate=config.sampling_rate,
        chunk_length_s=config.chunk_length_s,
        overlap=config.overlap,
    )

    if "best_state" in original_checkpoint:
        # we might have a training state saved, in which case discard the yaml results and just retain the weights
        original_checkpoint = original_checkpoint["best_state"]

    recursively_load_encodec_weights(original_checkpoint, model)
    return model, feature_extractor

def rename_decoder_keys(name):
    if "condition_provider.conditioners.description.output_proj" in name:
        name = name.replace("condition_provider.conditioners.description.output_proj", "model.encoder_projection")
    if "emb" in name:
        name = name.replace("emb", "decoder.model.decoder.embed_tokens")
    if "transformer" in name:
        name = name.replace("transformer", "decoder.model.decoder")
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
        name = name.replace("out_norm", "decoder.model.decoder.layer_norm")
    if "linears" in name:
        name = name.replace("linears", "decoder.lm_heads")
    return name


def rename_decoder_state_dict(state_dict: OrderedDict, d_model: int) -> Dict:
    """Function that takes the fairseq Musicgen state dict and renames it according to the HF
    module names. It further partitions the state dict into three: the decoder state dict (state_dict), the state dict
    for the LM head, and the state dict for the encoder projection."""
    keys = list(state_dict.keys())
    for key in keys:
        val = state_dict.pop(key)
        key = rename_decoder_keys(key)
        if "in_proj_weight" in key:
            # split fused qkv proj
            state_dict[key.replace("in_proj_weight", "q_proj.weight")] = val[:d_model, :]
            state_dict[key.replace("in_proj_weight", "k_proj.weight")] = val[d_model : 2 * d_model, :]
            state_dict[key.replace("in_proj_weight", "v_proj.weight")] = val[-d_model:, :]
        else:
            state_dict[key] = val
    return state_dict


def config_from_checkpoint(checkpoint: str) -> MusicgenConfig:
    if checkpoint == "small":
        d_model = 1024
        ffn_dim = d_model * 4
        num_layers = 24
        num_codebooks = 4
    config = MusicgenDecoderConfig(
        d_model=d_model, ffn_dim=ffn_dim, num_layers=num_layers, num_codebooks=num_codebooks, tie_word_embeddings=False,
    )
    return config

@torch.no_grad()
def convert_musicgen_checkpoint(checkpoint, pytorch_dump_folder=None, push_to_hub=False, device="cpu"):
    fairseq_model = MusicGen.get_pretrained(checkpoint, device=device)
    config = config_from_checkpoint(checkpoint)

    # TODO(SG): remove debugging statement
    fairseq_model.lm.transformer.layers = fairseq_model.lm.transformer.layers[:2]
    config.decoder.num_layers = 2

    # the t5 encoder state dict is 'hidden', so we retrieve using this hack
    text_encoder_state_dict = fairseq_model.lm.condition_provider.conditioners.description.__dict__["t5"].state_dict()
    audio_encoder_state_dict = fairseq_model.compression_model.state_dict()
    decoder_state_dict = fairseq_model.lm.state_dict()

    # convert weight names for audio encoder + musicgen decoder

    text_encoder = T5EncoderModel.from_pretrained(CHECKPOINT_TO_T5[checkpoint])
    audio_encoder_config = EncodecConfig(upsampling_ratios=[8, 5, 4, 4], target_bandwidths=[2.2], num_filters=64, sampling_rate=32_000,
                          codebook_size=2048, use_causal_conv=False, normalize=False, use_conv_shortcut=True)


    audio_encoder, feature_extractor = convert_encodec_checkpoint(audio_encoder_state_dict, config.audio_encoder)
    decoder_state_dict = rename_decoder_state_dict(decoder_state_dict, d_model=config.decoder.d_model)

    model = MusicgenForConditionalGeneration(config).eval()

    # load the text encoder model (one-to-one the same as T5EncoderModel)
    model.text_encoder.load_state_dict(text_encoder_state_dict)
    # load the audio encoder model (one-to-one the same as EncodecModel)
    model.audio_encoder = audio_encoder
    # load all other weights (encoder proj + decoder + lm heads) - expect that we'll be missing all text/audio encoder weights
    missing_keys, unexpected_keys = model.load_state_dict(decoder_state_dict, strict=False)

    for key in missing_keys.copy():
        if key.startswith(("text_encoder", "audio_encoder")) or key in EXPECTED_MISSING_KEYS:
            missing_keys.remove(key)

    if len(missing_keys) > 0:
        raise ValueError(f"Missing key(s) in state_dict: {missing_keys}")

    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected key(s) in state_dict: {unexpected_keys}")

    # check we can do a forward pass
    input_ids = torch.arange(0, 8, dtype=torch.long).reshape(2, -1)
    decoder_input_ids = input_ids.reshape(2, 4, -1)

    with torch.no_grad():
        logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits

    if logits.shape != (2, 1, 4, 2048):
        raise ValueError("Incorrect shape for logits")

    EXPECTED_SLICE = [-3.3180, -1.5341, -3.2796, -2.0692, -2.1036, -2.3339, 0.6623, -6.3549, 0.8477, -2.0866]

    if torch.max(torch.abs(logits[0, 0, 0, :10] - torch.tensor(EXPECTED_SLICE))) > 1e-4:
        raise ValueError("Logits exceed tolerance threshold")

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
