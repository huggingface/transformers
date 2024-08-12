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
"""Convert EnCodec checkpoints."""

import argparse

import torch

from transformers import (
    EncodecConfig,
    EncodecFeatureExtractor,
    EncodecModel,
    logging,
)


# checkpoints downloaded from:
# https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th
# https://huggingface.co/facebook/musicgen-small/resolve/main/compression_state_dict.bin
# https://dl.fbaipublicfiles.com/encodec/v0/encodec_48khz-7e698e3e.th


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.encodec")

MAPPING_QUANTIZER = {
    "quantizer.vq.layers.*._codebook.inited": "quantizer.layers.*.codebook.inited",
    "quantizer.vq.layers.*._codebook.cluster_size": "quantizer.layers.*.codebook.cluster_size",
    "quantizer.vq.layers.*._codebook.embed": "quantizer.layers.*.codebook.embed",
    "quantizer.vq.layers.*._codebook.embed_avg": "quantizer.layers.*.codebook.embed_avg",
}
MAPPING_ENCODER = {
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
MAPPING_ENCODER_48K = {
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
MAPPING_DECODER = {
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
MAPPING_DECODER_48K = {
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
MAPPING_24K = {
    **MAPPING_QUANTIZER,
    **MAPPING_ENCODER,
    **MAPPING_DECODER,
}
MAPPING_48K = {
    **MAPPING_QUANTIZER,
    **MAPPING_ENCODER,
    **MAPPING_ENCODER_48K,
    **MAPPING_DECODER,
    **MAPPING_DECODER_48K,
}
TOP_LEVEL_KEYS = []
IGNORE_KEYS = []


def set_recursively(hf_pointer, key, value, full_name, weight_type):
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


def should_ignore(name, ignore_keys):
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


def recursively_load_weights(orig_dict, hf_model, model_name):
    unused_weights = []

    if model_name in ["encodec_24khz", "encodec_32khz"]:
        MAPPING = MAPPING_24K
    elif model_name == "encodec_48khz":
        MAPPING = MAPPING_48K
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    for name, value in orig_dict.items():
        if should_ignore(name, IGNORE_KEYS):
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
                set_recursively(hf_model, mapped_key, value, name, weight_type)
            continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


@torch.no_grad()
def convert_checkpoint(
    model_name,
    checkpoint_path,
    pytorch_dump_folder_path,
    config_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = EncodecConfig.from_pretrained(config_path)
    else:
        config = EncodecConfig()

    if model_name == "encodec_24khz":
        pass  # config is already correct
    elif model_name == "encodec_32khz":
        config.upsampling_ratios = [8, 5, 4, 4]
        config.target_bandwidths = [2.2]
        config.num_filters = 64
        config.sampling_rate = 32_000
        config.codebook_size = 2048
        config.use_causal_conv = False
        config.normalize = False
        config.use_conv_shortcut = False
    elif model_name == "encodec_48khz":
        config.upsampling_ratios = [8, 5, 4, 2]
        config.target_bandwidths = [3.0, 6.0, 12.0, 24.0]
        config.sampling_rate = 48_000
        config.audio_channels = 2
        config.use_causal_conv = False
        config.norm_type = "time_group_norm"
        config.normalize = True
        config.chunk_length_s = 1.0
        config.overlap = 0.01
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = EncodecModel(config)

    feature_extractor = EncodecFeatureExtractor(
        feature_size=config.audio_channels,
        sampling_rate=config.sampling_rate,
        chunk_length_s=config.chunk_length_s,
        overlap=config.overlap,
    )
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    original_checkpoint = torch.load(checkpoint_path)
    if "best_state" in original_checkpoint:
        # we might have a training state saved, in which case discard the yaml results and just retain the weights
        original_checkpoint = original_checkpoint["best_state"]
    recursively_load_weights(original_checkpoint, model, model_name)
    model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        feature_extractor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="encodec_24khz",
        type=str,
        help="The model to convert. Should be one of 'encodec_24khz', 'encodec_32khz', 'encodec_48khz'.",
    )
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.model,
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
        args.push_to_hub,
    )
