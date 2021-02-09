# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert Wav2Vec2 checkpoint."""


import argparse

import fairseq
import torch

from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "wav2vec2.feature_projection.projection",
    "encoder.pos_conv.0": "wav2vec2.encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "wav2vec2.encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "wav2vec2.encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "wav2vec2.encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "wav2vec2.encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "wav2vec2.encoder.layers.*.layer_norm",
    "fc1": "wav2vec2.encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "wav2vec2.encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "wav2vec2.encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "wav2vec2.encoder.layer_norm",
    "w2v_model.layer_norm": "wav2vec2.feature_projection.layer_norm",
    "w2v_encoder.proj": "lm_head",
}


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    hf_shape = getattr(hf_pointer, weight_type).shape
    assert (
        hf_shape == value.shape
    ), f"Shape of hf {key + '.' + weight_type} is {hf_shape}, but should be {value.shape} for {full_name}"
    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    logger.info(f"{key + '.' + weight_type} was initialized from {full_name}.")


def recursively_load_weights(fairseq_model, hf_model):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    for name, value in fairseq_dict.items():
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                hf_model.wav2vec2.feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                if key in name:
                    is_used = True
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "weight" in name:
                        weight_type = "weight"
                    elif "bias" in name:
                        weight_type = "bias"
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    logger.info("Unused weights", unused_weights)


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            assert (
                value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape
            ), f"{full_name} has size {value.shape}, but {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            assert (
                value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape
            ), f"{full_name} has size {value.shape}, but {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            assert (
                value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape
            ), f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was found."
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            assert (
                value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape
            ), f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


@torch.no_grad()
def convert_wav2vec2_checkpoint(checkpoint_path, pytorch_dump_folder_path, dict_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    hf_wav2vec = Wav2Vec2ForCTC(Wav2Vec2Config())

    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides={"data": dict_path}
    )
    model = model[0].eval()

    recursively_load_weights(model, hf_wav2vec)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    args = parser.parse_args()
    convert_wav2vec2_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.dict_path)
