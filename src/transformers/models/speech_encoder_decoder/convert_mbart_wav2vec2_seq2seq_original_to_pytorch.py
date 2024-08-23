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
from torch import nn

from transformers import (
    MBart50Tokenizer,
    MBartConfig,
    MBartForCausalLM,
    SpeechEncoderDecoderConfig,
    SpeechEncoderDecoderModel,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
]


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    assert hf_shape == value.shape, (
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
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def recursively_load_weights_wav2vec2(fairseq_model, hf_model):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.feature_extractor
    adapter = hf_model.adapter

    for name, value in fairseq_dict.items():
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        elif any(x in name for x in ["adaptor", "w2v_encoder.proj.", "w2v_proj_ln."]):
            load_adapter(name, value, adapter, unused_weights)
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    if "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        weight_type = "weight"
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.bias.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            assert value.shape == feature_extractor.conv_layers[layer_id].conv.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape, (
                f"{full_name} has size {value.shape}, but {feature_extractor[layer_id].layer_norm.bias.data.shape} was"
                " found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            assert value.shape == feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape, (
                f"{full_name} has size {value.shape}, but"
                f" {feature_extractor[layer_id].layer_norm.weight.data.shape} was found."
            )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


def load_adapter(full_name, value, adapter, unused_weights):
    name = full_name.split("adaptor.")[-1]
    items = name.split(".")

    if items[1].isdigit():
        layer_id = int(items[1])
    else:
        layer_id = None

    if "adaptor" not in full_name:
        if "proj_ln" in full_name:
            # has to be layer norm
            if "bias" in name:
                assert (
                    value.shape == adapter.proj_layer_norm.bias.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj_layer_norm.bias.data.shape} was found."
                adapter.proj_layer_norm.bias.data = value
                logger.info(f"Adapter proj layer norm bias was initialized from {full_name}.")
            if "weight" in name:
                assert (
                    value.shape == adapter.proj_layer_norm.weight.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj_layer_norm.weight.data.shape} was found."
                adapter.proj_layer_norm.weight.data = value
        else:
            # has to be projection layer
            if "bias" in name:
                assert (
                    value.shape == adapter.proj.bias.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj.bias.data.shape} was found."
                adapter.proj.bias.data = value
                logger.info(f"Adapter proj layer bias was initialized from {full_name}.")
            if "weight" in name:
                assert (
                    value.shape == adapter.proj.weight.data.shape
                ), f"{full_name} has size {value.shape}, but {adapter.proj.weight.data.shape} was found."
                adapter.proj.weight.data = value
                logger.info(f"Adapter proj layer weight was initialized from {full_name}.")
    elif isinstance(layer_id, int):
        if "bias" in name:
            assert (
                value.shape == adapter.layers[layer_id].conv.bias.data.shape
            ), f"{full_name} has size {value.shape}, but {adapter.layers[layer_id].conv.bias.data.shape} was found."
            adapter.layers[layer_id].conv.bias.data = value
            logger.info(f"Adapter layer {layer_id} bias was initialized from {full_name}.")
        elif "weight" in name:
            assert (
                value.shape == adapter.layers[layer_id].conv.weight.data.shape
            ), f"{full_name} has size {value.shape}, but {adapter.layers[layer_id].conv.weight.data.shape} was found."
            adapter.layers[layer_id].conv.weight.data = value
            logger.info(f"Adapter layer {layer_id} bias was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    dict_path,
    config_yaml_path,
    encoder_config_path,
    decoder_config_path,
    add_adapter,
    adapter_kernel_size,
    adapter_stride,
    decoder_start_token_id,
    encoder_output_dim,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # load configs
    encoder_config = Wav2Vec2Config.from_pretrained(
        encoder_config_path,
        add_adapter=True,
        adapter_stride=adapter_stride,
        adapter_kernel_size=adapter_kernel_size,
        token_token=True,
        output_hidden_size=encoder_output_dim,
    )
    decoder_config = MBartConfig.from_pretrained(decoder_config_path)

    # load model
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides={
            "config_yaml": config_yaml_path,
            "data": "/".join(dict_path.split("/")[:-1]),
            "w2v_path": checkpoint_path,
            "load_pretrained_decoder_from": None,
        },
    )
    model = model[0].eval()

    # load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(encoder_config_path, token_token=True)

    # set weights for wav2vec2 encoder
    hf_encoder = Wav2Vec2Model(encoder_config)

    recursively_load_weights_wav2vec2(model.encoder, hf_encoder)

    # load decoder weights
    hf_decoder = MBartForCausalLM(decoder_config)
    missing_keys, unexpected_keys = hf_decoder.model.decoder.load_state_dict(model.decoder.state_dict(), strict=False)
    logger.warning(f"The following keys are missing when loading the decoder weights: {missing_keys}")
    logger.warning(f"The following keys are unexpected when loading the decoder weights: {unexpected_keys}")

    hf_wav2vec = SpeechEncoderDecoderModel(encoder=hf_encoder, decoder=hf_decoder)
    hf_wav2vec.config.tie_word_embeddings = False

    tokenizer = MBart50Tokenizer(dict_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    config = hf_wav2vec.config.to_dict()
    config["pad_token_id"] = tokenizer.pad_token_id
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.eos_token_id
    config["tokenizer_class"] = "mbart50"
    config["feature_extractor_type"] = "wav2vec2"

    config["decoder_start_token_id"] = tokenizer.eos_token_id
    config["forced_bos_token_id"] = 250004
    config["forced_eos_token_id"] = tokenizer.eos_token_id

    hf_wav2vec.config = SpeechEncoderDecoderConfig.from_dict(config)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_yaml_path", default=None, type=str, help="Path to yaml file of fine-tuned model")
    parser.add_argument(
        "--encoder_config_path",
        default="facebook/wav2vec2-xls-r-1b",
        type=str,
        help="Path to hf encoder wav2vec2 checkpoint config",
    )
    parser.add_argument(
        "--decoder_config_path",
        default="facebook/mbart-large-50-one-to-many-mmt",
        type=str,
        help="Path to hf decoder checkpoint config",
    )
    parser.add_argument("--add_adapter", default=True, type=bool, help="whethere to add model adapter layers")
    parser.add_argument("--adapter_stride", default=2, type=int, help="stride of adapter layers")
    parser.add_argument("--adapter_kernel_size", default=3, type=int, help="kernel size of adapter layers")
    parser.add_argument("--encoder_output_dim", default=1024, type=int, help="encoder output dim")
    parser.add_argument("--start_token_id", default=250004, type=int, help="`decoder_start_token_id` of model config")

    args = parser.parse_args()
    convert_wav2vec2_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.dict_path,
        args.config_yaml_path,
        encoder_config_path=args.encoder_config_path,
        decoder_config_path=args.decoder_config_path,
        add_adapter=args.add_adapter,
        adapter_kernel_size=args.adapter_kernel_size,
        adapter_stride=args.adapter_stride,
        decoder_start_token_id=args.start_token_id,
        encoder_output_dim=args.encoder_output_dim,
    )
