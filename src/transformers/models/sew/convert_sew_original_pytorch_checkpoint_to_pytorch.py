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
"""Convert SEW checkpoint."""


import argparse
import json
import os

import fairseq
import torch
from fairseq.data import Dictionary

# Register SEW's fairseq modules
from sew_asapp import tasks  # noqa: F401

from transformers import (
    SEWConfig,
    SEWForCTC,
    SEWModel,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.k_proj": "encoder.layers.*.attention.k_proj",
    "self_attn.v_proj": "encoder.layers.*.attention.v_proj",
    "self_attn.q_proj": "encoder.layers.*.attention.q_proj",
    "self_attn.out_proj": "encoder.layers.*.attention.out_proj",
    "self_attn_layer_norm": "encoder.layers.*.layer_norm",
    "fc1": "encoder.layers.*.feed_forward.intermediate_dense",
    "fc2": "encoder.layers.*.feed_forward.output_dense",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.upsample.0": "encoder.upsample.projection",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "layer_norm",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}


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


def recursively_load_weights(fairseq_model, hf_model, is_finetuned):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.sew.feature_extractor if is_finetuned else hf_model.feature_extractor

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
        else:
            for key, mapped_key in MAPPING.items():
                mapped_key = "sew." + mapped_key if (is_finetuned and mapped_key != "lm_head") else mapped_key

                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
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


def convert_config(model, is_finetuned):
    config = SEWConfig()
    if is_finetuned:
        fs_config = model.w2v_encoder.w2v_model.cfg
    else:
        fs_config = model.cfg

    config.conv_bias = fs_config.conv_bias
    conv_layers = eval(fs_config.conv_feature_layers)
    config.conv_dim = [x[0] for x in conv_layers]
    config.conv_kernel = [x[1] for x in conv_layers]
    config.conv_stride = [x[2] for x in conv_layers]
    config.feat_extract_activation = "gelu"
    config.feat_extract_norm = "layer" if fs_config.extractor_mode == "layer_norm" else "group"
    config.final_dropout = 0.0
    config.hidden_act = fs_config.activation_fn.name
    config.hidden_size = fs_config.encoder_embed_dim
    config.initializer_range = 0.02
    config.intermediate_size = fs_config.encoder_ffn_embed_dim
    config.layer_norm_eps = 1e-5
    config.layerdrop = fs_config.encoder_layerdrop
    config.num_attention_heads = fs_config.encoder_attention_heads
    config.num_conv_pos_embedding_groups = fs_config.conv_pos_groups
    config.num_conv_pos_embeddings = fs_config.conv_pos
    config.num_feat_extract_layers = len(conv_layers)
    config.num_hidden_layers = fs_config.encoder_layers
    config.squeeze_factor = fs_config.squeeze_factor

    # take care of any params that are overridden by the Wav2VecCtc model
    if is_finetuned:
        fs_config = model.cfg
        config.final_dropout = fs_config.final_dropout
        config.layerdrop = fs_config.layerdrop
    config.activation_dropout = fs_config.activation_dropout
    config.apply_spec_augment = fs_config.mask_prob > 0 or fs_config.mask_channel_prob > 0
    config.attention_dropout = fs_config.attention_dropout
    config.feat_proj_dropout = fs_config.dropout_input
    config.hidden_dropout = fs_config.dropout
    config.mask_feature_length = fs_config.mask_channel_length
    config.mask_feature_prob = fs_config.mask_channel_prob
    config.mask_time_length = fs_config.mask_length
    config.mask_time_prob = fs_config.mask_prob

    config.feature_extractor_type = "Wav2Vec2FeatureExtractor"
    config.tokenizer_class = "Wav2Vec2CTCTokenizer"

    return config


@torch.no_grad()
def convert_sew_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None, is_finetuned=True
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])

    if config_path is not None:
        config = SEWConfig.from_pretrained(config_path)
    else:
        config = convert_config(model[0], is_finetuned)
    model = model[0].eval()

    return_attention_mask = True if config.feat_extract_norm == "layer" else False
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=return_attention_mask,
    )

    if is_finetuned:
        if dict_path:
            target_dict = Dictionary.load(dict_path)

            # important change bos & pad token id since CTC symbol is <pad> and
            # not <s> as in fairseq
            target_dict.indices[target_dict.bos_word] = target_dict.pad_index
            target_dict.indices[target_dict.pad_word] = target_dict.bos_index
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(target_dict.indices, vocab_handle)
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.save_pretrained(pytorch_dump_folder_path)

        hf_model = SEWForCTC(config)
    else:
        hf_model = SEWModel(config)
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    recursively_load_weights(model, hf_model, is_finetuned)

    hf_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--is_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    args = parser.parse_args()
    convert_sew_checkpoint(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path, args.is_finetuned
    )
