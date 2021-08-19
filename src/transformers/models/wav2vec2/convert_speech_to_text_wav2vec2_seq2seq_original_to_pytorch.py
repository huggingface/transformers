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
import json
import os

import fairseq
import torch
from fairseq.data import Dictionary
from torch import nn

from transformers import (
    EncoderDecoderConfig,
    Speech2TextConfig,
    Speech2TextForCausalLM,
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

    assert (
        hf_shape == value.shape
    ), f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be {value.shape} for {full_name}"

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

    # if encoder has different dim to decoder -> use proj_weight
    proj_weight = None

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
        elif name.split(".")[0] == "proj":
            proj_weight = fairseq_model.proj
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                #                mapped_key = "wav2vec2." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
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
                        # TODO: don't match quantizer.weight_proj
                        weight_type = "weight"
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")

    return proj_weight


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


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_wav2vec2_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    encoder_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-lv60")
    decoder_config = Speech2TextConfig.from_pretrained(
        "facebook/s2t-small-mustc-en-fr-st", vocab_size=10224, decoder_layers=7
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)

    #    target_dict = Dictionary.load(dict_path)
    #
    # important change bos & pad token id since CTC symbol is <pad> and
    # not <s> as in fairseq
    #            config.bos_token_id = target_dict.pad_index
    #            config.pad_token_id = target_dict.bos_index
    #            config.eos_token_id = target_dict.eos_index
    #            config.vocab_size = len(target_dict.symbols)
    #            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
    #            if not os.path.isdir(pytorch_dump_folder_path):
    #                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
    #                return
    #            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    #            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
    #                json.dump(target_dict.indices, vocab_handle)
    #            tokenizer = Wav2Vec2CTCTokenizer(
    #                vocab_path,
    #                unk_token=target_dict.unk_word,
    #                pad_token=target_dict.pad_word,
    #                bos_token=target_dict.bos_word,
    #                eos_token=target_dict.eos_word,
    #                word_delimiter_token="|",
    #                do_lower_case=False,
    #            )
    #            return_attention_mask = True if config.feat_extract_norm == "layer" else False
    #            feature_extractor = Wav2Vec2FeatureExtractor(
    #                feature_size=1,
    #                sampling_rate=16000,
    #                padding_value=0,
    #                do_normalize=True,
    #                return_attention_mask=return_attention_mask,
    #            )
    #            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    #            processor.save_pretrained(pytorch_dump_folder_path)

    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
    )
    model = model[0].eval()

    # set weights for wav2vec2 encoder
    hf_encoder = Wav2Vec2Model(encoder_config)
    projection_layer = recursively_load_weights_wav2vec2(model.encoder, hf_encoder)

    hf_decoder = Speech2TextForCausalLM(decoder_config)
    missing_keys, unexpected_keys = hf_decoder.model.decoder.load_state_dict(model.decoder.state_dict(), strict=False)

    # set output linear layer
    unexpected_keys.remove("embed_out")
    hf_decoder.lm_head = model.decoder.embed_out

    # layer norm is init to identity matrix so leaving it is fine
    logger.warning(f"The following keys are missing when loading the decoder weights: {missing_keys}")
    logger.warning(f"The following keys are unexpected when loading the decoder weights: {unexpected_keys}")

    hf_wav2vec = SpeechEncoderDecoderModel(encoder=hf_encoder, decoder=hf_decoder)

    # add projection layer
    hf_wav2vec.enc_to_dec_proj.weight = nn.Parameter(projection_layer.weight)
    hf_wav2vec.enc_to_dec_proj.bias = nn.Parameter(projection_layer.bias)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()
    convert_wav2vec2_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path)
