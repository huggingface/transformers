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
from torch import nn

from transformers import (
    Speech2Text2Config,
    Speech2Text2ForCausalLM,
    Speech2Text2Tokenizer,
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


def create_vocab_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        words = [line.split(" ")[0] for line in lines]

    num_words = len(words)

    vocab_dict = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
    }

    vocab_dict.update({k: v for k, v in zip(words, range(4, num_words + 4))})
    return vocab_dict


@torch.no_grad()
def convert_wav2vec2_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    dict_path,
    encoder_config_path,
    decoder_config_path,
    vocab_size,
    num_decoder_layers,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    encoder_config = Wav2Vec2Config.from_pretrained(encoder_config_path)
    decoder_config = Speech2Text2Config.from_pretrained(
        decoder_config_path, vocab_size=vocab_size, decoder_layers=num_decoder_layers, do_stable_layer_norm=True
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0,
        do_normalize=True,
        return_attention_mask=True,
    )

    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
    )
    model = model[0].eval()

    # set weights for wav2vec2 encoder
    hf_encoder = Wav2Vec2Model(encoder_config)
    projection_layer = recursively_load_weights_wav2vec2(model.encoder, hf_encoder)

    hf_decoder = Speech2Text2ForCausalLM(decoder_config)
    missing_keys, unexpected_keys = hf_decoder.model.decoder.load_state_dict(model.decoder.state_dict(), strict=False)

    # set output linear layer
    unexpected_keys.remove("embed_out")
    hf_decoder.lm_head.weight = nn.Parameter(model.decoder.embed_out.detach())

    # layer norm is init to identity matrix so leaving it is fine
    logger.warning(f"The following keys are missing when loading the decoder weights: {missing_keys}")
    logger.warning(f"The following keys are unexpected when loading the decoder weights: {unexpected_keys}")

    hf_wav2vec = SpeechEncoderDecoderModel(encoder=hf_encoder, decoder=hf_decoder)
    hf_wav2vec.config.tie_word_embeddings = False

    # add projection layer
    hf_wav2vec.enc_to_dec_proj.weight = nn.Parameter(projection_layer.weight)
    hf_wav2vec.enc_to_dec_proj.bias = nn.Parameter(projection_layer.bias)

    vocab_dict = create_vocab_dict(dict_path)

    with open(os.path.join(pytorch_dump_folder_path, "vocab.json"), "w") as fp:
        json.dump(vocab_dict, fp)

    tokenizer = Speech2Text2Tokenizer(os.path.join(pytorch_dump_folder_path, "vocab.json"))
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    config = hf_wav2vec.config.to_dict()
    config["pad_token_id"] = tokenizer.pad_token_id
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.eos_token_id
    config["tokenizer_class"] = "speech_to_text_2"
    config["feature_extractor_type"] = "wav2vec2"

    hf_wav2vec.config = SpeechEncoderDecoderConfig.from_dict(config)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument(
        "--encoder_config_path",
        default="facebook/wav2vec2-large-lv60",
        type=str,
        help="Path to hf encoder wav2vec2 checkpoint config",
    )
    parser.add_argument(
        "--decoder_config_path",
        default="facebook/s2t-small-mustc-en-fr-st",
        type=str,
        help="Path to hf decoder s2t checkpoint config",
    )
    parser.add_argument("--vocab_size", default=10224, type=int, help="Vocab size of decoder")
    parser.add_argument("--num_decoder_layers", default=7, type=int, help="Number of decoder layers")

    args = parser.parse_args()
    convert_wav2vec2_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.dict_path,
        encoder_config_path=args.encoder_config_path,
        decoder_config_path=args.decoder_config_path,
        vocab_size=args.vocab_size,
        num_decoder_layers=args.num_decoder_layers,
    )
