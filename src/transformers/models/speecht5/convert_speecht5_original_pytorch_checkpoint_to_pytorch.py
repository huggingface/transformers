# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert SpeechT5 checkpoint."""


import argparse
# import json
# import os

import torch
# from fairseq.data import Dictionary

from transformers import (
    SpeechT5Config,
    # SpeechT5CTCTokenizer,
    # Wav2Vec2FeatureExtractor,
    SpeechT5ForConditionalGeneration,
    SpeechT5ForCTC,
    # SpeechT5ForPreTraining,
    # SpeechT5Processor,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.speecht5")

MAPPING = {
    "speech_encoder_prenet.layer_norm": "speecht5.encoder.speech_encoder_prenet.feature_projection.layer_norm",
    "speech_encoder_prenet.post_extract_proj": "speecht5.encoder.speech_encoder_prenet.feature_projection.projection",
    "speech_encoder_prenet.pos_conv.0": "speecht5.encoder.speech_encoder_prenet.pos_conv_embed.conv",
    "speech_encoder_prenet.mask_emb": "speecht5.encoder.speech_encoder_prenet.masked_spec_embed",
    "encoder.layers.*.self_attn.k_proj": "speecht5.encoder.layers.*.attention.k_proj",
    "encoder.layers.*.self_attn.v_proj": "speecht5.encoder.layers.*.attention.v_proj",
    "encoder.layers.*.self_attn.q_proj": "speecht5.encoder.layers.*.attention.q_proj",
    "encoder.layers.*.self_attn.out_proj": "speecht5.encoder.layers.*.attention.out_proj",
    "encoder.layers.*.self_attn_layer_norm": "speecht5.encoder.layers.*.layer_norm",
    "encoder.layers.*.fc1": "speecht5.encoder.layers.*.feed_forward.intermediate_dense",
    "encoder.layers.*.fc2": "speecht5.encoder.layers.*.feed_forward.output_dense",
    "encoder.layers.*.final_layer_norm": "speecht5.encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "speecht5.encoder.layer_norm",
    "encoder.pos_emb.pe_k": "speecht5.encoder.pos_emb.pe_k",
    "encoder.proj": "speecht5.encoder.ctc_proj",
    # "quantizer.weight_proj": "quantizer.weight_proj",
    # "quantizer.vars": "quantizer.codevectors",
    # "project_q": "project_q",
    # "final_proj": "project_hid",
    # "w2v_encoder.proj": "lm_head",
}
TOP_LEVEL_KEYS = [
    # "lm_head",
    # "quantizer.weight_proj",
    # "quantizer.codevectors",
    # "project_q",
    # "project_hid",
    #"speech_encoder_prenet",
]
IGNORE_KEYS = [
    "encoder.version",
    "encoder.layers.*.norm_k.weight",
    "encoder.layers.*.norm_k.bias",
    "speech_encoder_prenet.embed_positions._float_tensor",
    "decoder.version",
]


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
    else:
        hf_pointer.data = value

    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")


def should_ignore(name):
    for key in IGNORE_KEYS:
        if "*" in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        elif key in name:
            return True
    return False


def recursively_load_weights(fairseq_dict, hf_model):
    unused_weights = []

    for name, value in fairseq_dict.items():
        if should_ignore(name):
            logger.info(f"{name} was ignored")
            continue

        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                hf_model.speecht5.encoder.speech_encoder_prenet.feature_encoder,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                # mapped_key = "speecht5." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key

                if "*" in key:
                    prefix, suffix = key.split(".*.")
                    if prefix in name and suffix in name:
                        key = suffix

                # if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                if key in name:
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


def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


@torch.no_grad()
def convert_speecht5_checkpoint(
    task, checkpoint_path, pytorch_dump_folder_path, config_path=None, dict_path=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = SpeechT5Config.from_pretrained(config_path)
    else:
        config = SpeechT5Config()

    if task == "s2t":
    #     if dict_path:
    #         target_dict = Dictionary.load(dict_path)

    #         # important change bos & pad token id since CTC symbol is <pad> and
    #         # not <s> as in fairseq
    #         config.bos_token_id = target_dict.pad_index
    #         config.pad_token_id = target_dict.bos_index
    #         config.eos_token_id = target_dict.eos_index
    #         config.vocab_size = len(target_dict.symbols)
    #         vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
    #         if not os.path.isdir(pytorch_dump_folder_path):
    #             logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
    #             return
    #         os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    #         vocab_dict = target_dict.indices

    #         # fairseq has the <pad> and <s> switched
    #         vocab_dict["<pad>"] = 0
    #         vocab_dict["<s>"] = 1
    #         with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
    #             json.dump(vocab_dict, vocab_handle)
    #         tokenizer = SpeechT5CTCTokenizer(
    #             vocab_path,
    #             unk_token=target_dict.unk_word,
    #             pad_token=target_dict.pad_word,
    #             bos_token=target_dict.bos_word,
    #             eos_token=target_dict.eos_word,
    #             word_delimiter_token="|",
    #             do_lower_case=False,
    #         )
    #         return_attention_mask = True if config.feat_extract_norm == "layer" else False
    #         feature_extractor = Wav2Vec2FeatureExtractor(
    #             feature_size=1,
    #             sampling_rate=16000,
    #             padding_value=0,
    #             do_normalize=True,
    #             return_attention_mask=return_attention_mask,
    #         )
    #         processor = SpeechT5Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    #         processor.save_pretrained(pytorch_dump_folder_path)

        model = SpeechT5ForConditionalGeneration(config)
    # elif task == "pretrain":
    #     _model = SpeechT5ForPreTraining(config)
    else:
        raise ValueError(f"Unknown model name: {task}")

    fairseq_checkpoint = torch.load(checkpoint_path)
    recursively_load_weights(fairseq_checkpoint["model"], model)

    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="s2t",
        type=str,
        help=(
            "Type of the SpeechT5 model you'd like to convert. Should be one of 's2t', 'pretrain'."
        ),
    )
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_speecht5_checkpoint(
        args.task, args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path, args.dict_path,
    )
