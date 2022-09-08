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
    SpeechT5ForCTC,
    # SpeechT5ForPreTraining,
    # SpeechT5Processor,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_key(name):
    if "speech_encoder_prenet.feature_extractor." in name:
        name = name.replace("speech_encoder_prenet.feature_extractor.", "speecht5.speech_encoder_prenet.feature_encoder.")
        name = name.replace(".0.0.", ".0.conv.")
        name = name.replace(".0.2.weight", ".0.layer_norm.weight")
        name = name.replace(".0.2.bias", ".0.layer_norm.bias")
        name = name.replace(".0.weight", ".conv.weight")

    if "speech_encoder_prenet." in name:
        name = name.replace("speech_encoder_prenet.post_extract_proj.", "speecht5.speech_encoder_prenet.feature_projection.projection.")
        name = name.replace("speech_encoder_prenet.layer_norm.", "speecht5.speech_encoder_prenet.feature_projection.layer_norm.")
        name = name.replace("speech_encoder_prenet.pos_conv.0.", "speecht5.speech_encoder_prenet.pos_conv_embed.conv.")

    # Can ignore the following keys:
    # "speech_encoder_prenet.embed_positions._float_tensor"
    # "speech_encoder_prenet.mask_emb"

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            # key_split = key.split(".")
            # layer_num = int(key_split[0][6:]) - 1
            # transformer_num = int(key_split[3])
            # layer = model.get_submodule(f"{model_prefix}encoder.layer.{layer_num}")
            # dim = layer.transformer.layer[transformer_num].attention.attention.all_head_size
            # prefix = (
            #     f"{model_prefix}encoder.layer.{layer_num}.transformer.layer.{transformer_num}.attention.attention."
            # )
            # if "weight" in key:
            #     orig_state_dict[prefix + "query.weight"] = val[:dim, :]
            #     orig_state_dict[prefix + "key.weight"] = val[dim : dim * 2, :]
            #     orig_state_dict[prefix + "value.weight"] = val[-dim:, :]
            # else:
            #     orig_state_dict[prefix + "query.bias"] = val[:dim]
            #     orig_state_dict[prefix + "key.bias"] = val[dim : dim * 2]
            #     orig_state_dict[prefix + "value.bias"] = val[-dim:]
            pass
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


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

        model = SpeechT5ForCTC(config)
    # elif task == "pretrain":
    #     _model = SpeechT5ForPreTraining(config)
    else:
        raise ValueError(f"Unknown model name: {task}")

    fairseq_checkpoint = torch.load(checkpoint_path)
    new_state_dict = convert_state_dict(fairseq_checkpoint["model"], model)
    model.load_state_dict(new_state_dict, strict=False)  # TODO: remove strict=False for final model

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
