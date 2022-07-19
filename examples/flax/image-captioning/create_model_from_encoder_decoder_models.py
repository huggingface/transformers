#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
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
"""
Create a VisionEncoderDecoderModel instance from pretrained encoder/decoder models.

The cross-attention will be randomly initialized.
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    FlaxVisionEncoderDecoderModel,
    HfArgumentParser,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model will be written."},
    )
    encoder_model_name_or_path: str = field(
        metadata={
            "help": (
                "The encoder model checkpoint for weights initialization."
                "Don't set if you want to train an encoder model from scratch."
            )
        },
    )
    decoder_model_name_or_path: str = field(
        metadata={
            "help": (
                "The decoder model checkpoint for weights initialization."
                "Don't set if you want to train a decoder model from scratch."
            )
        },
    )
    encoder_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained encoder config name or path if not the same as encoder_model_name"}
    )
    decoder_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained decoder config name or path if not the same as decoder_model_name"}
    )


def main():
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_args_into_dataclasses()

    # Load pretrained model and tokenizer

    # Use explicit specified encoder config
    if model_args.encoder_config_name:
        encoder_config = AutoConfig.from_pretrained(model_args.encoder_config_name)
    # Use pretrained encoder model's config
    else:
        encoder_config = AutoConfig.from_pretrained(model_args.encoder_model_name_or_path)

    # Use explicit specified decoder config
    if model_args.decoder_config_name:
        decoder_config = AutoConfig.from_pretrained(model_args.decoder_config_name)
    # Use pretrained decoder model's config
    else:
        decoder_config = AutoConfig.from_pretrained(model_args.decoder_model_name_or_path)

    # necessary for `from_encoder_decoder_pretrained` when `decoder_config` is passed
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=model_args.encoder_model_name_or_path,
        decoder_pretrained_model_name_or_path=model_args.decoder_model_name_or_path,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )

    # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
    decoder_start_token_id = decoder_config.decoder_start_token_id
    pad_token_id = decoder_config.pad_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = decoder_config.bos_token_id
    if pad_token_id is None:
        pad_token_id = decoder_config.eos_token_id

    # This is necessary to make Flax's generate() work
    model.config.eos_token_id = decoder_config.eos_token_id
    model.config.decoder_start_token_id = decoder_start_token_id
    model.config.pad_token_id = pad_token_id

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.encoder_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.decoder_model_name_or_path)
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(model.config.pad_token_id)

    model.save_pretrained(model_args.output_dir)
    feature_extractor.save_pretrained(model_args.output_dir)
    tokenizer.save_pretrained(model_args.output_dir)


if __name__ == "__main__":
    main()
