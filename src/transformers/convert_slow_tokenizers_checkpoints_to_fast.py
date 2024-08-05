# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert slow tokenizers checkpoints in fast (serialization format of the `tokenizers` library)"""

import argparse
import os

import transformers

from .convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from .utils import logging


logging.set_verbosity_info()

logger = logging.get_logger(__name__)


TOKENIZER_CLASSES = {name: getattr(transformers, name + "Fast") for name in SLOW_TO_FAST_CONVERTERS}


def convert_slow_checkpoint_to_fast(tokenizer_name, checkpoint_name, dump_path, force_download):
    if tokenizer_name is not None and tokenizer_name not in TOKENIZER_CLASSES:
        raise ValueError(f"Unrecognized tokenizer name, should be one of {list(TOKENIZER_CLASSES.keys())}.")

    if tokenizer_name is None:
        tokenizer_names = TOKENIZER_CLASSES
    else:
        tokenizer_names = {tokenizer_name: getattr(transformers, tokenizer_name + "Fast")}

    logger.info(f"Loading tokenizer classes: {tokenizer_names}")

    for tokenizer_name in tokenizer_names:
        tokenizer_class = TOKENIZER_CLASSES[tokenizer_name]

        add_prefix = True
        if checkpoint_name is None:
            checkpoint_names = list(tokenizer_class.max_model_input_sizes.keys())
        else:
            checkpoint_names = [checkpoint_name]

        logger.info(f"For tokenizer {tokenizer_class.__class__.__name__} loading checkpoints: {checkpoint_names}")

        for checkpoint in checkpoint_names:
            logger.info(f"Loading {tokenizer_class.__class__.__name__} {checkpoint}")

            # Load tokenizer
            tokenizer = tokenizer_class.from_pretrained(checkpoint, force_download=force_download)

            # Save fast tokenizer
            logger.info(f"Save fast tokenizer to {dump_path} with prefix {checkpoint} add_prefix {add_prefix}")

            # For organization names we create sub-directories
            if "/" in checkpoint:
                checkpoint_directory, checkpoint_prefix_name = checkpoint.split("/")
                dump_path_full = os.path.join(dump_path, checkpoint_directory)
            elif add_prefix:
                checkpoint_prefix_name = checkpoint
                dump_path_full = dump_path
            else:
                checkpoint_prefix_name = None
                dump_path_full = dump_path

            logger.info(f"=> {dump_path_full} with prefix {checkpoint_prefix_name}, add_prefix {add_prefix}")

            if checkpoint in list(tokenizer.pretrained_vocab_files_map.values())[0]:
                file_path = list(tokenizer.pretrained_vocab_files_map.values())[0][checkpoint]
                next_char = file_path.split(checkpoint)[-1][0]
                if next_char == "/":
                    dump_path_full = os.path.join(dump_path_full, checkpoint_prefix_name)
                    checkpoint_prefix_name = None

                logger.info(f"=> {dump_path_full} with prefix {checkpoint_prefix_name}, add_prefix {add_prefix}")

            file_names = tokenizer.save_pretrained(
                dump_path_full, legacy_format=False, filename_prefix=checkpoint_prefix_name
            )
            logger.info(f"=> File names {file_names}")

            for file_name in file_names:
                if not file_name.endswith("tokenizer.json"):
                    os.remove(file_name)
                    logger.info(f"=> removing {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to output generated fast tokenizer files."
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help=(
            f"Optional tokenizer type selected in the list of {list(TOKENIZER_CLASSES.keys())}. If not given, will "
            "download and convert all the checkpoints from AWS."
        ),
    )
    parser.add_argument(
        "--checkpoint_name",
        default=None,
        type=str,
        help="Optional checkpoint name. If not given, will download and convert the canonical checkpoints from AWS.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Re-download checkpoints.",
    )
    args = parser.parse_args()

    convert_slow_checkpoint_to_fast(args.tokenizer_name, args.checkpoint_name, args.dump_path, args.force_download)
