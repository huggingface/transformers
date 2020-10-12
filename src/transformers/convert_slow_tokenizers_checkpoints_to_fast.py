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
""" Convert slow tokenizers checkpoints in fast (serialization format of the `tokenizers` library) """

import argparse
import os

import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from transformers.utils import logging


logging.set_verbosity_info()

logger = logging.get_logger(__name__)


FAST_TOKENIZER_CLASSES = {name + "Fast": getattr(transformers, name + "Fast") for name in SLOW_TO_FAST_CONVERTERS}


def convert_slow_checkpoint_to_fast(tokenizer_name, checkpoint_name, dump_path, force_download):
    if tokenizer_name is not None and tokenizer_name not in FAST_TOKENIZER_CLASSES:
        raise ValueError(
            "Unrecognized tokenizer name, should be one of {}.".format(list(FAST_TOKENIZER_CLASSES.keys()))
        )

    if tokenizer_name is None:
        tokenizer_names = FAST_TOKENIZER_CLASSES
    else:
        tokenizer_names = {tokenizer_name: getattr(transformers, tokenizer_name + "Fast")}

    print(f"Loading tokenizer classes: {tokenizer_names}")

    for tokenizer_name in tokenizer_names:
        tokenizer_class = FAST_TOKENIZER_CLASSES[tokenizer_name]

        if checkpoint_name is None:
            add_prefix = True
            checkpoint_names = list(tokenizer_class.max_model_input_sizes.keys())
        else:
            add_prefix = False
            checkpoint_names = [checkpoint_name]

        print(f"For tokenizer {tokenizer_class.__class__.__name__} loading checkpoints: {checkpoint_names}")

        for checkpoint in checkpoint_names:
            print(f"Loading {tokenizer_class.__class__.__name__} {checkpoint}")

            # Load tokenizer
            tokenizer = tokenizer_class.from_pretrained(checkpoint, force_download=force_download)

            # Save fast tokenizer
            print("Save fast tokenizer to {} with prefix {}".format(dump_path, checkpoint))

            # For organization names we create sub-directories
            if add_prefix and "/" in checkpoint:
                checkpoint_directory, checkpoint_prefix_name = checkpoint.split("/")
                dump_path_full = os.path.join(dump_path, checkpoint_directory)
            else:
                checkpoint_prefix_name = checkpoint
                dump_path_full = dump_path

            file_names = tokenizer.save_pretrained(
                dump_path_full, legacy_format=False, filename_prefix=checkpoint_prefix_name if add_prefix else None
            )
            print("=> File names {}".format(file_names))

            for file_name in file_names:
                if not file_name.endswith("tokenizer.json"):
                    os.remove(file_name)
                    print("=> removing {}".format(file_name))


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
        help="Optional tokenizer type selected in the list of {}. If not given, will download and convert all the checkpoints from AWS.".format(
            list(FAST_TOKENIZER_CLASSES.keys())
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
        help="Re-dowload checkpoints.",
    )
    args = parser.parse_args()

    convert_slow_checkpoint_to_fast(args.tokenizer_name, args.checkpoint_name, args.dump_path, args.force_download)
