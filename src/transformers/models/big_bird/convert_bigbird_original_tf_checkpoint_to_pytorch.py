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
"""Convert BigBird checkpoint."""

import argparse

from transformers import BigBirdConfig, BigBirdForPreTraining, BigBirdForQuestionAnswering, load_tf_weights_in_big_bird
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, big_bird_config_file, pytorch_dump_path, is_trivia_qa):
    # Initialise PyTorch model
    config = BigBirdConfig.from_json_file(big_bird_config_file)
    print(f"Building PyTorch model from configuration: {config}")

    if is_trivia_qa:
        model = BigBirdForQuestionAnswering(config)
    else:
        model = BigBirdForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=is_trivia_qa)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--big_bird_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--is_trivia_qa", action="store_true", help="Whether to convert a model with a trivia_qa head."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.big_bird_config_file, args.pytorch_dump_path, args.is_trivia_qa
    )
