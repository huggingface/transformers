# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import argparse

import torch

from transformers import MobileBertConfig, MobileBertForPreTraining, load_tf_weights_in_mobilebert
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, mobilebert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = MobileBertConfig.from_json_file(mobilebert_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = MobileBertForPreTraining(config)
    # Load weights from tf checkpoint
    model = load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path)
    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--mobilebert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained MobileBERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.mobilebert_config_file, args.pytorch_dump_path)
