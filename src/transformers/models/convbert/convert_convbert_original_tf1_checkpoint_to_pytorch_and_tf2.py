# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert ConvBERT checkpoint."""

import argparse

from transformers import ConvBertConfig, ConvBertModel, TFConvBertModel, load_tf_weights_in_convbert
from transformers.utils import logging


logging.set_verbosity_info()


def convert_orig_tf1_checkpoint_to_pytorch(tf_checkpoint_path, convbert_config_file, pytorch_dump_path):
    conf = ConvBertConfig.from_json_file(convbert_config_file)
    model = ConvBertModel(conf)

    model = load_tf_weights_in_convbert(model, conf, tf_checkpoint_path)
    model.save_pretrained(pytorch_dump_path)

    tf_model = TFConvBertModel.from_pretrained(pytorch_dump_path, from_pt=True)
    tf_model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--convbert_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained ConvBERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_orig_tf1_checkpoint_to_pytorch(args.tf_checkpoint_path, args.convbert_config_file, args.pytorch_dump_path)
