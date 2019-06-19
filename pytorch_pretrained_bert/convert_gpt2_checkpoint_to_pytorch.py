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
"""Convert OpenAI GPT checkpoint."""

from __future__ import absolute_import, division, print_function

import argparse
from io import open

import torch

from pytorch_pretrained_bert.modeling_gpt2 import (CONFIG_NAME, WEIGHTS_NAME,
                                                     GPT2Config,
                                                     GPT2Model,
                                                     load_tf_weights_in_gpt2)


def convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path, model_size):
    # Construct model
    if gpt2_config_file == "":
        if model_size == "117M":
            config = GPT2Config()
        else:
            config = GPT2Config(n_embd=1024, n_layer=24, n_head=16)
    else:
        config = GPT2Config(gpt2_config_file)
    model = GPT2Model(config)

    # Load weights from numpy
    load_tf_weights_in_gpt2(model, gpt2_checkpoint_path)

    # Save pytorch-model
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    print("Save PyTorch model to {}".format(pytorch_weights_dump_path))
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print("Save configuration file to {}".format(pytorch_config_dump_path))
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--gpt2_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the TensorFlow checkpoint path.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    parser.add_argument("--gpt2_config_file",
                        default = "",
                        type = str,
                        help = "An optional config json file corresponding to the pre-trained OpenAI model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--model_size",
                        default = "117M",
                        type = str,
                        help = "An optional argument that specifies whether the checkpoint to convert is from the 117M or the 345M model.")
    args = parser.parse_args()

    if args.model_size != "117M" or args.model_size != "345M":
        parser.error("--model_size must be either 117M or 345M.")

    convert_gpt2_checkpoint_to_pytorch(args.gpt2_checkpoint_path,
                                         args.gpt2_config_file,
                                         args.pytorch_dump_folder_path,
                                         args.model_size)
