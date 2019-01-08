# coding=utf-8
# Copyright 2018 The HugginFace Inc. team.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import json
import argparse
import tensorflow as tf
import torch
import numpy as np

from .modeling_openai import OpenAIGPTConfig, OpenAIGPTModel, CONFIG_NAME, WEIGHTS_NAME


def convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path):
    # Load weights from TF model
    print("Loading weights...")
    names = json.load(open(openai_checkpoint_folder_path + '/parameters_names.json', "r", encoding='utf-8'))
    shapes = json.load(open(openai_checkpoint_folder_path + '/params_shapes.json', "r", encoding='utf-8'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(openai_checkpoint_folder_path + '/params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    # if n_ctx > 0:
    #     init_params[0] = init_params[0][:n_ctx]
    # if n_special > 0:
    #     init_params[0] = np.concatenate(
    #         [init_params[1],
    #          (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
    #          init_params[0]
    #          ], 0)
    # else:
    #     init_params[0] = np.concatenate(
    #         [init_params[1],
    #          init_params[0]
    #          ], 0)
    # del init_params[1]
    # if n_transfer == -1:
    #     n_transfer = 0
    # else:
    #     n_transfer = 1 + n_transfer * 12

    init_params[0] = np.concatenate([init_params[1], init_params[0]], 0)
    del init_params[1]
    init_params = [arr.squeeze() for arr in init_params]

    # Construct model
    if openai_config_file == "":
        config = OpenAIGPTConfig()
    else:
        config = OpenAIGPTConfig(openai_config_file)
    model = OpenAIGPTModel(config)
    try:
        assert model.embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.weight.shape, init_params[0].shape)
        raise

    model.embed.weight.data = torch.from_numpy(init_params[0])
    names.pop(0)
    init_params.pop(0)

    for name, array in zip(names, init_params): # names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'g':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'b':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'w':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

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
    parser.add_argument("--openai_checkpoint_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the TensorFlow checkpoint path.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    parser.add_argument("--openai_config_file",
                        default = "",
                        type = str,
                        help = "An optional config json file corresponding to the pre-trained OpenAI model. \n"
                            "This specifies the model architecture.")
    args = parser.parse_args()
    convert_openai_checkpoint_to_pytorch(args.openai_checkpoint_folder_path,
                                         args.pytorch_dump_folder_path,
                                         args.openai_config_file)
