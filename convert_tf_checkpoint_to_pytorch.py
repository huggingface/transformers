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
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import tensorflow as tf
import torch
import numpy as np

from modeling import BertConfig, BertModel


def convert(config_path, ckpt_path, out_path=None):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(config_path)
    model = BertModel(config)

    # Load weights from TF model
    path = ckpt_path
    print("Converting TensorFlow checkpoint from {}".format(path))

    init_vars = tf.train.list_variables(path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading {} with shape {}".format(name, shape))
        array = tf.train.load_variable(path, name)
        print("Numpy array shape {}".format(array.shape))
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "bert/"
        print("Loading {}".format(name))
        name = name.split('/')
        if name[0] in ['redictions', 'eq_relationship']:
            print("Skipping")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    if out_path is not None:
        torch.save(model.state_dict(), out_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default=None,
                        type=str,
                        required=False,
                        help="Path to the output PyTorch model.")

    args = parser.parse_args()
    print(args)
    convert(args.bert_config_file, args.tf_checkpoint_path, args.pytorch_dump_path)
