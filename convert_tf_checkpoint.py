# coding=utf-8
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import tensorflow as tf
import torch

from .modeling_pytorch import BertConfig, BertModel

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--tf_checkpoint_path",
                    default = None,
                    type = str,
                    required = True,
                    help = "Path the TensorFlow checkpoint path.")
parser.add_argument("--bert_config_file",
                    default = None,
                    type = str,
                    required = True,
                    help = "The config json file corresponding to the pre-trained BERT model. \n"
                        "This specifies the model architecture.")
parser.add_argument("--pytorch_dump_path",
                    default = None,
                    type = str,
                    required = True,
                    help = "Path to the output PyTorch model.")

args = parser.parse_args()

def convert():
    # Load weights from TF model
    path = args.tf_checkpoint_path
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

    # Initialise PyTorch model and fill weights-in
    config = BertConfig.from_json_file(args.bert_config_file)
    model = BertModel(config)
    for name, array in zip(names, arrays):
        name = name[5:]  # skip "bert/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    torch.save(model.state_dict(), args.pytorch_dump_path)

if __name__ == "__main__":
    convert()
    return None
