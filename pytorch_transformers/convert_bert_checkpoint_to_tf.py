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
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from pytorch_transformers import BertConfig, TFBertForPreTraining, load_pt_weights_in_bert

import logging
logging.basicConfig(level=logging.INFO)

def convert_bert_checkpoint_to_tf(pytorch_checkpoint_path, bert_config_file, tf_dump_path):
    # Initialise TF model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building TensorFlow model from configuration: {}".format(str(config)))
    model = TFBertForPreTraining(config)

    # Load weights from tf checkpoint
    model = load_pt_weights_in_bert(model, config, pytorch_checkpoint_path)

    # Save pytorch-model
    print("Save TensorFlow model to {}".format(tf_dump_path))
    model.save_weights(tf_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--pytorch_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the PyTorch checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default = None,
                        type = str,
                        required = True,
                        help = "The config json file corresponding to the pre-trained BERT model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--tf_dump_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output Tensorflow dump file.")
    args = parser.parse_args()
    convert_bert_checkpoint_to_tf(args.pytorch_checkpoint_path,
                                     args.bert_config_file,
                                     args.tf_dump_path)
