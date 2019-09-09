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
""" Convert pytorch checkpoints to TensorFlow """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

from pytorch_transformers import is_torch_available

from pytorch_transformers import (BertConfig, TFBertForPreTraining, load_bert_pt_weights_in_tf2,
                                  GPT2Config, TFGPT2LMHeadModel, load_gpt2_pt_weights_in_tf2)

if is_torch_available():
    import torch
    import numpy as np
    from pytorch_transformers import BertForPreTraining, GPT2LMHeadModel
else:
    BertForPreTraining, GPT2LMHeadModel = None, None


import logging
logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    'bert': (BertConfig, TFBertForPreTraining, load_bert_pt_weights_in_tf2, BertForPreTraining),
    'gpt2': (GPT2Config, TFGPT2LMHeadModel, load_gpt2_pt_weights_in_tf2, GPT2LMHeadModel),
}

def convert_pt_checkpoint_to_tf(model_type, pytorch_checkpoint_path, config_file, tf_dump_path, compare_with_pt_model=False):
    if model_type not in MODEL_CLASSES:
        raise ValueError("Unrecognized model type, should be one of {}.".format(list(MODEL_CLASSES.keys())))

    config_class, model_class, loading_fct, pt_model_class = MODEL_CLASSES[model_type]

    # Initialise TF model
    config = config_class.from_json_file(config_file)
    print("Building TensorFlow model from configuration: {}".format(str(config)))
    tf_model = model_class(config)

    # Load weights from tf checkpoint
    tf_model = loading_fct(tf_model, config, pytorch_checkpoint_path)

    if compare_with_pt_model:
        inputs_list = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        tf_inputs = tf.constant(inputs_list)
        tfo = tf_model(tf_inputs, training=False)  # build the network

        pt_model = pt_model_class.from_pretrained(None,
                                                  config=config,
                                                  state_dict=torch.load(pytorch_checkpoint_path,
                                                                        map_location='cpu'))
        pt_inputs = torch.tensor(inputs_list)
        with torch.no_grad():
            pto = pt_model(pt_inputs)

        np_pt = pto[0].detach().numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print("Max absolute difference between models outputs {}".format(diff))

    # Save pytorch-model
    print("Save TensorFlow model to {}".format(tf_dump_path))
    tf_model.save_weights(tf_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--model_type",
                        default = None,
                        type = str,
                        required = True,
                        help = "Model type selcted in the list of {}.".format(list(MODEL_CLASSES.keys())))
    parser.add_argument("--pytorch_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the PyTorch checkpoint path.")
    parser.add_argument("--config_file",
                        default = None,
                        type = str,
                        required = True,
                        help = "The config json file corresponding to the pre-trained model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--tf_dump_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output Tensorflow dump file.")
    parser.add_argument("--compare_with_pt_model",
                        action='store_true',
                        help = "Compare Tensorflow and PyTorch model predictions.")
    args = parser.parse_args()
    convert_pt_checkpoint_to_tf(args.model_type.lower(),
                                args.pytorch_checkpoint_path,
                                args.config_file,
                                args.tf_dump_path,
                                compare_with_pt_model=args.compare_with_pt_model)
