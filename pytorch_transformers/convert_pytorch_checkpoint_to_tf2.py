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

import os
import argparse
import tensorflow as tf

from pytorch_transformers import is_torch_available, cached_path

from pytorch_transformers import (BertConfig, TFBertForPreTraining, load_bert_pt_weights_in_tf2,
                                  GPT2Config, TFGPT2LMHeadModel, load_gpt2_pt_weights_in_tf2,
                                  XLNetConfig, TFXLNetLMHeadModel, load_xlnet_pt_weights_in_tf2,
                                  XLMConfig, TFXLMWithLMHeadModel, load_xlm_pt_weights_in_tf2,
                                  TransfoXLConfig, TFTransfoXLLMHeadModel, load_transfo_xl_pt_weights_in_tf2,)

if is_torch_available():
    import torch
    import numpy as np
    from pytorch_transformers import (BertForPreTraining, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                      GPT2LMHeadModel, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                      XLNetLMHeadModel, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                      XLMWithLMHeadModel, XLM_PRETRAINED_MODEL_ARCHIVE_MAP, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                      TransfoXLLMHeadModel, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,)
else:
    (BertForPreTraining, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    GPT2LMHeadModel, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLNetLMHeadModel, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLMWithLMHeadModel, XLM_PRETRAINED_MODEL_ARCHIVE_MAP, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    TransfoXLLMHeadModel, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,) = (
        None, None, None,
        None, None, None,
        None, None, None,
        None, None, None,
        None, None, None,)


import logging
logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    'bert': (BertConfig, TFBertForPreTraining, load_bert_pt_weights_in_tf2, BertForPreTraining, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'gpt2': (GPT2Config, TFGPT2LMHeadModel, load_gpt2_pt_weights_in_tf2, GPT2LMHeadModel, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'xlnet': (XLNetConfig, TFXLNetLMHeadModel, load_xlnet_pt_weights_in_tf2, XLNetLMHeadModel, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'xlm': (XLMConfig, TFXLMWithLMHeadModel, load_xlm_pt_weights_in_tf2, XLMWithLMHeadModel, XLM_PRETRAINED_MODEL_ARCHIVE_MAP, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'transfo-xl': (TransfoXLConfig, TFTransfoXLLMHeadModel, load_transfo_xl_pt_weights_in_tf2, TransfoXLLMHeadModel, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP),
}

def convert_pt_checkpoint_to_tf(model_type, pytorch_checkpoint_path, config_file, tf_dump_path, compare_with_pt_model=False):
    if model_type not in MODEL_CLASSES:
        raise ValueError("Unrecognized model type, should be one of {}.".format(list(MODEL_CLASSES.keys())))

    config_class, model_class, loading_fct, pt_model_class, aws_model_maps, aws_config_map = MODEL_CLASSES[model_type]

    # Initialise TF model
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print("Building TensorFlow model from configuration: {}".format(str(config)))
    tf_model = model_class(config)

    # Load weights from tf checkpoint
    tf_model = loading_fct(tf_model, pytorch_checkpoint_path)

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
        assert diff <= 2e-2, "Error, model absolute difference is >2e-2"

    # Save pytorch-model
    print("Save TensorFlow model to {}".format(tf_dump_path))
    tf_model.save_weights(tf_dump_path, save_format='h5')


def convert_all_pt_checkpoints_to_tf(args_model_type, tf_dump_path, compare_with_pt_model=False):
    assert os.path.isdir(args.tf_dump_path), "--tf_dump_path should be a directory"

    if args_model_type is None:
        model_types = list(MODEL_CLASSES.keys())
    else:
        model_types = [args_model_type]

    for j, model_type in enumerate(model_types, start=1):
        print("=" * 100)
        print(" Converting model type {}/{}: {}".format(j, len(model_types), model_type))
        print("=" * 100)
        if model_type not in MODEL_CLASSES:
            raise ValueError("Unrecognized model type {}, should be one of {}.".format(model_type, list(MODEL_CLASSES.keys())))

        config_class, model_class, loading_fct, pt_model_class, aws_model_maps, aws_config_map = MODEL_CLASSES[model_type]

        for i, shortcut_name in enumerate(aws_config_map.keys(), start=1):
            print("-" * 100)
            print("    Converting checkpoint {}/{}: {}".format(i, len(aws_config_map), shortcut_name))
            print("-" * 100)
            if 'finetuned' in shortcut_name:
                print("    Skipping finetuned checkpoint ")
                continue
            config_file = cached_path(aws_config_map[shortcut_name], force_download=True)
            model_file = cached_path(aws_model_maps[shortcut_name], force_download=True)

            convert_pt_checkpoint_to_tf(model_type,
                                        model_file,
                                        config_file,
                                        os.path.join(tf_dump_path, shortcut_name + '-tf_model.h5'),
                                        compare_with_pt_model=compare_with_pt_model)
            os.remove(config_file)
            os.remove(model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tf_dump_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output Tensorflow dump file.")
    parser.add_argument("--model_type",
                        default = None,
                        type = str,
                        help = "Model type selected in the list of {}. If not given, will download and convert all the models from AWS.".format(list(MODEL_CLASSES.keys())))
    parser.add_argument("--pytorch_checkpoint_path",
                        default = None,
                        type = str,
                        help = "Path to the PyTorch checkpoint path or shortcut name to download from AWS. "
                               "If not given, will download and convert all the checkpoints from AWS.")
    parser.add_argument("--config_file",
                        default = None,
                        type = str,
                        help = "The config json file corresponding to the pre-trained model. \n"
                               "This specifies the model architecture. If not given and "
                               "--pytorch_checkpoint_path is not given or is a shortcut name"
                               "use the configuration associated to teh shortcut name on the AWS")
    parser.add_argument("--compare_with_pt_model",
                        action='store_true',
                        help = "Compare Tensorflow and PyTorch model predictions.")
    args = parser.parse_args()

    if args.pytorch_checkpoint_path is not None:
        convert_pt_checkpoint_to_tf(args.model_type.lower(),
                                    args.pytorch_checkpoint_path,
                                    args.config_file,
                                    args.tf_dump_path,
                                    compare_with_pt_model=args.compare_with_pt_model)
    else:
        convert_all_pt_checkpoints_to_tf(args.model_type.lower() if args.model_type is not None else None,
                                         args.tf_dump_path,
                                         compare_with_pt_model=args.compare_with_pt_model)
