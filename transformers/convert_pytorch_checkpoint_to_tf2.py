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

from transformers import is_torch_available, cached_path

from transformers import (load_pytorch_checkpoint_in_tf2_model,
    BertConfig, TFBertForPreTraining, TFBertForQuestionAnswering, TFBertForSequenceClassification, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  GPT2Config, TFGPT2LMHeadModel, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  XLNetConfig, TFXLNetLMHeadModel, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  XLMConfig, TFXLMWithLMHeadModel, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  TransfoXLConfig, TFTransfoXLLMHeadModel, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  OpenAIGPTConfig, TFOpenAIGPTLMHeadModel, OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  RobertaConfig, TFRobertaForMaskedLM, TFRobertaForSequenceClassification, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  DistilBertConfig, TFDistilBertForMaskedLM, TFDistilBertForQuestionAnswering, TFDistilBertForSequenceClassification, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  CTRLConfig, TFCTRLLMHeadModel, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  AlbertConfig, TFAlbertForMaskedLM, ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                  T5Config, TFT5WithLMHeadModel, T5_PRETRAINED_CONFIG_ARCHIVE_MAP)

if is_torch_available():
    import torch
    import numpy as np
    from transformers import (BertForPreTraining, BertForQuestionAnswering, BertForSequenceClassification, BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      GPT2LMHeadModel, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      XLNetLMHeadModel, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      XLMWithLMHeadModel, XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      TransfoXLLMHeadModel, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      OpenAIGPTLMHeadModel, OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      RobertaForMaskedLM, RobertaForSequenceClassification, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      DistilBertForMaskedLM, DistilBertForQuestionAnswering, DistilBertForSequenceClassification, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      CTRLLMHeadModel, CTRL_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      AlbertForMaskedLM, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
                                      T5WithLMHeadModel, T5_PRETRAINED_MODEL_ARCHIVE_MAP)
else:
    (BertForPreTraining, BertForQuestionAnswering, BertForSequenceClassification, BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    GPT2LMHeadModel, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLNetLMHeadModel, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLMWithLMHeadModel, XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
    TransfoXLLMHeadModel, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
    OpenAIGPTLMHeadModel, OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
    RobertaForMaskedLM, RobertaForSequenceClassification, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertForQuestionAnswering, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    CTRLLMHeadModel, CTRL_PRETRAINED_MODEL_ARCHIVE_MAP,
    AlbertForMaskedLM, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    T5WithLMHeadModel, T5_PRETRAINED_MODEL_ARCHIVE_MAP) = (
        None, None, None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None,
        None, None, None,
        None, None, None, None,
        None, None,
        None, None,
        None, None)


import logging
logging.basicConfig(level=logging.INFO)

MODEL_CLASSES = {
    'bert': (BertConfig, TFBertForPreTraining, BertForPreTraining, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'bert-large-uncased-whole-word-masking-finetuned-squad': (BertConfig, TFBertForQuestionAnswering, BertForQuestionAnswering, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'bert-large-cased-whole-word-masking-finetuned-squad': (BertConfig, TFBertForQuestionAnswering, BertForQuestionAnswering, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'bert-base-cased-finetuned-mrpc': (BertConfig, TFBertForSequenceClassification, BertForSequenceClassification, BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'gpt2': (GPT2Config, TFGPT2LMHeadModel, GPT2LMHeadModel, GPT2_PRETRAINED_MODEL_ARCHIVE_MAP, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'xlnet': (XLNetConfig, TFXLNetLMHeadModel, XLNetLMHeadModel, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'xlm': (XLMConfig, TFXLMWithLMHeadModel, XLMWithLMHeadModel, XLM_PRETRAINED_MODEL_ARCHIVE_MAP, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'transfo-xl': (TransfoXLConfig, TFTransfoXLLMHeadModel, TransfoXLLMHeadModel, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP, TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'openai-gpt': (OpenAIGPTConfig, TFOpenAIGPTLMHeadModel, OpenAIGPTLMHeadModel, OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP, OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'roberta': (RobertaConfig, TFRobertaForMaskedLM, RobertaForMaskedLM, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'roberta-large-mnli': (RobertaConfig, TFRobertaForSequenceClassification, RobertaForSequenceClassification, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'distilbert': (DistilBertConfig, TFDistilBertForMaskedLM, DistilBertForMaskedLM, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'distilbert-base-uncased-distilled-squad': (DistilBertConfig, TFDistilBertForQuestionAnswering, DistilBertForQuestionAnswering, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'distilbert-base-uncased-distilled-squad': (DistilBertConfig, TFDistilBertForQuestionAnswering, DistilBertForQuestionAnswering, DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP, DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'ctrl': (CTRLConfig, TFCTRLLMHeadModel, CTRLLMHeadModel, CTRL_PRETRAINED_MODEL_ARCHIVE_MAP, CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP),
    'albert': (AlbertConfig, TFAlbertForMaskedLM, AlbertForMaskedLM, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP, ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP),
    't5': (T5Config, TFT5WithLMHeadModel, T5WithLMHeadModel, T5_PRETRAINED_MODEL_ARCHIVE_MAP, T5_PRETRAINED_CONFIG_ARCHIVE_MAP),
}

def convert_pt_checkpoint_to_tf(model_type, pytorch_checkpoint_path, config_file, tf_dump_path, compare_with_pt_model=False, use_cached_models=True):
    if model_type not in MODEL_CLASSES:
        raise ValueError("Unrecognized model type, should be one of {}.".format(list(MODEL_CLASSES.keys())))

    config_class, model_class, pt_model_class, aws_model_maps, aws_config_map = MODEL_CLASSES[model_type]

    # Initialise TF model
    if config_file in aws_config_map:
        config_file = cached_path(aws_config_map[config_file], force_download=not use_cached_models)
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print("Building TensorFlow model from configuration: {}".format(str(config)))
    tf_model = model_class(config)

    # Load weights from tf checkpoint
    if pytorch_checkpoint_path in aws_model_maps:
        pytorch_checkpoint_path = cached_path(aws_model_maps[pytorch_checkpoint_path], force_download=not use_cached_models)
    # Load PyTorch checkpoint in tf2 model:
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path)

    if compare_with_pt_model:
        tfo = tf_model(tf_model.dummy_inputs, training=False)  # build the network

        state_dict = torch.load(pytorch_checkpoint_path, map_location='cpu')
        pt_model = pt_model_class.from_pretrained(pretrained_model_name_or_path=None,
                                                  config=config,
                                                  state_dict=state_dict)

        with torch.no_grad():
            pto = pt_model(**pt_model.dummy_inputs)

        np_pt = pto[0].numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print("Max absolute difference between models outputs {}".format(diff))
        assert diff <= 2e-2, "Error, model absolute difference is >2e-2: {}".format(diff)

    # Save pytorch-model
    print("Save TensorFlow model to {}".format(tf_dump_path))
    tf_model.save_weights(tf_dump_path, save_format='h5')


def convert_all_pt_checkpoints_to_tf(args_model_type, tf_dump_path, model_shortcut_names_or_path=None, config_shortcut_names_or_path=None,
                                     compare_with_pt_model=False, use_cached_models=False, remove_cached_files=False, only_convert_finetuned_models=False):
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

        config_class, model_class, pt_model_class, aws_model_maps, aws_config_map = MODEL_CLASSES[model_type]

        if model_shortcut_names_or_path is None:
            model_shortcut_names_or_path = list(aws_model_maps.keys())
        if config_shortcut_names_or_path is None:
            config_shortcut_names_or_path = model_shortcut_names_or_path

        for i, (model_shortcut_name, config_shortcut_name) in enumerate(
                zip(model_shortcut_names_or_path, config_shortcut_names_or_path), start=1):
            print("-" * 100)
            if '-squad' in model_shortcut_name or '-mrpc' in model_shortcut_name or '-mnli' in model_shortcut_name:
                if not only_convert_finetuned_models:
                    print("    Skipping finetuned checkpoint {}".format(model_shortcut_name))
                    continue
                model_type = model_shortcut_name
            elif only_convert_finetuned_models:
                print("    Skipping not finetuned checkpoint {}".format(model_shortcut_name))
                continue
            print("    Converting checkpoint {}/{}: {} - model_type {}".format(i, len(aws_config_map), model_shortcut_name, model_type))
            print("-" * 100)

            if config_shortcut_name in aws_config_map:
                config_file = cached_path(aws_config_map[config_shortcut_name], force_download=not use_cached_models)
            else:
                config_file = cached_path(config_shortcut_name, force_download=not use_cached_models)

            if model_shortcut_name in aws_model_maps:
                model_file = cached_path(aws_model_maps[model_shortcut_name], force_download=not use_cached_models)
            else:
                model_file = cached_path(model_shortcut_name, force_download=not use_cached_models)

            if os.path.isfile(model_shortcut_name):
                model_shortcut_name = 'converted_model'

            convert_pt_checkpoint_to_tf(model_type=model_type,
                                        pytorch_checkpoint_path=model_file,
                                        config_file=config_file,
                                        tf_dump_path=os.path.join(tf_dump_path, model_shortcut_name + '-tf_model.h5'),
                                        compare_with_pt_model=compare_with_pt_model)
            if remove_cached_files:
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
                               "use the configuration associated to the shortcut name on the AWS")
    parser.add_argument("--compare_with_pt_model",
                        action='store_true',
                        help = "Compare Tensorflow and PyTorch model predictions.")
    parser.add_argument("--use_cached_models",
                        action='store_true',
                        help = "Use cached models if possible instead of updating to latest checkpoint versions.")
    parser.add_argument("--remove_cached_files",
                        action='store_true',
                        help = "Remove pytorch models after conversion (save memory when converting in batches).")
    parser.add_argument("--only_convert_finetuned_models",
                        action='store_true',
                        help = "Only convert finetuned models.")
    args = parser.parse_args()

    # if args.pytorch_checkpoint_path is not None:
    #     convert_pt_checkpoint_to_tf(args.model_type.lower(),
    #                                 args.pytorch_checkpoint_path,
    #                                 args.config_file if args.config_file is not None else args.pytorch_checkpoint_path,
    #                                 args.tf_dump_path,
    #                                 compare_with_pt_model=args.compare_with_pt_model,
    #                                 use_cached_models=args.use_cached_models)
    # else:
    convert_all_pt_checkpoints_to_tf(args.model_type.lower() if args.model_type is not None else None,
                                        args.tf_dump_path,
                                        model_shortcut_names_or_path=[args.pytorch_checkpoint_path] if args.pytorch_checkpoint_path is not None else None,
                                        config_shortcut_names_or_path=[args.config_file] if args.config_file is not None else None,
                                        compare_with_pt_model=args.compare_with_pt_model,
                                        use_cached_models=args.use_cached_models,
                                        remove_cached_files=args.remove_cached_files,
                                        only_convert_finetuned_models=args.only_convert_finetuned_models)
