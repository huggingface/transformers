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

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import os
import argparse
import torch
import numpy as np
import tensorflow as tf
from pytorch_pretrained_bert.modeling import BertModel


def convert_pytorch_checkpoint_to_tf(model:BertModel, ckpt_dir:str, model_name:str):

    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:

    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transopse = (
        "dense.weight",
        "attention.self.query",
        "attention.self.key",
        "attention.self.value"
    )

    var_map = (
        ('layer.', 'layer_'),
        ('word_embeddings.weight', 'word_embeddings'),
        ('position_embeddings.weight', 'position_embeddings'),
        ('token_type_embeddings.weight', 'token_type_embeddings'),
        ('.', '/'),
        ('LayerNorm/weight', 'LayerNorm/gamma'),
        ('LayerNorm/bias', 'LayerNorm/beta'),
        ('weight', 'kernel')
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    session = tf.Session()
    state_dict = model.state_dict()
    tf_vars = []

    def to_tf_var_name(name:str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return 'bert/{}'.format(name)

    def assign_tf_var(tensor:np.ndarray, name:str):
        tmp_var = tf.Variable(initial_value=tensor)
        tf_var = tf.get_variable(dtype=tmp_var.dtype, shape=tmp_var.shape, name=name)
        op = tf.assign(ref=tf_var, value=tmp_var)
        session.run(tf.variables_initializer([tmp_var, tf_var]))
        session.run(fetches=[op, tf_var])
        return tf_var

    for var_name in state_dict:
        tf_name = to_tf_var_name(var_name)
        torch_tensor = state_dict[var_name].numpy()
        if any([x in var_name for x in tensors_to_transopse]):
            torch_tensor = torch_tensor.T
        tf_tensor = assign_tf_var(tensor=torch_tensor, name=tf_name)
        tf_vars.append(tf_tensor)
        print("{0}{1}initialized".format(tf_name, " " * (60 - len(tf_name))))

    saver = tf.train.Saver(tf_vars)
    saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="model name e.g. bert-base-uncased")
    parser.add_argument("--cache_dir",
                        type=str,
                        default=None,
                        required=False,
                        help="Directory containing pytorch model")
    parser.add_argument("--pytorch_model_path",
                        type=str,
                        required=True,
                        help="/path/to/<pytorch-model-name>.bin")
    parser.add_argument("--tf_cache_dir",
                        type=str,
                        required=True,
                        help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)
    
    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        state_dict=torch.load(args.pytorch_model_path),
        cache_dir=args.cache_dir
    )
    
    convert_pytorch_checkpoint_to_tf(
        model=model,
        ckpt_dir=args.tf_cache_dir,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
