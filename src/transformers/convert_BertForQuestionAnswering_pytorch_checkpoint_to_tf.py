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

"""Convert Huggingface Pytorch checkpoint to Huggingface Tensorflow checkpoint."""

import argparse
import os

import numpy as np
import tensorflow as tf
import torch

from transformers import BertForQuestionAnswering, TFBertForQuestionAnswering


def convert_BertForQuestionAnswering_pytorch_checkpoint_to_tf(model: BertForQuestionAnswering, ckpt_dir: str, model_name: str):

    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return: 
    Currently supported HF models:
        N BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        Y BertForQuestionAnswering
    """

    tensors_to_transpose = ("qa_outputs", "dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

    var_map = (        
        ("layer.", "layer_._"),        
        ("position_embeddings.weight", "position_embeddings/embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings/embeddings"),
        (".", "/"),
        ("layer_/_", "layer_._"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
        ('word_embeddings/kernel',"word_embeddings/weight")
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return "tf_bert_for_question_answering/{}".format(name)

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.compat.v1.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.compat.v1.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.compat.v1.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
                
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))


def main(raw_args=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model name e.g. bert-base-uncased")    
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="pretrained_model_name_or_path")
    parser.add_argument("--tf_cache_dir", type=str, required=True, help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)
    print(args)
    if os.path.exists(args.tf_cache_dir):
      
      raise Exception(f"Directory {args.tf_cache_dir} already exists")
    model = BertForQuestionAnswering.from_pretrained(args.pretrained_model_name_or_path)

    tfmodel = TFBertForQuestionAnswering.from_pretrained(args.model_name)
    convert_BertForQuestionAnswering_pytorch_checkpoint_to_tf(model=model, ckpt_dir=args.tf_cache_dir, model_name=args.model_name)
    tfmodel.load_weights(os.path.join(args.tf_cache_dir, args.model_name.replace("-", "_") + ".ckpt"))
    return tfmodel

if __name__ == "__main__":
    raw_args = ['--model_name', 'bert-base-uncased','--pretrained_model_name_or_path','twmkn9/bert-base-uncased-squad2','--tf_cache_dir','tfckpt']
    tfmodel = main(raw_args)
