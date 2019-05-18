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
import numpy as np
from pytorch_pretrained_bert.modeling import BertConfig, BertModel


def convert_hf_checkpoint_to_tf(model:BertModel, ckpt_dir:str):

    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: directory to save Tensorflow model

    Supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering

    Note:
        To keep tf out of package-level requirements, it's imported locally.
    """

    import tensorflow as tf

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    session = tf.Session()

    state_dict = model.state_dict()

    tf_vars = []

    def to_tf_var_name(name:str):
        """todo: compile as regex"""
        name = name.replace('layer.',                       'layer_')
        name = name.replace('word_embeddings.weight',       'word_embeddings')
        name = name.replace('position_embeddings.weight',   'position_embeddings')
        name = name.replace('token_type_embeddings.weight', 'token_type_embeddings')
        name = name.replace('.',                            '/')
        name = name.replace('LayerNorm/weight',             'LayerNorm/gamma')
        name = name.replace('LayerNorm/bias',               'LayerNorm/beta')
        name = name.replace('weight',                       'kernel')
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
        if var_name.endswith('dense.weight'):
            torch_tensor = torch_tensor.T
        tf_tensor = assign_tf_var(tensor=torch_tensor, name=tf_name)
        tf_vars.append(tf_tensor)
        print("{0}{1}initialized".format(tf_name, " " * (60 - len(tf_name))))

    saver = tf.train.Saver(tf_vars)
    saver.save(session, os.path.join(ckpt_dir, 'model'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="pretrained_model_name_or_path: either: \
                                - a str with the name of a pre-trained model to load selected in the list of: \
                                    . `bert-base-uncased` \
                                    . `bert-large-uncased` \
                                    . `bert-base-cased` \
                                    . `bert-large-cased` \
                                    . `bert-base-multilingual-uncased` \
                                    . `bert-base-multilingual-cased` \
                                    . `bert-base-chinese` \
                                - a path or url to a pretrained model archive containing: \
                                    . `bert_config.json` a configuration file for the model \
                                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance \
                                - a path or url to a pretrained model archive containing: \
                                    . `bert_config.json` a configuration file for the model \
                                    . `model.ckpt` a TensorFlow checkpoint")
    parser.add_argument("--config_file_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to bert config file.")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to a folder in which the TF model will be cached.")
    args = parser.parse_args()

    model = BertModel(
        config=BertConfig(args.config_file_path)
    ).from_pretrained(args.pretrained_model_name_or_path)

    convert_hf_checkpoint_to_tf(model=model, ckpt_dir=args.cache_dir)