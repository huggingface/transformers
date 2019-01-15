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
"""Convert Transformer XL checkpoint and datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import pickle

import tensorflow as tf
import torch
import numpy as np

from pytorch_pretrained_bert.modeling_transfo_xl import TransfoXLConfig, TransfoXLModel, CONFIG_NAME, WEIGHTS_NAME
from pytorch_pretrained_bert.tokenization_transfo_xl import VOCAB_NAME, CORPUS_NAME

# We do this to be able to load the python 2 datasets pickles
# See e.g. https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory/2121918#2121918
import pytorch_pretrained_bert.tokenization_transfo_xl as data_utils
data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules['data_utils'] = data_utils
sys.modules['vocabulary'] = data_utils

def build_tf_to_pytorch_map(model, config):
    """ A map of modules from TF to PyTorch.
        This time I use a map to keep the PyTorch model as identical to the original PyTorch model as possible.
    """
    tf_to_pt_map = {}
    # Embeddings cutoffs
    for i, (embed_l, proj_l) in enumerate(zip(model.word_emb.emb_layers, model.word_emb.emb_projs)):
        layer_str = "transformer/adaptive_embed/cutoff_%d/" % i
        tf_to_pt_map.update({
            layer_str + 'lookup_table': embed_l.weight,
            layer_str + 'proj_W': proj_l
            })

    # Transformer blocks
    for i, b in enumerate(model.layers):
        layer_str = "transformer/layer_%d/" % i
        tf_to_pt_map.update({
            layer_str + "rel_attn/LayerNorm/gamma": b.dec_attn.layer_norm.weight,
            layer_str + "rel_attn/LayerNorm/beta": b.dec_attn.layer_norm.bias,
            layer_str + "rel_attn/o/kernel": b.dec_attn.o_net.weight,
            layer_str + "rel_attn/qkv/kernel": b.dec_attn.qkv_net.weight,
            layer_str + "rel_attn/r/kernel": b.dec_attn.r_net.weight,
            layer_str + "ff/LayerNorm/gamma": b.pos_ff.layer_norm.weight,
            layer_str + "ff/LayerNorm/beta": b.pos_ff.layer_norm.bias,
            layer_str + "ff/layer_1/kernel": b.pos_ff.CoreNet[0].weight,
            layer_str + "ff/layer_1/bias": b.pos_ff.CoreNet[0].bias,
            layer_str + "ff/layer_2/kernel": b.pos_ff.CoreNet[3].weight,
            layer_str + "ff/layer_2/bias": b.pos_ff.CoreNet[3].bias,
        })

    # Softmax cutoffs
    for i, (out_l, proj_l, tie_proj) in enumerate(zip(
                            model.crit.out_layers,
                            model.crit.out_projs,
                            config.tie_projs)):
        layer_str = "transformer/adaptive_softmax/cutoff_%d/" % i
        if config.tie_weight:
            tf_to_pt_map.update({
                layer_str + 'b': out_l.bias})
        else:
            raise NotImplementedError
            # I don't think this is implemented in the TF code
            tf_to_pt_map.update({
                layer_str + 'lookup_table': out_l.weight,
                layer_str + 'b': out_l.bias})
        if not tie_proj:
            tf_to_pt_map.update({
                layer_str + 'proj': proj_l
                })

    # Relative positioning biases
    if config.untie_r:
        layer_str = "transformer/r_r_bias"
        layer_str_2 = "transformer/r_w_bias"
        r_r_list = []
        r_w_list = []
        for b in model.layers:
            r_r_list.append(b.dec_attn.r_r_bias)
            r_w_list.append(b.dec_attn.r_w_bias)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    tf_to_pt_map.update({
        'transformer/r_r_bias': r_r_list,
        'transformer/r_w_bias': r_w_list})
    return tf_to_pt_map


def convert_transfo_xl_checkpoint_to_pytorch(tf_checkpoint_path,
                                             transfo_xl_config_file,
                                             pytorch_dump_folder_path,
                                             transfo_xl_dataset_file):
    if transfo_xl_dataset_file:
        with open(transfo_xl_dataset_file, "rb") as fp:
            corpus = pickle.load(fp, encoding="latin1")
        # Save vocabulary and dataset cache as Dictionaries (should be better than pickles for the long-term)
        pytorch_vocab_dump_path = pytorch_dump_folder_path + '/' + VOCAB_NAME
        print("Save vocabulary to {}".format(pytorch_vocab_dump_path))
        torch.save(corpus.vocab.__dict__, pytorch_vocab_dump_path)

        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop('vocab', None)
        pytorch_dataset_dump_path = pytorch_dump_folder_path + '/' + CORPUS_NAME
        print("Save dataset to {}".format(pytorch_dataset_dump_path))
        torch.save(corpus_dict_no_vocab, pytorch_dataset_dump_path)

    if tf_checkpoint_path:
        config_path = os.path.abspath(transfo_xl_config_file)
        tf_path = os.path.abspath(tf_checkpoint_path)

        print("Converting Transformer XL checkpoint from {} with config at {}".format(tf_path, config_path))
        # Initialise PyTorch model
        # Construct model
        if transfo_xl_config_file == "":
            config = TransfoXLConfig()
        else:
            config = TransfoXLConfig(transfo_xl_config_file)
        print("Building PyTorch model from configuration: {}".format(str(config)))
        model = TransfoXLModel(config)

        # Build TF to PyTorch weights loading map
        tf_to_pt_map = build_tf_to_pytorch_map(model.transformer, config)

        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)
        tf_weights = {}
        for name, shape in init_vars:
            print("Loading TF weight {} with shape {}".format(name, shape))
            array = tf.train.load_variable(tf_path, name)
            tf_weights[name] = array

        for name, pointer in tf_to_pt_map.items():
            assert name in tf_weights
            array = tf_weights[name]
            # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
            # which are not required for using pretrained model
            if 'kernel' in name or 'proj_W' in name:
                array = np.transpose(array)
            if ('r_r_bias' in name or 'r_w_bias' in name) and len(pointer) > 1:
                # Here we will split the TF weigths
                assert len(pointer) == array.shape[0]
                for i, p_i in enumerate(pointer):
                    arr_i = array[i, ...]
                    try:
                        assert p_i.shape == arr_i.shape
                    except AssertionError as e:
                        e.args += (p_i.shape, arr_i.shape)
                        raise
                    print("Initialize PyTorch weight {} for layer {}".format(name, i))
                    p_i.data = torch.from_numpy(arr_i)
                continue
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
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the folder to store the PyTorch model or dataset/vocab.")
    parser.add_argument("--tf_checkpoint_path",
                        default = "",
                        type = str,
                        help = "An optional path to a TensorFlow checkpoint path to be converted.")
    parser.add_argument("--transfo_xl_config_file",
                        default = "",
                        type = str,
                        help = "An optional config json file corresponding to the pre-trained BERT model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--transfo_xl_dataset_file",
                        default = "",
                        type = str,
                        help = "An optional dataset file to be converted in a vocabulary.")
    args = parser.parse_args()
    convert_transfo_xl_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.transfo_xl_config_file,
                                     args.pytorch_dump_folder_path,
                                     args.transfo_xl_dataset_file)
