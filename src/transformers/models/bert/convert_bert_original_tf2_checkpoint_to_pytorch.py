# Copyright 2020 The HuggingFace Team. All rights reserved.
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

"""
This script can be used to convert a head-less TF2.x Bert model to PyTorch, as published on the official (now
deprecated) GitHub: https://github.com/tensorflow/models/tree/v2.3.0/official/nlp/bert

TF2.x uses different variable names from the original BERT (TF 1.4) implementation. The script re-maps the TF2.x Bert
weight names to the original names, so the model can be imported with Huggingface/transformer.

You may adapt this script to include classification/MLM/NSP/etc. heads.

Note: This script is only working with an older version of the TensorFlow models repository (<= v2.3.0).
      Models trained with never versions are not compatible with this script.
"""

import argparse
import os
import re

import tensorflow as tf
import torch

from transformers import BertConfig, BertModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_tf2_weights_in_bert(model, tf_checkpoint_path, config):
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    layer_depth = []
    for full_name, shape in init_vars:
        # logger.info(f"Loading TF weight {name} with shape {shape}")
        name = full_name.split("/")
        if full_name == "_CHECKPOINTABLE_OBJECT_GRAPH" or name[0] in ["global_step", "save_counter"]:
            logger.info(f"Skipping non-model layer {full_name}")
            continue
        if "optimizer" in full_name:
            logger.info(f"Skipping optimization layer {full_name}")
            continue
        if name[0] == "model":
            # ignore initial 'model'
            name = name[1:]
        # figure out how many levels deep the name is
        depth = 0
        for _name in name:
            if _name.startswith("layer_with_weights"):
                depth += 1
            else:
                break
        layer_depth.append(depth)
        # read data
        array = tf.train.load_variable(tf_path, full_name)
        names.append("/".join(name))
        arrays.append(array)
    logger.info(f"Read a total of {len(arrays):,} layers")

    # Sanity check
    if len(set(layer_depth)) != 1:
        raise ValueError(f"Found layer names with different depths (layer depth {list(set(layer_depth))})")
    layer_depth = list(set(layer_depth))[0]
    if layer_depth != 1:
        raise ValueError(
            "The model contains more than just the embedding/encoder layers. This script does not handle MLM/NSP"
            " heads."
        )

    # convert layers
    logger.info("Converting weights...")
    for full_name, array in zip(names, arrays):
        name = full_name.split("/")
        pointer = model
        trace = []
        for i, m_name in enumerate(name):
            if m_name == ".ATTRIBUTES":
                # variable names end with .ATTRIBUTES/VARIABLE_VALUE
                break
            if m_name.startswith("layer_with_weights"):
                layer_num = int(m_name.split("-")[-1])
                if layer_num <= 2:
                    # embedding layers
                    # layer_num 0: word_embeddings
                    # layer_num 1: position_embeddings
                    # layer_num 2: token_type_embeddings
                    continue
                elif layer_num == 3:
                    # embedding LayerNorm
                    trace.extend(["embeddings", "LayerNorm"])
                    pointer = getattr(pointer, "embeddings")
                    pointer = getattr(pointer, "LayerNorm")
                elif layer_num > 3 and layer_num < config.num_hidden_layers + 4:
                    # encoder layers
                    trace.extend(["encoder", "layer", str(layer_num - 4)])
                    pointer = getattr(pointer, "encoder")
                    pointer = getattr(pointer, "layer")
                    pointer = pointer[layer_num - 4]
                elif layer_num == config.num_hidden_layers + 4:
                    # pooler layer
                    trace.extend(["pooler", "dense"])
                    pointer = getattr(pointer, "pooler")
                    pointer = getattr(pointer, "dense")
            elif m_name == "embeddings":
                trace.append("embeddings")
                pointer = getattr(pointer, "embeddings")
                if layer_num == 0:
                    trace.append("word_embeddings")
                    pointer = getattr(pointer, "word_embeddings")
                elif layer_num == 1:
                    trace.append("position_embeddings")
                    pointer = getattr(pointer, "position_embeddings")
                elif layer_num == 2:
                    trace.append("token_type_embeddings")
                    pointer = getattr(pointer, "token_type_embeddings")
                else:
                    raise ValueError(f"Unknown embedding layer with name {full_name}")
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            elif m_name == "_attention_layer":
                # self-attention layer
                trace.extend(["attention", "self"])
                pointer = getattr(pointer, "attention")
                pointer = getattr(pointer, "self")
            elif m_name == "_attention_layer_norm":
                # output attention norm
                trace.extend(["attention", "output", "LayerNorm"])
                pointer = getattr(pointer, "attention")
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "LayerNorm")
            elif m_name == "_attention_output_dense":
                # output attention dense
                trace.extend(["attention", "output", "dense"])
                pointer = getattr(pointer, "attention")
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "dense")
            elif m_name == "_output_dense":
                # output dense
                trace.extend(["output", "dense"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "dense")
            elif m_name == "_output_layer_norm":
                # output dense
                trace.extend(["output", "LayerNorm"])
                pointer = getattr(pointer, "output")
                pointer = getattr(pointer, "LayerNorm")
            elif m_name == "_key_dense":
                # attention key
                trace.append("key")
                pointer = getattr(pointer, "key")
            elif m_name == "_query_dense":
                # attention query
                trace.append("query")
                pointer = getattr(pointer, "query")
            elif m_name == "_value_dense":
                # attention value
                trace.append("value")
                pointer = getattr(pointer, "value")
            elif m_name == "_intermediate_dense":
                # attention intermediate dense
                trace.extend(["intermediate", "dense"])
                pointer = getattr(pointer, "intermediate")
                pointer = getattr(pointer, "dense")
            elif m_name == "_output_layer_norm":
                # output layer norm
                trace.append("output")
                pointer = getattr(pointer, "output")
            # weights & biases
            elif m_name in ["bias", "beta"]:
                trace.append("bias")
                pointer = getattr(pointer, "bias")
            elif m_name in ["kernel", "gamma"]:
                trace.append("weight")
                pointer = getattr(pointer, "weight")
            else:
                logger.warning(f"Ignored {m_name}")
        # for certain layers reshape is necessary
        trace = ".".join(trace)
        if re.match(r"(\S+)\.attention\.self\.(key|value|query)\.(bias|weight)", trace) or re.match(
            r"(\S+)\.attention\.output\.dense\.weight", trace
        ):
            array = array.reshape(pointer.data.shape)
        if "kernel" in full_name:
            array = array.transpose()
        if pointer.shape == array.shape:
            pointer.data = torch.from_numpy(array)
        else:
            raise ValueError(
                f"Shape mismatch in layer {full_name}: Model expects shape {pointer.shape} but layer contains shape:"
                f" {array.shape}"
            )
        logger.info(f"Successfully set variable {full_name} to PyTorch layer {trace}")
    return model


def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path, pytorch_dump_path):
    # Instantiate model
    logger.info(f"Loading model based on config from {config_path}...")
    config = BertConfig.from_json_file(config_path)
    model = BertModel(config)

    # Load weights from checkpoint
    logger.info(f"Loading weights from checkpoint {tf_checkpoint_path}...")
    load_tf2_weights_in_bert(model, tf_checkpoint_path, config)

    # Save pytorch-model
    logger.info(f"Saving PyTorch model to {pytorch_dump_path}...")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow 2.x checkpoint path."
    )
    parser.add_argument(
        "--bert_config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the BERT model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model (must include filename).",
    )
    args = parser.parse_args()
    convert_tf2_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
