# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert BigBird checkpoint."""

import argparse
import math
import os

import torch

from transformers import BigBirdConfig, BigBirdForPreTraining, BigBirdForQuestionAnswering
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


_TRIVIA_QA_MAPPING = {
    "big_bird_attention": "attention/self",
    "output_layer_norm": "output/LayerNorm",
    "attention_output": "attention/output/dense",
    "output": "output/dense",
    "self_attention_layer_norm": "attention/output/LayerNorm",
    "intermediate": "intermediate/dense",
    "word_embeddings": "bert/embeddings/word_embeddings",
    "position_embedding": "bert/embeddings/position_embeddings",
    "type_embeddings": "bert/embeddings/token_type_embeddings",
    "embeddings": "bert/embeddings",
    "layer_normalization": "output/LayerNorm",
    "layer_norm": "LayerNorm",
    "trivia_qa_head": "qa_classifier",
    "dense": "intermediate/dense",
    "dense_1": "qa_outputs",
}


def load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=False):
    """Load tf checkpoints in a pytorch model."""

    def load_tf_weights_bert(init_vars, tf_path):
        names = []
        tf_weights = {}

        for name, shape in init_vars:
            array = tf.train.load_variable(tf_path, name)
            name = name.replace("bert/encoder/LayerNorm", "bert/embeddings/LayerNorm")
            logger.info(f"Loading TF weight {name} with shape {shape}")
            names.append(name)
            tf_weights[name] = array

        return names, tf_weights

    def load_tf_weights_trivia_qa(init_vars):
        names = []
        tf_weights = {}

        for i, var in enumerate(init_vars):
            name_items = var.name.split("/")

            if "transformer_scaffold" in name_items[0]:
                layer_name_items = name_items[0].split("_")
                if len(layer_name_items) < 3:
                    layer_name_items += [0]

                name_items[0] = f"bert/encoder/layer_{layer_name_items[2]}"

            name = "/".join([_TRIVIA_QA_MAPPING.get(x, x) for x in name_items])[:-2]  # remove last :0 in variable

            if "self/attention/output" in name:
                name = name.replace("self/attention/output", "output")

            if i >= len(init_vars) - 2:
                name = name.replace("intermediate", "output")

            logger.info(f"Loading TF weight {name} with shape {var.shape}")
            array = var.value().numpy()
            names.append(name)
            tf_weights[name] = array

        return names, tf_weights

    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")

    # Load weights from TF model
    init_vars = tf.saved_model.load(tf_path).variables if is_trivia_qa else tf.train.list_variables(tf_path)

    if len(init_vars) <= 0:
        raise ValueError("Loaded trained variables cannot be empty.")

    pt_names = list(model.state_dict().keys())

    if is_trivia_qa:
        names, tf_weights = load_tf_weights_trivia_qa(init_vars)
    else:
        names, tf_weights = load_tf_weights_bert(init_vars, tf_path)

    for txt_name in names:
        array = tf_weights[txt_name]
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        pt_name = []
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
                pt_name.append("weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
                pt_name.append("bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
                pt_name.append("weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
                pt_name.append("classifier")
            elif scope_names[0] == "transform":
                pointer = getattr(pointer, "transform")
                pt_name.append("transform")
                if ("bias" in name) or ("kernel" in name):
                    pointer = getattr(pointer, "dense")
                    pt_name.append("dense")
                elif ("beta" in name) or ("gamma" in name):
                    pointer = getattr(pointer, "LayerNorm")
                    pt_name.append("LayerNorm")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                    pt_name.append(f"{scope_names[0]}")
                except AttributeError:
                    logger.info(f"Skipping {m_name}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
                pt_name.append(f"{num}")
        if m_name[-11:] == "_embeddings" or m_name == "embeddings":
            pointer = getattr(pointer, "weight")
            pt_name.append("weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if len(array.shape) > len(pointer.shape) and math.prod(array.shape) == math.prod(pointer.shape):
                # print(txt_name, array.shape)
                if (
                    txt_name.endswith("attention/self/key/kernel")
                    or txt_name.endswith("attention/self/query/kernel")
                    or txt_name.endswith("attention/self/value/kernel")
                ):
                    array = array.transpose(1, 0, 2).reshape(pointer.shape)
                elif txt_name.endswith("attention/output/dense/kernel"):
                    array = array.transpose(0, 2, 1).reshape(pointer.shape)
                else:
                    array = array.reshape(pointer.shape)

            if pointer.shape != array.shape:
                raise ValueError(
                    f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched of {txt_name}."
                )
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        pt_weight_name = ".".join(pt_name)
        logger.info(f"Initialize PyTorch weight {pt_weight_name} from {txt_name}.")
        pointer.data = torch.from_numpy(array)
        tf_weights.pop(txt_name, None)
        pt_names.remove(pt_weight_name)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    logger.info(f"Weights not initialized in PyTorch model: {', '.join(pt_names)}.")
    return model


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, big_bird_config_file, pytorch_dump_path, is_trivia_qa):
    # Initialise PyTorch model
    config = BigBirdConfig.from_json_file(big_bird_config_file)
    print(f"Building PyTorch model from configuration: {config}")

    if is_trivia_qa:
        model = BigBirdForQuestionAnswering(config)
    else:
        model = BigBirdForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=is_trivia_qa)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--big_bird_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--is_trivia_qa", action="store_true", help="Whether to convert a model with a trivia_qa head."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.big_bird_config_file, args.pytorch_dump_path, args.is_trivia_qa
    )
