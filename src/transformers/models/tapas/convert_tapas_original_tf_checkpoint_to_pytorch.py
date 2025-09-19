# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert TAPAS checkpoint."""

import argparse
import os

import torch

from transformers import (
    TapasConfig,
    TapasForMaskedLM,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
    TapasTokenizer,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def load_tf_weights_in_tapas(model, config, tf_checkpoint_path):
    """
    Load tf checkpoints in a PyTorch model. This is an adaptation from load_tf_weights_in_bert

    - add cell selection and aggregation heads
    - take into account additional token type embedding layers
    """
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
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculate m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
                "seq_relationship",
            ]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        # in case the model is TapasForSequenceClassification, we skip output_bias and output_weights
        # since these are not used for classification
        if isinstance(model, TapasForSequenceClassification):
            if any(n in ["output_bias", "output_weights"] for n in name):
                logger.info(f"Skipping {'/'.join(name)}")
                continue
        # in case the model is TapasModel, we skip output_bias, output_weights, output_bias_cls and output_weights_cls
        # since this model does not have MLM and NSP heads
        if isinstance(model, TapasModel):
            if any(n in ["output_bias", "output_weights", "output_bias_cls", "output_weights_cls"] for n in name):
                logger.info(f"Skipping {'/'.join(name)}")
                continue
        # in case the model is TapasForMaskedLM, we skip the pooler
        if isinstance(model, TapasForMaskedLM):
            if any(n in ["pooler"] for n in name):
                logger.info(f"Skipping {'/'.join(name)}")
                continue
        # if first scope name starts with "bert", change it to "tapas"
        if name[0] == "bert":
            name[0] = "tapas"
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            # cell selection heads
            elif scope_names[0] == "output_bias":
                if not isinstance(model, TapasForMaskedLM):
                    pointer = getattr(pointer, "output_bias")
                else:
                    pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "output_weights")
            elif scope_names[0] == "column_output_bias":
                pointer = getattr(pointer, "column_output_bias")
            elif scope_names[0] == "column_output_weights":
                pointer = getattr(pointer, "column_output_weights")
            # aggregation head
            elif scope_names[0] == "output_bias_agg":
                pointer = getattr(pointer, "aggregation_classifier")
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights_agg":
                pointer = getattr(pointer, "aggregation_classifier")
                pointer = getattr(pointer, "weight")
            # classification head
            elif scope_names[0] == "output_bias_cls":
                pointer = getattr(pointer, "classifier")
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights_cls":
                pointer = getattr(pointer, "classifier")
                pointer = getattr(pointer, "weight")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name[-13:] in [f"_embeddings_{i}" for i in range(7)]:
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        # Added a check to see whether the array is a scalar (because bias terms in Tapas checkpoints can be
        # scalar => should first be converted to numpy arrays)
        if np.isscalar(array):
            array = np.array(array)
        pointer.data = torch.from_numpy(array)
    return model


def convert_tf_checkpoint_to_pytorch(
    task, reset_position_index_per_cell, tf_checkpoint_path, tapas_config_file, pytorch_dump_path
):
    # Initialise PyTorch model.
    # If you want to convert a checkpoint that uses absolute position embeddings, make sure to set reset_position_index_per_cell of
    # TapasConfig to False.

    # initialize configuration from json file
    config = TapasConfig.from_json_file(tapas_config_file)
    # set absolute/relative position embeddings parameter
    config.reset_position_index_per_cell = reset_position_index_per_cell

    # set remaining parameters of TapasConfig as well as the model based on the task
    if task == "SQA":
        model = TapasForQuestionAnswering(config=config)
    elif task == "WTQ":
        # run_task_main.py hparams
        config.num_aggregation_labels = 4
        config.use_answer_as_supervision = True
        # hparam_utils.py hparams
        config.answer_loss_cutoff = 0.664694
        config.cell_selection_preference = 0.207951
        config.huber_loss_delta = 0.121194
        config.init_cell_selection_weights_to_zero = True
        config.select_one_column = True
        config.allow_empty_column_selection = False
        config.temperature = 0.0352513

        model = TapasForQuestionAnswering(config=config)
    elif task == "WIKISQL_SUPERVISED":
        # run_task_main.py hparams
        config.num_aggregation_labels = 4
        config.use_answer_as_supervision = False
        # hparam_utils.py hparams
        config.answer_loss_cutoff = 36.4519
        config.cell_selection_preference = 0.903421
        config.huber_loss_delta = 222.088
        config.init_cell_selection_weights_to_zero = True
        config.select_one_column = True
        config.allow_empty_column_selection = True
        config.temperature = 0.763141

        model = TapasForQuestionAnswering(config=config)
    elif task == "TABFACT":
        model = TapasForSequenceClassification(config=config)
    elif task == "MLM":
        model = TapasForMaskedLM(config=config)
    elif task == "INTERMEDIATE_PRETRAINING":
        model = TapasModel(config=config)
    else:
        raise ValueError(f"Task {task} not supported.")

    print(f"Building PyTorch model from configuration: {config}")
    # Load weights from tf checkpoint
    load_tf_weights_in_tapas(model, config, tf_checkpoint_path)

    # Save pytorch-model (weights and configuration)
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # Save tokenizer files
    print(f"Save tokenizer files to {pytorch_dump_path}")
    tokenizer = TapasTokenizer(vocab_file=tf_checkpoint_path[:-10] + "vocab.txt", model_max_length=512)
    tokenizer.save_pretrained(pytorch_dump_path)

    print("Used relative position embeddings:", model.config.reset_position_index_per_cell)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task", default="SQA", type=str, help="Model task for which to convert a checkpoint. Defaults to SQA."
    )
    parser.add_argument(
        "--reset_position_index_per_cell",
        default=False,
        action="store_true",
        help="Whether to use relative position embeddings or not. Defaults to True.",
    )
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--tapas_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained TAPAS model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.task,
        args.reset_position_index_per_cell,
        args.tf_checkpoint_path,
        args.tapas_config_file,
        args.pytorch_dump_path,
    )
