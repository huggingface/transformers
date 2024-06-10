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

from transformers import (
    TapasConfig,
    TapasForMaskedLM,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
    TapasTokenizer,
    load_tf_weights_in_tapas,
)
from transformers.utils import logging


logging.set_verbosity_info()


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
