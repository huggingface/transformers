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
"""Convert TAPAS checkpoint."""


import argparse

import torch

from transformers import (
    TapasConfig,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
    load_tf_weights_in_tapas,
)
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, tapas_config_file, pytorch_dump_path):
    # Initialise PyTorch model. Defaults to TapasForQuestionAnswering with default SQA config.
    # Uncomment another config and/or model to change this. If you want to convert a checkpoint
    # that has absolute position embeddings, make sure to set reset_position_index_per_cell of
    # TapasConfig to False.

    # WTQ config
    # config = TapasConfig(
    #         # run_task_main.py hparams
    #         num_aggregation_labels = 4,
    #         use_answer_as_supervision = True,
    #         # hparam_utils.py hparams
    #         answer_loss_cutoff = 0.664694,
    #         cell_selection_preference = 0.207951,
    #         huber_loss_delta = 0.121194,
    #         init_cell_selection_weights_to_zero = True,
    #         select_one_column = True,
    #         allow_empty_column_selection = False,
    #         temperature = 0.0352513,
    # )

    # WikiSQL config
    # config = TapasConfig(
    #         # run_task_main.py hparams
    #         num_aggregation_labels = 4,
    #         use_answer_as_supervision = True,
    #         # hparam_utils.py hparams
    #         answer_loss_cutoff = 0.185567,
    #         cell_selection_preference = 0.611754,
    #         huber_loss_delta = 1265.74,
    #         init_cell_selection_weights_to_zero = False,
    #         select_one_column = False,
    #         allow_empty_column_selection = False,
    #         temperature = 0.107515,
    # )

    # WikiSQL-supervised config
    # config = TapasConfig(
    #         # run_task_main.py hparams
    #         num_aggregation_labels = 4,
    #         use_answer_as_supervision = False,
    #         # hparam_utils.py hparams
    #         answer_loss_cutoff = 36.4519,
    #         cell_selection_preference = 0.903421,
    #         huber_loss_delta = 222.088,
    #         init_cell_selection_weights_to_zero = True,
    #         select_one_column = True,
    #         allow_empty_column_selection = True,
    #         temperature = 0.763141,
    # )

    # SQA config
    config = TapasConfig()

    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = TapasModel(config)
    # model = TapasForQuestionAnswering(config)
    # model = TapasForSequenceClassification(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_tapas(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--tapas_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained TAPAS model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.tapas_config_file, args.pytorch_dump_path)
