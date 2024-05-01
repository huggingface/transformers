# coding=utf-8
# Copyright 2022 The Trajectory Transformers paper authors and The HuggingFace Inc. team. All rights reserved.
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
""" TrajectoryTransformer pytorch checkpoint conversion"""

import torch
import trajectory.utils as utils

from transformers import TrajectoryTransformerModel


class Parser(utils.Parser):
    dataset: str = "halfcheetah-medium-expert-v2"
    config: str = "config.offline"


def convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch(logbase, dataset, loadpath, epoch, device):
    """Converting Sequential blocks to ModuleList"""

    gpt, gpt_epoch = utils.load_model(logbase, dataset, loadpath, epoch=epoch, device=device)
    trajectory_transformer = TrajectoryTransformerModel(gpt.config)

    trajectory_transformer.tok_emb.load_state_dict(gpt.tok_emb.state_dict())
    trajectory_transformer.pos_emb = gpt.pos_emb
    trajectory_transformer.drop.load_state_dict(gpt.drop.state_dict())
    trajectory_transformer.ln_f.load_state_dict(gpt.ln_f.state_dict())
    trajectory_transformer.head.load_state_dict(gpt.head.state_dict())

    for i, block in enumerate(gpt.blocks):
        trajectory_transformer.blocks[i].ln1.load_state_dict(gpt.blocks[i].ln1.state_dict())
        trajectory_transformer.blocks[i].ln2.load_state_dict(gpt.blocks[i].ln2.state_dict())
        trajectory_transformer.blocks[i].attn.load_state_dict(gpt.blocks[i].attn.state_dict())

        trajectory_transformer.blocks[i].l1.load_state_dict(gpt.blocks[i].mlp[0].state_dict())
        trajectory_transformer.blocks[i].act.load_state_dict(gpt.blocks[i].mlp[1].state_dict())
        trajectory_transformer.blocks[i].l2.load_state_dict(gpt.blocks[i].mlp[2].state_dict())
        trajectory_transformer.blocks[i].drop.load_state_dict(gpt.blocks[i].mlp[3].state_dict())

    torch.save(trajectory_transformer.state_dict(), "pytorch_model.bin")


if __name__ == "__main__":
    """
    To run this script you will need to install the original repository to run the original model. You can find it
    here: https://github.com/jannerm/trajectory-transformer From this repository code you can also download the
    original pytorch checkpoints.

    Run with the command:

    ```sh
    >>> python convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py --dataset <dataset_name>
    ...     --gpt_loadpath <path_to_original_pytorch_checkpoint>
    ```
    """

    args = Parser().parse_args("plan")
    convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch(
        args.logbase, args.dataset, args.gpt_loadpath, args.gpt_epoch, args.device
    )
