# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import torch
from utils import logger


def _check_config_is_compatible(saved_config, config):
    if (saved_config.num_hidden_layers != config.num_hidden_layers or
            saved_config.hidden_size != config.hidden_size or
            saved_config.num_attention_heads != config.num_attention_heads):
        raise RuntimeError("Checkpoint being loaded does not match model definition.")


def _get_filename_or_prefix(config, step=None):
    phase = config.dataset
    model_type = config.model_type
    layers = config.num_hidden_layers
    hidden = config.hidden_size
    heads = config.num_attention_heads
    seqlen = config.sequence_length
    prefix = f"{phase}_{model_type}_L_{layers}_H_{hidden}_A_{heads}_seqlen_{seqlen}"
    if step is not None:
        filename = f"{prefix}_step_{step}.pt"
        return filename
    return prefix


def _load_checkpoint_from_file(config):
    abs_path_ckpt = os.path.abspath(config.checkpoint_file)

    # Return checkpoint if valid
    if os.path.isfile(abs_path_ckpt):
        try:
            checkpoint = torch.load(abs_path_ckpt)
            return checkpoint
        except Exception as e:
            logger(f"Failed with exception {e}.")
    else:
        raise RuntimeError("Please specify a PyTorch checkpoint file.")


def checkpoints_exist(config):
    path = os.path.abspath(config.checkpoint_dir)
    if os.path.exists(path):
        # All checkpoint files
        files = glob.glob(f"{os.path.join(path, '*.pt')}")
        if len(files) > 0:
            return True
    return False


def restore_checkpoint(config):
    checkpoint = _load_checkpoint_from_file(config)
    _check_config_is_compatible(checkpoint["config"], config)
    return checkpoint


def save_checkpoint(config, model, optimizer, step, metrics=None):
    if config.checkpoint_dir:
        abs_pathd = os.path.abspath(config.checkpoint_dir)
        os.makedirs(abs_pathd, exist_ok=True)
        filename = _get_filename_or_prefix(config, step)
        save_path = os.path.join(abs_pathd, filename)
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        logger(f"Saving checkpoint for step {step} to: {save_path}\n")
        torch.save({
            "step": step,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics,
            "config": config
        }, save_path)
