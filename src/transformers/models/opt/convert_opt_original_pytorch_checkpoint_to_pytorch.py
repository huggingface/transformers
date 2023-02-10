# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert OPT checkpoint."""


import argparse
from pathlib import Path

import torch

from transformers import OPTConfig, OPTModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def load_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "model" in sd.keys():
        sd = torch.load(checkpoint_path, map_location="cpu")["model"]

    # pop unnecessary weights
    keys_to_delete = [
        "decoder.version",
        "decoder.output_projection.weight",
    ]
    for key in keys_to_delete:
        if key in sd:
            sd.pop(key)

    keys_to_rename = {
        "decoder.project_in_dim.weight": "decoder.project_in.weight",
        "decoder.project_out_dim.weight": "decoder.project_out.weight",
        "decoder.layer_norm.weight": "decoder.final_layer_norm.weight",
        "decoder.layer_norm.bias": "decoder.final_layer_norm.bias",
    }
    for old_key, new_key in keys_to_rename.items():
        if old_key in sd:
            sd[new_key] = sd.pop(old_key)

    keys = list(sd.keys())
    for key in keys:
        if ".qkj_proj." in key:
            value = sd[key]
            # We split QKV in seperate Q,K,V

            q_name = key.replace(".qkv_proj.", ".q_proj.")
            k_name = key.replace(".qkv_proj.", ".k_proj.")
            v_name = key.replace(".qkv_proj.", ".v_proj.")

            depth = value.shape[0]
            assert depth % 3 == 0
            # `SequeuceParallelTransformerBlock` has QKV weight is separated in K,V,Q despite the naming:
            # https://cs.github.com/facebookresearch/metaseq/blob/51871bd73cd04c038f239ea2a26db1d7f6b37927/metaseq/modules/sequence_parallel_transformer_layer.py#L97
            k, v, q = torch.split(value, depth // 3, dim=0)

            sd[q_name] = q
            sd[k_name] = k
            sd[v_name] = v
            del sd[key]

    return sd


@torch.no_grad()
def convert_opt_checkpoint(checkpoint_path, pytorch_dump_folder_path, config=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    state_dict = load_checkpoint(checkpoint_path)

    if config is not None:
        config = OPTConfig.from_pretrained(config)
    else:
        config = OPTConfig()

    model = OPTModel(config).half().eval()
    model.load_state_dict(state_dict)

    # Check results
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq_path",
        type=str,
        help=(
            "path to fairseq checkpoint in correct format. You can find all checkpoints in the correct format here:"
            " https://huggingface.co/models?other=opt_metasq"
        ),
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--hf_config", default=None, type=str, help="Define HF config.")
    args = parser.parse_args()
    convert_opt_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, config=args.hf_config)
