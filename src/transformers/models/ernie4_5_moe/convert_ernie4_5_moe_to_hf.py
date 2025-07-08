# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
"""Converts a Ernie4.5 MoE model to Hugging Face format."""

# TODO: doesn't work + shared experts are done wrong atm

import argparse

from transformers import Ernie4_5_MoEForCausalLM, Ernie4_5Tokenizer


def save_ernie_model(model, save_dir_path, is_tied_weights=True):
    # workaround for shared pointers not detected during `save_pretrained`
    if is_tied_weights:
        model._keys_to_ignore_on_save = ['lm_head.weight']
    model.save_pretrained(save_dir_path)


# TODO: config conversion - only base 21B handled atm
def convert_ernie_4_5_model_to_hf(checkpoint_path):
    model = Ernie4_5_MoEForCausalLM.from_pretrained(checkpoint_path, device_map="cpu")
    model_dict = model.state_dict()

    # meta info for RoPE conversion
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    hidden_size = model.config.hidden_size

    num_heads = model.config.num_attention_heads
    dim_q = num_heads * head_dim
    num_kv_heads = model.config.num_key_value_heads
    dim_kv = num_kv_heads * head_dim

    # rotates to hf format (from even/odd to half/half)
    def permute(w, n_heads, dim1=dim_q, dim2=hidden_size):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for key, tensor in model_dict.items():
        if "q_proj" in key:
            model_dict[key] = permute(tensor, n_heads=num_heads)
        elif "k_proj" in key:
            model_dict[key] = permute(tensor, n_heads=num_kv_heads, dim1=dim_kv)
        else:
            model_dict[key] = tensor

    # load converted weights
    model.load_state_dict(model_dict, assign=True)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", type=str, default="baidu/ERNIE-4.5-21B-A3B-PT", help="Path to the checkpoint"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="AntonV/ERNIE-4.5-21B-A3B-PT",
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--convert_tokenizer",
        type=bool,
        default=True,
        help="Whether or not the tokenizer should be converted along with the model.",
    )
    args = parser.parse_args()

    model = convert_ernie_4_5_model_to_hf(args.checkpoint_path)
    if args.convert_tokenizer:
        tokenizer = Ernie4_5Tokenizer.from_pretrained(args.checkpoint_path, padding_side="left")
        tokenizer.save_pretrained(args.pytorch_dump_folder_path)

    save_ernie_model(model, args.pytorch_dump_folder_path, is_tied_weights=True) # TODO: change for 300B
    print(f"Saved converted checkpoint to {args.pytorch_dump_folder_path}")
