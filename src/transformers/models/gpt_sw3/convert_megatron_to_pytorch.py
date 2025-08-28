# Copyright 2022 The HuggingFace Inc. team and the AI-Sweden team. All rights reserved.
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
"""Convert GPT-SW3 megatron checkpoints to pytorch"""

import argparse
import os
from os.path import isfile

import torch

from transformers import GPT2Config


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def fix_query_key_value_ordering(param, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    # other versions store [num_heads * num_splits * hidden_size, :]
    saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
    param = param.view(*saved_shape)
    param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def convert_megatron_checkpoint(sd_megatron, config):
    """
    Converts a Megatron checkpoint to a HuggingFace GPT-SW3 checkpoint.
    """
    n_positions = config.n_positions
    layers = config.n_layer
    vocab_size = config.vocab_size
    heads = config.n_head
    hidden_size_per_head = config.n_embd // config.n_head

    word_embeddings = sd_megatron["model.language_model.embedding.word_embeddings.weight"][:vocab_size, :]
    sd_hf = {
        "transformer.wte.weight": word_embeddings,
        "transformer.wpe.weight": sd_megatron["model.language_model.embedding.position_embeddings.weight"],
        "transformer.ln_f.weight": sd_megatron["model.language_model.encoder.final_layernorm.weight"],
        "transformer.ln_f.bias": sd_megatron["model.language_model.encoder.final_layernorm.bias"],
    }

    pf = "model.language_model.encoder.layers."
    for i in range(layers):
        causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.bool))
        causal_mask = causal_mask.view(1, 1, n_positions, n_positions)
        sd_hf[f"transformer.h.{i}.attn.bias"] = causal_mask
        sd_hf[f"transformer.h.{i}.attn.masked_bias"] = torch.tensor(-1e4, dtype=torch.bfloat16)

        sd_hf[f"transformer.h.{i}.ln_1.weight"] = sd_megatron[f"{pf}{i}.input_layernorm.weight"]
        sd_hf[f"transformer.h.{i}.ln_1.bias"] = sd_megatron[f"{pf}{i}.input_layernorm.bias"]

        val1 = sd_megatron[f"{pf}{i}.self_attention.query_key_value.weight"]
        val1 = fix_query_key_value_ordering(val1, 3, heads, hidden_size_per_head)
        sd_hf[f"transformer.h.{i}.attn.c_attn.weight"] = val1.transpose(0, 1).contiguous()

        val2 = sd_megatron[f"{pf}{i}.self_attention.query_key_value.bias"]
        val2 = fix_query_key_value_ordering(val2, 3, heads, hidden_size_per_head)
        sd_hf[f"transformer.h.{i}.attn.c_attn.bias"] = val2

        sd_hf[f"transformer.h.{i}.attn.c_proj.weight"] = sd_megatron[f"{pf}{i}.self_attention.dense.weight"].transpose(
            0, 1
        )
        sd_hf[f"transformer.h.{i}.attn.c_proj.bias"] = sd_megatron[f"{pf}{i}.self_attention.dense.bias"]
        sd_hf[f"transformer.h.{i}.ln_2.weight"] = sd_megatron[f"{pf}{i}.post_attention_layernorm.weight"]
        sd_hf[f"transformer.h.{i}.ln_2.bias"] = sd_megatron[f"{pf}{i}.post_attention_layernorm.bias"]
        sd_hf[f"transformer.h.{i}.mlp.c_fc.weight"] = sd_megatron[f"{pf}{i}.mlp.dense_h_to_4h.weight"].transpose(0, 1)
        sd_hf[f"transformer.h.{i}.mlp.c_fc.bias"] = sd_megatron[f"{pf}{i}.mlp.dense_h_to_4h.bias"]
        sd_hf[f"transformer.h.{i}.mlp.c_proj.weight"] = sd_megatron[f"{pf}{i}.mlp.dense_4h_to_h.weight"].transpose(
            0, 1
        )
        sd_hf[f"transformer.h.{i}.mlp.c_proj.bias"] = sd_megatron[f"{pf}{i}.mlp.dense_4h_to_h.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    sd_hf["lm_head.weight"] = word_embeddings

    return sd_hf


def copy_config(config_hf, config_megatron):
    """Copy the config from Megatron to hf."""
    config_hf.vocab_size = 64000
    config_hf.n_positions = config_megatron["encoder_seq_length"]
    config_hf.n_embd = config_megatron["hidden_size"]
    config_hf.n_layer = config_megatron["num_layers"]
    config_hf.n_head = config_megatron["num_attention_heads"]
    config_hf.n_inner = config_megatron["ffn_hidden_size"]
    config_hf.activation_function = "gelu"
    config_hf.resid_pdrop = 0.1
    config_hf.embd_pdrop = 0.1
    config_hf.attn_pdrop = 0.1
    config_hf.layer_norm_epsilon = config_megatron["layernorm_epsilon"]  # 1e-5
    config_hf.initializer_range = config_megatron["init_method_std"]  # 0.02
    config_hf.apply_query_key_layer_scaling = config_megatron["apply_query_key_layer_scaling"]  # True
    config_hf.normalize_attention_scores = True
    config_hf.use_cache = True

    # This identifies the 6.7B (7B) model which uses a different tokenizer
    if config_megatron["hidden_size"] == 4096:
        config_hf.bos_token_id = 1  # <|endoftext|>
        config_hf.eos_token_id = 1  # <|endoftext|>
        config_hf.pad_token_id = 0  # <unk>
    else:
        config_hf.bos_token_id = 2  # <s>
        config_hf.eos_token_id = 3  # <|endoftext|>
        config_hf.pad_token_id = 0  # <pad>

    return config_hf


def main(args):
    print(args)

    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    if isfile(checkpoint_path):
        raise FileNotFoundError(f"ERROR! could not find file {checkpoint_path}")

    # Load the model.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load the config.
    config_megatron = checkpoint["hyper_parameters"]["cfg"]
    config_hf = GPT2Config()
    config_hf = copy_config(config_hf=config_hf, config_megatron=config_megatron)
    config_hf.architectures = ["GPT2LMHeadModel"]

    sd_megatron = checkpoint["state_dict"]

    # Convert.
    print("Converting")
    sd_hf = convert_megatron_checkpoint(sd_megatron, config_hf)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, sd_hf)

    config_hf.tokenizer_class = "GPTSw3Tokenizer"

    # Store the config to file.
    print("Saving config")
    config_hf.save_pretrained(save_path)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(save_path, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(sd_hf, output_checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="e.g. megatron_gpt--val_loss=2.42-step=38000-consumed_samples=54720000",
    )
    parser.add_argument("--save_path", type=str, required=True, help="e.g. /home/user/gpt-sw3/hf")
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    _args = parser.parse_args()
    main(_args)
