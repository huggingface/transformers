####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
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

####################################################################################################

import argparse
import json
import os
import re
import zipfile

import torch


####################################################################################################


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


####################################################################################################


def convert_megatron_checkpoint(args, input_state_dict):
    # The converted output model.
    output_state_dict = {}

    # The number of heads.
    heads = 16
    # The hidden_size per head.
    hidden_size_per_head = 64

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to 50257 rows.
    word_embeddings = word_embeddings[:50257, :]
    # Truncate the embedding table to 50257 rows.
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # The position embeddings.
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    # Read the hidden dimension.
    hidden_size = pos_embeddings.size(0)
    # DEBUG.
    assert hidden_size == heads * hidden_size_per_head
    # Store the position embeddings.
    output_state_dict["transformer.wpe.weight"] = pos_embeddings

    # The transformer.
    transformer = lm["transformer"]

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "attention.dense": ".attn.c_proj.",
        "mlp.dense_h_to_4h": ".mlp.c_fc.",
        "mlp.dense_4h_to_h": ".mlp.c_proj.",
    }

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"transformer.h.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):

            ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # Transpose the QKV matrix.
        elif op_name == "attention.query_key_value" and weight_or_bias == "weight":

            # Insert a tensor of 1x1xDxD bias.
            zeros = torch.ones(1, 1, hidden_size, hidden_size)
            output_state_dict[layer_name + ".attn.bias"] = zeros

            # Insert a "dummy" tensor for masked_bias.
            masked_bias = torch.tensor(-1e4)
            output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

            # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
            out_val = val.transpose(0, 1)
            # Store.
            output_state_dict[layer_name + ".attn.c_attn.weight"] = out_val

        # Transpose the bias.
        elif op_name == "attention.query_key_value" and weight_or_bias == "bias":

            # Store. No change of shape.
            output_state_dict[layer_name + ".attn.c_attn.bias"] = val

        # Transpose the weights.
        elif weight_or_bias == "weight":

            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val.transpose(0, 1)

        # Copy the bias.
        elif weight_or_bias == "bias":

            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val

    # The final layernorm.
    output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_state_dict["lm_head.weight"] = word_embeddings

    # The config.
    output_config = {
        "activation_function": "gelu_new",
        "architectures": ["GPT2LMHeadModel"],
        "attn_pdrop": 0.1,
        "bos_token_id": 50256,
        "embd_pdrop": 0.1,
        "eos_token_id": 50256,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gpt2",
        "n_ctx": 1024,
        "n_embd": 1024,
        "n_head": 16,
        "n_layer": 24,
        "n_positions": 1024,
        "resid_pdrop": 0.1,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "vocab_size": 50257,
    }

    # It should be done!
    return output_state_dict, output_config


####################################################################################################


def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument("path_to_checkpoint", type=str, help="Path to the ZIP file containing the checkpoint")
    args = parser.parse_args()

    # Extract the basename.
    basename = os.path.dirname(args.path_to_checkpoint)

    # Load the model.
    print('Extracting PyTorch state dictionary from "{}"'.format(args.path_to_checkpoint))
    with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
        with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
            input_state_dict = torch.load(pytorch_dict, map_location="cpu")

    # Convert.
    print("Converting")
    output_state_dict, output_config = convert_megatron_checkpoint(args, input_state_dict)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Store the config to file.
    output_config_file = os.path.join(basename, "config.json")
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
