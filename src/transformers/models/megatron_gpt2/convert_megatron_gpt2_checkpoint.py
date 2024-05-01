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

#
# Note: If when running this conversion script you're getting an exception:
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# you need to tell python where to find the clone of Megatron-LM, e.g.:
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py ...
#
# if you already have it cloned elsewhere, simply adjust the path to the existing path
#
# If the training was done using a Megatron-LM fork, e.g.,
# https://github.com/microsoft/Megatron-DeepSpeed/ then chances are that you need to have that one
# in your path, i.e., /path/to/Megatron-DeepSpeed/
#

import argparse
import os
import re
import zipfile

import torch

from transformers import AutoTokenizer, GPT2Config


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


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


####################################################################################################


def convert_megatron_checkpoint(args, input_state_dict, config):
    # The converted output model.
    output_state_dict = {}

    # old versions did not store training args
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # do not make the user write a config file when the exact dimensions/sizes are already in the checkpoint
        # from pprint import pprint
        # pprint(vars(ds_args))

        config.vocab_size = ds_args.padded_vocab_size
        config.n_positions = ds_args.max_position_embeddings
        config.n_embd = ds_args.hidden_size
        config.n_layer = ds_args.num_layers
        config.n_head = ds_args.num_attention_heads
        config.n_inner = ds_args.ffn_hidden_size
        # pprint(config)

    # The number of heads.
    heads = config.n_head
    # The hidden_size per head.
    hidden_size_per_head = config.n_embd // config.n_head
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to vocab_size rows.
    word_embeddings = word_embeddings[: config.vocab_size, :]
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # The position embeddings.
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    # Read the causal mask dimension (seqlen). [max_sequence_length, hidden_size]
    n_positions = pos_embeddings.size(0)
    if n_positions != config.n_positions:
        raise ValueError(
            f"pos_embeddings.max_sequence_length={n_positions} and config.n_positions={config.n_positions} don't match"
        )
    # Store the position embeddings.
    output_state_dict["transformer.wpe.weight"] = pos_embeddings

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "attention.dense": ".attn.c_proj.",
        "self_attention.dense": ".attn.c_proj.",
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
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "weight":
            # Insert a tensor of 1x1xDxD bias.
            causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(
                1, 1, n_positions, n_positions
            )
            output_state_dict[layer_name + ".attn.bias"] = causal_mask

            # Insert a "dummy" tensor for masked_bias.
            masked_bias = torch.tensor(-1e4, dtype=torch.float16)
            output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Megatron stores (3*D) x D but transformers-GPT2 expects D x 3*D.
            out_val = out_val.transpose(0, 1).contiguous()
            # Store.
            output_state_dict[layer_name + ".attn.c_attn.weight"] = out_val

        # Transpose the bias.
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "bias":
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Store. No change of shape.
            output_state_dict[layer_name + ".attn.c_attn.bias"] = out_val

        # Transpose the weights.
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val.transpose(0, 1)

        # Copy the bias.
        elif weight_or_bias == "bias":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val

    # DEBUG.
    assert config.n_layer == layer_idx + 1

    # The final layernorm.
    output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_state_dict["lm_head.weight"] = word_embeddings

    # It should be done!
    return output_state_dict


####################################################################################################


def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    args = parser.parse_args()

    # Extract the basename.
    basename = os.path.dirname(args.path_to_checkpoint)

    # Load the model.
    # the .zip is very optional, let's keep it for backward compatibility
    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    ds_args = input_state_dict.get("args", None)

    # Read the config, or default to the model released by NVIDIA.
    if args.config_file == "":
        if ds_args is not None:
            if ds_args.bias_gelu_fusion:
                activation_function = "gelu_fast"
            elif ds_args.openai_gelu:
                activation_function = "gelu_new"
            else:
                activation_function = "gelu"
        else:
            # in the very early days this used to be "gelu_new"
            activation_function = "gelu_new"

        # Spell out all parameters in case the defaults change.
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            activation_function=activation_function,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
        )
    else:
        config = GPT2Config.from_json_file(args.config_file)

    config.architectures = ["GPT2LMHeadModel"]

    # Convert.
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Add tokenizer class info to config
    # see https://github.com/huggingface/transformers/issues/13906)
    if ds_args is not None:
        tokenizer_type = ds_args.tokenizer_type
        if tokenizer_type == "GPT2BPETokenizer":
            tokenizer_model_name = "openai-community/gpt2"
        elif tokenizer_type == "PretrainedFromHF":
            tokenizer_model_name = ds_args.tokenizer_name_or_path
        else:
            raise ValueError(f"Unrecognized tokenizer_type {tokenizer_type}")
    else:
        tokenizer_model_name = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(basename)

    # Save tokenizer based on args
    print(f"Adding {tokenizer_class} tokenizer files")
    tokenizer.save_pretrained(basename)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
