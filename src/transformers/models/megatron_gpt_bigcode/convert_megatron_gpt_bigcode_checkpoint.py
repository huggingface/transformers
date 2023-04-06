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

import torch

from transformers import GPTBigCodeConfig, GPTBigCodeLMHeadModel, GPTBigCodeModel


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

# The simple map of names for "automated" rules.
NAME_MAP = {
    "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".attn.c_proj.",
    "mlp.dense_h_to_4h": ".mlp.c_fc.",
    "mlp.dense_4h_to_h": ".mlp.c_proj.",
    "self_attention.query_key_value": ".attn.c_attn.",
    "self_attention.query": ".attn.q_attn.",
    "self_attention.key_value": ".attn.kv_attn.",
}


def convert_megatron_checkpoint(input_state_dict, merge_qkv):
    # The converted output model.
    output_state_dict = {}
    ds_args = input_state_dict["args"]

    if ds_args is not None:
        if ds_args.bias_gelu_fusion:
            activation_function = "gelu_pytorch_tanh"
        elif ds_args.openai_gelu:
            activation_function = "gelu_new"
        else:
            activation_function = "gelu"
    else:
        # in the very early days this used to be "gelu_new"
        activation_function = "gelu_new"

    if ds_args.attention_head_type == "multihead":
        attention_type = 1
    else:
        assert ds_args.attention_head_type == "multiquery"
        attention_type = 2 if merge_qkv else 3

    attention_softmax_in_fp32 = ds_args.attention_softmax_in_fp32 or ds_args.apply_query_key_layer_scaling

    # Spell out all parameters in case the defaults change.
    config = GPTBigCodeConfig(
        architectures=["GPTBigCodeLMHeadModel"],
        vocab_size=ds_args.padded_vocab_size,
        n_positions=ds_args.max_position_embeddings,
        n_embd=ds_args.hidden_size,
        n_layer=ds_args.num_layers,
        n_head=ds_args.num_attention_heads,
        n_inner=ds_args.ffn_hidden_size,
        activation_function=activation_function,
        attention_type=attention_type,
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
        attention_softmax_in_fp32=attention_softmax_in_fp32,
        scale_attention_softmax_in_fp32=True,
    )

    # from pprint import pprint
    # pprint(vars(ds_args))
    # pprint(config)

    # Megatron-LM checkpoint version
    checkpoint_version = input_state_dict["checkpoint_version"]
    if checkpoint_version < 2.0:
        raise NotImplementedError(f"Checkpoint version {checkpoint_version} not supported.")

    # The model.
    model = input_state_dict["model"]["language_model"]

    # The word embeddings, truncated to to vocab_size rows.
    word_embeddings = model["embedding"]["word_embeddings"]["weight"][: config.vocab_size, :]
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # The position embeddings.
    output_state_dict["transformer.wpe.weight"] = model["embedding"]["position_embeddings"]["weight"]

    # The transformer.
    transformer = model["transformer"] if "transformer" in model else model["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

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

        # Concatenate QKV matrix.
        elif merge_qkv and (op_name == "self_attention.key_value"):
            # Query is before key_value in the dict.
            query = output_state_dict.pop(layer_name + ".attn.q_attn." + weight_or_bias)
            out_val = torch.cat([query, val], dim=0)
            output_state_dict[layer_name + ".attn.c_attn." + weight_or_bias] = out_val

        # Copy the parameters.
        else:
            output_state_dict[layer_name + NAME_MAP[op_name] + weight_or_bias] = val

    # DEBUG.
    assert config.n_layer == layer_idx + 1

    # The final layernorm.
    output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_state_dict["lm_head.weight"] = word_embeddings

    # It should be done!
    return config, output_state_dict


####################################################################################################


def main(argv=None):
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "--no_merge_qkv",
        dest="merge_qkv",
        action="store_false",
        help="Do not merge the query and key_value tensors (MQA).",
    )
    parser.add_argument(
        "--custom_model",
        action="store_true",
        help="Save as custom model so it can be used with huggingface transformers.",
    )
    parser.add_argument(
        "--save_dir", help="Path where the converted model is saved. Will use the checkpoint directory if not provided"
    )
    args = parser.parse_args(argv)

    # Extract the basename.
    basename = args.save_dir or os.path.dirname(args.path_to_checkpoint)

    # Load the model.
    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    # Convert.
    print("Converting")
    config, output_state_dict = convert_megatron_checkpoint(input_state_dict, args.merge_qkv)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    if args.custom_model:
        # Save custom model
        GPTBigCodeConfig.register_for_auto_class()
        GPTBigCodeModel.register_for_auto_class("AutoModelForCausalLM")
        hf_model = GPTBigCodeLMHeadModel(config)
        hf_model.load_state_dict(output_state_dict)
        hf_model.save_pretrained(basename)

    else:
        # Store the config to file.
        print("Saving config")
        config.save_pretrained(basename)

        # Store the state_dict to file.
        output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
        print(f'Saving checkpoint to "{output_checkpoint_file}"')
        torch.save(output_state_dict, output_checkpoint_file)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
