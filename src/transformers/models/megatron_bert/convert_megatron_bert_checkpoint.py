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

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Store the word embeddings.
    output_state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings

    # The position embeddings.
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    # Trained for 512 x 1024.
    assert pos_embeddings.size(0) == 512 and pos_embeddings.size(1) == 1024
    # Store the position embeddings.
    output_state_dict["bert.embeddings.position_embeddings.weight"] = pos_embeddings

    # The token-type embeddings.
    tokentype_embeddings = embeddings["tokentype_embeddings"]["weight"]
    # Store the position embeddings.
    output_state_dict["bert.embeddings.token_type_embeddings.weight"] = tokentype_embeddings

    # The transformer.
    transformer = lm["transformer"]

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "attention.dense": ".attention.output.dense.",
        "mlp.dense_h_to_4h": ".intermediate.dense.",
        "mlp.dense_4h_to_h": ".output.dense.",
    }

    # Keep track of the attention/query/value tensor.
    attention_qkv_weight = None

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
        layer_name = f"bert.encoder.layer.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):

            ln_name = "attention.ln" if op_name.startswith("input") else "ln"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # Transpose the QKV matrix.
        elif op_name == "attention.query_key_value" and weight_or_bias == "weight":

            # Make sure the QKV pointer is nil.
            assert attention_qkv_weight is None, ""

            # Store the tensor as we need the bias as well to interleave QKV and biases.
            attention_qkv_weight = val

        # Transpose the bias.
        elif op_name == "attention.query_key_value" and weight_or_bias == "bias":

            # Make sure we read the weight tensor.
            assert attention_qkv_weight is not None, ""

            # Split the QKV matrix into Q, K and V. Megatron stores Q,K,V interleaved.
            q = attention_qkv_weight[0 * 1024 : 1 * 1024, :]
            k = attention_qkv_weight[1 * 1024 : 2 * 1024, :]
            v = attention_qkv_weight[2 * 1024 : 3 * 1024, :]

            # Split the bias.
            q_bias = val[0 * 1024 : 1 * 1024]
            k_bias = val[1 * 1024 : 2 * 1024]
            v_bias = val[2 * 1024 : 3 * 1024]

            # Store.
            output_state_dict[f"{layer_name}.attention.self.query.weight"] = q
            output_state_dict[f"{layer_name}.attention.self.query.bias"] = q_bias
            output_state_dict[f"{layer_name}.attention.self.key.weight"] = k
            output_state_dict[f"{layer_name}.attention.self.key.bias"] = k_bias
            output_state_dict[f"{layer_name}.attention.self.value.weight"] = v
            output_state_dict[f"{layer_name}.attention.self.value.bias"] = v_bias

            # Clear the stored tensor.
            attention_qkv_weight = None

        # Copy weights and biases as is.
        elif weight_or_bias in ["weight", "bias"]:

            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + weight_or_bias] = val

    # The final layernorm.
    output_state_dict["bert.encoder.ln.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["bert.encoder.ln.bias"] = transformer["final_layernorm.bias"]

    # The config.
    output_config = {
        "vocab_size": word_embeddings.size(0),
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "hidden_act": "gelu_new",
        "intermediate_size": 4096,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.2,
        "layer_norm_eps": 1e-12,
        "gradient_checkpointing": False,
        "position_embedding_type": "absolute",
        "use_cache": False,
    }

    # The pooler.
    pooler = lm["pooler"]

    # Store the matrix and the bias.
    output_state_dict["bert.pooler.dense.weight"] = pooler["dense.weight"]
    output_state_dict["bert.pooler.dense.bias"] = pooler["dense.bias"]

    # The LM head from Megatron (for RACE).
    lm_head = model["lm_head"]

    # The transform matrix.
    output_state_dict["cls.predictions.transform.dense.weight"] = lm_head["dense.weight"]
    output_state_dict["cls.predictions.transform.dense.bias"] = lm_head["dense.bias"]

    # The transform LN.
    output_state_dict["cls.predictions.transform.LayerNorm.weight"] = lm_head["layernorm.weight"]
    output_state_dict["cls.predictions.transform.LayerNorm.bias"] = lm_head["layernorm.bias"]

    # For the decoder, we replicate the weights.
    output_state_dict["cls.predictions.decoder.weight"] = word_embeddings
    output_state_dict["cls.predictions.bias"] = lm_head["bias"]

    # The classifier from Megatron (for MLNI).
    binary_head = model["binary_head"]

    # Store the classifier.
    output_state_dict["cls.seq_relationship.weight"] = binary_head["weight"]
    output_state_dict["cls.seq_relationship.bias"] = binary_head["bias"]

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
    print(f'Extracting PyTorch state dictionary from "{args.path_to_checkpoint}"')
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
