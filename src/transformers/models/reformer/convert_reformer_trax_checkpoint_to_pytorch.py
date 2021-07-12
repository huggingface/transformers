# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert Reformer checkpoint."""


import argparse
import pickle

import numpy as np
import torch

from transformers import ReformerConfig, ReformerModelWithLMHead
from transformers.utils import logging


logging.set_verbosity_info()


def set_param(torch_layer, weight, bias=None):
    # set parameter of one layer
    assert torch_layer.weight.shape == weight.shape, f"{torch_layer} layer.weight does not match"
    torch_layer.weight = torch.nn.Parameter(weight)
    if bias is not None:
        assert torch_layer.bias.shape == bias.shape, f"{torch_layer} layer.bias does not match"
        torch_layer.bias = torch.nn.Parameter(bias)


def set_layer_weights_in_torch_lsh(weights, torch_layer, hidden_size):
    # set torch weights for 1-to-1 comparison
    np_query_key = np.asarray(weights[0])
    np_value = np.asarray(weights[1])
    np_dense = np.asarray(weights[2])

    set_param(
        torch_layer.self_attention.query_key,
        torch.tensor(np_query_key).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.self_attention.value,
        torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.output.dense,
        torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
    )


def set_layer_weights_in_torch_local(weights, torch_layer, hidden_size):
    # set torch weights for 1-to-1 comparison
    np_query = np.asarray(weights[0])
    np_key = np.asarray(weights[1])
    np_value = np.asarray(weights[2])
    np_dense = np.asarray(weights[3])

    set_param(
        torch_layer.self_attention.query,
        torch.tensor(np_query).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.self_attention.key,
        torch.tensor(np_key).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.self_attention.value,
        torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    set_param(
        torch_layer.output.dense,
        torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
    )


def set_block_weights_in_torch(weights, torch_block, hidden_size):
    # layernorm 1
    layer_norm_1 = weights[0][0][0]
    layer_norm_1_weight = np.asarray(layer_norm_1[0])
    layer_norm_1_bias = np.asarray(layer_norm_1[1])
    set_param(
        torch_block.attention.layer_norm,
        torch.tensor(layer_norm_1_weight),
        torch.tensor(layer_norm_1_bias),
    )

    # lsh weights + output
    attn_weights = weights[0][1]
    if len(attn_weights) < 4:
        set_layer_weights_in_torch_lsh(attn_weights, torch_block.attention, hidden_size)
    else:
        set_layer_weights_in_torch_local(attn_weights, torch_block.attention, hidden_size)

    # intermediate weighs
    intermediate_weights = weights[2][0][1][2]

    # Chunked Feed Forward
    if len(intermediate_weights) == 4:
        intermediate_weights = intermediate_weights[2]

    # layernorm 2
    layer_norm_2_weight = np.asarray(intermediate_weights[0][0])
    layer_norm_2_bias = np.asarray(intermediate_weights[0][1])
    set_param(
        torch_block.feed_forward.layer_norm,
        torch.tensor(layer_norm_2_weight),
        torch.tensor(layer_norm_2_bias),
    )

    # intermediate dense
    inter_dense_weight = np.asarray(intermediate_weights[1][0])
    inter_dense_bias = np.asarray(intermediate_weights[1][1])
    set_param(
        torch_block.feed_forward.dense.dense,
        torch.tensor(inter_dense_weight).transpose(0, 1).contiguous(),
        torch.tensor(inter_dense_bias),
    )

    # intermediate out
    out_dense_weight = np.asarray(intermediate_weights[4][0])
    out_dense_bias = np.asarray(intermediate_weights[4][1])
    set_param(
        torch_block.feed_forward.output.dense,
        torch.tensor(out_dense_weight).transpose(0, 1).contiguous(),
        torch.tensor(out_dense_bias),
    )


def set_model_weights_in_torch(weights, torch_model, hidden_size):
    # reformer model
    torch_model_reformer = torch_model.reformer

    # word embeds
    word_embeddings = np.asarray(weights[1])
    set_param(
        torch_model_reformer.embeddings.word_embeddings,
        torch.tensor(word_embeddings),
    )

    if isinstance(weights[3], tuple):
        position_embeddings = torch_model_reformer.embeddings.position_embeddings
        for emb_idx in range(len(position_embeddings.weights)):
            emb_weights = np.asarray(weights[3][emb_idx][0])
            assert (
                position_embeddings.weights[emb_idx].shape == emb_weights.shape
            ), f"{position_embeddings[emb_idx]} emb does not match"
            position_embeddings.weights[emb_idx] = torch.nn.Parameter(torch.tensor(emb_weights))

    trax_layer_weights = weights[5]
    assert len(torch_model_reformer.encoder.layers) * 4 == len(
        trax_layer_weights
    ), "HF and trax model do not have the same number of layers"
    for layer_idx, layer in enumerate(torch_model_reformer.encoder.layers):
        block_weights = trax_layer_weights[4 * layer_idx : 4 * (layer_idx + 1)]
        set_block_weights_in_torch(block_weights, layer, hidden_size)

    # output layer norm
    layer_norm_out_weight = np.asarray(weights[7][0])
    layer_norm_out_bias = np.asarray(weights[7][1])
    set_param(
        torch_model_reformer.encoder.layer_norm,
        torch.tensor(layer_norm_out_weight),
        torch.tensor(layer_norm_out_bias),
    )

    # output embeddings
    output_embed_weights = np.asarray(weights[9][0])
    output_embed_bias = np.asarray(weights[9][1])
    set_param(
        torch_model.lm_head.decoder,
        torch.tensor(output_embed_weights).transpose(0, 1).contiguous(),
        torch.tensor(output_embed_bias),
    )


def convert_trax_checkpoint_to_pytorch(trax_model_pkl_path, config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = ReformerConfig.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = ReformerModelWithLMHead(config)

    with open(trax_model_pkl_path, "rb") as f:
        model_weights = pickle.load(f)["weights"]

    set_model_weights_in_torch(model_weights, model, config.hidden_size)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--trax_model_pkl_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained Reformer model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_trax_checkpoint_to_pytorch(args.trax_model_pkl_path, args.config_file, args.pytorch_dump_path)
