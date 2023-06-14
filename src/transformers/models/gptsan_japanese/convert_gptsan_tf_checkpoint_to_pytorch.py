# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

"""Convert GPTSANJapanese checkpoints from the original repository to pytorch model."""

import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import torch


def convert_tf_gptsan_to_pt(args):
    parameter_file = os.path.join(args.tf_model_dir, "parameters.json")
    params = json.loads(open(parameter_file).read())
    if not params:
        raise ValueError(
            f"It seems that the json file at {parameter_file} is empty. Make sure you have a correct json file."
        )
    if not args.output.endswith(".pt"):
        args.output = args.output + ".pt"
    new_state = OrderedDict()
    with tf.device("/CPU:0"):
        reader = tf.train.load_checkpoint(args.tf_model_dir)
        shapes = reader.get_variable_to_shape_map()
        for key_name in shapes.keys():
            vnp = reader.get_tensor(key_name).astype(np.float16)
            if key_name.endswith("/adam_m") or key_name.endswith("/adam_v"):
                continue
            if key_name.startswith("pasts/"):
                if key_name.startswith("pasts/mlp"):
                    player = int(key_name[9])
                elif key_name.startswith("pasts/out"):
                    player = 8
                name = "model.sqout.%d.weight" % (player * 2)  # enter to nn.Sequencial with Tanh, so 2 at a time
                state = vnp.transpose([1, 0]).copy()  # Mesh-Tensorflow is a diagonal matrix
                new_state[name] = torch.tensor(state)
            elif key_name.startswith("model/moe"):
                player = int(key_name[9:].split("/")[0])
                if key_name.endswith("/switch_gating/kernel"):
                    name = "model.blocks.%d.feed_forward.mlp.router.classifier.weight" % player
                    state = vnp.transpose([1, 0]).copy()  # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
                elif key_name.endswith("/softmlp/kernel"):
                    name = "model.blocks.%d.feed_forward.soft_bypass_mlp.weight" % player
                    state = vnp.transpose([1, 0]).copy()  # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
                elif key_name.endswith("/wo/kernel") or key_name.endswith("/wi/kernel"):
                    nlayer = key_name[-9:-7]
                    for i in range(16):
                        name = "model.blocks.%d.feed_forward.mlp.experts.expert_%d.%s.weight" % (player, i, nlayer)
                        state = (
                            vnp[i].transpose([1, 0]).copy()
                        )  # In Mesh-Tensorflow, it is one array, so it is divided
                        new_state[name] = torch.tensor(state)
            elif key_name.startswith("model/mlp"):
                player = int(key_name[9:].split("/")[0])
                if key_name.endswith("/p1/kernel"):
                    name = "model.blocks.%d.feed_forward.mlp.wi.weight" % player
                    state = vnp.transpose([1, 0]).copy()  # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
                elif key_name.endswith("/p1/bias"):
                    name = "model.blocks.%d.feed_forward.mlp.wi.bias" % player
                    state = vnp.copy()  # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
                elif key_name.endswith("/p2/kernel"):
                    name = "model.blocks.%d.feed_forward.mlp.wo.weight" % player
                    state = vnp.transpose([1, 0]).copy()  # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
                elif key_name.endswith("/p2/bias"):
                    name = "model.blocks.%d.feed_forward.mlp.wo.bias" % player
                    state = vnp.copy()  # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
            elif key_name.startswith("model/ln"):
                player = int(key_name[8:].split("/")[0])
                if key_name.endswith("/b"):
                    name = "model.blocks.%d.feed_forward.norm.bias" % player
                    state = vnp.copy()  # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
                elif key_name.endswith("/g"):
                    name = "model.blocks.%d.feed_forward.norm.weight" % player
                    state = vnp.copy()  # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
            elif key_name.startswith("model/att"):
                player = int(key_name[9:].split("/")[0])
                if key_name.endswith("/qkv/kernel"):
                    state = vnp.copy()  # Compute same dimension as Mesh-tensorflow using einsum
                    state_q = state[:, 0, :, :]
                    state_k = state[:, 1, :, :]
                    state_v = state[:, 2, :, :]
                    state_q = (
                        state_q.reshape([state_q.shape[0], state_q.shape[1] * state_q.shape[2]])
                        .transpose([1, 0])
                        .copy()
                    )  # Mesh-Tensorflow is a diagonal matrix
                    state_k = (
                        state_k.reshape([state_k.shape[0], state_k.shape[1] * state_k.shape[2]])
                        .transpose([1, 0])
                        .copy()
                    )  # Mesh-Tensorflow is a diagonal matrix
                    state_v = (
                        state_v.reshape([state_v.shape[0], state_v.shape[1] * state_v.shape[2]])
                        .transpose([1, 0])
                        .copy()
                    )  # Mesh-Tensorflow is a diagonal matrix
                    name = "model.blocks.%d.self_attn.self_attn.q_proj.weight" % player
                    new_state[name] = torch.tensor(state_q)
                    name = "model.blocks.%d.self_attn.self_attn.k_proj.weight" % player
                    new_state[name] = torch.tensor(state_k)
                    name = "model.blocks.%d.self_attn.self_attn.v_proj.weight" % player
                    new_state[name] = torch.tensor(state_v)
                elif key_name.endswith("/o/kernel"):
                    name = "model.blocks.%d.self_attn.self_attn.out_proj.weight" % player
                    state = (
                        vnp.reshape([vnp.shape[0] * vnp.shape[1], vnp.shape[2]]).transpose([1, 0]).copy()
                    )  # Mesh-Tensorflow is a diagonal matrix
                    new_state[name] = torch.tensor(state)
            elif key_name.startswith("model/an"):
                player = int(key_name[8:].split("/")[0])
                if key_name.endswith("/b"):
                    name = "model.blocks.%d.self_attn.norm.bias" % player
                    state = vnp.copy()  # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
                elif key_name.endswith("/g"):
                    name = "model.blocks.%d.self_attn.norm.weight" % player
                    state = vnp.copy()  # same because it is one dimensional
                    new_state[name] = torch.tensor(state)
            elif (
                key_name.startswith("model/wte")
                or key_name.startswith("model/wpe")
                or key_name.startswith("model/ete")
            ):
                nlayer = {"wte": "embed_tokens", "wpe": "position_embeddings", "ete": "extra_position_embeddings"}[
                    key_name[-3:]
                ]
                name = "model.%s.weight" % nlayer
                state = vnp.copy()  # same in embedded
                new_state[name] = torch.tensor(state)
                if key_name.startswith("model/wte"):
                    name = "lm_head.weight"
                    state = vnp.copy()  # same in embedded
                    new_state[name] = torch.tensor(state)
            elif key_name.startswith("model/wob"):
                name = "final_logits_bias"
                state = vnp.copy()  # same in embedded
                state = state.reshape((1, -1))
                new_state[name] = torch.tensor(state)
            elif key_name == "model/dense/kernel":
                name = "model.last_project.weight"
                state = vnp.transpose([1, 0]).copy()  # Mesh-Tensorflow is a diagonal matrix
                new_state[name] = torch.tensor(state)
            elif key_name == "model/dense_1/bias":
                name = "model.last_project.bias"
                state = vnp.copy()  # same because it is one dimensional
                new_state[name] = torch.tensor(state)
    torch.save(new_state, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="model converter.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tf_model_dir", metavar="PATH", type=str, required=True, help="import model")
    parser.add_argument("--output", metavar="PATH", type=str, required=True, help="output model")
    args = parser.parse_args()
    convert_tf_gptsan_to_pt(args)
