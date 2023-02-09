# coding=utf-8
# Copyright 2022 The Google AI team and The HuggingFace Inc. team. All rights reserved.
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

import argparse

import torch

from transformers import TrillssonEfficientNetConfig, TrillssonEfficientNetModel


_MODELS = {"trillsson3": "https://tfhub.dev/google/nonsemantic-speech-benchmark/trillsson3/1"}


def loaf_tf_weights_in_trillsson_efficientnet(tf_checkpoint, pytorch_dump_folder_path):
    """Load TensorFlow model in a pytorch model."""
    try:
        import numpy as np

        import tensorflow_hub as hub
    except ImportError:
        print("Tensorflow is not installed. Please install it to load the weights.")
        raise
    # load the weights from the tensorflow checkpoint
    init_vars = hub.load(_MODELS[tf_checkpoint])
    tf_weights = {}
    for variable in init_vars.variables:
        print(f"Loading TF weight {variable.name} with shape {variable.shape}")
        tf_weights[variable.name] = variable.numpy()

    # init model
    config = TrillssonEfficientNetConfig()
    model = TrillssonEfficientNetModel(config)

    current_block = -1
    current_block_index = ""
    current_block_type = ""
    name = ""
    for vars_name in list(tf_weights.keys()):
        m_name = vars_name.split("_", 1)  # max split = 1
        pointer = model
        array = tf_weights[vars_name]
        # check type of current block in tf_weights
        for block_type in ["stem", "top", "block", "dense"]:
            if block_type in m_name[0]:
                current_block_type = block_type
        if current_block_type != "dense":
            pointer = getattr(pointer, current_block_type)
            if current_block_type == "block":
                block_index = m_name[0][-2:]
                if block_index != current_block_index:
                    current_block_index = block_index
                    current_block += 1
                pointer = pointer[current_block]
                scope_names = m_name[1].split("/")
            else:
                scope_names = vars_name.split("/")
            if "se" in scope_names[0]:
                pointer = getattr(pointer, "squeeze_excitation")
            for name in scope_names:
                if name in ["kernel:0", "gamma:0", "depthwise_kernel:0"]:
                    pointer = getattr(pointer, "weight")
                elif name in ["beta:0", "bias:0"]:
                    pointer = getattr(pointer, "bias")
                elif name == "moving_mean:0":
                    pointer = getattr(pointer, "running_mean")
                elif name == "moving_variance:0":
                    pointer = getattr(pointer, "running_var")
                elif name == "bn":
                    pointer = getattr(pointer, "normalization")
                else:
                    pointer = getattr(pointer, name)
        else:
            pointer = getattr(pointer, "dense")
            name = m_name[0].split("/")[1]
            if name == "kernel:0":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, "bias")

        if name == "depthwise_kernel:0":
            array = np.transpose(array, (3, 2, 0, 1))
            array = np.transpose(array, (1, 0, 2, 3))
        elif "kernel:0" in name:
            if len(pointer.shape) == 2:  # copying into linear layer
                array = array.squeeze().transpose()
            else:
                array = np.transpose(array, (3, 2, 0, 1))

        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise

        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(vars_name, None)

    print(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    if pytorch_dump_folder_path:
        model.save_pretrained(pytorch_dump_folder_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument("--checkpoint_path", type=str, help="TF checkpoints")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    loaf_tf_weights_in_trillsson_efficientnet(args.checkpoint_path, args.pytorch_dump_folder_path)
