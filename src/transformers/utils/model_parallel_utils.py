# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from math import ceil

import torch


def validate_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))

    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # Duplicate check
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    assert len(duplicate_blocks) == 0, (
        "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These "
        "attention blocks were specified more than once: " + str(duplicate_blocks)
    )
    assert len(missing_blocks) == 0, (
        "There are attention blocks for this model that are not specified in the device_map. Add these attention "
        "blocks to a device on the device_map: " + str(missing_blocks)
    )
    assert (
        len(extra_blocks) == 0
    ), "The device_map contains more attention blocks than this model has. Remove these from the device_map:" + str(
        extra_blocks
    )


def make_default_device_map(n_layers):
    """Returns a dictionary of layers distributed evenly across all devices."""
    n_gpus = torch.cuda.device_count()
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / n_gpus))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(range(n_gpus), layers_list))


def init_device_map(n_layers, device_map=None):
    """
    - creates a device_map if none was passed
    - validates that map is correct

    Args:
      n_layers - how many total layers to remap
    """
    if device_map is None:
        device_map = make_default_device_map(n_layers)
    validate_device_map(device_map, n_layers)
    return device_map


def get_layer_device(self):
    try:
        device = next(self.parameters(recurse=True)).device
    except StopIteration:
        device = None
    return device


def recursive_to(device, item):
    """
    Switch any tensors found in `item` to `device`. Currently can handle a single tensor, or any of the nested list,
    tuple and dict structures.
    """

    if torch.is_tensor(item):
        return item.to(device)

    elif isinstance(item, list):
        for i, x in enumerate(item):
            item[i] = recursive_to(device, x)
        return item

    elif isinstance(item, tuple):
        return tuple(recursive_to(device, list(item)))

    elif isinstance(item, dict):
        for k, v in item.items():
            item[k] = recursive_to(device, v)
        return item

    else:
        return item


def model_parallel_inputs_to_device(func):
    """
    This decorator is a noop unless self.model_parallel == True.

    It will try to find at least one parameter to read layer's .device from and then will automatically copy any inputs
    to that device before `forward` is called. Use it as:

    @model_parallel_inputs_to_device def forward(self, input1, input2, ...)

    It will do its magical thing only if all params of this layer are on the same device. If it is not the case use
    `model_parallel_inputs_to_specific_device` at the top of `forward`
    """

    def _call__mp(self, *input, **kwargs):

        if not hasattr(self, "model_parallel") or not self.model_parallel:
            return func(self, *input, **kwargs)

        # get device of any of the param of this layer
        try:
            device = next(self.parameters(recurse=True)).device
        except StopIteration:
            device = None

        # print(f"layer device: {device}")
        if device is not None:
            # torch.cuda.set_device(device)
            # print(f"auto-move inputs to {device}")

            input = recursive_to(device, input)
            kwargs = recursive_to(device, kwargs)

            return func(self, *input, **kwargs)

    return _call__mp


def model_parallel_inputs_to_specific_device(device, *input):
    """
    Similar to the model_parallel_inputs_to_device decorator, but this one is used for situations either when: 1. an
    explicit call is desired (similar to `model.to()`) 2. the layer has params on mixed devices and therefore a wrong
    device might get picked

    To use:

    @model_parallel_inputs_to_device def forward(self, input1, input2, ...): # get the desired device somewhere, e.g. a
    specific param or a module attribute device = self.fc1.device input1, input2 =
    model_parallel_inputs_to_specific_device(device, input1, input2) # this is the same as: input1 = input1.to(device)
    input2 = input2.to(device) # but it works on variables that contain tensors but don't have `.to()` otherwise
    """
    if device is None:
        raise ValueError("device cannot be None")
    # print(f"move specific inputs to {device}")
    input = recursive_to(device, input)
    # remove the need for the caller to perform "a, = foo(a)",
    # which otherwise will make `a` a tuple when it might not be one
    return input[0] if len(input) == 1 else input


# XXX: still used by gpt2 so leave here for now
assert_device_map = validate_device_map


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))
