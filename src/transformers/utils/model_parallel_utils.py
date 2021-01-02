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

import inspect
from math import ceil

import torch


def make_default_sub_device_map(n_layers):
    """Returns a dictionary of layers distributed evenly across all devices."""

    # XXX: in the future we can implement a smarter allocation based on the device memory, since:
    # 1. cards can be of different memory-size (uncommon, but this developer has this setup)
    # 2. also the first device of encoder is used as the main device which will have all the non-layer specific params on it and thus will take more memory, so ideally the default should put less layers on that device
    # but, of course, users can customize it to their liking in their code.
    # Except it is not possible to customize the map in Trainer-based scripts, like `finetune_trainer.py`, where the user can only switch --model_parallel flag on and no way to set the map.
    n_gpus = torch.cuda.device_count()

    # XXX: this function splits the layers evenly across all devices, so that the end result is that each encoder and decoder share devices, as compared to creating maps where either of the two completely takes over one of the devices - need to measure which approach is more efficient - i.e. minimizes inter-device copying.
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / n_gpus))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(range(n_gpus), layers_list))


def make_default_device_map(encoder_n_layers, decoder_n_layers):
    return {
        "encoder": make_default_sub_device_map(encoder_n_layers),
        "decoder": make_default_sub_device_map(decoder_n_layers),
    }


def validate_sub_device_map(n_layers, device_map, name):
    where = f"device_map['{name}']"
    possible_sub_device_map = make_default_sub_device_map(n_layers)
    error_msg = f"here is a possible entry for {where}:\n{possible_sub_device_map}"

    # general format
    gpu_ids = device_map.keys()
    assert all(isinstance(x, int) for x in gpu_ids), (
        f"{where}: All keys much be integers, corresponding to available gpu IDS)\n" + error_msg
    )
    layer_ids = [i for v in device_map.values() for i in v]
    assert all(isinstance(x, int) for x in layer_ids), (
        f"{where}: Values must contain only integers, corresponding to layer numbers\n" + error_msg
    )

    # reality check
    valid_gpu_ids = list(range(torch.cuda.device_count()))
    wrong_gpu_ids = [x for x in gpu_ids if x not in valid_gpu_ids]
    assert not len(wrong_gpu_ids), (
        f"All keys must correspond to available gpus IDs, but got: {wrong_gpu_ids}\n" + error_msg
    )

    duplicate_layer_ids = [i for i in set(layer_ids) if layer_ids.count(i) > 1]
    assert not len(duplicate_layer_ids), (
        f"{where}: duplicate layer numbers detected: {duplicate_layer_ids}\n"
        "Each layer number must be specified only once, remove duplicates"
    )

    valid_layer_ids = list(range(0, n_layers))
    missing_layer_ids = [i for i in valid_layer_ids if i not in layer_ids]
    assert not len(missing_layer_ids), (
        f"{where}: missing layer numbers detected: {missing_layer_ids}\n" "Add missing layers to the device map."
    )
    extra_layer_ids = [i for i in layer_ids if i not in valid_layer_ids]
    assert not len(extra_layer_ids), (
        f"{where}: non-existing layer numbers detected: {extra_layer_ids}\n"
        f"This {name} has only {n_layers} layers.\n"
        "Remove extraneous layers from the device map.\n"
    )


def validate_device_map(encoder_n_layers, decoder_n_layers, device_map):
    possible_device_map = make_default_device_map(encoder_n_layers, decoder_n_layers)
    error_msg = f"invalid device_map format detected, here is a possible device map {possible_device_map}"

    assert "encoder" in device_map and "decoder" in device_map, error_msg
    encoder_device_map = device_map["encoder"]
    decoder_device_map = device_map["decoder"]

    assert isinstance(encoder_device_map, dict) and isinstance(decoder_device_map, dict), error_msg

    validate_sub_device_map(encoder_n_layers, encoder_device_map, "encoder")
    validate_sub_device_map(decoder_n_layers, decoder_device_map, "decoder")


def init_device_map(encoder_n_layers, decoder_n_layers, device_map=None):
    """
    - creates a device_map if none was passed
    - validates that map is correct

    Args:
      encoder_n_layers - number of encoder layers to remap
      decoder_n_layers - number of decoder layers to remap
      device_map - use this user-supplied map
    """
    if device_map is None:
        device_map = make_default_device_map(encoder_n_layers, decoder_n_layers)
    validate_device_map(encoder_n_layers, decoder_n_layers, device_map)
    return device_map


def log_name_device(var, fallbackname=None):  # search from the outmost frame inwards
    """
    This helper is useful for debug tracing of devices of variables, e.g.:
      logger.info(f"MP {self.__class__.__name__} {log_name_device(attention_mask)}")
    if it can't deduce the variable name (or finds wrong name), pass the name explicitly, e.g.:
      logger.info(f"MP {self.__class__.__name__} {log_name_device(self.lm_head, 'self.lm_head')}")
    """

    if fallbackname is not None:
        name = fallbackname
    else:
        for f in reversed(inspect.stack()):
            name = "unknown"
            names = [x for x, val in f.frame.f_locals.items() if val is var]
            if len(names) > 0:
                name = names[0]
                break
    if var is None:
        return f"{name} val=None"

    device = None
    try:
        device = var.device
    except AttributeError:
        if hasattr(var, "parameters"):
            device = get_layer_device(var)
    return f"{name} {device}"


def get_layer_device(self):
    try:
        device = next(self.parameters(recurse=True)).device
    except StopIteration:
        device = None
    return device


# def to_dev(self, input):
#         try:
#             device = next(self.parameters(recurse=True)).device
#         except StopIteration:
#             device = None

#         if device is None:
#             raise ValueError(f"Can't find any params for {self.__class__}")
#         print(f"manual switch to {device}")
#         return input.to(device)


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


# def model_parallel_call(self, *input, **kwargs):

#     # get device of any of the param of this layer
#     try:
#         device = next(self.parameters(recurse=True)).device
#     except StopIteration:
#         device = None

#     # print(f"layer device: {device}")
#     if device is not None:
#         # torch.cuda.set_device(device)
#         input = recursive_to(device, input)
#         kwargs = recursive_to(device, kwargs)

#     return nn.Module.__call__(self, *input, **kwargs)


def print_layer_devices(self):
    try:
        device = next(self.parameters(recurse=True)).device
    except StopIteration:
        device = None
    print(f"device dump - looked up device {device}")
    for n, p in self.named_parameters():
        print(f"{n}: {p.device}")


# XXX: still used by t5 + gpt2 so leave here for now
# will be removed once the other functions above have been integrated
def assert_device_map(device_map, num_blocks):
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


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))
