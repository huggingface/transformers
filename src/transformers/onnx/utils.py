# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from ctypes import c_float, sizeof
from enum import Enum
from typing import Iterable

from transformers import is_torch_available


class ParameterFormat(Enum):
    Float = c_float

    @property
    def size(self) -> int:
        """
        Number of byte required for this data type

        Returns:
            Integer > 0
        """
        return sizeof(self.value)


def compute_effective_axis_dimension(dimension: int, fixed_dimension: int, num_token_to_add: int = 0) -> int:
    """

    Args:
        dimension:
        fixed_dimension:
        num_token_to_add:

    Returns:

    """
    # < 0 is possible if using a dynamic axis
    if dimension <= 0:
        dimension = fixed_dimension

    dimension -= num_token_to_add
    return dimension


def compute_serialized_parameters_size(num_parameters: int, dtype: ParameterFormat) -> int:
    """
    Compute the size taken by all the parameters in the given the storage format when serializing the model

    Args:
        num_parameters: Number of parameters to be saved
        dtype: The data format each parameter will be saved

    Returns:
        Size (in byte) taken to save all the parameters
    """
    return num_parameters * dtype.size


if is_torch_available():
    import torch

    def ort_compatible_forward_with_past_key_values_output(forward, num_layers):
        import functools

        if isinstance(num_layers, Iterable):
            num_layers = sum(num_layers)

        @functools.wraps(forward)
        def compatible_forward(*args, **kwargs):
            result = forward(*args, **kwargs)

            if "past_key_values" in result:
                if isinstance(result["past_key_values"][0], tuple) or isinstance(result["past_key_values"][0], list):
                    assert len(result["past_key_values"]) == num_layers and len(result["past_key_values"][0]) == 2
                present = []
                for i in range(num_layers):
                    # Since transformers v4.*, past key and values are separated outputs.
                    # Here we concatenate them into one tensor to be compatible with Attention operator.
                    present.append(
                        torch.cat((
                            result["past_key_values"][i][0].unsqueeze(0),
                            result["past_key_values"][i][1].unsqueeze(0)
                        ), dim=0)
                    )
                return {"logits": result["logits"], "past_key_values": tuple(present)}
            else:
                return result

        return compatible_forward

