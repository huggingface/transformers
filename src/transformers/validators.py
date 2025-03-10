# coding=utf-8
# Copyright 2025-present the HuggingFace Inc. team.
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
"""
Validators to be used with `huggingface_hub.utils.strict_dataclass`. We recommend using the validator(s) that best
describe the constraints of your dataclass fields, for a best user experience (e.g. better error messages).
"""

from typing import Callable, Optional

from .activations import ACT2CLS


# Numerical validators


def interval(min: Optional[int | float] = None, max: Optional[int | float] = None) -> Callable:
    """
    Parameterized validator that ensures that `value` is within the defined interval.
    Expected usage: `validated_field(interval(min=0), default=8)`
    """
    error_message = "Value must be"
    if min is not None:
        error_message += f" at least {min}"
    if min is not None and max is not None:
        error_message += " and"
    if max is not None:
        error_message += f" at most {max}"
    error_message += ", got {value}."

    min = min or float("-inf")
    max = max or float("inf")

    def _inner(value: int | float):
        if not min <= value <= max:
            raise ValueError(error_message.format(value=value))

    return _inner


def probability(value: float):
    """Ensures that `value` is a valid probability number, i.e. [0,1]."""
    if not 0 <= value <= 1:
        raise ValueError(f"Value must be a probability between 0.0 and 1.0, got {value}.")


# String validators


def activation_fn_key(value: str):
    """Ensures that `value` is a string corresponding to an activation function."""
    # TODO (joao): in python 3.11+, we can build a Literal type from the keys of ACT2CLS
    if value not in ACT2CLS:
        raise ValueError(
            f"Value must be one of {list(ACT2CLS.keys())}, got {value}. "
            "Make sure to use a string that corresponds to an activation function."
        )
