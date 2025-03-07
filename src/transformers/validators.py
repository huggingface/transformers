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

from typing import Iterable

from .activations import ACT2CLS


# Integer validators


def positive_int(value: int):
    """Ensures that `value` is a positive integer (including 0)."""
    if not value >= 0:
        raise ValueError(f"Value must be a positive integer, got {value}.")


def strictly_positive_int(value: int):
    """Ensures that `value` is a positive integer (excluding 0)."""
    if not value > 0:
        raise ValueError(f"Value must be a strictly positive integer, got {value}.")


def vocabulary_token(value: int, vocab_size: int):
    """Ensures that `value` is a valid vocabulary token index."""
    if not 0 <= value < vocab_size:
        raise ValueError(f"Value must be a token in the vocabulary, got {value}. (vocabulary size = {vocab_size})")


# Float validators


def positive_float(value: float):
    """Ensures that `value` is a positive float (including 0.0)."""
    if not value >= 0:
        raise ValueError(f"Value must be a positive float, got {value}.")


def probability(value: float):
    """Ensures that `value` is a valid probability number, i.e. [0,1]."""
    if not 0 <= value <= 1:
        raise ValueError(f"Value must be a probability between 0.0 and 1.0, got {value}.")


# String validators


def activation_function_key(value: str):
    """Ensures that `value` is a string corresponding to an activation function."""
    if value not in ACT2CLS:
        raise ValueError(
            f"Value must be one of {list(ACT2CLS.keys())}, got {value}. "
            "Make sure to use a string that corresponds to an activation function."
        )


def choice_str(value: str, choices: Iterable[str]):
    """Ensures that `value` is one of the choices in `choices`."""
    if value not in choices:
        raise ValueError(f"Value must be one of {choices}, got {value}")
