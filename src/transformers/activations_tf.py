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

import math

import tensorflow as tf
from packaging import version


def _gelu(x):
    """
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created. For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) Also see
    https://arxiv.org/abs/1606.08415
    """
    x = tf.convert_to_tensor(x)
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))

    return x * cdf


def _gelu_new(x):
    """
    Gaussian Error Linear Unit. This is a smoother version of the GELU. Original paper: https://arxiv.org/abs/1606.0841

    Args:
        x: float Tensor to perform activation

    Returns:
        `x` with the GELU activation applied.
    """
    x = tf.convert_to_tensor(x)
    pi = tf.cast(math.pi, x.dtype)
    coeff = tf.cast(0.044715, x.dtype)
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))

    return x * cdf


def mish(x):
    x = tf.convert_to_tensor(x)

    return x * tf.tanh(tf.math.softplus(x))


def gelu_fast(x):
    x = tf.convert_to_tensor(x)
    coeff1 = tf.cast(0.044715, x.dtype)
    coeff2 = tf.cast(0.7978845608, x.dtype)

    return 0.5 * x * (1.0 + tf.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))


if version.parse(tf.version.VERSION) >= version.parse("2.4"):

    def approximate_gelu_wrap(x):
        return tf.keras.activations.gelu(x, approximate=True)

    gelu = tf.keras.activations.gelu
    gelu_new = approximate_gelu_wrap
else:
    gelu = _gelu
    gelu_new = _gelu_new


ACT2FN = {
    "gelu": gelu,
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.activations.swish,
    "silu": tf.keras.activations.swish,
    "gelu_new": gelu_new,
    "mish": mish,
    "tanh": tf.keras.activations.tanh,
    "gelu_fast": gelu_fast,
}


def get_tf_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
