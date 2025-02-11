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
from packaging.version import parse


try:
    import tf_keras as keras
except (ModuleNotFoundError, ImportError):
    import keras

    if parse(keras.__version__).major > 2:
        raise ValueError(
            "Your currently installed version of Keras is Keras 3, but this is not yet supported in "
            "Transformers. Please install the backwards-compatible tf-keras package with "
            "`pip install tf-keras`."
        )


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


def quick_gelu(x):
    x = tf.convert_to_tensor(x)
    coeff = tf.cast(1.702, x.dtype)
    return x * tf.math.sigmoid(coeff * x)


def gelu_10(x):
    """
    Clip the range of possible GeLU outputs between [-10, 10]. This is especially useful for quantization purpose, as
    it allows mapping 2 negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created. For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) Also see
    https://arxiv.org/abs/1606.08415 :param x: :return:
    """
    return tf.clip_by_value(_gelu(x), -10, 10)


def glu(x, axis=-1):
    """
    Gated Linear Unit. Implementation as defined in the original paper (see https://arxiv.org/abs/1612.08083), where
    the input `x` is split in two halves across a dimension (`axis`), A and B, returning A * sigmoid(B).

    Args:
        `x`: float Tensor to perform activation
        `axis`: dimension across which `x` be split in half

    Returns:
        `x` with the GLU activation applied (with its size halved across the dimension `axis`).
    """
    a, b = tf.split(x, 2, axis=axis)
    return a * tf.math.sigmoid(b)


if parse(tf.version.VERSION) >= parse("2.4"):

    def approximate_gelu_wrap(x):
        return keras.activations.gelu(x, approximate=True)

    gelu = keras.activations.gelu
    gelu_new = approximate_gelu_wrap
else:
    gelu = _gelu
    gelu_new = _gelu_new


ACT2FN = {
    "gelu": gelu,
    "gelu_10": gelu_10,
    "gelu_fast": gelu_fast,
    "gelu_new": gelu_new,
    "glu": glu,
    "mish": mish,
    "quick_gelu": quick_gelu,
    "relu": keras.activations.relu,
    "sigmoid": keras.activations.sigmoid,
    "silu": keras.activations.swish,
    "swish": keras.activations.swish,
    "tanh": keras.activations.tanh,
}


def get_tf_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
