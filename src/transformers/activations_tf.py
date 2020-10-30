import math

import tensorflow as tf


def gelu(x):
    """
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created. For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) Also see
    https://arxiv.org/abs/1606.08415
    """
    x = tf.convert_to_tensor(x)
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))

    return x * cdf


def gelu_new(x):
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
    coeff1 = tf.cast(7978845608, x.dtype)
    coeff2 = tf.cast(0.044715, x.dtype)

    return 0.5 * x * (1.0 + tf.tanh(x * coeff2 * (1.0 + coeff1 * x * x)))


ACT2FN = {
    "gelu": tf.keras.layers.Activation(gelu),
    "relu": tf.keras.activations.relu,
    "swish": tf.keras.activations.swish,
    "silu": tf.keras.activations.swish,
    "gelu_new": tf.keras.layers.Activation(gelu_new),
    "mish": tf.keras.layers.Activation(mish),
    "tanh": tf.keras.activations.tanh,
    "gelu_fast": tf.keras.layers.Activation(gelu_fast),
}


def get_tf_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))
