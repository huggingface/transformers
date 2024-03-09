import torch.nn as nn
from copy import deepcopy

from transformers.activations import ACT2CLS, ClassInstantier

from .math_gelu import MathGELU


_BASE_ACTIVATIONS = deepcopy(ACT2CLS)
_BASE_ACTIVATIONS.update({
    "celu": nn.modules.CELU,
    "elu": nn.modules.ELU,
    "gelu_math_tanh": MathGELU,
    "selu": nn.modules.SELU,
    "hard_shrink": nn.modules.Hardshrink,
    "hard_sigmoid": nn.modules.Hardsigmoid,
    "hard_swish": nn.modules.Hardswish,
    "hard_tanh": nn.modules.Hardtanh,
    "log_sigmoid": nn.modules.LogSigmoid,
    "prelu": nn.modules.PReLU,
    "rrelu": nn.modules.RReLU,
    "softplus": nn.modules.Softplus,
    "soft_plus": nn.modules.Softplus,
    "soft_shrink": nn.modules.Softshrink,
    "soft_sign": nn.modules.Softsign,
    "tanh_shrink": nn.modules.Tanhshrink,
})

# instantiates the module when __getitem__ is called
_BASE_ACTIVATIONS = ClassInstantier(_BASE_ACTIVATIONS)


def get_base_activation(name: str) -> nn.Module:
    if name in _BASE_ACTIVATIONS:
        return _BASE_ACTIVATIONS[name]
    raise ValueError("invalid activation function")
