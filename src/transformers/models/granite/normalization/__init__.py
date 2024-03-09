import torch.nn as nn

from ..enums import NormalizationImplementation
from .layernorm import get_layernorm
from .rmsnorm import RMSNorm, get_rmsnorm


_NORMALIZATION_FUNCTIONS = {
    "layernorm": get_layernorm,
    "rmsnorm": get_rmsnorm,
}


def get_normalization_function(
    name: str,
    normalized_shape: int,
    eps: float = 1e-5,
    normalization_implementation: NormalizationImplementation = NormalizationImplementation.torch,
) -> nn.LayerNorm:
    if name in _NORMALIZATION_FUNCTIONS:
        return _NORMALIZATION_FUNCTIONS[name](
            normalized_shape, eps=eps, normalization_implementation=normalization_implementation
        )

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")
