import torch.nn as nn

from ....enums import NormalizationImplementation
from .apex import ApexRMSNorm
from .base import RMSNorm


_RMSNORM_MODULES = {
    NormalizationImplementation.torch.value: RMSNorm,
    NormalizationImplementation.apex.value: ApexRMSNorm,
}


def get_rmsnorm(
    normalized_shape: int,
    eps: float,
    normalization_implementation: NormalizationImplementation = NormalizationImplementation.torch,
) -> nn.LayerNorm:
    normalization_implementation = NormalizationImplementation(normalization_implementation)

    if normalization_implementation.value in _RMSNORM_MODULES:
        return _RMSNORM_MODULES[normalization_implementation.value](normalized_shape=normalized_shape, eps=eps)

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")
