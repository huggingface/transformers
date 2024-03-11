import torch.nn as nn

from ....enums import NormalizationImplementation
from .apex import ApexLayerNorm
from .apex_persistent import ApexPersistentLayerNorm


_LAYERNORM_MODULES = {
    NormalizationImplementation.torch.value: nn.LayerNorm,
    NormalizationImplementation.apex.value: ApexLayerNorm,
    NormalizationImplementation.apex_persistent.value: ApexPersistentLayerNorm,
}


def get_layernorm(
    normalized_shape: int,
    eps: float,
    normalization_implementation: NormalizationImplementation = NormalizationImplementation.torch,
) -> nn.LayerNorm:
    normalization_implementation = NormalizationImplementation(normalization_implementation)

    if normalization_implementation.value in _LAYERNORM_MODULES:
        return _LAYERNORM_MODULES[normalization_implementation.value](normalized_shape=normalized_shape, eps=eps)

    raise ValueError(f"unexpected `normalization_implementation` {normalization_implementation}")
