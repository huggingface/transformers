import torch.nn as nn

from .base import get_base_activation
from .glu import get_glu_activation, is_glu


def get_activation_function(name: str) -> nn.Module:
    return get_glu_activation(name) if is_glu(name) else get_base_activation(name)
