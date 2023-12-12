import torch
from ...configuration_utils import PretrainedConfig


class SigmaMoEConfiguration(PretrainedConfig):
    def __init__(self):
        pass


class SigmaMoEFeedForwardLayer(torch.nn.Module):

    def __init__(self, config: SigmaMoEConfiguration, is_sparse: bool):
        super().__init__()


class SigmaMoETransformerLayer(torch.nn.Module):
    """
    This layer can be a decoder or an encoder layer.
    
    If we get hidden encoder states in the forward,
    and this is a decoder layer, we use cross attention.
    
    If we don't get hidden encoder states in the forward
    and this is a decoder layer, we don't do cross attention.

    This module is a single layer, meaning that it has one
    self attention (always) and one cross attention (if decoder
    and enc-dec architecture) and one SigmaMoEFeedForwardLayer.
    The module also has layer norms, dropout and the standard
    residual connections.
    """

    def __init__(self, config: SigmaMoEConfiguration, is_sparse: bool):
        super().__init__()


if __name__ == "__main__":    
    layer = SigmaMoETransformerLayer(SigmaMoEConfiguration(), is_sparse=True)