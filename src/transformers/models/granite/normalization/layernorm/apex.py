import torch
import torch.nn as nn


try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
except:
    FusedLayerNormAffineFunction = None


class ApexLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: int, eps: float = 0.00001) -> None:
        if FusedLayerNormAffineFunction is None:
            raise ImportError("build apex from source")

        super().__init__(normalized_shape, eps=eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = FusedLayerNormAffineFunction.apply(
            input, self.weight, self.bias, self.normalized_shape, self.eps, True
        )
        return input
