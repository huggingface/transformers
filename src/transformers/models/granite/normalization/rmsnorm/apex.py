import torch

from .base import RMSNorm


try:
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction
except:
    FusedRMSNormAffineMixedDtypesFunction = None


class ApexRMSNorm(RMSNorm):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        if FusedRMSNormAffineMixedDtypesFunction is None:
            raise ImportError("build apex from source")

        super().__init__(normalized_shape, eps=eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = FusedRMSNormAffineMixedDtypesFunction.apply(input, self.weight, self.normalized_shape, self.eps, True)
        return input
