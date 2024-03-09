import torch
import torch.nn as nn


try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
except:
    FastLayerNormFN = None


_PERSISTENT_LAYERNORM_ALLOWED_HIDDEN_STATES = [
    1024,
    1536,
    2048,
    2304,
    3072,
    3840,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    12800,
    15360,
    16384,
    18432,
    20480,
    24576,
    25600,
    30720,
    32768,
    40960,
    49152,
    65536,
]


class ApexPersistentLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape: int, eps: float = 0.00001) -> None:
        if FastLayerNormFN is None:
            raise ImportError("build apex from source with --fast_layer_norm")

        super().__init__(normalized_shape, eps=eps)

        assert (
            self.normalized_shape[0] in _PERSISTENT_LAYERNORM_ALLOWED_HIDDEN_STATES
        ), "persistent layernorm kernel is not avilable for the specified hidden dimension"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = FastLayerNormFN.apply(input, self.weight, self.bias, self.eps, True)
        return input
