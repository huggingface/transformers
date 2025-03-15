import unittest

import pytest
import torch
from torch import nn

from transformers.testing_utils import require_torch_gpu
from transformers.utils.kernel_hub import (
    LayerRepository,
    register_layer_mapping,
    replace_hub_layer_forward,
    use_hub_layer_forward,
)


register_layer_mapping(
    "LlamaRMSNormForTesting",
    device_type="cuda",
    layer_repository=LayerRepository(
        layer_name="RMSNorm",
        repo_id="kernels-community/triton-layer-norm",
        revision="9fc83e639335d0c9a8ac2ecd7f64f7ebebc727f5",
    ),
)

register_layer_mapping(
    "LlamaRMSNormNonExistingDevice",
    device_type="bogus",
    layer_repository=LayerRepository(
        layer_name="RMSNorm",
        repo_id="kernels-community/triton-layer-norm",
        revision="9fc83e639335d0c9a8ac2ecd7f64f7ebebc727f5",
    ),
)

register_layer_mapping(
    "LlamaRMSNormNonExistingRepo",
    device_type="cuda",
    layer_repository=LayerRepository(
        layer_name="RMSNorm",
        repo_id="kernels-community/does-not-exist",
        revision="9fc83e639335d0c9a8ac2ecd7f64f7ebebc727f5",
    ),
)


class LlamaRMSNormReference(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        # Align parameters with hub kernel.
        self.bias = None
        self.drop = None
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@use_hub_layer_forward("LlamaRMSNormForTesting", use_fallback=False)
class LlamaRMSNormHub(LlamaRMSNormReference):
    pass


@require_torch_gpu
class KernelHubTest(unittest.TestCase):
    def test_use_hub_layer_forward(self):
        torch.manual_seed(0)
        layer_ref = LlamaRMSNormReference(64).cuda()
        layer_hub = LlamaRMSNormHub(64).cuda()
        X = torch.randn((8, 64), device="cuda", dtype=torch.float32)
        self.assertTrue(torch.allclose(layer_ref(X), layer_hub(X)))

    def test_replace_hub_layer_forward(self):
        class Layer(LlamaRMSNormReference):
            pass

        replace_hub_layer_forward(Layer, "LlamaRMSNormForTesting", use_fallback=False)

        torch.manual_seed(0)
        layer_ref = LlamaRMSNormReference(64).cuda()
        layer_hub = Layer(64).cuda()
        X = torch.randn((8, 64), device="cuda", dtype=torch.float32)
        self.assertTrue(torch.allclose(layer_ref(X), layer_hub(X)))


@use_hub_layer_forward("LlamaRMSNormNonExistingName", use_fallback=False)
class LlamaRMSNormHubNonExistingName(LlamaRMSNormReference):
    pass


@use_hub_layer_forward("LlamaRMSNormNonExistingDevice", use_fallback=False)
class LlamaRMSNormHubNonExistingDevice(LlamaRMSNormReference):
    pass


@use_hub_layer_forward("LlamaRMSNormNonExistingRepo", use_fallback=False)
class LlamaRMSNormHubNonExistingRepo(LlamaRMSNormReference):
    pass


@pytest.mark.parametrize(
    "layer_cls",
    [
        LlamaRMSNormHubNonExistingName,
        LlamaRMSNormHubNonExistingDevice,
        LlamaRMSNormHubNonExistingRepo,
    ],
)
@require_torch_gpu
def test_nonexisting_hub_layers_fails_without_fallback(layer_cls):
    layer = layer_cls(64).cuda()
    X = torch.randn((8, 64), device="cuda", dtype=torch.float32)
    with pytest.raises(Exception):
        layer(X)


@use_hub_layer_forward("LlamaRMSNormNonExistingName")
class LlamaRMSNormHubNonExistingNameFallback(LlamaRMSNormReference):
    pass


@use_hub_layer_forward("LlamaRMSNormNonExistingDevice")
class LlamaRMSNormHubNonExistingDeviceFallback(LlamaRMSNormReference):
    pass


@use_hub_layer_forward("LlamaRMSNormNonExistingRepo")
class LlamaRMSNormHubNonExistingRepoFallback(LlamaRMSNormReference):
    pass


@pytest.mark.parametrize(
    "layer_cls",
    [
        LlamaRMSNormHubNonExistingNameFallback,
        LlamaRMSNormHubNonExistingDeviceFallback,
        LlamaRMSNormHubNonExistingRepoFallback,
    ],
)
@require_torch_gpu
def test_nonexisting_hub_layers_succeeds_with_fallback(layer_cls):
    layer = layer_cls(64).cuda()
    X = torch.randn((8, 64), device="cuda", dtype=torch.float32)
    # Must not raise.
    layer(X)
