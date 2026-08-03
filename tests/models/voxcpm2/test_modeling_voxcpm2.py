# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import VoxCPM2Config, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.models.voxcpm2.modeling_voxcpm2 import (
        VoxCPM2CausalConv1d,
        VoxCPM2ScalarQuantizationLayer,
        VoxCPM2Snake1d,
    )


@require_torch
def test_scalar_quantization_matches_reference():
    config = VoxCPM2Config()
    config.lm_config.hidden_size = 2
    config.scalar_quantization_latent_dim = 3
    config.scalar_quantization_scale = 4
    layer = VoxCPM2ScalarQuantizationLayer(config)

    with torch.no_grad():
        layer.in_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]]))
        layer.in_proj.bias.zero_()
        layer.out_proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
        layer.out_proj.bias.zero_()

    assert set(layer.state_dict()) == {"in_proj.weight", "in_proj.bias", "out_proj.weight", "out_proj.bias"}
    assert layer.in_proj.weight.shape == (3, 2)
    assert layer.out_proj.weight.shape == (2, 3)

    hidden_states = torch.tensor([[[-1.0, 0.25], [0.5, 1.0]]])
    projected_states = torch.tanh(layer.in_proj(hidden_states))
    quantized_states = torch.round(projected_states * layer.scale) / layer.scale
    expected_output = layer.out_proj(quantized_states)

    layer.eval()
    torch.testing.assert_close(layer(hidden_states), expected_output, rtol=0, atol=0)

    layer.train()
    training_input = hidden_states.clone().requires_grad_()
    training_output = layer(training_input)
    torch.testing.assert_close(training_output, expected_output, rtol=0, atol=0)
    training_output.sum().backward()

    reference_input = hidden_states.clone().requires_grad_()
    layer.out_proj(torch.tanh(layer.in_proj(reference_input))).sum().backward()
    torch.testing.assert_close(training_input.grad, reference_input.grad, rtol=0, atol=0)


@require_torch
def test_snake_activation_matches_reference():
    layer = VoxCPM2Snake1d(3)
    with torch.no_grad():
        layer.alpha.copy_(torch.tensor([[[0.5], [1.0], [2.0]]]))

    assert list(layer.state_dict()) == ["alpha"]
    assert layer.alpha.shape == (1, 3, 1)

    hidden_states = torch.randn(2, 3, 4, 5, requires_grad=True)
    output = layer(hidden_states)
    reshaped_states = hidden_states.reshape(2, 3, -1)
    expected_output = reshaped_states + (layer.alpha + 1e-9).reciprocal() * torch.sin(
        layer.alpha * reshaped_states
    ).pow(2)
    expected_output = expected_output.reshape_as(hidden_states)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    output_gradient = hidden_states.grad.clone()
    hidden_states.grad = None
    expected_output.sum().backward()
    torch.testing.assert_close(output_gradient, hidden_states.grad, rtol=0, atol=0)


@require_torch
def test_causal_convolution_matches_reference():
    layer = VoxCPM2CausalConv1d(2, 4, kernel_size=6, stride=3, padding=2, output_padding=1)
    assert list(layer.state_dict()) == ["weight", "bias"]
    assert layer.padding == (0,)
    assert layer.causal_padding == 3

    hidden_states = torch.randn(2, 2, 19, requires_grad=True)
    output = layer(hidden_states)

    reference_input = hidden_states.detach().clone().requires_grad_()
    padded_states = torch.nn.functional.pad(reference_input, (layer.causal_padding, 0))
    expected_output = torch.nn.functional.conv1d(
        padded_states,
        layer.weight,
        layer.bias,
        stride=layer.stride,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    expected_output.sum().backward()
    torch.testing.assert_close(hidden_states.grad, reference_input.grad, rtol=0, atol=0)
