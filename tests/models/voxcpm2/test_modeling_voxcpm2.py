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

    from transformers.models.voxcpm2.modeling_voxcpm2 import VoxCPM2ScalarQuantizationLayer


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
