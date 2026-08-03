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

import math

from transformers import VoxCPM2Config, VoxCPM2TextConfig, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers.models.voxcpm2.modeling_voxcpm2 import (
        VoxCPM2Attention,
        VoxCPM2CausalConv1d,
        VoxCPM2CausalConvTranspose1d,
        VoxCPM2DecoderLayer,
        VoxCPM2RMSNorm,
        VoxCPM2ScalarQuantizationLayer,
        VoxCPM2SinusoidalPositionEmbedding,
        VoxCPM2Snake1d,
        VoxCPM2TimestepEmbedding,
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


@require_torch
def test_causal_transposed_convolution_matches_reference():
    layer = VoxCPM2CausalConvTranspose1d(4, 2, kernel_size=6, stride=3, padding=2, output_padding=1)
    assert list(layer.state_dict()) == ["weight", "bias"]
    assert layer.padding == (0,)
    assert layer.output_padding == (0,)
    assert layer.causal_trim == 3

    hidden_states = torch.randn(2, 4, 7, requires_grad=True)
    output = layer(hidden_states)

    reference_input = hidden_states.detach().clone().requires_grad_()
    expected_output = torch.nn.functional.conv_transpose1d(
        reference_input,
        layer.weight,
        layer.bias,
        stride=layer.stride,
        groups=layer.groups,
        dilation=layer.dilation,
    )
    expected_output = expected_output[..., : -layer.causal_trim]
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    expected_output.sum().backward()
    torch.testing.assert_close(hidden_states.grad, reference_input.grad, rtol=0, atol=0)

    zero_trim_layer = VoxCPM2CausalConvTranspose1d(2, 3, kernel_size=3)
    zero_trim_input = torch.randn(1, 2, 5)
    zero_trim_output = zero_trim_layer(zero_trim_input)
    expected_zero_trim_output = torch.nn.functional.conv_transpose1d(
        zero_trim_input, zero_trim_layer.weight, zero_trim_layer.bias
    )
    torch.testing.assert_close(zero_trim_output, expected_zero_trim_output, rtol=0, atol=0)


@require_torch
def test_sinusoidal_timestep_embedding_matches_reference():
    layer = VoxCPM2SinusoidalPositionEmbedding(8)
    assert not layer.state_dict()

    for timesteps in (torch.tensor(0.25), torch.tensor([0.0, 0.25, 1.0])):
        output = layer(timesteps)
        normalized_timesteps = timesteps.reshape(-1)
        exponent = math.log(10000) / 3
        frequencies = torch.exp(torch.arange(4, dtype=timesteps.dtype) * -exponent)
        embeddings = 1000.0 * normalized_timesteps.unsqueeze(1) * frequencies.unsqueeze(0)
        expected_output = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    assert layer(torch.tensor(0.5)).shape == (1, 8)
    assert layer(torch.tensor([0.5, 1.0])).shape == (2, 8)


@require_torch
def test_timestep_projection_matches_reference():
    layer = VoxCPM2TimestepEmbedding(4, 6, output_dim=5)
    assert list(layer.state_dict()) == [
        "linear_1.weight",
        "linear_1.bias",
        "linear_2.weight",
        "linear_2.bias",
    ]
    assert layer.linear_1.weight.shape == (6, 4)
    assert layer.linear_2.weight.shape == (5, 6)

    hidden_states = torch.randn(3, 4, requires_grad=True)
    output = layer(hidden_states)

    reference_input = hidden_states.detach().clone().requires_grad_()
    expected_output = torch.nn.functional.linear(reference_input, layer.linear_1.weight, layer.linear_1.bias)
    expected_output = torch.nn.functional.silu(expected_output)
    expected_output = torch.nn.functional.linear(expected_output, layer.linear_2.weight, layer.linear_2.bias)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    output.sum().backward()
    expected_output.sum().backward()
    torch.testing.assert_close(hidden_states.grad, reference_input.grad, rtol=0, atol=0)

    default_output_layer = VoxCPM2TimestepEmbedding(4, 6)
    assert default_output_layer(torch.randn(2, 4)).shape == (2, 6)


@require_torch
def test_attention_matches_reference():
    config = VoxCPM2TextConfig(
        vocab_size=32,
        hidden_size=6,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        kv_channels=4,
        use_mup=False,
        rope_parameters=None,
    )
    config._attn_implementation = "sdpa"
    layer = VoxCPM2Attention(config, layer_idx=0).eval()

    assert list(layer.state_dict()) == ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]
    assert layer.q_proj.weight.shape == (8, 6)
    assert layer.k_proj.weight.shape == (4, 6)
    assert layer.o_proj.weight.shape == (6, 8)

    hidden_states = torch.randn(1, 4, 6)
    angles = torch.randn(1, 4, 4)
    position_embeddings = (angles.cos(), angles.sin())
    for is_causal, rotary_embeddings in ((True, None), (False, None), (True, position_embeddings)):
        output, _ = layer(hidden_states, position_embeddings=rotary_embeddings, is_causal=is_causal)

        query_states = layer.q_proj(hidden_states).view(1, 4, 2, 4).transpose(1, 2)
        key_states = layer.k_proj(hidden_states).view(1, 4, 1, 4).transpose(1, 2)
        value_states = layer.v_proj(hidden_states).view(1, 4, 1, 4).transpose(1, 2)
        if rotary_embeddings is not None:
            cos, sin = (embedding.unsqueeze(1) for embedding in rotary_embeddings)
            rotated_query = torch.cat((-query_states[..., 2:], query_states[..., :2]), dim=-1)
            rotated_key = torch.cat((-key_states[..., 2:], key_states[..., :2]), dim=-1)
            query_states = query_states * cos + rotated_query * sin
            key_states = key_states * cos + rotated_key * sin

        expected_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=is_causal,
            enable_gqa=True,
            scale=layer.scaling,
        )
        expected_output = expected_output.transpose(1, 2).reshape(1, 4, 8)
        expected_output = layer.o_proj(expected_output)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)


@require_torch
def test_decoder_layer_residual_scaling():
    for use_mup in (False, True):
        config = VoxCPM2TextConfig(
            vocab_size=32,
            hidden_size=6,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            kv_channels=4,
            scale_depth=1.4,
            use_mup=use_mup,
            rope_parameters=None,
        )
        layer = VoxCPM2DecoderLayer(config, layer_idx=0)
        expected_keys = {
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        }
        assert set(layer.state_dict()) == expected_keys

        layer.input_layernorm = torch.nn.Identity()
        layer.post_attention_layernorm = torch.nn.Identity()
        layer.self_attn = torch.nn.Identity()
        layer.self_attn.forward = lambda hidden_states, **kwargs: (hidden_states, None)
        layer.mlp = torch.nn.Identity()

        hidden_states = torch.randn(1, 3, 6)
        output = layer(hidden_states, position_embeddings=None, is_causal=False)
        expected_scale = 1.4 / math.sqrt(2) if use_mup else 1.0
        expected_output = hidden_states + hidden_states * expected_scale
        expected_output = expected_output + expected_output * expected_scale
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        assert layer.residual_scale == expected_scale


@require_torch
def test_rms_normalization_matches_reference():
    layer = VoxCPM2RMSNorm(6, eps=1e-5)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]))

    assert list(layer.state_dict()) == ["weight"]
    hidden_states = torch.randn(2, 3, 6)
    variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
    expected_output = layer.weight * (hidden_states * torch.rsqrt(variance + layer.variance_epsilon))
    torch.testing.assert_close(layer(hidden_states), expected_output, rtol=0, atol=0)
