# coding=utf-8
# Copyright 2023 The VITS Authors and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch VITS model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "VitsConfig"


VITS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # "TODO",
    # See all VITS models at https://huggingface.co/models?filter=vits
]


@dataclass
class VitsModelOutput(ModelOutput):
    """
    Describes the outputs for the VITS model.

    Args:
        audio (`torch.FloatTensor` of shape `(batch_size, 1, sequence_length)`):
            Audio waveform predicted by the model.
        sequence_lengths  (`torch.FloatTensor` of shape `(batch_size,)`):
            The length in samples of each element in the `audio` batch.
    """

    audio: torch.FloatTensor = None
    sequence_lengths: torch.FloatTensor = None


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
    num_channels_int = num_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :num_channels_int, :])
    s_act = torch.sigmoid(in_act[:, num_channels_int:, :])
    acts = t_act * s_act
    return acts


class VitsPosteriorEncoder(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.out_channels = config.flow_size

        self.conv_pre = nn.Conv1d(config.spectrogram_bins, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=16)
        self.conv_proj = nn.Conv1d(config.hidden_size, self.out_channels * 2, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        inputs = self.conv_pre(inputs) * padding_mask
        inputs = self.wavenet(inputs, padding_mask, global_conditioning)
        stats = self.conv_proj(inputs) * padding_mask
        mean, log_stddev = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_stddev)) * padding_mask
        return z, mean, log_stddev


class VitsHifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class VitsHifiGan(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = nn.Conv1d(
            config.flow_size,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(
                    VitsHifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope)
                )

        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False)

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.upsample_initial_channel, 1)

    def apply_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()

    def remove_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()

    def forward(
        self, spectrogram: torch.FloatTensor, global_conditioning: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        r"""
        Converts a spectrogram into a speech waveform.

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`):
                Tensor containing the spectrograms.
            global_conditioning (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_size, 1)`, *optional*):
                Tensor containing speaker embeddings, for multispeaker models.

        Returns:
            `torch.FloatTensor`: Tensor of shape shape `(batch_size, 1, num_frames)` containing the speech waveform.
        """
        hidden_states = self.conv_pre(spectrogram)

        if global_conditioning is not None:
            hidden_states = hidden_states + self.cond(global_conditioning)

        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        waveform = torch.tanh(hidden_states)
        return waveform


class VitsWaveNet(torch.nn.Module):
    def __init__(self, config: VitsConfig, num_layers: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(config.wavenet_dropout)

        if config.speaker_embedding_size != 0:
            cond_layer = torch.nn.Conv1d(config.speaker_embedding_size, 2 * config.hidden_size * num_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(num_layers):
            dilation = config.wavenet_dilation_rate**i
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            in_layer = torch.nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=2 * config.hidden_size,
                kernel_size=config.wavenet_kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < num_layers - 1:
                res_skip_channels = 2 * config.hidden_size
            else:
                res_skip_channels = config.hidden_size

            res_skip_layer = torch.nn.Conv1d(config.hidden_size, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        outputs = torch.zeros_like(inputs)
        num_channels_tensor = torch.IntTensor([self.hidden_size])

        if global_conditioning is not None:
            global_conditioning = self.cond_layer(global_conditioning)

        for i in range(self.num_layers):
            hidden_states = self.in_layers[i](inputs)

            if global_conditioning is not None:
                cond_offset = i * 2 * self.hidden_size
                global_states = global_conditioning[:, cond_offset : cond_offset + 2 * self.hidden_size, :]
            else:
                global_states = torch.zeros_like(hidden_states)

            acts = fused_add_tanh_sigmoid_multiply(hidden_states, global_states, num_channels_tensor)
            acts = self.dropout(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_size, :]
                inputs = (inputs + res_acts) * padding_mask
                outputs = outputs + res_skip_acts[:, self.hidden_size :, :]
            else:
                outputs = outputs + res_skip_acts

        return outputs * padding_mask

    def remove_weight_norm(self):
        if self.speaker_embedding_size != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class VitsResidualCouplingLayer(nn.Module):
    def __init__(self, config: VitsConfig, mean_only: bool = False):
        super().__init__()
        self.half_channels = config.flow_size // 2
        self.mean_only = mean_only

        self.conv_pre = nn.Conv1d(self.half_channels, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=4)
        self.conv_post = nn.Conv1d(config.hidden_size, self.half_channels * (2 - mean_only), 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        x0, x1 = torch.split(inputs, [self.half_channels] * 2, dim=1)
        h = self.conv_pre(x0) * padding_mask
        h = self.wavenet(h, padding_mask, global_conditioning)
        stats = self.conv_post(h) * padding_mask

        if self.mean_only:
            mean = stats
            log_stddev = torch.zeros_like(mean)
        else:
            mean, log_stddev = torch.split(stats, [self.half_channels] * 2, dim=1)

        if not reverse:
            x1 = mean + x1 * torch.exp(log_stddev) * padding_mask
            outputs = torch.cat([x0, x1], dim=1)
            log_determinant = torch.sum(log_stddev, [1, 2])
            return outputs, log_determinant
        else:
            x1 = (x1 - mean) * torch.exp(-log_stddev) * padding_mask
            outputs = torch.cat([x0, x1], dim=1)
            return outputs


class VitsResidualCouplingBlock(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(config.prior_encoder_num_flows):
            self.flows.append(VitsResidualCouplingLayer(config, mean_only=True))
            self.flows.append(VitsFlip())

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                inputs, _ = flow(inputs, padding_mask, global_conditioning)
        else:
            for flow in reversed(self.flows):
                inputs = flow(inputs, padding_mask, global_conditioning, reverse=True)
        return inputs


class VitsDilatedDepthSeparableConv(nn.Module):
    def __init__(self, channels, kernel_size, num_layers, dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout_rate)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(num_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels))
            self.norms_2.append(nn.LayerNorm(channels))

    def forward(self, inputs, padding_mask, global_conditioning=None):
        if global_conditioning is not None:
            inputs = inputs + global_conditioning

        for i in range(self.num_layers):
            hidden_states = self.convs_sep[i](inputs * padding_mask)
            hidden_states = self.norms_1[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.convs_1x1[i](hidden_states)
            hidden_states = self.norms_2[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            hidden_states = self.dropout(hidden_states)
            inputs = inputs + hidden_states

        return inputs * padding_mask


class VitsConvFlow(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, num_layers, num_bins=10, tail_bound=5.0):
        super().__init__()
        self.filter_channels = filter_channels
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.conv_pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(filter_channels, kernel_size, num_layers, dropout_rate=0.0)
        self.conv_proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        x0, x1 = torch.split(inputs, [self.half_channels] * 2, dim=1)

        hidden_states = self.conv_pre(x0)
        hidden_states = self.conv_dds(hidden_states, padding_mask, global_conditioning)
        hidden_states = self.conv_proj(hidden_states) * padding_mask

        batch_size, channels, length = x0.shape
        hidden_states = hidden_states.reshape(batch_size, channels, -1, length).permute(0, 1, 3, 2)

        unnormalized_widths = hidden_states[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = hidden_states[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = hidden_states[..., 2 * self.num_bins :]

        x1, log_abs_det = self._unconstrained_rational_quadratic_spline(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tail_bound=self.tail_bound,
        )

        outputs = torch.cat([x0, x1], dim=1) * padding_mask
        if not reverse:
            log_determinant = torch.sum(log_abs_det * padding_mask, [1, 2])
            return outputs, log_determinant
        else:
            return outputs

    def _unconstrained_rational_quadratic_spline(
        self,
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse=False,
        tail_bound=1.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
    ):
        inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
        outside_interval_mask = ~inside_interval_mask

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)
        constant = np.log(np.exp(1 - min_derivative) - 1)

        unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0.0

        outputs[inside_interval_mask], logabsdet[inside_interval_mask] = self._rational_quadratic_spline(
            inputs=inputs[inside_interval_mask],
            unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
            unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
        return outputs, logabsdet

    def _rational_quadratic_spline(
        self,
        inputs,
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        inverse,
        left,
        right,
        bottom,
        top,
        min_bin_width,
        min_bin_height,
        min_derivative,
    ):
        if torch.min(inputs) < left or torch.max(inputs) > right:
            raise ValueError("Input to a transform is not within its domain")

        num_bins = unnormalized_widths.shape[-1]

        if min_bin_width * num_bins > 1.0:
            raise ValueError(f"Minimal bin width {min_bin_width} too large for the number of bins {num_bins}")
        if min_bin_height * num_bins > 1.0:
            raise ValueError(f"Minimal bin height {min_bin_height} too large for the number of bins {num_bins}")

        widths = nn.functional.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = min_derivative + nn.functional.softplus(unnormalized_derivatives)

        heights = nn.functional.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = nn.functional.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        bin_locations = cumheights if inverse else cumwidths
        bin_locations[..., -1] += 1e-6
        bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
        bin_idx = bin_idx[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if inverse:
            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            )
            c = -input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            if not (discriminant >= 0).all():
                raise RuntimeError(f"invalid discriminant {discriminant}")

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
            return outputs, -logabsdet
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
            return outputs, logabsdet


class VitsLog(nn.Module):
    def forward(self, inputs, padding_mask, reverse=False, **kwargs):
        if not reverse:
            outputs = torch.log(torch.clamp_min(inputs, 1e-5)) * padding_mask
            log_determinant = torch.sum(-outputs, [1, 2])
            return outputs, log_determinant
        else:
            outputs = torch.exp(inputs) * padding_mask
            return outputs


class VitsFlip(nn.Module):
    def forward(self, inputs, *args, reverse=False, **kwargs):
        outputs = torch.flip(inputs, [1])
        if not reverse:
            log_determinant = torch.zeros(outputs.size(0)).to(dtype=outputs.dtype, device=outputs.device)
            return outputs, log_determinant
        else:
            return outputs


class VitsElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.translate = nn.Parameter(torch.zeros(channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, inputs, padding_mask, reverse=False, **kwargs):
        if not reverse:
            outputs = self.translate + torch.exp(self.log_scale) * inputs
            outputs = outputs * padding_mask
            log_determinant = torch.sum(self.log_scale * padding_mask, [1, 2])
            return outputs, log_determinant
        else:
            outputs = (inputs - self.translate) * torch.exp(-self.log_scale) * padding_mask
            return outputs


class VitsStochasticDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.hidden_size

        self.conv_pre = nn.Conv1d(config.hidden_size, filter_channels, 1)
        self.conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.conv_dds = VitsDilatedDepthSeparableConv(
            filter_channels, kernel_size, 3, config.duration_predictor_dropout
        )

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, filter_channels, 1)

        self.flows = nn.ModuleList()
        self.flows.append(VitsElementwiseAffine(2))
        for _ in range(config.duration_predictor_num_flows):
            self.flows.append(VitsConvFlow(2, filter_channels, kernel_size, num_layers=3))
            self.flows.append(VitsFlip())

        self.log_flow = VitsLog()
        self.post_conv_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_conv_dds = VitsDilatedDepthSeparableConv(
            filter_channels, kernel_size, 3, config.duration_predictor_dropout
        )

        self.post_flows = nn.ModuleList()
        self.post_flows.append(VitsElementwiseAffine(2))
        for _ in range(config.duration_predictor_num_flows):
            self.post_flows.append(VitsConvFlow(2, filter_channels, kernel_size, num_layers=3))
            self.post_flows.append(VitsFlip())

    def forward(self, inputs, padding_mask, global_conditioning=None, w=None, reverse=False, noise_scale=1.0):
        inputs = torch.detach(inputs)
        inputs = self.conv_pre(inputs)

        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        inputs = self.conv_dds(inputs, padding_mask)
        inputs = self.conv_proj(inputs) * padding_mask

        if not reverse:
            h_w = self.post_conv_pre(w)
            h_w = self.post_conv_dds(h_w, padding_mask)
            h_w = self.post_conv_proj(h_w) * padding_mask

            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=inputs.device, dtype=inputs.dtype) * padding_mask
            z_q = e_q
            logdet_tot_q = 0
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, padding_mask, global_conditioning=inputs + h_w)
                logdet_tot_q += logdet_q

            z_u, z1 = torch.split(z_q, [1, 1], dim=1)
            u = torch.sigmoid(z_u) * padding_mask
            z0 = (w - u) * padding_mask

            logdet_tot_q += torch.sum(
                (nn.functional.logsigmoid(z_u) + nn.functional.logsigmoid(-z_u)) * padding_mask, [1, 2]
            )
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * padding_mask, [1, 2]) - logdet_tot_q

            z0, logdet_tot = self.log_flow(z0, padding_mask)
            z = torch.cat([z0, z1], dim=1)
            for flow in self.flows:
                z, logdet = flow(z, padding_mask, global_conditioning=inputs)
                logdet_tot = logdet_tot + logdet

            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * padding_mask, [1, 2]) - logdet_tot
            return nll + logq

        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow

            z = (
                torch.randn(inputs.size(0), 2, inputs.size(2)).to(device=inputs.device, dtype=inputs.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, padding_mask, global_conditioning=inputs, reverse=True)

            logw, _ = torch.split(z, [1, 1], dim=1)
            return logw


class VitsDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.duration_predictor_filter_channels

        self.dropout = nn.Dropout(config.duration_predictor_dropout)
        self.conv_1 = nn.Conv1d(config.hidden_size, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.hidden_size, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        inputs = torch.detach(inputs)

        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        inputs = self.conv_1(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_1(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        inputs = self.conv_2(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_2(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        inputs = self.proj(inputs * padding_mask)
        return inputs * padding_mask


class VitsAttention(nn.Module):
    """Multi-headed attention with relative positional representation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        window_size: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if window_size:
            self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.head_dim) * self.scaling)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, src_len)
            relative_logits = torch.matmul(query_states, key_relative_embeddings.transpose(-2, -1))
            rel_pos_bias = self._relative_position_to_absolute_position(relative_logits)
            attn_weights += rel_pos_bias

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.window_size is not None:
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, src_len)
            relative_weights = self._absolute_position_to_relative_position(attn_probs)
            rel_pos_bias = torch.matmul(relative_weights, value_relative_embeddings)
            attn_output += rel_pos_bias

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])

        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        return relative_embeddings[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        batch_heads, length, _ = x.size()

        # Concat columns of pad to shift from relative to absolute indexing.
        x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch_heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
        x_final = x_final[:, :length, length - 1 :]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch_heads, length, _ = x.size()

        # Pad along column
        x = nn.functional.pad(x, [0, length - 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length**2 + length * (length - 1)])

        # Add 0's in the beginning that will skew the elements after reshape
        x_flat = nn.functional.pad(x_flat, [length, 0, 0, 0])
        x_final = x_flat.view([batch_heads, length, 2 * length])[:, :, 1:]
        return x_final


class VitsFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_1 = nn.Conv1d(config.hidden_size, config.encoder_ffn_dim, config.ffn_kernel_size)
        self.conv_2 = nn.Conv1d(config.encoder_ffn_dim, config.hidden_size, config.ffn_kernel_size)
        self.dropout = nn.Dropout(config.activation_dropout)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    def forward(self, hidden_states, padding_mask):
        hidden_states = hidden_states.permute(0, 2, 1)
        padding_mask = padding_mask.permute(0, 2, 1)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states * padding_mask

        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states


class VitsEncoderLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.attention = VitsAttention(
            embed_dim=config.hidden_size,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = VitsFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states, padding_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VitsEncoder(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([VitsEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        hidden_states = hidden_states * padding_mask

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        padding_mask,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = hidden_states * padding_mask

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class VitsTextEncoder(nn.Module):
    """
    Transformer encoder that uses relative positional representation instead of absolute positional encoding.
    """

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.encoder = VitsEncoder(config)
        self.project = nn.Conv1d(config.hidden_size, config.flow_size * 2, kernel_size=1)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
        )

        last_hidden_state = encoder_outputs[0]

        stats = self.project(last_hidden_state.transpose(1, 2)).transpose(1, 2) * padding_mask
        means, log_variances = torch.split(stats, self.config.flow_size, dim=2)

        return (last_hidden_state, means, log_variances)


class VitsPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VitsConfig
    base_model_prefix = "vits"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (VitsTextEncoder)):
            module.gradient_checkpointing = value


VITS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VitsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


VITS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        speed (`float`, *optional*, defaults to 1.0):
            Speaking rate. Larger values are faster.
        noise_scale (`float`, *optional*, defaults to 1.0):
            How random the speech prediction is.
        noise_scale_duration (`float`, *optional*, defaults to 1.0):
            How random the duration prediction is.
        speaker_id (`int`, *optional*):
            Which speaker embedding to use. Only used for multispeaker models.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The VITS model.",
    VITS_START_DOCSTRING,
)
class VitsModel(VitsPreTrainedModel):
    def __init__(self, config: VitsConfig):
        super().__init__(config)
        self.config = config
        self.text_encoder = VitsTextEncoder(config)
        self.flow = VitsResidualCouplingBlock(config)
        self.decoder = VitsHifiGan(config)

        if config.use_stochastic_duration_prediction:
            self.duration_predictor = VitsStochasticDurationPredictor(config)
        else:
            self.duration_predictor = VitsDurationPredictor(config)

        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, config.speaker_embedding_size)

        # This is used only for training.
        self.posterior_encoder = VitsPosteriorEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.text_encoder

    @add_start_docstrings_to_model_forward(VITS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VitsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speed: float = 1.0,
        noise_scale: float = 1.0,
        noise_scale_duration: float = 1.0,
        speaker_id: Optional[int] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], VitsModelOutput]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, *optional*):
            Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
            computation.

        Returns:

        Example:

        ```python
        >>> from transformers import VitsMmsTokenizer, VitsModel, set_seed
        >>> import torch

        >>> tokenizer = VitsMmsTokenizer.from_pretrained("TODO")
        >>> model = VitsModel.from_pretrained("TODO")

        >>> inputs = tokenizer(text="Hello, my dog is cute", return_tensors="pt")

        >>> set_seed(555)  # make deterministic

        >>> # generate speech
        >>> outputs = model(inputs["input_ids"])
        >>> outputs.audio.shape
        torch.Size([1, 1, 33280])
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            input_padding_mask = attention_mask.unsqueeze(-1).float()
        else:
            input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()

        # Workaround for tokenizer: filter out padding tokens.
        input_ids[input_ids >= self.config.vocab_size] = 0

        hidden_states, means_prior, log_variances_prior = self.text_encoder(
            input_ids, input_padding_mask, attention_mask
        )

        hidden_states = hidden_states.transpose(1, 2)
        input_padding_mask = input_padding_mask.transpose(1, 2)

        if self.config.num_speakers > 1:
            if speaker_id is None:
                raise ValueError("Expected speaker_id")
            speaker_embeddings = self.embed_speaker(torch.tensor([speaker_id])).unsqueeze(-1)
        else:
            speaker_embeddings = None

        if labels is not None:
            raise NotImplementedError("Training of VITS is not supported yet.")

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states, input_padding_mask, speaker_embeddings, reverse=True, noise_scale=noise_scale_duration
            )
        else:
            log_duration = self.duration_predictor(hidden_states, input_padding_mask, speaker_embeddings)

        length_scale = 1.0 / speed
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)
        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        # Create a padding mask for the output lengths of shape (batch, 1, max_output_length)
        indices = torch.arange(predicted_lengths.max(), dtype=predicted_lengths.dtype, device=predicted_lengths.device)
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        # Reconstruct an attention tensor of shape (batch, 1, out_length, in_length)
        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(output_padding_mask, -1)
        attn = self._generate_path(duration, attn_mask)

        # Expand prior
        means_prior = torch.matmul(attn.squeeze(1), means_prior).transpose(1, 2)
        log_variances_prior = torch.matmul(attn.squeeze(1), log_variances_prior).transpose(1, 2)

        z_p = means_prior + torch.randn_like(means_prior) * torch.exp(log_variances_prior) * noise_scale
        z = self.flow(z_p, output_padding_mask, speaker_embeddings, reverse=True)

        predicted_audio = self.decoder((z * output_padding_mask), speaker_embeddings)
        sequence_lengths = predicted_lengths * 256

        if return_dict:
            return VitsModelOutput(audio=predicted_audio, sequence_lengths=sequence_lengths)

        return (predicted_audio, sequence_lengths)

    def _generate_path(self, duration, mask):
        batch_size, _, output_length, input_length = mask.shape
        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(output_length, dtype=duration.dtype, device=duration.device)
        path = indices.unsqueeze(0) < cum_duration
        path = path.to(mask.dtype).view(batch_size, input_length, output_length)
        path = path - nn.functional.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
        path = path.unsqueeze(1).transpose(2, 3) * mask
        return path
