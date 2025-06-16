# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from .configuration_hiftnet import HiFTNetConfig


logger = logging.get_logger(__name__)


@auto_docstring(
    # TODO: @eustlb, add docstring
    custom_intro=""" """
)
@auto_docstring
class HiFTNetPreTrainedModel(PreTrainedModel):
    # TODO: @eustlb, fill this
    config_class = HiFTNetConfig

    def _init_weights(self, module):
        pass


class HiFTNetResBlockLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, 1, dilation=dilation, padding=self._get_padding(kernel_size, dilation)
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
        )
        self.alpha1 = nn.Parameter(torch.ones(channels))
        self.alpha2 = nn.Parameter(torch.ones(channels))

        # apply weight norm
        nn.utils.parametrizations.weight_norm(self.conv1)
        nn.utils.parametrizations.weight_norm(self.conv2)

    def _get_padding(self, kernel_size, dilation):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, hidden_states, input_lengths):
        residual = hidden_states
        residual = residual + (1 / self.alpha1) * (
            torch.sin(self.alpha1 * residual) ** 2
        )  # TODO: to be replaced with snake activation function !!!!
        residual = self.conv1(residual.transpose(1, 2)).transpose(1, 2)
        residual = _mask_hidden_states(residual, input_lengths)

        residual = residual + (1 / self.alpha2) * (torch.sin(self.alpha2 * residual) ** 2)
        residual = self.conv2(residual.transpose(1, 2)).transpose(1, 2)
        residual = _mask_hidden_states(residual, input_lengths)

        return residual + hidden_states


class HiFTNetResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList([HiFTNetResBlockLayer(channels, kernel_size, dilation) for dilation in dilations])

    def forward(self, hidden_states, input_lengths=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, input_lengths)
        return hidden_states


class HiFTNetHarmonicNoiseSourceFilter(nn.Module):
    """
    Harmonic plus Noise Neural Source Filter.
    See: https://arxiv.org/abs/2309.09493
    Adapted from: https://github.com/yl4579/StyleTT2
    """

    def __init__(
        self,
        sampling_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amplitude=0.1,
        add_noise_std=0.003,
        voiced_threshold=0,
    ):
        """
        Args:
            samp_rate: (`int`):
                Sampling rate in Hz.
            upsample_scale: (`int`):
                Upsampling scale.
            harmonic_num: (`int`, *optional*, defaults to 0):
                Number of harmonic overtones.
            sine_amplitude: (`float`, *optional*, defaults to 0.1):
                Amplitude of sine-waveform.
            add_noise_std: (`float`, *optional*, defaults to 0.003):
                Standard deviation of Gaussian noise.
            voiced_threshold: (`float`, *optional*, defaults to 0):
                F0 threshold for U/V classification.
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.upsample_scale = upsample_scale
        self.harmonic_num = harmonic_num
        self.sine_amplitude = sine_amplitude
        self.add_noise_std = add_noise_std
        self.voiced_threshold = voiced_threshold

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini = torch.zeros(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] += rand_ini

        rad_values = F.interpolate(
            rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear"
        ).transpose(1, 2)
        phase = rad_values.cumsum(dim=1) * 2 * torch.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear"
        ).transpose(1, 2)
        sines = phase.sin()

        return sines

    def _sine_gen(self, f0):
        # generate sine waveforms
        fn = f0 * torch.arange(1, self.harmonic_num + 2, device=f0.device)
        sine_waves = self._f02sine(fn) * self.sine_amplitude

        # generate uv signal
        uv = (f0 > self.voiced_threshold).float()
        noise_amp = uv * self.add_noise_std + (1 - uv) * self.sine_amplitude / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise

        return sine_waves

    def forward(self, hidden_states):
        with torch.no_grad():
            sine_wavs = self._sine_gen(hidden_states)
        sine_merge = F.tanh(self.l_linear(sine_wavs))

        return sine_merge


class HiFTNetGeneratorLayer(nn.Module):
    def __init__(self, layer_idx, config, reflection_pad=False):
        super().__init__()
        self.layer_idx = layer_idx

        c_cur = config.upsample_initial_channel // (2 ** (layer_idx + 1))
        if layer_idx + 1 < len(config.upsample_rates):
            noise_conv_stride = math.prod(config.upsample_rates[layer_idx + 1 :])
            noise_conv_padding = (noise_conv_stride + 1) // 2
            noise_conv_kernel_size = noise_conv_stride * 2
            noise_res_kernel_size = 7
        else:
            noise_conv_stride = 1
            noise_conv_padding = 0
            noise_conv_kernel_size = 1
            noise_res_kernel_size = 11

        self.up = nn.ConvTranspose1d(
            config.upsample_initial_channel // (2**layer_idx),
            c_cur,
            config.upsample_kernel_sizes[layer_idx],
            config.upsample_rates[layer_idx],
            padding=(config.upsample_kernel_sizes[layer_idx] - config.upsample_rates[layer_idx]) // 2,
        )

        # TODO: @eustlb, remove comment
        # -> source downs in original code
        self.noise_conv = nn.Conv1d(
            config.n_fft + 2,
            c_cur,
            kernel_size=noise_conv_kernel_size,
            stride=noise_conv_stride,
            padding=noise_conv_padding,
        )

        # TODO: @eustlb, remove comment
        # -> source_resblocks in original code
        self.noise_res = HiFTNetResBlock(c_cur, noise_res_kernel_size, (1, 3, 5))

        # TODO: @eustlb, remove comment
        # -> resblocks in original code
        self.resblocks = nn.ModuleList(
            [
                HiFTNetResBlock(c_cur, kernel_size, dilation)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )

        self.reflection_pad = nn.ReflectionPad1d((1, 0)) if reflection_pad else nn.Identity()

        # apply weight norm
        nn.utils.parametrizations.weight_norm(self.up)

    def _noise_conv_out_length(self, input_lengths):
        if input_lengths is None:
            return None
        new_input_lengths = []
        for l in input_lengths:
            new_input_lengths.append(
                int(
                    (
                        l
                        + 2 * self.noise_conv.padding[0]
                        - self.noise_conv.dilation[0] * (self.noise_conv.kernel_size[0] - 1)
                        - 1
                    )
                    / self.noise_conv.stride[0]
                    + 1
                )
            )
        return new_input_lengths

    def _upsample_out_length(self, input_lengths):
        if input_lengths is None:
            return None
        new_input_lengths = []
        for l in input_lengths:
            new_input_lengths.append(
                (l - 1) * self.up.stride[0]
                - 2 * self.up.padding[0]
                + self.up.dilation[0] * (self.up.kernel_size[0] - 1)
                + self.up.output_padding[0]
                + 1
            )
        return new_input_lengths

    def _reflection_pad_out_length(self, input_lengths):
        if input_lengths is None:
            return None
        elif isinstance(self.reflection_pad, nn.Identity):
            return input_lengths
        else:
            return [l + 1 for l in input_lengths]

    def forward(self, hidden_states, hidden_states_source, input_lengths=None, source_lengths=None):
        hidden_states = F.leaky_relu(hidden_states, 0.1)
        hidden_states_source = self.noise_conv(hidden_states_source.transpose(1, 2)).transpose(1, 2)
        source_lengths = self._noise_conv_out_length(source_lengths)
        hidden_states_source = _mask_hidden_states(hidden_states_source, source_lengths)

        hidden_states_source = self.noise_res(hidden_states_source, source_lengths)
        hidden_states = self.up(hidden_states.transpose(1, 2))
        hidden_states_lengths = self._upsample_out_length(input_lengths)
        hidden_states = _mask_hidden_states(hidden_states.transpose(1, 2), hidden_states_lengths)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.reflection_pad(hidden_states).transpose(1, 2)
        hidden_states_lengths = self._reflection_pad_out_length(hidden_states_lengths)
        hidden_states = hidden_states + hidden_states_source
        hidden_states = sum(resblock(hidden_states, hidden_states_lengths) for resblock in self.resblocks) / len(
            self.resblocks
        )

        return hidden_states, hidden_states_lengths


# TODO: @eustlb, check if this naming convention is the best regarding Transformers
@auto_docstring
class HiFTNetModel(HiFTNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.n_fft = config.n_fft
        self.hop_length = config.hop_size
        self.win_length = config.n_fft
        self.window = torch.hann_window(config.n_fft)
        self.scale_factor = math.prod(config.upsample_rates) * config.hop_size

        # TODO: @eustlb, use config parameters
        self.conv_pre = nn.Conv1d(80, config.upsample_initial_channel, 7, 1, padding=3)

        self.f0_upsamp = nn.Upsample(scale_factor=self.scale_factor)
        self.m_source = HiFTNetHarmonicNoiseSourceFilter(
            sampling_rate=config.sampling_rate, upsample_scale=self.scale_factor, harmonic_num=8, voiced_threshold=10
        )

        self.layers = nn.ModuleList(
            [
                HiFTNetGeneratorLayer(
                    layer_idx, config, reflection_pad=True if layer_idx == len(config.upsample_rates) - 1 else False
                )
                for layer_idx in range(len(config.upsample_rates))
            ]
        )

        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2 ** (len(config.upsample_rates))),
            config.n_fft + 2,
            7,
            padding=3,
        )

        # apply weight norm
        nn.utils.parametrizations.weight_norm(self.conv_pre)
        nn.utils.parametrizations.weight_norm(self.conv_post)

    def _stft_output_length(self, length):
        return 1 + length // self.hop_length

    def forward(self, hidden_states, f0, input_lengths=None):
        # TODO: hidden states is actually the mel spectrogram, find a better name for it
        # TODO: f0 is the fundamental frequency, find a better name for it

        with torch.no_grad():
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
            har_source = self.m_source(f0)

            har_source_lengths = [l * self.scale_factor for l in input_lengths] if input_lengths is not None else None
            har_source = _mask_hidden_states(har_source, har_source_lengths)
            har_source = har_source.transpose(1, 2).squeeze(1)

            har_transform = torch.stft(
                har_source,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window.to(har_source.device),
                return_complex=True,
            )

            har_transform_lengths = (
                [self._stft_output_length(l) for l in har_source_lengths] if har_source_lengths is not None else None
            )
            har_transform = _mask_hidden_states(har_transform.transpose(1, 2), har_transform_lengths)
            har_transform = har_transform.transpose(1, 2)

            har_spec, har_phase = har_transform.abs(), har_transform.angle()
            har = torch.cat([har_spec, har_phase], dim=1).transpose(1, 2)

        hidden_states = self.conv_pre(hidden_states)

        for layer in self.layers:
            hidden_states, input_lengths = layer(hidden_states, har, input_lengths, har_transform_lengths)

        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states.transpose(1, 2))
        hidden_states = _mask_hidden_states(hidden_states.transpose(1, 2), har_transform_lengths)
        hidden_states = hidden_states.transpose(1, 2)
        spec = torch.exp(hidden_states[:, : self.n_fft // 2 + 1, :])
        phase = torch.sin(hidden_states[:, self.n_fft // 2 + 1 :, :])

        return spec, phase, har_source_lengths
    
        inverse_transform = torch.istft(
            spec * torch.exp(phase * 1j),
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window.to(har_source.device),
        )

        if har_source_lengths is not None:
            mask = (
                torch.arange(inverse_transform.size(1), device=inverse_transform.device)[None, :]
                < torch.tensor(har_source_lengths, device=inverse_transform.device)[:, None]
            )
            inverse_transform = inverse_transform * mask

        return inverse_transform, har_source_lengths


#TODO: @eustlb, check if this naming convention is the best regarding Transformers
class HiFTNetFundamentalFrequencyPredictor(HiFTNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.condnet = nn.Sequential(
            nn.Conv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.ELU(),
        )

        # apply weight norm
        for layer in self.condnet:
            if isinstance(layer, nn.Conv1d):
                nn.utils.parametrizations.weight_norm(layer)

        self.linear = nn.Linear(in_features=config.hidden_size, out_features=1)

    def forward(self, input_features: torch.Tensor, input_lengths=None) -> torch.Tensor:
        hidden_states = self.condnet(input_features)
        hidden_states = self.linear(hidden_states.transpose(1, 2))
        f0 = torch.abs(hidden_states.squeeze(-1))
        if input_lengths is not None:
            f0 = _mask_hidden_states(f0, input_lengths)
        return f0


class HiFTNetVocoder(HiFTNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.fundamental_frequency_predictor = HiFTNetFundamentalFrequencyPredictor(config)
        self.model = HiFTNetModel(config)
        self.window = torch.hann_window(config.n_fft)

        # TODO: @eustlb, a bit strange diff naming convention
        self.win_length = config.n_fft
        self.hop_length = config.hop_size

    def forward(self, input_features, input_lengths=None):
        fundamental_frequency = self.fundamental_frequency_predictor(input_features)
        spec, phase, har_source_lengths = self.model(input_features, fundamental_frequency, input_lengths)

        waveform = torch.istft(
            spec * torch.exp(phase * 1j),
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window.to(spec.device),
        )

        if har_source_lengths is not None:
            mask = (
                torch.arange(waveform.size(1), device=waveform.device)[None, :]
                < torch.tensor(har_source_lengths, device=waveform.device)[:, None]
            )
            waveform = waveform * mask

        return waveform, har_source_lengths, fundamental_frequency


def _mask_hidden_states(hidden_states: torch.FloatTensor, lengths: Optional[List[int]] = None) -> torch.FloatTensor:
    """Create boolean mask based on given input lengths and apply it to the hidden states.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Hidden states to mask.
        lengths (`List[int]`, *optional*):
            List of sequence lengths for each batch element. If None, no masking is applied.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`:
            Masked hidden states where values beyond each sequence length are set to 0.
    """

    # is not provided length or all lengths are the same and equal to the sequence lengthm
    # simply return the hidden states
    if lengths is None or (len(set(lengths)) == 1 and lengths[0] == hidden_states.shape[1]):
        return hidden_states

    _, seq_len, hidden_dim = hidden_states.shape
    mask = (
        torch.arange(seq_len, device=hidden_states.device)[None, :]
        < torch.tensor(lengths, device=hidden_states.device)[:, None]
    )
    mask = mask.unsqueeze(-1).expand(-1, -1, hidden_dim)
    return hidden_states * mask


__all__ = [
    "HiFTNetModel",
    "HiFTNetFundamentalFrequencyPredictor",
    "HiFTNetVocoder",
]
