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
        self.upsample = nn.Upsample(scale_factor=upsample_scale)

        # to merge source harmonics into a single excitation
        self.linear = torch.nn.Linear(harmonic_num + 1, 1)

    def forward(self, fundamental_frequency):
        # fundamental frequency: shape (batch_size, 1, sample_len)

        fundamental_frequency = self.upsample(fundamental_frequency[:, None]).transpose(1, 2)  # bs,n,t

        with torch.no_grad():
            # ----------------------------
            # sinusoidal source

            # shape (batch_size, num_harmonic + 2, sample_len)
            harmonic_overtones = fundamental_frequency * torch.arange(1, self.num_harmonic + 2, device=fundamental_frequency.device).unsqueeze(-1)
            harmonic_overtones = harmonic_overtones / self.sampling_rate % 1

            random_initialization = torch.rand(fundamental_frequency.shape[0], 1, self.num_harmonic + 1, device=fundamental_frequency.device)
            random_initialization[:, 0, :] = 0
            harmonic_overtones = harmonic_overtones + random_initialization

            harmonic_overtones = F.interpolate(
                harmonic_overtones, scale_factor=1 / self.upsample_scale, mode="linear"
            ) 
            phase =  2 * torch.pi * harmonic_overtones.cumsum(dim=-1)
            phase = F.interpolate(
                phase * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear"
            )
            sines = phase.sin()
            # ----------------------------

            # ----------------------------
            # noise source

            # voiced/unvoiced segments by thresholding
            is_voiced = fundamental_frequency > self.voiced_threshold
            noise_amp = is_voiced * self.noise_std + (1 - is_voiced) * self.sine_amplitude / 3
            noise = noise_amp * torch.randn_like(sines)
            # ----------------------------

            sines = sines * is_voiced + noise

        return F.tanh(self.linear(sines))


class Snake(nn.Module):
    """
    Snake activation function.
    See: https://arxiv.org/abs/2006.08195
    """
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
    
    def forward(self, hidden_states):
        return hidden_states + 1 / self.alpha * torch.sin(self.alpha * hidden_states) ** 2


class HiFTNetResidualBlockLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, 1, dilation=dilation, padding=self._get_padding(kernel_size, dilation)
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, 1, dilation=1, padding=self._get_padding(kernel_size, 1)
        )
        self.snake_activation_1 = Snake(channels)
        self.snake_activation_2 = Snake(channels)

        # apply weight norm
        nn.utils.parametrizations.weight_norm(self.conv1)
        nn.utils.parametrizations.weight_norm(self.conv2)

    def _get_padding(self, kernel_size, dilation):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, hidden_states, input_lengths):
        residual = self.snake_activation_1(hidden_states)
        residual = self.conv1(residual.transpose(1, 2)).transpose(1, 2)

        residual = _mask_hidden_states(residual, input_lengths)

        residual = self.snake_activation_2(residual)
        residual = self.conv2(residual.transpose(1, 2)).transpose(1, 2)
        residual = _mask_hidden_states(residual, input_lengths)

        return residual + hidden_states


class HiFTNetResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList([HiFTNetResidualBlockLayer(channels, kernel_size, dilation) for dilation in dilations])

    def forward(self, hidden_states, input_lengths=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, input_lengths)
        return hidden_states


class MultiReceptiveFieldFusion(nn.Module):
    """
    Multi-receptive field fusion with snake activation.
    See: https://arxiv.org/abs/2006.08195
    """
    def __init__(self, config):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [HiFTNetResidualBlock(config.hidden_size, kernel_size, dilation) for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)]
        )
    
    def forward(self, hidden_states, hidden_states_lengths):
        return sum(resblock(hidden_states, hidden_states_lengths) for resblock in self.resblocks) / len(self.resblocks) 
        

class HiFTNetGeneratorLayer(nn.Module):
    def __init__(self, config, layer_idx, reflection_pad=False):
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


        self.nsf_conv = nn.Conv1d(
            config.n_fft + 2,
            c_cur,
            kernel_size=noise_conv_kernel_size,
            stride=noise_conv_stride,
            padding=noise_conv_padding,
        )

        self.nsf_res = HiFTNetResidualBlock(c_cur, noise_res_kernel_size, (1, 3, 5))

        self.multi_receptive_field_fusion = nn.ModuleList(
            [
                HiFTNetResidualBlock(c_cur, kernel_size, dilation)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )

        self.reflection_pad = nn.ReflectionPad1d((1, 0)) if reflection_pad else nn.Identity()

        # apply weight norm
        nn.utils.parametrizations.weight_norm(self.up)

    def forward(self, hidden_states, hidden_states_source, input_lengths=None, source_lengths=None):
        #
        #           hidden states     source spectrogram (magnitude and phase)
        #               |                               |
        #               |                               |
        #               |                               |
        #               |                               |
        #               v                               v 
        #  ConvTranspose upsampling               NSF (see paper)
        #               |                               |
        #               |                               |
        #               └────────────── + ──────────────┘
        #                               |
        #              MRF with snake activation (see paper)
        #                               |
        #                               v
        #
        hidden_states = F.leaky_relu(hidden_states, 0.1)

        # -------------------------
        # Noise Source Filter (NSF) block
        hidden_states_source = self.nsf_conv(hidden_states_source.transpose(1, 2)).transpose(1, 2)
        source_lengths = self._nsf_conv_out_length(source_lengths)
        hidden_states_source = _mask_hidden_states(hidden_states_source, source_lengths)
        hidden_states_source = self.nsf_res(hidden_states_source, source_lengths)
        # -------------------------

        # -------------------------
        # ConvTranspose upsampling
        hidden_states = self.up(hidden_states.transpose(1, 2))
        hidden_states_lengths = self._upsample_out_length(input_lengths)
        hidden_states = _mask_hidden_states(hidden_states.transpose(1, 2), hidden_states_lengths)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.reflection_pad(hidden_states).transpose(1, 2)
        hidden_states_lengths = self._reflection_pad_out_length(hidden_states_lengths)
        # -------------------------

        # -------------------------
        # add source to hidden states
        hidden_states = hidden_states + hidden_states_source
        # -------------------------

        # -------------------------
        # multi-receptive field fusion (MRF) with snake activation
        # note: cuda api is async, resblock results are computed in parallel
        hidden_states = sum(resblock(hidden_states, hidden_states_lengths) for resblock in self.resblocks) / len(
            self.resblocks
        )
        # -------------------------

        return hidden_states, hidden_states_lengths
    
    def _nsf_conv_out_length(self, input_lengths):
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

@auto_docstring
class HiFTNetModel(HiFTNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.n_fft = config.n_fft
        self.hop_length = config.hop_size
        self.win_length = config.n_fft
        self.scale_factor = math.prod(config.upsample_rates) * config.hop_size

        self.source_generator = HiFTNetHarmonicNoiseSourceFilter(
            sampling_rate=config.sampling_rate, upsample_scale=self.scale_factor, harmonic_num=8, voiced_threshold=10
        )

        self.input_conv = nn.Conv1d(80, config.upsample_initial_channel, 7, 1, padding=3)

        self.layers = nn.ModuleList(
            [
                HiFTNetGeneratorLayer(
                    config, layer_idx, reflection_pad=True if layer_idx == len(config.upsample_rates) - 1 else False
                )
                for layer_idx in range(len(config.upsample_rates))
            ]
        )

        self.output_conv = nn.Conv1d(
            config.upsample_initial_channel // (2 ** (len(config.upsample_rates))),
            config.n_fft + 2,
            7,
            padding=3,
        )

        self.register_buffer(
            "window",
            torch.hann_window(config.n_fft),
            persistent=False
        )

        # apply weight norm
        nn.utils.parametrizations.weight_norm(self.input_conv)
        nn.utils.parametrizations.weight_norm(self.output_conv)

    def _stft_output_length(self, length):
        return 1 + length // self.hop_length

    def forward(self, input_features, fundamental_frequency, input_lengths=None):
        # 1. fundamental frequency to harmonic source magnitude and phase
        with torch.no_grad():
            source = self.source_generator(fundamental_frequency)

            source_lengths = [l * self.scale_factor for l in input_lengths] if input_lengths is not None else None
            source = _mask_hidden_states(source, source_lengths)
            source = source.transpose(1, 2).squeeze(1)

            source_spectrogram = torch.stft(
                source,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                return_complex=True,
            )

            source_spectrogram_lengths = (
                [self._stft_output_length(l) for l in source_lengths] if source_lengths is not None else None
            )
            source_spectrogram = _mask_hidden_states(source_spectrogram.transpose(1, 2), source_spectrogram_lengths)
            source_spectrogram = source_spectrogram.transpose(1, 2)

            source_magnitude, source_phase = source_spectrogram.abs(), source_spectrogram.angle()
            source = torch.cat([source_magnitude, source_phase], dim=1).transpose(1, 2)

        # 2. input conv
        hidden_states = self.input_conv(input_features)

        # 3. generator layers 
        for layer in self.layers:
            hidden_states, input_lengths = layer(hidden_states, source, input_lengths, source_lengths)

        # 4. output conv
        hidden_states = F.leaky_relu(hidden_states)
        hidden_states = self.output_conv(hidden_states.transpose(1, 2))
        hidden_states = _mask_hidden_states(hidden_states.transpose(1, 2), source_spectrogram_lengths)
        hidden_states = hidden_states.transpose(1, 2)

        magnitude = torch.exp(hidden_states[:, : self.n_fft // 2 + 1, :])
        phase = torch.sin(hidden_states[:, self.n_fft // 2 + 1 :, :])

        return magnitude, phase, source_lengths


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
        # 1. fundamental frequency from mel spectrogram
        with torch.no_grad():
            fundamental_frequency = self.fundamental_frequency_predictor(input_features)
        
        # 2. mel spectrogram to magnitude and phase
        magnitude, phase, source_lengths = self.model(input_features, fundamental_frequency, input_lengths)

        # 3. retreive waveform using inverse fourier transform
        waveform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=self.window.to(magnitude.device),
        )

        if source_lengths is not None:
            mask = (
                torch.arange(waveform.size(1), device=waveform.device)[None, :]
                < torch.tensor(source_lengths, device=waveform.device)[:, None]
            )
            waveform = waveform * mask

        return waveform, source_lengths, fundamental_frequency


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
