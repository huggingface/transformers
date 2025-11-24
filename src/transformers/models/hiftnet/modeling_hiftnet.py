# coding=utf-8
# Copyright 2024 Alibaba Inc, Resemble AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from:
#   - CosyVoice HiFiGAN implementation
#   - Original HiFi-GAN: https://github.com/jik876/hifi-gan
#   - BigVGAN: https://github.com/NVIDIA/BigVGAN
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
"""PyTorch HiFTNet model - Neural vocoder with source filter and ISTFTNet."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
from torch import pow, sin
from torch.distributions.uniform import Uniform
from torch.nn import Conv1d, ConvTranspose1d, Parameter
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_hiftnet import HiFTNetConfig


logger = logging.get_logger(__name__)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding to maintain sequence length."""
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function.

    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input

    Parameters:
        - alpha: trainable parameter

    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = False
    ):
        """
        Initialization.

        Args:
            in_features: shape of the input
            alpha: trainable parameter (default 1.0)
                   alpha is initialized to 1 by default, higher values = higher-frequency.
                   alpha will be trained along with the rest of your model.
            alpha_trainable: whether alpha is trainable
            alpha_logscale: whether to use log scale for alpha
        """
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
        return x


class ResBlock(nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""

    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: tuple[int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )
        # Note: Weights will be initialized by _init_weights in post_init()
        self.activations1 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            remove_weight_norm(self.convs2[idx])


class SineGen(nn.Module):
    """
    Definition of sine generator for neural source filter.

    SineGen(samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0)

    Args:
        samp_rate: sampling rate in Hz
        harmonic_num: number of harmonic overtones (default 0)
        sine_amp: amplitude of sine-waveform (default 0.1)
        noise_std: std of Gaussian noise (default 0.003)
        voiced_threshold: F0 threshold for U/V classification (default 0)
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        """Generate unvoiced/voiced signal."""
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f0: [B, 1, sample_len], Hz

        Returns:
            sine_waves: [B, harmonic_num+1, sample_len]
            uv: [B, 1, sample_len]
            noise: [B, harmonic_num+1, sample_len]
        """
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1))).to(f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i : i + 1, :] = f0 * (i + 1) / self.sampling_rate

        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-np.pi, high=np.pi)
        phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1)).to(F_mat.device)
        phase_vec[:, 0, :] = 0

        # generate sine waveforms
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        #        for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    """
    Source Module for harmonic-plus-noise neural source filter.

    Args:
        sampling_rate: sampling rate in Hz
        upsample_scale: total upsampling scale factor
        harmonic_num: number of harmonics above F0 (default: 0)
        sine_amp: amplitude of sine source signal (default: 0.1)
        add_noise_std: std of additive Gaussian noise (default: 0.003)
        voiced_threshold: threshold to set U/V given F0 (default: 0)
    """

    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold)

        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: F0 sampled [B, T, 1]

        Returns:
            sine_merge: [B, T, 1]
            noise: [B, T, 1]
            uv: [B, T, 1]
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x.transpose(1, 2))
            sine_wavs = sine_wavs.transpose(1, 2)
            uv = uv.transpose(1, 2)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class ConvRNNF0Predictor(nn.Module):
    """
    Convolutional RNN-based F0 predictor.

    Args:
        num_class: number of output classes (default 1 for F0 regression)
        in_channels: input feature dimension
        cond_channels: conditional feature dimension
    """

    def __init__(
        self,
        num_class: int = 1,
        in_channels: int = 80,
        cond_channels: int = 512,
    ):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] mel spectrogram

        Returns:
            f0: [B, T] predicted F0
        """
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class HiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet.

    This is the core HiFTNet vocoder model that combines a neural source filter
    with an inverse STFT network for high-quality speech synthesis.

    Reference: https://arxiv.org/abs/2309.09493
    """

    def __init__(self, config: HiFTNetConfig):
        super().__init__()

        self.config = config
        self.out_channels = 1
        self.nb_harmonics = config.nb_harmonics
        self.sampling_rate = config.sampling_rate
        self.lrelu_slope = config.lrelu_slope
        self.audio_limit = config.audio_limit

        # ISTFT parameters
        self.istft_params = {"n_fft": config.istft_n_fft, "hop_len": config.istft_hop_len}

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # Neural source filter
        upsample_scale = int(np.prod(config.upsample_rates) * config.istft_hop_len)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=config.sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_num=config.nb_harmonics,
            sine_amp=config.nsf_alpha,
            add_noise_std=config.nsf_sigma,
            voiced_threshold=config.nsf_voiced_threshold,
        )
        self.f0_upsamp = nn.Upsample(scale_factor=upsample_scale)

        # F0 predictor
        self.f0_predictor = ConvRNNF0Predictor(
            num_class=1,
            in_channels=config.f0_predictor_in_channels,
            cond_channels=config.f0_predictor_cond_channels,
        )

        # Pre-convolution
        self.conv_pre = weight_norm(Conv1d(config.in_channels, config.base_channels, 7, 1, padding=3))

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        config.base_channels // (2**i),
                        config.base_channels // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Source downsampling and residual blocks
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + config.upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)

        for i, (u, k, d) in enumerate(
            zip(
                downsample_cum_rates[::-1],
                config.source_resblock_kernel_sizes,
                config.source_resblock_dilation_sizes,
            )
        ):
            if u == 1:
                self.source_downs.append(
                    Conv1d(self.istft_params["n_fft"] + 2, config.base_channels // (2 ** (i + 1)), 1, 1)
                )
            else:
                self.source_downs.append(
                    Conv1d(
                        self.istft_params["n_fft"] + 2,
                        config.base_channels // (2 ** (i + 1)),
                        u * 2,
                        u,
                        padding=(u // 2),
                    )
                )

            self.source_resblocks.append(ResBlock(config.base_channels // (2 ** (i + 1)), k, d))

        # Main residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.base_channels // (2 ** (i + 1))
            for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        # Post-convolution
        self.conv_post = weight_norm(Conv1d(ch, self.istft_params["n_fft"] + 2, 7, 1, padding=3))
        # Note: Weights will be initialized by _init_weights in post_init()

        # Reflection padding and STFT window
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        stft_window = torch.from_numpy(get_window("hann", self.istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.register_buffer("stft_window", stft_window)

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for l in self.source_downs:
            remove_weight_norm(l)
        for l in self.source_resblocks:
            l.remove_weight_norm()
        # Remove weight norm from F0 predictor
        for module in self.f0_predictor.condnet:
            if hasattr(module, "weight"):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    pass

    def _stft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT."""
        spec = torch.stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(x.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute inverse STFT."""
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(magnitude.device),
        )
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor = None) -> torch.Tensor:
        """
        Decode mel spectrogram to waveform.

        Args:
            x: [B, C, T] mel spectrogram
            s: [B, 1, T_audio] source signal (optional, defaults to zeros)

        Returns:
            waveform: [B, T_audio]
        """
        if s is None:
            s = torch.zeros(x.size(0), 1, 0).to(x.device)

        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # Fusion with source
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])

        x = self._istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    def forward(self, speech_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            speech_feat: [B, T, C] mel spectrogram (will be transposed internally)

        Returns:
            generated_speech: [B, T_audio]
            f0: [B, T]
        """
        speech_feat = speech_feat.transpose(1, 2)  # [B, C, T]

        # Predict F0
        f0 = self.f0_predictor(speech_feat)

        # Generate source signal
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_upsampled, 1]
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)  # [B, 1, T_upsampled]

        # Generate waveform
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, f0

    @torch.inference_mode()
    def inference(
        self, speech_feat: torch.Tensor, cache_source: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference method with source caching support.

        Args:
            speech_feat: [B, C, T] mel spectrogram
            cache_source: [B, 1, T_cache] cached source signal (optional)

        Returns:
            generated_speech: [B, T_audio]
            s: [B, 1, T_audio] source signal for caching
        """
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(speech_feat.device)

        # Predict F0
        f0 = self.f0_predictor(speech_feat)

        # Generate source signal
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # [B, T_upsampled, 1]
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)  # [B, 1, T_upsampled]

        # Use cache_source to avoid glitch
        if cache_source.shape[2] != 0:
            s[:, :, : cache_source.shape[2]] = cache_source

        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s


class HiFTNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HiFTNetConfig
    base_model_prefix = "hiftnet"
    main_input_name = "speech_feat"

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()


class HiFTNetModel(HiFTNetPreTrainedModel):
    """
    HiFTNet vocoder model for converting mel spectrograms to waveforms.

    This model integrates the HiFTNet generator with neural source filter for high-quality
    speech synthesis. It's designed for inference-only use.

    Args:
        config (`HiFTNetConfig`): Model configuration class with all the parameters of the model.
    """

    def __init__(self, config: HiFTNetConfig):
        super().__init__(config)
        self.config = config
        self.hiftnet = HiFTGenerator(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        speech_feat: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Convert mel spectrogram to waveform.

        Args:
            speech_feat (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`):
                Mel spectrogram input.
            return_dict (`bool`, *optional*):
                Whether or not to return a dict. If `False`, returns a tuple.

        Returns:
            `torch.Tensor` of shape `(batch_size, audio_length)`:
                Generated waveform.
        """
        waveform, f0 = self.hiftnet(speech_feat)
        return waveform

    @torch.inference_mode()
    def generate(
        self,
        speech_feat: torch.Tensor,
        cache_source: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate waveform from mel spectrogram with caching support.

        Args:
            speech_feat (`torch.Tensor` of shape `(batch_size, feature_dim, sequence_length)`):
                Mel spectrogram input (already transposed).
            cache_source (`torch.Tensor`, *optional*):
                Cached source signal from previous generation for seamless streaming.

        Returns:
            tuple of `torch.Tensor`:
                - waveform of shape `(batch_size, audio_length)`
                - source signal of shape `(batch_size, 1, audio_length)` for caching
        """
        return self.hiftnet.inference(speech_feat, cache_source)


__all__ = ["HiFTNetModel", "HiFTNetPreTrainedModel", "HiFTNetConfig"]
