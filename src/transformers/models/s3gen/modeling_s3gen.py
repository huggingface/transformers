# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch S3Gen model."""

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window
from torch.distributions.uniform import Uniform
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, auto_docstring
from ..dac.modeling_dac import Snake1d
from ..s3tokenizer.configuration_s3tokenizer import S3TokenizerConfig
from ..s3tokenizer.feature_extraction_s3tokenizer import S3TokenizerFeatureExtractor
from ..s3tokenizer.modeling_s3tokenizer import S3TokenizerModel
from .configuration_s3gen import HiFTNetConfig, S3GenConfig


logger = logging.getLogger(__name__)

# Global state for mel spectrogram computation
mel_basis = {}
hann_window = {}


# Utility functions
def pad_list(xs, pad_value):
    """Perform padding for the list of tensors."""
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad


def extract_feature(audio):
    """Extract fbank features for speaker encoder."""
    features = []
    feature_lengths = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_lengths.append(feature.shape[0])
    features_padded = pad_list(features, pad_value=0)
    return features_padded, feature_lengths


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part."""
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def mel_spectrogram(
    y, n_fft=1024, num_mels=80, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False
):
    """Extract mel spectrogram from audio."""
    if isinstance(y, np.ndarray):
        y = torch.tensor(y).float()
    if len(y.shape) == 1:
        y = y[None,]

    global mel_basis, hann_window
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec


# CAMPPlus Speaker Encoder Components
class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.ModuleList()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut.append(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1), bias=False)
            )
            self.shortcut.append(nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        shortcut_out = x
        for layer in self.shortcut:
            shortcut_out = layer(shortcut_out)
        out += shortcut_out

        out = F.relu(out)
        return out


class FCM(nn.Module):
    """Frequency Channel Masking module."""

    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)
        out = F.relu(self.bn2(self.conv2(out)))
        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


def get_nonlinear(config_str, channels):
    """Create non-linear activation module."""
    nonlinear = nn.ModuleDict()
    if "batchnorm" in config_str:
        affine = "batchnorm_" not in config_str
        nonlinear["batchnorm"] = nn.BatchNorm1d(channels, affine=affine)
    if "relu" in config_str:
        nonlinear["relu"] = nn.ReLU(inplace=True)
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    """Compute mean and standard deviation statistics."""
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    """Time Delay Neural Network layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, f"Expect equal paddings, but got even kernel size ({kernel_size})"
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        for layer in self.nonlinear.values():
            x = layer(x)
        return x


class CAMLayer(nn.Module):
    """Context-Aware Masking layer."""

    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    """Dense TDNN layer with CAM."""

    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"Expect equal paddings, but got even kernel size ({kernel_size})"
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def bn_function(self, x):
        for layer in self.nonlinear1.values():
            x = layer(x)
        return self.linear1(x)

    def forward(self, x):
        x = self.bn_function(x)
        for layer in self.nonlinear2.values():
            x = layer(x)
        x = self.cam_layer(x)
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    """Dense TDNN block with CAM."""

    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module(f"tdnnd{i + 1}", layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    """Transition layer between blocks."""

    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for layer in self.nonlinear.values():
            x = layer(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    """Dense layer."""

    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        for layer in self.nonlinear.values():
            x = layer(x)
        return x


# ============================================================================
# HiFTNet Vocoder Components
# ============================================================================


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding to maintain sequence length."""
    return int((kernel_size * dilation - dilation) / 2)


class Snake(Snake1d):
    """
    Implementation of a sine-based periodic activation function.
    Inherits from Snake1d for modularity.
    """

    def __init__(
        self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = False
    ):
        super().__init__(in_features)
        # Re-initialize to match Chatterbox original shapes (C,) instead of (1, C, 1)
        # to ensure checkpoint compatibility.
        self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We override forward to handle the (C,) -> (1, C, 1) unsqueeze
        # while keeping the core logic identical to Snake1d.
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


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
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation,
                    padding=get_padding(kernel_size, dilation),
                )
            )
            self.convs2.append(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
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

    def apply_weight_norm(self):
        for idx in range(len(self.convs1)):
            weight_norm(self.convs1[idx])
            weight_norm(self.convs2[idx])

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
        # Using numeric keys to match state_dict
        self.condnet = nn.ModuleList(
            [
                nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1),
                nn.ELU(),
            ]
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] mel spectrogram

        Returns:
            f0: [B, T] predicted F0
        """
        for layer in self.condnet:
            x = layer(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))

    def apply_weight_norm(self):
        for module in self.condnet:
            if isinstance(module, nn.Conv1d):
                weight_norm(module)

    def remove_weight_norm(self):
        for module in self.condnet:
            if isinstance(module, nn.Conv1d):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    pass


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
        self.conv_pre = Conv1d(config.in_channels, config.base_channels, 7, 1, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    config.base_channels // (2**i),
                    config.base_channels // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
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
        self.conv_post = Conv1d(ch, self.istft_params["n_fft"] + 2, 7, 1, padding=3)
        # Note: Weights will be initialized by _init_weights in post_init()

        # Reflection padding and STFT window
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        stft_window = torch.from_numpy(get_window("hann", self.istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.register_buffer("stft_window", stft_window)

        # Apply weight normalization to match checkpoint
        self.apply_weight_norm()

    def apply_weight_norm(self):
        """Apply weight normalization to all relevant layers."""
        logger.info("Applying weight norm...")
        for l in self.ups:
            weight_norm(l)
        for l in self.resblocks:
            l.apply_weight_norm()
        weight_norm(self.conv_pre)
        weight_norm(self.conv_post)
        for l in self.source_resblocks:
            l.apply_weight_norm()
        # Apply weight norm to F0 predictor
        self.f0_predictor.apply_weight_norm()

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        logger.info("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for l in self.source_resblocks:
            l.remove_weight_norm()
        # Remove weight norm from F0 predictor
        self.f0_predictor.remove_weight_norm()

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
        else:
            # Smoothly fade-in the source when starting from scratch to reduce onset transients.
            n_fade = min(int(self.sampling_rate // 40), s.size(2))  # ~25ms at 24kHz
            if n_fade > 1:
                fade = (torch.cos(torch.linspace(torch.pi, 0, n_fade, device=s.device, dtype=s.dtype)) + 1) / 2
                s[:, :, :n_fade] *= fade

        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s


# ============================================================================
# Transformer Components (replacing diffusers dependency)
# ============================================================================


# Decorator replacement
def maybe_allow_in_graph(cls):
    """Decorator to allow class in graph (simplified version)."""
    return cls


# Linear layer replacement
class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used as a drop-in replacement for `torch.nn.Linear`.
    Simplified version without LoRA support.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)


# Activation functions
class GELU(nn.Module):
    """GELU activation function."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        if self.approximate == "tanh":
            return F.gelu(hidden_states, approximate="tanh")
        return F.gelu(hidden_states)


class GEGLU(nn.Module):
    """GEGLU activation function."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class ApproximateGELU(nn.Module):
    """Approximate GEGLU activation function."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate, approximate="tanh")


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components.

    Shape:
        - Input: (B, C, T) or (B, T, C)
        - Output: same shape as input

    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    """

    def __init__(self, in_features, out_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = out_features if isinstance(out_features, list) else [out_features]
        self.proj = LoRACompatibleLinear(in_features, out_features)

        # initialize alpha and beta
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


# Normalization layers
class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization."""

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim * 2)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization Zero."""

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim * 6)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 6)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, timestep, class_labels=None, hidden_dtype=None):
        emb = self.linear(self.silu(self.emb(timestep)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa)[:, None, :] + shift_msa[:, None, :]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# Attention processor
class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        out_bias: bool = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self.upcast_attention = upcast_attention

        self.to_q = LoRACompatibleLinear(query_dim, inner_dim, bias=bias)
        self.to_k = LoRACompatibleLinear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = LoRACompatibleLinear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(LoRACompatibleLinear(inner_dim, query_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # Compute attention
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        if self.upcast_attention:
            attention_probs = attention_probs.to(value.dtype)

        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)

        # Linear projection
        for layer in self.to_out:
            hidden_states = layer(hidden_states)

        return hidden_states


# Feed-forward layer
class FeedForward(nn.Module):
    """
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=True)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=True)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=True)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=True)
        elif activation_fn == "snakebeta":
            act_fn = SnakeBeta(dim, inner_dim)
        else:
            act_fn = GEGLU(dim, inner_dim, bias=True)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=True))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# BasicTransformerBlock
@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Optional[dict] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 2. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


# ============================================================================
# S3Gen Components
# ============================================================================


class CAMPPlus(nn.Module):
    """CAMPPlus speaker encoder."""

    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
        **kwargs,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = nn.ModuleDict()
        self.xvector["tdnn"] = TDNNLayer(
            channels, init_channels, 5, stride=2, dilation=1, padding=-1, config_str=config_str
        )

        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector[f"block{i + 1}"] = block
            channels = channels + num_layers * growth_rate
            self.xvector[f"transit{i + 1}"] = TransitLayer(channels, channels // 2, bias=False, config_str=config_str)
            channels //= 2

        self.xvector["out_nonlinear"] = get_nonlinear(config_str, channels)

        if self.output_level == "segment":
            self.xvector["stats"] = StatsPool()
            self.xvector["dense"] = DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
        else:
            assert self.output_level == "frame", "`output_level` should be set to 'segment' or 'frame'."

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)

        # Manual forward through ModuleDict in correct order
        for key in [
            "tdnn",
            "block1",
            "transit1",
            "block2",
            "transit2",
            "block3",
            "transit3",
            "out_nonlinear",
            "stats",
            "dense",
        ]:
            if key in self.xvector:
                module = self.xvector[key]
                if isinstance(module, nn.ModuleDict):
                    for sublayer in module.values():
                        x = sublayer(x)
                else:
                    x = module(x)

        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x

    def inference(self, audio_list):
        """Run inference on audio."""
        speech, speech_lengths = extract_feature(audio_list)
        results = self.forward(speech.to(torch.float32))
        return results


# Conformer Encoder Components
class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000, reverse: bool = False):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len
        # Lazily created on first forward to support meta-device initialization.
        self.pe = None

    def _build_pe(self, device: torch.device):
        pe = torch.zeros(self.max_len, self.d_model, device=device, dtype=torch.float32)
        position = torch.arange(0, self.max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Keep in float32 (as originally) for numerical stability; do not follow `x.dtype` (often fp16/bf16).
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pe is None or self.pe.is_meta or self.pe.device != x.device:
            self._build_pe(device=x.device)
        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int, apply_dropout: bool = True) -> torch.Tensor:
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        else:
            pos_emb = self.pe[:, :size]
        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pe is None or self.pe.is_meta or self.pe.device != x.device:
            self._build_pe(device=x.device)
        x = x * self.xscale
        pos_emb = self.position_encoding(offset, x.size(1), False)
        return self.dropout(x), self.dropout(pos_emb)


class EspnetRelPositionalEncoding(nn.Module):
    """ESPnet-style relative positional encoding."""

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None

    def extend_pe(self, x: torch.Tensor):
        if self.pe is not None:
            if self.pe.is_meta:
                self.pe = None
            elif self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model, device=x.device, dtype=torch.float32)
        pe_negative = torch.zeros(x.size(1), self.d_model, device=x.device, dtype=torch.float32)
        position = torch.arange(0, x.size(1), dtype=torch.float32, device=x.device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        pos_emb = self.pe[:, self.pe.size(1) // 2 - size + 1 : self.pe.size(1) // 2 + size]
        return pos_emb


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        n_batch = value.size(0)
        if mask.size(2) > 0:
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:, :, :, : scores.size(-1)]
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.forward_qkv(query, key, value)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding."""

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u.to(q.device)).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v.to(q.device)).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer."""

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class LinearNoSubsampling(nn.Module):
    """Linear transform without subsampling."""

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: nn.Module):
        super().__init__()
        self.out = nn.ModuleList(
            [
                nn.Linear(idim, odim),
                nn.LayerNorm(odim, eps=1e-5),
                nn.Dropout(dropout_rate),
            ]
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, offset: int = 0):
        for layer in self.out:
            x = layer(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module."""

    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)
            self.norm_final = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ):
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class Upsample1D(nn.Module):
    """1D upsampling layer."""

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    """Pre-lookahead layer for streaming."""

    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (2, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs
        return outputs


class UpsampleConformerEncoder(nn.Module):
    """Upsample Conformer Encoder."""

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self._output_size = output_size
        self.global_cmvn = global_cmvn

        if pos_enc_layer_type == "rel_pos_espnet":
            pos_enc_class = EspnetRelPositionalEncoding(output_size, positional_dropout_rate)
        else:
            pos_enc_class = RelPositionalEncoding(output_size, positional_dropout_rate)

        self.embed = LinearNoSubsampling(input_size, output_size, dropout_rate, pos_enc_class)
        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing

        activation = nn.SiLU() if activation_type == "swish" else nn.ReLU()

        encoder_selfattn_layer_args = (attention_heads, output_size, attention_dropout_rate, key_bias)
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)

        self.pre_lookahead_layer = PreLookaheadLayer(channels=512, pre_lookahead_len=3)
        self.encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args)
                    if selfattention_layer_type == "rel_selfattn"
                    else MultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None,
                    None,  # No CNN module
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(channels=512, out_channels=512, stride=2)

        if pos_enc_layer_type == "rel_pos_espnet":
            up_pos_enc_class = EspnetRelPositionalEncoding(output_size, positional_dropout_rate)
        else:
            up_pos_enc_class = RelPositionalEncoding(output_size, positional_dropout_rate)

        self.up_embed = LinearNoSubsampling(input_size, output_size, dropout_rate, up_pos_enc_class)
        self.up_encoders = nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args)
                    if selfattention_layer_type == "rel_selfattn"
                    else MultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None,
                    None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(4)
            ]
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self, xs: torch.Tensor, xs_lens: torch.Tensor, decoding_chunk_size: int = 0, num_decoding_left_chunks: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = masks

        xs = self.pre_lookahead_layer(xs)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        mask_pad = masks
        chunk_masks = masks
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_layers(
        self, xs: torch.Tensor, chunk_masks: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(
        self, xs: torch.Tensor, chunk_masks: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs


# CFM Decoder Components
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding layer."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.ReLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        assert stride == 1
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super().forward(x)
        return x


class Transpose(nn.Module):
    """Transpose module."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        return torch.transpose(x, self.dim0, self.dim1)


class CausalBlock1D(nn.Module):
    """Causal 1D block."""

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.block = nn.ModuleList(
            [
                CausalConv1d(dim, dim_out, 3),
                Transpose(1, 2),
                nn.LayerNorm(dim_out),
                Transpose(1, 2),
                nn.Mish(),
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x * mask
        for layer in self.block:
            x = layer(x)
        return x * mask


class CausalResnetBlock1D(nn.Module):
    """Causal ResNet block."""

    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.ModuleList([nn.Mish(), nn.Linear(time_emb_dim, dim_out)])
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)
        self.res_conv = CausalConv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)

        mlp_out = time_emb
        for layer in self.mlp:
            mlp_out = layer(mlp_out)
        h += mlp_out.unsqueeze(-1)

        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


# ConditionalDecoder implementation for S3Gen


class ConditionalDecoder(nn.Module):
    """Conditional decoder for CFM. Simplified U-Net architecture."""

    def __init__(
        self,
        in_channels=320,
        out_channels=80,
        causal=True,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(in_channels=in_channels, time_embed_dim=time_embed_dim, act_fn="silu")
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.static_chunk_size = 0

        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            # Create transformer blocks
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                        attention_bias=False,
                        only_cross_attention=False,
                        upcast_attention=False,
                    )
                    for _ in range(n_blocks)
                ]
            )

            downsample = (
                CausalConv1d(output_channel, output_channel, 3)
                if is_last
                else nn.Conv1d(output_channel, output_channel // 2, 3, stride=2, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=channels[-1], time_emb_dim=time_embed_dim)

            # Create transformer blocks
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=channels[-1],
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                        attention_bias=False,
                        only_cross_attention=False,
                        upcast_attention=False,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            # Create transformer blocks
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                        attention_bias=False,
                        only_cross_attention=False,
                        upcast_attention=False,
                    )
                    for _ in range(n_blocks)
                ]
            )

            upsample = (
                CausalConv1d(output_channel, output_channel, 3)
                if is_last
                else nn.ConvTranspose1d(output_channel, output_channel, 4, 2, 1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = CausalBlock1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        # Concatenate inputs
        x = torch.cat([x, mu], dim=1)
        if spks is not None:
            spks = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, spks], dim=1)
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            # Transpose for transformer blocks: (B, C, T) -> (B, T, C)
            for transformer_block in transformer_blocks:
                x = transformer_block(x.transpose(1, 2)).transpose(1, 2)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2] if x.shape[-1] < mask_down.shape[-1] else mask_down)

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            # Transpose for transformer blocks: (B, C, T) -> (B, T, C)
            for transformer_block in transformer_blocks:
                x = transformer_block(x.transpose(1, 2)).transpose(1, 2)

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = torch.cat([x[:, :, : skip.shape[-1]], skip], dim=1)
            x = resnet(x, mask_up, t)
            # Transpose for transformer blocks: (B, C, T) -> (B, T, C)
            for transformer_block in transformer_blocks:
                x = transformer_block(x.transpose(1, 2)).transpose(1, 2)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


class CausalConditionalCFM(nn.Module):
    """Causal Conditional Flow Matching."""

    def __init__(self, in_channels=240, spk_emb_dim=80, estimator=None):
        super().__init__()
        self.n_feats = in_channels
        self.spk_emb_dim = spk_emb_dim
        self.solver = "euler"
        self.sigma_min = 1e-6
        self.t_scheduler = "cosine"
        self.training_cfg_rate = 0.2
        self.inference_cfg_rate = 0.7
        self.estimator = estimator
        # Lazily materialized on first forward to support meta-device initialization.
        self.rand_noise = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        needed_len = mu.size(2)
        if (
            self.rand_noise is None
            or self.rand_noise.is_meta
            or self.rand_noise.device != mu.device
            or self.rand_noise.dtype != mu.dtype
            or self.rand_noise.size(2) < needed_len
        ):
            # Keep a small cache so repeated calls don't reallocate for slightly different lengths.
            cache_len = max(needed_len, 50 * 300)
            self.rand_noise = torch.randn(1, 80, cache_len, device=mu.device, dtype=mu.dtype)
        z = self.rand_noise[:, :, :needed_len] * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), None

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x.float()


class CausalMaskedDiffWithXvec(nn.Module):
    """Causal masked diffusion with speaker embedding."""

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
        self.fp16 = False

    @torch.inference_mode()
    def inference(
        self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, finalize
    ):
        if self.fp16:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        assert token.shape[0] == 1
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0, max=self.input_embedding.num_embeddings - 1)) * mask

        h, h_lengths = self.encoder(token, token_len)
        if not finalize:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(), mask=mask.unsqueeze(1), spks=embedding, cond=conds, n_timesteps=10
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None


# Main Model
class S3GenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = S3GenConfig
    base_model_prefix = "s3gen"
    main_input_name = "speech_tokens"
    supports_gradient_checkpointing = False


S3GEN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`S3GenConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

S3GEN_INPUTS_DOCSTRING = r"""
    Args:
        speech_tokens (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of speech tokens from S3 tokenizer.
        ref_wav (`torch.FloatTensor` of shape `(batch_size, audio_length)`, *optional*):
            Reference audio waveform for speaker conditioning.
        ref_sr (`int`, *optional*):
            Sample rate of the reference audio.
        ref_dict (`dict`, *optional*):
            Pre-computed reference embeddings dict (alternative to ref_wav).
        finalize (`bool`, *optional*, defaults to `True`):
            Whether this is the final chunk (for streaming).
"""


@auto_docstring
class S3GenModel(S3GenPreTrainedModel):
    """
    The S3Gen Model for converting speech tokens to mel spectrograms and waveforms.
    """

    def __init__(self, config: S3GenConfig):
        super().__init__(config)
        self.config = config

        # S3 Tokenizer for reference audio (initialized locally, weights loaded from checkpoint)
        tokenizer_config = S3TokenizerConfig()
        self.tokenizer = S3TokenizerModel(tokenizer_config, name="speech_tokenizer_v2_25hz")
        self.tokenizer_feature_extractor = S3TokenizerFeatureExtractor()

        # Speaker encoder
        self.speaker_encoder = CAMPPlus(
            feat_dim=config.speaker_feat_dim,
            embedding_size=config.speaker_embed_dim,
        )

        # Conformer encoder
        encoder = UpsampleConformerEncoder(
            output_size=config.encoder_output_size,
            attention_heads=config.encoder_attention_heads,
            linear_units=config.encoder_linear_units,
            num_blocks=config.encoder_num_blocks,
            dropout_rate=config.encoder_dropout_rate,
            positional_dropout_rate=config.encoder_dropout_rate,
            attention_dropout_rate=config.encoder_dropout_rate,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=config.token_embed_dim,
            use_cnn_module=False,
            macaron_style=False,
        )

        # CFM decoder
        estimator = ConditionalDecoder(
            in_channels=config.decoder_in_channels,
            out_channels=config.decoder_out_channels,
            causal=True,
            channels=config.decoder_channels,
            dropout=0.0,
            attention_head_dim=config.decoder_attention_head_dim,
            n_blocks=config.decoder_n_blocks,
            num_mid_blocks=config.decoder_num_mid_blocks,
            num_heads=config.decoder_num_heads,
            act_fn=config.decoder_act_fn,
        )
        decoder = CausalConditionalCFM(
            in_channels=config.decoder_in_channels,
            spk_emb_dim=config.decoder_out_channels,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=decoder,
            input_size=config.token_embed_dim,
            output_size=config.mel_bins,
            spk_embed_dim=config.speaker_embed_dim,
            vocab_size=config.vocab_size,
            input_frame_rate=config.input_frame_rate,
            token_mel_ratio=config.token_mel_ratio,
            pre_lookahead_len=config.pre_lookahead_len,
        )

        # CFM parameters stored in config for future use
        _ = (config.cfm_sigma_min, config.cfm_solver, config.cfm_t_scheduler, config.cfm_inference_cfg_rate)

        # HiFTNet vocoder
        hiftnet_config = HiFTNetConfig(
            sampling_rate=config.sampling_rate,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )
        self.mel2wav = HiFTGenerator(hiftnet_config)

        # Trim/fade buffer for reducing startup artifacts (glitches/clicks) from the vocoder.
        # Use a short fade-in (no trimming) to avoid reintroducing discontinuities.
        n_trim = 0  # ~10ms at 24kHz
        # Smooth fade-in from 0 -> 1.
        trim_fade = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False)

        self.post_init()

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_ref(self, ref_wav: torch.Tensor, ref_sr: int, device="auto"):
        """Extract reference embeddings from audio."""
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)

        # Resample to 24kHz for mel extraction
        ref_wav_24 = ref_wav
        if ref_sr != self.config.sampling_rate:
            import torchaudio

            resampler = torchaudio.transforms.Resample(ref_sr, self.config.sampling_rate).to(device)
            ref_wav_24 = resampler(ref_wav)

        ref_mels_24 = (
            mel_spectrogram(
                ref_wav_24,
                n_fft=self.config.n_fft,
                num_mels=self.config.mel_bins,
                sampling_rate=self.config.sampling_rate,
                hop_size=self.config.hop_length,
                win_size=self.config.win_size,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
            )
            .transpose(1, 2)
            .to(device)
        )

        # Resample to 16kHz for speaker encoder + tokenizer
        import torchaudio

        resampler_16 = torchaudio.transforms.Resample(ref_sr, 16000).to(device)
        ref_wav_16 = resampler_16(ref_wav).to(device)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16)

        # Tokenize reference (use feature extractor first)
        features = self.tokenizer_feature_extractor(
            ref_wav_16.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).to(device)
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(
            input_features=features.input_features, attention_mask=features.attention_mask, return_dict=False
        )

        # Ensure mel_len = 2 * token_len
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            ref_speech_tokens = ref_speech_tokens[:, : ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return {
            "prompt_token": ref_speech_tokens.to(device),
            "prompt_token_len": ref_speech_token_lens,
            "prompt_feat": ref_mels_24,
            "prompt_feat_len": None,
            "embedding": ref_x_vector,
        }

    @add_start_docstrings_to_model_forward(S3GEN_INPUTS_DOCSTRING)
    def forward(self, speech_tokens, ref_wav=None, ref_sr=None, ref_dict=None, finalize=False, **kwargs):
        """Generate mel spectrograms from tokens."""
        assert (ref_wav is None) ^ (ref_dict is None), "Must provide exactly one of ref_wav or ref_dict"

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(self.device)

        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        speech_token_lens = torch.LongTensor([speech_tokens.size(1)]).to(self.device)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            **ref_dict,
        )
        return output_mels

    @torch.inference_mode()
    def inference(self, speech_tokens, ref_wav=None, ref_sr=None, ref_dict=None, cache_source=None, finalize=True):
        """
        End-to-end inference: tokens → waveform.

        Args:
            speech_tokens: Speech token sequence
            ref_wav: Reference audio waveform (mutex with ref_dict)
            ref_sr: Reference audio sample rate (required with ref_wav)
            ref_dict: Pre-computed reference embeddings (mutex with ref_wav)
            cache_source: Cached source for streaming
            finalize: Whether to finalize generation
        """
        output_mels = self.forward(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)

        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(self.device)

        output_wavs, output_sources = self.mel2wav.inference(speech_feat=output_mels, cache_source=cache_source)

        # Reduce spillover artifacts at the start (non-inplace to avoid InferenceMode error)
        trim_fade = self.trim_fade.to(output_wavs.device)
        output_wavs = output_wavs.clone()  # Clone to allow inplace operation
        n_fade = len(trim_fade)
        if output_wavs.size(1) > n_fade:
            output_wavs[:, :n_fade] *= trim_fade

        return output_wavs, output_sources

    @torch.inference_mode()
    def generate(self, speech_tokens, ref_wav, ref_sr, cache_source=None, finalize=True):
        """
        Generate audio from speech tokens.

        This is an alias for the inference method, provided for consistency with
        HuggingFace generation API conventions.

        Args:
            speech_tokens (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Speech tokens to convert to audio.
            ref_wav (`torch.FloatTensor` of shape `(batch_size, audio_length)` or `(audio_length,)`):
                Reference audio for speaker embedding extraction.
            ref_sr (`int`):
                Sample rate of the reference audio.
            cache_source (`torch.FloatTensor`, *optional*):
                Cached source for streaming generation. Defaults to None.
            finalize (`bool`, *optional*, defaults to `True`):
                Whether to finalize the generation (used for streaming).

        Returns:
            `torch.FloatTensor`: Generated waveform of shape `(batch_size, audio_length)`.
        """
        output_wavs, _ = self.inference(
            speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, cache_source=cache_source, finalize=finalize
        )
        return output_wavs


__all__ = ["S3GenPreTrainedModel", "S3GenModel"]
