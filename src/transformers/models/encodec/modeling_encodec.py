# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and affiliates, and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch EnCodec model."""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn


# TODO: their stuff
from dataclasses import dataclass, field
import einops
from einops import rearrange, repeat


from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_encodec import EncodecConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "EncodecConfig"


ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Matthijs/encodec_24khz",
    "Matthijs/encodec_48khz",
    # See all EnCodec models at https://huggingface.co/models?filter=encodec
]


EncodedFrame = Tuple[torch.Tensor, Optional[torch.Tensor]]

_CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                 'time_layer_norm', 'layer_norm', 'time_group_norm'])


def _linear_overlap_add(frames: List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def _apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    if norm not in _CONV_NORMALIZATIONS:
        raise ValueError(f"Invalid normalization option: {norm}")
    if norm == 'weight_norm':
        return nn.utils.weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def _get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    if norm not in _CONV_NORMALIZATIONS:
        raise ValueError(f"Invalid normalization option: {norm}")
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def _get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def _pad1d(x: torch.Tensor, paddings: Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = nn.functional.pad(x, (0, extra_pad))
        padded = nn.functional.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return nn.functional.pad(x, paddings, mode, value)


def _unpad1d(x: torch.Tensor, paddings: Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


def _uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def _sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def _kmeans(samples, num_clusters: int, num_iters: int = 10):
    dim, dtype = samples.shape[-1], samples.dtype

    means = _sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = einops.rearrange(x, 'b ... t -> b t ...')
        x = super().forward(x)
        x = einops.rearrange(x, 'b t ... -> b ... t')
        return


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: Dict[str, Any] = {}, **kwargs):
        super().__init__()
        self.conv = _apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = _get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                 norm_kwargs: Dict[str, Any] = {}, **kwargs):
        super().__init__()
        self.convtr = _apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = _get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, causal: bool = False,
                 norm: str = 'none', norm_kwargs: Dict[str, Any] = {},
                 pad_mode: str = 'reflect'):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logger.warning('SConv1d has been initialized with stride > 1 and dilation > 1'
                          f' (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, groups=groups, bias=bias, causal=causal,
                               norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = _get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = _pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = _pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, causal: bool = False,
                 norm: str = 'none', trim_right_ratio: float = 1.,
                 norm_kwargs: Dict[str, Any] = {}):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                          causal=causal, norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

    def forward(self, x):
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = _unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = _unpad1d(y, (padding_left, padding_right))
        return y


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    Args:
        dim (int): Dimension of the input/output
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3)
        true_skip (bool): Whether to use true skip connection or a simple convolution as the skip connection.
    """
    def __init__(self, dim: int, kernel_sizes: List[int] = [3, 1], dilations: List[int] = [1, 1],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: Dict[str, Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                    causal=causal, pad_mode=pad_mode)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder."""
    def __init__(self, config: EncodecConfig):
        super().__init__()
        self.channels = config.audio_channels
        self.dimension = config.dimension
        self.n_filters = config.num_filters
        self.ratios = list(reversed(config.ratios))
        self.n_residual_layers = config.num_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, config.activation)
        mult = 1
        model = [
            SConv1d(
                config.audio_channels,
                mult * config.num_filters,
                config.kernel_size,
                norm=config.norm,
                norm_kwargs=config.norm_params,
                causal=config.causal,
                pad_mode=config.pad_mode,
            )
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * config.num_filters,
                        kernel_sizes=[config.residual_kernel_size, 1],
                        dilations=[config.dilation_base ** j, 1],
                        norm=config.norm,
                        norm_params=config.norm_params,
                        activation=config.activation,
                        activation_params=config.activation_params,
                        causal=config.causal,
                        pad_mode=config.pad_mode,
                        compress=config.compress,
                        true_skip=config.true_skip,
                    )
                ]
            # Add downsampling layers
            model += [
                act(**config.activation_params),
                SConv1d(
                    mult * config.num_filters,
                    mult * config.num_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=config.norm,
                    norm_kwargs=config.norm_params,
                    causal=config.causal,
                    pad_mode=config.pad_mode,
                )
            ]
            mult *= 2

        if config.lstm:
            model += [SLSTM(mult * config.num_filters, num_layers=config.lstm)]

        model += [
            act(**config.activation_params),
            SConv1d(
                mult * config.num_filters,
                config.dimension,
                config.last_kernel_size,
                norm=config.norm,
                norm_kwargs=config.norm_params,
                causal=config.causal,
                pad_mode=config.pad_mode,
            )
        ]

        self.model = nn.Sequential(*model)
        #TODO: nn.ModuleList

    def forward(self, x):
        return self.model(x)


class SEANetDecoder(nn.Module):
    """SEANet decoder."""
    def __init__(self, config: EncodecConfig):
        super().__init__()
        # TODO: do we need these properties?
        self.dimension = config.dimension
        self.channels = config.audio_channels
        self.n_filters = config.num_filters
        self.ratios = config.ratios
        self.n_residual_layers = config.num_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, config.activation)
        mult = int(2 ** len(self.ratios))
        model = [
            SConv1d(
                config.dimension,
                mult * config.num_filters,
                config.kernel_size,
                norm=config.norm,
                norm_kwargs=config.norm_params,
                causal=config.causal,
                pad_mode=config.pad_mode
            )
        ]

        if config.lstm:
            model += [SLSTM(mult * config.num_filters, num_layers=config.lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add upsampling layers
            model += [
                act(**config.activation_params),
                SConvTranspose1d(
                    mult * config.num_filters,
                    mult * config.num_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=config.norm,
                    norm_kwargs=config.norm_params,
                    causal=config.causal,
                    trim_right_ratio=config.trim_right_ratio,
                )
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * config.num_filters // 2,
                        kernel_sizes=[config.residual_kernel_size, 1],
                        dilations=[config.dilation_base ** j, 1],
                        activation=config.activation,
                        activation_params=config.activation_params,
                        norm=config.norm,
                        norm_params=config.norm_params,
                        causal=config.causal,
                        pad_mode=config.pad_mode,
                        compress=config.compress,
                        true_skip=config.true_skip,
                    )
                ]
            mult //= 2

        # Add final layers
        model += [
            act(**config.activation_params),
            SConv1d(
                config.num_filters,
                config.audio_channels,
                config.last_kernel_size,
                norm=config.norm,
                norm_kwargs=config.norm_params,
                causal=config.causal,
                pad_mode=config.pad_mode
            )
        ]
        # Add optional final activation to decoder (eg. tanh)
        if config.final_activation is not None:
            final_act = getattr(nn, config.final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                final_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)
        # TODO: use ModuleList

    def forward(self, z):
        y = self.model(z)
        return y


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: Union[Callable[..., torch.Tensor], Any] = _uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = _kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        #TODO distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], _sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        #TODO distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = nn.functional.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        #TODO
        # if self.training:
        #     # We do the expiry of code at that point as buffers are in sync
        #     # and all the workers will take the same decision.
        #     self.expire_codes_(x)
        #     ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
        #     embed_sum = x.t() @ embed_onehot
        #     ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
        #     cluster_size = (
        #         laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
        #         * self.cluster_size.sum()
        #     )
        #     embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        #     self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = codebook_dim if codebook_dim is not None else dim

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            logger.warning('When using RVQ in training model, first check '
                          'https://github.com/facebookresearch/encodec/issues/25 . '
                          'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                commit_loss = nn.functional.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        config: EncodecConfig,
        n_q: int = 8,
    ):
        super().__init__()

        self.n_q = n_q
        self.dimension = config.dimension
        self.bins = config.bins
        self.decay = config.decay
        self.kmeans_init = config.kmeans_init
        self.kmeans_iters = config.kmeans_iters
        self.threshold_ema_dead_code = config.threshold_ema_dead_code

        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(self, x: torch.Tensor, frame_rate: int, bandwidth: Optional[float] = None) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            frame_rate (int): Sample rate of the input tensor.
            bandwidth (float): Target bandwidth.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))

    def get_num_quantizers_for_bandwidth(self, frame_rate: int, bandwidth: Optional[float] = None) -> int:
        """Return n_q based on specified target bandwidth.
        """
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.:
            # bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as
            # bandwidth == 6.0
            n_q = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return n_q

    def get_bandwidth_per_quantizer(self, frame_rate: int):
        """Return bandwidth per quantizer for a given input frame rate.
        Each quantizer encodes a frame with lg(bins) bits.
        """
        return math.log2(self.bins) * frame_rate

    def encode(self, x: torch.Tensor, frame_rate: int, bandwidth: Optional[float] = None) -> torch.Tensor:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizers to use
        and returns indices for each quantizer.
        """
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        codes = self.vq.encode(x, n_q=n_q)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        """
        quantized = self.vq.decode(codes)
        return quantized


class EncodecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EncodecConfig
    base_model_prefix = "encodec"
    main_input_name = "input_values"
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
        if isinstance(module, (EncodecEncoder, EncodecDecoder)):
            module.gradient_checkpointing = value


#TODO
# class EncodecEncoder(EncodecPreTrainedModel):
#     """
#     Transformer encoder consisting of *config.encoder_layers* layers. Each layer is a [`EncodecEncoderLayer`].
#     """

#     def __init__(self, config: EncodecConfig):
#         super().__init__(config)
#         self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout)
#         self.layerdrop = config.encoder_layerdrop

#         self.layers = nn.ModuleList([EncodecEncoderLayer(config) for _ in range(config.encoder_layers)])

#         self.embed_positions = EncodecRelativePositionalEncoding(
#             config.hidden_size // config.encoder_attention_heads, config.encoder_max_relative_position
#         )

#         self.gradient_checkpointing = False

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         hidden_states: torch.FloatTensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutput]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
#                 Features extracted from the speech or text input by the encoder prenet.
#             attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
#                 `[0, 1]`:

#                 - 1 for tokens that are **not masked**,
#                 - 0 for tokens that are **masked**.

#                 [What are attention masks?](../glossary#attention-mask)
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
#                 Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

#                 - 1 indicates the head is **not masked**,
#                 - 0 indicates the head is **masked**.

#             output_hidden_states (`bool`, *optional*):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more detail.
#             return_dict (`bool`, *optional*):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # expand attention_mask
#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

#         hidden_states = self.layer_norm(hidden_states)
#         hidden_states = self.dropout(hidden_states)

#         position_bias = self.embed_positions(hidden_states)

#         deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None

#         # check if head_mask has a correct number of layers specified if desired
#         if head_mask is not None:
#             if head_mask.size()[0] != len(self.layers):
#                 raise ValueError(
#                     f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
#                     f" {head_mask.size()[0]}."
#                 )

#         for idx, encoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             dropout_probability = np.random.uniform(0, 1)

#             skip_the_layer = self.training and (dropout_probability < self.layerdrop)
#             if not skip_the_layer or deepspeed_zero3_is_enabled:
#                 # under deepspeed zero3 all gpus must run in sync
#                 if self.gradient_checkpointing and self.training:
#                     # create gradient checkpointing function
#                     def create_custom_forward(module):
#                         def custom_forward(*inputs):
#                             return module(*inputs, output_attentions)

#                         return custom_forward

#                     layer_outputs = torch.utils.checkpoint.checkpoint(
#                         create_custom_forward(encoder_layer),
#                         hidden_states,
#                         attention_mask,
#                         (head_mask[idx] if head_mask is not None else None),
#                         position_bias,
#                     )
#                 else:
#                     layer_outputs = encoder_layer(
#                         hidden_states,
#                         attention_mask=attention_mask,
#                         position_bias=position_bias,
#                         layer_head_mask=(head_mask[idx] if head_mask is not None else None),
#                         output_attentions=output_attentions,
#                     )
#                 hidden_states = layer_outputs[0]

#             if skip_the_layer:
#                 layer_outputs = (None, None)

#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         )


#TODO
# class EncodecDecoder(EncodecPreTrainedModel):
#     """
#     Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`EncodecDecoderLayer`]
#     """

#     def __init__(self, config: EncodecConfig):
#         super().__init__(config)
#         self.layerdrop = config.decoder_layerdrop

#         self.layers = nn.ModuleList([EncodecDecoderLayer(config) for _ in range(config.decoder_layers)])

#         self.gradient_checkpointing = False

#         # Initialize weights and apply final processing
#         self.post_init()

#     # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
#     def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
#         # create causal mask
#         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#         combined_attention_mask = None
#         if input_shape[-1] > 1:
#             combined_attention_mask = _make_causal_mask(
#                 input_shape,
#                 inputs_embeds.dtype,
#                 device=inputs_embeds.device,
#                 past_key_values_length=past_key_values_length,
#             )

#         if attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
#                 inputs_embeds.device
#             )
#             combined_attention_mask = (
#                 expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
#             )

#         return combined_attention_mask

#     def forward(
#         self,
#         hidden_states: Optional[torch.FloatTensor] = None,
#         attention_mask: Optional[torch.LongTensor] = None,
#         encoder_hidden_states: Optional[torch.FloatTensor] = None,
#         encoder_attention_mask: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
#         r"""
#         Args:
#             hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_size)`):
#                 Features extracted from the speech or text input by the decoder prenet.
#             attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

#                 - 1 for tokens that are **not masked**,
#                 - 0 for tokens that are **masked**.

#                 [What are attention masks?](../glossary#attention-mask)
#             encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
#                 Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
#                 of the decoder.
#             encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
#                 Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
#                 selected in `[0, 1]`:

#                 - 1 for tokens that are **not masked**,
#                 - 0 for tokens that are **masked**.

#                 [What are attention masks?](../glossary#attention-mask)
#             head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
#                 Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

#                 - 1 indicates the head is **not masked**,
#                 - 0 indicates the head is **masked**.

#             cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
#                 Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
#                 cross-attention on hidden heads. Mask values selected in `[0, 1]`:

#                 - 1 indicates the head is **not masked**,
#                 - 0 indicates the head is **masked**.

#             past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#                 Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
#                 shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
#                 shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

#                 Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
#                 cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

#                 If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
#                 that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
#                 all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
#                 shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
#                 `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
#                 control over how to convert `input_ids` indices into associated vectors than the model's internal
#                 embedding lookup matrix.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             output_hidden_states (`bool`, *optional*):
#                 Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
#                 for more detail.
#             return_dict (`bool`, *optional*):
#                 Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         input_shape = hidden_states.size()[:-1]

#         past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

#         attention_mask = self._prepare_decoder_attention_mask(
#             attention_mask, input_shape, hidden_states, past_key_values_length
#         )

#         # expand encoder attention mask
#         if encoder_hidden_states is not None and encoder_attention_mask is not None:
#             # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#             encoder_attention_mask = _expand_mask(encoder_attention_mask, hidden_states.dtype, tgt_len=input_shape[-1])

#         deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         # decoder layers
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None
#         all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
#         next_decoder_cache = () if use_cache else None

#         # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
#         for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
#             if attn_mask is not None:
#                 if attn_mask.size()[0] != (len(self.layers)):
#                     raise ValueError(
#                         f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
#                         f" {head_mask.size()[0]}."
#                     )

#         for idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             dropout_probability = random.uniform(0, 1)

#             skip_the_layer = self.training and (dropout_probability < self.layerdrop)
#             if skip_the_layer and not deepspeed_zero3_is_enabled:
#                 continue

#             past_key_value = past_key_values[idx] if past_key_values is not None else None

#             if self.gradient_checkpointing and self.training:

#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         # None for past_key_value
#                         return module(*inputs, output_attentions, use_cache)

#                     return custom_forward

#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer),
#                     hidden_states,
#                     attention_mask,
#                     encoder_hidden_states,
#                     encoder_attention_mask,
#                     head_mask[idx] if head_mask is not None else None,
#                     cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
#                     None,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=attention_mask,
#                     encoder_hidden_states=encoder_hidden_states,
#                     encoder_attention_mask=encoder_attention_mask,
#                     layer_head_mask=(head_mask[idx] if head_mask is not None else None),
#                     cross_attn_layer_head_mask=(
#                         cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
#                     ),
#                     past_key_value=past_key_value,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                 )
#             hidden_states = layer_outputs[0]

#             if use_cache:
#                 next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)

#                 if encoder_hidden_states is not None:
#                     all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None
#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions, all_cross_attentions]
#                 if v is not None
#             )

#         return BaseModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#             cross_attentions=all_cross_attentions,
#         )


#TODO
ENCODEC_BASE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncodecConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        encoder ([`EncodecEncoderWithSpeechPrenet`] or [`EncodecEncoderWithTextPrenet`] or `None`):
            The Transformer encoder module that applies the appropiate speech or text encoder prenet. If `None`,
            [`EncodecEncoderWithoutPrenet`] will be used and the `input_values` are assumed to be hidden states.
        decoder ([`EncodecDecoderWithSpeechPrenet`] or [`EncodecDecoderWithTextPrenet`] or `None`):
            The Transformer decoder module that applies the appropiate speech or text decoder prenet. If `None`,
            [`EncodecDecoderWithoutPrenet`] will be used and the `decoder_input_values` are assumed to be hidden
            states.
"""


#TODO
ENCODEC_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncodecConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


#TODO
ENCODEC_INPUTS_DOCSTRING = r"""
    Args:
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.

            </Tip>

        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_values`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read [`EncodecDecoder._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

        head_mask (`torch.FloatTensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_values` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_values` of shape `(batch_size, sequence_length)`. decoder_inputs_embeds (`torch.FloatTensor`
            of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
            `decoder_input_values` you can choose to directly pass an embedded representation. If `past_key_values` is
            used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is
            useful if you want more control over how to convert `decoder_input_values` indices into associated vectors
            than the model's internal embedding lookup matrix.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The EnCodec neural audio codec model.",
    ENCODEC_BASE_START_DOCSTRING,
)
class EncodecModel(EncodecPreTrainedModel):
    def __init__(self, config: EncodecConfig):
        super().__init__(config)
        self.config = config

        self.encoder = SEANetEncoder(config)
        self.decoder = SEANetDecoder(config)

        n_q = int(1000 * config.target_bandwidths[-1] // (math.ceil(config.sampling_rate / self.encoder.hop_length) * 10))
        self.quantizer = ResidualVectorQuantizer(config, n_q)

        self.frame_rate = math.ceil(self.config.sampling_rate / np.prod(self.encoder.ratios))
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))

        if 2 ** self.bits_per_codebook != self.quantizer.bins:
            raise ValueError("Number of quantizer bins must be a power of 2.")

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def encode(self, input_values: torch.Tensor, bandwidth: Optional[float] = None) -> List[EncodedFrame]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            bandwidth (`float`, *optional*):
                The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible bandwidth.

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform,
            along with rescaling factors for each segment when `normalize` is True.
            Each frames is a tuple `(codebook, scale)`, with `codebook` of shape `[B, K, T]`, with `K` the number of codebooks, `T` frames.
        """
        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.config.target_bandwidths}.")

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        segment_length = self.config.segment_length
        if segment_length is None:
            segment_length = input_length
            stride = input_length
        else:
            stride = self.config.segment_stride

        encoded_frames = []
        for offset in range(0, input_length, stride):
            frame = input_values[:, :, offset : offset + segment_length]
            encoded_frames.append(self._encode_frame(frame, bandwidth))

        return encoded_frames

    def _encode_frame(self, input_values: torch.Tensor, bandwidth: float) -> EncodedFrame:
        length = input_values.shape[-1]
        duration = length / self.config.sampling_rate

        if self.config.segment is not None and duration > 1e-5 + self.config.segment:
            raise RuntimeError(f"Duration of frame ({duration}) is longer than segment {self.config.segment}")

        if self.config.normalize:
            mono = input_values.mean(dim=1, keepdim=True)
            scale = mono.pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-8
            input_values = input_values / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        embeddings = self.encoder(input_values)
        codes = self.quantizer.encode(embeddings, self.frame_rate, bandwidth)
        codes = codes.transpose(0, 1)
        return codes, scale

    def decode(self, encoded_frames: List[EncodedFrame]) -> torch.Tensor:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.config.segment_length
        if segment_length is None:
            if len(encoded_frames) != 1:
                raise ValueError(f"Expected one frame, got {len(encoded_frames)}")
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.config.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        codes = codes.transpose(0, 1)
        embeddings = self.quantizer.decode(codes)
        outputs = self.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale.view(-1, 1, 1)
        return outputs

    #TODO @add_start_docstrings_to_model_forward(ENCODEC_INPUTS_DOCSTRING)
    #TODO @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: torch.Tensor, bandwidth: Optional[float] = None) -> torch.Tensor:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
            Float values of the input audio waveform.

        bandwidth (`float`, *optional*):
            The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible bandwidth.

        Returns:
        """
        return self.decode(self.encode(input_values, bandwidth))[..., :input_values.shape[-1]]
