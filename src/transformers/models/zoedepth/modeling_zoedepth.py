# coding=utf-8
# Copyright 2024 Intel Labs, OpenMMLab and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ZoeDepth (Dense Prediction Transformers) model.

This implementation is heavily inspired by OpenMMLab's implementation, found here:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/zoedepth_head.py.

"""


import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..auto import AutoBackbone
from .configuration_zoedepth import ZoeDepthConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ZoeDepthConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "Intel/zoedepth-base"
_EXPECTED_OUTPUT_SHAPE = [1, 577, 1024]


class ZoeDepthReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Map the N + 1 tokens to a set of N tokens, by taking into account the readout ([CLS]) token according to
       `config.readout_type`.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[ZoeDepthConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList()
        self._init_reassemble_zoedepth(config)

        self.neck_ignore_stages = config.neck_ignore_stages

    def _init_reassemble_zoedepth(self, config):
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            self.layers.append(ZoeDepthReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        if config.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            hidden_size = config.backbone_config.hidden_size
            for _ in range(len(config.neck_hidden_sizes)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    # Copied from transformers.models.dpt.modeling_dpt.DPTReassembleStage.forward
    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []

        for i, hidden_state in enumerate(hidden_states):
            if i not in self.neck_ignore_stages:
                # reshape to (batch_size, num_channels, height, width)
                cls_token, hidden_state = hidden_state[:, 0], hidden_state[:, 1:]
                batch_size, sequence_length, num_channels = hidden_state.shape
                if patch_height is not None and patch_width is not None:
                    hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
                else:
                    size = int(math.sqrt(sequence_length))
                    hidden_state = hidden_state.reshape(batch_size, size, size, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_shape = hidden_state.shape
                if self.config.readout_type == "project":
                    # reshape to (batch_size, height*width, num_channels)
                    hidden_state = hidden_state.flatten(2).permute((0, 2, 1))
                    readout = cls_token.unsqueeze(1).expand_as(hidden_state)
                    # concatenate the readout token to the hidden states and project
                    hidden_state = self.readout_projects[i](torch.cat((hidden_state, readout), -1))
                    # reshape back to (batch_size, num_channels, height, width)
                    hidden_state = hidden_state.permute(0, 2, 1).reshape(feature_shape)
                elif self.config.readout_type == "add":
                    hidden_state = hidden_state.flatten(2) + cls_token.unsqueeze(-1)
                    hidden_state = hidden_state.reshape(feature_shape)
                hidden_state = self.layers[i](hidden_state)
            out.append(hidden_state)

        return out


class ZoeDepthReassembleLayer(nn.Module):
    def __init__(self, config, channels, factor):
        super().__init__()
        # projection
        hidden_size = config.backbone_config.hidden_size
        self.projection = nn.Conv2d(in_channels=hidden_size, out_channels=channels, kernel_size=1)

        # up/down sampling depending on factor
        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor < 1:
            # so should downsample
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

    # Copied from transformers.models.dpt.modeling_dpt.DPTReassembleLayer.forward with DPT->ZoeDepth
    def forward(self, hidden_state):
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)
        return hidden_state


# Copied from transformers.models.dpt.modeling_dpt.DPTFeatureFusionStage with DPT->ZoeDepth
class ZoeDepthFeatureFusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(ZoeDepthFeatureFusionLayer(config))

    def forward(self, hidden_states):
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        # first layer only uses the last hidden_state
        fused_hidden_state = self.layers[0](hidden_states[0])
        fused_hidden_states.append(fused_hidden_state)
        # looping from the last layer to the second
        for hidden_state, layer in zip(hidden_states[1:], self.layers[1:]):
            fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


# Copied from transformers.models.dpt.modeling_dpt.DPTPreActResidualLayer with DPT->ZoeDepth
class ZoeDepthPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[ZoeDepthConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        use_bias_in_fusion_residual = (
            config.use_bias_in_fusion_residual
            if config.use_bias_in_fusion_residual is not None
            else not self.use_batch_norm
        )

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(config.fusion_hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


# Copied from transformers.models.dpt.modeling_dpt.DPTFeatureFusionLayer with DPT->ZoeDepth
class ZoeDepthFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[ZoeDepthConfig]`):
            Model configuration class defining the model architecture.
        align_corners (`bool`, *optional*, defaults to `True`):
            The align_corner setting for bilinear upsample.
    """

    def __init__(self, config, align_corners=True):
        super().__init__()

        self.align_corners = align_corners

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)

        self.residual_layer1 = ZoeDepthPreActResidualLayer(config)
        self.residual_layer2 = ZoeDepthPreActResidualLayer(config)

    def forward(self, hidden_state, residual=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


# Copied from transformers.models.dpt.modeling_dpt.DPTPreTrainedModel with DPT->ZoeDepth,dpt->zoedepth
class ZoeDepthPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ZoeDepthConfig
    base_model_prefix = "zoedepth"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


ZOEDEPTH_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

ZOEDEPTH_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class ZoeDepthNeck(nn.Module):
    """
    ZoeDepthNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For ZoeDepth, it includes 2 stages:

    * ZoeDepthReassembleStage
    * ZoeDepthFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    # Copied from transformers.models.dpt.modeling_dpt.DPTNeck.__init__ with DPT->ZoeDepth
    def __init__(self, config):
        super().__init__()
        self.config = config

        # postprocessing: only required in case of a non-hierarchical backbone (e.g. ViT, BEiT)
        if config.backbone_config is not None and config.backbone_config.model_type in ["swinv2"]:
            self.reassemble_stage = None
        else:
            self.reassemble_stage = ZoeDepthReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        # fusion
        self.fusion_stage = ZoeDepthFeatureFusionStage(config)

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        if self.reassemble_stage is not None:
            hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]
        # we need the last feature of `features`

        # fusion blocks
        output = self.fusion_stage(features)

        # we need the last 4 features of `output` as well

        return output, features[-1]


class ZoeDepthRelativeDepthEstimationHead(nn.Module):
    """
    Relative depth estimation head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in DPT's paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.projection = None
        if config.add_projection:
            self.projection = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        features = config.fusion_hidden_size
        self.conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # use last features
        hidden_states = hidden_states[self.config.head_in_index]

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = nn.ReLU()(hidden_states)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = nn.ReLU()(hidden_states)
        # we need the features here (after second conv + ReLu)
        features = hidden_states
        hidden_states = self.conv3(hidden_states)
        hidden_states = nn.ReLU()(hidden_states)

        predicted_depth = hidden_states.squeeze(dim=1)

        return predicted_depth, features


def log_binom(n, k, eps=1e-7):
    """log(nCk) using stirling approximation"""
    n = n + eps
    k = k + eps
    return n * torch.log(n) - k * torch.log(k) - (n - k) * torch.log(n - k + eps)


class LogBinomial(nn.Module):
    def __init__(self, n_classes=256, act=torch.softmax):
        """Compute log binomial distribution for n_classes

        Args:
            n_classes (int, optional): number of output classes. Defaults to 256.
        """
        super().__init__()
        self.K = n_classes
        self.act = act
        self.register_buffer("k_idx", torch.arange(0, n_classes).view(1, -1, 1, 1))
        self.register_buffer("K_minus_1", torch.Tensor([self.K - 1]).view(1, -1, 1, 1))

    def forward(self, x, t=1.0, eps=1e-4):
        """Compute log binomial distribution for x

        Args:
            x (torch.Tensor - NCHW): probabilities
            t (float, torch.Tensor - NCHW, optional): Temperature of distribution. Defaults to 1..
            eps (float, optional): Small number for numerical stability. Defaults to 1e-4.

        Returns:
            torch.Tensor -NCHW: log binomial distribution logbinomial(p;t)
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # make it nchw

        one_minus_x = torch.clamp(1 - x, eps, 1)
        x = torch.clamp(x, eps, 1)
        y = (
            log_binom(self.K_minus_1, self.k_idx)
            + self.k_idx * torch.log(x)
            + (self.K - 1 - self.k_idx) * torch.log(one_minus_x)
        )
        return self.act(y / t, dim=1)


class ZoeDepthConditionalLogBinomial(nn.Module):
    def __init__(
        self,
        in_features,
        condition_dim,
        n_classes=256,
        bottleneck_factor=2,
        p_eps=1e-4,
        max_temp=50,
        min_temp=1e-7,
        act=torch.softmax,
    ):
        """Conditional Log Binomial distribution

        Args:
            in_features (int): number of input channels in main feature
            condition_dim (int): number of input channels in condition feature
            n_classes (int, optional): Number of classes. Defaults to 256.
            bottleneck_factor (int, optional): Hidden dim factor. Defaults to 2.
            p_eps (float, optional): small eps value. Defaults to 1e-4.
            max_temp (float, optional): Maximum temperature of output distribution. Defaults to 50.
            min_temp (float, optional): Minimum temperature of output distribution. Defaults to 1e-7.
        """
        super().__init__()
        self.p_eps = p_eps
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.log_binomial_transform = LogBinomial(n_classes, act=act)
        bottleneck = (in_features + condition_dim) // bottleneck_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features + condition_dim, bottleneck, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            # 2 for p linear norm, 2 for t linear norm
            nn.Conv2d(bottleneck, 2 + 2, kernel_size=1, stride=1, padding=0),
            nn.Softplus(),
        )

    def forward(self, x, cond):
        """Forward pass

        Args:
            x (torch.Tensor - NCHW): Main feature
            cond (torch.Tensor - NCHW): condition feature

        Returns:
            torch.Tensor: Output log binomial distribution
        """
        pt = self.mlp(torch.concat((x, cond), dim=1))
        p, t = pt[:, :2, ...], pt[:, 2:, ...]

        p = p + self.p_eps
        p = p[:, 0, ...] / (p[:, 0, ...] + p[:, 1, ...])

        t = t + self.p_eps
        t = t[:, 0, ...] / (t[:, 0, ...] + t[:, 1, ...])
        t = t.unsqueeze(1)
        t = (self.max_temp - self.min_temp) * t + self.min_temp

        return self.log_binomial_transform(p, t)


class ZoeDepthSeedBinRegressor(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are bounded on (min_depth, max_depth) interval.

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Min depth value. Defaults to 1e-3.
            max_depth (float, optional): Max depth value. Defaults to 10.
        """
        super().__init__()
        self.version = "1_1"
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B = self._net(x)
        eps = 1e-3
        B = B + eps
        B_widths_normed = B / B.sum(dim=1, keepdim=True)
        B_widths = (self.max_depth - self.min_depth) * B_widths_normed  # .shape NCHW
        # pad has the form (left, right, top, bottom, front, back)
        B_widths = nn.functional.pad(B_widths, (0, 0, 0, 0, 1, 0), mode="constant", value=self.min_depth)
        B_edges = torch.cumsum(B_widths, dim=1)  # .shape NCHW

        B_centers = 0.5 * (B_edges[:, :-1, ...] + B_edges[:, 1:, ...])
        return B_widths_normed, B_centers


class ZoeDepthSeedBinRegressorUnnormed(nn.Module):
    def __init__(self, in_features, n_bins=16, mlp_dim=256, min_depth=1e-3, max_depth=10):
        """Bin center regressor network. Bin centers are unbounded

        Args:
            in_features (int): input channels
            n_bins (int, optional): Number of bin centers. Defaults to 16.
            mlp_dim (int, optional): Hidden dimension. Defaults to 256.
            min_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
            max_depth (float, optional): Not used. (for compatibility with SeedBinRegressor)
        """
        super().__init__()
        self.version = "1_1"
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.Softplus(),
        )

    def forward(self, x):
        """
        Returns tensor of bin_width vectors (centers). One vector b for every pixel
        """
        B_centers = self._net(x)
        return B_centers, B_centers


@torch.jit.script
def inv_attractor(dx, alpha: float = 300, gamma: int = 2):
    """Inverse attractor: dc = dx / (1 + alpha*dx^gamma), where dx = a - c, a = attractor point, c = bin center, dc = shift in bin center
    This is the default one according to the accompanying paper.

    Args:
        dx (torch.Tensor): The difference tensor dx = Ai - Cj, where Ai is the attractor point and Cj is the bin center.
        alpha (float, optional): Proportional Attractor strength. Determines the absolute strength. Lower alpha = greater attraction. Defaults to 300.
        gamma (int, optional): Exponential Attractor strength. Determines the "region of influence" and indirectly number of bin centers affected. Lower gamma = farther reach. Defaults to 2.

    Returns:
        torch.Tensor: Delta shifts - dc; New bin centers = Old bin centers + dc
    """
    return dx.div(1 + alpha * dx.pow(gamma))


class ZoeDepthAttractorLayer(nn.Module):
    def __init__(
        self,
        in_features,
        n_bins,
        n_attractors=16,
        mlp_dim=128,
        min_depth=1e-3,
        max_depth=10,
        alpha=300,
        gamma=2,
        kind="sum",
        memory_efficient=False,
    ):
        """
        Attractor layer for bin centers. Bin centers are bounded on the interval (min_depth, max_depth)
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.memory_efficient = memory_efficient

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_attractors * 2, 1, 1, 0),  # x2 for linear norm
            nn.ReLU(inplace=True),
        )

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True, is_for_query=False):
        """
        Args:
            x (torch.Tensor) : feature block; shape - n, c, h, w
            b_prev (torch.Tensor) : previous bin centers normed; shape - n, prev_nbins, h, w

        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers normed and scaled; shape - n, nbins, h, w
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(
                    prev_b_embedding, x.shape[-2:], mode="bilinear", align_corners=True
                )
            x = x + prev_b_embedding

        A = self._net(x)
        eps = 1e-3
        A = A + eps
        n, c, h, w = A.shape
        A = A.view(n, self.n_attractors, 2, h, w)
        A_normed = A / A.sum(dim=2, keepdim=True)  # n, a, 2, h, w
        A_normed = A[:, :, 0, ...]  # n, na, h, w

        b_prev = nn.functional.interpolate(b_prev, (h, w), mode="bilinear", align_corners=True)
        b_centers = b_prev

        # note: only attractor_type = "exp" is supported here, since no checkpoints were released with other attractor types
        distribution = inv_attractor

        if not self.memory_efficient:
            func = {"mean": torch.mean, "sum": torch.sum}[self.kind]
            # shape (N, nbins, height, width)
            delta_c = func(distribution(A_normed.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                # shape (N, nbins, height, width)
                delta_c += distribution(A_normed[:, i, ...].unsqueeze(1) - b_centers)

            if self.kind == "mean":
                delta_c = delta_c / self.n_attractors

        b_new_centers = b_centers + delta_c
        B_centers = (self.max_depth - self.min_depth) * b_new_centers + self.min_depth
        B_centers, _ = torch.sort(B_centers, dim=1)
        B_centers = torch.clip(B_centers, self.min_depth, self.max_depth)
        return b_new_centers, B_centers


class ZoeDepthAttractorLayerUnnormed(nn.Module):
    def __init__(
        self,
        in_features,
        n_bins,
        n_attractors=16,
        mlp_dim=128,
        min_depth=1e-3,
        max_depth=10,
        alpha=300,
        gamma=2,
        kind="sum",
        memory_efficient=False,
    ):
        """
        Attractor layer for bin centers. Bin centers are unbounded
        """
        super().__init__()

        self.n_attractors = n_attractors
        self.n_bins = n_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.memory_efficient = memory_efficient

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_attractors, 1, 1, 0),
            nn.Softplus(),
        )

    def forward(self, x, b_prev, prev_b_embedding=None, interpolate=True):
        """
        Args:
            x (torch.Tensor):
                Feature block; shape - n, c, h, w
            b_prev (torch.Tensor) :
                Previous bin centers normed; shape - n, prev_nbins, h, w

        Returns:
            tuple(torch.Tensor,torch.Tensor) : new bin centers unbounded; shape - n, nbins, h, w. Two outputs just to keep the API consistent with the normed version
        """
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = nn.functional.interpolate(
                    prev_b_embedding, x.shape[-2:], mode="bilinear", align_corners=True
                )
            x = x + prev_b_embedding

        A = self._net(x)
        n, c, h, w = A.shape

        b_prev = nn.functional.interpolate(b_prev, (h, w), mode="bilinear", align_corners=True)
        b_centers = b_prev

        dist = inv_attractor

        if not self.memory_efficient:
            func = {"mean": torch.mean, "sum": torch.sum}[self.kind]
            # .shape N, nbins, h, w
            delta_c = func(dist(A.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers, device=b_centers.device)
            for i in range(self.n_attractors):
                delta_c += dist(A[:, i, ...].unsqueeze(1) - b_centers)  # .shape N, nbins, h, w

            if self.kind == "mean":
                delta_c = delta_c / self.n_attractors

        b_new_centers = b_centers + delta_c
        B_centers = b_new_centers

        return b_new_centers, B_centers


class ZoeDepthProjector(nn.Module):
    def __init__(self, in_features, out_features, mlp_dim=128):
        """Projector MLP

        Args:
            in_features (int): input channels
            out_features (int): output channels
            mlp_dim (int, optional): hidden dimension. Defaults to 128.
        """
        super().__init__()

        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0),
        )

    def forward(self, x):
        return self._net(x)


class ZoeDepthMetricDepthEstimationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_bins = config.n_bins
        bin_embedding_dim = config.bin_embedding_dim
        min_depth = config.min_depth
        max_depth = config.max_depth
        n_attractors = config.num_attractors
        num_out_features = config.num_out_features
        attractor_alpha = config.attractor_alpha
        attractor_gamma = config.attractor_gamma
        attractor_kind = config.attractor_kind
        min_temp = config.min_temp
        max_temp = config.max_temp
        bin_centers_type = config.bin_centers_type

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.bin_centers_type = bin_centers_type

        # Bottleneck convolution
        bottleneck_features = config.bottleneck_features
        self.conv2 = nn.Conv2d(bottleneck_features, bottleneck_features, kernel_size=1, stride=1, padding=0)

        # Regressor and attractor
        if self.bin_centers_type == "normed":
            SeedBinRegressorLayer = ZoeDepthSeedBinRegressor
            Attractor = ZoeDepthAttractorLayer
        elif self.bin_centers_type == "softplus":
            SeedBinRegressorLayer = ZoeDepthSeedBinRegressorUnnormed
            Attractor = ZoeDepthAttractorLayerUnnormed

        self.seed_bin_regressor = SeedBinRegressorLayer(
            bottleneck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth
        )
        self.seed_projector = ZoeDepthProjector(bottleneck_features, bin_embedding_dim)

        self.projectors = nn.ModuleList(
            [ZoeDepthProjector(num_out, bin_embedding_dim) for num_out in num_out_features]
        )
        self.attractors = nn.ModuleList(
            [
                Attractor(
                    bin_embedding_dim,
                    n_bins,
                    n_attractors=n_attractors[i],
                    min_depth=min_depth,
                    max_depth=max_depth,
                    alpha=attractor_alpha,
                    gamma=attractor_gamma,
                    kind=attractor_kind,
                )
                for i in range(len(num_out_features))
            ]
        )

        N_MIDAS_OUT = 32
        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ZoeDepthConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp
        )

    def forward(self, out, rel_depth):
        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == "normed" or self.bin_centers_type == "hybrid2":
            b_prev = (seed_b_centers - self.min_depth) / (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation

        # concatenative relative depth with last. First interpolate relative depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        rel_cond = nn.functional.interpolate(rel_cond, size=last.shape[2:], mode="bilinear", align_corners=True)
        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(b_embedding, last.shape[-2:], mode="bilinear", align_corners=True)
        x = self.conditional_log_binomial(last, b_embedding)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        b_centers = nn.functional.interpolate(b_centers, x.shape[-2:], mode="bilinear", align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)

        return out


@add_start_docstrings(
    """
    ZoeDepth model with a metric depth estimation head on top.
    """,
    ZOEDEPTH_START_DOCSTRING,
)
class ZoeDepthForDepthEstimation(ZoeDepthPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # TODO perhaps just use AutoModelForDepthEstimation?
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        self.neck = ZoeDepthNeck(config)
        self.relative_head = ZoeDepthRelativeDepthEstimationHead(config)

        self.metric_head = ZoeDepthMetricDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ZOEDEPTH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
        >>> import torch
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-base")
        >>> model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-base")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        ...     predicted_depth = outputs.predicted_depth

        >>> # interpolate to original size
        >>> prediction = torch.nn.functional.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().cpu().numpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        hidden_states = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.backbone_config.patch_size
        patch_height = height // patch_size
        patch_width = width // patch_size

        hidden_states, features = self.neck(hidden_states, patch_height, patch_width)

        out = [features] + hidden_states

        relative_depth, features = self.relative_head(hidden_states)

        out = [features] + out

        metric_depth = self.metric_head(out, relative_depth)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        if not return_dict:
            if output_hidden_states:
                output = (metric_depth,) + outputs[1:]
            else:
                output = (metric_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=metric_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
