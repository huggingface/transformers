# coding=utf-8
# Copyright 2023 IBM and HuggingFace Inc. team. All Rights Reserved.
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
""" PyTorch PatchTSMixer model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_patchtsmixer import PatchTSMixerConfig
from .layers import (
    PatchMasking,
    PatchTSMixer,
    set_seed,
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSMixerConfig"


PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtsmixer-etth1-pretrain",
    # See all PatchTST models at https://huggingface.co/models?filter=patchtsmixer
]


PATCHTSMIXER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSMixerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PATCHTSMIXER_INPUTS_DOCSTRING = r"""
    Args:
        context_values (`torch.FloatTensor` of shape `(batch_size, seq_length, input_size)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

            For univariate time series, `input_size` dimension should be 1. For multivariate time series, it is > 1.

        target_values (`torch.FloatTensor` of shape `(batch_size, target_len, input_size)` for forecasting,
            `(batch_size, n_targets)` for regression, or `(batch_size,)` for classification, *optional*): Target values
            of the time series, that serve as labels for the model. The `target_values` is what the Transformer needs
            during training to learn to output, given the `context_values`. Note that, this is NOT required for a
            pretraining task.

            For a forecasting task, the shape is be `(batch_size, target_len, input_size)`. Even if we want to forecast
            only specific channels by setting the indices in `forecast_channel_indices` parameter, pass the target data
            with all channels, as channel Filtering for both prediction and target will be manually applied before the
            loss computation.

            For a classification task, it has a shape of `(batch_size,)`.

            For a regression task, it has a shape of `(batch_size, n_targets)`.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.
"""


class ForecastHead(nn.Module):
    """Forecast Head

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        head_dropout (float, optional): Head Dropout rate. Defaults to 0.2.
        forecast_len (int, optional): Forecast Length. Defaults to 16.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        forecast_channel_indices (list, optional):
            List of channel indices to forecast. If None, forecast all channels.
    """

    def __init__(
        self,
        num_patches: int,
        in_channels: int = 3,
        patch_size: int = 16,
        num_features: int = 16,
        forecast_len: int = 16,
        head_dropout: float = 0.2,
        mode: str = "common_channel",
        forecast_channel_indices: list = None,
    ):
        super().__init__()
        self.forecast_len = forecast_len
        self.nvars = in_channels
        self.num_features = num_features
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.forecast_channel_indices = forecast_channel_indices

        if self.forecast_channel_indices is not None:
            self.forecast_channel_indices.sort()
        self.mode = mode

        if self.mode in ["common_channel", "mix_channel"]:
            self.base_forecast_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear((num_patches * num_features), forecast_len),
            )
            self.flatten = nn.Flatten(start_dim=-2)

        else:
            self.base_forecast_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear((num_patches * num_features), forecast_len * in_channels),
            )
            self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x, y=None):
        """
        # x: [bs x num_patch x num_features] flatten mode or
            [bs x n_vars x num_patch x num_features] common_channel/mix_channel

        Output: [bs x forecast_len x nvars]

        """
        if self.mode in ["common_channel", "mix_channel"]:
            x = self.flatten(x)  # [bs x n_vars x num_patch * num_features]
            # x = torch.reshape(
            #     x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            # )  # [bs x n_vars x num_patch * num_features]

            forecast = self.base_forecast_block(x)  # [bs x n_vars x forecast_len]
            forecast = forecast.transpose(-1, -2)  # [bs x forecast_len x n_vars]

        else:
            x = self.flatten(x)  # x: [bs x num_patches*num_features]
            forecast = self.base_forecast_block(x)  # [bs x forecast_len * self.nvars]
            forecast = forecast.reshape(-1, self.forecast_len, self.nvars)  # y: [bs x forecast_len x n_vars]

        if self.forecast_channel_indices is not None:
            forecast = forecast[..., self.forecast_channel_indices]

        return forecast


class LinearHead(nn.Module):
    """LinearHead for Classification and Regression

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        head_dropout (float, optional): Head Dropout rate. Defaults to 0.2.
        head_agg (str, optional): Aggregation mode. Allowed values are use_last, max_pool, avg_pool.
                                Defaults to max_pool.
        output_range (list, optional): Output range of [low, high] to restrict sigmoid. Defaults to None.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
    """

    def __init__(
        self,
        num_patches: int = 5,
        in_channels: int = 3,
        num_features: int = 16,
        head_dropout: float = 0.2,
        output_dim: int = 1,
        output_range: list = None,
        head_agg: str = "max_pool",
        mode: str = "common_channel",
    ):
        super().__init__()
        self.nvars = in_channels
        self.num_features = num_features
        self.in_channels = in_channels
        self.head_dropout = head_dropout
        self.output_dim = output_dim
        self.mode = mode
        self.head_agg = head_agg
        self.output_range = output_range
        self.num_patches = num_patches

        if self.head_agg is None:
            mul_factor = self.num_patches
        else:
            mul_factor = 1

        if mode != "flatten":
            self.linear = nn.Linear(num_features * in_channels * mul_factor, output_dim)
            if self.head_agg is None:
                self.flatten = nn.Flatten(start_dim=-3)
            else:
                self.flatten = nn.Flatten(start_dim=-2)
        else:
            self.linear = nn.Linear(num_features * mul_factor, output_dim)
            if self.head_agg is None:
                self.flatten = nn.Flatten(start_dim=-2)
            else:
                self.flatten = None

        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, y=None):
        """
        # x: [bs x num_patch x num_features] flatten mode or
            [bs x n_vars x num_patch x num_features] common_channel/mix_channel
        Output: [bs x output_dim]
        """
        x = x.transpose(-1, -2)  # bs x num_features x num_patch or bs x n_vars x num_features x num_patch
        if self.head_agg == "use_last":
            x = x[..., -1]  # # bs x num_features (flatten) or # bs x n_vars x num_features (common_channel)
            # if self.mode  == "flatten":
            #     x = x[:,:,-1] # bs x num_features
            # else:
            #     x = x[:,:,:,-1] # bs x n_vars x num_features
        elif self.head_agg == "max_pool":
            x = x.max(dim=-1).values  # bs x n_vars x num_features or bs x num_features
        elif self.head_agg == "avg_pool":
            x = x.mean(dim=-1)  # bs x n_vars x num_features or bs x num_features

        if self.flatten:
            x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)  # bs x output_dim

        if self.output_range is not None:
            x = torch.sigmoid(x) * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        return x


class PatchTSMixerPreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = PatchTSMixerConfig
    base_model_prefix = "model"
    main_input_name = "context_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights"""
        # print("Module = ", module)
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PatchTSMixerEncoder)):
            module.gradient_checkpointing = value


class PretrainHead(nn.Module):
    """Pretrain head

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        input_size (int, optional): Number of input variables. Defaults to 1.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        head_dropout (float, optional): Head Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
    """

    def __init__(
        self,
        num_patches: int,
        num_features: int = 16,
        input_size: int = 1,
        patch_size: int = 16,
        head_dropout: float = 0,
        mode: str = "common_channel",
    ):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.input_size = input_size
        self.num_patches = num_patches

        if self.mode in ["common_channel", "mix_channel"]:
            self.base_pt_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_size),
            )
        else:
            self.base_pt_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_size * input_size),
            )

    def forward(self, x, y=None):
        """
        # flatten mode: [bs x num_patch x num_features] or common_channel/mix_channel mode: [bs x n_vars x num_patch x
        num_features]

        Output: z: [bs x n_vars x num_patch x patch_len]
        """

        if self.mode == "flatten":
            x = self.base_pt_block(x)  # x: [bs x num_patch x n_vars*patch_size]
            x = torch.reshape(
                x, (x.shape[0], x.shape[1], self.patch_size, self.input_size)
            )  # [bs x num_patch x patch_size x n_vars]
            x = x.permute(0, 3, 1, 2)  # [bs x nvars x num_patch  x patch_len]
            return x
        elif self.mode in ["common_channel", "mix_channel"]:
            forecast = self.base_pt_block(x)  # [bs x n_vars x num_patch x patch_size]
            return forecast


# Copied from transformers.models.patchtst.modeling_patchtst.positional_encoding
def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe is None:
        w_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zeros":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "normal":
        w_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == "sincos":
        pos_enc = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc - pos_enc.mean()
        pos_enc = pos_enc / (pos_enc.std() * 10)
        w_pos = pos_enc
    else:
        raise ValueError(
            f"{pe} is not a valid positional encoder. Available types are 'normal', 'zeros', 'zero', uniform', 'sincos', None."
        )
    return nn.Parameter(w_pos, requires_grad=learn_pe)


# Copied from transformers.models.patchtst.modeling_patchtst.compute_num_patches
def compute_num_patches(sequence_length, patch_length, stride):
    return (max(sequence_length, patch_length) - patch_length) // stride + 1


# Copied from transformers.models.patchtst.modeling_patchtst.Patchify
class Patchify(nn.Module):
    """
    Parameters:
    A class to patchify the time series sequence into different patches
        sequence_length (int, required): input sequence length.
        patch_length (int, required): patch length.
        stride (int, required): stride between patches.
    Returns:
        z: output tensor data [bs x num_input_channels x num_patches x patch_length]
    """

    def __init__(
        self,
        sequence_length: int,
        patch_length: int,
        stride: int,
        padding: bool = False,  # TODO: use this to set whether we want to pad zeros to the sequence
    ):
        super().__init__()

        assert (
            sequence_length > patch_length
        ), f"Sequence length ({sequence_length}) has to be greater than the patch length ({patch_length})"

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.stride = stride

        # get the number of patches
        self.num_patches = compute_num_patches(sequence_length, patch_length, stride)
        new_sequence_length = patch_length + stride * (self.num_patches - 1)
        self.s_begin = sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (torch.Tensor, required): Input of shape [bs x sequence_length x num_input_channels]
        Returns:
            x: output tensor data [bs x num_input_channels x num_patches x patch_length]
        """
        sequence_length = past_values.shape[-2]
        assert (
            sequence_length == self.sequence_length
        ), f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."

        x = past_values[:, self.s_begin :, :]  # x: [bs x new_sequence_length x nvars]
        x = x.unfold(
            dimension=-2, size=self.patch_length, step=self.stride
        )  # x: [bs x num_patches x num_input_channels x patch_length]
        x = x.transpose(-2, -3).contiguous()  # xb: [bs x num_input_channels x num_patches x patch_length]
        return x


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler with TimeSeries->PatchTSMixer
class PatchTSMixerStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along some given dimension `dim`, and then normalizes it
    by subtracting from the mean and dividing by the standard deviation.

    Args:
        dim (`int`):
            Dimension along which to calculate the mean and standard deviation.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        minimum_scale (`float`, *optional*, defaults to 1e-5):
            Default scale that is used for elements that are constantly zero along dimension `dim`.
    """

    def __init__(self, dim: int, keepdim: bool = False, minimum_scale: float = 1e-5):
        super().__init__()
        if not dim > 0:
            raise ValueError("Cannot compute scale along dim = 0 (batch dimension), please provide dim > 0")
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    @torch.no_grad()
    def forward(self, data: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denominator = weights.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * weights).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * weights) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeries->PatchTSMixer
class PatchTSMixerMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along dimension `dim`, and scales the data
    accordingly.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        default_scale (`float`, *optional*, defaults to `None`):
            Default scale that is used for elements that are constantly zero. If `None`, we use the scale of the batch.
        minimum_scale (`float`, *optional*, defaults to 1e-10):
            Default minimum possible scale that is used for any item.
    """

    def __init__(
        self, dim: int = -1, keepdim: bool = True, default_scale: Optional[float] = None, minimum_scale: float = 1e-10
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale

    @torch.no_grad()
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: (N, [C], T=1)
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler with TimeSeries->PatchTSMixer
class PatchTSMixerNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along dimension `dim`, and therefore applies no scaling to the input data.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
    """

    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


class InjectScalerStatistics4D(nn.Module):
    def __init__(self, num_features, num_patches, expansion=2):
        super().__init__()
        self.inverse_transform = nn.Sequential(
            nn.Linear(num_features + 2, expansion * num_features),
            nn.Linear(expansion * num_features, num_features),
        )

        self.map_scale = nn.Sequential(nn.Linear(2, 2 * expansion), nn.Linear(2 * expansion, 2))
        self.num_patches = num_patches

    def forward(self, z, loc, scale):
        """
        # revin_mean,revin_stddev: [bs x 1 x n_channels] z: [bs x in_channels x num_patch x num_features]

        output: [bs x in_channels x num_patch x num_features]
        """

        mean = loc.transpose(-1, -2)  # [bs x n_channels x 1 ]
        mean = mean.unsqueeze(-2)  # [bs x n_channels x 1 x 1]
        mean = mean.repeat(1, 1, self.num_patches, 1)  # [bs x n_channels x num_patch x 1]

        stdev = scale.transpose(-1, -2)  # [bs x n_channels x 1 ]
        stdev = stdev.unsqueeze(-2)  # [bs x n_channels x 1 x 1]
        stdev = stdev.repeat(1, 1, self.num_patches, 1)  # [bs x n_channels x num_patch x 1]

        concat_stats = torch.cat([mean, stdev], dim=-1)  # [bs x n_channels x num_patch x 2]

        concat_stats = self.map_scale(concat_stats)  # [bs x n_channels x num_patch x 2]

        z = torch.cat([z, concat_stats], dim=-1)  # [bs x channels x num_patch x num_features+2]
        z = self.inverse_transform(z)  # [bs x channels x num_patch x num_features]

        return z


@dataclass
class PatchTSMixerEncoderOutput(ModelOutput):
    """
    Base class for `PatchTSMixerEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, num_features)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class PatchTSMixerEncoder(PatchTSMixerPreTrainedModel):
    """
    Encoder for PatchTSMixer which inputs patched time-series and outputs patched embeddings.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.encoder = PatchTSMixer(
            num_patches=config.num_patches,
            patch_size=config.patch_len,
            in_channels=config.input_size,
            num_features=config.num_features,
            expansion_factor=config.expansion_factor,
            num_layers=config.num_layers,
            dropout=config.dropout,
            mode=config.mode,
            gated_attn=config.gated_attn,
            self_attn=config.self_attn,
            self_attn_heads=config.self_attn_heads,
            norm_mlp=config.norm_mlp,
            use_pe=config.use_pe,
            pe=config.pe,
            learn_pe=config.learn_pe,
        )

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=PatchTSMixerEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        context_values: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSMixerEncoderOutput]:
        r"""
        Args:
        context_values (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, patch_len)`):
            Patched input context.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        Returns:
        """

        # context_values: [bs  x n_vars x num_patches x patch_len]
        # return: [bs x n_vars x num_patches x num_features]
        last_hidden_state, hidden_states = self.encoder(context_values, output_hidden_states=output_hidden_states)
        return PatchTSMixerEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


@dataclass
class PatchTSMixerModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, num_features)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patched_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_len)`):
            Patched input data to the model.
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*):
            Bool Tensor indicating True in masked patches and False otherwise.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel. Used for revin denorm outside the model, if revin
            enabled.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patched_input: torch.FloatTensor = None
    mask: Optional[torch.FloatTensor] = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The PatchTSMixer Model for time-series forecasting.",
    PATCHTSMIXER_START_DOCSTRING,
)
class PatchTSMixerModel(PatchTSMixerPreTrainedModel):
    def __init__(self, config: PatchTSMixerConfig, mask_input: bool = False):
        super().__init__(config)

        set_seed(config.seed_number)

        self.encoder = PatchTSMixerEncoder(config)
        self.patching = Patchify(config.seq_len, patch_length=config.patch_len, stride=config.stride)

        if mask_input is True:
            self.masking = PatchMasking(
                mask_type=config.mask_type,
                mask_ratio=config.mask_ratio,
                mask_patches=config.mask_patches,
                mask_patch_ratios=config.mask_patch_ratios,
                channel_consistent_masking=config.channel_consistent_masking,
                d_size=config.d_size,
                cv_channel_indices=None,
                mask_value=config.mask_value,
            )
        else:
            self.masking = None

        if config.scaling == "mean" or config.scaling is True:
            self.scaler = PatchTSMixerMeanScaler(dim=1, keepdim=True)
        elif config.scaling == "std":
            self.scaler = PatchTSMixerStdScaler(dim=1, keepdim=True)
        else:
            self.scaler = PatchTSMixerNOPScaler(dim=1, keepdim=True)

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=PatchTSMixerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        context_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSMixerModelOutput]:
        r"""
        Args:
        context_values (`torch.FloatTensor` of shape `(batch_size, seq_length, input_size)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

            For univariate time series, `input_size` dimension should be 1. For multivariate time series, it is > 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.


        Returns:

        """

        mask = None
        if observed_mask is None:
            observed_mask = torch.ones_like(context_values)
        scaled_context_values, loc, scale = self.scaler(context_values, observed_mask)

        patched_x = self.patching(scaled_context_values)  # [bs x input_size x num_patch x patch_len

        enc_input = patched_x
        if self.masking is not None:
            enc_input, mask = self.masking(patched_x)
            # enc_input: [bs x input_size x num_patch x patch_len]
            # mask: [bs x input_size x num_patch]

        encoder_output = self.encoder(enc_input, output_hidden_states=output_hidden_states)

        return PatchTSMixerModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patched_input=patched_x,
            mask=mask,
            loc=loc,
            scale=scale,
        )


@dataclass
class PatchTSMixerForMaskPreTrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForMaskPreTrainingOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, patch_len)`):
            Prediction output from the pretrain head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForMaskPretraining(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for mask pretraining.

    Args:
        config (`PatchTSMixerConfig`, *mandatory*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)
        self.model = PatchTSMixerModel(config, mask_input=True)
        self.head = PretrainHead(
            num_patches=config.num_patches,
            num_features=config.num_features,
            input_size=config.input_size,
            patch_size=config.patch_len,
            head_dropout=config.head_dropout,
            mode=config.mode,
        )
        self.masked_loss = config.masked_loss
        if config.masked_loss is True:
            self.loss = torch.nn.MSELoss(reduction="none")
        else:
            self.loss = torch.nn.MSELoss(reduction="mean")

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    # @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForMaskPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        context_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForMaskPreTrainingOutput:
        r"""
        Args:
        context_values (`torch.FloatTensor` of shape `(batch_size, seq_length, input_size)`):
            Context values of the time series. For a pretraining task, this denotes the input time series to predict
            the masked portion. For a forecasting task, this denotes the history/past time series values. Similarly,
            for classification or regression tasks, it denotes the appropriate context values of the time series.

            For univariate time series, `input_size` dimension should be 1. For multivariate time series, it is > 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """

        # context_values: tensor [bs x seq_len x input_size]
        model_output = self.model(
            context_values, observed_mask=observed_mask, output_hidden_states=output_hidden_states
        )  # x.last_hidden_state: [bs x nvars x num_patch x num_features]
        x_hat = self.head(model_output.last_hidden_state)  # tensor [bs x nvars x num_patch x patch_len]

        if return_loss is True:
            loss_val = self.loss(x_hat, model_output.patched_input)
        else:
            loss_val = None

        # calculate masked_loss
        if self.masked_loss is True and loss_val is not None:
            loss_val = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        return PatchTSMixerForMaskPreTrainingOutput(
            prediction_logits=x_hat,  # tensor [bs x nvars x num_patch x patch_len]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )


@dataclass
class PatchTSMixerForForecastOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForForecastOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, forecast_len, input_size)`):
            Prediction output from the forecast head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForForecasting(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for forecasting application.

    Args:
        config (`PatchTSMixerConfig`, *mandatory*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = ForecastHead(
            num_patches=config.num_patches,
            in_channels=config.input_size,
            patch_size=config.patch_len,
            num_features=config.num_features,
            forecast_len=config.forecast_len,
            head_dropout=config.head_dropout,
            mode=config.mode,
            forecast_channel_indices=config.forecast_channel_indices,
        )
        self.loss = torch.nn.MSELoss(reduction="mean")

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForForecastOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        context_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForForecastOutput:
        r"""

        Returns:

        """

        # context_values: tensor [bs x seq_len x input_size]
        model_output = self.model(
            context_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
        )  # model_output: [bs x nvars x num_patch x num_features]

        y_hat = self.head(
            model_output.last_hidden_state,
        )  # tensor [bs x forecast_len x input_size]

        if target_values is not None and return_loss is True:
            if self.config.forecast_channel_indices is not None:
                y_hat_unscaled = (
                    y_hat * model_output.scale[..., self.config.forecast_channel_indices]
                    + model_output.loc[..., self.config.forecast_channel_indices]
                )
                loss_val = self.loss(y_hat_unscaled, target_values[..., self.config.forecast_channel_indices])
            else:
                y_hat_unscaled = y_hat * model_output.scale + model_output.loc
                loss_val = self.loss(y_hat_unscaled, target_values)
        else:
            loss_val = None

        return PatchTSMixerForForecastOutput(
            prediction_logits=y_hat_unscaled,  # tensor [bs x forecast_len x input_size]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )


@dataclass
class PatchTSMixerForClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForClassificationOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, n_classes)`):
            Prediction output from the classfication head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForClassification(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for classification application.

    Args:
        config (`PatchTSMixerConfig`, *mandatory*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = LinearHead(
            num_patches=config.num_patches,
            in_channels=config.input_size,
            num_features=config.num_features,
            head_dropout=config.head_dropout,
            output_dim=config.n_classes,
            output_range=config.output_range,
            head_agg=config.head_agg,
            mode=config.mode,
        )
        self.loss = torch.nn.CrossEntropyLoss()

        if config.scaling in ["std", "mean", True]:
            if config.mode == "flatten":
                raise ValueError("Scaling is not supported for classification task when mode == flatten")
            self.inject_scale = InjectScalerStatistics4D(
                num_features=config.num_features, num_patches=config.num_patches
            )
        else:
            self.inject_scale = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        context_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForClassificationOutput:
        r"""

        Returns:

        """

        model_output = self.model(
            context_values,
            output_hidden_states=output_hidden_states,
        )  # x: [bs x nvars x num_patch x num_features]

        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state, loc=model_output.loc, scale=model_output.scale
            )  # x: [bs x nvars x num_patch x num_features]

        y_hat = self.head(model_output.last_hidden_state)  # tensor [bs x n_labels]

        if target_values is not None and return_loss is True:
            loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForClassificationOutput(
            prediction_logits=y_hat,  # tensor [bs x n_labels]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )


@dataclass
class PatchTSMixerForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSMixerForRegressionOutput`].

    Args:
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, n_targets)`):
            Prediction output from the regression head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, input_size, num_patches, num_features)`):
            Backbone embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
    """

    prediction_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loss: Optional[torch.FloatTensor] = None


class PatchTSMixerForRegression(PatchTSMixerPreTrainedModel):
    r"""
    `PatchTSMixer` for regression application.

    Args:
        config (`PatchTSMixerConfig`, *mandatory*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__(config)

        self.model = PatchTSMixerModel(config)
        self.head = LinearHead(
            num_patches=config.num_patches,
            in_channels=config.input_size,
            num_features=config.num_features,
            head_dropout=config.head_dropout,
            output_dim=config.n_targets,
            output_range=config.output_range,
            head_agg=config.head_agg,
            mode=config.mode,
        )
        self.loss = torch.nn.MSELoss(reduction="mean")

        if config.scaling in ["std", "mean", True]:
            if config.mode == "flatten":
                raise Exception("Scaling is not supported for classification task when mode == flatten")
            self.inject_scale = InjectScalerStatistics4D(
                num_features=config.num_features, num_patches=config.num_patches
            )
        else:
            self.inject_scale = None

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(PATCHTSMIXER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=PatchTSMixerForRegressionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        context_values: torch.Tensor,
        target_values: torch.Tensor = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
    ) -> PatchTSMixerForRegressionOutput:
        r"""

        Returns:

        """

        # context_values: tensor [bs x seq_len x input_size]
        # target_values: tensor [bs x n_targets]

        model_output = self.model(
            context_values,
            output_hidden_states=output_hidden_states,
        )  # model_output: [bs x nvars x num_patch x num_features]

        if self.inject_scale is not None:
            model_output.last_hidden_state = self.inject_scale(
                model_output.last_hidden_state, loc=model_output.loc, scale=model_output.scale
            )  # x: [bs x nvars x num_patch x num_features]

        y_hat = self.head(model_output.last_hidden_state)  # tensor [bs x n_targets]

        if target_values is not None and return_loss is True:
            loss_val = self.loss(y_hat, target_values)
        else:
            loss_val = None

        return PatchTSMixerForRegressionOutput(
            prediction_logits=y_hat,  # tensor [bs x n_targets]
            last_hidden_state=model_output.last_hidden_state,  # x: [bs x nvars x num_patch x num_features]
            hidden_states=model_output.hidden_states,
            loss=loss_val,
        )
