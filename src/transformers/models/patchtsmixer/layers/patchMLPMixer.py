__all__ = ["PatchMLPMixer"]

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch.nn import functional as F
import logging
import sys
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import inspect
from collections import OrderedDict
from .basics import *
from .gated_attention import GatedAttention
from .norm import NormLayer
from .mixutils import get_class_params_via_inspect


# try:
#     from tsfm.models.layers.shift_attention import TSShiftBlock
#     from tsfm.models.layers.forecast_channel_mixer import ForecastChannelMixer
#     from tsfm.models.layers.hierarchy_tuner import HierarchyPredictionTuner
#     from tsfm.models.layers.hierarchy_tuner import HierarchyPretrainTuner
#     from tsfm.models.layers.heads import ForecastExogHead
#     from tsfm.models.layers.basics import check_forecast_masks
# except:
#     print("skipping tsfm imports")
# from tsfm.utils import get_class_params

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(
        self, in_features, out_features, expansion_factor, dropout, last_dropout=True
    ):
        super().__init__()
        num_hidden = in_features * expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.last_dropout = last_dropout
        if last_dropout:
            self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.fc2(x)
        if self.last_dropout:
            x = self.dropout2(x)
        return x


# class SGU(nn.Module):
#     """SGU (Not preferred to use.)"""

#     def __init__(self, d_ffn):
#         super().__init__()
#         self.norm = nn.LayerNorm(d_ffn)
#         self.proj = nn.Linear(d_ffn, d_ffn)
#         nn.init.constant_(self.proj.bias, 1.0)

#     def forward(self, x):
#         # print(x.shape)
#         u, v = x.chunk(2, dim=-1)
#         # print(u.shape)
#         # print(v.shape)

#         v = self.norm(v)
#         v = self.proj(v)
#         out = u * v
#         return out


# class gMLP(nn.Module):
#     """gMLP (Not preferred to use.)"""

#     def __init__(self, in_features, out_features, expansion_factor, dropout):
#         super().__init__()
#         num_hidden = in_features * expansion_factor
#         self.norm = nn.LayerNorm(in_features)
#         self.fc1 = nn.Linear(in_features, num_hidden * 2)
#         self.dropout1 = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(num_hidden, out_features)
#         self.dropout2 = nn.Dropout(dropout)
#         self.sgu = SGU(d_ffn=num_hidden)

#     def forward(self, x):
#         # residual = x
#         # x = self.norm(x)
#         x = self.dropout1(F.gelu(self.fc1(x)))
#         x = self.sgu(x)
#         x = self.dropout2(self.fc2(x))
#         # x = x + residual
#         return x


# class ChannelTokenMixer(nn.Module):
#     def __init__(
#         self,
#         num_features,
#         num_patches,
#         in_channels,
#         expansion_factor,
#         dropout,
#         mode,
#         gated_attn=False,
#         ffn="mlp",  # mlp, gmlp
#     ):
#         super().__init__()
#         self.mode = mode
#         self.norm = nn.LayerNorm(num_features)
#         if ffn == "gmlp":
#             self.mlp = gMLP(in_channels, in_channels, expansion_factor, dropout)
#         else:
#             self.mlp = MLP(in_channels, in_channels, expansion_factor, dropout)

#         self.gated_attn = gated_attn
#         if gated_attn:
#             self.attn_layer = nn.Linear(in_channels, in_channels)
#             self.attn_softmax = nn.Softmax(dim=-1)

#     def forward(self, x):
#         # x.shape == (batch_size, n_vars, num_patches, num_features)
#         residual = x
#         x = self.norm(x)

#         # x.shape == (batch_size, num_features, num_patches, n_vars)
#         x = x.transpose(1, 3)

#         x = self.mlp(x)

#         if self.gated_attn:
#             attn_weight = self.attn_softmax(self.attn_layer(x))
#             x = x * attn_weight

#         # x.shape == (batch_size, n_vars, num_patches, num_features)
#         x = x.transpose(1, 3)

#         out = x + residual
#         return out


class ChannelFeatureMixer(nn.Module):
    """ChannelFeatureMixer

    Args:
        num_features (int, optional): Hidden feature size. Defaults to 16.
        in_channels (int): Number of input channels. Defaults to 3.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
    """

    def __init__(
        self,
        num_features: int = 16,
        in_channels: int = 3,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = False,
        ffn: str = "mlp",
        norm_mlp="LayerNorm",
    ):
        super().__init__()
        self.mode = mode
        self.norm = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)
        if ffn == "gmlp":
            raise Exception("Use ffn = mlp. gmlp is not preferred")
            # self.mlp = gMLP(in_channels, in_channels, expansion_factor, dropout)
        else:
            self.mlp = MLP(in_channels, in_channels, expansion_factor, dropout)

        self.gated_attn = gated_attn
        if gated_attn:
            self.gab = GatedAttention(in_size=in_channels, out_size=in_channels)

    def forward(self, x):
        # x.shape == (batch_size, n_vars, num_patches, num_features)
        residual = x
        x = self.norm(x)

        # x.shape == (batch_size, num_features, num_patches, n_vars)
        x = x.permute(0, 3, 2, 1)

        if self.gated_attn:
            x = self.gab(x)

        x = self.mlp(x)

        # x.shape == (batch_size, n_vars, num_patches, num_features)
        x = x.permute(0, 3, 2, 1)

        out = x + residual
        return out


class PatchMixer(nn.Module):
    """PatchMixer

    Args:
        num_features (int, optional): Hidden feature size. Defaults to 16.
        num_patches (int): Number of patches to segment
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
    """

    def __init__(
        self,
        num_patches,
        num_features=16,
        expansion_factor=2,
        dropout=0.2,
        mode="common_channel",
        gated_attn=False,
        ffn="mlp",
        self_attn=False,
        self_attn_heads=1,
        norm_mlp="LayerNorm",
    ):
        super().__init__()
        if ffn != "mlp":
            raise Exception("only mlp ffn is allowed.")
        self.norm_mlp = norm_mlp
        self.mode = mode
        self.norm = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)

        self.self_attn = self_attn
        if ffn == "gmlp":
            raise Exception("Use ffn = mlp. gmlp is not preferred")
            # self.mlp = gMLP(num_patches, num_patches, expansion_factor, dropout)
        else:
            self.mlp = MLP(num_patches, num_patches, expansion_factor, dropout)

        self.gated_attn = gated_attn
        if gated_attn:
            self.gab = GatedAttention(in_size=num_patches, out_size=num_patches)

        if self_attn:
            self.self_attn_layer = MultiheadAttention(
                d_model=num_features,
                n_heads=self_attn_heads,
                attn_dropout=dropout,
                proj_dropout=dropout,
            )
            self.norm_attn = NormLayer(
                norm_mlp=norm_mlp, mode=mode, num_features=num_features
            )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features) if common_channel
        residual = x

        x = self.norm(x)

        if self.self_attn:
            x_tmp = x
            if self.mode in ["common_channel", "mix_channel"]:
                x_tmp = torch.reshape(
                    x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
                )
                #  (batch_size, num_patches, num_features) if flatten
                #  (batch_size, n_vars, num_patches, num_features) if common_channel

            x_attn, _ = self.self_attn_layer(x_tmp)

            if self.mode in ["common_channel", "mix_channel"]:
                x_attn = torch.reshape(
                    x_attn, (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                )
                #  (batch_size, num_patches, num_features) if flatten
                #  (batch_size, n_vars, num_patches, num_features) if common_channel

        # Transpose so that num_patches is the last dimension
        if self.mode == "flatten":
            x = x.transpose(1, 2)
        elif self.mode in ["common_channel", "mix_channel"]:
            x = x.transpose(2, 3)

        # x.shape == (batch_size, num_features, num_patches) if flatten
        # x.shape == (batch_size, n_vars, num_features, num_patches) if common_channel

        # if self.gated_attn:
        #     x = self.gab(x)

        x = self.mlp(x)

        if self.gated_attn:
            x = self.gab(x)

        # if self.gated_attn:
        #     attn_weight = self.attn_softmax(self.attn_layer(x))
        #     x = x * attn_weight

        # Transpose back
        if self.mode == "flatten":
            x = x.transpose(1, 2)
        elif self.mode in ["common_channel", "mix_channel"]:
            x = x.transpose(2, 3)

        # x.shape == (batch_size, num_patches, num_features) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        if self.self_attn:
            x = self.norm_attn(x + x_attn)

        out = x + residual
        return out


class FeatureMixer(nn.Module):
    """FeatureMixer

    Args:
        num_features (int, optional): Hidden feature size. Defaults to 16.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
    """

    def __init__(
        self,
        num_features=16,
        expansion_factor=2,
        dropout=0.2,
        gated_attn=False,
        ffn="mlp",  # mlp, gmlp
        mode="common_channel",
        norm_mlp="LayerNorm",
    ):
        super().__init__()
        self.norm_mlp = norm_mlp
        self.mode = mode
        self.norm = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)

        if ffn == "gmlp":
            raise Exception("Use ffn = mlp. gmlp is not preferred")
            # self.mlp = gMLP(num_features, num_features, expansion_factor, dropout)
        else:
            self.mlp = MLP(num_features, num_features, expansion_factor, dropout)

        self.gated_attn = gated_attn

        if self.gated_attn:
            self.gab = GatedAttention(in_size=num_features, out_size=num_features)

        if ffn != "mlp":
            raise Exception("only mlp ffn is allowed.")

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        residual = x

        x = self.norm(x)

        # if self.gated_attn:
        #     x = self.gab(x)

        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        if self.gated_attn:
            x = self.gab(x)

        out = x + residual
        return out


class GatedMixerLayer(nn.Module):

    """GatedMLPMixer: Follows the gMLP architecture (https://arxiv.org/pdf/2105.08050.pdf)

    Args:
        num_features (int, optional): Hidden feature size. Defaults to 16.
        num_patches (int): Number of patches to segment
        in_channels (int, optional): Number of input variables. Defaults to 3.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
    """

    def __init__(
        self,
        num_patches,
        num_features: int = 16,
        in_channels: int = 3,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = False,
        ffn: str = "mlp",  # mlp, gmlp
        self_attn: bool = False,
        self_attn_heads: int = 1,
        norm_mlp: str = "LayerNorm",
    ):
        super().__init__()

        self.mode = mode
        self.num_features = num_features
        self.num_patches = num_patches
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads

        self.norm1 = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)
        self.norm2 = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)

        self.feature_proj1 = nn.Linear(num_features, 2 * num_features)
        self.feature_proj2 = nn.Linear(num_features, num_features)
        self.spatial_proj = nn.Linear(num_patches, num_patches)

        nn.init.constant_(self.spatial_proj.bias, 1.0)
        nn.init.constant_(self.spatial_proj.weight, 0.0)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.gated_attn = gated_attn
        if gated_attn:
            self.spatial_gab = GatedAttention(in_size=num_patches, out_size=num_patches)
            self.feature_gab = GatedAttention(
                in_size=num_features * 2, out_size=num_features * 2
            )

        if self_attn:
            self.self_attn_layer = MultiheadAttention(
                d_model=num_features,
                n_heads=self_attn_heads,
                attn_dropout=dropout,
                proj_dropout=dropout,
            )

            self.norm_attn = NormLayer(
                norm_mlp=norm_mlp, mode=mode, num_features=num_features
            )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        residual = x

        x = self.norm1(x)

        x = self.dropout1(F.gelu(self.feature_proj1(x)))
        # x.shape == (batch_size, num_patches, num_features*2) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features*2) if common_channel

        if self.gated_attn:
            x = self.feature_gab(x)

        x1, x2 = x.chunk(2, dim=-1)
        # x1, x2 shape: (batch_size, n_vars, num_patches, num_features)

        x2 = self.norm2(x2)  # (batch_size, n_vars, num_patches, num_features)

        # x2 = self.layer_norm2(x2)  # (batch_size, n_vars, num_patches, num_features)

        if self.self_attn:
            x2_tmp = x2
            if self.mode in ["common_channel", "mix_channel"]:
                x2_tmp = torch.reshape(
                    x2, (x2.shape[0] * x2.shape[1], x2.shape[2], x2.shape[3])
                )  # (batch_size * n_vars, num_patches, num_features)

            x2_attn, _ = self.self_attn_layer(x2_tmp)

            if self.mode in ["common_channel", "mix_channel"]:
                x2_attn = torch.reshape(
                    x2_attn, (x2.shape[0], x2.shape[1], x2.shape[2], x2.shape[3])
                )  # (batch_size, n_vars, num_patches, num_features)

        if self.mode == "flatten":
            x2 = x2.transpose(1, 2)  # (batch_size, num_features, num_patches)
        elif self.mode in ["common_channel", "mix_channel"]:
            x2 = x2.transpose(2, 3)  # (batch_size, n_vars, num_features, num_patches)

        x2 = self.dropout2(
            self.spatial_proj(x2)
        )  # (batch_size, n_vars, num_features, num_patches)

        if self.gated_attn:
            x2 = self.spatial_gab(x2)

        if self.mode == "flatten":
            x2 = x2.transpose(1, 2)  # (batch_size, num_patches, num_features)
        elif self.mode in ["common_channel", "mix_channel"]:
            x2 = x2.transpose(2, 3)  # (batch_size, n_vars, num_patches, num_features)

        if self.self_attn:
            x2 = self.norm_attn(x2 + x2_attn)

        x = x1 * x2  # (batch_size, n_vars, num_patches, num_features)

        x = self.dropout3(
            self.feature_proj2(x)
        )  # (batch_size, n_vars, num_patches, num_features)

        x = x + residual  # (batch_size, n_vars, num_patches, num_features)

        # (batch_size, n_vars, num_patches, num_features)
        return x


class SwinMixerBlock(nn.Module):
    """MLPMixerLayer follows the MLP-Mixer architecture (https://arxiv.org/abs/2105.01601)

    Args:
        num_features (int, optional): Hidden feature size. Defaults to 16.
        num_patches (int): Number of patches to segment
        in_channels (int, optional): Number of input variables. Defaults to 3.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
        swin_level (int, optional): swin level. Defaults to 1

    """

    def __init__(
        self,
        num_patches: int,
        num_features: int = 16,
        in_channels: int = 3,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = False,
        ffn: str = "mlp",  # mlp, gmlp
        self_attn: bool = False,
        self_attn_heads: int = 1,
        norm_mlp="LayerNorm",
        swin_level=1,
        mixer_type: str = "base",  # base, gated
        num_layers: int = 8,
        shift_segment_len: int = 8,
        shift_attention: bool = False,
    ):
        super().__init__()
        mix_params = {}

        if mode == "flatten":
            raise Exception("SwinMixerBlock is not enabled when mode is flatten")
        if mixer_type == "base":
            mixer_class = (
                MixerLayer  # follow MLP-Mixer archi https://arxiv.org/abs/2105.01601
            )
        elif mixer_type == "gated":
            mixer_class = GatedMixerLayer  # follow gMLP archi https://arxiv.org/pdf/2105.08050.pdf

        elif mixer_type == "shift":
            mixer_class = (
                TSShiftBlock  # follow s2 MLP v2: https://arxiv.org/abs/2108.01072
            )
            mix_params["shift_segment_len"] = shift_segment_len
            mix_params["shift_attention"] = shift_attention
            # if num_patches%2 != 0 or num_patches%shift_segment_len !=0:
            #     raise Exception("For shift attention, num_patches should be even and divisible by shift_segment_len")

        self.swin_level = swin_level

        self.swin_factor = 2**swin_level

        if num_features % self.swin_factor != 0:
            raise Exception("num_features%(2 ** swin_level) should be zero")

        self.swin_mixers = nn.Sequential(
            *[
                mixer_class(
                    num_patches=num_patches * self.swin_factor,
                    num_features=num_features // self.swin_factor,
                    in_channels=in_channels,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                    mode=mode,
                    gated_attn=gated_attn,
                    ffn=ffn,
                    self_attn=self_attn,
                    self_attn_heads=self_attn_heads,
                    norm_mlp=norm_mlp,
                    **mix_params,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # input:
        # x.shape == (batch_size, num_patches, num_features) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        # after reshape
        #  x.shape == (batch_size, num_patches * swin_factor, num_features/swin_factor) if flatten
        # x.shape == (batch_size, n_vars, num_patches*swin_factor, num_features/swin_factor) if common_channel

        x = torch.reshape(
            x,
            (
                x.shape[0],
                x.shape[1],
                x.shape[2] * self.swin_factor,
                x.shape[3] // self.swin_factor,
            ),
        )

        #  x.shape == (batch_size, num_patches * swin_factor, num_features/swin_factor) if flatten
        # x.shape == (batch_size, n_vars, num_patches*swin_factor, num_features/swin_factor) if common_channel
        x = self.swin_mixers(x)

        # x.shape == (batch_size, num_patches, num_features) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features) if common_channel
        x = torch.reshape(
            x,
            (
                x.shape[0],
                x.shape[1],
                x.shape[2] // self.swin_factor,
                x.shape[3] * self.swin_factor,
            ),
        )

        return x


class MixerLayer(nn.Module):
    """MLPMixerLayer follows the MLP-Mixer architecture (https://arxiv.org/abs/2105.01601)

    Args:
        num_features (int, optional): Hidden feature size. Defaults to 16.
        num_patches (int): Number of patches to segment
        in_channels (int, optional): Number of input variables. Defaults to 3.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.

    """

    def __init__(
        self,
        num_patches: int,
        num_features: int = 16,
        in_channels: int = 3,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = False,
        ffn: str = "mlp",  # mlp, gmlp
        self_attn: bool = False,
        self_attn_heads: int = 1,
        norm_mlp="LayerNorm",
    ):
        super().__init__()
        self.patch_mixer = PatchMixer(
            num_patches=num_patches,
            num_features=num_features,
            expansion_factor=expansion_factor,
            dropout=dropout,
            mode=mode,
            gated_attn=gated_attn,
            ffn=ffn,
            self_attn=self_attn,
            self_attn_heads=self_attn_heads,
            norm_mlp=norm_mlp,
        )
        self.feature_mixer = FeatureMixer(
            num_features=num_features,
            expansion_factor=expansion_factor,
            dropout=dropout,
            gated_attn=gated_attn,
            ffn=ffn,
            mode=mode,
            norm_mlp=norm_mlp,
        )
        # define a cross series mixer

        self.mode = mode
        if mode == "mix_channel":
            self.channel_feature_mixer = ChannelFeatureMixer(
                num_features=num_features,
                in_channels=in_channels,
                expansion_factor=expansion_factor,
                dropout=dropout,
                mode=mode,
                gated_attn=gated_attn,
                ffn=ffn,
            )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        if self.mode == "mix_channel":
            # x = self.channel_token_mixer(x)
            x = self.channel_feature_mixer(x)

        x = self.patch_mixer(x)
        x = self.feature_mixer(x)

        # x.shape == (batch_size, num_patches, num_features)
        return x


# def backbone_match_params(func):
#     @wraps(func)
#     def wrapper(self, *args, **kwargs):
#         self.backbone_match_params = {}
#         self.backbone_match_params.update(kwargs)
#         self.backbone_match_params.update(args)
#         return func(self, *args, **kwargs)

#     return wrapper


class MLPMixer(nn.Module):
    """MLPMixer

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        num_layers (int, optional): Number of layers to use. Defaults to 8.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        use_pe (bool, optional): Use positional embedding. Defaults to False.
        pe (str, optional): Type of positional embedding to use. Defaults to "zeros".
        learn_pe (bool, optional): Make positional embedding learnable. Defaults to True.
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        beats (bool, optional): Enable backcast and forecast flows like nbeats (not preferred). Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        mixer_type (str, optional): Mixer Type to use. Allowed values are base, gated.
            base follows the MLP-Mixer architecture (https://arxiv.org/abs/2105.01601)
            gated follows the gMLP architecture (https://arxiv.org/pdf/2105.08050.pdf) Defaults to "base".
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
        swin_hier (int, optional): swin hier levels. If swin_hier is i, then we will have i levels with each level having n_layers.
        Level id starts with 0. num_patches at level i will be multipled by (2^i) and num_features at level i will be divided by (2^i).
        For Ex. if swin_hier is 3 - then we will have 3 levels:
            level 2: num_features//(2^2), num_patches*(2^2)
            level 1: num_features//(2^1), num_patches*(2^1)
            level 0: num_features//(2^0), num_patches*(2^0)
        swin_hier = 1 is same as one level mlp_mixer. This module gets disabled when swin_hier is 0 or neg value. Defaults to 0 (off mode).
        shift_segment_len (int, optional): Segment length to use when base type is shift. Default to 8.
        shift_attention(bool, optional): Enable attention when base type is shift. Defaults to False.
    """

    # @get_class_params
    def __init__(
        self,
        num_patches: int,
        patch_size: int = 16,
        in_channels: int = 3,
        num_features: int = 128,
        expansion_factor: int = 2,
        num_layers: int = 8,
        dropout: float = 0.5,
        mode: str = "common_channel",
        use_pe: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        gated_attn: bool = False,
        beats: bool = False,
        ffn: str = "mlp",  
        self_attn: bool = False,
        self_attn_heads: int = 1,
        mixer_type: str = "base",
        norm_mlp="LayerNorm",
        swin_hier: int = 0,
        shift_segment_len: int = 8,
        shift_attention: bool = False,
    ):
        # save class params which gets autovalidated during transfer weights.
        # prevents finetuning backbone with different parameters not used in the backbone.
        try:
            # work around as currentframe() behavior varies in cython mode
            input_frame = inspect.currentframe()
            self.class_params = get_class_params_via_inspect(input_frame)
            # remove in_channel from self.backbone_match_params as it can vary across backbone.
            # other params should match
            self.backbone_match_params = {}
            self.backbone_match_params.update(self.class_params)
            del self.backbone_match_params["in_channels"]
        except:
            self.backbone_match_params = {}
            self.class_params = {}
            logger.warning(
                "Inspect.currentframe is not working as expected! Manually ensure pre-train and fine-tune have the same backbone parameters."
            )

        super().__init__()

        if beats:
            raise Exception(
                "beats: True is not allowed due to poor performance. Set it to False"
            )
        if use_pe is True:
            raise Exception("set use_pe to False")
        
        if ffn != "mlp":
            raise Exception("Set ffn to mlp")
            
        self.mode = mode
        self.swin_hier = swin_hier

        if self.swin_hier > 0:
            if (2**self.swin_hier) > num_features:
                raise Exception("2^swin_hier should not be greater than num_features")

        if mode == "flatten":
            self.patcher = nn.Linear(in_channels * patch_size, num_features)

        elif mode in ["common_channel", "mix_channel"]:
            self.patcher = nn.Linear(patch_size, num_features)

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_layers = num_layers
        self.use_pe = use_pe

        if use_pe:
            self.W_pos = positional_encoding(pe, learn_pe, num_patches, num_features)

        self.mlp_mixer_encoder = MLPMixerEncoder(
            num_patches=num_patches,
            patch_size=patch_size,
            in_channels=in_channels,
            num_features=num_features,
            expansion_factor=expansion_factor,
            num_layers=num_layers,
            dropout=dropout,
            mode=mode,
            gated_attn=gated_attn,
            beats=beats,
            ffn=ffn,
            self_attn=self_attn,
            self_attn_heads=self_attn_heads,
            mixer_type=mixer_type,
            norm_mlp=norm_mlp,
            swin_hier=swin_hier,
            shift_segment_len=shift_segment_len,
            shift_attention=shift_attention,
        )

    def forward(self, x):
        # x: [bs  x n_vars x num_patch x patch_len]

        batch_size = x.shape[0]
        logger.debug(x.shape)

        if self.mode == "flatten":
            x = x.permute(0, 2, 1, 3)  # x: [bs  x num_patch x n_vars  x patch_len]
            x = torch.reshape(
                x, (batch_size, self.num_patches, self.in_channels * self.patch_size)
            )  # x: [bs x num_patch x patch_len * n_vars]

        # elif self.mode in ["common_channel", "mix_channel"]:
        #     x = x.permute(0, 2, 1, 3)  # x: [bs x n_vars x num_patch x patch_len]

        logger.debug(x.shape)
        patches = self.patcher(
            x
        )  # flatten: [bs x num_patch x num_features]   common_channel/mix_channel: [bs x n_vars x num_patch x num_features]

        logger.debug(x.shape)

        if self.use_pe:
            patches = patches + self.W_pos

        embedding = self.mlp_mixer_encoder(patches)

        logger.debug(x.shape)
        # embedding.shape == (batch_size, num_patches, num_features) if flatten
        # embedding.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        return embedding


class MLPMixerEncoder(nn.Module):
    """MLPMixerEncoder

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        num_layers (int, optional): Number of layers to use. Defaults to 8.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        use_pe (bool, optional): Use positional embedding. Defaults to False.
        pe (str, optional): Type of positional embedding to use. Defaults to "zeros".
        learn_pe (bool, optional): Make positional embedding learnable. Defaults to True.
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        beats (bool, optional): Enable backcast and forecast flows like nbeats (not preferred). Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        mixer_type (str, optional): Mixer Type to use. Allowed values are base, gated.
            base follows the MLP-Mixer architecture (https://arxiv.org/abs/2105.01601)
            gated follows the gMLP architecture (https://arxiv.org/pdf/2105.08050.pdf) Defaults to "base".
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
        swin_hier (int, optional): swin hier levels. If swin_hier is i, then we will have i levels with each level having n_layers.
        Level id starts with 0. num_patches at level i will be multipled by (2^i) and num_features at level i will be divided by (2^i).
        For Ex. if swin_hier is 3 - then we will have 3 levels:
            level 2: num_features//(2^2), num_patches*(2^2)
            level 1: num_features//(2^1), num_patches*(2^1)
            level 0: num_features//(2^0), num_patches*(2^0)
        swin_hier = 1 is same as one level mlp_mixer. This module gets disabled when swin_hier is 0 or neg value. Defaults to 0 (off mode).
        shift_segment_len (int, optional): Segment length to use when base type is shift. Default to 8.
        shift_attention(bool, optional): Enable attention when base type is shift. Defaults to False.
    """

    # @get_class_params
    def __init__(
        self,
        num_patches: int,
        patch_size: int = 16,
        in_channels: int = 3,
        num_features: int = 128,
        expansion_factor: int = 2,
        num_layers: int = 8,
        dropout: float = 0.5,
        mode: str = "common_channel",  # common_channel, flatten
        gated_attn: bool = False,
        beats: bool = False,
        ffn: str = "mlp",  # mlp, gmlp
        self_attn: bool = False,
        self_attn_heads: int = 1,
        mixer_type: str = "base",  # base, gated
        norm_mlp="LayerNorm",
        swin_hier: int = 0,
        shift_segment_len: int = 8,
        shift_attention: bool = False,
    ):
        try:
            # work around as currentframe() behavior varies in cython mode
            input_frame = inspect.currentframe()
            self.class_params = get_class_params_via_inspect(input_frame)
            if "in_channels" not in self.class_params:
                raise Exception("")
        except:
            self.class_params = {}
            logger.warning(
                "Inspect.currentframe is not working as expected! Manually ensure pre-train and fine-tune have the same backbone parameters."
            )

        super().__init__()
        self.mode = mode
        self.swin_hier = swin_hier

        if self.swin_hier > 0:
            if (2**self.swin_hier) > num_features:
                raise Exception("2^swin_hier should not be greater than num_features")

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_layers = num_layers

        self.beats = beats

        mix_params = {}
        if mixer_type == "base":
            mixer_class = (
                MixerLayer  # follow MLP-Mixer archi https://arxiv.org/abs/2105.01601
            )
        elif mixer_type == "gated":
            mixer_class = GatedMixerLayer  # follow gMLP archi https://arxiv.org/pdf/2105.08050.pdf
        elif mixer_type == "shift":
            mixer_class = (
                TSShiftBlock  # follow s2 MLP v2: https://arxiv.org/abs/2108.01072
            )
            mix_params["shift_segment_len"] = shift_segment_len
            mix_params["shift_attention"] = shift_attention
            # if num_patches%2 != 0 or num_patches%shift_segment_len !=0:
            #     raise Exception("For shift attention, num_patches should be even and divisible by shift_segment_len")

        if beats:
            # TO BE IMPROVED
            # architecture inspired from nbeats
            # not performing good as expected.
            self.mixers = nn.ModuleList()
            self.backcast = nn.ModuleList()
            self.forecast = nn.ModuleList()

            for i in range(num_layers):
                self.mixers.append(
                    mixer_class(
                        num_patches=num_patches,
                        num_features=num_features,
                        in_channels=in_channels,
                        expansion_factor=expansion_factor,
                        dropout=dropout,
                        mode=mode,
                        gated_attn=gated_attn,
                        ffn=ffn,
                        self_attn=self_attn,
                        self_attn_heads=self_attn_heads,
                        norm_mlp=norm_mlp,
                        **mix_params,
                    )
                )

                self.backcast.append(nn.Linear(num_features, num_features))
                self.forecast.append(nn.Linear(num_features, num_features))

        elif swin_hier > 0:
            self.mixers = nn.Sequential(
                *[
                    SwinMixerBlock(
                        num_patches=num_patches,
                        num_features=num_features,
                        in_channels=in_channels,
                        expansion_factor=expansion_factor,
                        dropout=dropout,
                        mode=mode,
                        gated_attn=gated_attn,
                        ffn=ffn,
                        self_attn=self_attn,
                        self_attn_heads=self_attn_heads,
                        norm_mlp=norm_mlp,
                        swin_level=i,
                        mixer_type=mixer_type,
                        num_layers=num_layers,
                        **mix_params,
                    )
                    for i in reversed(range(swin_hier))
                ]
            )

        else:
            self.mixers = nn.Sequential(
                *[
                    mixer_class(
                        num_patches=num_patches,
                        num_features=num_features,
                        in_channels=in_channels,
                        expansion_factor=expansion_factor,
                        dropout=dropout,
                        mode=mode,
                        gated_attn=gated_attn,
                        ffn=ffn,
                        self_attn=self_attn,
                        self_attn_heads=self_attn_heads,
                        norm_mlp=norm_mlp,
                        **mix_params,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(self, x):
        # flatten: [bs x num_patch x num_features]   common_channel/mix_channel: [bs x n_vars x num_patch x num_features]

        batch_size = x.shape[0]
        logger.debug(x.shape)

        patches = x
        if not self.beats:
            embedding = self.mixers(patches)
        else:
            # Nbeats style flow.
            # filter and pass only residual
            # not working well, as I am doing the residual and aggregation at embedding level and not at the
            # actual forecast level (as implemented in Nbeats)
            embedding = None
            for i in range(self.num_layers):
                patches = self.mixers[i](patches)
                backcast_patch = self.backcast[i](patches)
                forecast_patch = self.forecast[i](patches)

                patches = patches - backcast_patch
                if embedding is None:
                    embedding = forecast_patch
                else:
                    embedding = embedding + forecast_patch

        logger.debug(x.shape)
        # embedding.shape == (batch_size, num_patches, num_features) if flatten
        # embedding.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        return embedding


# Cell


class PatchMLPMixer(nn.Module):
    """
    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """

    """PatchMLPMixer

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        num_layers (int, optional): Number of layers to use. Defaults to 8.
        dropout (float, optional): Backbone Dropout rate. Defaults to 0.2.
        head_dropout (float, optional): Head Dropout rate. Defaults to 0.2.
        forecast_len (int, optional): Forecast Length. Defaults to 16.
        head_type (str, optional): Head Type to use. Allowed values are prediction, pretrain. Defaults to "prediction".
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten, 
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        use_pe (bool, optional): Use positional embedding. Defaults to False.
        pe (str, optional): Type of positional embedding to use. Defaults to "zeros".
        learn_pe (bool, optional): Make positional embedding learnable. Defaults to True.
        time_hierarchy (bool, optional): Enable time hierarchy in heads. Defaults to False.
        teacher_forcing (bool, optional): Enable teacher forcing in heads while using teacher forcing. Defaults to True.
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        beats (bool, optional): Enable backcast and forecast flows like nbeats (not preferred). Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        head_attn (bool, optional): Enable gated attention in head. (Not preferred) Defaults to False.
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        mixer_type (str, optional): Mixer Type to use. Allowed values are base, gated. 
            base follows the MLP-Mixer architecture (https://arxiv.org/abs/2105.01601)
            gated follows the gMLP architecture (https://arxiv.org/pdf/2105.08050.pdf) Defaults to "base".
        th_mode (str, optional): Mode to mix the hierarchy signals. Allowed values are plain, reconcile.
            reconcile not preferred based on current experiments. Defaults to "reconcile".
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
        swin_hier (int, optional): swin hier levels. If swin_hier is i, then we will have i levels with each level having n_layers.
        Level id starts with 0. num_patches at level i will be multipled by (2^i) and num_features at level i will be divided by (2^i). 
        For Ex. if swin_hier is 3 - then we will have 3 levels:
            level 2: num_features//(2^2), num_patches*(2^2)
            level 1: num_features//(2^1), num_patches*(2^1)
            level 0: num_features//(2^0), num_patches*(2^0)
        swin_hier = 1 is same as one level mlp_mixer. This module gets disabled when swin_hier is 0 or neg value. Defaults to 0 (off mode).
        forecast_channel_mixing (bool, optional): Enable cross-channel reconcilation head for finetuning. Default to False.
        cm_gated_attn(bool, optional): Enable gated attn in cross-channel reconcilation head. Default to False.
        cm_teacher_forcing(bool, optional): Eanble teacher forcing in cross-channel reconcilation head. Default to False.
        channel_context_length(int, optional): Surronding context length to use. Default to 0.
        shift_segment_len (int, optional): Segment length to use when base type is shift. Default to 8.
        shift_attention(bool, optional): Enable attention when base type is shift. Defaults to False.
        head_agg (str, optional): Aggregation mode. Allowed values are use_last, max_pool, avg_pool and None. 
                                Defaults to None.
        output_range (list, optional): Output range to restrict. Defaults to None.
    """

    # @get_class_params
    def __init__(
        self,
        num_patches: int,
        patch_size: int = 16,
        in_channels: int = 3,
        num_features: int = 16,
        expansion_factor: int = 2,
        num_layers: int = 8,
        dropout: float = 0.2,
        head_dropout: float = 0.2,
        forecast_len: int = 16,
        head_type: str = "prediction",
        mode: str = "common_channel",  # common_channel, flatten, mix_channel
        use_pe: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        time_hierarchy: bool = False,
        teacher_forcing: bool = False,
        gated_attn: bool = False,
        beats: bool = False,
        ffn: str = "mlp",  # mlp, gmlp
        head_attn: bool = False,
        self_attn: bool = False,
        self_attn_heads: bool = 1,
        mixer_type: str = "base",  # base, gated, shift
        th_mode: str = "reconcile",  # plain, reconcile
        norm_mlp="LayerNorm",
        forecast_channel_mixing=False,
        cm_gated_attn=True,
        cm_teacher_forcing=False,
        channel_context_length=0,
        swin_hier=0,
        shift_segment_len: int = 8,
        shift_attention: bool = False,
        output_range: list = None,
        head_agg: str = None,
        forecast_channel_indices: list = None,
        cv_channel_indices: list = None,
        finetune_channel_indices: list = None,
        mask_value: int = 0,
    ):
        try:
            # work around as currentframe() behavior varies in cython mode
            input_frame = inspect.currentframe()
            self.class_params = get_class_params_via_inspect(input_frame)
            if "in_channels" not in self.class_params:
                raise Exception("")
        except:
            self.class_params = {}
            logger.warning(
                "Inspect.currentframe is not working as expected! Manually ensure pre-train and fine-tune have the same backbone parameters."
            )

        super().__init__()

        if finetune_channel_indices is not None:
            raise Exception("Filtering finetune_channel_indices is not enabled in plain mixer. Use PatchMixerMAE or MultiLevelMixerMAE")
        
        assert head_type in [
            "pretrain",
            "prediction",
            "regression",
            "classification",
            "masked_forecast",
        ], "head type should be either masked_forecast, pretrain, prediction, regression, classification, forecast_with_exog"
        # Backbone

        self.sample_input_shape = (1, in_channels, num_patches, patch_size)
        self.head_type = head_type
        self.forecast_len = forecast_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.cv_channel_indices = cv_channel_indices
        self.mask_value = mask_value

        self.backbone = MLPMixer(
            num_patches=num_patches,
            patch_size=patch_size,
            in_channels=in_channels,
            num_features=num_features,
            expansion_factor=expansion_factor,
            num_layers=num_layers,
            dropout=dropout,
            mode=mode,
            use_pe=use_pe,
            pe=pe,
            learn_pe=learn_pe,
            gated_attn=gated_attn,
            beats=beats,
            ffn=ffn,
            self_attn=self_attn,
            self_attn_heads=self_attn_heads,
            mixer_type=mixer_type,
            norm_mlp=norm_mlp,
            swin_hier=swin_hier,
            shift_segment_len=shift_segment_len,
            shift_attention=shift_attention,
        )

        # Head

        if head_type == "prediction":
            self.head = PredictionHead(
                in_channels=in_channels,
                patch_size=patch_size,
                num_features=num_features,
                num_patches=num_patches,
                forecast_len=forecast_len,
                head_dropout=head_dropout,
                mode=mode,
                time_hierarchy=time_hierarchy,
                expansion_factor=expansion_factor,
                teacher_forcing=teacher_forcing,
                head_attn=head_attn,
                head_agg=head_agg,
                th_mode=th_mode,
                forecast_channel_mixing=forecast_channel_mixing,
                cm_gated_attn=cm_gated_attn,
                cm_teacher_forcing=cm_teacher_forcing,
                channel_context_length=channel_context_length,
            )
        elif head_type in ["regression", "classification"]:
            self.head = LinearHead(
                num_patches=num_patches,
                in_channels=in_channels,
                num_features=num_features,
                head_dropout=head_dropout,
                output_dim=forecast_len,
                output_range=output_range,
                head_agg=head_agg,
                mode=mode,
            )

        elif head_type == "pretrain":
            self.head = PretrainHead(
                num_features=num_features,
                num_patches=num_patches,
                in_channels=in_channels,
                patch_size=patch_size,
                head_dropout=head_dropout,
                mode=mode,
                time_hierarchy=time_hierarchy,
                teacher_forcing=teacher_forcing,
                th_mode=th_mode,
            )
        elif head_type == "masked_forecast":
            self.head = ForecastExogHead(num_patches=num_patches,
                        in_channels=in_channels,
                        patch_size=patch_size,
                        num_features=num_features,
                        forecast_len=forecast_len,
                        forecast_channel_indices=forecast_channel_indices,
                        cv_channel_indices = cv_channel_indices,
                        head_dropout=head_dropout,
                        forecast_channel_mixing=forecast_channel_mixing,
                        cm_gated_attn=cm_gated_attn,
                        cm_teacher_forcing=cm_teacher_forcing,
                        channel_context_length=channel_context_length,
                        d_size="3D",
                        )
        else:
            raise Exception("To be implemented..")

    def forward(self, z, y=None, cv_data = None):
        """
        z: tensor [bs  x n_vars x num_patch x patch_len]
        cv_data: [bs x forecast_len x nvars
        prediction head:
            time_hierarchy enabled (teacher forcing during training):
                y: ([bs x forecast_patches x nvars],[bs x forecast_len x nvars]) # y_hier, actual_y
            no time hierarchy:
                y: None
        pretrain head:
            time_hierarchy enabled (teacher forcing during training):
                y: ([bs x num_patch x n_vars],[bs x num_patch x n_vars x patch_len]) # y_hier, actual_y
            no time hierarchy:
                y: None

        Output:
        prediction head:
            time_hierarchy enabled (teacher forcing during training):
                z: ([bs x forecast_patches x nvars],[bs x forecast_len x nvars]) # y_hier, actual_y
            no time hierarchy:
                z: [bs x forecast_len x nvars]
        pretrain head:
            time_hierarchy enabled (teacher forcing during training):
                z: ([bs x num_patch x n_vars],[bs  x n_vars x num_patch x patch_len]) # y_hier, actual_y
            no time hierarchy:
                z: [bs  x n_vars x num_patch x patch_len]
        """

        logger.debug(z.shape)

        if cv_data is not None and self.cv_channel_indices is not None and len(self.cv_channel_indices)>0:
            cv_data = cv_data[:,:,self.cv_channel_indices]
        
        if self.head_type == "masked_forecast":
            check_forecast_masks(z = z, 
                                forecast_len = self.forecast_len,
                                patch_size = self.patch_size,
                                in_channels = self.in_channels,
                                cv_channel_indices = self.cv_channel_indices,
                                d_size = "4D",
                                mask_value = self.mask_value)
            
        z = self.backbone(
            z
        )  # flatten mode: [bs x num_patch x num_features] or  common_channel/mix_channel mode: [bs x n_vars x num_patch x num_features]
        if self.head_type == "masked_forecast":
            z = self.head(z, y, cv_data = cv_data)
        else:
            z = self.head(z, y)

        return z


class PretrainHead(nn.Module):
    """Pretrain head

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        head_dropout (float, optional): Head Dropout rate. Defaults to 0.2.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        time_hierarchy (bool, optional): Enable time hierarchy in heads. Defaults to False.
        teacher_forcing (bool, optional): Enable teacher forcing in heads while using teacher forcing. Defaults to True.
        th_mode (str, optional): Mode to mix the hierarchy signals. Allowed values are plain, reconcile.
            reconcile not preferred based on current experiments. Defaults to "reconcile".
    """

    def __init__(
        self,
        num_patches: int,
        num_features: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        head_dropout: float = 0,
        mode: str = "common_channel",  # flatten, common_channel
        time_hierarchy: bool = True,
        teacher_forcing: bool = False,
        th_mode: str = "reconcile",  # plain, reconcile
    ):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.time_hierarchy = time_hierarchy
        self.teacher_forcing = teacher_forcing
        self.num_patches = num_patches
        self.th_mode = th_mode

        if self.mode in ["common_channel", "mix_channel"]:
            self.base_forecast_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_size),
            )
        else:
            self.base_forecast_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_size * in_channels),
            )

        if self.time_hierarchy:
            if self.mode not in ["common_channel", "mix_channel"]:
                raise Exception(
                    "Hierarchy Tuner can be enabled only when backbone mode is common_channel or mix_channel"
                )

            self.ht_block = HierarchyPretrainTuner(
                patch_size=patch_size,
                num_features=num_features,
                num_patches=num_patches,
                head_dropout=head_dropout,
                teacher_forcing=teacher_forcing,
                th_mode=th_mode,
                hier_model_type="linear",
            )

    def forward(self, x, y=None):
        """
        # flatten mode: [bs x num_patch x num_features] or
        common_channel/mix_channel mode: [bs x n_vars x num_patch x num_features]

        time_hierarchy enabled (teacher forcing during training):
            y: ([bs x num_patch x n_vars],[bs x num_patch x n_vars x patch_len]) # y_hier, actual_y
        no time hierarchy:
            y: None

        Output:

        time_hierarchy enabled (teacher forcing during training):
            z: ([bs x n_vars x num_patch ],[bs x n_vars x num_patch  x patch_len]) # y_hier, actual_y
        no time hierarchy:
            z: [bs x n_vars x num_patch  x patch_len]

        """

        if self.mode == "flatten":
            x = self.base_forecast_block(x)  # x: [bs x num_patch x n_vars*patch_size]
            x = torch.reshape(
                x, (x.shape[0], x.shape[1], self.patch_size, self.in_channels)
            )  # [bs x num_patch x patch_size x n_vars]
            x = x.permute(0, 3, 1, 2)  # [bs x nvars x num_patch  x patch_len]
            return x
        elif self.mode in ["common_channel", "mix_channel"]:
            forecast = self.base_forecast_block(
                x
            )  # [bs x n_vars x num_patch x patch_size]
            # forecast = forecast.permute(
            #     0, 2, 1, 3
            # )  # [bs x num_patch x nvars x patch_size]

            if self.time_hierarchy:
                h_pred, forecast = self.ht_block(
                    x=x, base_forecast=forecast, y=y
                )  # ([bs x n_vars x num_patch],[bs x n_vars x num_patch  x patch_len])
                return h_pred, forecast
            else:
                return forecast


class PredictionHead(nn.Module):
    """PredictionHead

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        expansion_factor (int, optional): Expansion factor to use inside MLP. Defaults to 2.
        head_dropout (float, optional): Head Dropout rate. Defaults to 0.2.
        forecast_len (int, optional): Forecast Length. Defaults to 16.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        time_hierarchy (bool, optional): Enable time hierarchy in heads. Defaults to False.
        teacher_forcing (bool, optional): Enable teacher forcing in heads while using teacher forcing. Defaults to True.
        head_attn (bool, optional): Enable gated attention in head. (Not preferred) Defaults to False.
        th_mode (str, optional): Mode to mix the hierarchy signals. Allowed values are plain, reconcile.
            reconcile not preferred based on current experiments. Defaults to "reconcile".
        forecast_channel_mixing (bool, optional): Enable cross-channel reconcilation head for finetuning. Default to False.
        cm_gated_attn(bool, optional): Enable gated attn in cross-channel reconcilation head. Default to False.
        cm_teacher_forcing(bool, optional): Eanble teacher forcing in cross-channel reconcilation head. Default to False.
        channel_context_length(int, optional): Surronding context length to use. Default to 0.
        head_agg (str, optional): Aggregation mode. Allowed values are use_last, max_pool, avg_pool and None.
                                Defaults to None.

    """

    def __init__(
        self,
        num_patches: int,
        in_channels: int = 3,
        patch_size: int = 16,
        num_features: int = 16,
        forecast_len: int = 16,
        head_dropout: float = 0.2,
        mode: str = "common_channel",  # flatten, common_channel, mix_channel
        time_hierarchy: bool = True,
        th_mode: str = "reconcile",  # plain, reconcile
        expansion_factor: int = 2,
        teacher_forcing: bool = False,
        head_attn: bool = False,
        head_agg: str = None,
        forecast_channel_mixing=False,
        cm_gated_attn=True,
        cm_teacher_forcing=False,
        channel_context_length=0,
    ):
        super().__init__()
        self.forecast_len = forecast_len
        self.nvars = in_channels
        self.num_features = num_features
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.time_hierarchy = time_hierarchy
        self.mode = mode
        self.expansion_factor = expansion_factor
        self.head_attn = head_attn
        self.teacher_forcing = teacher_forcing
        self.th_mode = th_mode
        self.forecast_channel_mixing = forecast_channel_mixing
        self.cm_gated_attn = cm_gated_attn
        self.cm_teacher_forcing = cm_teacher_forcing
        self.head_agg = head_agg
        if self.head_attn:
            raise Exception(
                "head attn performance is not satisfactory and deprecated. Avoid using it."
            )

        if self.head_agg is not None:
            raise Exception(
                "Head aggregation is disabled in prediction head for forecasting. Please set it to None"
            )

        if self.forecast_channel_mixing:
            if self.mode not in ["common_channel", "mix_channel"]:
                raise Exception(
                    "Forecast channel mixing can be enabled only when backbone mode is common_channel or mix_channel"
                )

            self.fcm = ForecastChannelMixer(
                forecast_channels=in_channels,
                cm_expansion_factor=expansion_factor,
                cm_gated_attn=self.cm_gated_attn,
                cm_dropout=head_dropout,
                cm_teacher_forcing=self.cm_teacher_forcing,
                channel_context_length=channel_context_length,
            )

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

        if self.time_hierarchy:
            if self.mode not in ["common_channel", "mix_channel"]:
                raise Exception(
                    "Hierarchy Tuner can be enabled only when backbone mode is common_channel or mix_channel"
                )

            self.ht_block = HierarchyPredictionTuner(
                forecast_len=forecast_len,
                patch_size=patch_size,
                num_features=num_features,
                num_patches=num_patches,
                head_dropout=head_dropout,
                teacher_forcing=teacher_forcing,
                th_mode=th_mode,
                hier_model_type="linear",
            )

    def forward(self, x, y=None):
        """
        # x: [bs x num_patch x num_features] flatten mode or
            [bs x n_vars x num_patch x num_features] common_channel/mix_channel

        time_hierarchy enabled (teacher forcing during training):
            y: ([bs x forecast_patches x nvars],[bs x forecast_len x nvars]) # y_hier, actual_y
        no time hierarchy:
            y: None

        Output:

        time_hierarchy enabled (teacher forcing during training):
            output: ([bs x forecast_patches x nvars],[bs x forecast_len x nvars]) # y_hier, actual_y
        no time hierarchy:
            output: [bs x forecast_len x nvars]

        """

        if self.mode in ["common_channel", "mix_channel"]:
            x = self.flatten(x)  # [bs x n_vars x num_patch * num_features]
            # x = torch.reshape(
            #     x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            # )  # [bs x n_vars x num_patch * num_features]

            forecast = self.base_forecast_block(x)  # [bs x n_vars x forecast_len]
            forecast = forecast.transpose(-1, -2)  # [bs x forecast_len x n_vars]

            if self.forecast_channel_mixing:
                forecast = self.fcm(x=forecast, y=y)  # y: [bs x forecast_len x n_vars]

            if self.time_hierarchy:
                h_pred, forecast = self.ht_block(
                    x=x, base_forecast=forecast, y=y
                )  # [bs x forecast_patches x nvars],[bs x forecast_len x nvars]

            if self.time_hierarchy:
                return h_pred, forecast
            else:
                return forecast

        else:
            x = self.flatten(x)  # x: [bs x num_patches*num_features]
            forecast = self.base_forecast_block(x)  # [bs x forecast_len * self.nvars]
            forecast = forecast.reshape(
                -1, self.forecast_len, self.nvars
            )  # y: [bs x forecast_len x n_vars]
            return forecast


class LinearHead(nn.Module):
    """PredictionHead

    Args:
        num_patches (int): Number of patches to segment
        patch_size (int, optional): Patch length. Defaults to 16.
        in_channels (int, optional): Number of input variables. Defaults to 3.
        num_features (int, optional): Hidden feature size. Defaults to 16.
        head_dropout (float, optional): Head Dropout rate. Defaults to 0.2.
        head_agg (str, optional): Aggregation mode. Allowed values are use_last, max_pool, avg_pool.
                                Defaults to max_pool.
        output_range (list, optional): Output range to restrict. Defaults to None.

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
        x = x.transpose(
            -1, -2
        )  # bs x num_features x num_patch or bs x n_vars x num_features x num_patch
        if self.head_agg == "use_last":
            x = x[
                ..., -1
            ]  # # bs x num_features (flatten) or # bs x n_vars x num_features (common_channel)
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
            x = SigmoidRange(*self.output_range)(x)
        return x
