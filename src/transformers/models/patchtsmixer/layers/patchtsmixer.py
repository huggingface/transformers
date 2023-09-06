__all__ = ["PatchTSMixer"]

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
from .basics import SigmoidRange, positional_encoding
from .gated_attention import GatedAttention
from .norm import NormLayer
from .mixutils import get_class_params_via_inspect
from torch.nn.modules.activation import MultiheadAttention

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

        if ffn == "mlp":
            self.mlp = MLP(in_channels, in_channels, expansion_factor, dropout)
        else:
            raise Exception("Invalid ffn %s"%(ffn))

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
        num_patches: int,
        num_features: int=16,
        expansion_factor:int =2,
        dropout: float=0.2,
        mode: str="common_channel",
        gated_attn: bool=False,
        ffn: str="mlp",
        self_attn: bool=False,
        self_attn_heads: int=1,
        norm_mlp: str="LayerNorm",
    ):
        super().__init__()
        
        self.norm_mlp = norm_mlp
        self.mode = mode
        self.norm = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)

        self.self_attn = self_attn

        if ffn == "mlp":
            self.mlp = MLP(num_patches, num_patches, expansion_factor, dropout)
        else:
            raise Exception("Invalid ffn %s"%(ffn))

        self.gated_attn = gated_attn
        if gated_attn:
            self.gab = GatedAttention(in_size=num_patches, out_size=num_patches)

        if self_attn:
            self.self_attn_layer = MultiheadAttention(
                embed_dim=num_features,
                num_heads=self_attn_heads,
                dropout=dropout,
                add_bias_kv=True,
                add_zero_attn=False,
                batch_first=True,
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

            x_attn, _ = self.self_attn_layer(x_tmp,x_tmp,x_tmp, need_weights=False)
            
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
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        mode (str, optional): Mixer Mode. Determines how to process the channels. Allowed values: flatten,
            common_channel, mix_channel. In flatten, patch embedding encodes the patch information across all channels.
            In common_channel mode, patch embedding is independent of channels (Channel Independece). In mix_channel,
            we follow channel independence, but in addition to patch and feature mixing, we also do channel mixing.
            Defaults to "common_channel".
        
    """

    def __init__(
        self,
        num_features: int=16,
        expansion_factor: int=2,
        dropout: float=0.2,
        gated_attn: bool=False,
        ffn: str="mlp",
        mode: str="common_channel",
        norm_mlp: str="LayerNorm",
    ):
        super().__init__()
        self.norm_mlp = norm_mlp
        self.mode = mode
        self.norm = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)

        self.mlp = MLP(num_features, num_features, expansion_factor, dropout)

        self.gated_attn = gated_attn

        if self.gated_attn:
            self.gab = GatedAttention(in_size=num_features, out_size=num_features)


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

class PatchTSMixerLayer(nn.Module):
    """
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
        ffn: str = "mlp",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        norm_mlp: str="LayerNorm",
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


class PatchTSMixerBackbone(nn.Module):
    """PatchTSMixerBackbone

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
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        ffn (str, optional): MLP mode. Allowed values: mlp, gmlp. gmlp is not preferred. Defaults to "mlp".
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        mixer_type (str, optional): Mixer Type to use. Allowed values are base, gated.
            base follows the MLP-Mixer architecture (https://arxiv.org/abs/2105.01601)
            gated follows the gMLP architecture (https://arxiv.org/pdf/2105.08050.pdf) Defaults to "base".
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
    """

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
        gated_attn: bool = False,
        ffn: str = "mlp",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        mixer_type: str = "base",
        norm_mlp="LayerNorm",
    ):
        
        super().__init__()
        self.mode = mode
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_layers = num_layers

        
        mix_params = {}
        if mixer_type == "base":
            mixer_class = (
                PatchTSMixerLayer
            )
        else:
            raise Exception("mixer_type %s is not yet implemented"%(mixer_type))
        
        self.mixers = nn.ModuleList(
                [
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

        

    def forward(self, x, output_hidden_states: Optional[bool] = False):
        # flatten: [bs x num_patch x num_features]   common_channel/mix_channel: [bs x n_vars x num_patch x num_features]

        all_hidden_states = []

        logger.debug(x.shape)

        embedding = x

        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states is True:
                all_hidden_states.append(embedding)
        
        # embedding.shape == (batch_size, num_patches, num_features) if flatten
        # embedding.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        if output_hidden_states is True:
            return embedding, all_hidden_states
        else:
            return embedding, None


class PatchTSMixer(nn.Module):
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
        gated_attn (bool, optional): Enable Gated Attention. Defaults to False.
        self_attn (bool, optional): Enable Tiny self attention in addition to MLP mixing. Defaults to False.
        self_attn_heads (bool, optional): Self attention heads. Defaults to 1.
        norm_mlp (str, optional): Norm layer (BatchNorm or LayerNorm). Defaults to LayerNorm.
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
        gated_attn: bool = False,
        self_attn: bool = False,
        self_attn_heads: int = 1,
        norm_mlp="LayerNorm",
        use_pe: bool = False,
        pe: str = "zeros",
        learn_pe: bool = False,
    ):
        super().__init__()

        ffn = "mlp"
        mixer_type = "base"

        # if mode == "flatten":
        #     logger.warn("Use mode = common_channel or mix_channel. mode=flatten is not preferred due to poor performance")
            
        self.mode = mode
        self.use_pe = use_pe
    
        if mode == "flatten":
            self.patcher = nn.Linear(in_channels * patch_size, num_features)

        elif mode in ["common_channel", "mix_channel"]:
            self.patcher = nn.Linear(patch_size, num_features)

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_features = num_features
        self.num_layers = num_layers
        
        self.mlp_mixer_encoder = PatchTSMixerBackbone(
            num_patches=num_patches,
            patch_size=patch_size,
            in_channels=in_channels,
            num_features=num_features,
            expansion_factor=expansion_factor,
            num_layers=num_layers,
            dropout=dropout,
            mode=mode,
            gated_attn=gated_attn,
            ffn=ffn,
            self_attn=self_attn,
            self_attn_heads=self_attn_heads,
            mixer_type=mixer_type,
            norm_mlp=norm_mlp,
        )

        if use_pe:
            self.W_pos = positional_encoding(pe, learn_pe, num_patches, num_features)

    def forward(self, x, output_hidden_states: Optional[bool] = False):
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
        
        
        embedding, all_hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states = output_hidden_states)

        logger.debug(x.shape)
        # embedding.shape == (batch_size, num_patches, num_features) if flatten
        # embedding.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        return embedding, all_hidden_states


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
        forecast_channel_indices (list, optional): List of channel indices to forecast. If None, forecast all channels.
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
            forecast = forecast.reshape(
                -1, self.forecast_len, self.nvars
            )  # y: [bs x forecast_len x n_vars]
            
            
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
        output_range (list, optional): Output range to restrict. Defaults to None.
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
    """

    def __init__(
        self,
        num_patches: int,
        num_features: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        head_dropout: float = 0,
        mode: str = "common_channel",
    ):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches

        if self.mode in ["common_channel", "mix_channel"]:
            self.base_pt_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_size),
            )
        else:
            self.base_pt_block = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(num_features, patch_size * in_channels),
            )

    def forward(self, x, y=None):
        """
        # flatten mode: [bs x num_patch x num_features] or
        common_channel/mix_channel mode: [bs x n_vars x num_patch x num_features]

    
        Output:
        z: [bs x n_vars x num_patch  x patch_len]

        """

        if self.mode == "flatten":
            x = self.base_pt_block(x)  # x: [bs x num_patch x n_vars*patch_size]
            x = torch.reshape(
                x, (x.shape[0], x.shape[1], self.patch_size, self.in_channels)
            )  # [bs x num_patch x patch_size x n_vars]
            x = x.permute(0, 3, 1, 2)  # [bs x nvars x num_patch  x patch_len]
            return x
        elif self.mode in ["common_channel", "mix_channel"]:
            forecast = self.base_pt_block(
                x
            )  # [bs x n_vars x num_patch x patch_size]
            return forecast