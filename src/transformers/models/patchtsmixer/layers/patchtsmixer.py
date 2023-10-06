__all__ = ["PatchTSMixer"]

# Cell
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.activation import MultiheadAttention

from .basics import positional_encoding
from .gated_attention import GatedAttention
from .norm import NormLayer


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, expansion_factor, dropout, last_dropout=True):
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
            raise Exception("Invalid ffn %s" % (ffn))

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
        num_features: int = 16,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        mode: str = "common_channel",
        gated_attn: bool = False,
        ffn: str = "mlp",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        norm_mlp: str = "LayerNorm",
    ):
        super().__init__()

        self.norm_mlp = norm_mlp
        self.mode = mode
        self.norm = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)

        self.self_attn = self_attn

        if ffn == "mlp":
            self.mlp = MLP(num_patches, num_patches, expansion_factor, dropout)
        else:
            raise Exception("Invalid ffn %s" % (ffn))

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
            self.norm_attn = NormLayer(norm_mlp=norm_mlp, mode=mode, num_features=num_features)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features) if flatten
        # x.shape == (batch_size, n_vars, num_patches, num_features) if common_channel
        residual = x

        x = self.norm(x)

        if self.self_attn:
            x_tmp = x
            if self.mode in ["common_channel", "mix_channel"]:
                x_tmp = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
                #  (batch_size, num_patches, num_features) if flatten
                #  (batch_size, n_vars, num_patches, num_features) if common_channel

            x_attn, _ = self.self_attn_layer(x_tmp, x_tmp, x_tmp, need_weights=False)

            if self.mode in ["common_channel", "mix_channel"]:
                x_attn = torch.reshape(x_attn, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
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
        num_features: int = 16,
        expansion_factor: int = 2,
        dropout: float = 0.2,
        gated_attn: bool = False,
        ffn: str = "mlp",
        mode: str = "common_channel",
        norm_mlp: str = "LayerNorm",
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
        norm_mlp: str = "LayerNorm",
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
            base follows the MLP-Mixer architecture (https://arxiv.org/abs/2105.01601) gated follows the gMLP
            architecture (https://arxiv.org/pdf/2105.08050.pdf) Defaults to "base".
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
            mixer_class = PatchTSMixerLayer
        else:
            raise Exception("mixer_type %s is not yet implemented" % (mixer_type))

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

        embedding, all_hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)

        logger.debug(x.shape)
        # embedding.shape == (batch_size, num_patches, num_features) if flatten
        # embedding.shape == (batch_size, n_vars, num_patches, num_features) if common_channel

        return embedding, all_hidden_states
