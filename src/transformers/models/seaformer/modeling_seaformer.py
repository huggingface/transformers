# coding=utf-8
# Copyright 2023 NVIDIA The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Seaformer model."""


import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_seaformer import SeaformerConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "SeaformerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "seaformer-large"
_EXPECTED_OUTPUT_SHAPE = [1, 128, 64, 64]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "seaformer-large"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

SEAFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Inderpreet01/seaformer-semantic-segmentation-large",
    # See all Seaformer models at https://huggingface.co/models?filter=seaformer
]


def _make_divisible(value, divisor, min_value=None):
    """
    This function is taken from the original tf repo. It ensures that all layers have a channel number that is
    divisible by 8 It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * value:
        new_v += divisor
    return new_v


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(input, drop_prob: float = 0.0, training: bool = False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Seaformer
class SeaformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SeaformerConv2d(nn.Module):
    def __init__(
        self, inp_channel, out_channel, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, bias=False
    ):
        super().__init__()
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.add_module(
            "convolution", nn.Conv2d(self.inp_channel, self.out_channel, ks, stride, pad, dilation, groups, bias=bias)
        )

        bn = nn.BatchNorm2d(self.out_channel)
        # nn.init.constant_(bn.weight, bn_weight_init)
        # nn.init.constant_(bn.bias, 0)
        self.add_module("batchnorm", bn)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.batchnorm(self.convolution(hidden_states))


class SeaformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, hidden_act=nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = SeaformerConv2d(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = hidden_act
        self.dense2 = SeaformerConv2d(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense1(hidden_states)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.dense2(x)
        x = self.drop(x)
        return x


class SeaformerInvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, ks: int, stride: int, expand_ratio: int, hidden_act=None) -> None:
        super(SeaformerInvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        # assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(SeaformerConv2d(inp, hidden_dim, ks=1))
            layers.append(hidden_act)
        layers.extend(
            [
                # dw
                SeaformerConv2d(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks // 2, groups=hidden_dim),
                hidden_act,
                # pw-linear
                SeaformerConv2d(hidden_dim, oup, ks=1),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return hidden_states + self.conv(hidden_states)
        else:
            return self.conv(hidden_states)


class SeaformerStackedMV2Block(nn.Module):
    def __init__(self, mv2_blocks_cfgs, stem, inp_channel=16, hidden_act=nn.ReLU, width_mult=1.0):
        super().__init__()

        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(SeaformerConv2d(3, inp_channel, 3, 2, 1), hidden_act)

        self.mv2_blocks_cfgs = mv2_blocks_cfgs
        self.layers = []

        for i, (kernel_size, expand_ratio, out_channels, stride) in enumerate(mv2_blocks_cfgs):
            output_channel = _make_divisible(out_channels * width_mult, 8)
            exp_size = expand_ratio * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = "layer{}".format(i + 1)
            layer = SeaformerInvertedResidual(
                inp_channel,
                output_channel,
                ks=kernel_size,
                stride=stride,
                expand_ratio=expand_ratio,
                hidden_act=hidden_act,
            )
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.stem:
            hidden_states = self.stem_block(hidden_states)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            hidden_states = layer(hidden_states)
        return hidden_states


class SeaformerSqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, dim = hidden_states.shape
        hidden_states = hidden_states + F.interpolate(self.pos_embed, size=(dim), mode="linear", align_corners=False)

        return hidden_states


# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         return self.relu(hidden_states + 3) / 6


class SeaformerAttention(nn.Module):
    def __init__(self, dim=3, key_dim=64, num_attention_heads=8, attn_ratio=2, hidden_act=None):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_attention_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_attention_heads
        self.attn_ratio = attn_ratio

        self.to_q = SeaformerConv2d(dim, nh_kd, 1)
        self.to_k = SeaformerConv2d(dim, nh_kd, 1)
        self.to_v = SeaformerConv2d(dim, self.dh, 1)

        # if activation==None:
        #     activation = nn.ReLU

        self.proj = torch.nn.Sequential(hidden_act, SeaformerConv2d(self.dh, dim, bn_weight_init=0))

        self.proj_encode_row = torch.nn.Sequential(hidden_act, SeaformerConv2d(self.dh, self.dh, bn_weight_init=0))
        self.pos_emb_rowq = SeaformerSqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SeaformerSqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_column = torch.nn.Sequential(hidden_act, SeaformerConv2d(self.dh, self.dh, bn_weight_init=0))
        self.pos_emb_columnq = SeaformerSqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SeaformerSqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = SeaformerConv2d(
            self.dh + 2 * self.nh_kd,
            2 * self.nh_kd + self.dh,
            ks=3,
            stride=1,
            pad=1,
            dilation=1,
            groups=2 * self.nh_kd + self.dh,
        )
        self.act = hidden_act
        self.pwconv = SeaformerConv2d(2 * self.nh_kd + self.dh, dim, ks=1)
        # self.sigmoid = h_sigmoid()
        self.sigmoid = ACT2FN["h_sigmoid"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C, H, W = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # detail enhance
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_attention_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_attention_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_attention_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_attention_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_attention_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_attention_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)

        xx = self.sigmoid(xx) * qkv
        return xx


class SeaformerBlock(nn.Module):
    def __init__(
        self,
        dim=3,
        key_dim=64,
        num_attention_heads=8,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        hidden_act=nn.ReLU,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio

        self.attn = SeaformerAttention(
            dim, key_dim=key_dim, num_attention_heads=num_attention_heads, attn_ratio=attn_ratio, hidden_act=hidden_act
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = SeaformerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SeaformerMLP(in_features=dim, hidden_features=mlp_hidden_dim, hidden_act=hidden_act, drop=drop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.drop_path(self.attn(hidden_states))
        hidden_states = hidden_states + self.drop_path(self.mlp(hidden_states))
        return hidden_states


class SeaformerBasicLayer(nn.Module):
    def __init__(
        self,
        block_num=4,
        embedding_dim=3,
        key_dim=64,
        num_attention_heads=8,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        hidden_act=nn.ReLU,
    ):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                SeaformerBlock(
                    embedding_dim,
                    key_dim=key_dim,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    hidden_act=hidden_act,
                )
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # token * N
        for i in range(self.block_num):
            hidden_states = self.transformer_blocks[i](hidden_states)
        return hidden_states


class SeaformerFusionBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        embed_dim: int,
        activations=None,
    ) -> None:
        super(SeaformerFusionBlock, self).__init__()
        self.local_embedding = nn.Sequential()
        self.local_embedding.add_module(
            "conv", nn.Conv2d(in_channels=inp, out_channels=embed_dim, kernel_size=1, bias=False)
        )
        bn = nn.BatchNorm2d(embed_dim)
        # nn.init.constant_(bn.weight, 1)
        # nn.init.constant_(bn.bias, 0)
        self.local_embedding.add_module("batchnorm", bn)

        self.global_act = nn.Sequential()
        self.global_act.add_module(
            "conv", nn.Conv2d(in_channels=oup, out_channels=embed_dim, kernel_size=1, bias=False)
        )
        bn = nn.BatchNorm2d(embed_dim)
        # nn.init.constant_(bn.weight, 1)
        # nn.init.constant_(bn.bias, 0)
        self.global_act.add_module("batchnorm", bn)

        # self.act = h_sigmoid()
        self.act = ACT2FN["h_sigmoid"]

    def forward(self, x_local: torch.Tensor, x_global: torch.Tensor) -> torch.Tensor:
        """
        x_g: global features x_l: local features
        """
        B, C, H, W = x_local.shape
        B, C_c, H_c, W_c = x_global.shape

        local_feat = self.local_embedding(x_local)
        global_act = self.global_act(x_global)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode="bilinear", align_corners=False)
        out = local_feat * sig_act
        return out


class SeaformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.channels = config.channels
        self.depths = config.depths
        self.mv2_blocks_cfgs = config.mv2_blocks_cfgs
        if isinstance(config.hidden_act, str):
            self.hidden_act = ACT2FN[config.hidden_act]
        else:
            self.hidden_act = config.hidden_act
        # self.init_cfg = config.init_cfg
        # if self.init_cfg is not None:
        #     self.pretrained = self.init_cfg['checkpoint']

        for i in range(len(config.mv2_blocks_cfgs)):
            smb = SeaformerStackedMV2Block(
                mv2_blocks_cfgs=config.mv2_blocks_cfgs[i],
                stem=True if i == 0 else False,
                inp_channel=self.channels[i],
                hidden_act=self.hidden_act,
            )
            setattr(self, f"smb{i + 1}", smb)

        for i in range(len(config.depths)):
            dpr = [
                x.item() for x in torch.linspace(0, config.drop_path_rate, config.depths[i])
            ]  # stochastic depth decay rule
            trans = SeaformerBasicLayer(
                block_num=config.depths[i],
                embedding_dim=config.emb_dims[i],
                key_dim=config.key_dims[i],
                num_attention_heads=config.num_attention_heads,
                mlp_ratio=config.mlp_ratios[i],
                attn_ratio=config.attn_ratios,
                drop=0,
                attn_drop=0,
                drop_path=dpr,
                hidden_act=self.hidden_act,
            )
            setattr(self, f"trans{i + 1}", trans)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        pixel_values.shape[0]
        hidden_states = pixel_values

        num_smb_stage = len(self.mv2_blocks_cfgs)
        num_trans_stage = len(self.depths)
        x = pixel_values
        for i in range(num_smb_stage):
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)
            # 1/8 shared feat
            if i == 1:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (x,)

            if num_trans_stage + i >= num_smb_stage:
                trans = getattr(self, f"trans{i + num_trans_stage - num_smb_stage + 1}")

                for i in range(trans.block_num):
                    attn_out = trans.transformer_blocks[i].attn(x)
                    x = x + trans.transformer_blocks[i].drop_path(attn_out)
                    mlp_out = trans.transformer_blocks[i].mlp(x)
                    x = x + trans.transformer_blocks[i].drop_path(mlp_out)

                    if output_attentions:
                        all_self_attentions = all_self_attentions + (attn_out,)

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (x,)

        hidden_states = x

        # return outputs
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SeaformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SeaformerConfig
    base_model_prefix = "seaformer"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Conv2d):
            n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            n //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / n))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 0.01)
            if module.bias is not None:
                module.bias.data.zero_()


SEAFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SeaformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SEAFORMER_INPUTS_DOCSTRING = r"""

    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SeaformerImageProcessor.__call__`] for details.

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
    "The bare Seaformer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.",
    SEAFORMER_START_DOCSTRING,
)
# Copied from transformers.models.segformer.modeling_segformer.SegformerModel with SEGFORMER->SEAFORMER,Segformer->Seaformer
class SeaformerModel(SeaformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SeaformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SEAFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SeaformerDecodeHead(SeaformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        head_channels = config.decoder_channels
        in_channels = config.in_channels
        self.in_index = config.in_index

        self.linear_fuse = nn.Sequential()
        self.linear_fuse.add_module(
            "conv",
            nn.Conv2d(
                in_channels=head_channels,
                out_channels=head_channels,
                kernel_size=1,
                stride=1,
                groups=head_channels if config.is_depthwise else 1,
                bias=False,
            ),
        )
        bn = nn.BatchNorm2d(head_channels)
        # nn.init.constant_(bn.weight, 1)
        # nn.init.constant_(bn.bias, 0)
        self.linear_fuse.add_module("batchnorm", bn)

        self.linear_fuse.add_module("activate", nn.ReLU(inplace=True))

        for i in range(len(config.embed_dims)):
            fuse = SeaformerFusionBlock(
                in_channels[0] if i == 0 else config.embed_dims[i - 1],
                in_channels[i + 1],
                embed_dim=config.embed_dims[i],
            )
            setattr(self, f"fuse{i + 1}", fuse)
        self.embed_dims = config.embed_dims
        self.conv_seg = nn.Conv2d(head_channels, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # batch_size = encoder_hidden_states[-1].shape[0]
        xx = [encoder_hidden_states[i] for i in self.in_index]
        x_detail = xx[0]
        for i in range(len(self.embed_dims)):
            fuse = getattr(self, f"fuse{i + 1}")
            x_detail = fuse(x_detail, xx[i + 1])
        _c = self.linear_fuse(x_detail)
        logits = self.conv_seg(_c)
        return logits


@add_start_docstrings(
    """Seaformer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.""",
    SEAFORMER_START_DOCSTRING,
)
# Copied from transformers.models.segformer.modeling_segformer.SegformerForSemanticSegmentation with SEGFORMER->SEAFORMER,Segformer->Seaformer,segformer->seaformer
class SeaformerForSemanticSegmentation(SeaformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.seaformer = SeaformerModel(config)
        self.decode_head = SeaformerDecodeHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SEAFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, SeaformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/seaformer-b0-finetuned-ade-512-512")
        >>> model = SeaformerForSemanticSegmentation.from_pretrained("nvidia/seaformer-b0-finetuned-ade-512-512")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.seaformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()
            else:
                raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
