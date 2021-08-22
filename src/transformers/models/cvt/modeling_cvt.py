# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch CvT model. """


import collections.abc
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss

import scipy
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_cvt import CvTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "CvTConfig"
_CHECKPOINT_FOR_DOC = "google/CvT-base-patch16-224"

CVT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/CvT-base-patch16-224",
    # See all CvT models at https://huggingface.co/models?filter=CvT
]


# see for official implementation by
# https://github.com/microsoft/CvT


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class PatchEmbeddings(nn.Module):
    """Image to Conv Embedding"""

    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x


class CvTOutput(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CvTAttention(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        method="dw_bn",
        kernel_size=3,
        stride_kv=1,
        stride_q=1,
        padding_kv=1,
        padding_q=1,
        with_cls_token=True,
        **kwargs
    ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q, stride_q, "linear" if method == "avg" else method
        )
        self.conv_proj_k = self._build_projection(dim_in, dim_out, kernel_size, padding_kv, stride_kv, method)
        self.conv_proj_v = self._build_projection(dim_in, dim_out, kernel_size, padding_kv, stride_kv, method)

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
        if method == "dw_bn":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                dim_in,
                                dim_in,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=False,
                                groups=dim_in,
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(dim_in)),
                        ("rearrage", Rearrange("b c h w -> b (h w) c")),
                    ]
                )
            )
        elif method == "avg":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("avg", nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride, ceil_mode=True)),
                        ("rearrage", Rearrange("b c h w -> b (h w) c")),
                    ]
                )
            )
        elif method == "linear":
            proj = None
        else:
            raise ValueError("Unknown method ({})".format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, "b c h w -> b (h w) c")

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, "b c h w -> b (h w) c")

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, "b c h w -> b (h w) c")

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w, output_attentions=False):
        if self.conv_proj_q is not None or self.conv_proj_k is not None or self.conv_proj_v is not None:
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.proj_k(k), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.proj_v(v), "b t (h d) -> b h t d", h=self.num_heads)

        attn_score = torch.einsum("bhlk,bhtk->bhlt", [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum("bhlt,bhtv->bhlv", [attn, v])
        x = rearrange(x, "b h t d -> b t (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)

        if output_attentions:
            return x, attn

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T - 1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if hasattr(module, "conv_proj_q") and hasattr(module.conv_proj_q, "conv"):
            params = sum([p.numel() for p in module.conv_proj_q.conv.parameters()])
            flops += params * H_Q * W_Q

        if hasattr(module, "conv_proj_k") and hasattr(module.conv_proj_k, "conv"):
            params = sum([p.numel() for p in module.conv_proj_k.conv.parameters()])
            flops += params * H_KV * W_KV

        if hasattr(module, "conv_proj_v") and hasattr(module.conv_proj_v, "conv"):
            params = sum([p.numel() for p in module.conv_proj_v.conv.parameters()])
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class CvTLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__()

        self.with_cls_token = kwargs["with_cls_token"]

        self.norm1 = norm_layer(dim_in)
        self.attn = CvTAttention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = CvTOutput(in_features=dim_out, hidden_features=dim_mlp_hidden, act_layer=act_layer, drop=drop)

    def forward(self, x, h, w, output_attentions=False):
        res = x

        x = self.norm1(x)
        attn, all_attn = self.attn(x, h, w, output_attentions)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if output_attentions:
            return x, all_attn
        return x


class CvTEncoder(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        patch_stride=16,
        patch_padding=0,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = PatchEmbeddings(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        with_cls_token = kwargs["with_cls_token"]
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                CvTLayer(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

        if init == "xavier":
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logger.info("=> init weight of Linear from trunc norm")
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logger.info("=> init bias of Linear to zeros")
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logger.info("=> init weight of Linear from xavier uniform")
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logger.info("=> init bias of Linear to zeros")
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, "b c h w -> b (h w) c")

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

            x, attn = blk(x, H, W, output_attentions)
            if output_attentions:
                all_self_attentions = all_self_attentions + (attn,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x, all_hidden_states, all_self_attentions, cls_tokens


class CvTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CvTConfig
    base_model_prefix = "cvt"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


CvT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.CvTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

CvT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using :class:`~transformers.CvTFeatureExtractor`. See
            :meth:`transformers.CvTFeatureExtractor.__call__` for details.

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare CvT Model transformer outputting raw hidden-states without any specific head on top.",
    CvT_START_DOCSTRING,
)
class CvTModel(CvTPreTrainedModel):
    def __init__(self, config, in_chans=3):

        super().__init__(config)
        self.config = config
        self.num_stages = self.config.num_stages
        self.num_classes = self.config.num_classes
        if self.config.act_layer == "gelu":
            self.act_layer = nn.GELU
        if self.config.norm_layer == "layer_norm":
            self.norm_layer = nn.LayerNorm

        for i in range(self.num_stages):
            kwargs = {
                "patch_size": self.config.patch_size[i],
                "patch_stride": self.config.patch_stride[i],
                "patch_padding": self.config.patch_padding[i],
                "embed_dim": self.config.dim_embed[i],
                "depth": self.config.depth[i],
                "num_heads": self.config.num_heads[i],
                "mlp_ratio": self.config.mlp_ratio[i],
                "qkv_bias": self.config.qkv_bias[i],
                "drop_rate": self.config.drop_rate[i],
                "attn_drop_rate": self.config.attn_drop_rate[i],
                "drop_path_rate": self.config.drop_path_rate[i],
                "with_cls_token": self.config.cls_token[i],
                "method": self.config.qkv_proj_method[i],
                "kernel_size": self.config.kernel_qkv[i],
                "padding_q": self.config.padding_q[i],
                "padding_kv": self.config.padding_kv[i],
                "stride_kv": self.config.stride_kv[i],
                "stride_q": self.config.stride_q[i],
            }

            stage = CvTEncoder(
                in_chans=in_chans,
                init=self.config.init,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)

            in_chans = self.config.dim_embed[i]

        dim_embed = self.config.dim_embed[-1]
        self.norm = self.norm_layer(dim_embed)
        self.cls_token = self.config.cls_token[-1]

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f"stage{i}.pos_embed")
            layers.add(f"stage{i}.cls_token")

        return layers

    def forward_features(self, x, output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i in range(self.num_stages):
            x, hidden_states, attentions, cls_tokens = getattr(self, f"stage{i}")(
                x, output_attentions, output_hidden_states
            )
            if output_hidden_states:
                if i == 0:
                    all_hidden_states = all_hidden_states + hidden_states
                else:
                    all_hidden_states = all_hidden_states + hidden_states[1:]
            if output_attentions:
                all_self_attentions = all_self_attentions + attentions

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x, all_hidden_states, all_self_attentions

    @add_start_docstrings_to_model_forward(CvT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import CvTFeatureExtractor, CvTForImageClassification
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = CvTFeatureExtractor.from_pretrained('microsoft/cvt-base-384x384-in21k')
            >>> model = CvTModel.from_pretrained('microsoft/cvt-base-384x384-in21k')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        x, all_hidden_states, all_self_attentions = self.forward_features(
            pixel_values, output_attentions, output_hidden_states
        )

        if not return_dict:
            return x, all_hidden_states, all_self_attentions

        return BaseModelOutputWithPooling(
            last_hidden_state=all_hidden_states[-1],
            pooler_output=x,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """
    CvT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    CvT_START_DOCSTRING,
)
class CvTForImageClassification(CvTPreTrainedModel):
    def __init__(self, config, in_chans=3):

        super().__init__(config)
        self.config = config
        self.num_stages = self.config.num_stages
        self.num_classes = self.config.num_classes
        if self.config.act_layer == "gelu":
            self.act_layer = nn.GELU
        if self.config.norm_layer == "layer_norm":
            self.norm_layer = nn.LayerNorm

        for i in range(self.num_stages):
            kwargs = {
                "patch_size": self.config.patch_size[i],
                "patch_stride": self.config.patch_stride[i],
                "patch_padding": self.config.patch_padding[i],
                "embed_dim": self.config.dim_embed[i],
                "depth": self.config.depth[i],
                "num_heads": self.config.num_heads[i],
                "mlp_ratio": self.config.mlp_ratio[i],
                "qkv_bias": self.config.qkv_bias[i],
                "drop_rate": self.config.drop_rate[i],
                "attn_drop_rate": self.config.attn_drop_rate[i],
                "drop_path_rate": self.config.drop_path_rate[i],
                "with_cls_token": self.config.cls_token[i],
                "method": self.config.qkv_proj_method[i],
                "kernel_size": self.config.kernel_qkv[i],
                "padding_q": self.config.padding_q[i],
                "padding_kv": self.config.padding_kv[i],
                "stride_kv": self.config.stride_kv[i],
                "stride_q": self.config.stride_q[i],
            }

            stage = CvTEncoder(
                in_chans=in_chans,
                init=self.config.init,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)

            in_chans = self.config.dim_embed[i]

        dim_embed = self.config.dim_embed[-1]
        self.norm = self.norm_layer(dim_embed)
        self.cls_token = self.config.cls_token[-1]
        self.head = nn.Linear(dim_embed, self.num_classes) if self.num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f"stage{i}.pos_embed")
            layers.add(f"stage{i}.cls_token")

        return layers

    def forward_features(self, x, output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i in range(self.num_stages):
            x, hidden_states, attentions, cls_tokens = getattr(self, f"stage{i}")(
                x, output_attentions, output_hidden_states
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + hidden_states
            if output_attentions:
                all_self_attentions = all_self_attentions + attentions

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x, all_hidden_states, all_self_attentions

    @add_start_docstrings_to_model_forward(CvT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import CvTFeatureExtractor, CvTModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = CvTFeatureExtractor.from_pretrained('microsoft/cvt-base-384x384-in21k')
            >>> model = CvTForImageClassification.from_pretrained('microsoft/cvt-base-384x384-in21k')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
             >>> # model predicts one of the 1000 ImageNet classes
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        x, all_hidden_states, all_self_attentions = self.forward_features(
            pixel_values, output_attentions, output_hidden_states
        )
        logits = self.head(x)

        loss = None
        if labels is not None:
            if self.num_classes == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        if not return_dict:
            return logits, all_hidden_states, all_self_attentions

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits.view(-1, self.num_classes),
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
