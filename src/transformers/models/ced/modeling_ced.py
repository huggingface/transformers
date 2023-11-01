# coding=utf-8
# Copyright 2023 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch CED (Ced) model."""

import collections
import math
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_outputs import SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_ced import CedConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "CedConfig"
_CHECKPOINT_FOR_DOC = "mispeech/ced-tiny"
_SEQ_CLASS_EXPECTED_OUTPUT = 0
_SEQ_CLASS_EXPECTED_LOSS = 0.01

# Audio classification docstring
_SEQ_CLASS_CHECKPOINT = "mispeech/ced-tiny"


CED_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "mispeech/ced-tiny",
    "mispeech/ced-mini",
    "mispeech/ced-small",
    "mispeech/ced-base",
    # See all CED models at https://huggingface.co/models?search=mispeech%2Fced
]


class CedPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CedConfig
    base_model_prefix = "ced"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


Conv_Kernel = Union[int, Tuple[int, int]]


def to_2tuple(x: Any) -> Tuple[Any, Any]:
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class CedAudioPatchEmbed(nn.Module):
    def __init__(
        self,
        input_size: Conv_Kernel = 224,
        patch_size: Conv_Kernel = 16,
        patch_stride: Conv_Kernel = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = False,
    ):
        super().__init__()
        self.input_size = to_2tuple(input_size)
        self.patch_size = to_2tuple(patch_size)
        self.patch_stride = to_2tuple(patch_stride)
        self.grid_size = (self.input_size[0] // self.patch_stride[0], self.input_size[1] // self.patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))
        x = self.norm(x)
        return x


class CedAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if mask is not None:
        # # Mask is a tensor of shape [B, T, T]
        # # Different from self.causal == True, the mask might be something like:
        # # [False, False, True]
        # # [False, False, True]
        # # [True, True, True]
        # # We use -inf to pad here, since if we would pad by any number, the entries at rows only containing
        # # [True, True, True] would lead to weights such as: [0.33,0.33,0.33], which is not correct
        # mask_value = torch.as_tensor(-float('inf'))
        # print(mask.shape, attn.shape)
        # attn = attn.masked_fill(mask, mask_value)
        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]
            mask = torch.ones(i, j, device=q.device, dtype=torch.bool).triu(j - i + 1)
            attn = attn.masked_fill(mask, mask_value)
        attn = attn.softmax(dim=-1)
        # Only for the case that a mask with all True entries on a row is passed.
        # attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CedMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Drop path is taken from Timm
# https://github.com/huggingface/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/layers/drop.py#L155
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I (https://github.com/rwightman) created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class CedBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: Callable = CedAttention,
        attention_kwargs={},
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attention_type(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, **attention_kwargs
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = CedMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# Taken from timm
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


CED_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CedConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CED_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, n_mels, sequence_length)`):
            The sequence of audio features extracted from the audio signal. Can be obtained from a raw audio waveform
            using `~transformers.CedFeatureExtractor.__call__`.
"""


@add_start_docstrings(
    "The bare Ced Model transformer outputting raw hidden-states without any specific head on top.",
    CED_START_DOCSTRING,
)
class CedModel(CedPreTrainedModel):
    def __init__(self, config: CedConfig) -> None:
        super().__init__(config)
        self.config = config

        # Allowed length in number of frames, otherwise the positional embedding will throw an error
        self.maximal_allowed_length = self.config.target_length

        self.init_bn = torch.nn.BatchNorm2d(config.n_mels, momentum=0.01)

        self.patch_embed = CedAudioPatchEmbed(
            input_size=(config.n_mels, config.target_length),
            embed_dim=config.embed_dim,
            patch_size=config.patch_size,
            flatten=False,
            patch_stride=config.patch_stride,
        )

        self.time_pos_embed = nn.Parameter(torch.randn(1, config.embed_dim, 1, self.patch_embed.grid_size[1]) * 0.02)
        self.freq_pos_embed = nn.Parameter(torch.randn(1, config.embed_dim, self.patch_embed.grid_size[0], 1) * 0.02)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        self.blocks = nn.Sequential(
            *[
                CedBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attention_type=CedAttention,
                )
                for i in range(config.depth)
            ]
        )
        self.norm = norm_layer(config.embed_dim)

        # Make `check_config_attributes` happy
        if not self.config.name.startswith("ced"):
            raise NotImplementedError

        # Initialize weights and apply final processing
        self.post_init()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        x = x + self.freq_pos_embed[:, :, :, :]  # Just to support __getitem__ in posembed

        # x = rearrange(x, 'b c f t -> b (f t) c')
        x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))

        if self.config.pooling == "token":
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    @add_start_docstrings_to_model_forward(CED_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_SEQ_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(self, input_values: torch.Tensor):
        r"""
        Runs a forward pass of the CED model as an audio encoder.
        """
        x = torch.unsqueeze(input_values, 1)

        x = torch.permute(x, (0, 2, 1, 3))
        x = self.init_bn(x)
        x = torch.permute(x, (0, 2, 1, 3))

        if x.shape[-1] > self.maximal_allowed_length:
            splits = x.split(self.maximal_allowed_length, -1)

            if splits[-1].shape[-1] < self.maximal_allowed_length:
                if self.config.pad_last:
                    pad = torch.zeros(*x.shape[:-1], self.maximal_allowed_length, device=x.device)
                    pad[..., : splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = torch.flatten(splits, 0, 1)  # spl b c f t-> (spl b) c f t
            x = self.forward_head(self.ced(x))
            x = torch.reshape(x, (n_splits, -1, self.outputdim))  # (spl b) d -> spl b d, spl=n_splits

            if self.config.eval_avg == "mean":
                x = x.mean(0)
            elif self.config.eval_avg == "max":
                x = x.max(0)[0]
            else:
                raise ValueError(f"Unknown Eval average function ({self.eval_avg})")
        else:
            x = self.forward_features(x)

        return x


@add_start_docstrings(
    """
    Ced model with an audio classification head on top (a linear layer on top of the pooled output).
    """,
    CED_START_DOCSTRING,
)
class CedForAudioClassification(CedPreTrainedModel):
    def __init__(self, config: CedConfig) -> None:
        super().__init__(config)
        self.config = config

        self.encoder = CedModel(config)

        # Classifier head
        self.outputlayer = nn.Sequential(nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, config.outputdim))

        # Initialize weights and apply final processing
        self.post_init()

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "token":
            x = x[:, 0]
            return self.outputlayer(x).sigmoid()
        elif self.config.pooling == "mean":
            x = x.mean(1)
            return self.outputlayer(x).sigmoid()
        elif self.config.pooling == "logit":
            x = x.mean(1)
            return self.outputlayer(x)
        elif self.config.pooling == "dm":
            # Unpack using the frequency dimension, which is constant
            # 'b (f t) d -> b f t d', f=self.patch_embed.grid_size[0])
            x = torch.reshape(x, (x.shape[0], self.patch_embed.grid_size[0], -1, x.shape[3]))

            # First poolin frequency, then sigmoid the (B T D) output
            x = self.outputlayer(x.mean(1)).sigmoid()
            return x.mean(1)
        else:
            return x.mean(1)

    @add_start_docstrings_to_model_forward(CED_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_SEQ_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(self, input_values: torch.Tensor):
        r"""
        Runs a forward pass of the CED model for audio classification task.
        """
        x = self.encoder(input_values)
        x = self.forward_head(x)
        return x
