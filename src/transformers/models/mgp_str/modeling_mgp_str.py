# coding=utf-8
# Copyright 2022 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MGPSTR model."""


from functools import partial
from typing import Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from .configuration_mgp_str import MGPSTRConfig
from .helpers import DropPath, to_2tuple


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "MGPSTRConfig"
_TOKENIZER_FOR_DOC = "MGPSTRTokenizer"

# Base docstring
_CHECKPOINT_FOR_DOC = "alibaba-damo/mgp-str-base"

MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "alibaba-damo/mgp-str-base",
    # See all MGP-STR models at https://huggingface.co/models?filter=mgp-str
]


class MGPSTRPatchEmbeddings(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MGPSTRMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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


class MGPSTRAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MGPSTRLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MGPSTRAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MGPSTRMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MGPSTREncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(
            *[
                MGPSTRLayer(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x


class MGPSTRA3Module(nn.Module):
    def __init__(self, input_embed_dim, out_token=30):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(
            nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1, 1), stride=1, groups=8, bias=False),
            nn.Conv2d(input_embed_dim, out_token, kernel_size=(1, 1), stride=1, bias=False),
        )
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1, 1), stride=1, groups=8, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)

    def forward(self, x):
        x = self.token_norm(x)
        x = x.transpose(1, 2).unsqueeze(-1)
        selected = self.tokenLearner(x)
        selected = selected.flatten(2)
        selected = F.softmax(selected, dim=-1)
        feat = self.feat(x)
        feat = feat.flatten(2).transpose(1, 2)
        x = torch.einsum("...si,...id->...sd", selected, feat)

        x = self.norm(x)
        return selected, x


class MGPSTRPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MGPSTRConfig
    base_model_prefix = "mgp_str"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: MGPSTREncoder, value: bool = False) -> None:
        if isinstance(module, MGPSTREncoder):
            module.gradient_checkpointing = value


MGP_STR_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MGPSTRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The MGP-STR Model transformer with classification heads on top.",
    MGP_STR_START_DOCSTRING,
)
class MGPSTRModel(MGPSTRPreTrainedModel):
    def __init__(self, config: MGPSTRConfig):
        super().__init__(config)
        self.config = config
        self.num_tokens = 2 if config.distilled else 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.patch_embed = MGPSTRPatchEmbeddings(config.img_size, config.patch_size, config.in_chans, config.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        self.encoder = MGPSTREncoder(
            config.embed_dim,
            config.depth,
            config.num_heads,
            config.mlp_ratio,
            config.qkv_bias,
            config.drop_rate,
            config.attn_drop_rate,
            config.drop_path_rate,
            config.norm_layer,
            config.act_layer,
        )

        self.char_a3_module = MGPSTRA3Module(config.embed_dim, config.max_token_length)
        self.bpe_a3_module = MGPSTRA3Module(config.embed_dim, config.max_token_length)
        self.wp_a3_module = MGPSTRA3Module(config.embed_dim, config.max_token_length)

        self.char_head = nn.Linear(config.embed_dim, config.char_num_classes)
        self.bpe_head = nn.Linear(config.embed_dim, config.bpe_num_classes)
        self.wp_head = nn.Linear(config.embed_dim, config.wp_num_classes)

    def forward(self, pixel_values):
        r"""
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
                [`ViTImageProcessor.__call__`] for details.
        Returns:
            out(`List[torch.FloatTensor]`): The list of logits output of char, bpe, wp.
            attens(`List[torch.FloatTensor]`): The list of attention output of char, bpe, wp.

        Example:

        ```python
        >>> from transformers import (
        ...     MGPSTRModel,
        ...     MGPSTRProcessor,
        ... )
        >>> import requests
        >>> from PIL import Image

        >>> # load image from the IIIT-5k dataset
        >>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> processor = MGPSTRProcessor.from_pretrained("alibaba-damo/mgp-str-base")
        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> model = MGPSTRModel.from_pretrained("alibaba-damo/mgp-str-base")

        >>> # inference
        >>> generated_ids, attens = model(pixel_values)
        >>> outs = processor.batch_decode(generated_ids)
        >>> outs["generated_text"]
        '['ticket']'
        ```"""

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)

        attens = []
        out = []
        out_softmax = []

        # char
        char_attn, x_char = self.char_a3_module(x)
        char_out = self.char_head(x_char)
        attens.append(char_attn)
        out.append(char_out)
        out.append(F.softmax(char_out, dim=2))

        # bpe
        bpe_attn, x_bpe = self.bpe_a3_module(x)
        bpe_out = self.bpe_head(x_bpe)
        attens.append(bpe_attn)
        out.append(bpe_out)
        out.append(F.softmax(bpe_out, dim=2))

        # wp
        wp_attn, x_wp = self.wp_a3_module(x)
        wp_out = self.wp_head(x_wp)
        attens.append(wp_attn)
        out.append(wp_out)
        out.append(F.softmax(wp_out, dim=2))

        return out, attens
