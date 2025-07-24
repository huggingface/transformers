# coding=utf-8
# Copyright 2023 Alibaba Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MGP-STR model."""

import collections.abc
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, logging
from .configuration_mgp_str import MgpstrConfig


logger = logging.get_logger(__name__)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Mgpstr
class MgpstrDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    """
)
class MgpstrModelOutput(ModelOutput):
    r"""
    logits (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`):
        Tuple of `torch.FloatTensor` (one for the output of character of shape `(batch_size,
        config.max_token_length, config.num_character_labels)`, + one for the output of bpe of shape `(batch_size,
        config.max_token_length, config.num_bpe_labels)`, + one for the output of wordpiece of shape `(batch_size,
        config.max_token_length, config.num_wordpiece_labels)`) .

        Classification scores (before SoftMax) of character, bpe and wordpiece.
    a3_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`):
        Tuple of `torch.FloatTensor` (one for the attention of character, + one for the attention of bpe`, + one
        for the attention of wordpiece) of shape `(batch_size, config.max_token_length, sequence_length)`.

        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.
    """

    logits: tuple[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    a3_attentions: Optional[tuple[torch.FloatTensor]] = None


class MgpstrEmbeddings(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, config: MgpstrConfig):
        super().__init__()
        image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_tokens = 2 if config.distilled else 1

        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, config.hidden_size))
        self.pos_drop = nn.Dropout(p=config.drop_rate)

    def forward(self, pixel_values):
        batch_size, channel, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        patch_embeddings = self.proj(pixel_values)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)  # BCHW -> BNC

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedding_output = torch.cat((cls_tokens, patch_embeddings), dim=1)
        embedding_output = embedding_output + self.pos_embed
        embedding_output = self.pos_drop(embedding_output)

        return embedding_output


class MgpstrMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, config: MgpstrConfig, hidden_features):
        super().__init__()
        hidden_features = hidden_features or config.hidden_size
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, config.hidden_size)
        self.drop = nn.Dropout(config.drop_rate)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop(hidden_states)
        return hidden_states


class MgpstrAttention(nn.Module):
    def __init__(self, config: MgpstrConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attn_drop_rate)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_drop = nn.Dropout(config.drop_rate)

    def forward(self, hidden_states):
        batch_size, num, channel = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, num, 3, self.num_heads, channel // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attention_probs = (query @ key.transpose(-2, -1)) * self.scale
        attention_probs = attention_probs.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = (attention_probs @ value).transpose(1, 2).reshape(batch_size, num, channel)
        context_layer = self.proj(context_layer)
        context_layer = self.proj_drop(context_layer)
        return (context_layer, attention_probs)


class MgpstrLayer(nn.Module):
    def __init__(self, config: MgpstrConfig, drop_path=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MgpstrAttention(config)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = MgpstrDropPath(drop_path) if drop_path is not None else nn.Identity()
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        self.mlp = MgpstrMlp(config, mlp_hidden_dim)

    def forward(self, hidden_states):
        self_attention_outputs = self.attn(self.norm1(hidden_states))
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1]

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # second residual connection is done here
        layer_output = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))

        outputs = (layer_output, outputs)
        return outputs


class MgpstrEncoder(nn.Module):
    def __init__(self, config: MgpstrConfig):
        super().__init__()
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device="cpu")]

        self.blocks = nn.Sequential(
            *[MgpstrLayer(config=config, drop_path=dpr[i]) for i in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for _, blk in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = blk(hidden_states)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MgpstrA3Module(nn.Module):
    def __init__(self, config: MgpstrConfig):
        super().__init__()
        self.token_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.tokenLearner = nn.Sequential(
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=(1, 1), stride=1, groups=8, bias=False),
            nn.Conv2d(config.hidden_size, config.max_token_length, kernel_size=(1, 1), stride=1, bias=False),
        )
        self.feat = nn.Conv2d(
            config.hidden_size, config.hidden_size, kernel_size=(1, 1), stride=1, groups=8, bias=False
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.token_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
        selected = self.tokenLearner(hidden_states)
        selected = selected.flatten(2)
        attentions = F.softmax(selected, dim=-1)

        feat = self.feat(hidden_states)
        feat = feat.flatten(2).transpose(1, 2)
        feat = torch.einsum("...si,...id->...sd", attentions, feat)
        a3_out = self.norm(feat)

        return (a3_out, attentions)


@auto_docstring
class MgpstrPreTrainedModel(PreTrainedModel):
    config: MgpstrConfig
    base_model_prefix = "mgp_str"
    _no_split_modules = []

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, MgpstrEmbeddings):
            nn.init.trunc_normal_(module.pos_embed, mean=0.0, std=std)
            nn.init.trunc_normal_(module.cls_token, mean=0.0, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@auto_docstring
class MgpstrModel(MgpstrPreTrainedModel):
    def __init__(self, config: MgpstrConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = MgpstrEmbeddings(config)
        self.encoder = MgpstrEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.proj

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.FloatTensor], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return encoder_outputs
        return BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    MGP-STR Model transformer with three classification heads on top (three A^3 modules and three linear layer on top
    of the transformer encoder output) for scene text recognition (STR) .
    """
)
class MgpstrForSceneTextRecognition(MgpstrPreTrainedModel):
    config: MgpstrConfig
    main_input_name = "pixel_values"

    def __init__(self, config: MgpstrConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mgp_str = MgpstrModel(config)

        self.char_a3_module = MgpstrA3Module(config)
        self.bpe_a3_module = MgpstrA3Module(config)
        self.wp_a3_module = MgpstrA3Module(config)

        self.char_head = nn.Linear(config.hidden_size, config.num_character_labels)
        self.bpe_head = nn.Linear(config.hidden_size, config.num_bpe_labels)
        self.wp_head = nn.Linear(config.hidden_size, config.num_wordpiece_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_a3_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.FloatTensor], MgpstrModelOutput]:
        r"""
        output_a3_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of a3 modules. See `a3_attentions` under returned tensors
            for more detail.

        Example:

        ```python
        >>> from transformers import (
        ...     MgpstrProcessor,
        ...     MgpstrForSceneTextRecognition,
        ... )
        >>> import requests
        >>> from PIL import Image

        >>> # load image from the IIIT-5k dataset
        >>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

        >>> # inference
        >>> outputs = model(pixel_values)
        >>> out_strs = processor.batch_decode(outputs.logits)
        >>> out_strs["generated_text"]
        '["ticket"]'
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mgp_outputs = self.mgp_str(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = mgp_outputs[0]

        char_a3_out, char_attention = self.char_a3_module(sequence_output)
        bpe_a3_out, bpe_attention = self.bpe_a3_module(sequence_output)
        wp_a3_out, wp_attention = self.wp_a3_module(sequence_output)

        char_logits = self.char_head(char_a3_out)
        bpe_logits = self.bpe_head(bpe_a3_out)
        wp_logits = self.wp_head(wp_a3_out)

        all_a3_attentions = (char_attention, bpe_attention, wp_attention) if output_a3_attentions else None
        all_logits = (char_logits, bpe_logits, wp_logits)

        if not return_dict:
            outputs = (all_logits, all_a3_attentions) + mgp_outputs[1:]
            return tuple(output for output in outputs if output is not None)
        return MgpstrModelOutput(
            logits=all_logits,
            hidden_states=mgp_outputs.hidden_states,
            attentions=mgp_outputs.attentions,
            a3_attentions=all_a3_attentions,
        )


__all__ = ["MgpstrModel", "MgpstrPreTrainedModel", "MgpstrForSceneTextRecognition"]
