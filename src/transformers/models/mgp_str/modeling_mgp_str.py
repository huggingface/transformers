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
""" PyTorch MGP-STR model."""

import collections.abc
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mgp_str import MGPSTRConfig


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


def _drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however, the original name is
    misleading as 'Drop Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


@dataclass
class MGPSTRModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        logits (`tuple(torch.FloatTensor)` of shape `(batch_size, config.num_character_labels)`):
            Tuple of `torch.FloatTensor` (one for the output of character of shape `(batch_size,
            config.max_token_length, config.num_character_labels)`, + one for the output of bpe of shape `(batch_size,
            config.max_token_length, config.num_bpe_labels)`, + one for the output of wordpiece of shape `(batch_size,
            config.max_token_length, config.num_wordpiece_labels)`) .

            Classification scores (before SoftMax) of character, bpe and wordpiece.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, config.max_token_length,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        a3_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_a3_attentions=True` is passed or when `config.output_a3_attentions=True`):
            Tuple of `torch.FloatTensor` (one for the attention of character, + one for the attention of bpe`, + one
            for the attention of wordpiece) of shape `(batch_size, config.max_token_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    logits: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    a3_attentions: Optional[Tuple[torch.FloatTensor]] = None


class MGPSTRPatchEmbeddings(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, pixel_values):
        batch_size, channel, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        embeddings = self.proj(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)  # BCHW -> BNC
        embeddings = self.norm(embeddings)
        return embeddings


class MGPSTRMlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.drop(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop(hidden_states)
        return hidden_states


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


class MGPSTRLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = MGPSTRAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MGPSTRMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

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
    ):
        super().__init__()
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
                )
                for i in range(depth)
            ]
        )

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
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

    def forward(
        self,
        hidden_states,
    ):
        hidden_states = self.token_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(-1)
        selected = self.tokenLearner(hidden_states)
        selected = selected.flatten(2)
        attentions = F.softmax(selected, dim=-1)

        feat = self.feat(hidden_states)
        feat = feat.flatten(2).transpose(1, 2)
        feat = torch.einsum("...si,...id->...sd", attentions, feat)
        a3_out = self.norm(feat)

        return tuple((a3_out, attentions))


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

MGP_STR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
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
    "The bare MGP-STR Model transformer outputting raw hidden-states without any specific head on top.",
    MGP_STR_START_DOCSTRING,
)
class MGPSTRModel(MGPSTRPreTrainedModel):
    def __init__(self, config: MGPSTRConfig):
        super().__init__(config)
        self.config = config
        self.num_tokens = 2 if config.distilled else 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embed = MGPSTRPatchEmbeddings(
            config.image_size, config.patch_size, config.num_channels, config.hidden_size
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, config.hidden_size))
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        self.encoder = MGPSTREncoder(
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.mlp_ratio,
            config.qkv_bias,
            config.drop_rate,
            config.attn_drop_rate,
            config.drop_path_rate,
        )

    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch_size = pixel_values.shape[0]
        embedding_output = self.patch_embed(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedding_output = torch.cat((cls_tokens, embedding_output), dim=1)
        embedding_output = embedding_output + self.pos_embed
        embedding_output = self.pos_drop(embedding_output)

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


@add_start_docstrings(
    """
    MGP-STR Model transformer with three classification heads on top (three A^3 modules and three linear layer on top
    of the transformer encoder output) e.g. for scene text recognition (STR) .
    """,
    MGP_STR_START_DOCSTRING,
)
class MGPSTRForSceneTextRecognition(MGPSTRPreTrainedModel):
    def __init__(self, config: MGPSTRConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mgp_str = MGPSTRModel(config)

        self.char_a3_module = MGPSTRA3Module(config.hidden_size, config.max_token_length)
        self.bpe_a3_module = MGPSTRA3Module(config.hidden_size, config.max_token_length)
        self.wp_a3_module = MGPSTRA3Module(config.hidden_size, config.max_token_length)

        self.char_head = nn.Linear(config.hidden_size, config.num_character_labels)
        self.bpe_head = nn.Linear(config.hidden_size, config.num_bpe_labels)
        self.wp_head = nn.Linear(config.hidden_size, config.num_wordpiece_labels)

    @add_start_docstrings_to_model_forward(MGP_STR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MGPSTRModelOutput, config_class=MGPSTRConfig)
    def forward(
        self,
        pixel_values,
        output_attentions=False,
        output_a3_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        r"""
        Returns:
            `List[torch.FloatTensor]`: The list of logits output of char, bpe, wp. `List[torch.FloatTensor]`: The list
            of attention output of char, bpe, wp.

        Example:

        ```python
        >>> from transformers import (
        ...     MGPSTRProcessor,
        ...     MGPSTRForSceneTextRecognition,
        ... )
        >>> import requests
        >>> from PIL import Image

        >>> # load image from the IIIT-5k dataset
        >>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> processor = MGPSTRProcessor.from_pretrained("alibaba-damo/mgp-str-base")
        >>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

        >>> model = MGPSTRForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

        >>> # inference
        >>> outputs = model(pixel_values)
        >>> out_strs = processor.batch_decode(outputs.logits)
        >>> out_strs["generated_text"]
        '['ticket']'
        ```"""
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
        return MGPSTRModelOutput(
            logits=all_logits,
            hidden_states=mgp_outputs.hidden_states,
            attentions=mgp_outputs.attentions,
            a3_attentions=all_a3_attentions,
        )