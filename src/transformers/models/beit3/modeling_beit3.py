# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BEiT3 model."""


import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from transformers import PreTrainedModel, add_start_docstrings
from transformers.activations import get_activation
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    ImageClassifierOutputWithNoAttention,
    SequenceClassifierOutput,
)
from transformers.models.beit3.configuration_beit3 import Beit3Config
from transformers.utils import ModelOutput, add_start_docstrings_to_model_forward, logging


logger = logging.get_logger(__name__)


BEIT3_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Beit3Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


BEIT3_FOR_VISUALREASONING_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        image1_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.
        image2_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.
        padding_mask (`torch.LongTensor` of shape `({0})`):
            Padding mask for input tokens , of same shape as `input_ids`
            - 1 indicates the token is **not masked**,
            - 0 indicates the token is **masked**.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. A classification loss is computed (Cross-Entropy) against these labels.

"""


BEIT3_FOR_IMAGE_CLASSIFICATION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the classification loss. Indices should be in `[0, ..., config.num_labels - 1]`. A
            classification loss is computed (Cross-Entropy) against these labels.
"""


BEIT3_FOR_CAPTIONING_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.
        language_masked_pos (`torch.LongTensor` of shape `({0})`):
            language_masked_pos for denoting tokens for captioning

            - 1 indicates the token is **Present**,
            - 0 indicates the token is **absent**.
        text_len (`torch.LongTensor` of shape `({0})`):
            Length of text for captioning
        incremental_state (`Dict`):
            A Dictionary containing the incremental states layerwise
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the classification loss. Indices should be in `[0, ..., config.num_labels - 1]`. A
            classification loss is computed (Cross-Entropy) against these labels.
"""


BEIT3_FOR_VQA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.
        padding_mask (`torch.LongTensor` of shape `({0})`):
            Padding mask for input tokens , of same shape as `input_ids`

            - 1 indicates the token is **not masked**,
            - 0 indicates the token is **masked**.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the classification loss. Indices should be in `[0, ..., config.num_labels - 1]`. A
            classification loss is computed (Cross-Entropy) against these labels.
"""


BEIT3_FOR_TEXT_RETRIEVAL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BeitImageProcessor.__call__`] for details.
        padding_mask (`torch.LongTensor` of shape `({0})`):
            Padding mask for input tokens , of same shape as `input_ids`

            - 1 indicates the token is **not masked**,
            - 0 indicates the token is **masked**.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class Beit3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim * 4)
        self.dense1 = nn.Linear(config.embed_dim * 4, config.embed_dim * 2)
        self.norm2 = nn.LayerNorm(config.embed_dim * 2)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(config.embed_dim * 2, config.num_labels)

    def forward(self, hidden_states):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.dense2(hidden_states)


class Beit3MultiwayFeedForwardNetwork(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.dim = 1
        self.first = module
        self.second = copy.deepcopy(module)
        self.second.reset_parameters()

    def forward(self, hidden_states, split_position=-1):
        if split_position == -1:
            return self.first(hidden_states)
        if split_position == 0:
            return self.second(hidden_states)
        text_hidden, image_hidden = torch.split(
            hidden_states,
            [split_position, hidden_states.size(self.dim) - split_position],
            dim=self.dim,
        )
        y1, y2 = self.first(text_hidden), self.second(image_hidden)
        return torch.cat([y1, y2], dim=self.dim)


class Beit3Linear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
        self.linear_2 = nn.Linear(config.embed_dim, config.embed_dim, bias=True)

    def forward(self, hidden_states, split_position=-1):
        if split_position == -1:
            return self.linear_1(hidden_states)
        if split_position == 0:
            return self.linear_2(hidden_states)
        text_hidden, image_hidden = torch.split(
            hidden_states,
            [split_position, hidden_states.size(1) - split_position],
            dim=1,
        )
        y1, y2 = self.linear_1(text_hidden), self.linear_2(image_hidden)
        return torch.cat([y1, y2], dim=1)


class Beit3LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(config.embed_dim, eps=config.layernorm_eps)
        self.layernorm_2 = nn.LayerNorm(config.embed_dim, eps=config.layernorm_eps)

    def forward(self, hidden_states, split_position=-1):
        if split_position == -1:
            return self.layernorm_1(hidden_states)

        if split_position == 0:
            return self.layernorm_2(hidden_states)

        text_hidden, image_hidden = torch.split(
            hidden_states,
            [split_position, hidden_states.size(1) - split_position],
            dim=1,
        )
        text_hidden = self.layernorm_1(text_hidden)
        image_hidden = self.layernorm_2(image_hidden)
        hidden_states = torch.cat([text_hidden, image_hidden], dim=1)
        return hidden_states


class Beit3Embedder(nn.Module):
    def __init__(self, modules, dim=1):
        super().__init__()
        self.dim = dim
        self.first = modules[0]
        self.second = modules[1]
        self.split_position = -1

    def forward(self, hidden_states, positions, multiway_split_position=None):
        if multiway_split_position is not None:
            self.split_position = multiway_split_position
        if self.split_position == -1:
            return self.first(hidden_states, positions=positions)
        if self.split_position == 0:
            return self.second(hidden_states, positions=positions)
        text_hidden, image_hidden = torch.split(
            hidden_states,
            [self.split_position, hidden_states.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        y1, y2 = self.first(text_hidden, positions=positions), self.second(image_hidden, positions=positions)
        return torch.cat([y1, y2], dim=self.dim)


class Beit3VisionEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__()
        img_size = (config.img_size, config.img_size)
        patch_size = (config.patch_size, config.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(config.in_chans, config.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

    def num_position_embeddings(self):
        if self.cls_token is None:
            return self.num_patches
        else:
            return self.num_patches + 1

    def forward(self, hidden_states: torch.Tensor, masked_position: bool = None) -> torch.Tensor:
        hidden_states = self.proj(hidden_states).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.size()

        if masked_position is not None:
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            mask_position = masked_position.unsqueeze(-1).type_as(mask_token)
            hidden_states = hidden_states * (1 - mask_position) + mask_token * mask_position

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)

        return hidden_states


class Beit3PositionalEmbedding(nn.Embedding):
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor = None,
    ):
        if positions is None:
            positions = torch.arange(2, hidden_states.size(1) + 2, device=hidden_states.device).long().unsqueeze(0)
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class Beit3MultiheadAttention(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.key_proj = Beit3Linear(config)
        self.value_proj = Beit3Linear(config)
        self.query_proj = Beit3Linear(config)
        self.out_proj = Beit3Linear(config)
        self.inner_attn_ln = Beit3LayerNorm(config) if config.subln else None
        self.dropout_module = torch.nn.Dropout(config.attention_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        incremental_state: Dict = None,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        relative_pos: torch.Tensor = None,
    ):
        batch_size, target_length, embed_dim = query.size()

        key_batch_size, src_len, _ = key.size()

        query = (
            (self.query_proj(query) * self.scaling)
            .view(batch_size, target_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = self.key_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = query.reshape(batch_size * self.num_heads, target_length, self.head_dim)
        key = key.reshape(batch_size * self.num_heads, src_len, self.head_dim)
        value = value.reshape(batch_size * self.num_heads, src_len, self.head_dim)

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(batch_size * self.num_heads, -1, self.head_dim)
                prev_value = incremental_state["prev_value"].view(batch_size * self.num_heads, -1, self.head_dim)
                key = torch.cat([prev_key, key], dim=1)
                value = torch.cat([prev_value, value], dim=1)
            incremental_state["prev_key"] = key.view(batch_size, self.num_heads, -1, self.head_dim)
            incremental_state["prev_value"] = value.view(batch_size, self.num_heads, -1, self.head_dim)
            src_len = key.size(1)

        attn_weights = torch.bmm(query, key.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, target_length, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_length, src_len)

        if relative_pos is not None:
            relative_pos = relative_pos.view(attn_weights.size())
            attn_weights = attn_weights + relative_pos

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, value)
        attn = attn.transpose(0, 1).reshape(target_length, batch_size, embed_dim).transpose(0, 1)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        attn = self.out_proj(attn)
        attn_weights = attn_weights.view(batch_size, self.num_heads, target_length, src_len).transpose(1, 0)

        return attn, attn_weights


@dataclass
class Beit3ModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        encoder_out (`torch.Tensor` of shape `(1,)`):
            Output of Encoder
        encoder_embedding (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            encoder embedding states
        encoder_padding_mask (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            Padding mask used in encoder
        hidden_states (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            Encoder hidden states
        multiway_split_position (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            Split position denoting the point of split between text and image
    """

    encoder_out: Optional[torch.Tensor] = None
    encoder_embedding: Optional[torch.FloatTensor] = None
    encoder_padding_mask: Optional[torch.FloatTensor] = None
    hidden_states: List[Any] = None


class Beit3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Beit3Config
    base_model_prefix = "beit3"
    main_input_name = "input_ids"

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
        elif isinstance(
            module,
            (
                Beit3ForVisualReasoning,
                Beit3ForImageTextRetrieval,
                Beit3ForVisualQuestionAnswering,
                Beit3ForImageClassification,
                Beit3ForCaptioning,
            ),
        ):
            module.beit3.text_embedding.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Beit3Encoder):
            module.gradient_checkpointing = value

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()


class Beit3FeedForwardNetwork(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.embed_dim
        self.activation_fn = get_activation(config.activation_fn)
        self.activation_dropout = torch.nn.Dropout(config.activation_dropout)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(self.embed_dim, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, self.embed_dim)
        self.ffn_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_eps) if config.subln else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states.float()).type_as(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        if self.ffn_layernorm is not None:
            hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states.view(x_shape)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Beit3EncoderLayer(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.embed_dim
        self.self_attn = Beit3MultiheadAttention(config)
        self.self_attn_layer_norm = Beit3LayerNorm(config)
        self.dropout_module = torch.nn.Dropout(config.dropout)

        self.normalize_before = config.normalize_before
        self.ffn_dim = config.hidden_size

        self.ffn = Beit3MultiwayFeedForwardNetwork(
            Beit3FeedForwardNetwork(config),
        )
        self.final_layer_norm = Beit3LayerNorm(config)
        self.alpha = 1.0

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
        self,
        hidden_states,
        encoder_padding_mask,
        attn_mask=None,
        relative_pos=None,
        multiway_split_position=None,
        incremental_state=None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = hidden_states
        split_position = multiway_split_position if multiway_split_position is not None else -1
        hidden_states = self.self_attn_layer_norm(hidden_states, split_position=split_position)
        hidden_states, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            relative_pos=relative_pos,
            incremental_state=incremental_state,
        )
        hidden_states = self.dropout_module(hidden_states)

        hidden_states = self.residual_connection(hidden_states, residual)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states, split_position=split_position)
        hidden_states = self.ffn(hidden_states, split_position=split_position)

        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states, split_position=split_position)
        return hidden_states


class Beit3Encoder(nn.Module):
    def __init__(
        self,
        config,
        embed_positions=None,
    ):
        super().__init__()

        self.config = config
        self.dropout_module = torch.nn.Dropout(config.dropout)
        self.embed_positions = embed_positions
        self.layers = nn.ModuleList([])

        for i in range(config.layers):
            self.layers.append(Beit3EncoderLayer(config))
        self.num_layers = len(self.layers)
        if config.normalize_before and config.encoder_normalize_before:
            self.fc_norm = Beit3LayerNorm(config)
        else:
            self.fc_norm = None
        self.relative_position = None

        if config.subln:
            init_scale = math.sqrt(math.log(config.layers * 2))
            for name, p in self.named_parameters():
                if "fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name:
                    p.data.mul_(init_scale)

    def forward_embedding(
        self,
        src_tokens,
        token_embedding=None,
        positions=None,
        multiway_split_position=None,
    ):
        x = embed = token_embedding
        if self.embed_positions is not None:
            if src_tokens is not None:
                x = embed + self.embed_positions(
                    src_tokens, positions=positions, multiway_split_position=multiway_split_position
                )
            else:
                x = embed + self.embed_positions(
                    x, positions=positions, multiway_split_position=multiway_split_position
                )
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        encoder_padding_mask=None,
        attn_mask=None,
        return_all_hiddens=True,
        token_embeddings=None,
        multiway_split_position=None,
        incremental_state=None,
        positions=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_padding_mask is None:
            if src_tokens is not None:
                encoder_padding_mask = torch.zeros_like(src_tokens, device=src_tokens.device).bool()
            else:
                encoder_padding_mask = torch.zeros(
                    [token_embeddings.size(0), token_embeddings.size(1)],
                    device=token_embeddings.device,
                ).bool()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings, positions, multiway_split_position)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        hidden_states = []

        if return_all_hiddens:
            hidden_states.append(x)

        relative_pos_bias = None
        if self.relative_position is not None:
            relative_pos_bias = self.relative_position(batch_size=x.size(0), qlen=x.size(1), klen=x.size(1))

        # incremental_state is not None during inference if we use the bidirectional encoder as a generator as in s2s-ft (https://arxiv.org/abs/2110.13640)
        for idx, layer in enumerate(self.layers):
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if incremental_state is None else None,
                attn_mask=attn_mask,
                relative_pos=relative_pos_bias,
                multiway_split_position=multiway_split_position,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
            )
            if return_all_hiddens:
                hidden_states.append(x)

        if self.fc_norm is not None:
            x = self.fc_norm(x)

        if not return_dict:
            return [x, encoder_embedding, hidden_states]

        return Beit3ModelOutput(
            encoder_out=x,
            encoder_embedding=encoder_embedding,
            # encoder_padding_mask=encoder_padding_mask,
            hidden_states=hidden_states,
        )


@add_start_docstrings(
    """Beit3 is a multimodal foundation model, It achieves big convergence from bachbone architecture, pretraining task
    and model scaling. The key idea in BEiT-3 is to model images as another language. Beit3 uses multiway Transformers
    architecture which uses a shared self-attention module.""",
    BEIT3_START_DOCSTRING,
)
class Beit3Model(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.text_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.vision_embedding = Beit3VisionEmbedding(config)
        embed_positions = Beit3Embedder(
            modules=[
                Beit3PositionalEmbedding(self.vision_embedding.num_position_embeddings() + 2, config.embed_dim),
                Beit3PositionalEmbedding(config.max_source_positions, config.embed_dim),
            ],
            dim=1,
        )
        self.encoder = Beit3Encoder(
            config,
            embed_positions=embed_positions,
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.text_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.text_embedding = value

    def get_num_layers(self):
        return self.encoder.num_layers

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        text_padding_position=None,
        attn_mask=None,
        vision_masked_position=None,
        incremental_state=None,
        positions=None,
        return_dict=None,
        output_hidden_states=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is None:
            x = self.vision_embedding(pixel_values, vision_masked_position)
            encoder_padding_mask = None
            multiway_split_position = -1
        elif pixel_values is None:
            x = self.text_embedding(input_ids)
            encoder_padding_mask = text_padding_position
            multiway_split_position = 0
        else:
            x1 = self.vision_embedding(pixel_values, vision_masked_position)
            multiway_split_position = x1.size(1)
            x2 = self.text_embedding(input_ids)
            x = torch.cat([x1, x2], dim=1)

            if text_padding_position is not None:
                encoder_padding_mask = torch.cat(
                    [
                        torch.zeros(x1.shape[:-1]).to(x1.device).bool(),
                        text_padding_position,
                    ],
                    dim=1,
                )
            else:
                encoder_padding_mask = None
        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position,
            incremental_state=incremental_state,
            positions=positions,
            return_dict=return_dict,
        )
        if not return_dict:
            encoder_out.append(multiway_split_position)
        else:
            encoder_out["multiway_split_position"] = multiway_split_position

        return encoder_out


@add_start_docstrings(
    """Beit3ForVisualReasoning has a MLP head on top of Beit3Model. Beit3 is a multimodal foundation model, It achieves
    big convergence from bachbone architecture, pretraining task and model scaling. The key idea in BEiT-3 is to model
    images as another language. Beit3 uses multiway Transformers architecture which uses a shared self-attention
    module.""",
    BEIT3_START_DOCSTRING,
)
class Beit3ForVisualReasoning(Beit3PreTrainedModel):
    def __init__(self, config):
        super(Beit3ForVisualReasoning, self).__init__(config)
        self.beit3 = Beit3Model(config)
        self.classifier = Beit3MLP(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT3_FOR_VISUALREASONING_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        image1_values,
        image2_values,
        padding_mask,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        batch_size = input_ids.size()[0]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_input = torch.cat((image1_values, image2_values), dim=0)
        language_input = torch.cat((input_ids, input_ids), dim=0)
        padding_mask = torch.cat((padding_mask, padding_mask), dim=0)

        outputs = self.beit3(
            input_ids=language_input,
            pixel_values=vision_input,
            text_padding_position=padding_mask,
        )
        x = outputs.encoder_out
        multiway_split_position = outputs.multiway_split_position

        vision_cls = x[:, 0, :]
        language_cls = x[:, multiway_split_position, :]
        cls_rep = torch.cat((vision_cls, language_cls), dim=-1)
        a, b = torch.split(cls_rep, split_size_or_sections=[batch_size, batch_size], dim=0)
        cls_rep = torch.cat((a, b), dim=-1)

        logits = self.classifier(cls_rep)
        reshaped_logits = logits.contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (reshaped_logits,) + (outputs.hidden_states,) if output_hidden_states else (reshaped_logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
        )


@add_start_docstrings(
    """Beit3ForImageClassification has a Linear head on top of Beit3Model for classification. Beit3 is a multimodal
    foundation model, It achieves big convergence from bachbone architecture, pretraining task and model scaling. The
    key idea in BEiT-3 is to model images as another language. Beit3 uses multiway Transformers architecture which uses
    a shared self-attention module.""",
    BEIT3_START_DOCSTRING,
)
class Beit3ForImageClassification(Beit3PreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config):
        super(Beit3ForImageClassification, self).__init__(config)
        embed_dim = config.embed_dim
        self.beit3 = Beit3Model(config)
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.num_labels = config.num_labels
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT3_FOR_IMAGE_CLASSIFICATION_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[Any], ImageClassifierOutputWithNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.beit3(input_ids=None, pixel_values=pixel_values)
        encoder_out = encoder_outputs.encoder_out
        logits = self.classifier(self.fc_norm(encoder_out[:, 1:, :].mean(1)))
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,)
            output = (output + (encoder_outputs.hidden_states,)) if output_hidden_states else output
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """Beit3ForCaptioning has a Linear head on top of Beit3Model for Image captioning . Beit3 is a multimodal
    foundation model, It achieves big convergence from bachbone architecture, pretraining task and model scaling. The
    key idea in BEiT-3 is to model images as another language. Beit3 uses multiway Transformers architecture which uses
    a shared self-attention module.""",
    BEIT3_START_DOCSTRING,
)
class Beit3ForCaptioning(Beit3PreTrainedModel):
    def __init__(self, config):
        super(Beit3ForCaptioning, self).__init__(config)
        embed_dim = config.embed_dim
        self.beit3 = Beit3Model(config)
        self.label_smoothing = config.label_smoothing
        self.mlm_classifier = nn.Linear(embed_dim, config.vocab_size)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction="none")
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT3_FOR_CAPTIONING_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        pixel_values,
        padding_mask,
        language_masked_pos,
        text_len=None,
        incremental_state=None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_len = text_len if text_len is not None else input_ids.size(1)
        image_len = self.beit3.vision_embedding.num_position_embeddings()
        max_len = text_len + image_len
        uni_mask = torch.zeros((max_len, max_len), dtype=torch.long, device=input_ids.device)
        i_start, i_end = 0, image_len
        t_start, t_end = image_len, max_len
        # triangle mask for caption to caption
        uni_mask[t_start:t_end, t_start:t_end] = torch.tril(
            torch.ones(text_len, text_len, dtype=torch.long, device=input_ids.device)
        )
        # full attention for caption to image
        uni_mask[t_start:t_end, i_start:i_end] = 1
        # full attention for image to image
        uni_mask[i_start:i_end, i_start:i_end] = 1
        uni_mask = 1 - uni_mask

        if incremental_state is not None:
            for idx in range(self.get_num_layers()):
                if idx not in incremental_state:
                    incremental_state[idx] = {}

        # for incremental decoding
        positions = None
        if pixel_values is None:
            uni_mask = uni_mask[-2:]
            padding_mask = None
            # start position (2 (fairseq starts at 2) + cur_position) is equal to text_len
            positions = (
                torch.arange(text_len, input_ids.size(1) + text_len, device=input_ids.device).long().unsqueeze(0)
            )

        outputs = self.beit3(
            input_ids=input_ids,
            pixel_values=pixel_values,
            text_padding_position=padding_mask,
            attn_mask=uni_mask,
            incremental_state=incremental_state,
            positions=positions,
        )
        if pixel_values is not None:
            text_feats = outputs.encoder_out[:, image_len:]
        else:
            text_feats = outputs.encoder_out

        if language_masked_pos is not None:
            text_feats = text_feats[language_masked_pos.bool()]

        logits = self.mlm_classifier(text_feats)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            eps = self.label_smoothing
            n_class = logits.size(1)
            one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = self.log_soft(logits)
            loss = self.kl(log_prb, one_hot).sum(1)
            loss = loss.mean()

        if not return_dict:
            output = (logits,)
            output = output + (outputs.hidden_states,) if output_hidden_states else output
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss if loss is not None else None,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class Beit3Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@add_start_docstrings(
    """Beit3ForVisualQuestionAnswering has a Linear head on top of Beit3Model for visual question answering . Beit3 is a
multimodal
    foundation model, It achieves big convergence from bachbone architecture, pretraining task and model scaling. The
    key idea in BEiT-3 is to model images as another language. Beit3 uses multiway Transformers architecture which uses
    a shared self-attention module.""",
    BEIT3_START_DOCSTRING,
)
class Beit3ForVisualQuestionAnswering(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.embed_dim
        self.num_labels = config.num_labels
        self.beit3 = Beit3Model(config)
        self.pooler = Beit3Pooler(config)
        self.pooler.apply(self._init_weights)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, config.num_labels),
        )
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT3_FOR_VQA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        pixel_values,
        padding_mask,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[Any], SequenceClassifierOutput]:
        encoder_outputs = self.beit3(
            input_ids=input_ids,
            pixel_values=pixel_values,
            text_padding_position=padding_mask,
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        x = encoder_outputs.encoder_out
        cls_rep = self.pooler(x)
        logits = self.classifier(cls_rep)
        reshaped_logits = logits.view(-1, self.num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            log_softmax = nn.LogSoftmax(dim=-1)
            reshaped_logits = log_softmax(reshaped_logits)
            loss = loss_fct(reshaped_logits, labels.contiguous())
        if not return_dict:
            output = (
                (reshaped_logits,) + (encoder_outputs.hidden_states,) if output_hidden_states else (reshaped_logits,)
            )
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=encoder_outputs.hidden_states,
        )


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class Beit3ImageTextMatchingModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        similarity (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
    """

    loss: Optional[torch.Tensor] = None
    text_hidden: Optional[torch.FloatTensor] = None
    image_hidden: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    """Beit 3 Model transformer with a 'language' modeling head on top. BEiT does masked image modeling by predicting
    visual tokens of a Vector-Quantize Variational Autoencoder (VQ-VAE), whereas other vision models like ViT and DeiT
    predict RGB pixel values. As a result, this class is incompatible with [`AutoModelForMaskedImageModeling`], so you
    will need to use [`BeitForMaskedImageModeling`] directly if you wish to do masked image modeling with BEiT.""",
    BEIT3_START_DOCSTRING,
)
class Beit3ForImageTextRetrieval(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.embed_dim
        self.beit3 = Beit3Model(config)
        self.language_classifier = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_classifier = nn.Linear(embed_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT3_FOR_TEXT_RETRIEVAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        padding_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Any], Beit3ImageTextMatchingModelOutput]:
        outputs = self.beit3(
            input_ids=None,
            pixel_values=pixel_values,
            text_padding_position=None,
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_out = outputs.encoder_out
        vision_cls = self.vision_classifier(vision_out[:, 0, :])
        vision_cls = F.normalize(vision_cls, dim=-1)

        outputs = self.beit3(
            input_ids=input_ids,
            pixel_values=None,
            text_padding_position=padding_mask,
        )
        text_out = outputs.encoder_out
        text_cls = self.language_classifier(text_out[:, 0, :])
        text_cls = F.normalize(text_cls, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(vision_cls, text_cls.t()) * logit_scale
        similarity = clip_loss(logits_per_text)

        if not return_dict:
            outputs = (similarity,)
            return (
                outputs
                + (
                    text_out,
                    vision_out,
                )
                if output_hidden_states
                else outputs
            )

        return Beit3ImageTextMatchingModelOutput(similarity, text_out, vision_out)
