# coding=utf-8
# Copyright 2022 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
""" PyTorch OwlViT model."""


from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_owlvit import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/owlvit-base"

OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/owlvit-base",
    # See all OwlViT models at https://huggingface.co/models?filter=owlvit
]



# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/OwlViT.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->owlvit
def owlvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->OwlViT
class OwlViTOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`OwlViTVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`OwlViTTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`OwlViTVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class OwlViTObjectDetectionOutput(ModelOutput):
    """
    Output type of [`OwlViTForObjectDetection`].
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~OwlViTFeatureExtractor.post_process`] to retrieve the unnormalized bounding
            boxes.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`OwlViTTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`OwlViTVisionModel`].
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, *optional*):
            Class embeddings of all image patches.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    class_embeds: torch.FloatTensor = None


class OwlViTVisionEmbeddings(nn.Module):
    def __init__(self, config: OwlViTVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_positions = (self.embed_dim // self.patch_size) ** 2 + 1
        self.position_embedding = nn.Parameter(torch.rand(self.num_positions, self.embed_dim))

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]

        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding
        return embeddings


class OwlViTTextEmbeddings(nn.Module):
    def __init__(self, config: OwlViTTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.rand(config.max_position_embeddings, embed_dim))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        embeddings = inputs_embeds + self.position_embedding
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->OwlViT
class OwlViTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->OwlViT
class OwlViTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->OwlViT
class OwlViTEncoderLayer(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OwlViTAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = OwlViTMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.clip.modeling_clip.CLIPPreTrainedModel with CLIP->OwlViT,clip->owlvit
class OwlViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OwlViTConfig
    base_model_prefix = "owlvit"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, OwlViTTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            nn.init.normal_(module.position_embedding, mean=0.0, std=factor * 0.02)
        elif isinstance(module, OwlViTVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.position_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, OwlViTAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, OwlViTMLP):
            factor = self.config.initializer_factor
            in_proj_std = (
                (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, OwlViTModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, OwlViTEncoder):
            module.gradient_checkpointing = value


OWLVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`OwlViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

OWLVIT_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

OWLVIT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

OWLVIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

OWLVIT_OBJ_DETECTION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`CLIPFeatureExtractor`]. See [`CLIPFeatureExtractor.__call__`] for details.
        input_ids (`torch.LongTensor` of shape `(batch_size, num_text_queries, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`OwlViTTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
"""

# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->OwlViT
class OwlViTEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`OwlViTEncoderLayer`].

    Args:
        config: OwlViTConfig
    """

    def __init__(self, config: OwlViTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([OwlViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class OwlViTTextTransformer(nn.Module):
    def __init__(self, config: OwlViTTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = OwlViTTextEmbeddings(config)
        self.encoder = OwlViTEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids)

        bsz, seq_len = input_shape
        # OWLVIT's text model uses causal mask, prepare it here.
        # https://github.com/openai/OWLVIT/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/owlvit/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len)
        mask.fill_(torch.tensor(float("-inf")))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


class OwlViTTextModel(OwlViTPreTrainedModel):
    config_class = OwlViTTextConfig

    def __init__(self, config: OwlViTTextConfig):
        super().__init__(config)
        self.text_model = OwlViTTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, OwlViTTextModel

        >>> model = OwlViTTextModel.from_pretrained("google/owlvit-base")
        >>> tokenizer = CLIPTokenizer.from_pretrained("google/owlvit-base")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class OwlViTVisionTransformer(nn.Module):
    def __init__(self, config: OwlViTVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = OwlViTVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim)
        self.encoder = OwlViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim)

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        if train:
            pooled_output = self.post_layernorm(pooled_output)
        else:
            pooled_output = self.post_layernorm(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class OwlViTVisionModel(OwlViTPreTrainedModel):
    config_class = OwlViTVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: OwlViTVisionConfig):
        super().__init__(config)
        self.vision_model = OwlViTVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, OwlViTVisionModel

        >>> model = OwlViTVisionModel.from_pretrained("google/owlvit-base")
        >>> processor = CLIPProcessor.from_pretrained("google/owlvit-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@add_start_docstrings(OWLVIT_START_DOCSTRING)
class OwlViTModel(OwlViTPreTrainedModel):
    config_class = OwlViTConfig

    def __init__(self, config: OwlViTConfig):
        super().__init__(config)

        if not isinstance(config.text_config, OwlViTTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type OwlViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, OwlViTVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type OwlViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = OwlViTTextTransformer(text_config)
        self.vision_model = OwlViTVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`OwlViTTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base")
        >>> tokenizer = CLIPTokenizer.from_pretrained("google/owlvit-base")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use OWLVIT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(OWLVIT_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: Optional[bool] = True,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`OwlViTVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base")
        >>> processor = CLIPProcessor.from_pretrained("google/owlvit-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use OWLVIT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        # Return projected output if in training mode
        if train:
            image_features = self.visual_projection(pooled_output)
        else:
            image_features = pooled_output
        return image_features

    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTOutput, config_class=OwlViTConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        normalize: Optional[bool] = True,
    ) -> Union[Tuple, OwlViTOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, OwlViTModel

        >>> model = OwlViTModel.from_pretrained("google/owlvit-base")
        >>> processor = CLIPProcessor.from_pretrained("google/owlvit-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use OWLVIT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        if normalize:
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text, logits_per_image = None, None
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = None
        if return_loss:
            loss = owlvit_loss(logits_per_text)

        if not return_dict:
            output = (text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return OwlViTOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class OwlViTBoxPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()

        width = config.vision_config.hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, 4)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output


class OwlViTClassPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()

        out_dim = config.text_config.hidden_size
        query_dim = config.vision_config.hidden_size

        self.dense0 = nn.Linear(query_dim, out_dim)
        self.logit_shift = nn.Linear(query_dim, 1)
        self.logit_scale = nn.Linear(query_dim, 1)
        self.elu = nn.ELU()

    def forward(
        self, 
        image_embeds: torch.FloatTensor, 
        query_embeds: torch.FloatTensor, 
        query_mask: torch.Tensor,
    ) -> Tuple[torch.FloatTensor]:

        image_class_embeds = self.dense0(image_embeds)

        # Normalize image and text features
        image_class_embeds /= torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        query_embeds /= torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6

        # Get class predictions
        pred_logits = torch.einsum('...pd,...qd->...pq', image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)
   
            pred_logits = pred_logits.to(torch.float64)
            pred_logits = torch.where(query_mask==0, -1e6, pred_logits)

        return (pred_logits, image_class_embeds)


class OwlViTImageTextEmbedder(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()

        self.clip = OwlViTModel(config)
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size)

    def forward(
        self, 
        pixel_values: Optional[torch.FloatTensor] = None, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        image_embeds, text_embeds = None, None

        # Encode text
        if input_ids is not None:
            text_embeds = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_embeds.unsqueeze(0)

        # Encode image 
        if pixel_values is not None:
            image_embeds = self.clip.get_image_features(pixel_values, train=False)

            # Resize class token
            new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
            class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

            # Merge image embedding with class tokens
            image_embeds = image_embeds[:, 1:, :] * class_token_out  
            image_embeds = self.layer_norm(image_embeds)

        return (image_embeds, text_embeds)


class OwlViTForObjectDetection(OwlViTPreTrainedModel):
    def __init__(self, config: OwlViTConfig):
        super().__init__(config)

        self._embedder = OwlViTImageTextEmbedder(config)
        self._class_head = OwlViTClassPredictionHead(config)
        self._box_head = OwlViTBoxPredictionHead(config)
        self.sigmoid = nn.Sigmoid()

    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):

        # Computes normalized xy corner coords from feature_map.
        assert feature_map.ndim == 4  # [B, H, W, C]
        h, w = feature_map.shape[1:3]

        xy = np.stack(np.meshgrid(np.arange(1, w+1), np.arange(1, h+1)), axis=-1).astype(np.float32)
        xy /= np.array([w, h], np.float32)

        # Flatten h, w dimensions
        xy = xy.reshape(*(xy.shape[:-3] + (-1, 2)))
        xy = torch.from_numpy(xy)
      
        return xy

    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:

        # The box center is biased to its position on the feature grid:
        xy = self.normalize_grid_corner_coordinates(feature_map)
        xy = torch.clip(xy, 0.0, 1.0)

        # Unnormalize xy 
        xy_bias = torch.log(xy + 1e-4) - torch.log1p(-xy + 1e-4)

        # The box size is biased to the patch size
        wh = torch.full_like(xy_bias, 1.0 / feature_map.shape[-2])
        wh_bias = torch.log(wh + 1e-4) - torch.log1p(-wh + 1e-4)

        # Compute box bias
        box_bias = torch.cat([xy_bias, wh_bias], dim=-1)
        return box_bias

    def box_predictor(
        self, 
        image_feats: torch.FloatTensor, 
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats: 
                Features extracted from the image, returned by the`embedder` function.
            feature_map: 
                A spatial re-arrangement of image_features, also returned by the `embedder` function.

        Returns:
            pred_boxes: 
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self._box_head(image_feats)
 
        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(
        self, 
        image_feats: torch.FloatTensor, 
        query_embeds: torch.FloatTensor, 
        query_mask: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Args:
            image_feats: 
                Features extracted from the image embedder.
            query_embeds: 
                Text query embeddings.
            query_mask: 
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        (pred_logits, image_class_embeds) = self._class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds)

    def image_embedder(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        # Returns a 2D map of image features.
        (image_embeds, _ ) = self._embedder(pixel_values=pixel_values)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0], 
            int(np.sqrt(image_embeds.shape[1])), 
            int(np.sqrt(image_embeds.shape[1])),  
            image_embeds.shape[-1]
        )
        image_embeds = image_embeds.reshape(new_size)

        return image_embeds

    def text_embedder(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.FloatTensor:

        # Returns text embeddings
        (_, text_feats) = self._embedder(input_ids=input_ids, attention_mask=attention_mask)

        return text_feats

    @add_start_docstrings_to_model_forward(OWLVIT_OBJ_DETECTION_INPUTS_DOCSTRING)
    def forward(
        self, 
        pixel_values: torch.FloatTensor, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> OwlViTObjectDetectionOutput:

        # Embed images
        feature_map = self.image_embedder(pixel_values)
        b, h, w, d = feature_map.shape
        image_feats = torch.reshape(feature_map, (b, h*w, d))

        # Embed text queries
        query_embeds = self.text_embedder(input_ids, attention_mask)

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.unsqueeze(0)
        query_mask = (input_ids[..., 0] > 0)

        # Predict object classes [batch_size, num_patches, num_queries+1]
        (pred_logits, class_embeds) = self.class_predictor(image_feats, query_embeds, query_mask)

        # Predict object boxes
        pred_boxes = self.box_predictor(image_feats, feature_map)

        return OwlViTObjectDetectionOutput(
            image_embeds=feature_map,
            text_embeds=query_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
        )
