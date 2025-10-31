# coding=utf-8
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BEiT3 model."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from transformers import PreTrainedModel
from transformers.activations import get_activation
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    ImageClassifierOutput,
    SequenceClassifierOutput,
)
from transformers.models.beit3.configuration_beit3 import Beit3Config
from transformers.utils import ModelOutput, logging

from ...utils import auto_docstring


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIPTextModel->Beit3Model, CLIPVisionModel->Beit3Model, CLIP->Beit3ImageTextMatching
class Beit3ImageTextMatchingOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for image-text similarity.
    logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
        The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
        The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
        similarity scores.
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`Beit3Model`].
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The image embeddings obtained by applying the projection layer to the pooled output of [`Beit3Model`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`Beit3Model`].
    vision_model_output (`BaseModelOutputWithPooling`):
        The output of the [`Beit3Model`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class Beit3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Concatenation of two image embeddings and two text embeddings
        in_features = config.hidden_size * 4
        hidden_features = config.hidden_size * 2

        self.norm1 = nn.LayerNorm(in_features, eps=config.layer_norm_eps)

        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features, eps=config.layer_norm_eps)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, config.num_labels)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.dense2(hidden_states)


class Beit3FeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.activation_fn = get_activation(config.activation_fn)
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        self.ffn_layernorm = (
            LayerNorm(config.intermediate_size, eps=config.layer_norm_eps) if config.sub_layernorm else None
        )

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


class Beit3MultiwayFeedForwardNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text = Beit3FeedForwardNetwork(config)
        self.image = Beit3FeedForwardNetwork(config)

    def forward(self, hidden_states: torch.Tensor, split_position: int = -1):
        if split_position == -1:
            return self.image(hidden_states)
        if split_position == 0:
            return self.text(hidden_states)
        image_hidden, text_hidden = torch.split(
            hidden_states,
            [split_position, hidden_states.size(1) - split_position],
            dim=1,
        )
        image_out, text_out = self.image(image_hidden), self.text(text_hidden)
        return torch.cat([image_out, text_out], dim=1)


class Beit3AttentionLinearProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.image = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor, split_position: int = -1):
        if split_position == -1:
            return self.image(hidden_states)
        if split_position == 0:
            return self.text(hidden_states)
        image_hidden, text_hidden = torch.split(
            hidden_states,
            [split_position, hidden_states.size(1) - split_position],
            dim=1,
        )
        image_hidden, text_hidden = self.image(image_hidden), self.text(text_hidden)
        return torch.cat([image_hidden, text_hidden], dim=1)


class Beit3LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.image = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, split_position: int = -1):
        if split_position == -1:
            return self.image(hidden_states)

        if split_position == 0:
            return self.text(hidden_states)

        image_hidden, text_hidden = torch.split(
            hidden_states,
            [split_position, hidden_states.size(1) - split_position],
            dim=1,
        )
        text_hidden = self.text(text_hidden)
        image_hidden = self.image(image_hidden)
        hidden_states = torch.cat([image_hidden, text_hidden], dim=1)
        return hidden_states


class Beit3PositionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_size = (config.image_size, config.image_size)
        patch_size = (config.patch_size, config.patch_size)
        self.num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) + 1

        # being consistent with original implementation with Fairseq, which starts from 2 for position embedding
        self.image = nn.Embedding(self.num_patches + 2, config.hidden_size)
        self.text = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, text_end_position: int, multiway_split_position: int = -1):
        if text_end_position is None:
            positions = torch.arange(2, hidden_states.size(1) + 2, device=hidden_states.device).long().unsqueeze(0)
        else:
            positions = text_end_position
        if multiway_split_position == -1:
            return self.image(positions)
        if multiway_split_position == 0:
            return self.text(positions)
        image_hidden, text_hidden = torch.split(
            hidden_states,
            [multiway_split_position, hidden_states.size(1) - multiway_split_position],
            dim=1,
        )
        if text_end_position is None:
            text_positions = torch.arange(2, text_hidden.size(1) + 2, device=text_hidden.device).long().unsqueeze(0)
            image_positions = torch.arange(2, image_hidden.size(1) + 2, device=image_hidden.device).long().unsqueeze(0)
        else:
            text_positions = text_end_position
            image_positions = text_end_position

        image_representatations, text_representations = self.image(image_positions), self.text(text_positions)
        return torch.cat([image_representatations, text_representations], dim=1)


class Beit3VisionEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__()
        image_size = (config.image_size, config.image_size)
        patch_size = (config.patch_size, config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.num_position_embeddings = self.num_patches + 1

    def forward(self, hidden_states: torch.Tensor, masked_position: bool | None = None) -> torch.Tensor:
        hidden_states = self.projection(hidden_states).flatten(2).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.size()

        if masked_position is not None:
            mask_token = self.mask_token.expand(batch_size, seq_len, -1)
            mask_position = masked_position.unsqueeze(-1).type_as(mask_token)
            hidden_states = hidden_states * (1 - mask_position) + mask_token * mask_position

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)

        return hidden_states


class Beit3MultiheadAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add layer norm after performing self-attention (as explained in the Beit3 paper).
    """

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.key_proj = Beit3AttentionLinearProjection(config)
        self.value_proj = Beit3AttentionLinearProjection(config)
        self.query_proj = Beit3AttentionLinearProjection(config)
        self.out_proj = Beit3AttentionLinearProjection(config)
        self.inner_attn_ln = Beit3LayerNorm(config) if config.sub_layernorm else None
        self.dropout_module = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        attention_mask: torch.Tensor = None,
        image_text_attention_mask: torch.Tensor = None,
        multiway_split_position=-1,
        output_attentions: bool | None = None,
    ):
        batch_size, target_length, embed_dim = query.size()

        _, src_len, _ = key.size()

        query = (
            (self.query_proj(query, split_position=multiway_split_position) * self.scaling)
            .view(batch_size, target_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key_proj(key, split_position=multiway_split_position)
            .view(batch_size, src_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value_proj(value, split_position=multiway_split_position)
            .view(batch_size, src_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        query = query.reshape(batch_size * self.num_heads, target_length, self.head_dim)
        key = key.reshape(batch_size * self.num_heads, src_len, self.head_dim)
        value = value.reshape(batch_size * self.num_heads, src_len, self.head_dim)

        if past_key_values is not None:
            prev_key = past_key_values[0].view(batch_size * self.num_heads, -1, self.head_dim)
            prev_value = past_key_values[1].view(batch_size * self.num_heads, -1, self.head_dim)
            key = torch.cat([prev_key, key], dim=1)
            value = torch.cat([prev_value, value], dim=1)
            past_key_values[0] = key.view(batch_size, self.num_heads, -1, self.head_dim)
            past_key_values[1] = value.view(batch_size, self.num_heads, -1, self.head_dim)
            src_len = key.size(1)

        attn_weights = torch.bmm(query, key.transpose(1, 2))

        if image_text_attention_mask is not None:
            image_text_attention_mask = image_text_attention_mask.unsqueeze(0)
            attn_weights += image_text_attention_mask

        if attention_mask is not None:
            attention_mask = 1 - attention_mask.type_as(query)

            attn_weights = attn_weights.view(batch_size, self.num_heads, target_length, src_len)
            attn_weights = attn_weights.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, target_length, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_probs, value)
        attn = attn.transpose(0, 1).reshape(target_length, batch_size, embed_dim).transpose(0, 1)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn, split_position=multiway_split_position)

        attn = self.out_proj(attn, split_position=multiway_split_position)

        outputs = (attn, attn_probs) if output_attentions else (attn,)

        return outputs


class Beit3EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Beit3MultiheadAttention(config)
        self.self_attn_layer_norm = Beit3LayerNorm(config)
        self.dropout_module = nn.Dropout(config.dropout)

        self.normalize_before = config.sub_layernorm
        self.ffn_dim = config.intermediate_size

        self.ffn = Beit3MultiwayFeedForwardNetwork(config)
        self.final_layer_norm = Beit3LayerNorm(config)
        self.alpha = 1.0

    def forward(
        self,
        hidden_states,
        attention_mask,
        image_text_mask=None,
        multiway_split_position=None,
        past_key_values=None,
        output_attentions=None,
    ):
        if image_text_mask is not None:
            image_text_mask = image_text_mask.masked_fill(image_text_mask.to(torch.bool), -1e8)

        residual = hidden_states
        split_position = multiway_split_position if multiway_split_position is not None else -1
        hidden_states = self.self_attn_layer_norm(hidden_states, split_position=split_position)
        output = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            image_text_attention_mask=image_text_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            multiway_split_position=split_position,
            output_attentions=output_attentions,
        )

        attention_weights = None
        if output_attentions:
            hidden_states, attention_weights = output
        else:
            hidden_states = output[0]
        hidden_states = self.dropout_module(hidden_states)

        hidden_states = residual * self.alpha + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states, split_position=split_position)
        hidden_states = self.ffn(hidden_states, split_position=split_position)

        hidden_states = residual * self.alpha + hidden_states
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states, split_position=split_position)
        if output_attentions:
            return hidden_states, attention_weights
        return hidden_states


class Beit3Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.dropout_module = nn.Dropout(config.dropout)
        self.embed_positions = Beit3PositionEmbeddings(config)

        self.layers = nn.ModuleList([Beit3EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.fc_norm = Beit3LayerNorm(config) if config.sub_layernorm and config.encoder_normalize_before else None

        self.gradient_checkpointing = False

    def add_position_embeddings(
        self,
        hidden_state,
        text_end_positions=None,
        multiway_split_position=None,
    ):
        if self.embed_positions is not None:
            hidden_state = hidden_state + self.embed_positions(
                hidden_state, text_end_position=text_end_positions, multiway_split_position=multiway_split_position
            )
        hidden_state = self.dropout_module(hidden_state)
        return hidden_state

    def forward(
        self,
        hidden_state,
        attention_mask=None,
        image_text_mask=None,
        output_hidden_states=None,
        output_attentions=None,
        multiway_split_position=None,
        past_key_values=None,
        text_end_positions=None,
        return_dict=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        hidden_state = self.add_position_embeddings(hidden_state, text_end_positions, multiway_split_position)
        hidden_state = hidden_state * (attention_mask.unsqueeze(-1).type_as(hidden_state))

        # past_key_values is not None during inference if we use the bidirectional encoder as a generator as in s2s-ft (https://arxiv.org/abs/2110.13640)
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            encoder_layer_outputs = layer(
                hidden_state,
                attention_mask=attention_mask if past_key_values is None else None,
                image_text_mask=image_text_mask,
                multiway_split_position=multiway_split_position,
                past_key_values=past_key_values[idx] if past_key_values is not None else None,
                output_attentions=output_attentions,
            )
            if output_attentions:
                hidden_state, attention_weights = encoder_layer_outputs
                all_self_attentions = all_self_attentions + (attention_weights,)
            else:
                hidden_state = encoder_layer_outputs

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if self.fc_norm is not None:
            hidden_state = self.fc_norm(hidden_state, split_position=multiway_split_position)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_state,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


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
                Beit3ForImagesAndTextClassification,
                Beit3ForImageTextRetrieval,
                Beit3ForQuestionAnswering,
                Beit3ForImageClassification,
                Beit3ForCaptioning,
            ),
        ):
            module.beit3.text_embedding.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Beit3Encoder):
            module.gradient_checkpointing = value


@auto_docstring(
    custom_intro="""
    BEiT-3 is a general-purpose multimodal foundation model that excels in both vision and vision-language tasks. It
        utilizes [Multiway transformers] (https://arxiv.org/abs/2208.10442) for deep fusion and modality-specific
        encoding, and unifies masked modeling on images, texts, and image-text pairs, achieving top performance on
        multiple benchmarks.
    """
)
class Beit3Model(Beit3PreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)

        self.text_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vision_embedding = Beit3VisionEmbedding(config)
        self.encoder = Beit3Encoder(config)

        self.pooler = Beit3Pooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        return self.text_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.text_embedding = value

    def get_num_layers(self):
        return self.encoder.num_layers

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        image_text_mask: Optional[torch.FloatTensor] = None,
        vision_masked_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        output_hidden_states: Optional[torch.LongTensor] = None,
        output_attentions: Optional[torch.LongTensor] = None,
        return_dict: Optional[torch.LongTensor] = None,
    ):
        r"""
        Examples:

        ```python
        >>> from transformers import Beit3Processor, Beit3Model
        >>> from PIL import Image
        >>> import requests

        >>> processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")
        >>> model = Beit3Model.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "This is photo of a cat"

        >>> inputs = processor(text=text, images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # last hidden states have a sequence length equal to
        >>> # the number of text tokens + number of image patches + 1 (for the CLS token)
        >>> # which in this case equals 8 + (224//16) + 1 = 205
        >>> last_hidden_state = outputs.last_hidden_state
        >>> print(last_hidden_state.shape)
        torch.Size([1, 205, 768])
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_end_positions = None
        if input_ids is None and pixel_values is None:
            raise ValueError("You have to specify at least input_ids or pixel_values")

        if input_ids is None:
            embeddings = self.vision_embedding(pixel_values, vision_masked_position)
            multiway_split_position = -1
        elif pixel_values is None:
            embeddings = self.text_embedding(input_ids)
            multiway_split_position = 0
        else:
            vision_embeddings = self.vision_embedding(pixel_values, vision_masked_position)
            multiway_split_position = vision_embeddings.size(1)
            text_embeddings = self.text_embedding(input_ids)
            embeddings = torch.cat([vision_embeddings, text_embeddings], dim=1)

            if attention_mask is not None:
                # Shape of ones_for_vision_padding is (num_image,((config.image_size/config.patch_size)^2) + 1) .
                # + 1 for CLS token
                ones_for_vision_padding = torch.ones(vision_embeddings.shape[:-1]).to(vision_embeddings.device).bool()
                attention_mask = torch.cat([ones_for_vision_padding, attention_mask], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device).bool()

        encoder_outputs = self.encoder(
            hidden_state=embeddings,
            attention_mask=attention_mask,
            image_text_mask=image_text_mask,
            multiway_split_position=multiway_split_position,
            past_key_values=past_key_values,
            text_end_positions=text_end_positions,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            output = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return output + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Beit3ForImagesAndTextClassification with a MLP head on top of Beit3Model, for images and text classification tasks
    such as NLVR2
    """
)
class Beit3ForImagesAndTextClassification(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.beit3 = Beit3Model(config)
        self.classifier = Beit3MLP(config)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import Beit3Processor, Beit3ForImagesAndTextClassification
        >>> from PIL import Image
        >>> import requests
        >>> import torch

        >>> model = Beit3ForImagesAndTextClassification.from_pretrained("Raghavan/beit3_base_patch16_224_nlvr2")

        >>> right_image_url = "https://i.ytimg.com/vi/DtTND8frecg/hqdefault.jpg"
        >>> right_image = Image.open(requests.get(right_image_url, stream=True).raw)
        >>> left_image_url = "https://cevgroup.org/wp-content/uploads/2018/10/WAP-7_class_electric_locomotive_of_Indian_Railways.jpg"
        >>> left_image = Image.open(requests.get(left_image_url, stream=True).raw)

        >>> text = "Power lines can be seen above the train in the image on the right."

        >>> processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_224_nlvr2")
        >>> inputs = processor(text=text, images=[left_image, right_image], return_tensors="pt")
        >>> outputs = model(
        ...     input_ids=inputs["input_ids"],
        ...     pixel_values=inputs["pixel_values"].unsqueeze(0),
        ... )
        >>> predicted_class_idx = outputs.logits.argmax(-1).item()
        >>> predicted_class = model.config.id2label[predicted_class_idx]
        >>> print("Predicted class:", predicted_class)
        Predicted class: False
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.size()[0]
        image1_values, image2_values = pixel_values.split(1, dim=1)
        image1_values = image1_values.squeeze(1)
        image2_values = image2_values.squeeze(1)
        vision_input = torch.cat((image1_values, image2_values), dim=0)
        language_input = torch.cat((input_ids, input_ids), dim=0)

        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape)

        attention_mask = torch.cat((attention_mask, attention_mask), dim=0)

        outputs = self.beit3(
            input_ids=language_input,
            pixel_values=vision_input,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        multiway_split_position = (self.config.image_size // self.config.patch_size) ** 2 + 1

        vision_cls = last_hidden_state[:, 0, :]
        language_cls = last_hidden_state[:, multiway_split_position, :]
        cls_rep = torch.cat((vision_cls, language_cls), dim=-1)
        vision_cls_rep, language_cls_rep = torch.split(cls_rep, split_size_or_sections=[batch_size, batch_size], dim=0)
        cls_rep = torch.cat((vision_cls_rep, language_cls_rep), dim=-1)

        logits = self.classifier(cls_rep)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Beit3ForImageClassification has a Linear head on top of Beit3Model for classification. Beit3 is a multimodal
    foundation model, The key idea in BEiT-3 is to model images as another language. Beit3 uses multiway Transformers
    architecture which uses a shared self-attention module.
    """
)
class Beit3ForImageClassification(Beit3PreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.hidden_size
        self.beit3 = Beit3Model(config)
        self.fc_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(embed_dim, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.num_labels = config.num_labels
        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[Any], ImageClassifierOutput]:
        r"""
        Examples:

        ```python
        >>> from transformers import Beit3Processor, Beit3ForImageClassification
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")
        >>> model = Beit3ForImageClassification.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> predicted_class_idx = outputs.logits.argmax(-1).item()
        >>> predicted_class = model.config.id2label[predicted_class_idx]
        >>> print("Predicted class:", predicted_class)
        Predicted class: remote control, remote
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit3(
            input_ids=None,
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state if return_dict else outputs[0]
        patch_tokens = last_hidden_state[:, 1:, :]
        logits = self.classifier(self.fc_norm(patch_tokens.mean(1)))

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Beit3ForCaptioning has a linear head on top of Beit3Model for image captioning. BEiT-3 is a multimodal
    foundation model, The key idea in BEiT-3 is to model images as another language. Beit3 uses multiway Transformers
    architecture which uses a shared self-attention module.
    """
)
class Beit3ForCaptioning(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.beit3 = Beit3Model(config)
        self.mlm_classifier = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        language_masked_pos: Optional[torch.LongTensor] = None,
        text_len: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        r"""
        Examples:

        ```python
        >>> from transformers import Beit3ForCaptioning, Beit3Processor
        >>> from PIL import Image
        >>> import requests
        >>> import torch
        >>> import numpy as np
        >>> from io import BytesIO

        >>> url = (
        ...     "https://cdn.britannica.com/79/232779-050-6B0411D7/German-Shepherd-dog-Alsatian.jpg"
        ... )
        >>> image = Image.open(BytesIO(requests.get(url).content))

        >>> model = Beit3ForCaptioning.from_pretrained("Raghavan/beit3_base_patch16_480_coco_captioning")

        >>> processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_480_coco_captioning")
        >>> inputs = processor(text=["This is photo of a dog"], images=image, return_tensors="pt")

        >>> language_masked_pos = torch.zeros_like(inputs.input_ids)
        >>> # Mask the language_masked_pos on position of dog token (token to be filled)
        >>> language_masked_pos[:, 6] = 1
        >>> # Set the token to be filled with the special token id 64001
        >>> inputs.input_ids[:, 6] = 64001
        >>> output = model(
        ...     input_ids=inputs.input_ids,
        ...     pixel_values=inputs.pixel_values,
        ...     attention_mask=torch.ones_like(language_masked_pos),
        ...     language_masked_pos=language_masked_pos,
        ... )
        >>> processor.tokenizer.batch_decode([output.logits.argmax(-1)])[0]
        'dog'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_len = text_len if text_len is not None else input_ids.size(1)
        image_len = self.beit3.vision_embedding.num_position_embeddings
        max_len = text_len + image_len
        image_text_mask = torch.zeros((max_len, max_len), dtype=torch.long, device=input_ids.device)
        i_start, i_end = 0, image_len
        t_start, t_end = image_len, max_len
        # triangle mask for caption to caption
        image_text_mask[t_start:t_end, t_start:t_end] = torch.tril(
            torch.ones(text_len, text_len, dtype=torch.long, device=input_ids.device)
        )
        # full attention for caption to image
        image_text_mask[t_start:t_end, i_start:i_end] = 1
        # full attention for image to image
        image_text_mask[i_start:i_end, i_start:i_end] = 1
        image_text_mask = 1 - image_text_mask

        outputs = self.beit3(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            image_text_mask=image_text_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        text_feats = outputs.last_hidden_state[:, image_len:]

        if language_masked_pos is not None:
            text_feats = text_feats[language_masked_pos.bool()]

        logits = self.mlm_classifier(text_feats)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported for Beit3ForCaptioning.")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Beit3Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_state):
        cls_embedding = hidden_state[:, 0, :]
        cls_embedding = self.norm(cls_embedding)
        pooled_output = self.dense(cls_embedding)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@auto_docstring(
    custom_intro="""
    BEiT-3 model with a linear head on top for visual question answering. BEiT-3 is a
    multimodal foundation model. The key idea in BEiT-3 is to model images as another language. Beit3 uses a multiway
    Transformers architecture which uses a shared self-attention module.
    """
)
class Beit3ForQuestionAnswering(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.hidden_size
        self.num_labels = config.num_labels
        self.beit3 = Beit3Model(config, add_pooling_layer=True)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2, eps=config.layer_norm_eps),
            nn.GELU(),
            nn.Linear(embed_dim * 2, config.num_labels),
        )
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[Any], SequenceClassifierOutput]:
        r"""
        Examples:

        ```python
        >>> from transformers import Beit3Processor, Beit3ForQuestionAnswering
        >>> from PIL import Image
        >>> import requests

        >>> processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_480_vqa")
        >>> model = Beit3ForQuestionAnswering.from_pretrained("Raghavan/beit3_base_patch16_480_vqa")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> question = "How many cats are there?"

        >>> inputs = processor(text=question, images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> # model predicts one of the 3129 possible answers
        >>> predicted_answer_idx = logits.argmax(-1).item()
        >>> predicted_answer = model.config.id2label[predicted_answer_idx]
        >>> print("Predicted answer:", predicted_answer)
        Predicted answer: 2
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit3(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.KLDivLoss(reduction="batchmean")
            log_softmax = nn.LogSoftmax(dim=-1)
            logits = log_softmax(logits)
            loss = loss_fct(logits, labels.contiguous())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@auto_docstring(
    custom_intro="""
    BEiT-3 Transformer model with language and image classifier heads on top for image-text retrieval.
    """
)
class Beit3ForImageTextRetrieval(Beit3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.hidden_size
        self.beit3 = Beit3Model(config)
        self.language_classifier = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_classifier = nn.Linear(embed_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[Any], Beit3ImageTextMatchingOutput]:
        r"""
        Examples:

        ```python
        >>> from transformers import Beit3Processor, Beit3ForImageTextRetrieval
        >>> from PIL import Image
        >>> import requests

        >>> processor = Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_384_coco_retrieval")
        >>> model = Beit3ForImageTextRetrieval.from_pretrained("Raghavan/beit3_base_patch16_384_coco_retrieval")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["This is photo of a cat", "This is photo of a dog"], images=[image, image], return_tensors="pt"
        ... )

        >>> outputs = model(
        ...     input_ids=inputs["input_ids"],
        ...     pixel_values=inputs["pixel_values"],
        ...     return_loss=True,
        ... )

        >>> loss = outputs.loss.detach().numpy()
        >>> print(round(float(outputs.loss.detach().numpy()), 4))
        1.8435
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # forward images through the model
        vision_outputs = self.beit3(
            input_ids=None,
            pixel_values=pixel_values,
            attention_mask=None,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        vision_last_hidden_state = vision_outputs.last_hidden_state if return_dict else vision_outputs[0]
        image_embeds = self.vision_classifier(vision_last_hidden_state[:, 0, :])

        # forward text through the model
        text_outputs = self.beit3(
            input_ids=input_ids,
            pixel_values=None,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        text_last_hidden_state = text_outputs.last_hidden_state if return_dict else text_outputs[0]
        text_embeds = self.language_classifier(text_last_hidden_state[:, 0, :])

        # normalized features
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(image_embeds, text_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return Beit3ImageTextMatchingOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


__all__ = [
    "Beit3PreTrainedModel",
    "Beit3Model",
    "Beit3ForImagesAndTextClassification",
    "Beit3ForImageClassification",
    "Beit3ForCaptioning",
    "Beit3ForQuestionAnswering",
    "Beit3ForImageTextRetrieval",
]
