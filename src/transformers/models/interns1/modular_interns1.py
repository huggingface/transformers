# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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


from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ..convnextv2.modeling_convnextv2 import ConvNextV2DropPath
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging
)
from ..auto import AutoModel
from ...modeling_outputs import BaseModelOutput
from ..qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from ..internvl.modeling_internvl import (InternVLPreTrainedModel,
                                          InternVLVisionRMSNorm,
                                          InternVLVisionAttention,
                                          InternVLVisionModelOutputWithPooling,
                                          InternVLVisionPatchEmbeddings,
                                          InternVLVisionMLP,
                                          InternVLVisionPreTrainedModel,
                                          InternVLMultiModalProjector,
                                          InternVLModelOutputWithPast,
                                          InternVLVisionEmbeddings,
                                          InternVLCausalLMOutputWithPast,
                                          InternVLForConditionalGeneration,
                                          InternVLModel)
from .configuration_interns1 import InternS1Config, InternS1VisionConfig

logger = logging.get_logger(__name__)


class InternS1VisionRMSNorm(InternVLVisionRMSNorm):
    pass


class InternS1VisionAttention(InternVLVisionAttention):
    pass


@auto_docstring
class InternS1VisionPreTrainedModel(InternVLVisionPreTrainedModel):
    config: InternS1VisionConfig


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of [`InternS1VisionModel`].
    """
)
class InternS1VisionModelOutputWithPooling(InternVLVisionModelOutputWithPooling):
    pass


class InternS1VisionPatchEmbeddings(InternVLVisionPatchEmbeddings):

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # Note: more robust
        embeddings = self.projection(pixel_values.to(self.projection.weight.dtype))
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)


class InternS1VisionEmbeddings(InternVLVisionEmbeddings):
    pass


class InternS1VisionMLP(InternVLVisionMLP):
    pass


class InternS1DropPath(ConvNextV2DropPath):
    pass


NORM2FN = {"layer_norm": nn.LayerNorm, "rms_norm": InternS1VisionRMSNorm}


class InternS1VisionLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: InternS1VisionConfig, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = InternS1VisionAttention(config)
        self.mlp = InternS1VisionMLP(config)
        # InternS1 uses different layernorm implementations for different models
        self.layernorm_before = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = NORM2FN[config.norm_type](config.hidden_size, eps=config.layer_norm_eps)

        init_values = config.layer_scale_init_value
        self.lambda_1 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.lambda_2 = nn.Parameter(init_values * torch.ones(config.hidden_size), requires_grad=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Note: Compared to InternVL, we have added support for drop_path to facilitate user fine-tuning.
        self.drop_path1 = InternS1DropPath(drop_path_rate)
        self.drop_path2 = InternS1DropPath(drop_path_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        attention_output, attention_weights = self.attention(
            self.layernorm_before(hidden_states),  # in InternS1Vision, layernorm is applied before self-attention
            output_attentions=output_attentions,
        )

        attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.drop_path1(attention_output) + hidden_states

        # in InternS1Vision, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.mlp(layer_output)
        layer_output = self.dropout(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path2(layer_output) + hidden_states

        return layer_output, attention_weights


class InternS1VisionEncoder(nn.Module):
    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__()
        self.config = config
        dpr = np.linspace(0.0, float(config.drop_path_rate), int(config.num_hidden_layers))
        self.layer = nn.ModuleList([
            InternS1VisionLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring
class InternS1VisionModel(InternS1VisionPreTrainedModel):
    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__(config)
        self.config = config

        self.embeddings = InternS1VisionEmbeddings(config)
        self.encoder = InternS1VisionEncoder(config)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[tuple, InternS1VisionModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output, _ = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        return InternS1VisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class InternS1PreTrainedModel(InternVLPreTrainedModel):
    pass


class InternS1MultiModalProjector(InternVLMultiModalProjector):
    pass


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for InternS1 outputs, with hidden states and attentions.
    """
)
class InternS1ModelOutputWithPast(InternVLModelOutputWithPast):
    r"""
    router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

        Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
        loss for Mixture of Experts models.
    """
    router_logits: Optional[tuple[torch.FloatTensor]] = None


class InternS1Model(InternVLModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: InternS1Config):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = InternS1MultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)

        # Note: We aim for InternS1 to support both dense and MoE configurations in its LLM component.
        self.is_moe_model = False
        if hasattr(config.text_config, 'output_router_logits'):
            self.is_moe_model = True

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> InternS1ModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if self.is_moe_model:
            output_router_logits = (
                output_router_logits if output_router_logits is not None else
                self.config.text_config.output_router_logits
            )
            kwargs['output_router_logits'] = output_router_logits

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id

            n_image_tokens = (special_image_mask).sum()
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_features = image_features.shape[0] * image_features.shape[1]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        return InternS1ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits if self.is_moe_model else None,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for InternS1 causal language model (or autoregressive) outputs.
    """
)
class InternS1CausalLMOutputWithPast(InternVLCausalLMOutputWithPast):
    r"""
    aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
        aux_loss for the sparse modules.
    router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

        Raw router logtis (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
        loss for Mixture of Experts models.
    """

    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None


class InternS1ForConditionalGeneration(InternVLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: InternS1Config):
        super().__init__(config)
        self.model = InternS1Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Note: We aim for InternS1 to support both dense and MoE configurations in its LLM component.
        self.is_moe_model = False
        if hasattr(config.text_config, 'output_router_logits'):
            self.is_moe_model = True
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, InternS1CausalLMOutputWithPast]:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> torch_device = "cuda"
        >>> processor = AutoProcessor.from_pretrained("internlm/Intern-S1")
        >>> model = AutoModelForImageTextToText.from_pretrained(
        ...     "internlm/Intern-S1", torch_dtype=torch.bfloat16, device_map=torch_device
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
        ...             },
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
        ...             },
        ...             {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(torch_device)
        >>> generate_ids = model.generate(**inputs, max_new_tokens=200)
        >>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if self.is_moe_model:
            output_router_logits = (
                output_router_logits if output_router_logits is not None else self.config.text_config.output_router_logits
            )
            kwargs['output_router_logits'] = output_router_logits

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        aux_loss = None
        if self.is_moe_model and output_router_logits and labels is not None:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

        return InternS1CausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits if self.is_moe_model else None,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "InternS1VisionPreTrainedModel",
    "InternS1VisionModel",
    "InternS1PreTrainedModel",
    "InternS1Model",
    "InternS1ForConditionalGeneration",
]
