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

import copy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..auto import AutoModel
from ..convnextv2.modeling_convnextv2 import ConvNextV2DropPath
from ..internvl.modeling_internvl import (
    InternVLCausalLMOutputWithPast,
    InternVLForConditionalGeneration,
    InternVLModel,
    InternVLModelOutputWithPast,
    InternVLMultiModalProjector,
    InternVLPreTrainedModel,
    InternVLVisionAttention,
    InternVLVisionEmbeddings,
    InternVLVisionLayer,
    InternVLVisionMLP,
    InternVLVisionModel,
    InternVLVisionModelOutputWithPooling,
    InternVLVisionPatchEmbeddings,
    InternVLVisionPreTrainedModel,
    InternVLVisionRMSNorm,
)
from ..internvl.processing_internvl import InternVLProcessor
from ..internvl.video_processing_internvl import InternVLVideoProcessor
from ..qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from .configuration_intern_s1 import InternS1Config, InternS1VisionConfig


logger = logging.get_logger(__name__)


class InternS1Processor(InternVLProcessor):
    pass


class InternS1VideoProcessor(InternVLVideoProcessor):
    pass


class InternS1VisionRMSNorm(InternVLVisionRMSNorm):
    pass


class InternS1VisionAttention(InternVLVisionAttention):
    pass


class InternS1VisionEncoder(nn.Module):
    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__()
        self.config = config
        dpr = np.linspace(0.0, float(config.drop_path_rate), int(config.num_hidden_layers))
        dpr_configs = []
        for idx in range(config.num_hidden_layers):
            copy_config = copy.deepcopy(config)
            copy_config.drop_path_rate = dpr[idx]
            dpr_configs.append(copy_config)
        self.layers = nn.ModuleList([InternS1VisionLayer(dpr_configs[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, **kwargs)
        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class InternS1VisionPreTrainedModel(InternVLVisionPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": InternS1VisionEncoder,
        "attentions": InternS1VisionAttention,
    }


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


class InternS1VisionLayer(InternVLVisionLayer):
    def __init__(self, config: InternS1VisionConfig) -> None:
        super().__init__(config)

        # Note: Compared to InternVL, we have added support for drop_path to facilitate user fine-tuning.
        self.drop_path1 = InternS1DropPath(config.drop_path_rate)
        self.drop_path2 = InternS1DropPath(config.drop_path_rate)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        attention_output, _ = self.attention(
            self.layernorm_before(hidden_states),  # in InternS1Vision, layernorm is applied before self-attention
            **kwargs,
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

        return layer_output


class InternS1VisionModel(InternVLVisionModel):
    pass


class InternS1PreTrainedModel(InternVLPreTrainedModel):
    _can_compile_fullgraph = False


class InternS1MultiModalProjector(InternVLMultiModalProjector):
    pass


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
        if hasattr(config.text_config, "output_router_logits"):
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
                output_router_logits
                if output_router_logits is not None
                else self.config.text_config.output_router_logits
            )
            kwargs["output_router_logits"] = output_router_logits

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
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
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


class InternS1CausalLMOutputWithPast(InternVLCausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
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
        if hasattr(config.text_config, "output_router_logits"):
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> InternS1CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> torch_device = "auto"
        >>> processor = AutoProcessor.from_pretrained("internlm/Intern-S1-hf")
        >>> model = AutoModelForImageTextToText.from_pretrained(
        ...     "internlm/Intern-S1-hf", dtype=torch.bfloat16, device_map=torch_device
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
        >>> generate_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> print(processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if self.is_moe_model:
            output_router_logits = (
                output_router_logits
                if output_router_logits is not None
                else self.config.text_config.output_router_logits
            )
            kwargs["output_router_logits"] = output_router_logits

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
    "InternS1VideoProcessor",
    "InternS1Processor",
]
