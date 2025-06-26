# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

import torch
from torch import nn

from transformers.models.llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)

from ...activations import ACT2FN
from ...utils import auto_docstring, is_torchdynamo_compiling, logging
from .configuration_vipllava import VipLlavaConfig


logger = logging.get_logger(__name__)


class VipLlavaModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class VipLlavaCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class VipLlavaMultiModalProjector(nn.Module):
    def __init__(self, config: VipLlavaConfig):
        super().__init__()
        num_feature_layers = 1 if isinstance(config.vision_feature_layers, int) else len(config.vision_feature_layers)
        self.projector_layernorm = nn.LayerNorm(
            num_feature_layers * config.vision_config.hidden_size, eps=config.projector_layernorm_eps
        )

        self.linear_1 = nn.Linear(
            num_feature_layers * config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=True,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.projector_layernorm(hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class VipLlavaPreTrainedModel(LlavaPreTrainedModel):
    pass


class VipLlavaModel(LlavaModel):
    def get_image_features(
        self, pixel_values: torch.FloatTensor, vision_feature_layers: Optional[Union[int, list[int]]] = None
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
            vision_feature_layers (`Union[int, list[int]]`):
                The vision feature layer, or the list of indexes of the layers to select
                the vision feature.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_feature_layers = (
            vision_feature_layers if vision_feature_layers is not None else self.config.vision_feature_layers
        )
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        # If multiple feature layers are provided (which is usually the case)
        # then the image features are concatenated after the CLS is removed.
        if isinstance(vision_feature_layers, int):
            image_features = image_outputs.hidden_states[vision_feature_layers][:, 1:]
        else:
            # Usually, we select the features from index 1: the layers -2, -5, -8, -11 and 6
            image_features = [image_outputs.hidden_states[index][:, 1:] for index in vision_feature_layers]
            image_features = torch.cat(image_features, dim=-1)
        image_features = self.multi_modal_projector(image_features)
        return image_features

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layers: Optional[Union[int, list[int]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **lm_kwargs,
    ) -> Union[tuple, VipLlavaModelOutputWithPast]:
        r"""
        vision_feature_layers (`Union[int, list[int]]`, *optional*):
            The vision feature layer, or the list of indexes of the layers to select
            the vision feature.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layers = (
            vision_feature_layers if vision_feature_layers is not None else self.config.vision_feature_layers
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values, vision_feature_layers=vision_feature_layers
            )

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
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
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        output = VipLlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
        return output if return_dict else output.to_tuple()


class VipLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def get_image_features(
        self, pixel_values: torch.FloatTensor, vision_feature_layers: Optional[Union[int, list[int]]] = None
    ):
        return self.model.get_image_features(pixel_values=pixel_values, vision_feature_layers=vision_feature_layers)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layers: Optional[Union[int, list[int]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[tuple, VipLlavaCausalLMOutputWithPast]:
        r"""
        vision_feature_layers (`Union[int, list[int]]`, *optional*):
            The vision feature layer, or the list of indexes of the layers to select
            the vision feature.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, VipLlavaForConditionalGeneration

        >>> model = VipLlavaForConditionalGeneration.from_pretrained("llava-hf/vip-llava-7b-hf", device_map="auto", torch_dtype=torch.float16)
        >>> processor = AutoProcessor.from_pretrained("llava-hf/vip-llava-7b-hf")

        >>> prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{}###Assistant:"
        >>> question = "Can you please describe this image?"
        >>> prompt = prompt.format(question)
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/compel-neg.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=text, images=image, return_tensors="pt").to(0, torch.float16)

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=20)
        >>> processor.decode(generate_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        The image features a brown and white cat sitting on a green surface, with a red ball in its
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layers = (
            vision_feature_layers if vision_feature_layers is not None else self.config.vision_feature_layers
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            vision_feature_layers=vision_feature_layers,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **lm_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return VipLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = ["VipLlavaModel", "VipLlavaForConditionalGeneration", "VipLlavaPreTrainedModel"]
