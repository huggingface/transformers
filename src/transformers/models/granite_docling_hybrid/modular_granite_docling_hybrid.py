# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch GraniteDoclingHybrid model."""

import torch

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache
from ..idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3CausalLMOutputWithPast,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
)
from .configuration_granite_docling_hybrid import GraniteDoclingHybridConfig


class GraniteDoclingHybridBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast):
    pass


class GraniteDoclingHybridCausalLMOutputWithPast(Idefics3CausalLMOutputWithPast):
    pass


class HybridMambaAttentionDynamicCache(HybridMambaAttentionDynamicCache):
    pass


class GraniteDoclingHybridPreTrainedModel(Idefics3PreTrainedModel):
    config_class = GraniteDoclingHybridConfig


class GraniteDoclingHybridModel(Idefics3Model):
    config_class = GraniteDoclingHybridConfig

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | GraniteDoclingHybridBaseModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
            the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
            max_num_images is the maximum number of images among the batch_size samples in the batch.
            Padding images are not needed beyond padding the pixel_values at the entrance of the model.
            For efficiency, we only pass through the vision_model's forward the real images by
            discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
            image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, _ = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, _, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_hidden_states = self.get_image_features(
                pixel_values, pixel_attention_mask, return_dict=True
            ).pooler_output
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        # Initialize HybridMambaAttentionDynamicCache for GraniteMoeHybrid text model
        if use_cache and not isinstance(past_key_values, HybridMambaAttentionDynamicCache):
            past_key_values = HybridMambaAttentionDynamicCache(
                self.text_model.config,
                batch_size=batch_size,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        return GraniteDoclingHybridBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The GraniteDoclingHybrid Model with a language modeling head. It is made up of a SigLIP vision encoder,
    with a GraniteMoeHybrid language model on top.
    """
)
class GraniteDoclingHybridForConditionalGeneration(Idefics3ForConditionalGeneration):
    config_class = GraniteDoclingHybridConfig

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | GraniteDoclingHybridCausalLMOutputWithPast:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `GraniteDoclingHybridForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return GraniteDoclingHybridCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten to handle HybridMambaAttentionDynamicCache initialization

        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # Initialize HybridMambaAttentionDynamicCache if needed
        if model_inputs.get("use_cache", True) and not isinstance(
            model_inputs.get("past_key_values"), HybridMambaAttentionDynamicCache
        ):
            cache_source = model_inputs.get("inputs_embeds")
            if cache_source is None:
                cache_source = model_inputs.get("decoder_inputs_embeds")
            if cache_source is not None:
                batch_size = cache_source.shape[0]
                dtype = cache_source.dtype
                device = cache_source.device
            else:
                input_tensor = model_inputs.get("input_ids")
                if input_tensor is None:
                    input_tensor = model_inputs.get("decoder_input_ids")
                if input_tensor is None:
                    input_tensor = input_ids
                if input_tensor is None:
                    raise ValueError("Unable to determine batch size for GraniteMoeHybrid cache initialization.")
                batch_size = input_tensor.shape[0]
                dtype = self.model.text_model.get_input_embeddings().weight.dtype
                device = input_tensor.device

            model_inputs["past_key_values"] = HybridMambaAttentionDynamicCache(
                self.model.text_model.config,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
            )

        cache_position = model_inputs.get("cache_position", cache_position)
        if cache_position is None:
            cache_position = torch.zeros(1, dtype=torch.long, device=self.device)

        if image_hidden_states is not None or cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


__all__ = [
    "GraniteDoclingHybridForConditionalGeneration",
    "GraniteDoclingHybridModel",
    "GraniteDoclingHybridPreTrainedModel",
]
