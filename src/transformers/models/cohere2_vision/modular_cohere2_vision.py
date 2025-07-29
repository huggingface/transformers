# coding=utf-8
# Copyright 2025 the Cohere Inc. team. All rights reserved.
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
"""PyTorch AyaVision model."""

from typing import Optional, Union

import torch
from torch import nn

from transformers.models.aya_vision.modeling_aya_vision import (
    AyaVisionCausalLMOutputWithPast,
    AyaVisionForConditionalGeneration,
    AyaVisionModel,
    AyaVisionModelOutputWithPast,
)

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from .configuration_cohere2_vision import Cohere2VisionConfig


logger = logging.get_logger(__name__)


class Cohere2VisionMultiModalProjector(nn.Module):
    def __init__(self, config: Cohere2VisionConfig):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor
        self.intermediate_size = config.alignment_intermediate_size
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * (config.downsample_factor**2), self.intermediate_size, bias=True
        )
        self.act = ACT2FN["silu"]
        self.linear_2 = nn.Linear(self.intermediate_size // 2, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        image_features = self.pixel_shuffle(image_features)
        hidden_states = self.linear_1(image_features)

        # Split along last dimension and apply SwiGLU
        x, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * x

        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_shuffle(self, image_features):  # B, S, D
        batch_size, seq_length, feature_dim = image_features.shape
        height = width = int(seq_length**0.5)
        image_features = image_features.reshape(image_features.shape[0], width, height, -1)
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            batch_size, width, int(height / self.downsample_factor), int(channels * self.downsample_factor)
        )
        image_features = image_features.permute(0, 2, 1, 3)
        image_features = image_features.reshape(
            batch_size, int(height / self.downsample_factor), int(width / self.downsample_factor), -1
        )
        image_features = image_features.permute(0, 2, 1, 3)
        return image_features


class Cohere2VisionModelOutputWithPast(AyaVisionModelOutputWithPast):
    pass


class Cohere2VisionCausalLMOutputWithPast(AyaVisionCausalLMOutputWithPast):
    pass


class Cohere2VisionModel(AyaVisionModel):
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_num_patches: torch.Tensor,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
        Returns:
            image_features (List[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """

        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches.tolist(), dim=0)
        # pad image_features to the same length and stack them
        padded_image_features = []
        max_patch_len = max([img.shape[0] for img in image_features])
        for img in image_features:
            padded_image_features.append(
                torch.cat(
                    [
                        img,
                        torch.zeros(max_patch_len - img.shape[0], *img.shape[1:], device=img.device, dtype=img.dtype),
                    ],
                    dim=0,
                )
            )
        padded_image_features = torch.stack(padded_image_features, dim=0)
        return padded_image_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_num_patches: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, Cohere2VisionModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_num_patches)
            inputs_embeds = self._merge_image_text_embeddings(input_ids, image_features, inputs_embeds)

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
            **kwargs,
        )

        return Cohere2VisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def _merge_image_text_embeddings(self, input_ids, image_features, inputs_embeds):
        is_img_patch = input_ids == self.config.image_token_id
        image_encoding_to_sequence_map = (is_img_patch.cumsum(axis=1) * is_img_patch).to(torch.int64)  # [bsz, seqlen]

        if len(image_features.shape) == 4:
            image_features = image_features.unsqueeze(0)  # add batch dimension
        B, I, H, W, D = image_features.shape
        image_features = image_features.reshape(B, I * W * H, D)

        dev_img = image_features.device
        pad_tensor = torch.zeros((B, 1, D), dtype=image_features.dtype, device=dev_img)
        image_features = torch.cat((pad_tensor, image_features), dim=1)

        batch_indices = torch.arange(B, device=dev_img).unsqueeze(1).expand(B, image_encoding_to_sequence_map.shape[1])
        image_encoding_to_sequence_map = image_encoding_to_sequence_map.to(dev_img, non_blocking=True)
        gathered = image_features[batch_indices, image_encoding_to_sequence_map]  # [B, S, D]

        dev_out = inputs_embeds.device
        gathered = gathered.to(dev_out, dtype=inputs_embeds.dtype, non_blocking=True)
        is_img_patch = is_img_patch.to(dev_out, non_blocking=True)

        output = inputs_embeds * (~is_img_patch).unsqueeze(-1) + gathered * is_img_patch.unsqueeze(-1)
        return output


class Cohere2VisionForConditionalGeneration(AyaVisionForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_num_patches: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Cohere2VisionCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Cohere2VisionForConditionalGeneration
        >>> import torch

        >>> torch_device = "cuda:0"
        >>> processor = AutoProcessor.from_pretrained("CohereForAI/aya-vision-8b", use_fast=True)
        >>> model = Cohere2VisionForConditionalGeneration.from_pretrained("CohereForAI/aya-vision-8b", device_map=torch_device)

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://pbs.twimg.com/media/Fx7YvfQWYAIp6rZ?format=jpg&name=medium",
        ...             },
        ...             {"type": "text", "text": "चित्र में लिखा पाठ क्या कहता है?"},
        ...         ],
        ...     }
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", device=torch_device
        ... ).to(model.device)

        >>> gen_tokens = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.3)
        >>> processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_num_patches=image_num_patches,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            image_sizes=image_sizes,
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

        return Cohere2VisionCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "Cohere2VisionForConditionalGeneration",
    "Cohere2VisionPreTrainedModel",  # noqa: F822
    "Cohere2VisionModel",
]
