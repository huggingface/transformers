# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
# Written by Orr Zohar
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
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import DynamicCache
from ...utils import (
    logging,
)
from ..idefics3.configuration_idefics3 import Idefics3Config, Idefics3VisionConfig
from ..idefics3.image_processing_idefics3 import Idefics3ImageProcessor
from ..idefics3.modeling_idefics3 import (
    Idefics3BaseModelOutputWithPast,
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
    Idefics3VisionTransformer,
)


logger = logging.get_logger(__name__)


class SmolVLMVisionConfig(Idefics3VisionConfig):
    model_type = "smolvlm_vision"
    pass


class SmolVLMPreTrainedModel(Idefics3PreTrainedModel):
    pass
    

class SmolVLMVisionTransformer(Idefics3VisionTransformer):
    pass


class SmolVLMConfig(Idefics3Config):
    model_type = "smolvlm"
    pass


class SmolVLMImageProcessor(Idefics3ImageProcessor):
    pass


class SmolVLMBaseModelOutputWithPast(Idefics3BaseModelOutputWithPast):
    pass
    




class SmolVLMModel(Idefics3Model):
    """
    A subclass of Idefics3Model. We do *not* remove or block the call to inputs_merger
    in forward. Instead, we override inputs_merger here with custom logic.
    """

    def inputs_merger(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.Tensor, image_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge text embeddings with image embeddings out-of-place (no in-place indexing).
        The shapes are something like:
          - input_ids:          (B, T)
          - inputs_embeds:      (B, T, D)
          - image_hidden_states:(N, S, D) where N is total images across the batch,
            S is #patches (or #slots) per image, D is embedding dim.
        Returns:
          A tensor of (B, T, D).
        """

        B, T, D_text = inputs_embeds.shape
        N, S, D_img = image_hidden_states.shape
        
        image_offset = 0
        merged_outputs: List[torch.Tensor] = []

        # Iterate through each sample
        for b_idx, (cur_ids, cur_embeds) in enumerate(zip(input_ids, inputs_embeds)):
            # Find positions of <image> tokens in the text
            image_positions = (cur_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_positions)
            
            # If no <image> => text-only
            if num_image_tokens == 0:
                # NOTE: this is important for DeepSpeed.
                empty_slice = image_hidden_states[0][:0, :]  # shape (0, D)
                merged_text_only = torch.cat([cur_embeds, empty_slice], dim=0)
                merged_outputs.append(merged_text_only)
                continue
                
            # Typically, if each image is S embeddings, we expect the total # of <image> tokens
            # in this sample to be multiple of S => each group of S tokens = 1 image
            if num_image_tokens % S != 0:
                raise ValueError(
                    f"Sample {b_idx} has {num_image_tokens} <image> tokens, not a multiple of S={S}. "
                    "Cannot map them to blocks of shape (S, D)."
                )
                
            positions_list = image_positions.tolist()
            chunks = [positions_list[i : i + S] for i in range(0, num_image_tokens, S)]
            
            segments = []
            text_start = 0

            # For each chunk (each chunk => 1 image)
            for chunk in chunks:
                cur_block = image_hidden_states[image_offset]
                image_offset += 1
                
                # We'll iterate over the S positions in ascending order
                for i_s, pos in enumerate(chunk):
                    if pos > text_start:
                        segments.append(cur_embeds[text_start:pos])
                    # Then add one row from cur_block => shape (1, D)
                    row_of_block = cur_block[i_s : i_s + 1, :]
                    segments.append(row_of_block)
                    text_start = pos + 1

            # leftover text after the final <image> token
            if text_start < T:
                segments.append(cur_embeds[text_start:])

            # cat them into a single (T_b, D) tensor
            merged_sample = torch.cat(segments, dim=0)
            merged_outputs.append(merged_sample)

        merged_outputs = torch.stack(merged_outputs)
        return merged_outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SmolVLMBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image

            if not any(real_images_inds):
                # no images, leave one empty image.
                real_images_inds[0] = True

            pixel_values = pixel_values[real_images_inds].contiguous()

            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state

            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return SmolVLMBaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class SmolVLMForConditionalGeneration(Idefics3ForConditionalGeneration):
    """
    A subclass of Idefics3ForConditionalGeneration that uses SmolVLMModel
    instead of the default Idefics3Model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = SmolVLMModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "SmolVLMVisionConfig",
    "SmolVLMConfig",
    "SmolVLMImageProcessor",
    "SmolVLMForConditionalGeneration",
    "SmolVLMPreTrainedModel",
    "SmolVLMModel",
    "SmolVLMVisionTransformer",
]
