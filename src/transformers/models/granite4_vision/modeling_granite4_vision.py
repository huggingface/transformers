# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Granite 4 Vision model."""

import math
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import ModelOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ...utils.import_utils import can_return_tuple
from ..auto import AutoModel
from ..granitemoehybrid.modeling_granitemoehybrid import HybridMambaAttentionDynamicCache
from ..llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModelOutputWithPast,
    LlavaNextPreTrainedModel,
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
    unpad_image,
)
from .configuration_granite4_vision import Granite4VisionConfig
from .downsampling import WindowQFormerDownsampler


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Granite4VisionConfig"


class Granite4VisionForConditionalGeneration(LlavaNextForConditionalGeneration):
    config_class = Granite4VisionConfig

    def __init__(self, config: Granite4VisionConfig):
        LlavaNextPreTrainedModel.__init__(self, config)

        self.model = Granite4VisionModel(config)

        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        self.post_init()

    def generate(self, *args, **kwargs) -> torch.LongTensor:
        # When loaded with a LoRA adapter, disable the adapter for text-only
        # inputs (no pixel_values) so the base LLM runs standalone.
        pixel_values = kwargs.get("pixel_values", None)
        if hasattr(self, "_hf_peft_config_loaded") and self._hf_peft_config_loaded:
            if pixel_values is not None:
                self.enable_adapters()
            else:
                self.disable_adapters()
        return super().generate(*args, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaNextCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        outputs = self.model(
            input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
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

        hidden_states = outputs.last_hidden_state

        loss = None
        logits = self.lm_head(hidden_states)
        logits = logits / self.config.text_config.logits_scaling
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:, :]

        return LlavaNextCausalLMOutputWithPast(
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
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        model_inputs = self._init_hybrid_cache(**model_inputs)
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes

        return model_inputs

    def _init_hybrid_cache(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """Handle HybridMambaAttentionDynamicCache for GraniteMoeHybrid language model."""
        empty_past_kv = past_key_values is None or (
            isinstance(past_key_values, DynamicCache) and past_key_values[0][0] is None
        )

        if not empty_past_kv:
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
        elif use_cache:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.model.language_model.config, input_ids.shape[0], self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )

        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs


class Granite4VisionModel(LlavaNextPreTrainedModel):
    config_class = Granite4VisionConfig

    def __init__(self, config: Granite4VisionConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.spatial_projectors = None

        if config.deepstack_layer_map is None:
            raise ValueError("deepstack_layer_map must be specified in config")
        if config.downsample_rate is None:
            raise ValueError("downsample_rate must be specified in config")

        self.downsample_rate = config.downsample_rate

        # Deepstack projectors: one per (vision_layer, llm_layer) pair
        self.layerwise_projectors = nn.ModuleList(
            [WindowQFormerDownsampler(config) for _ in range(len(config.deepstack_layer_map))]
        )

        # Spatial sampling projectors: 4 offset groups (TL, TR, BL, BR)
        if config.use_spatial_sampling:
            self.spatial_projectors = nn.ModuleList(
                [WindowQFormerDownsampler(config, spatial_offset=i) for i in range(4)]
            )

        self.image_newline = None
        if config.use_image_newline_parameter:
            embed_std = 1 / math.sqrt(config.text_config.hidden_size)
            self.image_newline = nn.Parameter(
                torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std
            )

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModel.from_config(config.text_config)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def pack_and_unpad_image_features(
        self, image_features, image_sizes, vision_feature_select_strategy, image_newline=None
    ):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`list[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_select_strategy (`str`)
                The feature selection strategy used to select the vision feature from the vision backbone.
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`list[int]`)
                token length of each image in image_features
        """
        from fractions import Fraction

        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                if self.layerwise_projectors is not None:
                    ds_rate = Fraction(self.downsample_rate)
                    height = int(height * ds_rate)
                    width = int(width * ds_rate)

                if (
                    np.prod(image_feature.shape) % (num_patch_height * num_patch_width * height * width) != 0
                    and vision_feature_select_strategy == "default"
                ):
                    logger.warning_once(
                        "Image feature shape does not line up with the provided patch size. "
                        "You may be using the `default` vision_feature_select_strategy with a"
                        " visual encoder that does not have CLS."
                    )

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features[0].device)
        return new_image_features, feature_lens

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ):
        """
        Extract image features via deepstack (multi-layer) and spatial sampling projections.

        Runs the vision tower once, then:
        1. Deepstack: for each (vision_layer, llm_layer) in deepstack_layer_map,
           extracts features from that vision layer, downsamples via interpolation + QFormer,
           and pairs them with the target LLM layer.
        2. Spatial: if enabled, extracts the spatial_vision_layer and creates 4 spatial
           offset groups (TL, TR, BL, BR), each targeting a different LLM layer.

        Args:
            pixel_values: Image tensors of shape (batch, num_patches, C, H, W) or (N, C, H, W).
            image_sizes: Actual image sizes (num_images, 2).
            vision_feature_layer: Unused (kept for API compatibility).
            vision_feature_select_strategy: "default" (remove CLS) or "full".
        Returns:
            List of (llm_layer_idx, packed_features) tuples for injection during forward pass.
        """
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        if pixel_values.dim() == 5:
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        # Deepstack features: extract from multiple vision layers, downsample via interpolation
        all_features = []
        for projection_idx, (vision_layer, llm_layer) in enumerate(self.config.deepstack_layer_map):
            selected_feature = vision_outputs.hidden_states[vision_layer]

            if vision_feature_select_strategy == "default":
                selected_feature = selected_feature[:, 1:]

            projected_features = self.layerwise_projectors[projection_idx](selected_feature)
            projected_features = torch.split(projected_features, image_num_patches, dim=0)

            packed_features, _ = self.pack_and_unpad_image_features(
                projected_features,
                image_sizes,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=self.image_newline,
            )

            all_features.append((llm_layer, packed_features))

        # Spatial features: extract 4 offset groups from a single vision layer
        if self.config.use_spatial_sampling:
            spatial_feature = vision_outputs.hidden_states[self.config.spatial_vision_layer]

            if vision_feature_select_strategy == "default":
                spatial_feature = spatial_feature[:, 1:]

            for group_idx, llm_layer in enumerate(self.config.spatial_target_layers):
                projected_group = self.spatial_projectors[group_idx](spatial_feature)
                projected_group_split = torch.split(projected_group, image_num_patches, dim=0)

                packed_group, _ = self.pack_and_unpad_image_features(
                    projected_group_split,
                    image_sizes,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_newline=self.image_newline,
                )

                all_features.append((llm_layer, packed_group))

        return all_features

    def get_image_token_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Build a boolean mask over inputs_embeds marking positions of <image> tokens,
        and verify that the count matches the number of image feature vectors.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_index

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )
        return special_image_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, LlavaNextModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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

        # Extract deepstack + spatial features and prepare for layer-by-layer injection
        deepstack_features = []
        vision_mask = None
        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            for idx, (llm_layer_idx, packed_features) in enumerate(image_features):
                concat_features = torch.cat(packed_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                if idx == 0:
                    vision_mask = self.get_image_token_mask(
                        input_ids, inputs_embeds=inputs_embeds, image_features=concat_features
                    )
                    inputs_embeds = inputs_embeds.masked_fill(vision_mask, 0.0)
                deepstack_features.append((llm_layer_idx, concat_features))

        # Custom forward pass with vision injection at specific LLM layers
        hidden_states = inputs_embeds * self.language_model.embedding_multiplier

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.language_model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        mamba_mask = self.language_model._update_mamba_mask(attention_mask, cache_position)

        position_embeddings = None
        if self.language_model.rotary_emb is not None:
            position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # Layer-by-layer forward with vision injection
        for layer_idx, decoder_layer in enumerate(self.language_model.layers):
            # Inject vision features at this layer if configured
            for target_layer, features_for_layer in deepstack_features:
                if layer_idx == target_layer:
                    hidden_states = hidden_states.masked_scatter(
                        vision_mask, (hidden_states[vision_mask] + features_for_layer.flatten()).view(-1)
                    )

            layer_mask = mamba_mask if decoder_layer.layer_type == "mamba" else causal_mask

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions and layer_outputs[1] is not None:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.language_model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        return LlavaNextModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


__all__ = ["Granite4VisionForConditionalGeneration", "Granite4VisionModel", "Granite4VisionConfig"]

# Made with Bob
