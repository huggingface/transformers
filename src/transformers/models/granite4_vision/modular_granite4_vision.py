# Copyright 2025 IBM. All rights reserved.
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

from fractions import Fraction

import numpy as np
import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation.utils import GenerationMixin
from ...image_processing_utils import BatchFeature, select_best_resolution
from ...image_utils import ImageInput
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TransformersKwargs, can_return_tuple, logging
from ..llava_next.configuration_llava_next import LlavaNextConfig
from ..llava_next.image_processing_llava_next import LlavaNextImageProcessor, LlavaNextImageProcessorKwargs
from ..llava_next.image_processing_pil_llava_next import LlavaNextImageProcessorPil
from ..llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextModelOutputWithPast,
    LlavaNextPreTrainedModel,
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
    unpad_image,
)
from ..llava_next.processing_llava_next import LlavaNextProcessor
from .downsampling_granite4_vision import WindowQFormerDownsampler


logger = logging.get_logger(__name__)


# ── Image processing ──────────────────────────────────────────────────────


class Granite4VisionImageProcessorKwargs(LlavaNextImageProcessorKwargs):
    pass


class Granite4VisionImageProcessor(LlavaNextImageProcessor):
    valid_kwargs = Granite4VisionImageProcessorKwargs

    def preprocess(
        self, images: ImageInput | list[ImageInput], *args, **kwargs: Unpack[Granite4VisionImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, *args, **kwargs)


# Re-define Kwargs inheriting from ImagesKwargs for PIL file inlining (same pattern as llava_onevision)
class Granite4VisionImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    image_grid_pinpoints (`list[list[int]]`, *optional*):
        A list of possible resolutions to use for processing high resolution images. The best resolution is selected
        based on the original size of the image.
    """

    image_grid_pinpoints: list[list[int]]


class Granite4VisionImageProcessorPil(LlavaNextImageProcessorPil):
    valid_kwargs = Granite4VisionImageProcessorKwargs

    def preprocess(
        self, images: ImageInput | list[ImageInput], *args, **kwargs: Unpack[Granite4VisionImageProcessorKwargs]
    ) -> BatchFeature:
        return super().preprocess(images, *args, **kwargs)


# ── Output classes ──────────────────────────────────────────────────────────


class Granite4VisionModelOutputWithPast(LlavaNextModelOutputWithPast):
    pass


class Granite4VisionCausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    pass


# ── Config ──────────────────────────────────────────────────────────────────


class Granite4VisionConfig(LlavaNextConfig):
    r"""
    downsample_rate (`str`, *optional*):
        Fractional downsample rate for the Window Q-Former projector, e.g. `"1/4"` or `"3/8"`.
        The numerator is the query window side, the denominator is the key window side.
    use_image_newline_parameter (`bool`, *optional*, defaults to `True`):
        Whether to add a learnable newline embedding between image patch rows.
    deepstack_layer_map (`list`, *optional*):
        List of `[vision_layer_idx, llm_layer_idx]` pairs. Features from each vision encoder layer
        are projected and injected at the corresponding LLM decoder layer during forward pass.
    use_spatial_sampling (`bool`, *optional*, defaults to `False`):
        Whether to enable spatial offset sampling, which creates 4 groups (TL, TR, BL, BR) from
        a single vision layer, each injected at a different LLM layer.
    spatial_vision_layer (`int`, *optional*, defaults to `-1`):
        Index of the vision encoder layer used for spatial sampling.
    spatial_target_layers (`list`, *optional*, defaults to `[12, 15, 18, 21]`):
        Target LLM layers for the 4 spatial offset groups.
    projector_dropout (`float`, *optional*, defaults to `0.1`):
        Dropout probability in the Window Q-Former projector.
    image_grid_pinpoints (`list`, *optional*):
        A list of possible resolutions to use for processing high resolution images. Each item in the list should be a
        tuple or list of the form `(height, width)`.
    """

    model_type = "granite4_vision"

    downsample_rate: str | None = None
    use_image_newline_parameter: bool = True
    deepstack_layer_map: list | None = None
    use_spatial_sampling: bool = False
    spatial_vision_layer: int = -1
    spatial_target_layers: list | None = None
    projector_dropout: float = 0.1

    def __post_init__(self, **kwargs):
        if self.deepstack_layer_map is not None:
            self.deepstack_layer_map = [(int(v), int(l)) for v, l in self.deepstack_layer_map]

        if self.spatial_target_layers is None:
            self.spatial_target_layers = [12, 15, 18, 21]

        super().__post_init__(**kwargs)


# ── Processor ───────────────────────────────────────────────────────────────


class Granite4VisionProcessor(LlavaNextProcessor):
    model_type = "granite4_vision"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",
        num_additional_image_tokens=0,
        downsample_rate=None,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        num_additional_image_tokens (`int`, *optional*, defaults to `0`):
            Number of additional tokens added to the image embeddings, such as CLS (+1).
        downsample_rate (`str`, *optional*):
            Fractional downsample rate (e.g. `"1/4"`), used to adjust the number of image tokens
            when computing token counts for padding/truncation.
        """
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            vision_feature_select_strategy=vision_feature_select_strategy,
            chat_template=chat_template,
            image_token=image_token,
            num_additional_image_tokens=num_additional_image_tokens,
        )
        self.downsample_rate = downsample_rate

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        image_grid_pinpoints = self.image_processor.image_grid_pinpoints

        height_best_resolution, width_best_resolution = select_best_resolution(
            [orig_height, orig_width], image_grid_pinpoints
        )
        scale_height, scale_width = height_best_resolution // height, width_best_resolution // width

        patches_height = height // self.patch_size
        patches_width = width // self.patch_size
        if self.downsample_rate is not None:
            ds_rate = Fraction(self.downsample_rate)
            patches_height = int(patches_height * ds_rate)
            patches_width = int(patches_width * ds_rate)

        unpadded_features, newline_features = self._get_unpadded_features(
            orig_height, orig_width, patches_height, patches_width, scale_height, scale_width
        )
        base_features = patches_height * patches_width + self.num_additional_image_tokens
        num_image_tokens = unpadded_features + newline_features + base_features
        return num_image_tokens


# ── Model ───────────────────────────────────────────────────────────────────


class Granite4VisionPreTrainedModel(LlavaNextPreTrainedModel):
    pass


class Granite4VisionModel(LlavaNextModel):
    config_class = Granite4VisionConfig

    def __init__(self, config: Granite4VisionConfig):
        super().__init__(config)

        # Replace parent's single multi_modal_projector with deepstack projectors
        del self.multi_modal_projector

        self.spatial_projectors = None
        self.downsample_rate = config.downsample_rate
        self.projector_dropout = config.projector_dropout
        # Inherited from LlavaNextConfig (unused — kept for config compatibility)
        self.projector_hidden_act = config.projector_hidden_act
        self.multimodal_projector_bias = config.multimodal_projector_bias

        # Deepstack projectors: one per (vision_layer, llm_layer) pair
        self.layerwise_projectors = nn.ModuleList(
            [WindowQFormerDownsampler(config) for _ in range(len(config.deepstack_layer_map))]
        )

        # Spatial sampling projectors: 4 offset groups (TL, TR, BL, BR)
        if config.use_spatial_sampling:
            self.spatial_projectors = nn.ModuleList(
                [WindowQFormerDownsampler(config, spatial_offset=i) for i in range(4)]
            )

        # Override parent's image_newline based on config flag
        if not config.use_image_newline_parameter:
            self.image_newline = None

        self.pad_token_id = getattr(self.config, "pad_token_id", None) or -1

    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Overrides the parent to apply downsample_rate to height/width calculations.
        """
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
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
    ):
        """
        Extract image features via deepstack (multi-layer) and spatial sampling projections.

        Runs the vision tower once, then:
        1. Deepstack: for each (vision_layer, llm_layer) in deepstack_layer_map,
           extracts features from that vision layer, downsamples via interpolation + QFormer,
           and pairs them with the target LLM layer.
        2. Spatial: if enabled, extracts the spatial_vision_layer and creates 4 spatial
           offset groups (TL, TR, BL, BR), each targeting a different LLM layer.

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

            packed_features, _ = self.pack_image_features(
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

                packed_group, _ = self.pack_image_features(
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
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

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
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | Granite4VisionModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
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

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        if hasattr(self.language_model, "_update_mamba_mask") and any(
            lt in getattr(self.language_model.config, "layer_types", []) for lt in ("mamba", "hybrid")
        ):
            mamba_mask = self.language_model._update_mamba_mask(attention_mask, past_key_values)
        else:
            mamba_mask = None

        position_embeddings = None
        if self.language_model.rotary_emb is not None:
            position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = None

        # Layer-by-layer forward with vision injection
        for layer_idx, decoder_layer in enumerate(self.language_model.layers):
            # Inject vision features at this layer if configured
            for target_layer, features_for_layer in deepstack_features:
                if layer_idx == target_layer:
                    hidden_states = hidden_states.masked_scatter(
                        vision_mask, (hidden_states[vision_mask] + features_for_layer.flatten()).view(-1)
                    )

            layer_mask = mamba_mask if getattr(decoder_layer, "layer_type", None) == "mamba" else causal_mask

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        hidden_states = self.language_model.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        return Granite4VisionModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


# ── ForConditionalGeneration ────────────────────────────────────────────────


class Granite4VisionForConditionalGeneration(LlavaNextForConditionalGeneration):
    config_class = Granite4VisionConfig

    def __init__(self, config: Granite4VisionConfig):
        super().__init__(config)
        self.model = Granite4VisionModel(config)

    def merge_lora_adapters(self):
        """Merge LoRA adapter weights into base weights and replace PEFT wrappers with base layers."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            for attr_name, child in list(module.named_children()):
                if isinstance(child, BaseTunerLayer):
                    child.merge()
                    setattr(module, attr_name, child.get_base_layer())
        if hasattr(self, "peft_config"):
            del self.peft_config
        self._hf_peft_config_loaded = False
        return self

    def generate(self, *args, **kwargs) -> torch.LongTensor:
        # When loaded with a LoRA adapter, disable the adapter for text-only
        # inputs (no pixel_values) so the base LLM runs standalone.
        pixel_values = kwargs.get("pixel_values")
        if hasattr(self, "_hf_peft_config_loaded") and self._hf_peft_config_loaded:
            if pixel_values is not None:
                self.enable_adapters()
            else:
                self.disable_adapters()
        return GenerationMixin.generate(self, *args, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Granite4VisionCausalLMOutputWithPast:
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

        return Granite4VisionCausalLMOutputWithPast(
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
        logits_to_keep=None,
        **kwargs,
    ):
        is_first = kwargs.get("is_first_iteration", False)
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        model_inputs = self._init_hybrid_cache(**model_inputs)
        if is_first:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes

        return model_inputs

    def _init_hybrid_cache(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """Handle HybridMambaAttentionDynamicCache for GraniteMoeHybrid language model."""
        empty_past_kv = past_key_values is None or (
            isinstance(past_key_values, DynamicCache) and past_key_values.get_seq_length() == 0
        )

        if use_cache and empty_past_kv:
            past_key_values = DynamicCache(config=self.model.language_model.config)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv and input_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and (input_ids is None or empty_past_kv):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs


__all__ = [
    "Granite4VisionConfig",
    "Granite4VisionImageProcessor",
    "Granite4VisionImageProcessorPil",
    "Granite4VisionProcessor",
    "Granite4VisionPreTrainedModel",
    "Granite4VisionModel",
    "Granite4VisionForConditionalGeneration",
]
