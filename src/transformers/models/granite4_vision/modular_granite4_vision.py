# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...image_processing_utils import select_best_resolution
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, can_return_tuple
from ...utils.output_capturing import capture_outputs
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..granite.configuration_granite import GraniteConfig
from ..granite.modeling_granite import GraniteAttention, GraniteDecoderLayer, GraniteModel, GraniteRotaryEmbedding
from ..llava_next.configuration_llava_next import LlavaNextConfig
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


# ── Output classes ──────────────────────────────────────────────────────────


@dataclass
class Granite4VisionModelOutputWithPast(LlavaNextModelOutputWithPast):
    """
    Args:
        deepstack_features (`list[tuple[int, list[torch.Tensor]]]`, *optional*):
            List of `(llm_layer_idx, packed_features)` pairs produced by the deepstack
            and spatial projectors. Each entry targets one LLM decoder layer; `packed_features`
            is a per-image list of tensors of shape `(num_image_tokens, hidden_size)`.
    """

    deepstack_features: list | None = None


@dataclass
class Granite4VisionCausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    """
    Args:
        deepstack_features (`list[tuple[int, list[torch.Tensor]]]`, *optional*):
            List of `(llm_layer_idx, packed_features)` pairs. See `Granite4VisionModelOutputWithPast`.
    """

    deepstack_features: list | None = None


@dataclass
class Granite4VisionImageFeaturesOutput(ModelOutput):
    """
    Output of `Granite4VisionModel.get_image_features`.

    Args:
        deepstack_features (`list[tuple[int, list[torch.Tensor]]]`):
            List of `(llm_layer_idx, packed_features)` pairs. Each entry targets one LLM
            decoder layer; `packed_features` is a per-image list of tensors of shape
            `(num_image_tokens, hidden_size)`.
    """

    deepstack_features: list | None = None


# ── Config ──────────────────────────────────────────────────────────────────


class Granite4VisionTextConfig(GraniteConfig):
    model_type = "granite4_vision_text"
    base_config_key = "text_config"


class Granite4VisionConfig(LlavaNextConfig):
    r"""
    downsample_rate (`str`, *optional*):
        Fractional downsample rate for the Window Q-Former projector, e.g. `"1/4"` or `"3/8"`.
        The numerator is the query window side, the denominator is the key window side.
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
    qformer_config (`dict` or `Blip2QFormerConfig`, *optional*):
        Configuration for the Window Q-Former projector. If `None`, defaults are derived from
        `vision_config.hidden_size`.
    image_grid_pinpoints (`list`, *optional*):
        A list of possible resolutions to use for processing high resolution images. Each item in the list should be a
        tuple or list of the form `(height, width)`.
    """

    model_type = "granite4_vision"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig, "qformer_config": AutoConfig}

    multimodal_projector_bias = AttributeError()
    projector_hidden_act = AttributeError()

    downsample_rate: str | None = None
    deepstack_layer_map: list | None = None
    use_spatial_sampling: bool = False
    spatial_vision_layer: int = -1
    spatial_target_layers: list | None = None
    projector_dropout: float = 0.1
    qformer_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.deepstack_layer_map is not None:
            self.deepstack_layer_map = [(int(v), int(l)) for v, l in self.deepstack_layer_map]

        if self.spatial_target_layers is None:
            self.spatial_target_layers = [12, 15, 18, 21]

        # Peek at vision hidden_size before super() to build a fully-specified qformer_config,
        # avoiding any runtime field patching after super().
        if isinstance(self.vision_config, dict):
            vision_hidden_size = self.vision_config.get("hidden_size", 1152)
        elif self.vision_config is not None:
            vision_hidden_size = self.vision_config.hidden_size
        else:
            vision_hidden_size = 1152

        # Convert qformer_config dict → object before super() so _attn_implementation.setter
        # (called inside super().__post_init__) sees a config object, not a raw dict.
        if isinstance(self.qformer_config, dict):
            model_type = self.qformer_config.get("model_type", "blip_2_qformer")
            self.qformer_config = CONFIG_MAPPING[model_type](**self.qformer_config)
        elif self.qformer_config is None:
            self.qformer_config = CONFIG_MAPPING["blip_2_qformer"](
                num_hidden_layers=1,
                intermediate_size=3072,
                cross_attention_frequency=1,
                max_position_embeddings=2048,
                use_qformer_text_input=False,
                hidden_size=vision_hidden_size,
                num_attention_heads=vision_hidden_size // 64,
                encoder_hidden_size=vision_hidden_size,
            )

        super().__post_init__(**kwargs)


# ── Processor ───────────────────────────────────────────────────────────────


class Granite4VisionProcessor(LlavaNextProcessor):
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


# ── Downsampling helpers ─────────────────────────────────────────────────────


def interpolate_downsample(image_features: torch.Tensor, orig_side: int, new_side: int) -> torch.Tensor:
    """Spatial downsampling via area interpolation."""
    batch, _, channels = image_features.size()
    spatial = image_features.view(batch, orig_side, orig_side, channels).permute(0, 3, 1, 2)
    spatial = torch.nn.functional.interpolate(spatial, size=(new_side, new_side), mode="area")
    return spatial.permute(0, 2, 3, 1).flatten(1, 2)


def spatial_offset_downsample(image_features: torch.Tensor, orig_side: int, offset: int = 0) -> torch.Tensor:
    """Sample one position from each 2x2 block; offset selects which corner (0=TL,1=TR,2=BL,3=BR)."""
    offset_h, offset_w = [(0, 0), (0, 1), (1, 0), (1, 1)][offset]
    new_side = orig_side // 2
    batch, _, channels = image_features.shape
    grid = image_features.reshape(batch, orig_side, orig_side, channels)
    grid = grid.reshape(batch, new_side, 2, new_side, 2, channels)
    return grid[:, :, offset_h, :, offset_w, :].reshape(batch, -1, channels)


class Granite4VisionWindowQFormerDownsampler(nn.Module):
    """Window-based QFormer downsampler that processes image patches in windows."""

    def __init__(self, config, spatial_offset=None):
        super().__init__()
        llm_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size

        self.dropout = nn.Dropout(config.projector_dropout)
        self._spatial_offset = spatial_offset
        self._downsample_rate = config.downsample_rate

        self.qformer = AutoModel.from_config(config.qformer_config)

        self.image_side = config.vision_config.image_size // config.vision_config.patch_size
        query_side_str, window_side_str = config.downsample_rate.split("/")
        self.query_side, self.window_side = int(query_side_str), int(window_side_str)
        self.query_length = self.query_side**2
        self.norm = nn.LayerNorm(vision_hidden_size, eps=1e-6)
        self.query = nn.Parameter(torch.empty(1, self.query_length, vision_hidden_size))
        self.image_positions = nn.Parameter(torch.empty(1, self.window_side**2, vision_hidden_size))
        self.out_linear = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)

    def _windowed_raster(self, features, side, window_size):
        """(B, side*side, C) raster -> (B*num_win*num_win, window_size*window_size, C)"""
        batch, _, channels = features.shape
        num_win = side // window_size
        return (
            features.view(batch, side, side, channels)
            .view(batch, num_win, window_size, num_win, window_size, channels)
            .transpose(2, 3)
            .flatten(0, 2)
            .flatten(1, 2)
        )

    def _unwindowed_raster(self, windowed_features, num_win, window_size):
        """(B*num_win*num_win, window_size*window_size, C) -> (B, (num_win*window_size)^2, C)"""
        batch_win, _, channels = windowed_features.shape
        if batch_win % (num_win * num_win) != 0:
            raise ValueError(
                f"Expected batch_win ({batch_win}) to be divisible by num_win^2 ({num_win**2})."
            )
        batch = batch_win // (num_win * num_win)
        side = num_win * window_size
        return (
            windowed_features.view(batch, num_win, num_win, window_size, window_size, channels)
            .transpose(2, 3)
            .contiguous()
            .view(batch, side, side, channels)
            .flatten(1, 2)
        )

    def forward(self, image_features):
        batch, hw, channels = image_features.shape
        if self.image_side * self.image_side != hw:
            raise ValueError(
                f"Expected image_features with {self.image_side**2} spatial tokens, got {hw}. "
                "Check that the vision encoder image_size and patch_size match the config."
            )
        num_windows = self.image_side // self.window_side
        interp_side = int(self.image_side * Fraction(self._downsample_rate))
        image_features = self.norm(image_features)
        enc = self._windowed_raster(image_features, self.image_side, self.window_side)

        if self._spatial_offset is not None:
            downsampled = spatial_offset_downsample(image_features, self.image_side, self._spatial_offset)
        else:
            downsampled = interpolate_downsample(image_features, self.image_side, interp_side)

        downsampled_side = num_windows * self.query_side
        downsampled_w = self._windowed_raster(downsampled, downsampled_side, self.query_side)

        query_embeds = self.query + downsampled_w
        encoder_embeds = self.dropout(enc + self.image_positions)
        out_w = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_embeds,
            return_dict=True,
        ).last_hidden_state

        out = self._unwindowed_raster(out_w, num_win=num_windows, window_size=self.query_side)
        out = self.dropout(out)
        return self.out_linear(out)


# ── Model ───────────────────────────────────────────────────────────────────


class Granite4VisionTextRotaryEmbedding(GraniteRotaryEmbedding):
    pass


class Granite4VisionPreTrainedModel(LlavaNextPreTrainedModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Granite4VisionTextRotaryEmbedding):
            # Non-persistent buffers (inv_freq, original_inv_freq) are replaced with
            # torch.empty_like() garbage by _move_missing_keys_from_meta_to_device.
            # Recompute them here so _initialize_missing_keys restores correct values.
            rope_type = module.config.rope_parameters.get("rope_type", "default")
            if rope_type != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = module.compute_default_rope_parameters
            inv_freq, attention_scaling = rope_init_fn(module.config, module.inv_freq.device)
            init.copy_(module.inv_freq, inv_freq)
            init.copy_(module.original_inv_freq, inv_freq)
            module.attention_scaling = attention_scaling
        if isinstance(module, Granite4VisionWindowQFormerDownsampler):
            embed_std = 1 / math.sqrt(module.query.shape[-1])
            init.normal_(module.query, mean=0.0, std=embed_std)
            init.normal_(module.image_positions, mean=0.0, std=embed_std)


class Granite4VisionTextAttention(GraniteAttention):
    pass


class Granite4VisionTextDecoderLayer(GraniteDecoderLayer):
    pass


class Granite4VisionTextModel(Granite4VisionPreTrainedModel, GraniteModel):
    """Granite LLM backbone with deepstack feature injection support."""

    base_model_prefix = "model"
    _no_split_modules = ["Granite4VisionTextDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": Granite4VisionTextDecoderLayer,
        "attentions": Granite4VisionTextAttention,
    }

    def __init__(self, config: Granite4VisionTextConfig):
        super().__init__(config)

    def _deepstack_inject(
        self,
        hidden_states: torch.Tensor,
        vision_mask: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Add projected vision features into the image-token positions of hidden_states."""
        vision_mask = vision_mask.to(hidden_states.device)
        features = features.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[vision_mask] = hidden_states[vision_mask] + features
        return hidden_states

    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        vision_mask: torch.BoolTensor | None = None,
        deepstack_features: dict | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        vision_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Boolean mask marking image token positions. Required when `deepstack_features` is provided.
        deepstack_features (`dict[int, torch.Tensor]`, *optional*):
            Mapping from LLM layer index to projected vision features of shape `(num_image_tokens, hidden_size)`.
            Features are added into image-token positions of hidden states before the corresponding decoder layer.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            ).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if deepstack_features is not None and layer_idx in deepstack_features:
                hidden_states = self._deepstack_inject(hidden_states, vision_mask, deepstack_features[layer_idx])

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Granite4VisionModel(LlavaNextModel):
    config_class = Granite4VisionConfig

    def __init__(self, config: Granite4VisionConfig):
        super().__init__(config)

        # Replace parent's single multi_modal_projector with layerwise_projectors
        del self.multi_modal_projector

        self.spatial_projectors = None
        self.downsample_rate = config.downsample_rate
        self.projector_dropout = config.projector_dropout

        # Deepstack projectors: one per (vision_layer, llm_layer) pair
        self.layerwise_projectors = nn.ModuleList(
            [Granite4VisionWindowQFormerDownsampler(config) for _ in range(len(config.deepstack_layer_map))]
        )

        # Spatial sampling projectors: 4 offset groups (TL, TR, BL, BR)
        if config.use_spatial_sampling:
            self.spatial_projectors = nn.ModuleList(
                [Granite4VisionWindowQFormerDownsampler(config, spatial_offset=i) for i in range(4)]
            )

        self.pad_token_id = (
            self.config.text_config.pad_token_id if self.config.text_config.pad_token_id is not None else -1
        )

        # Replace the inherited LLM backbone with our deepstack-aware subclass
        self.language_model = Granite4VisionTextModel(config.text_config)

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
                    raise ValueError(
                        "Image feature shape does not line up with the provided patch size. "
                        "You may be using the `default` vision_feature_select_strategy with a "
                        "visual encoder that does not have CLS token."
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
        **kwargs,
    ) -> Granite4VisionImageFeaturesOutput:
        """
        Extract image features via deepstack (multi-layer) and spatial sampling projections.

        Runs the vision tower once, then:
        1. Deepstack: for each (vision_layer, llm_layer) in deepstack_layer_map,
           extracts features from that vision layer, downsamples via interpolation + QFormer,
           and pairs them with the target LLM layer.
        2. Spatial: if enabled, extracts the spatial_vision_layer and creates 4 spatial
           offset groups (TL, TR, BL, BR), each targeting a different LLM layer.
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

        vision_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

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

        return Granite4VisionImageFeaturesOutput(deepstack_features=all_features)

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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Granite4VisionModelOutputWithPast:
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

        # Build deepstack injection map and scatter initial image embeddings
        deepstack_features = None
        vision_mask = None
        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            deepstack_features = {}
            for idx, (llm_layer_idx, packed_features) in enumerate(image_features.deepstack_features):
                concat_features = torch.cat(packed_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                if idx == 0:
                    vision_mask_3d = self.get_placeholder_mask(
                        input_ids, inputs_embeds=inputs_embeds, image_features=concat_features
                    )
                    vision_mask = vision_mask_3d[..., 0]
                    inputs_embeds = inputs_embeds.masked_fill(vision_mask_3d, 0.0)
                deepstack_features[llm_layer_idx] = concat_features

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            vision_mask=vision_mask,
            deepstack_features=deepstack_features,
            **kwargs,
        )

        return Granite4VisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            deepstack_features=image_features.deepstack_features if pixel_values is not None else None,
        )


# ── ForConditionalGeneration ────────────────────────────────────────────────


class Granite4VisionForConditionalGeneration(LlavaNextForConditionalGeneration):
    config_class = Granite4VisionConfig

    def __init__(self, config: Granite4VisionConfig):
        super().__init__(config)
        self.model = Granite4VisionModel(config)

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
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Granite4VisionCausalLMOutputWithPast:
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
            deepstack_features=outputs.deepstack_features,
        )



__all__ = [
    "Granite4VisionConfig",
    "Granite4VisionTextConfig",
    "Granite4VisionProcessor",
    "Granite4VisionPreTrainedModel",
    "Granite4VisionTextModel",
    "Granite4VisionModel",
    "Granite4VisionForConditionalGeneration",
]
