# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""LocateAnything: a MoonViT vision encoder + MLP projector + Qwen2.5 language model for visual grounding."""

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_cu_seqlens, get_vision_position_ids
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
)
from ..qwen2_vl.modeling_qwen2_vl import VisionRotaryEmbedding
from ..video_llama_3.modeling_video_llama_3 import (
    VideoLlama3VisionAttention,
    VideoLlama3VisionEncoder,
    VideoLlama3VisionEncoderLayer,
)


logger = logging.get_logger(__name__)


@auto_docstring
@strict
class LocateAnythingVisionConfig(PreTrainedConfig):
    r"""
    init_pos_emb_height (`int`, *optional*, defaults to 64):
        Height of the learnable position embedding grid that is bicubically interpolated to each image's grid.
    init_pos_emb_width (`int`, *optional*, defaults to 64):
        Width of the learnable position embedding grid.
    spatial_merge_size (`int`, *optional*, defaults to 2):
        Side length of the square patch-merge window applied before the multimodal projector.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        Base period of the 2D rotary position embedding.
    """

    model_type = "locateanything_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    spatial_merge_size: int = 2
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    initializer_range: float = 0.02


@auto_docstring
@strict
class LocateAnythingConfig(PreTrainedConfig):
    r"""
    image_token_id (`int`, *optional*, defaults to 151665):
        Token id used as the placeholder for image patch embeddings.
    image_start_token_id (`int`, *optional*, defaults to 151666):
        Token id that opens an image span (`<img>`).
    image_end_token_id (`int`, *optional*, defaults to 151667):
        Token id that closes an image span (`</img>`).
    box_start_token_id (`int`, *optional*, defaults to 151668):
        Token id that opens a bounding-box span (`<box>`).
    box_end_token_id (`int`, *optional*, defaults to 151669):
        Token id that closes a bounding-box span (`</box>`).
    coord_start_token_id (`int`, *optional*, defaults to 151677):
        First token id of the quantized coordinate vocabulary.
    coord_end_token_id (`int`, *optional*, defaults to 152677):
        Last token id of the quantized coordinate vocabulary.
    ref_start_token_id (`int`, *optional*, defaults to 151672):
        Token id that opens a referring-expression span (`<ref>`).
    ref_end_token_id (`int`, *optional*, defaults to 151673):
        Token id that closes a referring-expression span (`</ref>`).
    none_token_id (`int`, *optional*, defaults to 4064):
        Token id emitted for an empty box.
    """

    model_type = "locateanything"
    sub_configs = {"vision_config": LocateAnythingVisionConfig, "text_config": AutoConfig}

    image_token_id: int = 151665
    image_start_token_id: int = 151666
    image_end_token_id: int = 151667
    box_start_token_id: int = 151668
    box_end_token_id: int = 151669
    coord_start_token_id: int = 151677
    coord_end_token_id: int = 152677
    ref_start_token_id: int = 151672
    ref_end_token_id: int = 151673
    none_token_id: int = 4064
    tie_word_embeddings: bool = True

    vision_config: dict | LocateAnythingVisionConfig | None = None
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = CONFIG_MAPPING[self.text_config.get("model_type", "qwen2")](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()

        super().__post_init__(**kwargs)

        # `magi` is an optional MTP-time NVIDIA kernel this integration does not implement; the published checkpoint
        # ships it as the default, so unset it (and the sub-configs, via the setter) and let the standard selection apply.
        if self._attn_implementation == "magi":
            self._attn_implementation = None


class LocateAnythingVisionRotaryEmbedding(VisionRotaryEmbedding):
    pass


class LocateAnythingVisionAttention(VideoLlama3VisionAttention):
    pass


class LocateAnythingVisionEncoderLayer(VideoLlama3VisionEncoderLayer):
    pass


class LocateAnythingLearnable2DInterpPosEmb(nn.Module):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.height = config.init_pos_emb_height
        self.width = config.init_pos_emb_width
        self.weight = nn.Parameter(torch.empty(self.height, self.width, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        pos_embeds = []
        for _, height, width in image_grid_thw.tolist():
            if (height, width) == (self.height, self.width):
                pos_embeds.append(self.weight.flatten(end_dim=1))
            else:
                interpolated = F.interpolate(
                    self.weight.permute(2, 0, 1).unsqueeze(0), size=(height, width), mode="bicubic"
                )
                pos_embeds.append(interpolated.squeeze(0).permute(1, 2, 0).flatten(end_dim=1))
        return hidden_states + torch.cat(pos_embeds)


class LocateAnythingPatchEmbed(nn.Module):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.pos_emb = LocateAnythingLearnable2DInterpPosEmb(config)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(pixel_values).view(pixel_values.size(0), -1)
        hidden_states = self.pos_emb(hidden_states, image_grid_thw)
        return hidden_states


class LocateAnythingVisionEncoder(VideoLlama3VisionEncoder):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__(config)
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = LocateAnythingVisionRotaryEmbedding(head_dim // 2, theta=config.rope_theta)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, image_grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutput:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width of each image's patch grid.
        """
        position_ids = get_vision_position_ids(image_grid_thw, 1)
        cu_seqlens = get_vision_cu_seqlens(image_grid_thw)
        rotary = self.rotary_emb(position_ids).repeat(1, 2)
        position_embeddings = (rotary.cos(), rotary.sin())
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
            )
        return BaseModelOutput(last_hidden_state=self.final_layernorm(hidden_states))


class LocateAnythingMultiModalProjector(nn.Module):
    def __init__(self, config: LocateAnythingConfig):
        super().__init__()
        self.merge_size = config.vision_config.spatial_merge_size
        merged_dim = config.vision_config.hidden_size * self.merge_size * self.merge_size
        self.pre_norm = nn.LayerNorm(merged_dim)
        self.linear_1 = nn.Linear(merged_dim, config.text_config.hidden_size)
        self.act = ACT2FN["gelu"]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)

    def forward(self, image_features: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        merge = self.merge_size
        dim = image_features.shape[-1]
        chunks = image_features.split((image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist(), dim=0)
        outputs = []
        for chunk, (_, height, width) in zip(chunks, image_grid_thw.tolist()):
            new_height, new_width = height // merge, width // merge
            chunk = chunk.view(new_height, merge, new_width, merge, dim).permute(0, 2, 1, 3, 4)
            chunk = chunk.reshape(new_height * new_width, merge * merge * dim)
            chunk = self.linear_2(self.act(self.linear_1(self.pre_norm(chunk))))
            outputs.append(chunk)
        return torch.cat(outputs, dim=0)


@auto_docstring
class LocateAnythingPreTrainedModel(PreTrainedModel):
    config: LocateAnythingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LocateAnythingVisionEncoderLayer", "Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, LocateAnythingLearnable2DInterpPosEmb):
            init.normal_(module.weight)
        elif isinstance(module, LocateAnythingVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class LocateAnythingVisionModel(LocateAnythingPreTrainedModel):
    config: LocateAnythingVisionConfig
    main_input_name = "pixel_values"
    input_modalities = "image"
    _can_record_outputs = {
        "hidden_states": LocateAnythingVisionEncoderLayer,
        "attentions": LocateAnythingVisionAttention,
    }

    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__(config)
        self.patch_embed = LocateAnythingPatchEmbed(config)
        self.encoder = LocateAnythingVisionEncoder(config)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    def forward(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutput:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width of each image's patch grid.
        """
        hidden_states = self.patch_embed(pixel_values, image_grid_thw)
        return self.encoder(hidden_states, image_grid_thw, **kwargs)


class LocateAnythingModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class LocateAnythingCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class LocateAnythingModel(LlavaModel):
    def __init__(self, config: LocateAnythingConfig):
        super().__init__(config)
        self.vision_tower = LocateAnythingVisionModel._from_config(config.vision_config)
        self.multi_modal_projector = LocateAnythingMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @auto_docstring
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width of each image's patch grid.
        """
        vision_outputs: BaseModelOutput = self.vision_tower(
            pixel_values=pixel_values.to(self.vision_tower.dtype), image_grid_thw=image_grid_thw, **kwargs
        )
        image_features = self.multi_modal_projector(vision_outputs.last_hidden_state, image_grid_thw)
        return BaseModelOutputWithPooling(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=image_features,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LocateAnythingModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width of each image's patch grid.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs: BaseModelOutputWithPast = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return LocateAnythingModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class LocateAnythingForConditionalGeneration(LlavaForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LocateAnythingCausalLMOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width of each image's patch grid.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for the language-modeling loss; indices in `[0, ..., config.text_config.vocab_size]` or -100.
        """
        outputs: LocateAnythingModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return LocateAnythingCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "LocateAnythingVisionConfig",
    "LocateAnythingConfig",
    "LocateAnythingPreTrainedModel",
    "LocateAnythingVisionModel",
    "LocateAnythingModel",
    "LocateAnythingForConditionalGeneration",
]
