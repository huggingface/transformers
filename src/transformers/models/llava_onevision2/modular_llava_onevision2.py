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

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_position_ids
from ..auto import AutoModel
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)
from ..qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionAttention
from ..qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from ..siglip.modeling_siglip import SiglipMLP
from .configuration_llava_onevision2 import LlavaOnevision2Config, LlavaOnevision2VisionConfig


class LlavaOnevision2ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class LlavaOnevision2CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class LlavaOnevision2ImageProcessor(Qwen2VLImageProcessor):
    temporal_patch_size = 1


class LlavaOnevision2ImageProcessorPil(Qwen2VLImageProcessorPil):
    temporal_patch_size = 1


def rotate_half(x):
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


class LlavaOnevision2VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.config = config
        unit = config.hidden_size // config.num_attention_heads // 32
        for axis, width in zip("thw", (4 * unit, 6 * unit, 6 * unit)):
            frequency = 1.0 / (config.rope_theta ** (torch.arange(width, dtype=torch.float32) / width))
            self.register_buffer(f"inv_freq_{axis}", frequency, persistent=False)

    def forward(self, positions):
        frequencies = torch.cat(
            [
                positions[:, index, None].float() * getattr(self, f"inv_freq_{axis}").float()
                for index, axis in enumerate("thw")
            ],
            dim=-1,
        )
        # The checkpoint repeats the entire frequency table before interleaved rotation.
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos(), frequencies.sin()


class LlavaOnevision2VisionEmbeddings(nn.Module):
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels * config.temporal_patch_size
        self.hidden_size = config.hidden_size
        self.patch_embedding = nn.Conv2d(
            self.num_channels, config.hidden_size, config.patch_size, stride=config.patch_size, bias=False
        )

    def forward(self, pixel_values):
        patches = pixel_values.reshape(-1, self.num_channels, self.patch_size, self.patch_size)
        return self.patch_embedding(patches.to(self.patch_embedding.weight.dtype)).reshape(-1, self.hidden_size)


class LlavaOnevision2VisionMLP(SiglipMLP):
    pass


class LlavaOnevision2VisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__(config)
        self.attention_dropout = config.attention_dropout


class LlavaOnevision2VisionBlock(GradientCheckpointingLayer):
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        norm = nn.LayerNorm if config.layer_norm_type == "layer_norm" else nn.RMSNorm
        self.layer_norm1 = norm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = norm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = LlavaOnevision2VisionAttention(config)
        self.mlp = LlavaOnevision2VisionMLP(config)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states), **kwargs)
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class LlavaOnevision2VisionEncoder(nn.Module):
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([LlavaOnevision2VisionBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states


class LlavaOnevision2VisionPatchMerger(nn.Module):
    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)
        if config.use_patch_position_encoding:
            if config.patch_position_encoding_type != "absolute":
                raise ValueError("Only absolute merger position embeddings are supported")
            self.pos_emb_h = nn.Embedding(config.max_position_embeddings, config.out_hidden_size)
            self.pos_emb_w = nn.Embedding(config.max_position_embeddings, config.out_hidden_size)

    def forward(self, hidden_states, patch_positions):
        hidden_states = self.fc2(self.act(self.fc1(self.ln_q(hidden_states).reshape(-1, self.hidden_size))))
        if self.config.use_patch_position_encoding:
            positions = patch_positions[:: self.config.spatial_merge_size**2] // self.config.spatial_merge_size
            hidden_states = hidden_states + self.pos_emb_h(positions[:, 1]) + self.pos_emb_w(positions[:, 2])
        return hidden_states


class LlavaOnevision2PreTrainedModel(LlavaPreTrainedModel):
    config_class = LlavaOnevision2Config
    _no_split_modules = ["LlavaOnevision2VisionBlock"]
    _can_compile_fullgraph = False
    _supports_flex_attn = False

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, LlavaOnevision2VisionRotaryEmbedding):
            for axis in "thw":
                buffer = getattr(module, f"inv_freq_{axis}")
                values = 1.0 / (
                    module.config.rope_theta ** (torch.arange(buffer.numel(), dtype=torch.float32) / buffer.numel())
                )
                init.copy_(buffer, values)


@auto_docstring
class LlavaOnevision2VisionModel(LlavaOnevision2PreTrainedModel):
    config_class = LlavaOnevision2VisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _can_record_outputs = {"hidden_states": LlavaOnevision2VisionBlock}

    def __init__(self, config: LlavaOnevision2VisionConfig):
        super().__init__(config)
        self.embeddings = LlavaOnevision2VisionEmbeddings(config)
        norm = nn.LayerNorm if config.layer_norm_type == "layer_norm" else nn.RMSNorm
        self.layernorm_pre = norm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = LlavaOnevision2VisionEncoder(config)
        self.video_rope = LlavaOnevision2VisionRotaryEmbedding(config)
        self.merger = LlavaOnevision2VisionPatchMerger(config)
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        patch_positions: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width dimensions of each patch grid.
        patch_positions (`torch.LongTensor` of shape `(num_patches, 3)`, *optional*):
            Source `(time, height, width)` coordinates in spatial merge block order.
        """
        hidden_states = self.layernorm_pre(self.embeddings(pixel_values))
        if patch_positions is None:
            patch_positions = get_vision_position_ids(
                image_grid_thw, self.config.spatial_merge_size, include_temporal=True
            )
        patch_positions = patch_positions.reshape(-1, 3)
        if len(patch_positions) != len(hidden_states):
            raise ValueError("patch_positions must contain one coordinate per input patch")
        frame_counts = image_grid_thw[:, 0]
        spatial_sizes = image_grid_thw[:, 1:].prod(-1)
        if (frame_counts * spatial_sizes).sum() != len(hidden_states):
            raise ValueError("image_grid_thw does not match the number of input patches")
        if self.config.frame_windows_size:
            window = self.config.frame_windows_size
            windows_per_grid = (frame_counts + window - 1) // window
            sizes = spatial_sizes.repeat_interleave(windows_per_grid)
            frames_per_window = torch.full_like(sizes, window)
            frames_per_window[windows_per_grid.cumsum(0) - 1] = (frame_counts - 1) % window + 1
            lengths = frames_per_window * sizes
        else:
            lengths = frame_counts * spatial_sizes
        cu_seqlens = nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)).to(hidden_states.device)
        hidden_states = self.encoder(
            hidden_states, position_embeddings=self.video_rope(patch_positions), cu_seqlens=cu_seqlens, **kwargs
        )
        return BaseModelOutputWithPooling(last_hidden_state=self.merger(hidden_states, patch_positions))


@auto_docstring
class LlavaOnevision2Model(LlavaModel):
    config_class = LlavaOnevision2Config

    def __init__(self, config: LlavaOnevision2Config):
        PreTrainedModel.__init__(self, config)
        self.visual = LlavaOnevision2VisionModel(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        patch_positions: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width dimensions of each patch grid.
        patch_positions (`torch.LongTensor` of shape `(num_patches, 3)`, *optional*):
            Source coordinates for image, frame or codec patches.
        """
        outputs = self.visual(
            pixel_values, image_grid_thw, patch_positions=patch_positions, return_dict=True, **kwargs
        )
        sizes = (image_grid_thw.prod(-1) // self.config.vision_config.spatial_merge_size**2).tolist()
        outputs.pooler_output = outputs.last_hidden_state.split(sizes)
        return outputs

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        patch_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LlavaOnevision2ModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height and width dimensions of each patch grid.
        patch_positions (`torch.LongTensor` of shape `(num_patches, 3)`, *optional*):
            Source coordinates for image, frame or codec patches.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("Specify pixel_values or mm_encoder_outputs, not both")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        encoder_outputs = mm_encoder_outputs or {}
        if pixel_values is not None:
            encoder_outputs = {
                "image": self.get_image_features(pixel_values, image_grid_thw, patch_positions, **kwargs)
            }
        if encoder_outputs.get("image") is not None:
            image_features = torch.cat(encoder_outputs["image"].pooler_output, dim=0).to(inputs_embeds)
            mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)
        if position_ids is None and attention_mask is not None and attention_mask.ndim == 2:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -inputs_embeds.shape[1] :]
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            **kwargs,
        )
        return LlavaOnevision2ModelOutputWithPast(**outputs, image_outputs=encoder_outputs.get("image"))


@auto_docstring
class LlavaOnevision2ForConditionalGeneration(LlavaForConditionalGeneration):
    config_class = LlavaOnevision2Config

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        patch_positions: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Temporal, height and width dimensions of each patch grid.
        patch_positions (`torch.LongTensor` of shape `(num_patches, 3)`, *optional*):
            Source coordinates for image, frame or codec patches.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, patch_positions, **kwargs)

    @can_return_tuple
    @auto_docstring
    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        patch_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LlavaOnevision2CausalLMOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            Temporal, height and width dimensions of each patch grid.
        patch_positions (`torch.LongTensor` of shape `(num_patches, 3)`, *optional*):
            Source coordinates for image, frame or codec patches.
        """
        # Keep LLaVA's logits and loss logic, replacing only the base-model call.
        super().forward(**kwargs)
        outputs = self.model(  # noqa: F841
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            patch_positions=patch_positions,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )


__all__ = [
    "LlavaOnevision2PreTrainedModel",
    "LlavaOnevision2VisionModel",
    "LlavaOnevision2Model",
    "LlavaOnevision2ForConditionalGeneration",
    "LlavaOnevision2ImageProcessor",
    "LlavaOnevision2ImageProcessorPil",
]
