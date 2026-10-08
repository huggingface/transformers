# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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
import itertools
import math

import torch
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...distributed.utils import is_dtensor
from ...image_processing_utils import BatchFeature
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling, SizeDict
from ...masking_utils import create_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from ...processing_utils import MultiModalData, ProcessingKwargs, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling
from ...utils.output_capturing import OutputRecorder
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos
from ..ernie4_5.modeling_ernie4_5 import apply_rotary_pos_emb  # noqa: F401
from ..hunyuan_vl.configuration_hunyuan_vl import HunYuanVLConfig, HunYuanVLVisionConfig
from ..hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLRMSNorm,
    HunYuanVLVisionPatchEmbed,
    HunYuanVLVisionPatchMerger,
    HunYuanVLVisionTransformer,
)
from ..hy_v4.configuration_hy_v4 import HYV4Config
from ..hy_v4.modeling_hy_v4 import (
    HYV4Attention,
    HYV4DecoderLayer,
    HYV4Model,
    HYV4PreTrainedModel,
    HYV4TopkRouter,
    repeat_kv,
)
from ..qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from ..qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor, Qwen3VLVideoProcessorInitKwargs


@auto_docstring(checkpoint="tencent/Hy4")
@strict
class HYV4VLVisionConfig(HunYuanVLVisionConfig):
    r"""
    interpolate_mode (`str`, *optional*, defaults to `"bilinear"`):
        Interpolation mode for the learned patch positional embeddings.
    out_hidden_size (`int`, *optional*, defaults to 6144):
        Width of the visual features passed to the text backbone.
    max_image_size (`int`, *optional*, defaults to 2048):
        Image size used to define the learned positional embedding grid.
    """

    img_max_token_num = AttributeError()
    min_image_size = AttributeError()
    max_vit_seq_len = AttributeError()
    text_hidden_size = AttributeError()
    temporal_patch_size = AttributeError()

    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 40
    out_hidden_size: int = 6144


@auto_docstring(checkpoint="tencent/Hy4")
@strict
class HYV4VLTextConfig(HYV4Config):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of expert groups for routing.
    topk_group (`int`, *optional*, defaults to 1):
        Number of expert groups kept during routing.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer MLP type, either `"dense"` or `"sparse"`. Defaults to one dense layer followed by sparse layers.
    index_topk (`int`, *optional*, defaults to 2048):
        Maximum number of key positions selected by each DSA query.
    index_head_dim (`int`, *optional*, defaults to 128):
        Hidden dimension of each DSA indexer head.
    index_n_heads (`int`, *optional*, defaults to 32):
        Number of DSA indexer heads.
    indexer_types (`list[str]`, *optional*):
        Per-layer DSA indexer type, either `"full"` or `"shared"` (reuses the previous full layer's selection).
    hc_mult (`int`, *optional*, defaults to 4):
        Number of hidden-state channels maintained by iHC.
    hc_magnitude (`float`, *optional*, defaults to 2.0):
        Scale applied to the iHC post-gating branch.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical epsilon added to iHC sigmoid gates.
    learnable_sink_init (`float`, *optional*, defaults to 0.0):
        Initial value of each learned attention-sink logit.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Magnitude of the routed-expert SwiGLU clamp. Values at or below zero disable the clamp.
    """

    model_type = "hy_v4_vl_text"
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section"}
    base_model_tp_plan = {
        "embed_tokens": "embedding_rowwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.gate_proj": "colwise",
        "layers.*.self_attn.sinks": "colwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    hidden_size: int = 6144
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    index_n_heads: int = 32
    output_router_logits: bool = False

    def convert_rope_params_to_dict(self, **kwargs):
        kwargs = PreTrainedConfig.convert_rope_params_to_dict(self, **kwargs)
        self.rope_parameters.setdefault("mrope_section", [12, 10, 10])
        return kwargs

    def validate_architecture(self):
        PreTrainedConfig.validate_architecture(self)
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2:
            raise ValueError("`qk_rope_head_dim` must be a positive even integer.")
        if self.nested_rope_parameter_keys(self.rope_parameters):
            raise ValueError("HY4-VL requires a single multimodal RoPE configuration.")
        mrope_section = self.rope_parameters["mrope_section"]
        if not isinstance(mrope_section, (list, tuple)) or len(mrope_section) != 3:
            raise ValueError("HY4-VL requires three `mrope_section` entries (time, height, width).")
        if any(type(section) is not int or section < 0 for section in mrope_section):
            raise ValueError("`mrope_section` entries must be non-negative integers.")
        expected = self.qk_rope_head_dim // 2
        if sum(mrope_section) != expected:
            raise ValueError(f"`mrope_section` must sum to `qk_rope_head_dim // 2` = {expected}.")
        if mrope_section[1] > len(range(1, expected, 3)) or mrope_section[2] > len(range(2, expected, 3)):
            raise ValueError("`mrope_section` height/width entries exceed the interleaved frequency slots.")
        if self.rope_parameters.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("HY4-VL rotates the complete `qk_rope_head_dim` slice.")


@auto_docstring(checkpoint="tencent/Hy4")
@strict
class HYV4VLConfig(HunYuanVLConfig):
    """Configuration for the HY4-VL text and vision backbones."""

    im_start_id = AttributeError()
    im_end_id = AttributeError()
    im_newline_id = AttributeError()

    model_type = "hy_v4_vl"
    sub_configs = {"vision_config": HYV4VLVisionConfig, "text_config": HYV4VLTextConfig}

    video_token_id: int = 120124
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        for key, config_class in self.sub_configs.items():
            config = getattr(self, key)
            if isinstance(config, dict):
                setattr(self, key, config_class(**config))
            elif config is None:
                setattr(self, key, config_class())
        PreTrainedConfig.__post_init__(self, **kwargs)


class HYV4VLAttention(HYV4Attention):
    def __init__(self, config: HYV4VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Latent norms use the checkpoint `rms_norm_eps`
        self.q_a_layernorm = HYV4VLRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = HYV4VLRMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)


class HYV4VLTopkRouter(HYV4TopkRouter):
    pass


class HYV4VLDecoderLayer(HYV4DecoderLayer):
    pass


@auto_docstring
class HYV4VLPreTrainedModel(HYV4PreTrainedModel):
    _can_record_outputs = {
        "hidden_states": HYV4VLDecoderLayer,
        "attentions": HYV4VLAttention,
        "router_logits": OutputRecorder(HYV4VLTopkRouter, index=0),
    }
    config: HYV4VLConfig
    input_modalities = ("image", "video", "text")
    _no_split_modules = ["HYV4VLDecoderLayer", "HYV4VLVisionBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, HYV4VLVisionPatchMerger):
            init.normal_(module.image_newline, mean=0.0, std=module.out_hidden_size**-0.5)


class HYV4VLRMSNorm(HunYuanVLRMSNorm):
    pass


class HYV4VLVisionPatchEmbed(HunYuanVLVisionPatchEmbed):
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        patch_pos_embed = self.position_embedding.weight[1:, :].reshape(
            1, self.position_edge, self.position_edge, self.embed_dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2).float()
        # Key difference: an explicit `scale_factor` of `(size + 0.1) / edge`, which shifts the sampling grid
        scale_h = (int(height) + 0.1) / self.position_edge
        scale_w = (int(width) + 0.1) / self.position_edge
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(scale_h, scale_w),
            mode=self.config.interpolate_mode,
            align_corners=False,
        )
        return patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim).to(embeddings.dtype)


class HYV4VLVisionPatchMerger(HunYuanVLVisionPatchMerger):
    def __init__(self, config: HYV4VLVisionConfig):
        nn.Module.__init__(self)
        self.config = config
        hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.out_hidden_size = config.out_hidden_size

        self.before_rms = HYV4VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj_conv = nn.Conv2d(
            config.hidden_size,
            hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.proj_act = ACT2FN[config.hidden_act]
        self.proj_out = nn.Conv2d(hidden_size, config.out_hidden_size, kernel_size=1)
        self.image_newline = nn.Parameter(torch.empty(config.out_hidden_size))

    def forward(self, hidden_states: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        hidden_states = self.before_rms(hidden_states)
        dtype = hidden_states.dtype
        # Frames are merged independently, without HunYuanVL's final MLP or boundary embeddings.
        hidden_states = hidden_states.reshape(-1, *size, self.config.hidden_size).permute(0, 3, 1, 2)
        hidden_states = self.proj_conv(hidden_states)
        hidden_states = self.proj_act(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = torch.cat(
            [
                hidden_states,
                self.image_newline.reshape(1, channels, 1, 1)
                .expand(batch_size, channels, height, 1)
                .to(dtype, non_blocking=True),
            ],
            dim=-1,
        )
        hidden_states = hidden_states.reshape(batch_size, channels, -1).permute(0, 2, 1)
        return hidden_states.reshape(1, -1, self.out_hidden_size)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Shared by both towers: only the text attention has learned sinks
    sinks = getattr(module, "sinks", None)
    if sinks is None:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    else:
        if is_dtensor(sinks):
            sinks = sinks.to_local()
        sinks = sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        scores = nn.functional.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)[..., :-1]
        attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class HYV4VLVisionTransformer(HunYuanVLVisionTransformer):
    config: HYV4VLVisionConfig
    # Only the text attention (sinks + DSA) is restricted to eager
    _supports_sdpa = True
    _supports_flash_attn = True


class HYV4VLRotaryEmbedding(Qwen3VLTextRotaryEmbedding):
    def __init__(self, config: HYV4VLTextConfig, device=None):
        super().__init__(config, device=device)
        if device is not None:
            self.to(device)


class HYV4VLTextModel(HYV4Model):
    config: HYV4VLTextConfig
    input_modalities = ("text",)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # As in Qwen2-VL, an optional leading text axis is used to build the causal mask
        text_position_ids = None
        if position_ids.shape[0] == 4:
            text_position_ids, position_ids = position_ids[0], position_ids[1:]

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = {
                "indexed_attention": create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                    # Always build the mask so the DSA indexer sees causality.
                    allow_is_causal_skip=False,
                )
            }

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        topk_indices = None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping["indexed_attention"],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prev_topk_indices=topk_indices,
                **kwargs,
            )
        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring
class HYV4VLModel(Qwen2VLModel):
    config: HYV4VLConfig

    def __init__(self, config: HYV4VLConfig):
        super().__init__(config)
        self.vision_tower = HYV4VLVisionTransformer(config.vision_config)
        del self.visual

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int] | torch.Tensor,
        temporal_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """All frame positions, with a newline column and a fresh position block for each frame."""
        grid_t, grid_h, grid_w = (int(value) for value in grid_thw)
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size
        position_height, position_width = torch.meshgrid(
            torch.arange(merged_h, dtype=torch.long, device=device),
            torch.arange(merged_w + 1, dtype=torch.long, device=device),
            indexing="ij",
        )
        frame_positions = torch.stack(
            [torch.zeros_like(position_height), position_height, position_width], dim=0
        ).flatten(1)
        # All three axes advance past the previous frame, including its newline column.
        offsets = torch.arange(grid_t, device=device) * max(merged_h, merged_w + 1) + start_position
        return (frame_positions[:, None, :] + offsets[None, :, None]).flatten(1)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Same as [`Qwen2VLModel.get_rope_index`], except that every image and every video frame is a `(height,
        width + 1)` block after which the running position advances past the block's maximum.
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        position_ids = torch.zeros(
            3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
        )
        image_grids = iter(image_grid_thw) if image_grid_thw is not None else iter(())
        video_grids = iter(video_grid_thw) if video_grid_thw is not None else iter(())
        image_count = video_count = 0
        rope_deltas = []

        for batch_idx, current_input_ids in enumerate(input_ids):
            token_type = mm_token_type_ids[batch_idx]
            valid_token_mask = None
            if attention_mask is not None:
                valid_token_mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[valid_token_mask]
                token_type = token_type[valid_token_mask]

            current_position_ids = torch.zeros(
                3, current_input_ids.shape[-1], dtype=input_ids.dtype, device=input_ids.device
            )
            running = 0

            for modality, group in itertools.groupby(enumerate(token_type.tolist()), lambda pair: pair[1]):
                group = list(group)
                span_start, span_end = group[0][0], group[-1][0] + 1
                span_length = span_end - span_start

                if modality == 0:
                    text_index = torch.arange(span_length, dtype=input_ids.dtype, device=input_ids.device) + running
                    current_position_ids[:, span_start:span_end] = text_index
                    running += span_length
                    continue

                if modality == 1:
                    grid = next(image_grids, None)
                    if grid is None:
                        raise ValueError("Found more image placeholder spans than entries in `image_grid_thw`.")
                    image_count += 1
                else:
                    grid = next(video_grids, None)
                    if grid is None:
                        raise ValueError("Found more video placeholder spans than entries in `video_grid_thw`.")
                    video_count += 1

                vision_positions = self.get_vision_position_ids(
                    running, grid, spatial_merge_size=spatial_merge_size, device=input_ids.device
                )
                if span_length != vision_positions.shape[1]:
                    raise ValueError(
                        f"Placeholder span length does not match the visual grid: span_length={span_length}, "
                        f"expected {vision_positions.shape[1]}."
                    )
                current_position_ids[:, span_start:span_end] = vision_positions
                running = int(vision_positions.max()) + 1

            if valid_token_mask is not None:
                position_ids[:, batch_idx, valid_token_mask] = current_position_ids
            else:
                position_ids[:, batch_idx] = current_position_ids
            rope_deltas.append(running - len(current_input_ids))

        if image_grid_thw is not None and image_count != len(image_grid_thw):
            raise ValueError(
                "Found fewer image placeholder spans than entries in `image_grid_thw`: "
                f"spans={image_count}, images={len(image_grid_thw)}."
            )
        if video_grid_thw is not None and video_count != len(video_grid_thw):
            raise ValueError(
                "Found fewer video placeholder spans than entries in `video_grid_thw`: "
                f"spans={video_count}, videos={len(video_grid_thw)}."
            )

        rope_deltas = torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, rope_deltas

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if position_ids is not None:
            if not is_torchdynamo_compiling() and (past_key_values is None or past_key_values.get_seq_length() == 0):
                # Explicit prefill positions define the offset for later automatic decoding, too.
                vision_positions = position_ids[-3:] if position_ids.ndim == 3 else position_ids.unsqueeze(0)
                if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
                    valid_lengths = attention_mask.long().sum(-1)
                    vision_positions = vision_positions.masked_fill(~attention_mask.bool().unsqueeze(0), -1)
                elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
                    valid_lengths = position_ids[0].amax(-1) + 1
                else:
                    valid_lengths = inputs_embeds.shape[1]
                self.rope_deltas = (vision_positions.amax(dim=(0, 2)) + 1 - valid_lengths).unsqueeze(1)
            return position_ids
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_length == 0:
            # A new request must not reuse offsets from a previous multimodal sequence.
            self.rope_deltas = None
            if image_grid_thw is not None or video_grid_thw is not None:
                if mm_token_type_ids is None:
                    raise ValueError("Multimodal prefill requires mm_token_type_ids or explicit position_ids.")
                # The scanner uses modality spans, not token values, so embeddings-only inputs are sufficient.
                position_input = input_ids if input_ids is not None else mm_token_type_ids.long()
                position_ids, self.rope_deltas = self.get_rope_index(
                    position_input,
                    mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                return position_ids

        batch_size, seq_length = inputs_embeds.shape[:2]
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            positions = attention_mask.long().cumsum(-1) - 1
            positions = positions.masked_fill(attention_mask == 0, 0)[:, -seq_length:]
        else:
            positions = torch.arange(past_length, past_length + seq_length, device=inputs_embeds.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
        position_ids = positions.unsqueeze(0).expand(3, -1, -1).to(inputs_embeds.device)
        if past_length > 0 and self.rope_deltas is not None:
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(inputs_embeds.device)
        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeModelOutputWithPast:
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs.")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("image") is not None:
            image_embeds = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if mm_encoder_outputs.get("video") is not None:
            video_embeds = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
        )
        kwargs.setdefault("output_router_logits", self.config.text_config.output_router_logits)
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return MoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        vision_outputs = self.vision_tower(pixel_values.to(self.vision_tower.dtype), grid_thw=image_grid_thw, **kwargs)
        # Each merged row carries one extra newline token
        merge = self.config.vision_config.spatial_merge_size
        split_sizes = image_grid_thw[:, 0] * (image_grid_thw[:, 1] // merge) * (image_grid_thw[:, 2] // merge + 1)
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output.squeeze(0), split_sizes.tolist())
        return vision_outputs

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        vision_outputs = self.vision_tower(
            pixel_values_videos.to(self.vision_tower.dtype), grid_thw=video_grid_thw, **kwargs
        )
        merge = self.config.vision_config.spatial_merge_size
        split_sizes = video_grid_thw[:, 0] * (video_grid_thw[:, 1] // merge) * (video_grid_thw[:, 2] // merge + 1)
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output.squeeze(0), split_sizes.tolist())
        return vision_outputs


@auto_docstring
class HYV4VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    _keep_in_fp32_modules_strict = HYV4VLPreTrainedModel._keep_in_fp32_modules_strict + ["lm_head"]
    _tp_plan = {"lm_head": "colwise_gather_output"}

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoProcessor, DistributedConfig, HYV4VLForConditionalGeneration

        >>> model_path = "tencent/Hy4"  # Tentative release identifier.
        >>> processor = AutoProcessor.from_pretrained(model_path)
        >>> model = HYV4VLForConditionalGeneration.from_pretrained(
        ...     model_path, distributed_config=DistributedConfig(tp_plan="auto"),
        ... )

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "path/to/your/image.jpg"},
        ...             {"type": "text", "text": "Describe this image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, reasoning_effort="no_think",
        ...     return_tensors="pt", return_dict=True,
        ... ).to(model.device)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        >>> generated_trimmed = generated_ids[0][inputs["input_ids"].shape[-1]:]
        >>> print(processor.decode(generated_trimmed, skip_special_tokens=True))
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            mm_encoder_outputs=mm_encoder_outputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # Key difference: the LM head is kept in fp32
        logits = self.lm_head(hidden_states[:, slice_indices, :].to(self.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def _expand_multimodal_outputs(
        self,
        input_ids: torch.LongTensor,
        mm_encoder_output: dict,
        expand_size: int = 1,
        inputs_embeds: torch.Tensor | None = None,
    ) -> dict:
        for modality in ("image", "video"):
            outputs = mm_encoder_output.get(modality)
            if outputs is None or outputs.pooler_output is None:
                continue
            token_id = getattr(self.config, f"{modality}_token_id")
            if (input_ids is None or input_ids.numel() == 0) and inputs_embeds is not None:
                token_embed = self.get_input_embeddings()(
                    torch.tensor(token_id, device=inputs_embeds.device, dtype=torch.long)
                )
                counts = (inputs_embeds == token_embed).all(-1).sum(-1)
            else:
                counts = (input_ids == token_id).sum(-1)
            sample_ends = set(itertools.accumulate(counts.tolist()))
            features = outputs.pooler_output
            feature_ends = itertools.accumulate(feature.shape[0] for feature in features)
            offsets = [0] + [index + 1 for index, end in enumerate(feature_ends) if end in sample_ends]
            # This base repeats individual images; repeat the whole sample's visual sequence instead.
            outputs.pooler_output = [
                feature
                for start, end in zip(offsets[:-1], offsets[1:])
                for _ in range(expand_size)
                for feature in features[start:end]
            ]
        return mm_encoder_output


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_image_size: int,
) -> tuple[int, int]:
    """Resize a frame under both the clip's pixel budget and the per-frame area limit."""
    if min(num_frames, height, width, factor, min_pixels, max_pixels, max_image_size) <= 0:
        raise ValueError("Video dimensions, resize factor, and pixel budgets must be positive.")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    per_frame_budget = min(max_pixels / num_frames, max_image_size)
    min_cells = math.ceil(min_pixels / factor**2)
    max_cells = math.floor(per_frame_budget / factor**2)
    if min_cells > max_cells:
        raise ValueError("The video pixel budget cannot fit a factor-aligned frame at the minimum pixel area.")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > per_frame_budget:
        beta = math.sqrt((height * width) / per_frame_budget)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    if min_pixels <= h_bar * w_bar <= per_frame_budget:
        return h_bar, w_bar

    # Only repair boundary cases where clamping or rounding violated the area limits.
    # Enumerate the shorter grid edge to find a feasible shape close to the input aspect ratio.
    candidates = []
    for short_edge in range(1, math.isqrt(max_cells) + 1):
        lower = max(short_edge, math.ceil(min_cells / short_edge))
        upper = max_cells // short_edge
        if lower <= upper:
            long_edge = min(upper, max(lower, round(short_edge * max(height, width) / min(height, width))))
            grid_h, grid_w = (short_edge, long_edge) if height <= width else (long_edge, short_edge)
            candidates.append((grid_h, grid_w))
    grid_h, grid_w = min(
        candidates, key=lambda grid: (abs(math.log(grid[0] * width / (grid[1] * height))), -grid[0] * grid[1])
    )
    return grid_h * factor, grid_w * factor


class HYV4VLVideoProcessorInitKwargs(Qwen3VLVideoProcessorInitKwargs, total=False):
    r"""
    size (`dict[str, int]`, *optional*):
        Minimum per-frame pixel area (`shortest_edge`) and total clip pixel budget (`longest_edge`).
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    min_frames (`int`, *optional*, defaults to 2):
        The minimum number of frames to sample from a video.
    max_frames (`int`, *optional*, defaults to 1024):
        The maximum number of frames to sample from a video.
    max_image_size (`dict[str, int]`, *optional*):
        Per-frame pixel-area limit, specified by the `longest_edge` key.
    """

    cap_pixels_per_frame = AttributeError()
    max_video_tokens = AttributeError()
    max_image_size: dict[str, int]


@auto_docstring
class HYV4VLVideoProcessor(Qwen3VLVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 4096, "longest_edge": 33554432}
    max_image_size = {"longest_edge": 1024 * 1024}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    temporal_patch_size = 1
    min_frames = 2
    max_frames = 1024
    do_sample_frames = False
    cap_pixels_per_frame = AttributeError()
    max_video_tokens = AttributeError()
    valid_kwargs = HYV4VLVideoProcessorInitKwargs

    def resize(
        self,
        videos: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        max_image_size: dict[str, int],
        **kwargs,
    ) -> "torch.Tensor":
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(f"`size` dict must contain 'shortest_edge' and 'longest_edge' keys but got {size}.")
        height, width = smart_resize(
            num_frames=videos.shape[1],
            height=videos.shape[-2],
            width=videos.shape[-1],
            factor=factor,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
            max_image_size=max_image_size["longest_edge"],
        )
        return BaseVideoProcessor.resize(
            self,
            image=videos,
            size=SizeDict(height=height, width=width),
            resample=resample,
        )

    def patchify(
        self,
        videos: "torch.Tensor",
        patch_size: int,
        merge_size: int,
        temporal_patch_size: int,
    ) -> tuple["torch.Tensor", int, int, int]:
        batch_size, num_frames, channel, resized_height, resized_width = videos.shape
        if pad := -num_frames % temporal_patch_size:
            videos = torch.cat((videos, videos[:, -1:].expand(-1, pad, -1, -1, -1)), dim=1)
            num_frames += pad

        grid_t = num_frames // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = videos.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Key difference: row-major order within each merged block, as in the image processor
        patches = patches.permute(0, 1, 4, 5, 7, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, grid_t, grid_h, grid_w

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        max_image_size: dict[str, int],
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_resize:
                stacked_videos = self.resize(
                    videos=stacked_videos,
                    size=size,
                    resample=resample,
                    factor=patch_size * merge_size,
                    max_image_size=max_image_size,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches, grid_t, grid_h, grid_w = self.patchify(
                stacked_videos,
                patch_size=patch_size,
                merge_size=merge_size,
                temporal_patch_size=temporal_patch_size,
            )
            processed_videos_grouped[shape] = patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * len(stacked_videos)

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        return BatchFeature(
            data={
                "pixel_values_videos": torch.cat(processed_videos, dim=0),
                "video_grid_thw": torch.tensor(processed_grids),
            },
            tensor_type=return_tensors,
        )

    def get_num_of_video_patches(self, num_frames: int, height: int, width: int, videos_kwargs=None) -> int:
        """
        Number of video patches for a given clip size.
        Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
        without a video.
        """
        grid_t, grid_h, grid_w = self.get_video_grid(num_frames, height, width, videos_kwargs)
        return grid_t * grid_h * grid_w

    def get_video_grid(
        self, num_frames: int, height: int, width: int, videos_kwargs: dict | None = None
    ) -> tuple[int, int, int]:
        videos_kwargs = videos_kwargs or {}
        size = videos_kwargs.get("size", self.size)
        max_image_size = videos_kwargs.get("max_image_size", self.max_image_size)
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        if videos_kwargs.get("do_resize", self.do_resize):
            height, width = smart_resize(
                num_frames=num_frames,
                height=height,
                width=width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
                max_image_size=max_image_size["longest_edge"],
            )
        grid_t = (num_frames + -num_frames % temporal_patch_size) // temporal_patch_size
        return grid_t, height // patch_size, width // patch_size


class HYV4VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "return_mm_token_type_ids": True,
        },
    }


@auto_docstring
class HYV4VLProcessor(Qwen2VLProcessor):
    valid_processor_kwargs = HYV4VLProcessorKwargs

    @staticmethod
    def _num_frame_tokens(grid_h: int, grid_w: int, merge_size: int) -> int:
        # One extra newline token per merged row
        return (grid_h // merge_size) * (grid_w // merge_size + 1)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        _, grid_h, grid_w = image_inputs["image_grid_thw"][image_idx]
        merge_size = kwargs.get("merge_size") or self.image_processor.merge_size
        return self.image_token * self._num_frame_tokens(grid_h, grid_w, merge_size)

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        grid_t, grid_h, grid_w = video_inputs["video_grid_thw"][video_idx]
        merge_size = kwargs.get("merge_size") or self.video_processor.merge_size
        return self.video_token * (grid_t * self._num_frame_tokens(grid_h, grid_w, merge_size))

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = {**kwargs, **kwargs.get("images_kwargs", {})}
            merge_size = images_kwargs.get("merge_size") or self.image_processor.merge_size
            grids = [
                self.image_processor.get_number_of_image_patches(height, width, images_kwargs)
                for height, width in image_sizes
            ]
            vision_data["num_image_patches"] = [grid_h * grid_w for grid_h, grid_w in grids]
            vision_data["num_image_tokens"] = [self._num_frame_tokens(*grid, merge_size) for grid in grids]
        if video_sizes is not None:
            videos_kwargs = {**kwargs, **kwargs.get("videos_kwargs", {})}
            merge_size = videos_kwargs.get("merge_size") or self.video_processor.merge_size
            num_video_tokens = []
            for num_frames, height, width in video_sizes:
                grid_t, grid_h, grid_w = self.video_processor.get_video_grid(num_frames, height, width, videos_kwargs)
                num_video_tokens.append(grid_t * self._num_frame_tokens(grid_h, grid_w, merge_size))
            vision_data["num_video_tokens"] = num_video_tokens
        return MultiModalData(**vision_data)


__all__ = [
    "HYV4VLConfig",
    "HYV4VLVisionConfig",
    "HYV4VLTextConfig",
    "HYV4VLPreTrainedModel",
    "HYV4VLModel",
    "HYV4VLTextModel",
    "HYV4VLVisionTransformer",
    "HYV4VLForConditionalGeneration",
    "HYV4VLVideoProcessor",
    "HYV4VLProcessor",
]
