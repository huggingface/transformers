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
from ...image_utils import PILImageResampling, SizeDict
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling
from ...processing_utils import MultiModalData, ProcessingKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import accepts_precomputed_kwargs
from ...video_processing_utils import BaseVideoProcessor
from ..ernie4_5.modeling_ernie4_5 import apply_rotary_pos_emb  # noqa: F401
from ..hunyuan_vl.configuration_hunyuan_vl import HunYuanVLConfig, HunYuanVLVisionConfig
from ..hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor
from ..hunyuan_vl.image_processing_pil_hunyuan_vl import HunYuanVLImageProcessorPil
from ..hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLRMSNorm,
    HunYuanVLVisionPatchEmbed,
    HunYuanVLVisionTransformer,
)
from ..hy_v4.configuration_hy_v4 import HYV4Config
from ..hy_v4.modeling_hy_v4 import HYV4Attention, HYV4Model, HYV4PreTrainedModel, repeat_kv
from ..qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding
from ..qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor, Qwen3VLVideoProcessorInitKwargs


@auto_docstring
@strict
class HYV4VLVisionConfig(HunYuanVLVisionConfig):
    r"""
    interpolate_mode (`str`, *optional*, defaults to `"bilinear"`):
        Interpolation mode used when resizing learned patch positional embeddings to match the current image grid.
    out_hidden_size (`int`, *optional*, defaults to 6144):
        Output hidden size produced by the vision tower before it is consumed by the text backbone.
    img_max_token_num (`int`, *optional*, defaults to 4096):
        Maximum image token count expected by the vision stack.
    max_image_size (`int`, *optional*, defaults to 2048):
        Maximum supported image size, which sets the size of the learned positional grid.
    min_image_size (`int`, *optional*, defaults to 512):
        Minimum supported image size.
    max_vit_seq_len (`int`, *optional*, defaults to 16384):
        Maximum sequence length produced by the vision transformer.
    text_hidden_size (`int`, *optional*, defaults to 3072):
        Hidden size of the text backbone. Unused by HY4-VL, whose projector outputs `out_hidden_size`.
    """

    model_type = "hy_v4_vl_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_hidden_layers: int = 40
    out_hidden_size: int = 6144


@auto_docstring
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
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}
    base_model_tp_plan = {
        "embed_tokens": "embedding_rowwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.gate_proj": "colwise",
        "layers.*.self_attn.sinks": "colwise_local",
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

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        rope_parameters = self.rope_parameters
        rope_parameters.setdefault("mrope_section", [12, 10, 10])
        mrope_section = rope_parameters["mrope_section"]
        if len(mrope_section) != 3:
            raise ValueError(
                f"HY4-VL uses three multimodal RoPE axes (time, height, width), got {len(mrope_section)}."
            )
        expected = self.qk_rope_head_dim // 2
        total = sum(int(section) for section in mrope_section)
        if total != expected:
            raise ValueError(
                f"`mrope_section` must sum to `qk_rope_head_dim // 2` = {expected}, got {mrope_section} (sum {total})."
            )
        rope_parameters["mrope_section"] = [int(section) for section in mrope_section]


@auto_docstring
@strict
class HYV4VLConfig(HunYuanVLConfig):
    r"""
    im_start_id (`int`, *optional*, defaults to 120118):
        Token id marking the beginning of an image span.
    im_end_id (`int`, *optional*, defaults to 120119):
        Token id marking the end of an image span.
    im_newline_id (`int`, *optional*, defaults to 120121):
        Token id of the newline separator appended to every merged visual row.
    video_token_id (`int`, *optional*, defaults to 120124):
        Token id of the video placeholder.

    Example:

    ```python
    >>> from transformers import HYV4VLConfig, HYV4VLForConditionalGeneration
    >>>
    >>> configuration = HYV4VLConfig()
    >>> model = HYV4VLForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "hy_v4_vl"
    sub_configs = {"vision_config": HYV4VLVisionConfig, "text_config": HYV4VLTextConfig}

    video_token_id: int = 120124
    tie_word_embeddings: bool = False


class HYV4VLAttention(HYV4Attention):
    def __init__(self, config: HYV4VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Latent norms use the checkpoint `rms_norm_eps`
        self.q_a_layernorm = HYV4VLRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = HYV4VLRMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)


@auto_docstring
class HYV4VLPreTrainedModel(HYV4PreTrainedModel):
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


class HYV4VLVisionPatchMerger(nn.Module):
    def __init__(self, config: HYV4VLVisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.out_hidden_size = config.out_hidden_size

        self.norm = HYV4VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj_conv = nn.Conv2d(
            config.hidden_size,
            self.hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.act = ACT2FN[config.hidden_act]
        self.proj_out = nn.Conv2d(self.hidden_size, config.out_hidden_size, kernel_size=1)
        self.image_newline = nn.Parameter(torch.empty(config.out_hidden_size))

    def forward(self, hidden_states: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        height, width = int(size[0]), int(size[1])
        hidden_states = self.norm(hidden_states)
        # Frames are merged independently
        hidden_states = hidden_states.reshape(-1, height, width, self.config.hidden_size).permute(0, 3, 1, 2)
        hidden_states = self.proj_out(self.act(self.proj_conv(hidden_states)))

        hidden_states = hidden_states.permute(0, 2, 3, 1)
        num_frames, merged_height = hidden_states.shape[0], hidden_states.shape[1]
        # One newline embedding per merged row
        hidden_states = torch.cat(
            [
                hidden_states,
                self.image_newline.reshape(1, 1, 1, self.out_hidden_size).expand(
                    num_frames, merged_height, 1, self.out_hidden_size
                ),
            ],
            dim=2,
        )
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
    pass


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
    ) -> BaseModelOutputWithPast:
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
                "deepseek_sparse_attention": create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                    # Always build the mask so the DSA indexer sees causality.
                    allow_is_causal_skip=False,
                )
            }

        input_ids = None
        attention_mask = causal_mask_mapping
        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )


@auto_docstring
class HYV4VLModel(Qwen2VLModel):
    config: HYV4VLConfig

    def __init__(self, config: HYV4VLConfig):
        super().__init__(config)
        self.vision_tower = HYV4VLVisionTransformer(config.vision_config)
        del self.visual

    def get_vision_position_ids(
        self,
        grid_hw: list[int] | torch.Tensor,
        spatial_merge_size: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """`(time, height, width)` positions of one frame; the width axis includes the newline column."""
        grid_h, grid_w = (int(value) for value in grid_hw)
        merged_h = grid_h // spatial_merge_size
        merged_w = grid_w // spatial_merge_size

        position_height, position_width = torch.meshgrid(
            torch.arange(merged_h, dtype=torch.long, device=device),
            torch.arange(merged_w + 1, dtype=torch.long, device=device),
            indexing="ij",
        )
        position_height = position_height.flatten()
        position_width = position_width.flatten()
        return torch.stack([torch.zeros_like(position_height), position_height, position_width], dim=0)

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
                    frames = 1
                else:
                    grid = next(video_grids, None)
                    if grid is None:
                        raise ValueError("Found more video placeholder spans than entries in `video_grid_thw`.")
                    video_count += 1
                    frames = int(grid[0])

                frame_positions = self.get_vision_position_ids(
                    grid[1:], spatial_merge_size=spatial_merge_size, device=input_ids.device
                )
                tokens_per_frame = frame_positions.shape[1]
                if span_length != tokens_per_frame * frames:
                    raise ValueError(
                        f"Placeholder span length does not match the visual grid: span_length={span_length}, "
                        f"expected {tokens_per_frame * frames} ({frames} frame(s) x {tokens_per_frame} tokens)."
                    )

                for frame_idx in range(frames):
                    start = span_start + frame_idx * tokens_per_frame
                    end = start + tokens_per_frame
                    current_position_ids[:, start:end] = frame_positions + running
                    running += int(frame_positions.max()) + 1

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

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring
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

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring
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


class HYV4VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    pass


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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | HYV4VLCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> import os
        >>> import torch
        >>> from transformers import AutoProcessor, DistributedConfig, HYV4VLForConditionalGeneration

        >>> model_path = "/path/to/hy4-vl-checkpoint"
        >>> torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        >>> processor = AutoProcessor.from_pretrained(model_path)
        >>> model = HYV4VLForConditionalGeneration.from_pretrained(
        ...     model_path, dtype=torch.bfloat16,
        ...     distributed_config=DistributedConfig(tp_size=int(os.environ["WORLD_SIZE"])),
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

        return HYV4VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def _get_image_nums_and_video_nums(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # No shared `vision_start_token_id`: each image or video is one contiguous run of placeholder tokens
        token_ids = (self.config.image_token_id, self.config.video_token_id)
        if inputs_embeds is None:
            masks = [input_ids == token_id for token_id in token_ids]
        else:
            embeds = self.get_input_embeddings()(torch.tensor(token_ids, device=inputs_embeds.device))
            masks = [(inputs_embeds == embed).all(-1) for embed in embeds]
        image_nums, video_nums = (
            (mask & ~nn.functional.pad(mask[:, :-1], (1, 0), value=False)).sum(dim=1) for mask in masks
        )
        return image_nums, video_nums


@auto_docstring
class HYV4VLImageProcessor(HunYuanVLImageProcessor):
    pass


@auto_docstring
class HYV4VLImageProcessorPil(HunYuanVLImageProcessorPil):
    pass


class HYV4VLVideoProcessorInitKwargs(Qwen3VLVideoProcessorInitKwargs, total=False):
    r"""
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
    cap_pixels_per_frame (`bool`, *optional*, defaults to `False`):
        Unused, the per-frame limit is `max_image_size`.
    max_video_tokens (`int`, *optional*, defaults to 768):
        Unused, see `cap_pixels_per_frame`.
    max_image_size (`dict[str, int]`, *optional*):
        Per-frame pixel-area limit, specified by the `longest_edge` key.
    """

    max_image_size: dict[str, int]


@auto_docstring
class HYV4VLVideoProcessor(Qwen3VLVideoProcessor):
    size = {"shortest_edge": 4096, "longest_edge": 33554432}
    max_image_size = {"longest_edge": 1024 * 1024}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    temporal_patch_size = 1
    min_frames = 2
    max_frames = 1024
    do_sample_frames = False
    cap_pixels_per_frame = False
    valid_kwargs = HYV4VLVideoProcessorInitKwargs

    def __init__(self, **kwargs: Unpack[HYV4VLVideoProcessorInitKwargs]):
        # Checkpoint spellings of `size` and `max_image_size`
        size = dict(kwargs.pop("size", None) or self.size)
        if (min_pixels := kwargs.pop("min_pixels", None)) is not None:
            size["shortest_edge"] = min_pixels
        if (max_pixels := kwargs.pop("max_pixels", None)) is not None:
            size["longest_edge"] = max_pixels
        if (video_max_image_size := kwargs.pop("video_max_image_size", None)) is not None:
            kwargs["max_image_size"] = {"longest_edge": int(video_max_image_size) ** 2}
        for unused in ("timestamp_token_per_frame", "timestamp_token_reserve", "spatial_patch_size"):
            kwargs.pop(unused, None)
        BaseVideoProcessor.__init__(self, size=size, **kwargs)

    def _get_frame_size(
        self, num_frames: int, height: int, width: int, factor: int, size: SizeDict
    ) -> tuple[int, int]:
        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        # `longest_edge` is a budget for the whole clip, further capped per frame by `max_image_size`
        per_frame_budget = min(size.longest_edge / max(num_frames, 1), self.max_image_size["longest_edge"])
        h_bar = max(factor, round(height / factor) * factor)
        w_bar = max(factor, round(width / factor) * factor)
        if h_bar * w_bar > per_frame_budget:
            beta = math.sqrt((height * width) / per_frame_budget)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < size.shortest_edge:
            beta = math.sqrt(size.shortest_edge / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def resize(
        self,
        videos: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        factor: int,
        **kwargs,
    ) -> "torch.Tensor":
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError(f"`size` dict must contain 'shortest_edge' and 'longest_edge' keys but got {size}.")
        height, width = self._get_frame_size(videos.shape[1], *videos.shape[-2:], factor, size)
        return BaseVideoProcessor.resize(
            self,
            image=videos,
            size=SizeDict(height=height, width=width),
            # The reference implementation resizes with PIL's default (BICUBIC)
            resample=PILImageResampling.BICUBIC,
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
        size = videos_kwargs.get("size") or self.size
        size = SizeDict(
            shortest_edge=videos_kwargs.get("min_pixels") or size["shortest_edge"],
            longest_edge=videos_kwargs.get("max_pixels") or size["longest_edge"],
        )
        patch_size = videos_kwargs.get("patch_size") or self.patch_size
        merge_size = videos_kwargs.get("merge_size") or self.merge_size
        temporal_patch_size = videos_kwargs.get("temporal_patch_size") or self.temporal_patch_size
        if videos_kwargs.get("do_resize", self.do_resize):
            height, width = self._get_frame_size(num_frames, height, width, patch_size * merge_size, size)
        return math.ceil(num_frames / temporal_patch_size), height // patch_size, width // patch_size


class HYV4VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
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
        _, grid_h, grid_w = (int(value) for value in image_inputs["image_grid_thw"][image_idx])
        merge_size = kwargs.get("merge_size") or self.image_processor.merge_size
        return self.image_token * self._num_frame_tokens(grid_h, grid_w, merge_size)

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        grid_t, grid_h, grid_w = (int(value) for value in video_inputs["video_grid_thw"][video_idx])
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
    "HYV4VLImageProcessor",
    "HYV4VLImageProcessorPil",
    "HYV4VLVideoProcessor",
    "HYV4VLProcessor",
]
