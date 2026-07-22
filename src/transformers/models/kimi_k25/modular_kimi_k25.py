# Copyright 2026 the HuggingFace Team. All rights reserved.
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
import time
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...utils.generic import get_max_seqlen, is_flash_attention_requested, maybe_autocast
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_cu_seqlens, get_vision_position_ids
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding
from ..glm4v.modeling_glm4v import Glm4vForConditionalGeneration
from ..llava.modeling_llava import LlavaCausalLMOutputWithPast, LlavaModelOutputWithPast
from ..qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLPreTrainedModel,
    Qwen2VLVisionBlock,
    VisionAttention,
    VisionMlp,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor


logger = logging.get_logger(__name__)


def get_vision_bicubic_indices_and_weights(
    grid_thw: torch.Tensor, num_grid_per_side: int, kwargs: dict | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-patch 16-tap bicubic gather indices/weights for resampling a square learned
    `(num_grid_per_side, num_grid_per_side)` position-embedding table to each image's `(h, w)`,
    or pop `"bicubic_indices"`/`"bicubic_weights"` from `kwargs`.

    Reproduces `F.interpolate(mode="bicubic", align_corners=False)` (Keys cubic kernel, `a=-0.75`)
    as `(total_patches, 16)` indices + weights, consumed by a single fused `F.embedding_bag`. Fully
    vectorised over packed patches (ragged `(h, w)` handled with `repeat_interleave`, no per-image
    loop), so it traces and supports dynamic shapes like the other grid_thw precompute helpers.
    """
    if kwargs is not None:
        bicubic_indices = kwargs.pop("bicubic_indices", None)
        bicubic_weights = kwargs.pop("bicubic_weights", None)
        if bicubic_indices is not None and bicubic_weights is not None:
            return bicubic_indices, bicubic_weights

    a = -0.75
    side = num_grid_per_side
    device = grid_thw.device
    offsets = torch.arange(-1, 3, device=device)  # the 4 bicubic taps: floor-1 .. floor+2

    def cubic_weights(distance):
        # Keys convolution kernel (a=-0.75): near lobe for |d| <= 1, far lobe for 1 < |d| < 2.
        near = ((a + 2) * distance - (a + 3)) * distance * distance + 1
        far = ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a
        return torch.where(distance <= 1, near, far)

    def axis_taps_weights(index, size):
        src = (index + 0.5) * side / size - 0.5  # source coordinate, align_corners=False
        floor = torch.floor(src)
        taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)  # (total, 4)
        return taps, cubic_weights((src[:, None] - floor[:, None] - offsets).abs())

    # Per-patch (row, col) within its image, derived from packed offsets — no per-image loop.
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(F.pad(counts.cumsum(0)[:-1], (1, 0)), counts)
    within = (torch.arange(counts.sum(), device=device) - starts) % (heights * widths)
    h_taps, h_weights = axis_taps_weights(within // widths, heights)
    w_taps, w_weights = axis_taps_weights(within % widths, widths)
    # 2D separable: outer of the 4 h-taps × 4 w-taps → 16 taps per patch.
    bicubic_indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 16)
    bicubic_weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 16)
    return bicubic_indices, bicubic_weights


def get_vision_frame_index(grid_thw: torch.Tensor, kwargs: dict | None = None) -> torch.Tensor:
    """Per-patch index into a temporal embedding table whose row `0` is a zero pad, or pop
    `"frame_index"` from `kwargs`.

    Single-frame clips (`t == 1`, images) map every patch to `0` (no temporal term); frame `f` of a
    multi-frame clip maps to `f + 1`. Precomputable, so the encoder avoids a per-clip `if t > 1` loop.
    """
    if kwargs is not None and (frame_index := kwargs.pop("frame_index", None)) is not None:
        return frame_index
    device = grid_thw.device
    parts = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        # t == 1 → [0] (padded row 0 = zero); t > 1 → [1..t] → time_emb[0..t-1]
        frames = torch.arange(t, device=device) + int(t > 1)
        parts.append(frames.repeat_interleave(h * w))
    return torch.cat(parts)


def get_vision_temporal_merge_index(
    grid_thw: torch.Tensor, kernel_height: int, kernel_width: int, kwargs: dict | None = None
) -> torch.Tensor:
    """Gather index regrouping a flat patch sequence into `(total_merged, t, kernel_height *
    kernel_width)` for the temporal-pooling merger, or pop `"temporal_merge_index"` from `kwargs`.

    Row `m` collects the `t` frames × `kernel_height*kernel_width` source patches that pool into
    merged token `m`; the caller means over the frame axis. Precomputable, so the encoder avoids a
    per-clip `grid_thw.tolist()` loop.
    """
    if kwargs is not None and (index := kwargs.pop("temporal_merge_index", None)) is not None:
        return index
    device = grid_thw.device
    running, rows = 0, []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        new_h, new_w = h // kernel_height, w // kernel_width
        base = torch.arange(running, running + t * h * w, device=device).view(
            t, new_h, kernel_height, new_w, kernel_width
        )
        # (t, new_h, new_w, kh, kw) → (new_h*new_w, t, kh*kw): frame axis kept for the caller's mean.
        base = base.permute(1, 3, 0, 2, 4).reshape(new_h * new_w, t, kernel_height * kernel_width)
        rows.append(base)
        running += t * h * w
    return torch.cat(rows, dim=0)


@auto_docstring(checkpoint="moonshotai/Kimi-K2.6")
@strict
class Kimi_K25VisionConfig(PreTrainedConfig):
    r"""
    pos_emb_height (`int`, *optional*):
        Initial position embedding height.
    pos_emb_width (`int`, *optional*):
        Initial position embedding width.
    pos_emb_time (`int`, *optional*):
        Initial position embedding time dimension.
    merge_kernel_size (`tuple[int] | list[int]`, *optional*):
        Kernel size for patch merging.
    """

    model_type = "kimi_k25_vision"

    patch_size: int = 14
    pos_emb_height: int = 64
    pos_emb_width: int = 64
    pos_emb_time: int = 4
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    hidden_act: str = "gelu_pytorch_tanh"
    merge_kernel_size: tuple[int, int] | list[int] = (2, 2)
    rope_parameters: dict | None = None  # defaults set by `RopeConfigMixin`
    max_position_embeddings: int | None = None


@auto_docstring(checkpoint="moonshotai/Kimi-K2.6")
@strict
class Kimi_K25Config(PreTrainedConfig):
    r"""
    projection_hidden_size (`int`, *optional*, defaults to `1152`):
        The output hidden size for multimodal projector.
    projection_layer_norm_eps (`float`, *optional*, defaults to `1e-5`):
        Layer norm epsilon for projector.
    """

    model_type = "kimi_k25"
    sub_configs = {"text_config": AutoConfig, "vision_config": Kimi_K25VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_hidden_size: int | None = 1152
    projection_layer_norm_eps: float = 1e-5
    image_token_id: int = 163605
    video_token_id: int = 163840
    vision_start_token_id: int = 163602
    vision_end_token_id: int = 163604
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # BC: load from remote config on the hub where the model-type points to remote config
        if isinstance(self.text_config, dict):
            model_type = self.text_config.get("model_type", "deepseek_v3")
            if model_type == "kimi_k2":
                model_type = "deepseek_v3"
            self.text_config = CONFIG_MAPPING[model_type](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["deepseek_v3"]()
        else:
            model_type = self.text_config.model_type
            if model_type == "kimi_k2":
                self.text_config.model_type = "deepseek_v3"

        if isinstance(self.vision_config, dict):
            self.vision_config = Kimi_K25VisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Kimi_K25VisionConfig()
        super().__post_init__(**kwargs)


class Kimi_K25ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


class Kimi_K25CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


class Kimi_K25VisionPositionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_frames = config.pos_emb_time

        self.position_embeddings = nn.Parameter(
            torch.zeros(config.pos_emb_height, config.pos_emb_width, config.hidden_size)
        )
        # Side of the (square) learned grid; the exporter's input preparer reads it to precompute
        # the bicubic gather indices.
        self.num_grid_per_side = config.pos_emb_height

        # Time-axis pos_emb are an additive sinusoidal table, i.e. add pos to hiddens rather than rotating
        time_position_embeddings = self.compute_pos_embed()
        self.register_buffer("time_position_embeddings", time_position_embeddings, persistent=False)

    def compute_pos_embed(self):
        position_ids = torch.arange(self.num_frames, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, dtype=torch.int64).to(dtype=torch.float) / self.dim))
        freqs = torch.outer(position_ids, inv_freq)  # (M, D/2)
        pos_embed = torch.cat([freqs.sin(), freqs.cos()], dim=1)  # (M, D)
        # Prepend a zero row so frame index 0 (single-frame clips) adds no temporal offset.
        return torch.cat([pos_embed.new_zeros(1, self.dim), pos_embed])  # (M+1, D)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        # Spatial: bicubically resample the learned grid to each image's (h, w) as a fused weighted
        # gather (`embedding_bag`), equivalent to a per-image `F.interpolate(mode="bicubic")` but a
        # single traceable op over all patches — and faster.
        table = self.position_embeddings.flatten(0, 1)
        bicubic_indices, bicubic_weights = get_vision_bicubic_indices_and_weights(
            grid_thw, self.num_grid_per_side, kwargs=kwargs
        )
        pos = F.embedding_bag(bicubic_indices, table, per_sample_weights=bicubic_weights.to(table.dtype), mode="sum")
        # Temporal: add a per-frame sinusoid. Row 0 of the table is a zero pad, so single-frame clips
        # (frame index 0) get none.
        pos = pos + self.time_position_embeddings[get_vision_frame_index(grid_thw, kwargs=kwargs)]
        return hidden_states + pos.to(hidden_states.dtype)


class Kimi_K25VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = (
            config.patch_size if not isinstance(config.patch_size, int) else (config.patch_size, config.patch_size)
        )
        self.proj = nn.Conv2d(3, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = Kimi_K25VisionPositionEmbeddings(config)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.proj(pixel_values).view(pixel_values.size(0), -1)
        hidden_states = self.pos_emb(hidden_states, grid_thw, **kwargs)
        return hidden_states


# Similarly to gemma4, applies the same freq to H and W grids
# The difference is that gemma4 stacks H/W embeds on `dim`, while Kimi interleaves them
class Kimi_K25VisionRotaryEmbedding(Gemma4VisionRotaryEmbedding):
    def forward(self, x, position_ids):
        position_ids_expanded = position_ids.transpose(0, 1)[..., None].float()  # (positions, 2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :].float().expand(position_ids_expanded.shape[0], 2, -1).to(x.device)
        )  # (positions, 2, freq_dim)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(1, 2).flatten(1)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos, sin


class Kimi_K25VisionMLP(VisionMlp):
    pass


# Difference from Qwen: unfused qkv as we chunk and permute qk proj when converting!
class Kimi_K25VisionAttention(VisionAttention):
    def __init__(self, config: Kimi_K25VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=True)
        del self.qkv

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(2, 1)
        key_states = key_states.transpose(2, 1)
        value_states = value_states.transpose(2, 1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


# Don't copy `init` from Qwen-VL due to non-standard config naming in Qwen
class Kimi_K25VisionEncoderLayer(Qwen2VLVisionBlock):
    def __init__(self, config):
        nn.Module.__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = Kimi_K25VisionAttention(config=config)
        self.mlp = Kimi_K25VisionMLP(config.hidden_size, config.intermediate_size, config.hidden_act)


class Kimi_K25PreTrainedModel(Qwen2VLPreTrainedModel):
    _no_split_modules = ["Kimi_K25VisionEncoderLayer"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, Kimi_K25VisionPositionEmbeddings):
            buffer_value = module.compute_pos_embed()
            init.copy_(module.time_position_embeddings, buffer_value)
            init.trunc_normal_(module.position_embeddings, mean=0.0)


class Kimi_K25VisionModel(Kimi_K25PreTrainedModel):
    config: Kimi_K25VisionConfig
    input_modalities = ("image", "video")
    _can_record_outputs = {
        "hidden_states": Kimi_K25VisionEncoderLayer,
        "attentions": Kimi_K25VisionAttention,
    }

    def __init__(self, config: Kimi_K25VisionConfig):
        super().__init__(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_embed = Kimi_K25VisionPatchEmbed(config)

        self.rotary_emb = Kimi_K25VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList([Kimi_K25VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-05)
        self.post_init()

    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        hidden_states = self.patch_embed(pixel_values, grid_thw=grid_thw, **kwargs)
        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1, kwargs=kwargs)
        position_ids = position_ids.transpose(0, 1).flip(0)  # (2, positions)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens = get_vision_cu_seqlens(grid_thw, merge_temporal=True, kwargs=kwargs)
        max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs=kwargs)

        for block in self.layers:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.final_layernorm(hidden_states)
        merge_index = get_vision_temporal_merge_index(grid_thw, *self.merge_kernel_size, kwargs=kwargs)
        pooled_hidden_states = hidden_states[merge_index].mean(dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_hidden_states,
        )


class Kimi_K25MultimodalProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.vision_config.hidden_size * (
            config.vision_config.merge_kernel_size[0] * config.vision_config.merge_kernel_size[1]
        )
        self.pre_norm = nn.LayerNorm(config.projection_hidden_size, eps=config.projection_layer_norm_eps)

        self.in_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(self.hidden_size, config.text_config.hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = self.pre_norm(hidden_states).view(batch_size, -1, self.hidden_size)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class Kimi_K25Model(Kimi_K25PreTrainedModel):
    def __init__(self, config: Kimi_K25Config):
        super().__init__(config)
        self.vision_tower = Kimi_K25VisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.mm_projector = Kimi_K25MultimodalProjection(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        vision_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, **kwargs)
        image_embeds = self.mm_projector(vision_outputs.pooler_output).squeeze(1)
        merge_kernel_size = self.vision_tower.merge_kernel_size[0] * self.vision_tower.merge_kernel_size[1]
        split_sizes = (image_grid_thw.prod(-1) // merge_kernel_size).tolist()
        vision_outputs.pooler_output = torch.split(image_embeds, split_sizes)
        return vision_outputs

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input videos.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                n_video_tokens * inputs_embeds.shape[-1] == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Kimi_K25ModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """

        if inputs_embeds is None:
            multimodal_mask = (input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = 0
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return Kimi_K25ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Kimi_K25ForConditionalGeneration(Glm4vForConditionalGeneration):
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
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Kimi_K25CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Kimi_K25ForConditionalGeneration

        >>> model = Kimi_K25ForConditionalGeneration.from_pretrained("moonshotai/Kimi-K2.6")
        >>> processor = AutoProcessor.from_pretrained("moonshotai/Kimi-K2.6")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """

        outputs: Kimi_K25ModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Kimi_K25CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_position_ids_for_generation(self, **kwargs):
        raise AttributeError("Kimi doesn't use m-rope!")

    def _get_image_nums_and_video_nums(self, **super_kwargs):
        raise AttributeError()

    def _expand_inputs_for_generation(self, **super_kwargs):
        raise AttributeError("Uses normal super call")


class Kimi_K25ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class Kimi_K25Processor(Qwen2VLProcessor):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        ProcessorMixin.__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.image_token = "<|media_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = (
            "<|kimi_k25_video_placeholder|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        merge_length = self.video_processor.merge_size**2
        temporal_patch_size = self.video_processor.temporal_patch_size

        num_chunks = video_inputs["num_chunks_per_video"][video_idx]
        start = 0 if video_idx == 0 else np.cumsum(video_inputs["num_chunks_per_video"])[video_idx - 1]
        video_grid_thw = video_inputs["video_grid_thw"][start : start + num_chunks]
        video_structure = ""

        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.fps is None:
            logger.warning_once(
                "SmolVLM requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
            )
            metadata.fps = 24

        for chunk_id in range(num_chunks):
            current_chunk = metadata.timestamps[
                (chunk_id * temporal_patch_size) : (chunk_id + 1) * temporal_patch_size
            ]
            timestamp = float(current_chunk[0])
            current_chunk = metadata.timestamps[chunk_id : chunk_id + temporal_patch_size]
            timestamp_str = time.strftime("%H:%M:%S", time.gmtime(timestamp)) + f".{int(timestamp % 1 * 1000):03d}"
            num_frame_tokens = video_grid_thw[chunk_id][1:].prod() // merge_length
            video_tokens = num_frame_tokens * self.video_token
            video_structure += f"{timestamp_str}<|media_begin|>video<|media_content|>{video_tokens}<|media_end|>"
        return video_structure

    @property
    def model_input_names(self) -> list[str]:
        model_input_names = []
        for attribute_name in self.get_attributes():
            attribute = getattr(self, attribute_name, None)
            if attribute is not None:
                attr_input_names = getattr(attribute, "model_input_names")
                model_input_names.extend(attr_input_names)
        return [name for name in model_input_names if name not in self.unused_input_names]

    @property
    def unused_input_names(self):
        return ["num_chunks_per_video"]


__all__ = [
    "Kimi_K25Config",
    "Kimi_K25VisionConfig",
    "Kimi_K25ForConditionalGeneration",
    "Kimi_K25Model",
    "Kimi_K25PreTrainedModel",
    "Kimi_K25VisionModel",
    "Kimi_K25Processor",
]
