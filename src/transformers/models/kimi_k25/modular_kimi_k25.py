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

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...utils.generic import is_flash_attention_requested, maybe_autocast
from ...utils.output_capturing import capture_outputs
from ...video_utils import VideoInput
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
    rope_parameters: dict | None = None
    max_position_embeddings: int | None = None


@auto_docstring(checkpoint="moonshotai/Kimi-K2.6")
@strict
class Kimi_K25Config(PreTrainedConfig):
    r"""
    projection_layer_norm_eps (`float`, *optional*):
        Layer norm epsilon for projector.
    """

    model_type = "kimi_k25"
    sub_configs = {"text_config": AutoConfig, "vision_config": Kimi_K25VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_hidden_size: int | None = 1152
    projection_hidden_act: str = "gelu"
    projection_layer_norm_eps: float = 1e-5
    image_token_id: int = 163605
    video_token_id: int = 163606
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
            torch.randn(config.pos_emb_height, config.pos_emb_width, config.hidden_size)
        )
        time_position_embeddings = self.compute_pos_embed()
        self.register_buffer("time_position_embeddings", time_position_embeddings, persistent=False)

    def compute_pos_embed(self):
        position_ids = torch.arange(self.num_frames, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, dtype=torch.int64).to(dtype=torch.float) / self.dim))
        freqs = torch.outer(position_ids, inv_freq)  # (M, D/2)
        pos_embed = torch.cat([freqs.sin(), freqs.cos()], dim=1)  # (M, D)
        return pos_embed.unsqueeze(1)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thw.tolist():
            if t > self.num_frames:
                raise ValueError(
                    f"Got an input with {t} frames. Number of frames should be less than config.pos_emb_time=({self.num_frames})"
                )

            # Apply learned positions on h/w grids with optional interpolation for bigger images
            if (h, w) == self.position_embeddings.shape[:-1]:
                position_embeddings = self.position_embeddings.flatten(0, 1)
            else:
                position_embeddings = self.position_embeddings.permute(2, 0, 1).unsqueeze(0)
                position_embeddings = F.interpolate(
                    position_embeddings,
                    size=(h, w),
                    mode="bicubic",
                )
                position_embeddings = position_embeddings.squeeze(0).permute(1, 2, 0).flatten(0, 1)

            position_embeddings = position_embeddings.unsqueeze(0).repeat(t, 1, 1)
            # Add RoPE positions for time grid if processing videos
            if t > 1:
                position_embeddings = position_embeddings + self.time_position_embeddings[0:t]

            pos_embs.append(position_embeddings.flatten(0, 1))
        hidden_states = hidden_states + torch.cat(pos_embs, dim=0)
        return hidden_states


class Kimi_K25VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_size = (
            config.patch_size if not isinstance(config.patch_size, int) else (config.patch_size, config.patch_size)
        )
        self.proj = nn.Conv2d(3, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_emb = Kimi_K25VisionPositionEmbeddings(config)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(pixel_values).view(pixel_values.size(0), -1)
        hidden_states = self.pos_emb(hidden_states, grid_thw)
        return hidden_states


# Similarly to gemma4, applies the same freq to H and W grids
# The difference is that gemma4 stcak H/W embeds, while Kimi interleaves them
class Kimi_K25VisionRotaryEmbedding(Gemma4VisionRotaryEmbedding):
    def forward(self, x, position_ids):
        position_ids_expanded = position_ids.permute(1, 2, 0)[..., None].float()  # shape (bs, positions, 2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, None, :]
            .float()
            .expand(position_ids_expanded.shape[0], position_ids_expanded.shape[1], 2, -1)
            .to(x.device)
        )  # shape (bs, positions, 2, freq_dim)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() * position_ids_expanded.float()).transpose(2, 3).flatten(2)
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
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
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

    def temporal_patch_merger(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> list[torch.Tensor]:
        r"""
        Merges temporal frames by spatially pooling patch embeddings across time.

        For each video clip defined by `grid_thw`, the method reshapes the flat patch sequence
        into a `(T, H, W)` grid, averages over the temporal dimension, then rearranges spatial
        patches into groups of `kernel_height * kernel_width` — matching the merged-token layout
        expected by downstream layers.

        Args:
            hidden_states (`torch.Tensor` of shape `(total_patches, hidden_dim)`):
                Concatenated patch embeddings for all clips in the batch. `total_patches` equals
                the sum of `t * h * w` over all entries in `grid_thw`.
            grid_thw (`torch.Tensor` of shape `(batch_size, 3)`):
                Temporal and spatial grid dimensions for each clip, where each row is
                `(num_frames, grid_height, grid_width)`. `grid_height` and `grid_width` must be
                divisible by `kernel_height` and `kernel_width` respectively.

        Returns:
            `torch.Tensor` of shape `(total_merged_patches, kernel_height * kernel_width, hidden_dim)`:
                Temporally pooled patch embeddings. `total_merged_patches` equals the sum of
                `(h // kernel_height) * (w // kernel_width)` over all clips.
        """
        hidden_dim = hidden_states.size(-1)
        kernel_height, kernel_width = self.merge_kernel_size

        outputs = []
        pre_sum = 0
        for t, h, w in grid_thw.tolist():
            # Get the current sequence
            seq = hidden_states[pre_sum : pre_sum + t * h * w]
            # Reshape along self.merge_kernel_size and concat to the last dimension
            new_height, new_width = h // kernel_height, w // kernel_width
            reshaped_seq = seq.view(t, new_height, kernel_height, new_width, kernel_width, hidden_dim)
            reshaped_seq = reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)  # temporal pooling
            padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
            outputs.append(padded_seq)
            pre_sum += t * h * w

        return torch.cat(outputs, dim=0)

    def get_position_ids(self, grid_thw: torch.Tensor) -> torch.Tensor:
        all_position_ids = []
        for t, h, w in grid_thw.tolist():
            h_ids = torch.arange(h, device=grid_thw.device)
            w_ids = torch.arange(w, device=grid_thw.device)

            # (h, w, 2) grid of (row, col) coordinates
            grid = torch.stack(torch.meshgrid(h_ids, w_ids, indexing="xy"), dim=-1)

            # (h*w, 2) -> repeat for each temporal frame in case of videos -> (t*h*w, 2)
            all_position_ids.append(grid.reshape(-1, 2).repeat(t, 1))
        position_ids = torch.cat(all_position_ids, dim=0).unsqueeze(0)
        return position_ids.permute(2, 0, 1)  # (2, batch, seq_len)

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
        hidden_states = self.patch_embed(pixel_values, grid_thw=grid_thw)
        position_ids = self.get_position_ids(grid_thw=grid_thw)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thw.dtype, device=grid_thw.device),
                grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for block in self.layers:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.final_layernorm(hidden_states)
        pooled_hidden_states = self.temporal_patch_merger(hidden_states, grid_thw)

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
        image_embeds = self.mm_projector(vision_outputs.pooler_output)
        vision_outputs.pooler_output = image_embeds
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
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )
        return special_image_mask

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
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
            video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=video_embeds)
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

        >>> model = Kimi_K25ForConditionalGeneration.from_pretrained("TODO")
        >>> processor = AutoProcessor.from_pretrained("TODO")

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
            return_tensors="pt"
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


class Kimi_K25ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


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
        self.video_token_id = self.image_token_id

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[Kimi_K25ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Kimi_K25ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            num_chunks_per_video = videos_inputs.pop("num_chunks_per_video")
            video_grid_thw = []
            start = 0
            for num in num_chunks_per_video:
                video_grid_thw.append(videos_inputs["video_grid_thw"][start : start + num])
                start += num

            # If user has not requested video metadata, pop it
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            temporal_patch_size = self.video_processor.temporal_patch_size
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_chunks = num_chunks_per_video[index]
                    video_structure = ""

                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "SmolVLM requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        )
                    metadata.fps = 24 if metadata.fps is None else metadata.fps

                    for chunk_id in range(num_chunks):
                        current_chunk = metadata.timestamps[
                            (chunk_id * temporal_patch_size) : (chunk_id + 1) * temporal_patch_size
                        ]
                        timestamp = float(current_chunk[0])
                        current_chunk = metadata.timestamps[chunk_id : chunk_id + temporal_patch_size]
                        timestamp_str = (
                            time.strftime("%H:%M:%S", time.gmtime(timestamp)) + f".{int(timestamp % 1 * 1000):03d}"
                        )
                        num_frame_tokens = video_grid_thw[index][chunk_id][1:].prod() // merge_length
                        video_tokens = num_frame_tokens * "<|placeholder|>"  # * len(current_chunk)
                        video_structure += (
                            f"{timestamp_str}<|media_begin|>video<|media_content|>{video_tokens}<|media_end|>"
                        )

                    text[i] = text[i].replace(self.video_token, video_structure, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        raise [name for name in ProcessorMixin.model_input_names if name not in "num_chunks_per_video"]


__all__ = [
    "Kimi_K25Config",
    "Kimi_K25VisionConfig",
    "Kimi_K25ForConditionalGeneration",
    "Kimi_K25Model",
    "Kimi_K25PreTrainedModel",
    "Kimi_K25VisionModel",
    "Kimi_K25Processor",
]
