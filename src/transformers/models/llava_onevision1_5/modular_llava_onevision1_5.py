# Copyright 2025 The LLaVA-OneVision team and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LLaVA-OneVision-1.5 model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging, torch_compilable_check
from ...utils.generic import can_return_tuple, get_max_seqlen, is_flash_attention_requested, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..auto import AutoModel
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
)
from ..qwen2_vl.modeling_qwen2_vl import (
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    VisionAttention,
    VisionRotaryEmbedding,
)
from ..qwen3.modeling_qwen3 import Qwen3Model
from .configuration_llava_onevision1_5 import (
    LlavaOnevision1_5Config,
    LlavaOnevision1_5TextConfig,
    LlavaOnevision1_5VisionConfig,
)


logger = logging.get_logger(__name__)


class LlavaOnevision1_5RiceRotaryEmbedding(VisionRotaryEmbedding):
    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class LlavaOnevision1_5VisionPatchEmbed(nn.Module):
    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [config.patch_size, config.patch_size]
        self.proj = nn.Conv2d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class LlavaOnevision1_5VisionPatchMerger(nn.Module):
    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class LlavaOnevision1_5VisionMlp(nn.Module):
    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LlavaOnevision1_5VisionAttention(VisionAttention):
    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        nn.Module.__init__(self)
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        attn_weights = None

        if is_flash_attention_requested(self.config):
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, attn_weights = attention_interface(
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
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = []
            attn_weights_chunks = []
            for q, k, v in zip(*splits):
                chunk_output, chunk_attn_weights = attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )
                attn_outputs.append(chunk_output)
                if chunk_attn_weights is not None:
                    attn_weights_chunks.append(chunk_attn_weights)
            attn_output = torch.cat(attn_outputs, dim=1)
            if attn_weights_chunks:
                attn_weights = torch.cat(attn_weights_chunks, dim=2)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output, attn_weights


class LlavaOnevision1_5VisionBlock(nn.Module):
    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = LlavaOnevision1_5VisionAttention(config=config)
        self.mlp = LlavaOnevision1_5VisionMlp(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_output, _ = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@auto_docstring
class LlavaOnevision1_5PreTrainedModel(PreTrainedModel):
    config: LlavaOnevision1_5Config
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaOnevision1_5VisionBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.get_text_config().initializer_range
        if isinstance(module, LlavaOnevision1_5VisionModel):
            init.normal_(module.class_embedding, mean=0.0, std=std)
            init.normal_(module.class_pos_emb, mean=0.0, std=std)
        elif isinstance(module, LlavaOnevision1_5RiceRotaryEmbedding):
            # This buffer is non-persistent (never part of the checkpoint) and does not follow the
            # `original_inv_freq`/`rope_type` convention used by the generic `RotaryEmbedding` handling in
            # `PreTrainedModel._init_weights`, so it needs to be recomputed explicitly here. Without this, the
            # meta-device fast-load path used by `from_pretrained` would leave it uninitialized.
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


@auto_docstring
class LlavaOnevision1_5VisionModel(LlavaOnevision1_5PreTrainedModel):
    config: LlavaOnevision1_5VisionConfig
    input_modalities = ("image", "video")
    main_input_name = "pixel_values"
    _no_split_modules = ["LlavaOnevision1_5VisionBlock"]
    _can_record_outputs = {
        "hidden_states": LlavaOnevision1_5VisionBlock,
        "attentions": LlavaOnevision1_5VisionAttention,
    }

    def __init__(self, config: LlavaOnevision1_5VisionConfig) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = LlavaOnevision1_5VisionPatchEmbed(config)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = LlavaOnevision1_5RiceRotaryEmbedding(head_dim // 2)

        scale = config.hidden_size**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(config.hidden_size))
        self.class_pos_emb = nn.Parameter(torch.randn(1, head_dim // 2))

        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.blocks = nn.ModuleList([LlavaOnevision1_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = LlavaOnevision1_5VisionPatchMerger(config)

        self.gradient_checkpointing = False
        self.post_init()

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            t = int(t.item())
            h = int(h.item())
            w = int(w.item())
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = int(grid_thw[:, 1:].max().item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
        """
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        segment_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).to(torch.long)
        expected_total_patches = int(segment_seqlens.sum().item())
        actual_total_patches = hidden_states.shape[0]

        compressed_inputs = expected_total_patches != actual_total_patches
        if compressed_inputs:
            if expected_total_patches % actual_total_patches != 0:
                raise ValueError(
                    "Image features and image tokens do not match after vision patch embedding, "
                    f"expected {expected_total_patches} patches from image_grid_thw but got {actual_total_patches}."
                )

            compression_factor = expected_total_patches // actual_total_patches
            if torch.any(segment_seqlens % compression_factor != 0):
                raise ValueError(
                    "image_grid_thw patch segments are incompatible with compressed vision features: "
                    f"segment lengths {segment_seqlens.tolist()} cannot be divided by compression factor {compression_factor}."
                )

            compressed_seqlens = segment_seqlens // compression_factor
            compressed_rotary_pos_emb = []
            rotary_start = 0
            for raw_seg_len, compressed_seg_len in zip(segment_seqlens.tolist(), compressed_seqlens.tolist()):
                rotary_end = rotary_start + raw_seg_len
                compressed_rotary_pos_emb.append(rotary_pos_emb[rotary_start:rotary_end:compression_factor][:compressed_seg_len])
                rotary_start = rotary_end
            rotary_pos_emb = torch.cat(compressed_rotary_pos_emb, dim=0)
            segment_seqlens = compressed_seqlens

        cu_seqlens = segment_seqlens.cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu = cu_seqlens.to(torch.long)
        num_segments = cu.numel() - 1
        cls_token = self.class_embedding.to(hidden_states.dtype).unsqueeze(0)

        total_patches = cu[-1].item()
        img_feats = hidden_states.shape[0]
        new_total = total_patches + num_segments
        embed_dim = hidden_states.size(-1)
        new_hidden = hidden_states.new_empty((new_total, embed_dim))
        new_rotary_pos_emb = rotary_pos_emb.new_empty((new_total, rotary_pos_emb.shape[-1]))

        write_ptr = 0
        new_cu = [0]
        for i in range(1, num_segments + 1):
            seg_start = cu[i - 1].item()
            seg_end = cu[i].item()
            seg_len = seg_end - seg_start
            new_hidden[write_ptr] = cls_token
            new_rotary_pos_emb[write_ptr] = self.class_pos_emb
            new_hidden[write_ptr + 1 : write_ptr + 1 + seg_len] = hidden_states[seg_start:seg_end]
            new_rotary_pos_emb[write_ptr + 1 : write_ptr + 1 + seg_len] = rotary_pos_emb[seg_start:seg_end]
            write_ptr += 1 + seg_len
            new_cu.append(write_ptr)

        hidden_states = new_hidden
        cu_seqlens = torch.tensor(new_cu, device=hidden_states.device, dtype=torch.int32)
        rotary_pos_emb = new_rotary_pos_emb

        hidden_states = self.pre_layernorm(hidden_states)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        new_hidden = hidden_states.new_empty((img_feats, embed_dim))
        for i in range(1, num_segments + 1):
            seg_start = cu[i - 1].item()
            seg_end = cu[i].item()
            new_seg_start = new_cu[i - 1]
            new_seg_end = new_cu[i]
            new_hidden[seg_start:seg_end] = hidden_states[new_seg_start + 1 : new_seg_end]
        hidden_states = new_hidden
        merge_block = self.spatial_merge_size**2
        if (not compressed_inputs) and hidden_states.shape[0] % merge_block == 0:
            merged_hidden_states = self.merger(hidden_states)
        else:
            merged_hidden_states = hidden_states
        return BaseModelOutputWithPooling(
            last_hidden_state=merged_hidden_states,
            pooler_output=merged_hidden_states,
        )


@auto_docstring
class LlavaOnevision1_5TextPreTrainedModel(PreTrainedModel):
    config: LlavaOnevision1_5TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True


@auto_docstring
class LlavaOnevision1_5TextModel(Qwen3Model):
    config: LlavaOnevision1_5TextConfig
    _can_record_outputs = {
        "hidden_states": LlavaOnevision1_5TextDecoderLayer,
        "attentions": LlavaOnevision1_5TextAttention,
    }


@auto_docstring(
    custom_intro="""
    Base class for LLaVA-OneVision-1.5 outputs, with hidden states and attentions.
    """
)
@dataclass
class LlavaOnevision1_5ModelOutputWithPast(LlavaModelOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    Base class for LLaVA-OneVision-1.5 causal language model (or autoregressive) outputs.
    """
)
@dataclass
class LlavaOnevision1_5CausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    pass


@auto_docstring(
    custom_intro="""
    The LLaVA-OneVision-1.5 model which consists of a RICE vision backbone and a Qwen3 language model,
    without a language modeling head.
    """
)
class LlavaOnevision1_5Model(LlavaModel):
    def __init__(self, config: LlavaOnevision1_5Config):
        PreTrainedModel.__init__(self, config)
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, num_channels * patch_size * patch_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_embeds = vision_outputs.pooler_output
        return BaseModelOutputWithPooling(
            last_hidden_state=image_embeds,
            pooler_output=image_embeds,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @merge_with_config_defaults
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains video last hidden states from the vision tower and apply multimodal projection."
    )
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, num_channels * temporal_patch_size * patch_size * patch_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
                The temporal, height and width of feature shape of each video in LLM.
        """
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw, return_dict=True, **kwargs)
        video_embeds = vision_outputs.pooler_output
        return BaseModelOutputWithPooling(
            last_hidden_state=video_embeds,
            pooler_output=video_embeds,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
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
                n_image_tokens == image_features.shape[0],
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, "
                f"features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                n_video_tokens == video_features.shape[0],
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, "
                f"features: {video_features.shape[0]}",
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
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,  # noqa: ARG002 (accepted for processor compatibility)
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LlavaOnevision1_5ModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        video_features = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, **kwargs)
            image_features = image_outputs.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

            if input_ids is not None:
                n_image_tokens = int((input_ids == self.config.image_token_id).sum().item())
                if image_features.shape[0] != n_image_tokens and n_image_tokens % image_features.shape[0] == 0:
                    image_features = image_features.repeat_interleave(n_image_tokens // image_features.shape[0], dim=0)

            special_image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)
            video_features = video_outputs.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)

            if input_ids is not None:
                n_video_tokens = int((input_ids == self.config.video_token_id).sum().item())
                if video_features.shape[0] != n_video_tokens and n_video_tokens % video_features.shape[0] == 0:
                    video_features = video_features.repeat_interleave(n_video_tokens // video_features.shape[0], dim=0)

            _, special_video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return LlavaOnevision1_5ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


@auto_docstring(
    custom_intro="""
    The LLAVA-OneVision-1.5 model which consists of a RICE vision backbone and a Qwen3 language model.
    """
)
class LlavaOnevision1_5ForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config: LlavaOnevision1_5Config):
        PreTrainedModel.__init__(self, config)
        self.model = LlavaOnevision1_5Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width of feature shape of each image in LLM.
        """
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width of feature shape of each video in LLM.
        """
        return self.model.get_video_features(
            pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,  # noqa: ARG002 (accepted for processor compatibility)
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LlavaOnevision1_5CausalLMOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
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

        return LlavaOnevision1_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


__all__ = [
    "LlavaOnevision1_5PreTrainedModel",
    "LlavaOnevision1_5TextPreTrainedModel",
    "LlavaOnevision1_5VisionModel",
    "LlavaOnevision1_5TextModel",
    "LlavaOnevision1_5Model",
    "LlavaOnevision1_5ForConditionalGeneration",
]
