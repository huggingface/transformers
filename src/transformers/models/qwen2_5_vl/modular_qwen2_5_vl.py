# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2.5-VL model."""

import itertools
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ProcessingKwargs, Unpack
from ...utils import auto_docstring, logging
from ...utils.generic import get_max_seqlen, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...video_utils import VideoInput
from ...vision_utils import (
    get_vision_attention_seqlens,
    get_vision_position_ids,
    get_vision_window_index,
)
from ..llama.modeling_llama import LlamaRMSNorm
from ..qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLTextConfig
from ..qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    PatchMerger,
    Qwen2VLCausalLMOutputWithPast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLModelOutputWithPast,
    Qwen2VLPreTrainedModel,
    TransformersKwargs,
    VisionAttention,
    VisionRotaryEmbedding,
)
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor


logger = logging.get_logger(__name__)


def get_mrope_position_ids(
    config,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    attention_mask: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """M-RoPE decoder positions for a sequence of interleaved text and vision runs.

    Per batch row, over the unpadded tokens, walk `mm_token_type_ids` span by span (0=text, 1=image,
    2=video): a text run counts 1D positions on all three axes, and a vision span lays a
    `(temporal, height, width)` grid then advances the text position by its largest spatial extent. A
    video's temporal axis is scaled by this family's clock, `tokens_per_second * second_per_grid_ts`.
    Returns `(position_ids, rope_deltas)`, the deltas being what `generate` advances decode positions by.
    """
    vision_config = config.vision_config
    spatial_merge_size = vision_config.spatial_merge_size
    tokens_per_second = vision_config.tokens_per_second
    grids = {1: image_grid_thw, 2: video_grid_thw}

    position_ids = torch.full(
        (3, input_ids.shape[0], input_ids.shape[1]), 0, dtype=input_ids.dtype, device=input_ids.device
    )
    rope_deltas = []
    for batch_idx, token_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        valid_tokens = None
        if attention_mask is not None:
            valid_tokens = attention_mask[batch_idx].bool()
            token_ids, input_token_type = token_ids[valid_tokens], input_token_type[valid_tokens]

        counter, current_position, blocks = defaultdict(int), 0, []
        for modality_type, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            length = len(list(group))
            # 0 is a text run; 1 and 2 are image and video spans.
            if modality_type == 0:
                positions = torch.arange(current_position, current_position + length, device=token_ids.device)
                blocks.append(positions.expand(3, length))
                current_position += length
                continue

            grid_thw = grids[modality_type][counter[modality_type]]
            counter[modality_type] += 1
            grid_t = grid_thw[0].item()
            grid_h, grid_w = grid_thw[1].item() // spatial_merge_size, grid_thw[2].item() // spatial_merge_size
            time_interval = 1
            if modality_type == 2 and second_per_grid_ts is not None:
                time_interval = tokens_per_second * int(second_per_grid_ts[counter[modality_type] - 1])
            temporal = torch.arange(grid_t, device=token_ids.device) * time_interval + current_position
            height = torch.arange(grid_h, device=token_ids.device) + current_position
            width = torch.arange(grid_w, device=token_ids.device) + current_position
            axes = torch.meshgrid(temporal, height, width, indexing="ij")
            blocks.append(torch.stack(axes, dim=0).reshape(3, -1))
            current_position += int(max(grid_thw[1], grid_thw[2])) // spatial_merge_size

        positions = torch.cat(blocks, dim=1).to(device=position_ids.device, dtype=position_ids.dtype)
        if valid_tokens is not None:
            position_ids[:, batch_idx, valid_tokens] = positions
        else:
            position_ids[:, batch_idx] = positions
        rope_deltas.append(positions.max() + 1 - len(token_ids))
    return position_ids, torch.tensor(rope_deltas, device=input_ids.device).unsqueeze(1)


@auto_docstring(checkpoint="Qwen/Qwen2-VL-7B-Instruct")
@strict
class Qwen2_5_VLVisionConfig(PreTrainedConfig):
    r"""
    tokens_per_second (`int`, *optional*, defaults to 41):
        Number of tokens to merge for each second of video.
    window_size (`int`, *optional*, defaults to 11):
        Size of windows.
    out_hidden_size (`int`, *optional*, defaults to 3584):
        The output hidden size of the vision model.
    fullatt_block_indexes (`int`, *optional*, defaults to `[7, 15, 23, 31]`):
        Indices of layers with full attention
    """

    model_type = "qwen2_5_vl_vision"
    base_config_key = "vision_config"

    depth: int = 32
    hidden_size: int = 3584
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    tokens_per_second: int = 4
    window_size: int = 112
    out_hidden_size: int = 3584
    fullatt_block_indexes: list[int] | tuple[int, ...] = (7, 15, 23, 31)
    initializer_range: float = 0.02


class Qwen2_5_VLTextConfig(Qwen2VLTextConfig):
    pass


class Qwen2_5_VLConfig(Qwen2VLConfig):
    pass


class Qwen2_5_VLRMSNorm(LlamaRMSNorm):
    pass


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VisionPatchEmbed(PatchEmbed):
    pass


class Qwen2_5_VisionRotaryEmbedding(VisionRotaryEmbedding):
    pass


class Qwen2_5_VLPatchMerger(PatchMerger):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__(dim, context_dim, spatial_merge_size)
        self.ln_q = Qwen2_5_VLRMSNorm(context_dim, eps=1e-6)


class Qwen2_5_VLVisionAttention(VisionAttention):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__(config)
        self.dim = config.hidden_size


class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2_5_VLRMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2_5_VLRMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor`):
            Cumulative sequence lengths used for packed variable-length attention in Flash Attention kernels.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VLPreTrainedModel(Qwen2VLPreTrainedModel):
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Qwen2_5_VisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]
    _input_embed_layer = "patch_embed"
    _can_record_outputs = {
        "hidden_states": Qwen2_5_VLVisionBlock,
        "attentions": Qwen2_5_VLVisionAttention,
    }

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

        self.post_init()

    def rot_pos_emb(self, grid_thw):
        warnings.warn(
            f"`{self.__class__.__name__}.rot_pos_emb` is deprecated and will be removed in v5.11. Use `get_vision_position_ids` from `transformers.vision_utils` and apply the rotary embedding module.",
            FutureWarning,
            stacklevel=2,
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        warnings.warn(
            f"`{self.__class__.__name__}.get_window_index` is deprecated and will be removed in v5.11. Use `get_vision_window_index` from `transformers.vision_utils` instead.",
            FutureWarning,
            stacklevel=2,
        )
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            window_size=self.window_size,
            patch_size=self.patch_size,
        )
        return window_index, cu_window_seqlens.tolist()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            window_size=self.window_size,
            patch_size=self.patch_size,
            kwargs=kwargs,
        )
        max_window_seqlen = get_max_seqlen(
            cu_window_seqlens, self.config, kwargs=kwargs, kwarg_name="max_window_seqlen"
        )

        hidden_states = self.patch_embed(hidden_states)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_window_seqlen

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                max_seqlen=max_seqlen_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        reverse_indices = torch.argsort(window_index)
        merged_hidden_states = self.merger(hidden_states)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


class Qwen2_5_VLModelOutputWithPast(Qwen2VLModelOutputWithPast):
    pass


class Qwen2_5_VLModel(Qwen2VLModel):
    config: Qwen2_5_VLConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's sizes. The utility expects a `vision + text`
        sequence and will error out otherwise. For pure text sequence, please rely on model's auto-inferred
        position ids. In a mixed vision + text sequence, vision tokens use 3D RoPE (temporal, height, width)
        while text tokens use standard 1D RoPE.

        Example:
            Temporal patches: 3; Height patches: 2; Width patches: 2
            Each vision input results in (temporal x height × width) positions. Here: 3 x 2 × 2 = 12 positions total.

            Temporal position IDs are spaced by:
                `interval = tokens_per_second * temporal_patch_size / fps`

                If fps = 1; tokens_per_second = 25; temporal_patch_size = 2, temporal IDs increase by 50 for each temporal patch:
                `[0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]`

            Height IDs repeat per row: `[0, 0, 1, 1, ...]`
            Width IDs alternate per column: `[0, 1, 0, 1, ...]`
            Text tokens follow standard 1D RoPE and the position IDs grow consequently with a step of `1`

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        return get_mrope_position_ids(
            self.config,
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            second_per_grid_ts=second_per_grid_ts,
        )

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: torch.Tensor | None,
        second_per_grid_ts: torch.Tensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        # Kept rather than inherited: this model falls back to 1D positions when `mm_token_type_ids` is
        # missing, where the mixin raises. Callers pass grids without it (`test_video_forward`), so
        # inheriting would be a breaking change for them.
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        can_compute_mrope = (
            input_ids is not None
            and mm_token_type_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        )

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                second_per_grid_ts=second_per_grid_ts,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        # Use pre-calculated rope-deltas to infer correct 3D position ids during incremental
        # generation (past_key_values_length > 0) or when only inputs_embeds is provided (no input_ids
        # to recompute from). Skip when input_ids is provided without past_key_values to avoid shape
        # mismatches from stale rope_deltas (e.g., training forward pass after generation).
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=position_ids.device)
        else:
            # Can't build correct 3D positions. Let the model infer it
            position_ids = None
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
        second_per_grid_ts: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2_5_VLModelOutputWithPast:
        r"""
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, **kwargs).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, **kwargs).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen2_5_VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    pass


class Qwen2_5_VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

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
        second_per_grid_ts: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen2_5_VLCausalLMOutputWithPast:
        r"""
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

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

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


class Qwen2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Qwen2_5_VLProcessor(Qwen2VLProcessor):
    def _process_videos(self, videos: VideoInput, **kwargs):
        processed_data, video_replacements = super()._process_videos(videos, **kwargs)
        video_grid_thw = processed_data["video_grid_thw"]

        video_metadata = processed_data["video_metadata"]
        fps = [metadata.sampled_fps for metadata in video_metadata]

        if isinstance(fps, (int, float)):
            second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
        elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
            second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
        else:
            raise ValueError(
                f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
            )
        processed_data["second_per_grid_ts"] = second_per_grid_ts
        return processed_data, video_replacements

    @property
    def model_input_names(self):
        return super().model_input_names + ["second_per_grid_ts", "mm_token_type_ids"]


__all__ = [
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLVisionConfig",
    "Qwen2_5_VLTextConfig",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel",
    "Qwen2_5_VLProcessor",
    "Qwen2_5_VLTextModel",  # noqa: F822
]
