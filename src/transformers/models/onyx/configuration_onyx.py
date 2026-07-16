# coding=utf-8
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
from __future__ import annotations

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring
class OnyxConfig(PreTrainedConfig):
    model_type = "onyx"

    def __init__(
        self,
        hidden_size: int = 6656,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        intermediate_size: int = 19968,
        vocab_size: int = 202048,
        rms_norm_eps: float = 1e-5,
        post_norm_eps: float = 1e-8,
        rope_theta: float = 500_000.0,
        max_position_embeddings: int = 16384,
        use_qk_norm: bool = True,
        qk_scale_factor: float = 43.7840518911,
        use_attn_output_gate: bool = True,
        output_multiplier: float = 0.19611613513818404,
        output_soft_cap_temp: float | None = 20.0,
        normalize_tok_embeddings: bool = True,
        sliding_window: int = 2048,
        sliding_window_pattern: list[int] | None = None,
        every_n_layers_nope: int = 4,
        no_rope_layers: list[int] | None = None,
        layer_types: list[str] | None = None,
        vision_latent_dim: int = 1536,
        vision_output_dim: int = 6144,
        vision_layers: int = 50,
        vision_heads: int = 16,
        vision_mlp_ratio: float = 8960 / 1536,
        vision_patch_size: int = 14,
        vision_patch_temporal: int = 2,
        vision_downsample_factor: int = 2,
        vision_sparse_attention_factor: int = 4,
        vision_pos_emb_grid_h: int = 32,
        vision_pos_emb_grid_w: int = 32,
        vision_adapter_dim: int = 4096,
        patch_token_id: int = 200092,
        video_token_id: int = 200091,
        vid_start_id: int = 200082,
        vid_end_id: int = 200083,
        vid_frame_sep_id: int = 200087,
        video_num_frames: int = 96,
        video_sampling_fps: float = 2.0,
        has_vision: bool = True,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = False,
        bos_token_id: int = 200000,
        eos_token_id: int = 200001,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.post_norm_eps = post_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = use_qk_norm
        self.qk_scale_factor = qk_scale_factor
        self.use_attn_output_gate = use_attn_output_gate
        self.output_multiplier = output_multiplier
        self.output_soft_cap_temp = output_soft_cap_temp
        self.normalize_tok_embeddings = normalize_tok_embeddings
        self.sliding_window = sliding_window
        self.every_n_layers_nope = every_n_layers_nope
        self.hidden_act = hidden_act
        self.vision_latent_dim = vision_latent_dim
        self.vision_output_dim = vision_output_dim
        self.vision_layers = vision_layers
        self.vision_heads = vision_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_patch_size = vision_patch_size
        self.vision_patch_temporal = vision_patch_temporal
        self.vision_downsample_factor = vision_downsample_factor
        self.vision_sparse_attention_factor = vision_sparse_attention_factor
        self.vision_pos_emb_grid_h = vision_pos_emb_grid_h
        self.vision_pos_emb_grid_w = vision_pos_emb_grid_w
        self.vision_adapter_dim = vision_adapter_dim
        self.patch_token_id = patch_token_id
        self.video_token_id = video_token_id
        self.vid_start_id = vid_start_id
        self.vid_end_id = vid_end_id
        self.vid_frame_sep_id = vid_frame_sep_id
        self.video_num_frames = video_num_frames
        self.video_sampling_fps = video_sampling_fps
        self.has_vision = has_vision

        if sliding_window_pattern is None:
            sliding_window_pattern = [sliding_window, sliding_window, sliding_window, 0]
        self.sliding_window_pattern = sliding_window_pattern

        # iRoPE: NoPE layers determined by counting backward from last layer.
        # 1 = RoPE, 0 = NoPE.
        if no_rope_layers is None:
            no_rope_layers = []
            for layer_idx in range(num_hidden_layers):
                backward_idx = num_hidden_layers - layer_idx - 1
                is_nope = backward_idx % every_n_layers_nope == 0
                no_rope_layers.append(0 if is_nope else 1)
        self.no_rope_layers = no_rope_layers

        # sliding_attention for RoPE layers, full_attention for NoPE layers.
        if layer_types is None:
            nope_freq = every_n_layers_nope or 1
            layer_types = []
            for layer_idx in range(num_hidden_layers):
                count_backward = layer_idx + nope_freq - num_hidden_layers % nope_freq
                cfg_val = sliding_window_pattern[count_backward % len(sliding_window_pattern)]
                layer_types.append("sliding_attention" if cfg_val > 0 else "full_attention")
        self.layer_types = layer_types

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


__all__ = ["OnyxConfig"]
