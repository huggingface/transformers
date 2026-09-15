# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""ZGCM model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="zgcagi/ZGCM-1-7B")
@strict
class ZgcmConfig(PreTrainedConfig):
    r"""
    partial_rotary_factor (`float`, *optional*, defaults to 0.334):
        Fraction of each attention head rotated, rounded down to an even dimension.
    window_attn_skip_freq (`int`, *optional*, defaults to 6):
        Every nth layer uses full attention when `layer_types` is omitted.
    attention_gate_layers (`list[bool]`, *optional*):
        Whether to apply an elementwise sigmoid gate to each layer's attention output.
    rope_theta (`float`, *optional*, defaults to 10000000.0):
        Base period of the rotary embeddings.
    """

    model_type = "zgcm"
    keys_to_ignore_at_inference = ["past_key_values"]
    vocab_size: int = 155136
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.334
    max_position_embeddings: int = 262144
    sliding_window: int = 128
    window_attn_skip_freq: int = 6
    layer_types: list[str] | None = None
    attention_gate_layers: list[bool] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    pad_token_id: int | None = 154820
    eos_token_id: int | list[int] | None = 154820
    bos_token_id: int | None = None
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        if self.window_attn_skip_freq <= 0 or self.sliding_window <= 0:
            raise ValueError("window_attn_skip_freq and sliding_window must be positive")
        if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        rotary_dim -= rotary_dim % 2
        if not 0 < self.partial_rotary_factor <= 1 or rotary_dim <= 0:
            raise ValueError("partial_rotary_factor must produce a positive even rotary dimension")
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.window_attn_skip_freq == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers or any(
            kind not in ("full_attention", "sliding_attention") for kind in self.layer_types
        ):
            raise ValueError("layer_types must contain one valid attention type per layer")
        if self.attention_gate_layers is None:
            self.attention_gate_layers = [kind == "sliding_attention" for kind in self.layer_types]
        if len(self.attention_gate_layers) != self.num_hidden_layers:
            raise ValueError("attention_gate_layers must contain one entry per layer")
        super().__post_init__(**kwargs)


__all__ = ["ZgcmConfig"]
