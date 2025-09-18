# coding=utf-8
# Copyright 2025 Dustin Loring
# Website: dustinwloring1988.github.io
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

from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...modeling_rope_utils import rope_config_validation


class BlueberryConfig(PretrainedConfig):
    model_type = "blueberry"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        max_position_embeddings=1024,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        hidden_act="silu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        attention_bias=True,
        attention_dropout=0.0,
        rms_norm_eps=1e-05,
        rope_scaling=None,
        rope_theta=150000,
        sliding_window=128,
        tie_word_embeddings=False,
        use_cache=True,
        layer_types=None,
        vocab_size=100000,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_proj_to_labels = summary_proj_to_labels
        self.summary_first_dropout = summary_first_dropout
        self.scale_attn_weights = scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        # Default rope_scaling required by the spec
        if rope_scaling is None:
            rope_scaling = {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn",
                "truncate": False,
            }
        # Backward-compat normalization
        if isinstance(rope_scaling, dict) and "type" in rope_scaling and "rope_type" not in rope_scaling:
            rope_scaling["rope_type"] = rope_scaling.pop("type")
        self.rope_scaling = rope_scaling

        self.sliding_window = sliding_window
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache

        # Set up layer_types default if not provided
        if layer_types is None:
            # Use provided pattern from the spec; if fewer than num_hidden_layers, repeat as needed and trim
            default_pattern = [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ]
            repeated = (default_pattern * ((num_hidden_layers + len(default_pattern) - 1) // len(default_pattern)))[
                :num_hidden_layers
            ]
            layer_types = repeated
        self.layer_types = layer_types

        # Validate
        layer_type_validation(self.layer_types, self.num_hidden_layers)
        rope_config_validation(self)


__all__ = ["BlueberryConfig"]

