# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Configuration for the normalized latent recurrent Dory architecture."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
@strict
class DoryConfig(PreTrainedConfig):
    r"""
    ngpt_alpha_init (`float`, *optional*, defaults to 0.05):
        Initial interpolation rate for the normalized residual updates.
    n_input_layers (`int`, *optional*, defaults to 16):
        Number of physical blocks before the recurrent section.
    n_recurrent_layers (`int`, *optional*):
        Number of shared blocks in the recurrent section. Defaults to the remaining blocks.
    n_output_layers (`int`, *optional*, defaults to 16):
        Number of physical blocks after the recurrent section.
    n_recurrent_loops (`int`, *optional*, defaults to 2):
        Number of passes through the shared recurrent blocks.
    recurrent_kv_cache_mode (`str`, *optional*, defaults to `"per_loop"`):
        `"per_loop"` keeps independent KV histories for each recurrent pass. `"last_loop"` shares
        a history and overwrites the current chunk on each pass.

    ```python
    >>> from transformers import DoryConfig, DoryForCausalLM

    >>> configuration = DoryConfig()
    >>> model = DoryForCausalLM(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "dory"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 10_000.0

    vocab_size: int = 131072
    hidden_size: int = 2560
    num_hidden_layers: int = 72
    intermediate_size: int = 7168
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 8192
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    # Also normalizes the learned nGPT scales at inference time.
    initializer_range: float = 0.0177
    sliding_window: int | None = 1024
    rope_parameters: dict | None = None
    layer_types: list[str] | None = None
    ngpt_alpha_init: float = 0.05
    n_input_layers: int = 16
    n_recurrent_layers: int | None = None
    n_output_layers: int = 16
    n_recurrent_loops: int = 2
    # "last_loop" reuses one cache, overwritten by each recurrent loop.
    recurrent_kv_cache_mode: str = "per_loop"

    def __post_init__(self, **kwargs):
        # Published checkpoints include these now-fixed architectural fields.
        legacy_values = {
            "hidden_act": ("silu",),
            "mlp_bias": (False,),
            "attention_bias": (False,),
            "attention_output_gate_logit_scale_init": (None, self.hidden_size**0.5),
        }
        for key, supported_values in legacy_values.items():
            if key in kwargs:
                value = kwargs.pop(key)
                if value not in supported_values:
                    raise ValueError(f"Dory requires {key} to be one of {supported_values!r}, got {value!r}")
        if self.layer_types is None:
            self.layer_types = [
                "mlp" if i % 2 else "full_attention" if i % 8 == 6 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters is None:
            self.rope_parameters = {
                kind: {
                    "rope_type": "default",
                    "rope_theta": 10_000_000.0 if kind == "full_attention" else 10_000.0,
                }
                for kind in ("sliding_attention", "full_attention")
                if kind in self.layer_types
            }
        if self.n_recurrent_layers is None:
            self.n_recurrent_layers = self.num_hidden_layers - self.n_input_layers - self.n_output_layers
        super().__post_init__(**kwargs)

    def validate_layer_type(self):
        # Each physical block is attention-only or MLP-only.
        valid_types = {"sliding_attention", "full_attention", "mlp"}
        if self.layer_types is None or any(kind not in valid_types for kind in self.layer_types):
            raise ValueError("layer_types must be a list of sliding_attention, full_attention, or mlp")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries")

    def validate_architecture(self):
        if self.layer_types is None or self.n_recurrent_layers is None:
            raise ValueError("Dory layer layout must be initialized")
        partitions = (
            self.n_input_layers,
            self.n_recurrent_layers,
            self.n_output_layers,
        )
        if any(n < 0 for n in partitions) or sum(partitions) != self.num_hidden_layers:
            raise ValueError("Dory input, recurrent, and output layers must sum to num_hidden_layers")
        if self.n_recurrent_loops < 1:
            raise ValueError("n_recurrent_loops must be a positive integer")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
        if any(n <= 0 for n in (self.head_dim, self.num_attention_heads, self.num_key_value_heads)):
            raise ValueError("Attention dimensions must be positive integers")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive")
        if "sliding_attention" in self.layer_types and (self.sliding_window is None or self.sliding_window <= 0):
            raise ValueError("sliding_window must be a positive integer")
        if self.tie_word_embeddings:
            raise ValueError("Dory requires untied embeddings")

    def validate_rope(self):
        if self.layer_types is None:
            raise ValueError("Dory layer layout must be initialized")
        attention_types = set(self.layer_types) - {"mlp"}
        if self.rope_parameters is None or (
            set(self.rope_parameters) - {"sliding_attention", "full_attention"}
            or attention_types - self.rope_parameters.keys()
        ):
            raise ValueError("rope_parameters must define each attention block type")
        for rope in self.rope_parameters.values():
            if not isinstance(rope, dict) or rope.get("rope_type") != "default" or rope.get("rope_theta", 0) <= 0:
                raise ValueError("Dory requires default RoPE with a positive rope_theta")
            if rope.get("partial_rotary_factor", 1.0) != 1.0:
                raise ValueError("Dory requires full-head rotary embeddings (partial_rotary_factor=1.0)")
            if self.head_dim % 2:
                raise ValueError("Rotary dimensions must be even")


__all__ = ["DoryConfig"]
