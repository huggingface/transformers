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
"""Dory model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
@strict
class DoryConfig(PreTrainedConfig):
    r"""
    ngpt_alpha_init (`float`, *optional*, defaults to 0.05):
        Initial value of the nGPT residual interpolation rates `lambda_1`/`lambda_2`.
    n_input_layers (`int`, *optional*, defaults to 8):
        Number of decoder layers run once before the recurrent section.
    n_recurrent_layers (`int`, *optional*):
        Number of decoder layers in the recurrent section. Defaults to the remaining layers.
    n_output_layers (`int`, *optional*, defaults to 8):
        Number of decoder layers run once after the recurrent section.
    n_recurrent_loops (`int`, *optional*, defaults to 2):
        Number of passes through the recurrent section. More loops trade compute for accuracy.
    recurrent_kv_cache_mode (`str`, *optional*, defaults to `"per_loop"`):
        `"per_loop"` keeps one KV cache per recurrent loop. `"last_loop"` keeps a single KV cache per recurrent layer,
        overwritten by each loop, so cache memory does not grow with the loop count.

    ```python
    >>> from transformers import DoryConfig, DoryForCausalLM

    >>> configuration = DoryConfig()
    >>> model = DoryForCausalLM(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "dory"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 131072
    hidden_size: int = 2560
    intermediate_size: int = 7168
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    # nGPT scaled parameters (paper §2.6) are stored as `s` and used as `s * s_init / initializer_range`
    initializer_range: float = 0.0177
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    sliding_window: int = 1024
    layer_types: list[str] | None = None
    ngpt_alpha_init: float = 0.05
    n_input_layers: int = 8
    n_recurrent_layers: int | None = None
    n_output_layers: int = 8
    n_recurrent_loops: int = 2
    recurrent_kv_cache_mode: str = "per_loop"

    def __post_init__(self, **kwargs):
        if self.layer_types is not None and "mlp" in self.layer_types:
            # Released `config.json`s list attention and mlp as separate blocks: each (attention, mlp) pair is one layer
            self.layer_types = self.layer_types[::2]
            self.num_hidden_layers //= 2
            self.n_input_layers //= 2
            self.n_output_layers //= 2
            if self.n_recurrent_layers is not None:
                self.n_recurrent_layers //= 2
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(self.num_hidden_layers)
            ]
        if self.n_recurrent_layers is None:
            self.n_recurrent_layers = self.num_hidden_layers - self.n_input_layers - self.n_output_layers
        # The recurrent layers run once per loop, their cache keeps one history per loop
        recurrent_end = self.n_input_layers + self.n_recurrent_layers
        self.layer_types = [
            f"recurrent_{layer_type.removeprefix('recurrent_')}"
            if self.n_input_layers <= i < recurrent_end
            else layer_type
            for i, layer_type in enumerate(self.layer_types)
        ]
        if self.rope_parameters is None:
            rope_thetas = {"sliding_attention": 10_000.0, "full_attention": 10_000_000.0}
            self.rope_parameters = {
                layer_type: {"rope_type": "default", "rope_theta": rope_thetas[layer_type]}
                for layer_type in {layer_type.removeprefix("recurrent_") for layer_type in self.layer_types}
            }
        # Recurrent layers use the rope of their attention type
        for layer_type in set(self.layer_types):
            self.rope_parameters.setdefault(layer_type, self.rope_parameters[layer_type.removeprefix("recurrent_")])
        super().__post_init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        # Serialize the layout `__post_init__` converted from, as released checkpoints (and vLLM / llama.cpp) store it:
        # each decoder layer is an attention block followed by an mlp block
        layer_types = [layer_type.removeprefix("recurrent_") for layer_type in self.layer_types]
        output["layer_types"] = [block for layer_type in layer_types for block in (layer_type, "mlp")]
        output["rope_parameters"] = {
            layer_type: rope
            for layer_type, rope in output["rope_parameters"].items()
            if not layer_type.startswith("recurrent_")
        }
        for key in ("num_hidden_layers", "n_input_layers", "n_recurrent_layers", "n_output_layers"):
            output[key] = 2 * getattr(self, key)
        return output

    def validate_architecture(self):
        if self.n_input_layers + self.n_recurrent_layers + self.n_output_layers != self.num_hidden_layers:
            raise ValueError("n_input_layers + n_recurrent_layers + n_output_layers must equal num_hidden_layers")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError(
                f"recurrent_kv_cache_mode must be per_loop or last_loop, got {self.recurrent_kv_cache_mode}"
            )


__all__ = ["DoryConfig"]
