# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
#
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
"""GraniteMoeHybrid model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="ibm-granite/granite-speech-3.2-8b")
@strict
class GraniteMoeHybridConfig(PreTrainedConfig):
    r"""
    embedding_multiplier (`float`, *optional*, defaults to 1.0):
        embedding multiplier.
    logits_scaling (`float`, *optional*, defaults to 1.0):
        divisor for output logits.
    residual_multiplier (`float`, *optional*, defaults to 1.0):
        residual multiplier.
    attention_multiplier (`float`, *optional*, defaults to 1.0):
        attention multiplier.
    shared_intermediate_size (`int`, *optional*, defaults to 1024):
        intermediate size for shared experts.
    position_embedding_type (`str`, *optional*):
        Positional embedding type to be used; defaults to None. Allowed options: `[None, "rope"]`

    Example:

    ```python
    >>> from transformers import GraniteMoeHybridModel, GraniteMoeHybridConfig

    >>> # Initializing a GraniteMoeHybrid config
    >>> configuration = GraniteMoeHybridConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granitemoehybrid"
    attribute_map = {
        "layers_block_type": "layer_types",
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    embedding_multiplier: int | float | None = 1.0
    logits_scaling: int | float | None = 1.0
    residual_multiplier: int | float | None = 1.0
    attention_multiplier: int | float | None = 1.0
    num_local_experts: int | None = 8
    num_experts_per_tok: int | None = 2
    output_router_logits: bool | None = False
    router_aux_loss_coef: float | None = 0.001
    shared_intermediate_size: int = 1024
    position_embedding_type: str | None = None
    layer_types: list[str] | None = None
    mamba_n_heads: int | None = 128
    mamba_n_groups: int | None = 1
    mamba_d_state: int | None = 256
    mamba_d_head: int | str | None = "auto"
    mamba_d_conv: int | None = 4
    mamba_expand: int | None = 2
    mamba_chunk_size: int | None = 256
    mamba_conv_bias: bool | None = True
    mamba_proj_bias: bool | None = False
    time_step_min: float | None = 0.001
    time_step_max: float | None = 0.1
    time_step_limit: list[float, float] | tuple[float, float] | None = (0.0, float("inf"))

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        mamba_intermediate = self.mamba_expand * self.hidden_size
        if self.mamba_d_head == "auto":
            self.mamba_d_head = mamba_intermediate // self.mamba_n_heads

        self.time_step_limit = tuple(self.time_step_limit) if self.time_step_limit is not None else None
        if self.layer_types is None:
            self.layer_types = ["mamba"] * self.num_hidden_layers

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""

        mamba_intermediate = self.mamba_expand * self.hidden_size
        if mamba_intermediate % self.mamba_n_heads != 0:
            raise ValueError("mamba_n_heads must divide mamba_expand * hidden_size")

        if self.mamba_d_head * self.mamba_n_heads != mamba_intermediate:
            raise ValueError("The dimensions for the Mamba head state do not match the model intermediate_size")

    # overwrite the function to use in `HybridMambaAttentionDynamicCache`
    @property
    def layers_block_type(self):
        return self.layer_types


__all__ = ["GraniteMoeHybridConfig"]
