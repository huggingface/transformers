# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""Bamba model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@strict
@auto_docstring(
    custom_intro="""
    The BambaModel is a hybrid [mamba2](https://github.com/state-spaces/mamba) architecture with SwiGLU.
    The checkpoints are  jointly trained by IBM, Princeton, and UIUC.
    """,
    checkpoint="ibm-fms/Bamba-9.8b-2.2T-hf",
)
class BambaConfig(PreTrainedConfig):
    r"""
    num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
        Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
        integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
        logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
        sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
        significantly.
    attn_layer_indices (`list`, *optional*):
        Specifies the layer indices that will have full attention. Must contain values at most num_hidden_layers.
    z_loss_coefficient (`float`, *optional*, defaults to 0.0):
        Coefficient for auxiliary z-loss used to control logit growth during training
    """

    model_type = "bamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 128000
    tie_word_embeddings: bool = False
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 8
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    num_logits_to_keep: int | None = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    max_position_embeddings: int = 262144
    attention_dropout: float | int | None = 0.0
    attn_layer_indices: list[int] | None = None
    mamba_n_heads: int | None = 128
    mamba_d_head: str | int | None = "auto"
    mamba_n_groups: int | None = 1
    mamba_d_state: int | None = 256
    mamba_d_conv: int | None = 4
    mamba_expand: int | None = 2
    mamba_chunk_size: int | None = 256
    mamba_conv_bias: bool | None = True
    mamba_proj_bias: bool | None = False
    time_step_min: float | None = 0.001
    time_step_max: float | None = 0.1
    time_step_limit: list[float] | tuple[float, float] | None = (0.0, float("inf"))
    z_loss_coefficient: float | None = 0.0
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self, **kwargs):
        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # for the mamba_v2, must satisfy the following
        if self.mamba_d_head == "auto":
            self.mamba_d_head = self.mamba_expand * self.hidden_size // self.mamba_n_heads

        self.time_step_limit = tuple(self.time_step_limit) if self.time_step_limit is not None else None
        kwargs["partial_rotary_factor"] = 0.5  # hardcode for BC

        super().__post_init__(**kwargs)

    @property
    def layers_block_type(self):
        return [
            "attention" if (self.attn_layer_indices and i in self.attn_layer_indices) else "mamba"
            for i in range(self.num_hidden_layers)
        ]

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        mamba_intermediate = self.mamba_expand * self.hidden_size
        if mamba_intermediate % self.mamba_n_heads != 0:
            raise ValueError("mamba_n_heads must divide mamba_expand * hidden_size")

        if self.mamba_d_head * self.mamba_n_heads != mamba_intermediate:
            raise ValueError("The dimensions for the Mamba head state do not match the model intermediate_size")


__all__ = ["BambaConfig"]
