# Copyright 2025 TII and the HuggingFace Inc. team. All rights reserved.
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
"""FalconH1 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="tiiuae/Falcon-H1-1.5B-Deep-Instruct")
@strict
class FalconH1Config(PreTrainedConfig):
    r"""
    num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
        Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
        integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
        logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
        sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
        significantly.
    projectors_bias (`bool`, *optional*, defaults to `False`):
        Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the attention block
    lm_head_multiplier (`float`, *optional*, defaults to 1.0):
        The multiplier for the LM head. This is used to scale the output of the LM head.
    embedding_multiplier (`float`, *optional*, defaults to 1.0):
        The multiplier for the embedding layer. This is used to scale the output of the embedding layer.
    mlp_multipliers (`list[float]`, *optional*):
        The multipliers for the MLP layers. This is used to scale the output of the MLP layers. The first value is
        the multiplier of gate layer, the second value is the multiplier of the down_proj layer.
    key_multiplier (`float`, *optional*):
        The multiplier for the key layer. This is used to scale the output of the key layer.
    attention_out_multiplier (`float`, *optional*):
        The multiplier for the attention output layer. This is used to scale the output of the attention output
    attention_in_multiplier (`float`, *optional*):
        The multiplier for the attention input layer. This is used to scale the output of the attention input layer.
    ssm_multipliers (`list[float]`, *optional*):
        The multipliers for the SSM layers. This is used to scale the output of the SSM layers.
    ssm_in_multiplier (`float`, *optional*):
        The multiplier for the SSM input layer. This is used to scale the output of the SSM input layer.
    ssm_out_multiplier (`float`, *optional*):
        The multiplier for the SSM output layer. This is used to scale the output of the SSM output layer.
    """

    model_type = "falcon_h1"
    attribute_map = {"layer_types": "layers_block_type"}
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
    use_cache: bool | None = True
    num_logits_to_keep: int | None = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    max_position_embeddings: int = 8192
    attention_dropout: float | int | None = 0.0
    mamba_d_ssm: int | None = 1024
    mamba_n_heads: int | None = 128
    mamba_d_head: str | int | None = "auto"
    mamba_n_groups: int | None = 1
    mamba_d_state: int | None = 256
    mamba_d_conv: int | None = 4
    mamba_expand: int | None = 2
    mamba_chunk_size: int | None = 256
    mamba_conv_bias: bool | None = True
    mamba_proj_bias: bool | None = False
    mamba_norm_before_gate: bool | None = True
    mamba_rms_norm: bool | None = False
    time_step_min: float | None = 0.001
    time_step_max: float | None = 0.1
    time_step_limit: list[float, float] | tuple[float, float] | None = (0.0, float("inf"))
    projectors_bias: bool | None = False
    rope_parameters: RopeParameters | dict | None = None
    lm_head_multiplier: float | None = 1.0
    embedding_multiplier: float | None = 1.0
    mlp_multipliers: list[float] | None = None
    key_multiplier: float | None = 1.0
    attention_out_multiplier: float | None = 1.0
    attention_in_multiplier: float | None = 1.0
    ssm_multipliers: list[float] | None = None
    ssm_in_multiplier: float | None = 1.0
    ssm_out_multiplier: float | None = 1.0
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # for the mamba_v2, must satisfy the following
        mamba_intermediate = self.mamba_expand * self.hidden_size if self.mamba_d_ssm is None else self.mamba_d_ssm
        if self.mamba_d_head == "auto":
            self.mamba_d_head = mamba_intermediate // self.mamba_n_heads

        self.time_step_limit = tuple(self.time_step_limit) if self.time_step_limit is not None else None
        if self.mlp_multipliers is None:
            self.mlp_multipliers = [1.0, 1.0]

        if self.ssm_multipliers is None:
            self.ssm_multipliers = [1.0, 1.0, 1.0, 1.0, 1.0]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        mamba_intermediate = self.mamba_expand * self.hidden_size if self.mamba_d_ssm is None else self.mamba_d_ssm

        if mamba_intermediate % self.mamba_n_heads != 0:
            raise ValueError("mamba_n_heads must divide mamba_expand * hidden_size")

        if self.mamba_d_head * self.mamba_n_heads != mamba_intermediate:
            raise ValueError("The dimensions for the Mamba head state do not match the model intermediate_size")

    @property
    def layers_block_type(self):
        return ["hybrid" for i in range(self.num_hidden_layers)]


__all__ = ["FalconH1Config"]
