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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ibm-fms/FalconH1-9.8b-2.2T-hf")
class FalconH1Config(PreTrainedConfig):
    r"""
    num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
        Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
        integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
        logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
        sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
        significantly.
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
    projectors_bias (`bool`, *optional*, defaults to `False`):
        Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the attention block
    """

    model_type = "falcon_h1"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 128000,
        tie_word_embeddings: bool | None = False,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 14336,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        hidden_act: str | None = "silu",
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: int | None = True,
        num_logits_to_keep: int | None = 1,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        max_position_embeddings: int | None = 8192,
        attention_dropout: float | None = 0.0,
        mamba_d_ssm: int | None = 1024,
        mamba_n_heads: int | None = 128,
        mamba_d_head: str | None = "auto",
        mamba_n_groups: int | None = 1,
        mamba_d_state: int | None = 256,
        mamba_d_conv: int | None = 4,
        mamba_expand: int | None = 2,
        mamba_chunk_size: int | None = 256,
        mamba_conv_bias: bool | None = True,
        mamba_proj_bias: bool | None = False,
        mamba_norm_before_gate: bool | None = True,
        mamba_rms_norm: bool | None = False,
        time_step_min: float | None = 0.001,
        time_step_max: float | None = 0.1,
        time_step_limit: tuple[float, float] | None = (0.0, float("inf")),
        projectors_bias: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        lm_head_multiplier: float | None = 1.0,
        embedding_multiplier: float | None = 1.0,
        mlp_multipliers: int | None = None,
        key_multiplier: int | None = None,
        attention_out_multiplier: int | None = None,
        attention_in_multiplier: int | None = None,
        ssm_multipliers: int | None = None,
        ssm_in_multiplier: int | None = None,
        ssm_out_multiplier: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.attention_bias = False
        self.mlp_bias = False

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.projectors_bias = projectors_bias
        mamba_intermediate = mamba_expand * hidden_size if mamba_d_ssm is None else mamba_d_ssm

        if mamba_intermediate % mamba_n_heads != 0:
            raise ValueError("mamba_n_heads must divide mamba_expand * hidden_size")

        # for the mamba_v2, must satisfy the following
        if mamba_d_head == "auto":
            mamba_d_head = mamba_intermediate // mamba_n_heads

        if mamba_d_head * mamba_n_heads != mamba_intermediate:
            raise ValueError("The dimensions for the Mamba head state do not match the model intermediate_size")

        self.mamba_d_ssm = mamba_d_ssm
        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_head = mamba_d_head
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        self.mamba_norm_before_gate = mamba_norm_before_gate
        self.mamba_rms_norm = mamba_rms_norm
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_limit = tuple(time_step_limit) if time_step_limit is not None else None

        self.lm_head_multiplier = lm_head_multiplier
        self.embedding_multiplier = embedding_multiplier

        if mlp_multipliers is not None:
            self.mlp_multipliers = mlp_multipliers
        else:
            self.mlp_multipliers = [1.0, 1.0]

        if attention_out_multiplier is not None:
            self.attention_out_multiplier = attention_out_multiplier
        else:
            self.attention_out_multiplier = 1.0

        if attention_in_multiplier is not None:
            self.attention_in_multiplier = attention_in_multiplier
        else:
            self.attention_in_multiplier = 1.0

        if key_multiplier is not None:
            self.key_multiplier = key_multiplier
        else:
            self.key_multiplier = 1.0

        if ssm_multipliers is not None:
            self.ssm_multipliers = ssm_multipliers
        else:
            self.ssm_multipliers = [1.0, 1.0, 1.0, 1.0, 1.0]

        if ssm_in_multiplier is not None:
            self.ssm_in_multiplier = ssm_in_multiplier
        else:
            self.ssm_in_multiplier = 1.0

        if ssm_out_multiplier is not None:
            self.ssm_out_multiplier = ssm_out_multiplier
        else:
            self.ssm_out_multiplier = 1.0

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_parameters = rope_parameters
        super().__init__(**kwargs)

    @property
    def layers_block_type(self):
        return ["attention" for i in range(self.num_hidden_layers)]


__all__ = ["FalconH1Config"]
