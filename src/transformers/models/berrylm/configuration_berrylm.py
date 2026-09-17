# Copyright 2026 The RWB AI Assist team and The HuggingFace Inc. team. All rights reserved.
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
"""BerryLM-OS model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="RWB/BerryLM-OS")
@strict
class BerryLMConfig(PreTrainedConfig):
    r"""
    attn_output_gate (`bool`, *optional*, defaults to `True`):
        Whether the full-attention layers gate their output with a sigmoid branch of the query projection.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each key head in linear attention.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head in linear attention.
    linear_num_key_heads (`int`, *optional*, defaults to 16):
        Number of key heads used in linear attention layers.
    linear_num_value_heads (`int`, *optional*, defaults to 32):
        Number of value heads used in linear attention layers. Must be a multiple of `linear_num_key_heads`.
    attn_res_block_size (`int`, *optional*, defaults to 8):
        Block size of the Gated Block AttnRes mixer: every decoder layer reads a softmax mixture of the residual
        streams committed every `attn_res_block_size` layers (the embeddings count as block 0) and the current
        stream, gated back toward the identity by a per-layer scalar. `0` disables the mixer.
    attn_res_gated (`bool`, *optional*, defaults to `True`):
        Whether the mixer output is gated toward the identity (`y = x + tanh(g) * (mix - x)`, `g = 0` is the exact
        identity). `False` feeds the plain mixture to the layer.
    attn_res_eps (`float`, *optional*, defaults to 1e-06):
        Epsilon of the RMS normalization applied to the mixer keys.
    kda_gate_bottleneck (`int`, *optional*, defaults to 128):
        Rank of the low-rank per-channel forget gate of the linear-attention layers
        (`g = -exp(A_log) * softplus(f_up(f_down(x)) + dt_bias)`, shape `[linear_num_value_heads,
        linear_key_head_dim]`; `f_down_proj` / `f_up_proj`).
    kda_safe_gate (`bool`, *optional*, defaults to `False`):
        Whether the forget gate is clamped from below by `kda_gate_lower_bound`.
    kda_gate_lower_bound (`float`, *optional*):
        Lower bound of the log-decay when `kda_safe_gate` is enabled.

    ```python
    >>> from transformers import BerryLMModel, BerryLMConfig

    >>> # Initializing a BerryLM style configuration
    >>> configuration = BerryLMConfig()

    >>> # Initializing a model from the BerryLM-OS style configuration
    >>> model = BerryLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "berrylm"
    keys_to_ignore_at_inference = ["past_key_values"]

    # The small KDA gate projections (f_down_proj / f_up_proj) and the AttnRes parameters stay replicated.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }

    vocab_size: int = 180224
    hidden_size: int = 2048
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    attn_output_gate: bool = True
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    layer_types: list[str] | None = None
    attn_res_block_size: int = 8
    attn_res_gated: bool = True
    attn_res_eps: float = 1e-6
    kda_gate_bottleneck: int = 128
    kda_safe_gate: bool = False
    kda_gate_lower_bound: float | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC
        # `full_attention_interval` only seeds `layer_types` (one full-attention layer per `interval` layers).
        interval_pattern = kwargs.pop("full_attention_interval", 4)
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        bad = sorted({t for t in self.layer_types if t not in ("linear_attention", "full_attention")})
        if bad:
            raise ValueError(f"`layer_types` must contain only 'linear_attention'/'full_attention', got {bad}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` has {len(self.layer_types)} entries for {self.num_hidden_layers} hidden layers"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError("`linear_num_value_heads` must be a multiple of `linear_num_key_heads`")
        if self.attn_res_block_size < 0:
            raise ValueError("`attn_res_block_size` must be >= 0")
        if self.kda_safe_gate and self.kda_gate_lower_bound is None:
            raise ValueError("`kda_gate_lower_bound` is required when `kda_safe_gate=True`")

        super().__post_init__(**kwargs)


__all__ = ["BerryLMConfig"]
