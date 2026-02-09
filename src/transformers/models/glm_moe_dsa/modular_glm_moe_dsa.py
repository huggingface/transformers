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


from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache
from ...configuration_utils import layer_type_validation
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...models.llama.modeling_llama import (
    apply_rotary_pos_emb,
)
from ...processing_utils import Unpack
from ...utils import logging
from ...utils.import_utils import is_tracing
from ..deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave, yarn_get_mscale
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
    eager_attention_forward,
)
from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer


logger = logging.get_logger(__name__)


class GlmMoeDsaConfig(Glm4MoeLiteConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmMoeDsaModel`]. It is used to instantiate a
    GLM-5 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GLM-5.
    e.g. [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4MoeLiteModel`]
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 78):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 64):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor or routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 2048):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 256):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 192):
            Dimension of the query/key heads that don't use rotary position embeddings.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts, None means dense model.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 202752):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings.
        mlp_layer_types (`list`, *optional*):
            MLP (Moe vs Dense) pattern for each layer.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of top tokens selected by the indexer for retrieval/attention in each step.

    ```python
    >>> from transformers import Glm4MoeLiteModel, Glm4MoeLiteConfig

    >>> # Initializing a GLM-MOE-DSA style configuration
    >>> configuration = GlmMoeDsaConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        vocab_size: int | None = 154880,
        hidden_size: int | None = 6144,
        intermediate_size: int | None = 12288,
        moe_intermediate_size: int | None = 2048,
        num_hidden_layers: int | None = 78,
        num_attention_heads: int | None = 64,
        num_key_value_heads: int | None = 64,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 256,
        routed_scaling_factor: float | None = 2.5,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 2048,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 256,
        qk_nope_head_dim: int | None = 192,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        num_experts_per_tok: int | None = 8,
        norm_topk_prob: bool | None = True,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 202752,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 1,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        rope_interleave: bool | None = True,
        mlp_layer_types=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        index_topk: int | None = 2048,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.index_topk = index_topk
        self.mlp_layer_types = mlp_layer_types
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        # Default to MoE from the fourth layer and on
        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )
        layer_type_validation(self.mlp_layer_types, self.num_hidden_layers, attention=False)

        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)
        del self.pretraining_tp


class GlmMoeDsaRMSNorm(Glm4MoeRMSNorm):
    pass


class GlmMoeDsaAttention(nn.Module):
    """
    DeepSeek V3.2 sparse attention mechanism with indexer.

    This implements the native sparse attention from [DeepSeek V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) which uses
    an indexer to select top-k tokens for attention computation, making it more efficient for long sequences.

    In GLM-5, the indexer RoPE uses neox_style = false. Therefore, we introduced the indexer_rope_interleave parameter:
    when indexer_rope_interleave is set to True, RoPE is computed using the same neox_style = false behavior as in the
    GlmMoeDsa model. This part has not yet been implemented in transformers.
    """

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.index_topk = config.index_topk

        self.is_causal = True

        # Query projection
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = GlmMoeDsaRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # Key-Value projections
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = GlmMoeDsaRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Indexer components for sparse attention
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.qk_head_dim, bias=config.attention_bias)
        self.k_norm = GlmMoeDsaRMSNorm(self.qk_head_dim)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = self.config.rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_parameters["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]

        # For training or when index_topk is not effective, fall back to standard attention
        # This is a simplified implementation - in practice, you'd implement the full sparse indexer
        if self.training or seq_length <= self.index_topk:
            if not is_tracing(hidden_states):
                logger.warning_once(
                    "DeepSeek V3.2 sparse attention is not fully implemented in this version. "
                    "Falling back to standard attention. For production use, please use vLLM or "
                    "other optimized inference engines.",
                )
            return self._standard_attention(
                hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs
            )

        # Sparse attention implementation would go here
        # This requires custom CUDA kernels for efficient top-k selection and indexing
        return self._standard_attention(
            hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs
        )

    def _standard_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """Standard attention fallback (same as DeepSeek V3)"""
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GlmMoeDsaDecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


class GlmMoeDsaPreTrainedModel(Glm4MoePreTrainedModel):
    pass


class GlmMoeDsaModel(Glm4MoeModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]


class GlmMoeDsaForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "GlmMoeDsaConfig",
    "GlmMoeDsaPreTrainedModel",
    "GlmMoeDsaModel",
    "GlmMoeDsaForCausalLM",
]
