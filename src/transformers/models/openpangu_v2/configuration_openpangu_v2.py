# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""OpenPangu_v2 model configuration"""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="openpangu/openPangu-2.0-Flash")
@strict
class OpenPanguV2Config(PreTrainedConfig):
    model_type = "openpangu_v2"
    keys_to_ignore_at_inference = ["past_key_values"]
    tie_word_embeddings: bool = True

    # Default tensor parallel plan for base model `OpenPangu_v2`
    base_model_tp_plan = {
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        attention_dropout: float | None = 0.0,
        block_post_layernorm_idx: list[int] | None = None,
        dsa_layers: list[str] | None = None,
        first_k_dense_replace: int | None = 0,
        head_dim: int | None = None,
        hidden_act: str | None = "silu",
        hidden_size: int | None = None,
        index_head_dim: int | None = None,
        index_n_heads: int | None = None,
        index_topk: int | None = None,
        intermediate_size: int | None = None,
        kv_lora_rank: int | None = None,
        layer_types: list[str] | None = None,
        max_position_embeddings: int | None = None,
        mhc_num_stream: int | None = None,
        mhc_recur_norm: int | None = None,
        mhc_use_gamma: bool | None = None,
        moe_intermediate_size: int | None = None,
        n_routed_experts: int | None = None,
        n_shared_experts: int | None = None,
        norm_topk_prob: bool | None = None,
        num_attention_heads: int | None = None,
        num_experts_per_tok: int | None = None,
        num_hidden_layers: int | None = 0,
        num_key_value_heads: int | None = None,
        pad_token_id: int | None = 2,
        param_sink_number: int | None = 0,
        q_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        rms_norm_eps: float | None = 1e-5,
        rope_interleave: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        routed_scaling_factor: float | None = None,
        router_sliding_window: int | None = 0,
        sandwich_norm: bool | None = False,
        sliding_window: int | list[int] | None = None,
        swa_layers: list[str] | None = None,
        use_cache: bool | None = True,
        use_mhc: bool | None = False,
        v_head_dim: int | None = None,
        vocab_size: int | None = None,
        **kwargs,
    ):
        r"""
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention probabilities.
        block_post_layernorm_idx (`list[int]`, *optional*):
            Indices of the decoder layers after which an additional block-level RMSNorm is applied.
        dsa_layers (`list[int]`, *optional*):
            Indices of the layers that run the Dynamic Sparse Attention (DSA) indexer.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of shallow layers that use a dense MLP instead of a MoE block before switching to MoE.
        head_dim (`int`, *optional*):
            Dimension of each attention head. When `qk_rope_head_dim` is provided it is derived from it.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function used by the MLP blocks.
        hidden_size (`int`, *optional*):
            Dimension of the hidden states.
        index_head_dim (`int`, *optional*):
            Per-head dimension of the DSA indexer projections.
        index_n_heads (`int`, *optional*):
            Number of heads used by the DSA indexer.
        index_topk (`int`, *optional*):
            Number of top-k tokens selected per query by the DSA indexer for sparse attention.
        intermediate_size (`int`, *optional*):
            Dimension of the "intermediate" (i.e. feed-forward) layer in the dense MLP blocks.
        kv_lora_rank (`int`, *optional*):
            Latent rank used to compress the key/value states in Multi-head Latent Attention (MLA).
        layer_types (`list[str]`, *optional*):
            Attention type of each decoder layer, one of `"full_attention"` or `"sliding_attention"`.
        max_position_embeddings (`int`, *optional*):
            Maximum sequence length the model was trained with.
        mhc_num_stream (`int`, *optional*):
            Number of streams in the mHC residual path. Hidden states are repeated along the feature dim by
            this factor when mHC is enabled.
        mhc_recur_norm (`int`, *optional*):
            Number of Sinkhorn iterations performed by the mHC module when redistributing the stream weights.
        mhc_use_gamma (`bool`, *optional*):
            Whether to apply a learnable gamma scale inside the mHC pre-normalization.
        moe_intermediate_size (`int`, *optional*):
            Per-expert intermediate dimension of each routed expert in the MoE blocks.
        n_routed_experts (`int`, *optional*):
            Number of routed experts available in each MoE block.
        n_shared_experts (`int`, *optional*):
            Number of shared experts always activated in each MoE block.
        norm_topk_prob (`bool`, *optional*):
            Whether to normalize the selected experts' routing weights so they sum to one.
        num_attention_heads (`int`, *optional*):
            Number of attention heads for each attention layer.
        num_experts_per_tok (`int`, *optional*):
            Number of routed experts selected for each token.
        num_hidden_layers (`int`, *optional*, defaults to 0):
            Number of decoder layers.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads. Defaults to `num_attention_heads` when `None`.
        pad_token_id (`int`, *optional*, defaults to 2):
            Token id of the padding token.
        param_sink_number (`int`, *optional*, defaults to 0):
            Number of learnable "sink" key/value tokens prepended to every layer's KV cache (streaming
            attention sinks).
        q_lora_rank (`int`, *optional*):
            Latent rank used to compress the query projection in MLA. When `None`, the query is projected
            directly without low-rank compression.
        qk_nope_head_dim (`int`, *optional*):
            Per-head dimension of the non-rotary component of the query/key tensors.
        qk_rope_head_dim (`int`, *optional*):
            Per-head dimension of the rotary (RoPE) component of the query/key tensors.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon used by the RMSNorm layers.
        rope_interleave (`bool`, *optional*, defaults to `False`):
            Whether to use interleaved RoPE (as in the original DeepSeek implementation) instead of
            half-rotated (NeoX/Llama-style) RoPE.
        rope_parameters (`RopeParameters` or `dict`, *optional*):
            RoPE scaling / configuration parameters. See [`RopeParameters`].
        routed_scaling_factor (`float`, *optional*):
            Scaling factor multiplied onto the routed experts' weights before adding the shared expert.
        router_sliding_window (`int`, *optional*, defaults to 0):
            Window size of the convolutional Mixture-of-Memory-Experts (MoME) path. `0` disables MoME.
        sandwich_norm (`bool`, *optional*, defaults to `False`):
            Whether to add extra pre- and post-MLP RMSNorms (sandwich norm) inside the decoder layer.
        sliding_window (`int` or `list[int]`, *optional*):
            Window size(s) used by the sliding-window attention layers.
        swa_layers (`list[int]`, *optional*):
            Indices of the layers that use sliding-window attention. Used to derive `layer_types` when the
            latter is `None`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use the KV cache for fast autoregressive decoding.
        use_mhc (`bool`, *optional*, defaults to `False`):
            Whether to enable the multi-stream Hopfield-controlled (mHC) residual path.
        v_head_dim (`int`, *optional*):
            Per-head dimension of the value states in Multi-head Latent Attention.
        vocab_size (`int`, *optional*):
            Vocabulary size of the OpenPanguV2 model.

        Example:

        ```python
        >>> from transformers import OpenPanguV2Model, OpenPanguV2Config

        >>> # Initializing an OpenPanguV2 style configuration
        >>> configuration = OpenPanguV2Config()

        >>> # Initializing a model from the OpenPanguV2 style configuration
        >>> model = OpenPanguV2Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
        """
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types

        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        if qk_rope_head_dim is not None and qk_nope_head_dim is not None:
            self.head_dim = qk_rope_head_dim
            self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave

        self.sliding_window = sliding_window
        self.swa_layers = swa_layers

        self.param_sink_number = param_sink_number
        self.router_sliding_window = router_sliding_window
        self.sandwich_norm = sandwich_norm
        self.block_post_layernorm_idx = block_post_layernorm_idx
        self.use_mhc = use_mhc
        self.mhc_use_gamma = mhc_use_gamma
        self.mhc_recur_norm = mhc_recur_norm
        self.mhc_num_stream = mhc_num_stream

        # Indexer (DSA) parameters
        self.dsa_layers = dsa_layers
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads

        if self.layer_types is None:
            if self.swa_layers is not None:
                self.layer_types = [
                    "sliding_attention" if i in self.swa_layers else "full_attention"
                    for i in range(self.num_hidden_layers)
                ]
            else:
                self.layer_types = ["full_attention" for _ in range(self.num_hidden_layers)]

        if (
            num_hidden_layers is not None
            and self.layer_types is not None
            and len(self.layer_types) != num_hidden_layers
        ):
            raise ValueError(
                f"`num_hidden_layers` ({num_hidden_layers}) must be equal to the number of layer types "
                f"({len(layer_types)})"
            )

        super().__init__(**kwargs)


__all__ = ["OpenPanguV2Config"]
