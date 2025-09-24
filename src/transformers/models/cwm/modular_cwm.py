# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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

import torch

from ...utils import logging
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
# Import only Gemma3's sliding window attention implementation
from ..gemma3.modeling_gemma3 import Gemma3Attention


logger = logging.get_logger(__name__)


class CwmTextConfig(LlamaConfig):
    """
    Text configuration class for CWM (Code World Model).

    CWM uses Llama3 architecture with Gemma3's interleaved sliding window attention.
    This is the main configuration class since CWM is text-only.
    """

    model_type = "cwm_text"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=6144,
        intermediate_size=21504,
        num_hidden_layers=64,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        eos_token_id=[128001, 128008, 128009],
        bos_token_id=128000,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        attention_bias=False,
        attention_dropout=0.0,
        # Sliding window attention parameters (from Gemma3)
        sliding_window=8192,
        layer_types=None,
        query_pre_attn_scalar=128,  # Set to head_dim for proper scaling
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        rope_local_base_freq=10000.0,
        use_bidirectional_attention=False,
        pretraining_tp=1,
        mlp_bias=False,
        rope_scaling=None,
        **kwargs,
    ):
        # Set rope_scaling to match your exact config.json if not provided
        if rope_scaling is None:
            rope_scaling = {
                "factor": 16.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }

        # Set layer_types based on your conversion script pattern if not provided
        if layer_types is None:
            # Generate pattern: layers 0, 4, 8, 12, etc. are full attention, others are sliding
            layer_types = []
            for i in range(num_hidden_layers):
                if i % 4 == 0:  # Every 4th layer starting from 0
                    layer_types.append("full_attention")
                else:
                    layer_types.append("sliding_attention")

        # Call LlamaConfig.__init__ with standard Llama parameters
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rope_scaling=rope_scaling,
            pretraining_tp=pretraining_tp,
            mlp_bias=mlp_bias,
            **kwargs,
        )

        # Add CWM-specific sliding window parameters
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.rope_local_base_freq = rope_local_base_freq
        self.use_bidirectional_attention = use_bidirectional_attention

        if use_bidirectional_attention:
            self.sliding_window = (self.sliding_window // 2) + 1


class CwmConfig(CwmTextConfig):
    model_type = "cwm"


# Use Llama components for weight compatibility
class CwmMLP(LlamaMLP):
    pass


class CwmRMSNorm(LlamaRMSNorm):
    """
    CWM RMSNorm that handles both Llama and Gemma3 parameter styles.
    - Llama calls: CwmRMSNorm(hidden_size, eps=rms_norm_eps)
    - Gemma3 calls: CwmRMSNorm(dim=head_dim, eps=rms_norm_eps)
    """
    def __init__(self, *args, hidden_size=None, dim=None, eps=None, **kwargs):
        # Gemma3 call with dim parameter
        if dim is not None:
            hidden_size = dim
        elif hidden_size is not None:
            pass  # Use provided hidden_size
        elif len(args) > 0:
            # Handle positional arg (Llama-style)
            hidden_size = args[0]
        else:
            raise ValueError("Must provide either hidden_size, dim, or positional argument")
        
        eps = eps if eps is not None else 1e-6
        super().__init__(hidden_size, eps=eps)


class CwmRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class CwmAttention(Gemma3Attention):
    """
    Gemma3Attention for sliding window.
    """
    pass


class CwmDecoderLayer(LlamaDecoderLayer):
    """
    CWM decoder layer using Llama structure but with Gemma3 sliding window attention.
    """
    def __init__(self, config, layer_idx: int):
        # Initialize as Llama layer first
        super().__init__(config, layer_idx)
        # Replace the attention with CwmAttention (which has sliding window support)
        self.self_attn = CwmAttention(config, layer_idx)
        # Store layer index for attention module
        self.self_attn.layer_idx = layer_idx


class CwmPreTrainedModel(LlamaPreTrainedModel):
    config_class = CwmTextConfig
    base_model_prefix = "model"


class CwmModel(LlamaModel):
    """
    CWM model using Llama architecture with sliding window attention.
    This maintains Llama weight structure (model.layers.X.self_attn.q_proj, etc.)
    """
    config_class = CwmTextConfig

    def __init__(self, config):
        super().__init__(config)
        # Replace decoder layers with CwmDecoderLayer (which has sliding window attention)
        self.layers = torch.nn.ModuleList(
            [CwmDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class CwmForCausalLM(LlamaForCausalLM):
    """
    CWM For Causal Language Modeling using Llama structure with sliding window attention.
    This maintains weight compatibility with your Llama3-based checkpoint.
    """
    config_class = CwmTextConfig

    def __init__(self, config):
        super().__init__(config)
        # Replace the model with CwmModel (which has sliding window attention)
        self.model = CwmModel(config)


__all__ = [
    "CwmTextConfig",  # Primary config class
    "CwmConfig",      # Main config class
    "CwmPreTrainedModel",
    "CwmModel",
    "CwmForCausalLM",
]
