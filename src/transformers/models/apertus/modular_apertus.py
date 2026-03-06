# Copyright 2025 the HuggingFace Inc. team and the Swiss AI Initiative. All rights reserved.
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
from collections.abc import Callable

import torch
from torch import nn

from ...activations import ACT2CLS
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..nemotron.modeling_nemotron import NemotronMLP


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="swiss-ai/Apertus-8B")
class ApertusConfig(PreTrainedConfig):
    r"""
    ```python
    >>> from transformers import ApertusModel, ApertusConfig

    >>> # Initializing a Apertus-8B style configuration
    >>> configuration = ApertusConfig()

    >>> # Initializing a model from the Apertus-8B style configuration
    >>> model = ApertusModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "apertus"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 12000000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 131072,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 14336,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "xielu",
        max_position_embeddings: int | None = 65536,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = 3,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | None = {
            "rope_type": "llama3",
            "rope_theta": 12000000.0,
            "factor": 8.0,
            "original_max_position_embeddings": 8192,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
        },
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


class ApertusMLP(NemotronMLP):
    def __init__(self, config):
        super().__init__(config)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act == "xielu":
            self.act_fn = ACT2CLS["xielu"](dtype=config.dtype)


class ApertusRMSNorm(LlamaRMSNorm):
    pass


class ApertusRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class ApertusAttention(LlamaAttention):
    def __init__(self, config: ApertusConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.q_norm = ApertusRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = ApertusRMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ApertusDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: ApertusConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_layernorm = ApertusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layernorm = ApertusRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        del self.input_layernorm
        del self.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ApertusPreTrainedModel(LlamaPreTrainedModel):
    pass


class ApertusModel(LlamaModel):
    pass


class ApertusForCausalLM(LlamaForCausalLM):
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, ApertusForCausalLM

        >>> model = ApertusForCausalLM.from_pretrained("swiss-ai/Apertus-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        return super().forward(**super_kwargs)


class ApertusForTokenClassification(LlamaForTokenClassification):
    pass


__all__ = [
    "ApertusConfig",
    "ApertusModel",
    "ApertusForCausalLM",
    "ApertusForTokenClassification",
    "ApertusPreTrainedModel",
]
