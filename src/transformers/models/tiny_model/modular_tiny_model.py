# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""PyTorch TinyModel model."""

from collections.abc import Callable

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="noanabeshima/tiny_model")
@strict(accept_kwargs=True)
class TinyModelConfig(PreTrainedConfig):
    r"""
    attention_output_bias (`bool`, *optional*, defaults to `True`):
        Whether to use a bias in the attention output projection.
    mlp_bias (`bool`, *optional*, defaults to `True`):
        Whether to use biases in the feed-forward projections.
    lm_head_bias (`bool`, *optional*, defaults to `True`):
        Whether to use a bias in the language-modeling head.
    embedding_initializer_range (`float`, *optional*, defaults to 0.0001):
        Standard deviation used to initialize token and position embeddings.
    """

    model_type = "tiny_model"

    vocab_size: int = 10_000
    hidden_size: int = 768
    intermediate_size: int = 3_072
    num_hidden_layers: int = 4
    num_attention_heads: int = 16
    max_position_embeddings: int = 256
    hidden_act: str = "relu"
    attention_bias: bool = False
    attention_output_bias: bool = True
    mlp_bias: bool = True
    lm_head_bias: bool = True
    initializer_range: float = 0.02
    embedding_initializer_range: float = 1e-4
    bos_token_id: int | None = 9_996
    eos_token_id: int | list[int] | None = 9_997
    pad_token_id: int | None = 9_998
    tie_word_embeddings: bool = False

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class TinyModelAttention(nn.Module):
    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_output_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TinyModelMLP(nn.Module):
    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class TinyModelDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.self_attn = TinyModelAttention(config)
        self.mlp = TinyModelMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@auto_docstring
class TinyModelPreTrainedModel(PreTrainedModel):
    config: TinyModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TinyModelDecoderLayer"]
    _supports_attention_backend = True
    _supports_sdpa = True
    _can_record_outputs = {
        "hidden_states": TinyModelDecoderLayer,
        "attentions": TinyModelAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=self.config.embedding_initializer_range)


@auto_docstring
class TinyModel(TinyModelPreTrainedModel):
    def __init__(self, config: TinyModelConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([TinyModelDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds + self.embed_positions(position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask, **kwargs)

        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class TinyModelForCausalLM(TinyModelPreTrainedModel, GenerationMixin):
    def __init__(self, config: TinyModelConfig):
        super().__init__(config)
        self.model = TinyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        self.post_init()
        self.generation_config.use_cache = False

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("TinyModel cannot generate from `inputs_embeds` without key/value caching.")
        return super().prepare_inputs_for_generation(input_ids, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        if use_cache:
            raise ValueError("TinyModel does not support key/value caching; set `use_cache=False`.")
        if past_key_values is not None:
            raise ValueError("TinyModel does not support `past_key_values`.")

        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :]).contiguous()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "TinyModelConfig",
    "TinyModel",
    "TinyModelForCausalLM",
    "TinyModelPreTrainedModel",
]
