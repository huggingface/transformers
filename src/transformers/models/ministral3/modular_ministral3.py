from collections.abc import Callable

import torch

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ..mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralModel,
    MistralPreTrainedModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


def _get_llama_4_attn_scale(positions_ids: torch.Tensor, beta: float, max_position_embeddings: int) -> torch.Tensor:
    scaling = 1 + beta * torch.log(1 + torch.floor(positions_ids / max_position_embeddings))
    return scaling.unsqueeze(-1)


class Ministral3Attention(MistralAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states * _get_llama_4_attn_scale(
            cache_position,
            self.config.rope_parameters.get("llama_4_scaling_beta"),
            self.config.rope_parameters.get("original_max_position_embeddings"),
        ).to(query_states.dtype)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self.config, "sliding_window", None),  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Ministral3DecoderLayer(MistralDecoderLayer):
    pass


@auto_docstring
class Ministral3PreTrainedModel(MistralPreTrainedModel):
    pass


@auto_docstring
class Ministral3Model(MistralModel):
    pass


@auto_docstring
class Ministral3ForCausalLM(MistralForCausalLM):
    pass


class Ministral3ForTokenClassification(GenericForTokenClassification, Ministral3PreTrainedModel):
    pass


class Ministral3ForSequenceClassification(GenericForSequenceClassification, Ministral3PreTrainedModel):
    pass


class Ministral3ForQuestionAnswering(GenericForQuestionAnswering, Ministral3PreTrainedModel):
    pass


__all__ = [
    "Ministral3ForCausalLM",
    "Ministral3ForQuestionAnswering",
    "Ministral3Model",
    "Ministral3PreTrainedModel",
    "Ministral3ForSequenceClassification",
    "Ministral3ForTokenClassification",
]
