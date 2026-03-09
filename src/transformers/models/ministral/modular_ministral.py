import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..mistral.configuration_mistral import MistralConfig
from ..qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2MLP,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)


@auto_docstring(checkpoint="mistralai/Ministral-8B-Instruct-2410")
class MinistralConfig(MistralConfig, PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import MinistralModel, MinistralConfig

    >>> # Initializing a Ministral 8B style configuration
    >>> configuration = MinistralConfig()

    >>> # Initializing a model from the Ministral 8B style configuration
    >>> model = MinistralModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ministral"

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 14336,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096 * 32,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | None = None,
        sliding_window: int | None = 4096,
        attention_dropout: float | None = 0.0,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.head_dim = head_dim

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.sliding_window is not None else "full_attention"
            ] * num_hidden_layers

        self.rope_parameters = rope_parameters

        PreTrainedConfig.__init__(self, **kwargs)


class MinistralMLP(Qwen2MLP):
    pass


class MinistralAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Match Mistral: q/k/v do not have bias
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)


class MinistralRMSNorm(Qwen2RMSNorm):
    pass


class MinistralDecoderLayer(Qwen2DecoderLayer):
    pass


class MinistralPreTrainedModel(Qwen2PreTrainedModel):
    pass


class MinistralRotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class MinistralModel(Qwen2Model):
    def __init__(self, config: MinistralConfig):
        super().__init__(config)
        del self.has_sliding_layers

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class MinistralForCausalLM(Qwen2ForCausalLM):
    pass


class MinistralForSequenceClassification(Qwen2ForSequenceClassification):
    pass


class MinistralForTokenClassification(Qwen2ForTokenClassification):
    pass


class MinistralForQuestionAnswering(Qwen2ForQuestionAnswering):
    pass


__all__ = [
    "MinistralConfig",
    "MinistralPreTrainedModel",
    "MinistralModel",
    "MinistralForCausalLM",
    "MinistralForSequenceClassification",
    "MinistralForTokenClassification",
    "MinistralForQuestionAnswering",
]
