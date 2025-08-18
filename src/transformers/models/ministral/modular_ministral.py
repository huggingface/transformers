"""PyTorch Ministral model"""

from torch import nn

from ..mistral.configuration_mistral import MistralConfig
from ..qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)


class MinistralConfig(MistralConfig):
    model_type = "ministral"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0,
        layer_types=None,
        **kwargs,
    ):
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
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            sliding_window=sliding_window,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        self.layer_types = layer_types

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if self.sliding_window is not None else "full_attention"
            ] * num_hidden_layers


class MinistralRMSNorm(Qwen2RMSNorm):
    pass


class MinistralRotaryEmbedding(Qwen2RotaryEmbedding):
    pass


class MinistralAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Match Mistral: q/k/v do not have bias
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)


class MinistralPreTrainedModel(Qwen2PreTrainedModel):
    pass


class MinistralModel(Qwen2Model):
    pass


class MinistralForCausalLM(Qwen2ForCausalLM):
    pass


class MinistralForSequenceClassification(Qwen2ForSequenceClassification):
    pass


class MinistralForTokenClassification(Qwen2ForTokenClassification):
    pass


class MinistralForQuestionAnswering(Qwen2ForQuestionAnswering):
    pass


__all__ = [
    "MinistralPreTrainedModel",
    "MinistralModel",
    "MinistralForCausalLM",
    "MinistralForSequenceClassification",
    "MinistralForTokenClassification",
    "MinistralForQuestionAnswering",
    "MinistralConfig",
]
