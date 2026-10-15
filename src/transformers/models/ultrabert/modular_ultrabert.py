# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
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
from typing import Literal

import torch
from torch import nn

from ...activations import ACT2FN
from ...configuration_utils import strict
from ...masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from ...modeling_layers import GenericForQuestionAnswering
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..eurobert.configuration_eurobert import EuroBertConfig
from ..eurobert.modeling_eurobert import (
    EuroBertAttention,
    EuroBertDecoderLayer,
    EuroBertMLP,
    EuroBertModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llama.modeling_llama import LlamaRMSNorm
from ..modernbert.modeling_modernbert import (
    ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
    ModernBertForTokenClassification,
    ModernBertRotaryEmbedding,
)
from ..olmo3.modeling_olmo3 import Olmo3PreTrainedModel


@auto_docstring(checkpoint="dmis-lab/UltraBERT-150M")
@strict
class UltrabertConfig(EuroBertConfig):
    r"""
    sliding_window (`int`, *optional*, defaults to 128):
        Inclusive attention radius on either side of each token in sliding attention layers.
    prediction_head_size (`int`, *optional*, defaults to 768):
        Intermediate dimension of the two-layer masked language modeling prediction head.
    classifier_pooling (`str`, *optional*, defaults to `"cls"`):
        Pooling method for sequence classification. Either `"cls"` or `"mean"`.
    """

    model_type = "ultrabert"

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    use_cache: bool = False
    max_position_embeddings: int = 32768
    sliding_window: int = 128
    layer_types: list[str] | None = None
    prediction_head_size: int = 768
    classifier_dropout: float = 0.0
    classifier_pooling: Literal["cls", "mean"] = "cls"
    pretraining_tp = AttributeError()
    mask_token_id = AttributeError()

    def __post_init__(self, **kwargs):
        # BC -> existing checkpoints store attention windows per layer.
        if "attention_windows" in kwargs:
            attention_windows = kwargs.pop("attention_windows")
            self.sliding_window = next((window for window in attention_windows if window), self.sliding_window)
            if self.layer_types is None:
                self.layer_types = [
                    "sliding_attention" if window else "full_attention" for window in attention_windows
                ]
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 3 else "full_attention" for i in range(self.num_hidden_layers)
            ]

        # BC -> existing checkpoints store RoPE bases in rope_theta.
        rope_theta = kwargs.pop("rope_theta", None)
        if rope_theta is not None and self.rope_parameters is None:
            self.rope_parameters = {
                kind: {"rope_type": "default", "rope_theta": theta}
                for kind, theta in zip(self.layer_types, rope_theta)
            }
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 2_560_000.0},
            }
        super().__post_init__(**kwargs)


class UltrabertRMSNorm(LlamaRMSNorm):
    pass


class UltrabertRotaryEmbedding(ModernBertRotaryEmbedding):
    pass


class UltrabertAttention(EuroBertAttention):
    def __init__(self, config: UltrabertConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # +1 is needed because flash attention sets inclusive boundaries (see modeling_flash_attention_utils.py)
        self.sliding_window = (
            config.sliding_window + 1 if config.layer_types[layer_idx] == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Unlike EuroBERT, this has no KV cache and it uses a per-layer attention window.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class UltrabertMLP(EuroBertMLP):
    pass


class UltrabertDecoderLayer(EuroBertDecoderLayer):
    pass


class UltrabertPreTrainedModel(Olmo3PreTrainedModel):
    config: UltrabertConfig


class UltrabertModel(EuroBertModel):
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        # Unlike EuroBERT, this selects attention masks and RoPE parameters by layer type.
        mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        attention_masks = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }

        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        hidden_states = inputs_embeds
        for i, encoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_masks[layer_type],
                position_ids=position_ids,
                position_embeddings=position_embeddings[layer_type],
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class UltrabertPredictionHead(nn.Module):
    def __init__(self, config: UltrabertConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.prediction_head_size, bias=False)
        self.linear2 = nn.Linear(config.prediction_head_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN["gelu_pytorch_tanh"]
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return self.decoder(hidden_states)


class UltrabertForMaskedLM(ModernBertForMaskedLM):
    _tied_weights_keys = {"head.decoder.weight": "model.embed_tokens.weight"}

    def __init__(self, config: UltrabertConfig):
        super().__init__(config)
        self.decoder = nn.Identity()
        self.sparse_prediction = False
        self.sparse_pred_ignore_index = -100

    def get_output_embeddings(self):
        # UltraBERT doesn't normally tie embeddings, so output embeddings differ from input embeddings
        return self.head.decoder

    def set_output_embeddings(self, new_embeddings):
        # UltraBERT doesn't normally tie embeddings, so output embeddings differ from input embeddings
        self.head.decoder = new_embeddings


class UltrabertForSequenceClassification(ModernBertForSequenceClassification):
    def __init__(self, config: UltrabertConfig):
        super().__init__(config)
        self.head = nn.Identity()


class UltrabertForTokenClassification(ModernBertForTokenClassification):
    def __init__(self, config: UltrabertConfig):
        super().__init__(config)
        self.head = nn.Identity()


class UltrabertForQuestionAnswering(GenericForQuestionAnswering, UltrabertPreTrainedModel):
    pass


__all__ = [
    "UltrabertConfig",
    "UltrabertPreTrainedModel",
    "UltrabertModel",
    "UltrabertForMaskedLM",
    "UltrabertForSequenceClassification",
    "UltrabertForTokenClassification",
    "UltrabertForQuestionAnswering",
]
