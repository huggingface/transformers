# Copyright 2026 The Jina-AI and HuggingFace Inc. teams. All rights reserved.
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
from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import CrossEntropyLoss

from ...integrations import use_kernelized_func
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..clip.modeling_clip import CLIPMLP
from ..gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from ..llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding, apply_rotary_pos_emb
from ..xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from ..xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaEmbeddings,
    XLMRobertaForMaskedLM,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaLMHead,
    XLMRobertaModel,
    XLMRobertaPooler,
    XLMRobertaPreTrainedModel,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="jinaai/jina-embeddings-v3-hf")
@strict
class JinaEmbeddingsV3Config(XLMRobertaConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import JinaEmbeddingsV3Config, JinaEmbeddingsV3Model

    >>> # Initializing a Jina-Embeddings-V3 jinaai/jina-embeddings-v3-hf style configuration
    >>> configuration = JinaEmbeddingsV3Config()

    >>> # Initializing a model (with random weights) from the jinaai/jina-embeddings-v3-hf style configuration
    >>> model = JinaEmbeddingsV3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "jina_embeddings_v3"
    default_theta = 20000.0

    vocab_size: int = 250002
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    max_position_embeddings: int = 8194
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-5
    rope_parameters: RopeParameters | dict | None = None

    add_cross_attention = AttributeError()
    is_decoder = AttributeError()


class JinaEmbeddingsV3Embeddings(XLMRobertaEmbeddings):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__(config)

        del self.padding_idx
        del self.position_embeddings

    def create_position_ids_from_inputs_embeds():
        raise AttributeError("Not needed for JinaEmbeddingsV3")

    def create_position_ids_from_input_ids():
        raise AttributeError("Not needed for JinaEmbeddingsV3")

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        embeddings = inputs_embeds
        if inputs_embeds is None:
            embeddings = self.word_embeddings(input_ids)

        input_shape = embeddings.shape[:-1]
        device = embeddings.device

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # NOTE: We assume either pos ids to have bsz == 1 (broadcastable) or bsz == effective bsz (input_shape[0])
                buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
                buffered_token_type_ids = torch.gather(buffered_token_type_ids, dim=1, index=position_ids)
                token_type_ids = buffered_token_type_ids.expand(*input_shape)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class JinaEmbeddingsV3RotaryEmbedding(LlamaRotaryEmbedding):
    pass


@use_kernelized_func(apply_rotary_pos_emb)
class JinaEmbeddingsV3Attention(LlamaAttention):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__(config)
        self.is_causal = False
        self.attention_dropout = config.attention_probs_dropout_prob

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

        del self.layer_idx
        del self.num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
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
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class JinaEmbeddingsV3MLP(CLIPMLP):
    pass


class JinaEmbeddingsV3Layer(GPTNeoXLayer):
    def __init__(self, config: JinaEmbeddingsV3Config):
        super().__init__(config)
        self.self_attn = JinaEmbeddingsV3Attention(config=config)

        self.post_attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.post_mlp_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout_prob)

        del self.use_parallel_residual
        del self.input_layernorm
        del self.attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        attention_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + self.post_attention_dropout(attention_output)
        hidden_states = self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_mlp_dropout(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        return hidden_states


class JinaEmbeddingsV3Pooler(XLMRobertaPooler):
    pass


class JinaEmbeddingsV3PreTrainedModel(XLMRobertaPreTrainedModel):
    _can_record_outputs = {
        "hidden_states": JinaEmbeddingsV3Layer,
        "attentions": JinaEmbeddingsV3Attention,
    }


@auto_docstring
class JinaEmbeddingsV3Model(XLMRobertaModel):
    def __init__(self, config: JinaEmbeddingsV3Config, add_pooling_layer=True):
        super().__init__(config)
        self.rotary_emb = JinaEmbeddingsV3RotaryEmbedding(config)
        self.layers = nn.ModuleList([JinaEmbeddingsV3Layer(config) for _ in range(config.num_hidden_layers)])
        del self.encoder

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling | tuple:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            seq_length = input_ids.shape[1]
            device = input_ids.device
        else:
            seq_length = inputs_embeds.shape[1]
            device = inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)[None, :]

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=embedding_output,
            attention_mask=attention_mask,
        )

        hidden_states = embedding_output
        position_embeddings = self.rotary_emb(embedding_output, position_ids)

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        sequence_output = hidden_states
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )

    def _create_attention_masks(self):
        raise AttributeError("Not needed for JinaEmbeddingsV3")


class JinaEmbeddingsV3LMHead(XLMRobertaLMHead):
    pass


class JinaEmbeddingsV3ForMaskedLM(XLMRobertaForMaskedLM):
    def __init__(self, config):
        JinaEmbeddingsV3PreTrainedModel.__init__(self, config=config)

        self.lm_head = JinaEmbeddingsV3LMHead(config)
        self.roberta = JinaEmbeddingsV3Model(config, add_pooling_layer=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        r"""
        token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.

            [What are token type IDs?](../glossary#token-type-ids)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        sequence_output = outputs[0]

        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class JinaEmbeddingsV3ForSequenceClassification(XLMRobertaForSequenceClassification):
    pass


class JinaEmbeddingsV3ForTokenClassification(XLMRobertaForTokenClassification):
    pass


class JinaEmbeddingsV3ForQuestionAnswering(XLMRobertaForQuestionAnswering):
    pass


__all__ = [
    "JinaEmbeddingsV3Config",
    "JinaEmbeddingsV3PreTrainedModel",
    "JinaEmbeddingsV3Model",
    "JinaEmbeddingsV3ForMaskedLM",
    "JinaEmbeddingsV3ForSequenceClassification",
    "JinaEmbeddingsV3ForTokenClassification",
    "JinaEmbeddingsV3ForQuestionAnswering",
    "JinaEmbeddingsV3Layer",
]
