# coding=utf-8
# Copyright 2021 Tel AViv University, AllenAI and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Splinter model."""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
    QuestionAnsweringModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.deprecation import deprecate_kwarg
from .configuration_splinter import SplinterConfig


logger = logging.get_logger(__name__)


class SplinterEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> tuple:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.align.modeling_align.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# Copied from transformers.models.align.modeling_align.AlignTextSelfAttention with AlignText->Splinter
class SplinterSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_dropout = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5

    @deprecate_kwarg("encoder_hidden_states", version="4.54.0")
    @deprecate_kwarg("encoder_attention_mask", version="4.54.0")
    @deprecate_kwarg("past_key_value", version="4.54.0")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

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
            head_mask=head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Splinter
class SplinterSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.align.modeling_align.AlignTextAttention with AlignText->Splinter
class SplinterAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SplinterSelfAttention(config)
        self.output = SplinterSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @deprecate_kwarg("encoder_hidden_states", version="4.54.0")
    @deprecate_kwarg("encoder_attention_mask", version="4.54.0")
    @deprecate_kwarg("past_key_value", version="4.54.0")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Splinter
class SplinterIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Splinter
class SplinterOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.align.modeling_align.AlignTextLayer with AlignText->Splinter
class SplinterLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SplinterAttention(config)
        self.intermediate = SplinterIntermediate(config)
        self.output = SplinterOutput(config)

    @deprecate_kwarg("encoder_hidden_states", version="4.54.0")
    @deprecate_kwarg("encoder_attention_mask", version="4.54.0")
    @deprecate_kwarg("past_key_value", version="4.54.0")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.align.modeling_align.AlignTextEncoder with AlignText->Splinter
class SplinterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SplinterLayer(config) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    @deprecate_kwarg("encoder_hidden_states", version="4.54.0")
    @deprecate_kwarg("encoder_attention_mask", version="4.54.0")
    @deprecate_kwarg("past_key_values", version="4.54.0")
    @deprecate_kwarg("use_cache", version="4.54.0")
    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring
class SplinterPreTrainedModel(PreTrainedModel):
    config_class = SplinterConfig
    base_model_prefix = "splinter"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@auto_docstring
class SplinterModel(SplinterPreTrainedModel):
    """
    The model is an encoder (with only self-attention) following the architecture described in [Attention is all you
    need](https://huggingface.co/papers/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = SplinterEmbeddings(config)
        self.encoder = SplinterEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @deprecate_kwarg("encoder_hidden_states", version="4.54.0")
    @deprecate_kwarg("encoder_attention_mask", version="4.54.0")
    @deprecate_kwarg("past_key_values", version="4.54.0")
    @deprecate_kwarg("use_cache", version="4.54.0")
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        token_type_ids (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SplinterFullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_act="gelu"):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense = nn.Linear(self.input_dim, self.output_dim)
        self.act_fn = ACT2FN[hidden_act]
        self.LayerNorm = nn.LayerNorm(self.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(inputs)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class QuestionAwareSpanSelectionHead(nn.Module):
    """
    Implementation of Question-Aware Span Selection (QASS) head, described in Splinter's paper:

    """

    def __init__(self, config):
        super().__init__()

        self.query_start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.query_end_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.end_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)

        self.start_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.end_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, inputs, positions):
        _, _, dim = inputs.size()
        index = positions.unsqueeze(-1).repeat(1, 1, dim)  # [batch_size, num_positions, dim]
        gathered_reps = torch.gather(inputs, dim=1, index=index)  # [batch_size, num_positions, dim]

        query_start_reps = self.query_start_transform(gathered_reps)  # [batch_size, num_positions, dim]
        query_end_reps = self.query_end_transform(gathered_reps)  # [batch_size, num_positions, dim]
        start_reps = self.start_transform(inputs)  # [batch_size, seq_length, dim]
        end_reps = self.end_transform(inputs)  # [batch_size, seq_length, dim]

        hidden_states = self.start_classifier(query_start_reps)  # [batch_size, num_positions, dim]
        start_reps = start_reps.permute(0, 2, 1)  # [batch_size, dim, seq_length]
        start_logits = torch.matmul(hidden_states, start_reps)

        hidden_states = self.end_classifier(query_end_reps)
        end_reps = end_reps.permute(0, 2, 1)
        end_logits = torch.matmul(hidden_states, end_reps)

        return start_logits, end_logits


@auto_docstring
class SplinterForQuestionAnswering(SplinterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.splinter = SplinterModel(config)
        self.splinter_qass = QuestionAwareSpanSelectionHead(config)
        self.question_token_id = config.question_token_id

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        question_positions: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, QuestionAnsweringModelOutput]:
        r"""
        token_type_ids (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `batch_size, sequence_length`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        question_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            The positions of all question tokens. If given, start_logits and end_logits will be of shape `(batch_size,
            num_questions, sequence_length)`. If None, the first question token in each sequence in the batch will be
            the only one for which start_logits and end_logits are calculated and they will be of shape `(batch_size,
            sequence_length)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        question_positions_were_none = False
        if question_positions is None:
            if input_ids is not None:
                question_position_for_each_example = torch.argmax(
                    (torch.eq(input_ids, self.question_token_id)).int(), dim=-1
                )
            else:
                question_position_for_each_example = torch.zeros(
                    inputs_embeds.size(0), dtype=torch.long, layout=inputs_embeds.layout, device=inputs_embeds.device
                )
            question_positions = question_position_for_each_example.unsqueeze(-1)
            question_positions_were_none = True

        outputs = self.splinter(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        start_logits, end_logits = self.splinter_qass(sequence_output, question_positions)

        if question_positions_were_none:
            start_logits, end_logits = start_logits.squeeze(1), end_logits.squeeze(1)

        if attention_mask is not None:
            start_logits = start_logits + (1 - attention_mask) * torch.finfo(start_logits.dtype).min
            end_logits = end_logits + (1 - attention_mask) * torch.finfo(end_logits.dtype).min

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of Splinter as a span selection model.
    """
)
class SplinterForPreTrainingOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when start and end positions are provided):
        Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    start_logits (`torch.FloatTensor` of shape `(batch_size, num_questions, sequence_length)`):
        Span-start scores (before SoftMax).
    end_logits (`torch.FloatTensor` of shape `(batch_size, num_questions, sequence_length)`):
        Span-end scores (before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: Optional[torch.FloatTensor] = None
    end_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@auto_docstring(
    custom_intro="""
    Splinter Model for the recurring span selection task as done during the pretraining. The difference to the QA task
    is that we do not have a question, but multiple question tokens that replace the occurrences of recurring spans
    instead.
    """
)
class SplinterForPreTraining(SplinterPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.splinter = SplinterModel(config)
        self.splinter_qass = QuestionAwareSpanSelectionHead(config)
        self.question_token_id = config.question_token_id

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        question_positions: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, SplinterForPreTrainingOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, num_questions, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        token_type_ids (`torch.LongTensor` of shape `batch_size, num_questions, sequence_length`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `batch_size, num_questions, sequence_length`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_questions, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        start_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        question_positions (`torch.LongTensor` of shape `(batch_size, num_questions)`, *optional*):
            The positions of all question tokens. If given, start_logits and end_logits will be of shape `(batch_size,
            num_questions, sequence_length)`. If None, the first question token in each sequence in the batch will be
            the only one for which start_logits and end_logits are calculated and they will be of shape `(batch_size,
            sequence_length)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if question_positions is None and start_positions is not None and end_positions is not None:
            raise TypeError("question_positions must be specified in order to calculate the loss")

        elif question_positions is None and input_ids is None:
            raise TypeError("question_positions must be specified when input_embeds is used")

        elif question_positions is None:
            question_positions = self._prepare_question_positions(input_ids)

        outputs = self.splinter(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        batch_size, sequence_length, dim = sequence_output.size()
        # [batch_size, num_questions, sequence_length]
        start_logits, end_logits = self.splinter_qass(sequence_output, question_positions)

        num_questions = question_positions.size(1)
        if attention_mask is not None:
            attention_mask_for_each_question = attention_mask.unsqueeze(1).expand(
                batch_size, num_questions, sequence_length
            )
            start_logits = start_logits + (1 - attention_mask_for_each_question) * torch.finfo(start_logits.dtype).min
            end_logits = end_logits + (1 - attention_mask_for_each_question) * torch.finfo(end_logits.dtype).min

        total_loss = None
        # [batch_size, num_questions, sequence_length]
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            start_positions.clamp_(0, max(0, sequence_length - 1))
            end_positions.clamp_(0, max(0, sequence_length - 1))

            # Ignore zero positions in the loss. Splinter never predicts zero
            # during pretraining and zero is used for padding question
            # tokens as well as for start and end positions of padded
            # question tokens.
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            start_loss = loss_fct(
                start_logits.view(batch_size * num_questions, sequence_length),
                start_positions.view(batch_size * num_questions),
            )
            end_loss = loss_fct(
                end_logits.view(batch_size * num_questions, sequence_length),
                end_positions.view(batch_size * num_questions),
            )
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return SplinterForPreTrainingOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _prepare_question_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows, flat_positions = torch.where(input_ids == self.config.question_token_id)
        num_questions = torch.bincount(rows)
        positions = torch.full(
            (input_ids.size(0), num_questions.max()),
            self.config.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        cols = torch.cat([torch.arange(n) for n in num_questions])
        positions[rows, cols] = flat_positions
        return positions


__all__ = [
    "SplinterForQuestionAnswering",
    "SplinterForPreTraining",
    "SplinterLayer",
    "SplinterModel",
    "SplinterPreTrainedModel",
]
