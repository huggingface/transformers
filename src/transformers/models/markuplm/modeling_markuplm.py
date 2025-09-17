# coding=utf-8
# Copyright 2022 Microsoft Research Asia and the HuggingFace Inc. team.
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
"""PyTorch MarkupLM model."""

import os
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import auto_docstring, can_return_tuple, logging
from .configuration_markuplm import MarkupLMConfig


logger = logging.get_logger(__name__)


class XPathEmbeddings(nn.Module):
    """Construct the embeddings from xpath tags and subscripts.

    We drop tree-id in this version, as its info can be covered by xpath.
    """

    def __init__(self, config):
        super().__init__()
        self.max_depth = config.max_depth

        self.xpath_unitseq2_embeddings = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.activation = nn.ReLU()
        self.xpath_unitseq2_inner = nn.Linear(config.xpath_unit_hidden_size * self.max_depth, 4 * config.hidden_size)
        self.inner2emb = nn.Linear(4 * config.hidden_size, config.hidden_size)

        self.xpath_tag_sub_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.max_xpath_tag_unit_embeddings, config.xpath_unit_hidden_size)
                for _ in range(self.max_depth)
            ]
        )

        self.xpath_subs_sub_embeddings = nn.ModuleList(
            [
                nn.Embedding(config.max_xpath_subs_unit_embeddings, config.xpath_unit_hidden_size)
                for _ in range(self.max_depth)
            ]
        )

    def forward(self, xpath_tags_seq=None, xpath_subs_seq=None):
        xpath_tags_embeddings = []
        xpath_subs_embeddings = []

        for i in range(self.max_depth):
            xpath_tags_embeddings.append(self.xpath_tag_sub_embeddings[i](xpath_tags_seq[:, :, i]))
            xpath_subs_embeddings.append(self.xpath_subs_sub_embeddings[i](xpath_subs_seq[:, :, i]))

        xpath_tags_embeddings = torch.cat(xpath_tags_embeddings, dim=-1)
        xpath_subs_embeddings = torch.cat(xpath_subs_embeddings, dim=-1)

        xpath_embeddings = xpath_tags_embeddings + xpath_subs_embeddings

        xpath_embeddings = self.inner2emb(self.dropout(self.activation(self.xpath_unitseq2_inner(xpath_embeddings))))

        return xpath_embeddings


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class MarkupLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.max_depth = config.max_depth

        self.xpath_embeddings = XPathEmbeddings(config)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings.create_position_ids_from_inputs_embeds
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        xpath_tags_seq=None,
        xpath_subs_seq=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # prepare xpath seq
        if xpath_tags_seq is None:
            xpath_tags_seq = self.config.tag_pad_id * torch.ones(
                tuple(list(input_shape) + [self.max_depth]), dtype=torch.long, device=device
            )
        if xpath_subs_seq is None:
            xpath_subs_seq = self.config.subs_pad_id * torch.ones(
                tuple(list(input_shape) + [self.max_depth]), dtype=torch.long, device=device
            )

        words_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        xpath_embeddings = self.xpath_embeddings(xpath_tags_seq, xpath_subs_seq)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings + xpath_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->MarkupLM
class MarkupLMSelfOutput(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class MarkupLMIntermediate(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->MarkupLM
class MarkupLMOutput(nn.Module):
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


# Copied from transformers.models.bert.modeling_bert.BertPooler
class MarkupLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->MarkupLM
class MarkupLMPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->MarkupLM
class MarkupLMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = MarkupLMPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->MarkupLM
class MarkupLMOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = MarkupLMLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


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


# Copied from transformers.models.align.modeling_align.AlignTextSelfAttention with AlignText->MarkupLM
class MarkupLMSelfAttention(nn.Module):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
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


# Copied from transformers.models.align.modeling_align.AlignTextAttention with AlignText->MarkupLM
class MarkupLMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MarkupLMSelfAttention(config)
        self.output = MarkupLMSelfOutput(config)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
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


# Copied from transformers.models.align.modeling_align.AlignTextLayer with AlignText->MarkupLM
class MarkupLMLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MarkupLMAttention(config)
        self.intermediate = MarkupLMIntermediate(config)
        self.output = MarkupLMOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
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


# Copied from transformers.models.align.modeling_align.AlignTextEncoder with AlignText->MarkupLM
class MarkupLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MarkupLMLayer(config) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    @can_return_tuple
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
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
class MarkupLMPreTrainedModel(PreTrainedModel):
    config: MarkupLMConfig
    base_model_prefix = "markuplm"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights with Bert->MarkupLM
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
        elif isinstance(module, MarkupLMLMPredictionHead):
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


@auto_docstring
class MarkupLMModel(MarkupLMPreTrainedModel):
    # Copied from transformers.models.clap.modeling_clap.ClapTextModel.__init__ with ClapText->MarkupLM
    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = MarkupLMEmbeddings(config)
        self.encoder = MarkupLMEncoder(config)

        self.pooler = MarkupLMPooler(config) if add_pooling_layer else None

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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        xpath_tags_seq: Optional[torch.LongTensor] = None,
        xpath_subs_seq: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithPooling]:
        r"""
        xpath_tags_seq (`torch.LongTensor` of shape `(batch_size, sequence_length, config.max_depth)`, *optional*):
            Tag IDs for each token in the input sequence, padded up to config.max_depth.
        xpath_subs_seq (`torch.LongTensor` of shape `(batch_size, sequence_length, config.max_depth)`, *optional*):
            Subscript IDs for each token in the input sequence, padded up to config.max_depth.

        Examples:

        ```python
        >>> from transformers import AutoProcessor, MarkupLMModel

        >>> processor = AutoProcessor.from_pretrained("microsoft/markuplm-base")
        >>> model = MarkupLMModel.from_pretrained("microsoft/markuplm-base")

        >>> html_string = "<html> <head> <title>Page Title</title> </head> </html>"

        >>> encoding = processor(html_string, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 4, 768]
        ```"""
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

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring
class MarkupLMForQuestionAnswering(MarkupLMPreTrainedModel):
    # Copied from transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ with bert->markuplm, Bert->MarkupLM
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        xpath_tags_seq: Optional[torch.Tensor] = None,
        xpath_subs_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        xpath_tags_seq (`torch.LongTensor` of shape `(batch_size, sequence_length, config.max_depth)`, *optional*):
            Tag IDs for each token in the input sequence, padded up to config.max_depth.
        xpath_subs_seq (`torch.LongTensor` of shape `(batch_size, sequence_length, config.max_depth)`, *optional*):
            Subscript IDs for each token in the input sequence, padded up to config.max_depth.

        Examples:

        ```python
        >>> from transformers import AutoProcessor, MarkupLMForQuestionAnswering
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/markuplm-base-finetuned-websrc")
        >>> model = MarkupLMForQuestionAnswering.from_pretrained("microsoft/markuplm-base-finetuned-websrc")

        >>> html_string = "<html> <head> <title>My name is Niels</title> </head> </html>"
        >>> question = "What's his name?"

        >>> encoding = processor(html_string, questions=question, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**encoding)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
        >>> processor.decode(predict_answer_tokens).strip()
        'Niels'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.markuplm(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

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

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    MarkupLM Model with a `token_classification` head on top.
    """
)
class MarkupLMForTokenClassification(MarkupLMPreTrainedModel):
    # Copied from transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ with bert->markuplm, Bert->MarkupLM
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.markuplm = MarkupLMModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        xpath_tags_seq: Optional[torch.Tensor] = None,
        xpath_subs_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        xpath_tags_seq (`torch.LongTensor` of shape `(batch_size, sequence_length, config.max_depth)`, *optional*):
            Tag IDs for each token in the input sequence, padded up to config.max_depth.
        xpath_subs_seq (`torch.LongTensor` of shape `(batch_size, sequence_length, config.max_depth)`, *optional*):
            Subscript IDs for each token in the input sequence, padded up to config.max_depth.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForTokenClassification
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/markuplm-base")
        >>> processor.parse_html = False
        >>> model = AutoModelForTokenClassification.from_pretrained("microsoft/markuplm-base", num_labels=7)

        >>> nodes = ["hello", "world"]
        >>> xpaths = ["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"]
        >>> node_labels = [1, 2]
        >>> encoding = processor(nodes=nodes, xpaths=xpaths, node_labels=node_labels, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**encoding)

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.markuplm(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = outputs[0]
        prediction_scores = self.classifier(sequence_output)  # (batch_size, seq_length, node_type_size)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, self.config.num_labels),
                labels.view(-1),
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    MarkupLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """
)
class MarkupLMForSequenceClassification(MarkupLMPreTrainedModel):
    # Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ with bert->markuplm, Bert->MarkupLM
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.markuplm = MarkupLMModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        xpath_tags_seq: Optional[torch.Tensor] = None,
        xpath_subs_seq: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        xpath_tags_seq (`torch.LongTensor` of shape `(batch_size, sequence_length, config.max_depth)`, *optional*):
            Tag IDs for each token in the input sequence, padded up to config.max_depth.
        xpath_subs_seq (`torch.LongTensor` of shape `(batch_size, sequence_length, config.max_depth)`, *optional*):
            Subscript IDs for each token in the input sequence, padded up to config.max_depth.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForSequenceClassification
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/markuplm-base")
        >>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/markuplm-base", num_labels=7)

        >>> html_string = "<html> <head> <title>Page Title</title> </head> </html>"
        >>> encoding = processor(html_string, return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**encoding)

        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.markuplm(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "MarkupLMForQuestionAnswering",
    "MarkupLMForSequenceClassification",
    "MarkupLMForTokenClassification",
    "MarkupLMModel",
    "MarkupLMPreTrainedModel",
]
