# Copyright 2022 The HuggingFace and Meta Team.  All rights reserved.
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
import torch.nn as nn


class BertLayerFast(nn.Module):
    def __init__(self, bert_layer):
        r"""
        A simple conversion of the BERTLayer to its `Fast` implementation.

        Args:
            bert_layer (`torch.nn.Module`):
                The original BERT Layer where the weights needs to be retrieved.
        """
        super().__init__()
        # In_proj layer
        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.self.query.weight,
                    bert_layer.attention.self.key.weight,
                    bert_layer.attention.self.value.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    bert_layer.attention.self.query.bias,
                    bert_layer.attention.self.key.bias,
                    bert_layer.attention.self.value.bias,
                ]
            )
        )

        # Out proj layer
        self.out_proj_weight = bert_layer.attention.output.dense.weight
        self.out_proj_bias = bert_layer.attention.output.dense.bias

        # Linear layer 1
        self.linear1_weight = bert_layer.intermediate.dense.weight
        self.linear1_bias = bert_layer.intermediate.dense.bias

        # Linear layer 2
        self.linear2_weight = bert_layer.output.dense.weight
        self.linear2_bias = bert_layer.output.dense.bias

        # Layer norm 1
        self.norm1_eps = bert_layer.attention.output.LayerNorm.eps
        self.norm1_weight = bert_layer.attention.output.LayerNorm.weight
        self.norm1_bias = bert_layer.attention.output.LayerNorm.bias

        # Layer norm 2
        self.norm2_weight = bert_layer.output.LayerNorm.weight
        self.norm2_bias = bert_layer.output.LayerNorm.bias

        # Model hyper parameters
        self.num_heads = bert_layer.attention.self.num_attention_heads
        self.embed_dim = bert_layer.attention.self.all_head_size

        del bert_layer

        # Last step: set the last layer to `False` -> this will be set to `True` when converting the model
        self.is_last_layer = False

    def forward(self, hidden_states, attention_mask, *_):
        r"""
        This is just a wrapper around the forward function proposed in:
        https://github.com/huggingface/transformers/pull/19553
        """
        if hidden_states.is_nested:
            attention_mask = None

        if attention_mask is not None:
            # attention mask comes in with values 0 and -inf. we convert to torch.nn.TransformerEncoder style bool mask
            # 0->false->keep this token -inf->true->mask this token
            attention_mask = attention_mask.bool()
            attention_mask = torch.reshape(attention_mask, (attention_mask.shape[0], attention_mask.shape[-1]))
            seqlen = attention_mask.shape[1]
            lengths = torch.sum(~attention_mask, 1)
            if not all([l == seqlen for l in lengths]):
                hidden_states = torch._nested_tensor_from_mask(hidden_states, ~attention_mask)
            attention_mask = None

        hidden_states = torch._transformer_encoder_layer_fwd(
            hidden_states,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.out_proj_weight,
            self.out_proj_bias,
            True,  # TODO use_gelu. make it not hardcoded
            False,  # norm_first, currently not supported
            self.norm1_eps,
            self.norm1_weight,
            self.norm1_bias,
            self.norm2_weight,
            self.norm2_bias,
            self.linear1_weight,
            self.linear1_bias,
            self.linear2_weight,
            self.linear2_bias,
            attention_mask,  # TODO fix this
        )
        if hidden_states.is_nested and self.is_last_layer:
            hidden_states = hidden_states.to_padded_tensor(0.0)
        return (hidden_states,)

    # return (nested_hidden_states,)
