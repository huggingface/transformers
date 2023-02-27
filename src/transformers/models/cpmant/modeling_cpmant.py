# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch CPMAnt"""


import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CPMAntConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "cpm-ant-10b"
_CONFIG_FOR_DOC = "CPMAntConfig"
_TOKENIZER_FOR_DOC = "CPMAntTokenizer"

CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openbmb/cpm-ant-10b",
    # See all CPMAnt models at https://huggingface.co/models?filter=cpmant
]


# Adapted from Bert
def load_tf_weights_in_cpmant(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch"""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise AssertionError("Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class CPMAntLayerNorm(nn.Module):
    """
    We use Root Mean Square (RMS) Layer Normalization, please see https://arxiv.org/abs/1910.07467 for details."
    """

    def __init__(
        self,
        config: CPMAntConfig,
        init_var: float = 1.0,
    ):
        super().__init__()

        self.eps = config.eps
        self.dim_norm = config.dim_model
        self.weight = torch.nn.parameter.Parameter(torch.full((config.dim_model,), init_var))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        if hidden_states.size(-1) != self.dim_norm:
            raise AssertionError("hidden_states.size(-1) != self.dim_norm")
        return rms_layernorm(hidden_states, self.weight, self.eps)


class CPMAntAttention(nn.Module):
    def __init__(self, config: CPMAntConfig):
        super().__init__()
        self.dim_model = config.dim_model
        self.num_heads = config.num_heads
        self.dim_head = config.dim_head

        self.project_q = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.project_k = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.project_v = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)

        self.attention_out = nn.Linear(self.num_heads * self.dim_head, self.dim_model, bias=False)

        self.softmax = torch.nn.Softmax(dim=-1)

        if config.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ):
        """
        Args:
            hidden_q (`torch.Tensor`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            hidden_kv (`torch.Tensor` of shape `(batch, len_k, dim_model)`)):
                Tensor *key_value* and *query* of shape `(batch, len_k, dim_model)`
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            past_key_values (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)

        query = self.project_q(hidden_q)
        key = self.project_k(hidden_kv)
        value = self.project_v(hidden_kv)

        query = query.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        if past_key_values is not None:
            key = torch.cat([past_key_values[0], key], dim=-2)
            value = torch.cat([past_key_values[1], value], dim=-2)
            len_k = key.size(-2)

        # (batch_size, num_heads, len_q, dim_head) @ (batch_size, num_heads, dim_head, len_k) -> (batch_size, num_heads, len_q, len_k)
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim_head)
        score = score + position_bias

        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == torch.tensor(False),
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
        )
        score = self.softmax(score)

        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == torch.tensor(False),
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
        )
        if output_attentions:
            attn_weights = score
        else:
            attn_weights = None

        if self.dropout is not None:
            score = self.dropout(score)

        # (batch_size, num_heads, len_q, len_k) @ (batch_size, num_heads, len_k, dim_head) -> (batch_size, num_heads, len_q, dim_head)
        score = torch.matmul(score, value)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3)
        score = score.contiguous().view(batch_size, len_q, self.num_heads * self.dim_head)

        score = self.attention_out(score)

        if use_cache:
            return score, attn_weights, (key, value)

        return score, attn_weights


class CPMAntSelfAttentionBlock(nn.Module):
    def __init__(self, config: CPMAntConfig):
        super().__init__()
        self.layernorm_before_attention = CPMAntLayerNorm(config)
        self.self_attention = CPMAntAttention(config)
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        outputs = self.layernorm_before_attention(hidden_states)
        outputs = self.self_attention(outputs, outputs, attention_mask, position_bias, output_attentions, past_key_values, use_cache)
        if use_cache:
            outputs, attn_weights, current_key_value = outputs
        else:
            outputs, attn_weights = outputs

        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = hidden_states + outputs

        if use_cache:
            return (hidden_states, attn_weights, current_key_value)

        return hidden_states, attn_weights


class DenseGatedACT(nn.Module):
    def __init__(self, config: CPMAntConfig):
        super().__init__()
        self.w_0 = nn.Linear(config.dim_model, config.dim_ff, bias=False)
        self.w_1 = nn.Linear(config.dim_model, config.dim_ff, bias=False)
        self.act = torch.nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        """Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        gate_score = self.act(self.w_0(hidden_states))
        hidden_states = self.w_1(hidden_states)

        hidden_states = gate_score * hidden_states
        return hidden_states


class CPMAntFeedForward(nn.Module):
    def __init__(self, config: CPMAntConfig):
        super().__init__()
        self.w_in = DenseGatedACT(config)
        if config.dropout_p is not None:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

        self.w_out = nn.Linear(config.dim_ff, config.dim_model, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        hidden_states = self.w_in(hidden_states)

        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_out(hidden_states)

        return hidden_states


class CPMAntFFNBlock(nn.Module):
    def __init__(self, config: CPMAntConfig):
        super().__init__()
        self.layernorm_before_ffn = CPMAntLayerNorm(config)
        self.ffn = CPMAntFeedForward(config)
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Hidden states before feed forward layer.
        """
        ln_outputs = self.layernorm_before_ffn(hidden_states)
        outputs = self.ffn(ln_outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = hidden_states + outputs
        return hidden_states


class CPMAntTransformerBlock(nn.Module):
    def __init__(self, config: CPMAntConfig):
        super().__init__()
        self.self_att = CPMAntSelfAttentionBlock(config)
        self.ffn = CPMAntFFNBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input to the layer of shape `(batch, seq_len, dim_model)`
            attention_mask (`torch.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_len, seq_len)`
            position_bias (`torch.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_len, seq_len)`
            past_key_values (`Tuple[torch.Tensor, torch.Tensor])`, *optional*):
                Cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        current_key_value = None
        hidden_states = self.self_att(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            hidden_states, attn_weights, current_key_value = hidden_states
        else:
            hidden_states, attn_weights = hidden_states
        hidden_states = self.ffn(hidden_states)

        if use_cache:
            return (hidden_states, attn_weights, current_key_value)

        return hidden_states, attn_weights


class CPMAntEncoder(nn.Module):
    def __init__(self, config: CPMAntConfig):
        super().__init__()
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList([CPMAntTransformerBlock(config) for ith in range(self.num_layers)])

        self.output_layernorm = CPMAntLayerNorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input to the layer of shape `(batch, seq_len, dim_model)`
            attention_mask (`torch.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_len, seq_len)`
            position_bias (`torch.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_len, seq_len)`
            past_key_values (`Tuple[torch.Tensor, torch.Tensor])`, *optional*):
                Cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        if not use_cache:
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                layer_outputs = layer(hidden_states, attention_mask, position_bias, output_attentions, None, use_cache)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
            hidden_states = self.output_layernorm(hidden_states)
            if output_attentions:
                all_hidden_states += (hidden_states,)
            return hidden_states, None, all_hidden_states, all_self_attns

        current_key_values = []
        for i, module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            module_outputs = module(
                hidden_states,
                attention_mask,
                position_bias,
                output_attentions=output_attentions,
                past_key_values=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
            )
            hidden_states = module_outputs[0]
            if output_attentions:
                all_self_attns += (module_outputs[1],)
            current_key_values.append(module_outputs[-1])
        hidden_states = self.output_layernorm(hidden_states)
        if output_attentions:
            all_hidden_states += (hidden_states,)
        return hidden_states, current_key_values, all_hidden_states, all_self_attns


# Copied from BertIntermediate
class CPMAntIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CPMAntSegmentPositionEmbedding(nn.Module):
    def __init__(
        self,
        num_heads: int = 32,
        num_segments: int = 32,
        num_buckets: int = 512,
        max_distance: int = 2048,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.num_segments = num_segments

        self.relative_attention_bias = torch.nn.parameter.Parameter(
            torch.empty(num_segments * num_segments + num_buckets, num_heads)
        )

    def forward(
        self,
        key_pos: torch.Tensor,
        query_pos: torch.Tensor,
        key_segment: torch.Tensor,
        query_segment: torch.Tensor,
    ):
        with torch.no_grad():
            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)

            assert key_pos.size(0) == query_pos.size(0), "key_pos.size(0) != query_pos.size(0)"
            assert keylen == key_segment.size(1) and querylen == query_segment.size(1), "keylen != key_segment.size(1) or querylen != query_segment.size(1)"

            key_pos = key_pos.view(batch, -1, keylen)
            query_pos = query_pos.view(batch, querylen, -1)
            key_segment = key_segment.view(batch, -1, keylen)
            query_segment = query_segment.view(batch, querylen, -1)

            relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
            relative_position_bucket = relative_position_bucket + self.num_buckets  # 与相对位置编码区间不重叠

            # (batch, len_q, len_k)
            absolute_position_bucket = self._position_bucket(
                torch.arange(keylen, dtype=torch.int32, device=relative_position_bucket.device)[None, :]
                - torch.arange(querylen, dtype=torch.int32, device=relative_position_bucket.device)[:, None],
                bidirectional=self.bidirectional,
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            relative_position_bucket = torch.where(
                (key_segment == query_segment),
                absolute_position_bucket[None, :, :],
                relative_position_bucket,
            )

        # (batch, len_q, len_k, num_heads)
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        # (batch, num_heads, len_q, len_k)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _segment_relative_position_bucket(self, query_segment, key_segment):
        return query_segment * self.num_segments + key_segment

    def _position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.int32)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
        return relative_buckets


# Copied from BertOutput
class CPMAntOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CPMAntPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CPMAntConfig
    load_tf_weights = load_tf_weights_in_cpmant
    base_model_prefix = "cpmant"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CPMAntEncoder):
            module.gradient_checkpointing = value


CPMANT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters
        config ([`~CPMAntConfig`]): Model configuration class with all the parameters of the
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CPMANT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`CPMAntTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        length (`torch.Tensor` of shape `(batch)`, *optional*):
            The: length of input tokens.
        context (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
            The Boolean value determines whether the model makes a prediction for that position
        position (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
            Indices of position of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        segment (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
            A sequence of tokens that is processed together as a unit.
        span (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
            A contiguous sequence of tokens within the input text.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare CPMAnt Model outputting raw hidden-states without any specific head on top.",
    CPMANT_START_DOCSTRING,
)
class CPMAntModel(CPMAntPreTrainedModel):
    def __init__(self, config: CPMAntConfig):
        super().__init__(config)
        self.encoder = CPMAntEncoder(config)
        self.segment_embedding = nn.Embedding(config.segment_types, config.dim_model)
        self.input_embedding = nn.Embedding(
            config.vocab_size + config.prompt_types * config.prompt_length, config.dim_model
        )
        self.position_bias = CPMAntSegmentPositionEmbedding()
        self.prompt_length = config.prompt_length
        self.vocab_size = config.vocab_size

    def get_input_embeddings(self):
        embeddings = {
            "segment": self.segment_embedding,
            "input_ids": self.input_embedding,
            "position": self.position_bias,
        }
        return embeddings

    def set_input_embeddings(self, embeddings, **kwargs):
        self.segment_embedding = embeddings["segment"]
        self.input_embedding = embeddings["input_ids"]
        self.position_bias = embeddings["position"]

    def _prepare_attention_mask(self, input_ids, span, context, length):
        batch = input_ids.size(0)
        seqlen = input_ids.size(1)
        device = input_ids.device
        directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
        attention_mask = context[:, None, :] | (
            context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
        )
        attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
        # mask for left padding
        mask_1d = torch.tensor(list(range(seqlen - self.prompt_length))[::-1], device=device)[None, :].repeat(batch, 1) < length[:, None]
        mask_1d = torch.cat((torch.ones(batch, self.prompt_length).bool(), mask_1d), dim=1)
        attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
        return attention_mask

    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        # attention_mask: Optional[torch.Tensor] = None,  # dummy parameter for text-generation pipeline
        **kwargs
    ):
        r"""
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`CPMAntTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)

            length (`torch.Tensor` of shape `(batch)`, *optional*):
                The length of input tokens.
            context (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                The Boolean value determines whether the model makes a prediction for that position
            position (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Indices of position of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.max_position_embeddings - 1]`.

                [What are position IDs?](../glossary#position-ids)
            segment (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                A sequence of tokens that is processed together as a unit.
            span (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                A contiguous sequence of tokens within the input text.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # add prompts ahead
        if input_ids.dtype != torch.int32:
            input_ids = input_ids.int()
        dtype, device = input_ids.dtype, input_ids.device
        segment = torch.where(input_ids != 0, 2, 0).to(dtype=dtype, device=device)
        length = (segment != 0).sum(-1).to(dtype=dtype, device=device)
        input_ids = torch.cat(
            (
                torch.arange(self.prompt_length * 2 + self.vocab_size, self.prompt_length * 3 + self.vocab_size, dtype=dtype, device=device).repeat(input_ids.size(0), 1),
                input_ids,
            ),
            dim=1,
        )
        batch, seq_length = input_ids.size()
        segment = torch.cat(
            (torch.zeros(batch, self.prompt_length, dtype=dtype, device=device), segment), dim=1
        )
        context = torch.full((batch, seq_length), 1, dtype=dtype, device=device)
        position = torch.arange(seq_length, dtype=dtype, device=device).repeat(batch, 1)
        span = torch.full((batch, seq_length), 0, dtype=dtype, device=device)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.encoder.num_layers)
            input_ids = input_ids.contiguous()
            hidden_states = self.input_embedding(input_ids)
            segment_states = self.segment_embedding(segment)
            hidden_states = hidden_states + segment_states
        else:
            past_length = past_key_values[0][0].size(-2)
            segment_states = self.segment_embedding(segment)
            hidden_states = self.input_embedding(input_ids) + segment_states[:, -1:, :]

        attention_mask = self._prepare_attention_mask(input_ids, span, context, length)
        position_bias = self.position_bias(position, position, segment, segment)

        attention_mask = attention_mask[:, past_length:, :]
        position_bias = position_bias[:, :, past_length:, :]
        hidden_states = hidden_states[:, past_length:, :]

        hidden_states, present_key_values, all_hidden_states, all_attentions = self.encoder(
            hidden_states, attention_mask, position_bias, output_attentions, output_hidden_states, past_key_values, use_cache
        )

        if past_length == 0:
            hidden_states = hidden_states[:, self.prompt_length:, :]

        if not return_dict:
            return tuple(v for v in [hidden_states, present_key_values, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


@add_start_docstrings(
    """
    The CPMAnt Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    CPMANT_START_DOCSTRING,
)
class CPMAntForCausalLM(CPMAntPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    
    def __init__(self, config: CPMAntConfig):
        super().__init__(config)
        self.cpmant = CPMAntModel(config)
        
        # lm_head.weight is tied to cpmant.input_embedding.weight
        self.lm_head = nn.Linear(config.dim_model, config.vocab_size, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        attention_mask: Optional[torch.Tensor] = None,  # dummy parameter for text-generation pipeline
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`CPMAntTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)

            length (`torch.Tensor` of shape `(batch)`, *optional*):
                The length of input tokens.
            context (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                The Boolean value determines whether the model makes a prediction for that position
            position (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                Indices of position of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.max_position_embeddings - 1]`.

                [What are position IDs?](../glossary#position-ids)
            segment (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                A sequence of tokens that is processed together as a unit.
            span (`torch.Tensor` of shape `(batch_size, seq_len)`, *optional*):
                A contiguous sequence of tokens within the input text.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        model_output = self.cpmant(input_ids, output_attentions, output_hidden_states, past_key_values, use_cache, return_dict)
        hidden_states = model_output.last_hidden_state if return_dict else model_output[0]
        
        logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + model_output
            return output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=model_output.past_key_values,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
        )

    def get_input_embeddings(self):
        return self.cpmant.input_embedding

    def set_input_embeddings(self, embeddings):
        self.cpmant.input_embedding = embeddings
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        input_ids = input_ids.int()
        # save the memory usage of dummy attention mask
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = torch.zeros(1, 1)

        return {
            "input_ids": input_ids,
            "use_cache": kwargs["use_cache"],
            "past_key_values": kwargs.get("past_key_values", None),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        past_key_values = [list(each) if each is not None else each for each in past_key_values]
        for key_value_layer in past_key_values:
            key_value_layer[0] = key_value_layer[0][beam_idx]
            key_value_layer[1] = key_value_layer[1][beam_idx]
        return past_key_values
