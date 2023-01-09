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
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, CausalLMOutput
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


@torch.jit.script  # type: ignore
def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class CPMAntLayerNorm(nn.Module):
    """RMS LayerNorm"""

    def __init__(
        self,
        dim_norm: int,
        eps: float = 1e-6,
        init_var: float = 1.0,
    ):
        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = torch.nn.parameter.Parameter(torch.full((dim_norm,), init_var))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """  # noqa: E501
        if x.size(-1) != self.dim_norm:
            raise AssertionError("x.size(-1) != self.dim_norm")
        return rms_layernorm(x, self.weight, self.eps)


class CPMAntAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        dropout_p: Optional[float] = None,
        use_cache: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.use_cache = use_cache
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.project_q = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.project_k = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.project_v = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)

        self.attention_out = nn.Linear(self.num_heads * self.dim_head, self.dim_model, bias=False)

        self.softmax = torch.nn.Softmax(dim=-1)

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            hidden_q (`torch.Tensor`):
                Indices of input sequence tokens of shape `(batch, len_q, dim_model)`. It will be embedded by model's
                internal embedding lookup matrix.
            hidden_kv (`torch.Tensor` of shape `(batch, len_k, dim_model)`)):
                Tensor *key_value* and *query* of shape `(batch, len_k, dim_model)`
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            past_kv (`Tuple(torch.FloatTensor)`, *optional*): Cached past key and value projection states.
        """  # noqa: E501

        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)

        query = self.project_q(hidden_q)
        key = self.project_k(hidden_kv)
        value = self.project_v(hidden_kv)

        query = query.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        if past_kv is not None:
            key = torch.cat([past_kv[0], key], dim=-2)
            value = torch.cat([past_kv[1], value], dim=-2)
            len_k = key.size(-2)

        # (b, n_h, len_q, d_h) @ (b, n_h, d_h, len_k) -> (b, n_h, len_q, len_k)
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

        if self.dropout is not None:
            score = self.dropout(score)

        # (b, n_h, len_q, len_k) @ (b, n_h, len_k, d_h) -> (b, n_h, len_q, d_h)
        score = torch.matmul(score, value)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3)
        score = score.contiguous().view(batch_size, len_q, self.num_heads * self.dim_head)

        score = self.attention_out(score)

        if self.use_cache:
            return score, (key, value)

        return score


class CPMAntSelfAttentionBlock(nn.Module):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual
    connection.

        Args:
            dim_model (int): Main dimension of modules in transformer blocks.
            num_heads (int): Number of attention heads in the Transformer encoder.
            dim_head (int): Dimension of attention heads for each attention layer in the Transformer encoder.
            eps (float, optional): The epsilon used by the layer normalization layers.
            dropout_p (float, optional): Defaults to 0.
            use_cache (bool, optional): Whether to use cache.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        use_cache: Optional[bool] = False,
    ):
        super().__init__()
        self.use_cache = use_cache
        self.layernorm_before_attention = CPMAntLayerNorm(
            dim_norm=dim_model,
            eps=eps,
        )
        self.self_attention = CPMAntAttention(
            dim_model=dim_model,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout_p=dropout_p,
            use_cache=self.use_cache,
        )
        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): Cached past key and value projection states.
        """  # noqa: E501
        x = self.layernorm_before_attention(hidden_states)
        x = self.self_attention(x, x, attention_mask, position_bias, past_key_value)
        if self.use_cache:
            x, current_key_value = x

        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x

        if self.use_cache:
            return (hidden_states, current_key_value)

        return hidden_states


class DenseGatedACT(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
    ):
        super().__init__()

        self.w_0 = nn.Linear(dim_in, dim_ff, bias=False)
        self.w_1 = nn.Linear(dim_in, dim_ff, bias=False)
        self.act = torch.nn.GELU()

    def forward(self, x: torch.Tensor):
        """Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            x (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """  # noqa: E501
        gate_score = self.act(self.w_0(x))
        x = self.w_1(x)

        x = gate_score * x
        return x


class CPMAntFeedForward(nn.Module):
    r"""FeedForward module

    Args:
        dim_in (int): input dimension.
        dim_ff (int): middle dimension.
        dim_out (int, optional): output dimension. Defaults to None, which means dim_in = dim_out.
        bias (bool, optional):
            whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dropout_p: Optional[float] = None,
    ):
        super().__init__()

        self.w_in = DenseGatedACT(
            dim_in=dim_model,
            dim_ff=dim_ff,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.w_out = nn.Linear(dim_ff, dim_model, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """  # noqa: E501
        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.w_out(x)

        return x


class CPMAntFFNBlock(nn.Module):
    """The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): Main dimension of modules in transformer blocks.
        dim_ff (int): Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        eps (float, optional): The epsilon used by the layer normalization layers.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
    ):
        super().__init__()

        self.layernorm_before_ffn = CPMAntLayerNorm(
            dim_model,
            eps=eps,
        )

        self.ffn = CPMAntFeedForward(
            dim_model,
            dim_ff,
            dropout_p=dropout_p,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
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
        """  # noqa: E501
        x = self.layernorm_before_ffn(hidden_states)
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states


class CPMAntTransformerBlock(nn.Module):
    """The whole transformer block. A sequence of operation. Consists of self-attention block[, cross-attention block] and
    feed-forward block.

        Args:
            dim_model (int): Main dimension of modules in transformer blocks.
            dim_ff (int): Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
            num_heads (int): Number of attention heads in the Transformer encoder.
            dim_head (int): Dimension of attention heads for each attention layer in the Transformer encoder.
            eps (float, optional): The epsilon used by the layer normalization layers.
            dropout_p (float, optional): Defaults to 0.
            use_cache (bool, optional): Whether to use cache.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        use_cache: Optional[bool] = False,
        mask_att: bool = False,
        mask_ffn: bool = False,
    ):
        super().__init__()
        self.mask_att = mask_att
        self.mask_ffn = mask_ffn
        self.use_cache = use_cache
        if not self.mask_att:
            self.self_att = CPMAntSelfAttentionBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_head=dim_head,
                eps=eps,
                dropout_p=dropout_p,
                use_cache=self.use_cache,
            )

        if not self.mask_ffn:
            self.ffn = CPMAntFFNBlock(
                dim_model=dim_model,
                dim_ff=dim_ff,
                eps=eps,
                dropout_p=dropout_p,
            )

    def forward(
        self,
        self_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        self_position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            self_hidden_states (`torch.Tensor`): input to the layer of shape `(batch, seq_enc, dim_model)`
            self_attention_mask (`torch.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_enc, seq_enc)`
            self_position_bias (`torch.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_enc, seq_enc)`
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """  # noqa: E501

        current_key_value = None
        if not self.mask_att:
            hidden_states = self.self_att(
                self_hidden_states,
                attention_mask=self_attention_mask,
                position_bias=self_position_bias,
                past_key_value=past_key_value,
            )
            if self.use_cache:
                hidden_states, current_key_value = hidden_states
        else:
            hidden_states = self_hidden_states

        # (batch, dim_model, len_seq)
        if not self.mask_ffn:
            hidden_states = self.ffn(hidden_states)

        if self.use_cache:
            return (hidden_states, current_key_value)

        return hidden_states


class CPMAntEncoder(nn.Module):
    """Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): Number of layers.
        dim_model (int): Main dimension of modules in transformer blocks.
        dim_ff (int): Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_heads (int): Number of attention heads in the Transformer encoder.
        dim_head (int): Dimension of attention heads for each attention layer in the Transformer encoder.
        eps (float, optional): The epsilon used by the layer normalization layers.
        dropout_p (float, optional): Defaults to 0.
        use_cache (bool, optional): Whether to use cache.
    """  # noqa: E501

    def __init__(
        self,
        num_layers: int = 48,
        dim_model: int = 4096,
        dim_ff: int = 10240,
        num_heads: int = 32,
        dim_head: int = 128,
        eps: float = 1e-6,
        use_cache: Optional[bool] = False,
        dropout_p: Optional[float] = 0.0,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_cache = use_cache
        if mask_modules is not None:
            if len(mask_modules) == num_layers:
                raise ValueError("The total number of masks should equal to num_layers")
            for mask_module in mask_modules:
                if len(mask_module) != 2:
                    raise ValueError("For encoder, each mask should be (mask_att, mask_ffn)")
        else:
            mask_modules = [(False, False)] * num_layers

        self.layers = nn.ModuleList(
            [
                CPMAntTransformerBlock(
                    dim_model=dim_model,
                    dim_ff=dim_ff,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    eps=eps,
                    dropout_p=dropout_p,
                    mask_att=mask_modules[ith][0],
                    mask_ffn=mask_modules[ith][1],
                    use_cache=self.use_cache,
                )
                for ith in range(num_layers)
            ]
        )

        self.output_layernorm = CPMAntLayerNorm(dim_norm=dim_model, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`): input to the layer of shape `(batch, seq_enc, dim_model)`
            attention_mask (`torch.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_enc, seq_enc)`
            position_bias (`torch.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_enc, seq_enc)`
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """  # noqa: E501
        if not self.use_cache:
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask, position_bias)
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states

        current_key_values = []
        for i, module in enumerate(self.layers):
            hidden_states = module(
                hidden_states,
                attention_mask,
                position_bias,
                past_key_value=past_key_values[i] if past_key_values else None,
            )
            current_key_values.append(hidden_states[1])
            hidden_states = hidden_states[0]
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states, current_key_values


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

            if key_pos.size(0) != query_pos.size(0):
                raise AssertionError("key_pos.size(0) != query_pos.size(0)")
            if keylen != key_segment.size(1) or querylen != query_segment.size(1):
                raise AssertionError("keylen != key_segment.size(1) or querylen != query_segment.size(1)")

            key_pos = key_pos.view(batch, -1, keylen)
            query_pos = query_pos.view(batch, querylen, -1)
            key_segment = key_segment.view(batch, -1, keylen)
            query_segment = query_segment.view(batch, querylen, -1)

            relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
            relative_position_bucket = relative_position_bucket + self.num_buckets  # 与相对位置编码区间不重叠

            # b*q*k
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

    Parameters:
        config ([`~CPMAntConfig`]): Model configuration class with all the parameters of the
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CPMANT_INPUTS_DOCSTRING = r"""
    Args:
        input (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`CPMAntTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        length (`torch.Tensor` of shape `(batch)`, *optional*):
            The length of input tokens.
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
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare CPMAnt Model outputting raw hidden-states without any specific head on top.",
    CPMANT_START_DOCSTRING,
)
class CPMAntModel(CPMAntPreTrainedModel):
    def __init__(self, config: CPMAntConfig):
        super().__init__(config)
        self.encoder = CPMAntEncoder()
        self.prompt_embedding = nn.Embedding(config.prompt_types * config.prompt_length, config.dim_model)
        self.segment_embedding = nn.Embedding(config.segment_types, config.dim_model)
        self.input_embedding = nn.Embedding(config.vocab_size, config.dim_model)
        self.position_bias = CPMAntSegmentPositionEmbedding()
        self.prompt_length = config.prompt_length

    def get_input_embeddings(self):
        embeddings = {
            "prompt": self.prompt_embedding,
            "segment": self.segment_embedding,
            "input": self.input_embedding,
            "position": self.position_bias,
        }
        return embeddings

    def set_input_embeddings(self, embeddings, **kwargs):
        self.prompt_embedding = embeddings["prompt"]
        self.segment_embedding = embeddings["segment"]
        self.input_embedding = embeddings["input"]
        self.position_bias = embeddings["position"]

    def _prepare_attention_mask(self, input, span, context, length):
        batch = input.size(0)
        device = input.device
        seqlen = input.size(1)
        directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
        attention_mask = context[:, None, :] | (
            context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
        )
        attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
        # mask for left paddding
        mask_1d = torch.tensor(list(range(seqlen))[::-1], device=device)[None, :].repeat(batch, 1) < length[:, None]
        attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
        return attention_mask

    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input: Optional[torch.Tensor] = None,
        length: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        segment: Optional[torch.Tensor] = None,
        span: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = False,
        **kwargs,
    ):
        r"""
        Args:
            input (`torch.Tensor` of shape `(batch_size, seq_len)`):
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
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        input_prompt = input[:, : self.prompt_length].contiguous()
        input_ids = input[:, self.prompt_length :].contiguous()

        prompt_states = self.prompt_embedding(input_prompt)
        hidden_states = self.input_embedding(input_ids)
        segment_states = self.segment_embedding(segment)
        hidden_states = torch.cat([prompt_states, hidden_states], 1) + segment_states

        attention_mask = self._prepare_attention_mask(input, span, context, length)
        position_bias = self.position_bias(position, position, segment, segment)

        hidden_states = self.encoder(hidden_states, attention_mask, position_bias)
        logits = F.linear(hidden_states, self.input_embedding.weight)

        if not return_dict:
            return tuple(v for v in [logits, hidden_states] if v is not None)

        return BaseModelOutput(hidden_states=hidden_states)


@add_start_docstrings(
    """
    The CPMAnt Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    CPMANT_START_DOCSTRING,
)
class CPMAntForCausalLM(CPMAntPreTrainedModel):
    def __init__(self, config: CPMAntConfig):
        super().__init__(config)
        self.encoder = CPMAntEncoder(use_cache=config.use_cache)
        self.prompt_embedding = nn.Embedding(config.prompt_types * config.prompt_length, config.dim_model)
        self.segment_embedding = nn.Embedding(config.segment_types, config.dim_model)
        self.input_embedding = nn.Embedding(config.vocab_size, config.dim_model)
        self.position_bias = CPMAntSegmentPositionEmbedding()
        self.prompt_length = config.prompt_length
        self.lm_head = nn.Linear(config.dim_model, config.vocab_size, bias=False)

    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input: Optional[torch.Tensor] = None,
        length: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        segment: Optional[torch.Tensor] = None,
        span: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Args:
            input (`torch.Tensor` of shape `(batch_size, seq_len)`):
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
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*):
                Cached past key and value projection states.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.encoder.num_layers)
            input_prompt = input[:, : self.prompt_length].contiguous()
            input_ids = input[:, self.prompt_length :].contiguous()

            prompt_states = self.prompt_embedding(input_prompt)
            hidden_states = self.input_embedding(input_ids)
            segment_states = self.segment_embedding(segment)
            hidden_states = torch.cat([prompt_states, hidden_states], 1) + segment_states

        else:
            past_length = past_key_values[0][0].size(-2)
            segment_states = self.segment_embedding(segment)
            hidden_states = self.input_embedding(input) + segment_states[:, -1:, :]

        attention_mask = self._prepare_attention_mask(input, span, context, length, past_length)
        position_bias = self.position_bias(position, position, segment, segment)

        attention_mask = attention_mask[:, past_length:, :]
        position_bias = position_bias[:, :, past_length:, :]

        self.encoder.use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        hidden_states, present_key_values = self.encoder(hidden_states, attention_mask, position_bias, past_key_values)
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return tuple(v for v in [logits, hidden_states, present_key_values] if v is not None)

        return CausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
        )

    def _prepare_attention_mask(self, input, span, context, length, past_length):
        batch = input.size(0)
        seqlen = input.size(1) + past_length
        device = input.device
        directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
        attention_mask = context[:, None, :] | (
            context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
        )
        attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
        # mask for left paddding
        mask_1d = torch.tensor(list(range(seqlen))[::-1], device=device)[None, :].repeat(batch, 1) < length[:, None]
        attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask
        return attention_mask

    def get_input_embeddings(self):
        return self.lm_head

    def set_input_embeddings(self, embeddings, **kwargs):
        self.lm_head = embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        input_tensors = list(map(self._convert_to_tensors, input_ids))
        keys = set(input_tensors[0].keys())
        padded = {}
        for key in keys:
            padded[key] = self.pad(input_tensors, key, padding_side="left")
        return padded

    def _convert_to_tensors(self, input_ids, task_id=2):
        model_inputs = {}
        input_ids = torch.cat((torch.tensor([6]), input_ids), axis=0)
        input_ids = [j for j in input_ids if j != torch.tensor(1)]

        model_inputs["input"] = [x + self.prompt_length * task_id for x in range(self.prompt_length)] + input_ids
        model_inputs["length"] = len(model_inputs["input"])
        model_inputs["position"] = list(range(len(model_inputs["input"])))
        model_inputs["span"] = [0] * len(model_inputs["input"])
        model_inputs["context"] = [True] * len(model_inputs["input"])
        model_inputs["segment"] = [0] * self.prompt_length + [2] * len(input_ids)

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs

    def _reorder_cache(past_key_values, beam_idx):
        past_key_values = [list(each) if each is not None else each for each in past_key_values]  # type: ignore # noqa: E501
        for key_value_layer in past_key_values:
            if key_value_layer is not None:
                key_value_layer[0] = key_value_layer[0][beam_idx]
                key_value_layer[1] = key_value_layer[1][beam_idx]
        return past_key_values

    def pad(self, orig_items, key, padding_value=0, padding_side="left"):
        items = []
        if isinstance(orig_items[0][key], list):
            if not isinstance(orig_items[0][key][0], torch.Tensor):
                raise TypeError("The type of orig_items[0][key][0] should be tensor!")
            for it in orig_items:
                for tr in it[key]:
                    items.append({key: tr})
        else:
            if not isinstance(orig_items[0][key][0], torch.Tensor):
                raise TypeError("The type of orig_items[0][key][0] should be tensor!")
            items = orig_items

        batch_size = len(items)
        shape = items[0][key].shape
        dim = len(shape)
        if dim > 3:
            raise ValueError(f"input should have at most 3 dimensions, got {dim}")
        max_length = max(item[key].shape[-1] for item in items)
        min_length = min(item[key].shape[-1] for item in items)
        dtype = items[0][key].dtype

        if dim == 1:
            return torch.cat([item[key] for item in items], dim=0)
        elif dim == 2:
            if max_length == min_length:
                return torch.cat([item[key] for item in items], dim=0)
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        for i, item in enumerate(items):
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0])] = item[key][0].clone()
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
                else:
                    tensor[i, : len(item[key][0]), :] = item[key][0].clone()

        return tensor
