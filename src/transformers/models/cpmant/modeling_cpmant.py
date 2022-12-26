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
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
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
            # assert (
            #     pointer.shape == array.shape
            # ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            if pointer.shape != array.shape:
                raise AssertionError("Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits = logits.view(batch_size, -1).contiguous()
        for index in range(len(logits)):

            sorted_logits, sorted_indices = torch.sort(logits[index].view(-1), descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[index][indices_to_remove] = filter_value

        logits = logits.view(batch_size, -1).contiguous()

    return logits


class BeamHypotheses:
    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty

        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / cur_len**self.length_penalty


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
        dtype: torch.dtype = torch.float,
        eps: float = 1e-6,
        init_var: float = 1.0,
    ):

        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = torch.nn.parameter.Parameter(torch.full((dim_norm,), init_var, dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """  # noqa: E501
        # assert x.size(-1) == self.dim_norm
        if x.size(-1) != self.dim_norm:
            raise AssertionError("x.size(-1) != self.dim_norm")
        return rms_layernorm(x, self.weight, self.eps)


class CPMAntLinear(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.float,
        scale_before: bool = False,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale_before = scale_before

        self.weight = torch.nn.parameter.Parameter(torch.empty((dim_out, dim_in), dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (`torch.Tensor` of shape `(batch, seq_len, dim_in)`).
        """  # noqa: E501
        if self.scale_before:
            x = x / math.sqrt(self.dim_in)
            x = F.linear(x, self.weight)
        else:
            x = F.linear(x, self.weight)
            x = x / math.sqrt(self.dim_in)
        return x


class CPMAntAttention(nn.Module):
    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        dtype: torch.dtype = torch.float,
        dropout_p: Optional[float] = None,
    ) -> None:

        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.project_q = CPMAntLinear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_k = CPMAntLinear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)
        self.project_v = CPMAntLinear(self.dim_model, self.num_heads * self.dim_head, dtype=dtype)

        self.attention_out = CPMAntLinear(self.num_heads * self.dim_head, self.dim_model, dtype=dtype)

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
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            hidden_q (`torch.Tensor` of shape `(batch, len_q, dim_model)`))
                Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            hidden_kv (`torch.Tensor` of shape `(batch, len_k, dim_model)`))
                obj:*torch.Tensor* of shape `(batch, len_k, dim_model)`): Length of input sequence before padding.
            attention_mask (`torch.Tensor` of shape `(batch, len_q, len_k)`))
                Used to avoid performing attention on padding token indices.
            position_bias (`torch.Tensor` of shape `(num_heads, len_q, len_k)`))
                Provide positional information about tensor *key_value* and *query*.
            use_cache (`bool`): Whether use cache.
            past_kv (`Tuple[torch.Tensor, torch.Tensor]`): The past key value.
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

        if use_cache:
            return score, (key, value)

        return score


class CPMAntSelfAttentionBlock(nn.Module):
    """The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual
    connection.

        Args:
            dim_model (int): Main dimension of modules in transformer blocks.
            num_heads (int): Number of attention heads in the Transformer encoder.
            dim_head (int): Dimension of attention heads for each attention layer in the Transformer encoder.
            dtype (optional): Defaults to torch.float.
            eps (float, optional): The epsilon used by the layer normalization layers.
            dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_head: int,
        dtype=torch.float,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
    ):
        super().__init__()

        self.layernorm_before_attention = CPMAntLayerNorm(
            dim_norm=dim_model,
            dtype=dtype,
            eps=eps,
        )
        self.self_attention = CPMAntAttention(
            dim_model=dim_model,
            num_heads=num_heads,
            dim_head=dim_head,
            dtype=dtype,
            dropout_p=dropout_p,
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
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Input of self-attention block. It can be the embedding of a batch of sequences.
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            use_cache (`bool`): Whether use cache.
            past_key_value (`Tuple[torch.Tensor, torch.Tensor]`): The past key value.
        """  # noqa: E501
        x = self.layernorm_before_attention(hidden_states)
        x = self.self_attention(x, x, attention_mask, position_bias, use_cache, past_key_value)
        if use_cache:
            x, current_key_value = x

        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x

        if use_cache:
            return hidden_states, current_key_value

        return hidden_states


class DenseGatedACT(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_ff: int,
        dtype=torch.float,
    ):
        super().__init__()

        self.w_0 = CPMAntLinear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            scale_before=False,
        )

        self.w_1 = CPMAntLinear(
            dim_in=dim_in,
            dim_out=dim_ff,
            dtype=dtype,
            scale_before=False,
        )
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
        dtype (optional): Defaults to torch.float.
        init_mean (float, optional):
            mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in
            feed-forward layer. Defaults to 0.
        init_std (float, optional):
            std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)` for fully-connected module used in
            feed-forward layer. Defaults to 0.02.
        bias (bool, optional):
            whether to use bias term in fully-connected layers used in feed-forward module. Defaults to False.
        activate_fn (str, optional): Defaults to `gated_gelu`.
        dropout_p (int, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dtype=torch.float,
        dropout_p: Optional[float] = None,
    ):

        super().__init__()

        self.w_in = DenseGatedACT(
            dim_in=dim_model,
            dim_ff=dim_ff,
            dtype=dtype,
        )

        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.w_out = CPMAntLinear(
            dim_in=dim_ff,
            dim_out=dim_model,
            dtype=dtype,
            scale_before=True,
        )

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
        dtype (optional): Defaults to torch.float.
        eps (float, optional): The epsilon used by the layer normalization layers.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        dtype=torch.float,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0,
    ):
        super().__init__()

        self.layernorm_before_ffn = CPMAntLayerNorm(
            dim_model,
            dtype=dtype,
            eps=eps,
        )

        self.ffn = CPMAntFeedForward(
            dim_model,
            dim_ff,
            dtype=dtype,
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
            dtype (optional): Defaults to torch.float.
            eps (float, optional): The epsilon used by the layer normalization layers.
            dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_heads: int,
        dim_head: int,
        dtype=torch.float,
        eps: float = 1e-6,
        dropout_p: Optional[float] = None,
        mask_att: bool = False,
        mask_ffn: bool = False,
    ):
        super().__init__()
        self.mask_att = mask_att
        self.mask_ffn = mask_ffn

        if not self.mask_att:
            self.self_att = CPMAntSelfAttentionBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                dim_head=dim_head,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
            )

        if not self.mask_ffn:
            self.ffn = CPMAntFFNBlock(
                dim_model=dim_model,
                dim_ff=dim_ff,
                dtype=dtype,
                eps=eps,
                dropout_p=dropout_p,
            )

    def forward(
        self,
        self_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor,
        self_position_bias: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            self_hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            self_attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            self_position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            use_cache (`bool`): Whether use cache.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): Cached past key and value projection states.
        """  # noqa: E501

        current_key_value = None
        if not self.mask_att:
            hidden_states = self.self_att(
                self_hidden_states,
                attention_mask=self_attention_mask,
                position_bias=self_position_bias,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
            if use_cache:
                hidden_states, current_key_value = hidden_states
        else:
            hidden_states = self_hidden_states

        # (batch, dim_model, len_seq)
        if not self.mask_ffn:
            hidden_states = self.ffn(hidden_states)

        if use_cache:
            return hidden_states, current_key_value

        return hidden_states


class CPMAntEncoder(nn.Module):
    """Layers of encoder transformer blocks plus an final layernorm.

    Args:
        num_layers (int): Number of layers.
        dim_model (int): Main dimension of modules in transformer blocks.
        dim_ff (int): Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_heads (int): Number of attention heads in the Transformer encoder.
        dim_head (int): Dimension of attention heads for each attention layer in the Transformer encoder.
        dtype (optional): Defaults to torch.float.
        eps (float, optional): The epsilon used by the layer normalization layers.
        dropout_p (float, optional): Defaults to 0.
    """  # noqa: E501

    def __init__(
        self,
        num_layers: int = 48,
        dim_model: int = 4096,
        dim_ff: int = 10240,
        num_heads: int = 32,
        dim_head: int = 128,
        dtype: torch.dtype = torch.float,
        eps: float = 1e-6,
        dropout_p: Optional[float] = 0.0,
        mask_modules: Optional[List[Tuple[bool, bool]]] = None,
    ):

        super().__init__()
        self.num_layers = num_layers

        if mask_modules is not None:
            # assert len(mask_modules) == num_layers, "The total number of masks should equal to num_layers"
            if len(mask_modules) == num_layers:
                raise ValueError("The total number of masks should equal to num_layers")
            for mask_module in mask_modules:
                # assert len(mask_module) == 2, "For encoder, each mask should be (mask_att, mask_ffn)"
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
                    dtype=dtype,
                    eps=eps,
                    dropout_p=dropout_p,
                    mask_att=mask_modules[ith][0],
                    mask_ffn=mask_modules[ith][1],
                )
                for ith in range(num_layers)
            ]
        )

        self.output_layernorm = CPMAntLayerNorm(dim_norm=dim_model, dtype=dtype, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of `(batch, seq_enc, dim_model)`):
                Input of encoder, might be the embedding of a batch of sequences.
            attention_mask (`torch.Tensor` of `(batch, seq_enc, seq_enc)`):
                Avoid invalid areas to participate in the calculation.
            position_bias (`torch.Tensor` of shape `(num_heads, seq_enc, seq_enc)`):
                Provides position information to attention mechanism.
            use_cache (`bool`): Whether use cache.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): Cached past key and value projection states.
        """  # noqa: E501
        if not use_cache:
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask, position_bias)
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states
        with torch.no_grad():
            current_key_values = []
            for i, module in enumerate(self.layers):
                hidden_states = module(
                    hidden_states,
                    attention_mask,
                    position_bias,
                    past_key_value=past_key_values[i] if past_key_values else None,
                    use_cache=use_cache,
                )
                current_key_values.append(hidden_states[1])
                hidden_states = hidden_states[0]
            hidden_states = self.output_layernorm(hidden_states)
            return hidden_states, current_key_values


class CPMAntEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        embedding_size: int = 4096,
        dtype: torch.dtype = torch.float,
    ):

        super().__init__()

        self.dim_model = embedding_size
        self.weight = torch.nn.parameter.Parameter(torch.empty(vocab_size, embedding_size, dtype=dtype))

    def forward(self, ids: torch.Tensor):
        """
        Args:
            ids (`torch.Tensor`): Indices of input sequence tokens of shape (batch_size, seq_len).
        """  # noqa: E501

        embeds = F.embedding(ids, self.weight) / math.sqrt(self.dim_model)
        return embeds

    def projection(self, x: torch.Tensor):
        """
        Args:
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection
        map embed_size back to vocab_size.
            x (`torch.Tensor`): Input of projection of shape (batch, seq_len, dim_model)
        """  # noqa: E501
        logits = F.linear(x / math.sqrt(self.dim_model), self.weight)
        return logits


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
        dtype: torch.dtype = torch.float,
    ):

        super().__init__()

        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.num_segments = num_segments

        self.relative_attention_bias = torch.nn.parameter.Parameter(
            torch.empty(num_segments * num_segments + num_buckets, num_heads, dtype=dtype)
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

            # assert key_pos.size(0) == query_pos.size(0)
            # assert keylen == key_segment.size(1) and querylen == query_segment.size(1)
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
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`CPMAntTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare CPMAnt Model transformer outputting raw hidden-states without any specific head on top.",
    CPMANT_START_DOCSTRING,
)
class CPMAntModel(CPMAntPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config: CPMAntConfig):

        super().__init__(config)
        self.encoder = CPMAntEncoder()
        self.prompt_embedding = CPMAntEmbeddings()
        self.segment_embedding = CPMAntEmbeddings(vocab_size=config.segment_types)
        self.input_embedding = CPMAntEmbeddings(vocab_size=config.vocab_size)
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

    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input: torch.Tensor,
        length: torch.Tensor,
        context: torch.Tensor,
        position: torch.Tensor,
        segment: torch.Tensor,
        span: torch.Tensor,
    ):
        """
        Args:
            input (`torch.Tensor`): tokenized ids, shape = `(batch, seq_len)`
            length (`torch.Tensor`): length of input, shape = `(batch)`
            context (`torch.Tensor`): context determines whether model predicts, shape = `(batch, seq_len)`
            position (`torch.Tensor`): position of input, shape = `(batch, seq_len)`
            segment (`torch.Tensor`): segment of input, shape = `(batch, seq_len)`
            span (`torch.Tensor`): span the context of input, shape = `(batch, seq_len)`
        """
        batch = input.size(0)
        seqlen = input.size(1)
        input_prompt = input[:, : self.prompt_length].contiguous()
        input_ids = input[:, self.prompt_length :].contiguous()

        prompt_states = self.prompt_embedding(input_prompt)
        hidden_states = self.input_embedding(input_ids)
        segment_states = self.segment_embedding(segment)
        hidden_states = torch.cat([prompt_states, hidden_states], 1) + segment_states

        with torch.no_grad():
            device = input.device
            directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(
                -1, 1
            )
            attention_mask = context[:, None, :] | (
                context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
            )
            attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
            mask_1d = torch.arange(seqlen, device=device)[None, :].repeat(batch, 1) < length[:, None]
            attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask

        position_bias = self.position_bias(position, position, segment, segment)

        hidden_states = self.encoder(hidden_states, attention_mask, position_bias)
        logits = self.input_embedding.projection(hidden_states)
        return logits, hidden_states


class CPMAntForCausalLM(CPMAntPreTrainedModel):
    def __init__(self, config: CPMAntConfig):
        super().__init__(config)
        self.encoder = CPMAntEncoder()
        self.prompt_embedding = CPMAntEmbeddings()
        self.segment_embedding = CPMAntEmbeddings(vocab_size=config.segment_types)
        self.input_embedding = CPMAntEmbeddings(vocab_size=config.vocab_size)
        self.position_bias = CPMAntSegmentPositionEmbedding()
        self.prompt_length = config.prompt_length
        self.lm_head = self.input_embedding.projection

    def forward(
        self,
        input: torch.Tensor,
        length: torch.Tensor,
        context: torch.Tensor,
        position: torch.Tensor,
        segment: torch.Tensor,
        span: torch.Tensor,
        past_key_values=None,
    ):
        """
        Args:
            input (`torch.Tensor`): tokenized ids, shape = `(batch, seq_len)`
            length (`torch.Tensor`): length of input, shape = `(batch)`
            context (`torch.Tensor`): context determines whether model predicts, shape = `(batch, seq_len)`
            position (`torch.Tensor`): position of input, shape = `(batch, seq_len)`
            segment (`torch.Tensor`): segment of input, shape = `(batch, seq_len)`
            span (`torch.Tensor`): span the context of input, shape = `(batch, seq_len)`
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        batch = input.size(0)

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

        seqlen = past_length + input.size(1)

        with torch.no_grad():
            device = input.device
            directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(
                -1, 1
            )
            attention_mask = context[:, None, :] | (
                context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
            )
            attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])
            # mask for left paddding
            mask_1d = (
                torch.tensor(list(range(seqlen))[::-1], device=device)[None, :].repeat(batch, 1) < length[:, None]
            )
            attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask

        position_bias = self.position_bias(position, position, segment, segment)

        attention_mask = attention_mask[:, past_length:, :]
        position_bias = position_bias[:, :, past_length:, :]

        hidden_states, present_key_values = self.encoder(
            hidden_states, attention_mask, position_bias, True, past_key_values
        )
        logits = self.lm_head(hidden_states)
        return logits, hidden_states, present_key_values

    def get_input_embeddings(self):
        return self.lm_head

    def set_input_embeddings(self, embeddings, **kwargs):
        self.lm_head = embeddings

    def prepare_inputs_for_generation(self, input_ids):
        input_tensors = list(map(self._convert_to_tensors, input_ids))
        keys = set(input_tensors[0].keys())
        padded = {}
        for key in keys:
            padded[key] = self.pad(input_tensors, key, padding_side="left")
        return padded

    def _convert_to_tensors(self, input_ids, task_id=2):
        model_inputs = {}
        input_ids = [6] + input_ids
        input_ids = [j for j in input_ids if j != 1]

        model_inputs["input"] = [x + self.prompt_length * task_id for x in range(self.prompt_length)] + input_ids
        model_inputs["length"] = len(model_inputs["input"])
        model_inputs["position"] = list(range(len(model_inputs["input"])))
        model_inputs["span"] = [0] * len(model_inputs["input"])
        model_inputs["context"] = [True] * len(model_inputs["input"])
        model_inputs["segment"] = [0] * self.prompt_length + [2] * len(input_ids)

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0)

        return model_inputs

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

    def generate(self, input_ids, **kwargs):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().tolist()
        if not isinstance(input_ids[0], list):
            input_ids = [input_ids]
        model_inputs = self.prepare_inputs_for_generation(input_ids)
        with torch.inference_mode():
            output_ids = self._decode(model_inputs, **kwargs)
        result = [ii + oi for ii, oi in zip(input_ids, output_ids)]
        return torch.tensor(result)

    def _decode(
        self, model_inputs, beam_size=3, max_length=50, repetition_penalty=1.2, repetition_window=None, **kwargs
    ):
        """
        Args:
            model_inputs (dict): {input, context, segment, length, span, position}.
            beam_size (int, optional, defaults to 3): beam size of beam search.
            max_length (int, optional, defaults to 50): maximum generation length.
            repetition_penalty (float, optional, defaults to 1.2):
                repetition penalty coefficient, 1.0 means no penalty.
            repetition_window (int, optional, defaults to None):
                window size of repetition penalty, None means that all output tokens are penalized.
        """  # noqa: E501
        # generate_length + 1 for EOS token
        max_length += 1

        # expand dimmension
        batch_size = model_inputs["input"].size(0)
        input = (
            model_inputs["input"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        length = (
            model_inputs["length"]
            .unsqueeze(1)
            .expand(batch_size, beam_size)
            .contiguous()
            .view(
                batch_size * beam_size,
            )
        )
        context = (
            model_inputs["context"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        position = (
            model_inputs["position"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        segment = (
            model_inputs["segment"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )
        span = (
            model_inputs["span"]
            .unsqueeze(1)
            .expand(batch_size, beam_size, -1)
            .contiguous()
            .view(batch_size * beam_size, -1)
        )

        done = [False for _ in range(batch_size)]

        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(beam_size, max_length, length_penalty=1, early_stopping=False) for _ in range(batch_size)
        ]

        pred_start_index = input.size(-1)
        past_key_values = None
        for i in range(max_length + 1):
            if i == 0:
                logits, _, past_key_values = self(
                    input=input,
                    length=length,
                    context=context,
                    position=position,
                    segment=segment,
                    span=span,
                    past_key_values=past_key_values,
                )
            else:
                logits, _, past_key_values = self(
                    input=input[:, -1:],
                    length=length,
                    context=context,
                    position=position,
                    segment=segment,
                    span=span,
                    past_key_values=past_key_values,
                )

            # skip all steps when we are done with each sentence
            if all(done):
                break

            # (batch * beam, seqlen, model_dim)
            logits = logits[:, -1, :]

            if i == 0:
                logits[:, 7] = -float("inf")
                logits[:, 4] = -float("inf")

            self.apply_repetition_penalty(
                logits,
                batch_size,
                beam_size,
                input,
                repetition_penalty,
                pred_start_index,
                input.size(-1) - 1,
                repetition_window,
            )
            scores = F.log_softmax(logits, dim=-1)

            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * beam_size, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(batch_size, -1)  # (batch_size, beam_size * vocab_size)
            next_scores, next_words = torch.topk(next_scores, 2 * beam_size, dim=1, largest=True, sorted=True)

            if not (next_scores.size() == next_words.size() == (batch_size, 2 * beam_size)):
                raise AssertionError(
                    "next_scores.size(), next_words.size(), (batch_size, 2 * beam_size) are not equal. "
                )
            next_batch_beam = []

            for sent_id in range(batch_size):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item(), i)
                if done[sent_id]:
                    next_batch_beam.extend([(0, 0, 0)] * beam_size)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                    # get beam and word IDs
                    beam_id = torch.div(idx, scores.size(-1), rounding_mode="floor")
                    word_id = idx % scores.size(-1)

                    # end of sentence, or next word
                    if word_id == 7 or i == max_length:
                        generated_hyps[sent_id].add(
                            input[sent_id * beam_size + beam_id, pred_start_index:].clone().cpu().tolist(),
                            value.item(),
                        )
                    else:
                        next_sent_beam.append((value, word_id, sent_id * beam_size + beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                # assert len(next_sent_beam) == 0 if i == max_length else beam_size
                if i == max_length and len(next_sent_beam) != 0:
                    raise AssertionError("i == max_length and len(next_sent_beam) != 0")
                if i != max_length and len(next_sent_beam) != beam_size:
                    raise AssertionError("i != beam_size and len(next_sent_beam) != beam_size")
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, 0, 0)] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                # assert len(next_batch_beam) == beam_size * (sent_id + 1)
                if len(next_batch_beam) != beam_size * (sent_id + 1):
                    raise AssertionError("len(next_batch_beam) != beam_size * (sent_id + 1)")

            # we have reached the last step
            if i == max_length:
                break

            # sanity check / prepare next batch
            # assert len(next_batch_beam) == batch_size * beam_size
            if len(next_batch_beam) != batch_size * beam_size:
                raise AssertionError("len(next_batch_beam) != batch_size * beam_size")
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input.new([x[1] for x in next_batch_beam])
            beam_idx = length.new([x[2] for x in next_batch_beam]).long()

            # re-order batch and internal states
            input = input[beam_idx, :]

            past_key_values = [list(each) if each is not None else each for each in past_key_values]  # type: ignore # noqa: E501
            for key_value_layer in past_key_values:
                if key_value_layer is not None:
                    key_value_layer[0] = key_value_layer[0][beam_idx]
                    key_value_layer[1] = key_value_layer[1][beam_idx]

            # update input ids
            input = torch.cat([input, beam_words.unsqueeze(1)], dim=-1)
            length += 1
            context = torch.cat(
                [context, torch.ones((context.size(0), 1), dtype=torch.int, device=context.device)],
                dim=-1,
            )
            position = torch.cat([position, position[:, -1:] + 1], dim=-1)
            segment = torch.cat([segment, segment[:, -1:]], dim=-1)  # segment id always the same as the previous token
            span = torch.cat([span, span[:, -1:]], dim=-1)

        # select the best hypotheses
        results = []
        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            results.append(best_hyp)

        return results

    def apply_repetition_penalty(
        self,
        logits,
        batch_size,
        num_beams,
        prev_output_tokens,
        repetition_penalty,
        start_idx=None,
        end_idx=None,
        window_size=None,
    ):
        # only conduct repetition penalty for the output
        # assert repetition_penalty >= 1, "repetition penalty coefficient should >= 1"
        if repetition_penalty < 1:
            raise ValueError("repetition penalty coefficient should >= 1")
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        for i in range(batch_size * num_beams):
            if start_idx is None or end_idx is None:
                output_tokens = prev_output_tokens[i].tolist()
            else:
                if end_idx >= start_idx:
                    if window_size:
                        output_tokens = prev_output_tokens[i][
                            max(start_idx, end_idx + 1 - window_size) : end_idx + 1
                        ].tolist()
                    else:
                        output_tokens = prev_output_tokens[i][start_idx : end_idx + 1].tolist()
                else:
                    output_tokens = []
            for previous_token in set(output_tokens):
                # if score < 0 then repetition penalty has to
                # multiplied to reduce the previous token probability
                if logits[i, previous_token] < 0:
                    logits[i, previous_token] *= repetition_penalty
                else:
                    logits[i, previous_token] /= repetition_penalty
