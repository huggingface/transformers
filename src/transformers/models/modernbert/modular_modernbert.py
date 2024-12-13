# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
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

import math
from enum import Enum
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer, prune_qkv_linear_layer
from ...utils import (
    is_flash_attn_2_available,
    is_torch_greater_or_equal,
    logging,
)
from ..gemma.modeling_gemma import GemmaRotaryEmbedding, apply_rotary_pos_emb


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    RotaryEmbedding = None

if is_torch_greater_or_equal("2.5"):
    pass


_CHECKPOINT_FOR_DOC = "answerdotai/modernbert-base"

logger = logging.get_logger(__name__)


class ModernBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ModernBertModel`]. It is used to instantiate an ModernBert
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ModernBert-7B.
    e.g. [answerdotai/modernbert-base](https://huggingface.co/answerdotai/modernbert-base)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the ModernBert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ModernBertModel`]
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu"`
            if not specified.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to 256): scaling factor used on the attention scores
        final_logit_softcapping (`float`, *optional*, defaults to 30.0): scaling factor when applying tanh softcapping on the logits.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0): scaling factor when applying tanh softcapping on the attention scores.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.

    ```python
    >>> from transformers import ModernBertModel, ModernBertConfig
    >>> # Initializing a ModernBert modernbert-7b style configuration
    >>> configuration = ModernBertConfig()
    >>> # Initializing a model from the modernbert-7b style configuration
    >>> model = ModernBertModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "modernbert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50368,
        hidden_size=768,
        intermediate_size=1152,
        num_hidden_layers=22,
        num_attention_heads=12,
        hidden_activation="gelu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,
        norm_eps=1e-5,
        norm_bias=False,
        pad_token_id=50283,
        eos_token_id=50282,
        bos_token_id=50281,
        cls_token_id=50281,
        sep_token_id=50282,
        tie_word_embeddings=True,
        global_rope_theta=160000.0,
        attention_bias=False,
        attention_dropout=0.0,
        global_attn_every_n_layers=3,
        local_attention=128,
        local_rope_theta=10000.0,
        skip_first_prenorm=True,
        embedding_norm=True,
        embedding_dropout=0.0,
        mlp_bias=False,
        mlp_dropout=0.0,
        unpad_inputs=None,
        unpad_no_grad=True,
        decoder_bias=True,
        classifier_dropout=0.0,
        classifier_pooling="mean",
        classifier_norm=True,
        classifier_bias=False,
        classifier_activation="gelu",
        deterministic_flash_attn=False,
        sparse_prediction=False,
        sparse_pred_ignore_index=-100,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_cutoff_factor = initializer_cutoff_factor
        self.norm_eps = norm_eps
        self.norm_bias = norm_bias
        self.global_rope_theta = global_rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.global_attn_every_n_layers = global_attn_every_n_layers
        self.local_attention = local_attention
        self.local_rope_theta = local_rope_theta
        self.skip_first_prenorm = skip_first_prenorm
        self.embedding_norm = embedding_norm
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.unpad_inputs = unpad_inputs
        self.unpad_no_grad = unpad_no_grad
        self.decoder_bias = decoder_bias
        self.classifier_dropout = classifier_dropout
        self.classifier_pooling = classifier_pooling
        self.classifier_bias = classifier_bias
        self.classifier_norm = classifier_norm
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index

        if unpad_inputs is None:
            self.unpad_inputs = self._attn_implementation == "flash_attention_2"


class ModernBertModuleType(str, Enum):
    in_module = "in"
    out_module = "out"
    embedding = "emb"
    final_out = "final_out"


class ModernBertPoolingType(str, Enum):
    cls = "cls"
    mean = "mean"
    max = "max"


# Copyright 2023 OLMo Authors
# License: Apache-2.0


def _init_modernbert_weights(
    config: ModernBertConfig,
    module: Union[nn.Linear, nn.Embedding],
    module_type: ModernBertModuleType,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    """
    if module_type is None:
        raise RuntimeError("When using the full megatron init, every module must have a type.")

    cutoff_factor = config.initializer_cutoff_factor
    if cutoff_factor is None:
        cutoff_factor = 3

    if module_type == ModernBertModuleType.in_module:
        std = config.initializer_range  # for att_proj (same as QKV), ff_proj
    elif module_type == ModernBertModuleType.out_module:
        std = config.initializer_range / math.sqrt(2.0 * config.num_hidden_layers)  # for attn_out, ff_out
    elif module_type == ModernBertModuleType.embedding:
        std = config.initializer_range  # token embeddings (wte)
    elif module_type == ModernBertModuleType.final_out:
        std = config.hidden_size**-0.5  # final output (ff_out)
    else:
        raise RuntimeError(f"Unknown module type '{module_type}'")

    nn.init.trunc_normal_(
        module.weight,
        mean=0.0,
        std=std,
        a=-cutoff_factor * std,
        b=cutoff_factor * std,
    )

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _unpad_modernbert_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels


def _pad_modernbert_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length
        position_ids: (total_nnz) or None
        labels: (total_nnz) or None

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
        padded_labels: (batch, seqlen) or None
    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    padded_labels = None
    if labels is not None:
        padded_labels = torch.full(
            (batch * seqlen,), fill_value=ignore_index, dtype=labels.dtype, device=labels.device
        )
        padded_labels[indices] = labels
        padded_labels = padded_labels.view(batch, seqlen)

    return padded_inputs, padded_labels

    # Copyright (c) 2023, Tri Dao.
    # License: Apache-2.0

    # if is_flash_attn_2_available():


class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # (total_nnz, 3, nheads, headdim)
        total_nnz, three, nheads, headdim = qkv.shape
        assert three == 3
        if qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
            qk = qkv[:, :2].view(total_nnz, -1, headdim)
            apply_rotary(
                qk,
                cos,
                sin,
                seqlen_offsets=0,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=False,
                inplace=True,
            )
        else:
            q, k = qkv[:, 0, :, :], qkv[:, 1, :, :]
            apply_rotary(
                q,
                cos,
                sin,
                seqlen_offsets=0,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=False,
                inplace=True,
            )
            apply_rotary(
                k,
                cos,
                sin,
                seqlen_offsets=0,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                interleaved=False,
                inplace=True,
            )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        if do.is_contiguous():
            total_nnz, three, nheads, headdim = do.shape
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads) dimensions, we get the same tensor
            dqk = do[:, :2].view(total_nnz, -1, headdim)
            apply_rotary(
                dqk,
                cos,
                sin,
                seqlen_offsets=0,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=False,
                inplace=True,
                conjugate=True,
            )
        else:
            dq, dk = do[:, 0, :, :], do[:, 1, :, :]
            apply_rotary(
                dq,
                cos,
                sin,
                seqlen_offsets=0,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=False,
                inplace=True,
                conjugate=True,
            )
            apply_rotary(
                dk,
                cos,
                sin,
                seqlen_offsets=0,
                cu_seqlens=cu_seqlens,
                max_seqlen=ctx.max_seqlen,
                interleaved=False,
                inplace=True,
                conjugate=True,
            )

        return do, None, None, None, None, None, None


def apply_rotary_unpadded(
    qkv,
    cos,
    sin,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        qkv: (total_nnz, 3, nheads, headdim) - input tensor for packed QKV.
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (total_nnz, dim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


class ModernBertUnpaddedRotaryEmbedding(RotaryEmbedding):
    """
    The rotary position embeddings applied directly to unpadded sequences.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        max_seqlen: if max_seqlen, device, and dtype are provided, we precompute the cos_sin_cache
            up to max_seqlen. If the max_seqlen, device, or dtype during training/inference differ,
            the cos_sin_cache wll be recomputed during the forward pass.
        """
        super().__init__(dim=dim, base=base, pos_idx_in_fp32=True, device=device, interleaved=False)
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply rotary embedding *inplace* to qkv.
        qkv: (total_nnz, 3, nheads, headdim)
        cu_seqlens: (batch + 1,) cumulative sequence lengths
        max_seqlen: int max seq length in the batch
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"


class ModernBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.drop = nn.Dropout(config.embedding_dropout) if config.embedding_dropout > 0.0 else nn.Identity()

    def _init_weights(self, reset_params: bool = False):
        _init_modernbert_weights(self.config, self.tok_embeddings, module_type=ModernBertModuleType.embedding)
        if reset_params:
            self.norm.reset_parameters()

    @torch.compile(dynamic=True)
    def forward(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))


class ModernBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout) if config.mlp_dropout > 0.0 else nn.Identity()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def _init_weights(self, reset_params: bool = False):
        _init_modernbert_weights(self.config, self.Wi, module_type=ModernBertModuleType.in_module)
        _init_modernbert_weights(self.config, self.Wo, module_type=ModernBertModuleType.out_module)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertRotaryEmbedding(GemmaRotaryEmbedding):
    pass


def eager_attention_forward(
    self: "ModernBertAttention",
    qkv: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    attention_mask: torch.Tensor,
    bs: int,
    seqlen: int,
    dim: int,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = self.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if self.training:
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, seqlen, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def flash_attention_forward(
    self: "ModernBertAttention",
    qkv: torch.Tensor,
    rotary_emb: ModernBertUnpaddedRotaryEmbedding,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # (total_seqlen, 3, nheads, headdim)
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)

        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            deterministic=self.deterministic_flash_attn,
            window_size=local_attention,
        )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            deterministic=self.deterministic_flash_attn,
            window_size=local_attention,
        )
    return (attn.view(bs, dim),)


# def flex_attention_forward(
#     config: ModernBertConfig,
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     mask: Optional[torch.Tensor],
#     output_attentions: bool = False,
#     **_kwargs,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

#     attn_output = flex_attention(
#         query,
#         key,
#         value,
#         enable_gqa=True,
#         scale=config.scaling,
#         return_lse=output_attentions,
#     )
#     if not output_attentions:
#         attn_weights = None
#     else:
#         attn_output, attn_weights = attn_output

#     attn_output = attn_output.transpose(1, 2).contiguous()
#     return attn_output, attn_weights


def sdpa_attention_forward(
    self: "ModernBertAttention",
    qkv: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    attention_mask: torch.Tensor,
    bs: int,
    seqlen: int,
    dim: int,
    **_kwargs,
) -> Tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = self.rotary_emb(qkv, position_ids=position_ids)
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=self.attention_dropout if self.training else 0.0,
        attn_mask=attention_mask,
    ).transpose(1, 2)
    attn_output = attn_output.view(bs, seqlen, dim)
    return (attn_output,)


MODERNBERT_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    # "flex_attention": flex_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


class ModernBertAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional details.
    """

    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias)

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
        else:
            self.local_attention = (-1, -1)

        rope_theta = config.global_rope_theta
        max_position_embeddings = config.max_position_embeddings
        if self.local_attention != (-1, -1):
            if config.local_rope_theta is not None:
                rope_theta = config.local_rope_theta
            max_position_embeddings = config.local_attention

        if config._attn_implementation == "flash_attention_2":
            self.rotary_emb = ModernBertUnpaddedRotaryEmbedding(
                dim=self.head_dim, max_seqlen=max_position_embeddings, base=rope_theta
            )
        else:
            self.rotary_emb = ModernBertRotaryEmbedding(
                dim=self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta
            )

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)

        # Prune linear layers
        self.Wqkv = prune_qkv_linear_layer(self.Wqkv, index)
        self.Wo = prune_linear_layer(self.Wo, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_heads = self.num_heads - len(heads)
        self.all_head_size = self.head_dim * self.num_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def _init_weights(self, reset_params: bool = False):
        _init_modernbert_weights(self.config, self.Wqkv, module_type=ModernBertModuleType.in_module)
        _init_modernbert_weights(self.config, self.Wo, module_type=ModernBertModuleType.out_module)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        """Perform self-attention.

        There are two attention implementations supported: PyTorch's SDPA attention and Flash Attention 2.

        The arguments are unpadded. The SDPA implementation of attention requires padded arguments while the
        Flash Attention implementation does not. If using SDPA we first call `pad_input`. Once we compute
        attention, we re-unpad our outputs for the other layers. The pad/unpad operations add overhead, but not
        sending pad tokens through ffs saves compute.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen)

        Returns:
            attention: (total_nnz, dim)
        """
        qkv = self.Wqkv(hidden_states)

        attn_kwargs = {
            "output_attentions": output_attentions,
            "dim": self.all_head_size,
        }
        if self.config._attn_implementation == "flash_attention_2":
            bs = hidden_states.shape[0]
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)

            attn_kwargs.update(
                {
                    "local_attention": self.local_attention,
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": max_seqlen,
                    "bs": bs,
                }
            )
        else:
            bs, seqlen = hidden_states.shape[:2]
            qkv = qkv.view(bs, seqlen, 3, self.num_heads, self.head_dim)

            attn_kwargs.update(
                {
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "bs": bs,
                    "seqlen": seqlen,
                }
            )

        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logger.warning_once(
                    "Outputting attentions is only supported with the 'eager' attention implementation, "
                    'not with "sdpa". Falling back to `attn_implementation="eager"`.'
                )
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logger.warning_once(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`.'
                    " Setting `output_attentions=False`."
                )

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](
            self,
            qkv=qkv,
            rotary_emb=self.rotary_emb,
            **attn_kwargs,
        )
        hidden_states = attn_outputs[0]

        return (self.out_drop(self.Wo(hidden_states)),) + attn_outputs[1:]  # add attentions if outputted


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if config.skip_first_prenorm and config.embedding_norm and layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)

    def _init_weights(self, reset_params: bool = False):
        if reset_params:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        """Forward pass for a ModernBert layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            position_ids: (total_nnz,)
            attention_mask: (batch, max_seqlen)
            cu_seqlens: (batch + 1,)
            max_seqlen: int
        """
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_outputs[0]
        hidden_states = hidden_states + self.compiled_mlp(hidden_states)

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation] if config.classifier_activation else nn.Identity()
        self.norm = (
            nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
            if config.classifier_norm
            else nn.Identity()
        )

    def _init_weights(self, reset_params: bool = False):
        if reset_params:
            self.norm.reset_parameters()
        _init_modernbert_weights(self.config, self.dense, module_type=ModernBertModuleType.in_module)

    def reset_parameters(self):
        self._init_weights(reset_params=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


class ModernBertPoolingHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation] if config.classifier_activation else nn.Identity()
        self.norm = (
            nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
            if config.classifier_norm
            else nn.Identity()
        )
        self.drop = torch.nn.Dropout(config.classifier_dropout) if config.classifier_dropout > 0 else nn.Identity()
        self.pooling_type = ModernBertPoolingType(config.classifier_pooling)

    def forward(self, hidden_states: torch.Tensor, pool: Optional[bool] = True) -> torch.Tensor:
        if pool:
            if self.pooling_type == ModernBertPoolingType.cls:
                output = hidden_states[:, 0]
            elif self.pooling_type == ModernBertPoolingType.mean:
                output = hidden_states.mean(dim=1)
            elif self.pooling_type == ModernBertPoolingType.max:
                output = hidden_states.max(dim=1)[0]
        else:
            output = hidden_states

        return self.drop(self.norm(self.act(self.dense(output))))

    def _init_weights(self, reset_params: bool = False):
        _init_modernbert_weights(self.config, self.dense, module_type=ModernBertModuleType.out_module)
        if reset_params and hasattr(self.norm, "reset_parameters"):
            self.norm.reset_parameters()

    def reset_parameters(self):
        self._init_weights(reset_params=True)


class ModernBertPreTrainedModel(PreTrainedModel):
    config_class = ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(
        self,
        module: Union[ModernBertEncoderLayer, ModernBertAttention, ModernBertMLP, ModernBertEmbeddings],
        reset_params: bool = False,
    ):
        if isinstance(module, (ModernBertEncoderLayer, ModernBertAttention, ModernBertMLP, ModernBertEmbeddings)):
            module._init_weights(reset_params=reset_params)

    @torch.no_grad()
    def _unpad_inputs_no_grad(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        return self._unpad_inputs(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
        )

    def _unpad_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor | int | None]:
        return _unpad_modernbert_input(
            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
        )

    @torch.no_grad()
    def _pad_outputs_no_grad(
        self,
        inputs: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        seqlen: int,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        return self._pad_outputs(
            inputs=inputs,
            indices=indices,
            batch_size=batch_size,
            seqlen=seqlen,
            labels=labels,
            ignore_index=ignore_index,
        )

    def _pad_outputs(
        self,
        inputs: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        seqlen: int,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        return _pad_modernbert_output(
            inputs=inputs, indices=indices, batch=batch_size, seqlen=seqlen, labels=labels, ignore_index=ignore_index
        )


class ModernBertModel(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.layers[layer].attn.prune_heads(heads)

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        if module and hasattr(module, "_init_weights"):
            super()._init_weights(module, reset_params=reset_params)
        elif isinstance(reset_params, bool):
            self.embeddings._init_weights(reset_params=reset_params)

            if reset_params:
                self.final_norm.reset_parameters()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            batch_size, seq_len = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.bool)

        repad = False
        if self.config.unpad_inputs:
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if self.config.unpad_no_grad:
                    input_ids, indices, cu_seqlens, max_seqlen, *_ = self._unpad_inputs_no_grad(
                        input_ids, attention_mask
                    )
                else:
                    input_ids, indices, cu_seqlens, max_seqlen, *_ = self._unpad_inputs(input_ids, attention_mask)
        elif position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embeddings(input_ids)

        # expand attention_mask
        if self.config._attn_implementation != "flash_attention_2" and attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cu_seqlens,
                    max_seqlen,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if repad:
            hidden_states, _ = self._pad_outputs(hidden_states, indices, batch_size, seq_len)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ModernBertForMaskedLM(ModernBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)

        if config.tie_word_embeddings:
            decoder_weights = self.model.embeddings.tok_embeddings.weight
        else:
            decoder_weights = nn.Linear(config.hidden_size, config.vocab_size, bias=False).weight
        self.decoder = nn.Linear(decoder_weights.size(1), decoder_weights.size(0), bias=config.decoder_bias)
        self.decoder.weight = decoder_weights

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index
        # Initialize weights and apply final processing
        self._init_weights(reset_params=False)

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            super()._init_weights(module, reset_params=reset_params)
        else:
            assert isinstance(reset_params, bool)
            self.model._init_weights(reset_params=reset_params)
            self.head._init_weights(reset_params=reset_params)

            # Output weights.
            if not self.config.tie_word_embeddings:
                _init_modernbert_weights(self.config, self.decoder, module_type=ModernBertModuleType.out_module)

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.unpad_inputs:
            if indices is None and cu_seqlens is None and max_seqlen is None:
                batch_size, seq_len = input_ids.shape[:2]
                if self.config.unpad_no_grad:
                    input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = self._unpad_inputs_no_grad(
                        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )
                else:
                    input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = self._unpad_inputs(
                        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = self.compiled_head(last_hidden_state)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        if self.config.unpad_inputs:
            if self.config.unpad_no_grad:
                logits, _ = self._pad_outputs_no_grad(logits, indices, batch_size, seq_len)
            else:
                logits, _ = self._pad_outputs(logits, indices, batch_size, seq_len)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModernBertForSequenceClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPoolingHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self._init_weights(reset_params=False)

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            super()._init_weights(module, reset_params=reset_params)
        else:
            assert isinstance(reset_params, bool)
            self.model._init_weights(reset_params=reset_params)
            self.head._init_weights(reset_params=reset_params)
            _init_modernbert_weights(self.config, self.classifier, module_type=ModernBertModuleType.final_out)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        pooled_output = self.head(last_hidden_state)
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

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModernBertForTokenClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernBertModel(config)
        self.drop = nn.Dropout(config.classifier_dropout) if config.classifier_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self._init_weights(reset_params=False)

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        assert (module is None) != (reset_params is None), "arg module xor reset_params must be specified"
        if module:
            super()._init_weights(module, reset_params=reset_params)
        else:
            assert isinstance(reset_params, bool)
            self.model._init_weights(reset_params=reset_params)
            _init_modernbert_weights(self.config, self.classifier, module_type=ModernBertModuleType.final_out)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
