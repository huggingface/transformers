# Copyright 2024 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team. All rights reserved.
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
from contextlib import nullcontext
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
)
from ...utils.import_utils import is_triton_available
from ..gemma.modeling_gemma import GemmaRotaryEmbedding, apply_rotary_pos_emb


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    RotaryEmbedding = object

_CHECKPOINT_FOR_DOC = "answerdotai/ModernBERT-base"
_CONFIG_FOR_DOC = "ModernBertConfig"

logger = logging.get_logger(__name__)


class ModernBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ModernBertModel`]. It is used to instantiate an ModernBert
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ModernBERT-base.
    e.g. [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50368):
            Vocabulary size of the ModernBert model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ModernBertModel`]
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 1152):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder. Will default to `"gelu"`
            if not specified.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
            The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        norm_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the normalization layers.
        pad_token_id (`int`, *optional*, defaults to 50283):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 50282):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 50281):
            Beginning of stream token id.
        cls_token_id (`int`, *optional*, defaults to 50281):
            Classification token id.
        sep_token_id (`int`, *optional*, defaults to 50282):
            Separation token id.
        mask_token_id (`int`, *optional*, defaults to 50284):
            Mask token id.
        global_rope_theta (`float`, *optional*, defaults to 160000.0):
            The base period of the global RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        global_attn_every_n_layers (`int`, *optional*, defaults to 3):
            The number of layers between global attention layers.
        local_attention (`int`, *optional*, defaults to 128):
            The window size for local attention.
        local_rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the local RoPE embeddings.
        embedding_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP layers.
        mlp_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the MLP layers.
        decoder_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the decoder layers.
        classifier_pooling (`str`, *optional*, defaults to `"cls"`):
            The pooling method for the classifier. Should be either `"cls"` or `"mean"`. In local attention layers, the
            CLS token doesn't attend to all tokens on long sequences.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the classifier.
        classifier_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the classifier.
        classifier_activation (`str`, *optional*, defaults to `"gelu"`):
            The activation function for the classifier.
        deterministic_flash_attn (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic flash attention. If `False`, inference will be faster but not deterministic.
        sparse_prediction (`bool`, *optional*, defaults to `False`):
            Whether to use sparse prediction for the masked language model instead of returning the full dense logits.
        sparse_pred_ignore_index (`int`, *optional*, defaults to -100):
            The index to ignore for the sparse prediction.
        reference_compile (`bool`, *optional*):
            Whether to compile the layers of the model which were compiled during pretraining. If `None`, then parts of
            the model will be compiled if 1) `triton` is installed, 2) the model is not on MPS, 3) the model is not
            shared between devices, and 4) the model is not resized after initialization. If `True`, then the model may
            be faster in some scenarios.
        repad_logits_with_grad (`bool`, *optional*, defaults to `False`):
            When True, ModernBertForMaskedLM keeps track of the logits' gradient when repadding for output. This only
            applies when using Flash Attention 2 with passed labels. Otherwise output logits always have a gradient.
        is_causal (`bool`, *optional*, defaults to `False`):
            Whether to use causal attention (e.g. decoder-only models). For standard encoder-decoder models, this should
            be set to `False`.
        num_future_masks (`int`, *optional*, defaults to 3):
            The number of masks tokens to append to each iterative encoder-only token generation (is_casual=False).
    Examples:

    ```python
    >>> from transformers import ModernBertModel, ModernBertConfig

    >>> # Initializing a ModernBert style configuration
    >>> configuration = ModernBertConfig()

    >>> # Initializing a model from the modernbert-base style configuration
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
        mask_token_id=50284,
        global_rope_theta=160000.0,
        attention_bias=False,
        attention_dropout=0.0,
        global_attn_every_n_layers=3,
        local_attention=128,
        local_rope_theta=10000.0,
        embedding_dropout=0.0,
        mlp_bias=False,
        mlp_dropout=0.0,
        decoder_bias=True,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout=0.0,
        classifier_bias=False,
        classifier_activation="gelu",
        deterministic_flash_attn=False,
        sparse_prediction=False,
        sparse_pred_ignore_index=-100,
        reference_compile=None,
        repad_logits_with_grad=False,
        is_causal=False,
        num_future_masks=3,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            mask_token_id=mask_token_id,
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
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.mlp_dropout = mlp_dropout
        self.decoder_bias = decoder_bias
        self.classifier_pooling = classifier_pooling
        self.classifier_dropout = classifier_dropout
        self.classifier_bias = classifier_bias
        self.classifier_activation = classifier_activation
        self.deterministic_flash_attn = deterministic_flash_attn
        self.sparse_prediction = sparse_prediction
        self.sparse_pred_ignore_index = sparse_pred_ignore_index
        self.reference_compile = reference_compile
        self.repad_logits_with_grad = repad_logits_with_grad
        self.is_causal = is_causal
        self.num_future_masks = num_future_masks
        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.'
            )


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
) -> torch.Tensor:
    """
    Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
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

    return padded_inputs


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
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        # We need qkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
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

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        # We need dqkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
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
        self.drop = nn.Dropout(config.embedding_dropout)

    @torch.compile(dynamic=True)
    def compiled_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)))

    def forward(
        self, input_ids: torch.LongTensor = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(self.norm(inputs_embeds))
        else:
            hidden_states = (
                self.compiled_embeddings(input_ids)
                if self.config.reference_compile
                else self.drop(self.norm(self.tok_embeddings(input_ids)))
            )
        return hidden_states


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
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertRotaryEmbedding(GemmaRotaryEmbedding):
    def __init__(self, config: ModernBertConfig, dim: int, base: float, device: Optional[torch.device] = None):
        super().__init__(self, config=config, device=device)
        inv_freq, self.attention_scaling = self.rope_init_fn(None, device, dim=dim, base=base)


def eager_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    is_causal: bool = False,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]

    # Get RoPE embeddings
    if past_key_value is not None:
        # Generate RoPE embeddings ONLY for the new token's position
        qkv_new = torch.stack([query, key, value], dim=2).transpose(1, 2)
        cos, sin = module.rotary_emb(qkv_new, position_ids=position_ids)

        query_new, key_new = apply_rotary_pos_emb(query, key, cos, sin)

        # Concatenate with past keys (which already have correct RoPE applied)
        past_k, past_v = past_key_value
        key = torch.cat([past_k, key_new], dim=-2)
        value = torch.cat([past_v, value], dim=-2)
        query = query_new
    else:
        # For initial forward pass
        position_ids = torch.arange(query.size(-2), device=query.device)
        position_ids = position_ids.unsqueeze(0).expand(bs, -1)
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

    # Calculate attention scores
    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    final_attention_mask = None
    if local_attention != (-1, -1):
        final_attention_mask = sliding_window_mask

    # Handle masking similar to SDPA version
    if is_causal:
        if past_key_value is not None:
            # For cached case, we only need the last row of the sliding window mask
            if final_attention_mask is not None:
                final_attention_mask = final_attention_mask[:, :, -1:, :]
            causal_mask = torch.zeros((1, 1, 1, key.size(-2)), device=query.device)
        else:
            # Create causal mask for the full sequence
            seq_len = key.size(-2)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=query.device), diagonal=1
            ).expand(1, 1, seq_len, seq_len)

        # Combine masks if local attention is active
        if final_attention_mask is not None:
            final_attention_mask = final_attention_mask + causal_mask
        else:
            final_attention_mask = causal_mask

    # Apply attention mask from input if provided and no other masks are active
    elif final_attention_mask is None and attention_mask is not None:
        final_attention_mask = attention_mask

    # Apply attention mask if any
    if final_attention_mask is not None:
        attn_weights = attn_weights + final_attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)

    outputs = (attn_output,)
    if output_attentions:
        outputs = outputs + (attn_weights,)
    if use_cache:
        outputs = outputs + ((key, value),)
    return outputs


def flash_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    rotary_emb: ModernBertUnpaddedRotaryEmbedding,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    is_causal: bool = False,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    **_kwargs,
) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    original_seq_len = qkv.size(0)
    if past_key_value is not None:
        past_k, past_v = past_key_value
        # Reshape from [batch, num_heads, seq, head_dim] to [seq, num_heads, head_dim]
        past_k = past_k.squeeze(0).transpose(0, 1)
        past_v = past_v.squeeze(0).transpose(0, 1)

        q = qkv[:, 0]
        k = torch.cat([past_k, qkv[:, 1]], dim=0)
        v = torch.cat([past_v, qkv[:, 2]], dim=0)
        q = torch.cat([torch.zeros_like(past_k, device=q.device, dtype=q.dtype), q], dim=0)
        qkv = torch.stack([q, k, v], dim=1)

    # Flash attention logic stays exactly the same
    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    if convert_dtype:
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
            causal=is_causal,
        )
        attn = attn.to(orig_dtype)
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
            causal=is_causal,
        )

    if past_key_value is not None:
        attn = attn[-original_seq_len:]

    outputs = (attn.view(bs, dim),)
    if use_cache:
        k, v = qkv[:, 1], qkv[:, 2]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        outputs = outputs + ((k, v),)
    return outputs


def sdpa_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: Tuple[int, int],
    bs: int,
    dim: int,
    is_causal: bool = False,
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: Optional[bool] = False,
    **_kwargs,
) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]

    # Get RoPE embeddings
    if past_key_value is not None:
        # Generate RoPE embeddings ONLY for the new token's position
        qkv_new = torch.stack([query, key, value], dim=2).transpose(1, 2)
        cos, sin = module.rotary_emb(qkv_new, position_ids=position_ids)

        query_new, key_new = apply_rotary_pos_emb(query, key, cos, sin)

        # Concatenate with past keys (which already have correct RoPE applied)
        past_k, past_v = past_key_value
        key = torch.cat([past_k, key_new], dim=-2)
        value = torch.cat([past_v, value], dim=-2)
        query = query_new
    else:
        # For initial forward pass
        position_ids = torch.arange(query.size(-2), device=query.device)
        position_ids = position_ids.unsqueeze(0).expand(bs, -1)
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

    final_attention_mask = None
    if local_attention != (-1, -1):
        final_attention_mask = sliding_window_mask

    # Figure out the attention mask
    #   if is_causal, we need to create a causal mask
    #   unless it is_causal and past_key_value is not None, then all can be attended
    #   unless sliding_window_mask is provided for local attention, we need to combine masks
    #   and if not is_causal we can simply use the provided mask
    if is_causal:
        if past_key_value is not None:
            # For cached case, we only need the last row of the sliding window mask
            if final_attention_mask is not None:
                final_attention_mask = final_attention_mask[:, :, -1:, :]
            causal_mask = torch.zeros((1, 1, 1, key.size(-2)), device=query.device)
        else:
            # Create causal mask for the full sequence
            seq_len = key.size(-2)
            # Create upper triangular matrix of ones (for positions to mask)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), torch.finfo(attention_mask.dtype).min, device=query.device), diagonal=1
            ).expand(1, 1, seq_len, seq_len)

        # Combine masks if local attention is active
        if final_attention_mask is not None:
            final_attention_mask = final_attention_mask + causal_mask
        else:
            final_attention_mask = causal_mask

    # Apply attention mask from input if provided and no other masks are active
    elif final_attention_mask is None and attention_mask is not None:
        final_attention_mask = attention_mask

    attn_output = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=final_attention_mask,
            is_causal=False,  # attn masks handled above
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    outputs = (attn_output,)
    if use_cache:
        outputs = outputs + ((key, value),)
    return outputs


MODERNBERT_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
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
            self.rotary_emb = ModernBertRotaryEmbedding(config=config, dim=self.head_dim, base=rope_theta)

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](
            self,
            qkv=qkv,
            rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            output_attentions=output_attentions,
            is_causal=self.config.is_causal,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_outputs[0]
        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        hidden_states = hidden_states + mlp_output

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


MODERNBERT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ModernBertConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare ModernBert Model outputting raw hidden-states without any specific head on top.",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertPreTrainedModel(PreTrainedModel):
    config_class = ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
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

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        if isinstance(module, ModernBertEmbeddings):
            init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ModernBertMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, ModernBertForMaskedLM):
            init_weight(module.decoder, stds["out"])
        elif isinstance(module, (ModernBertForSequenceClassification, ModernBertForTokenClassification)):
            init_weight(module.classifier, stds["final_out"])
        elif isinstance(module, ModernBertForCausalLM):
            init_weight(module.decoder, stds["out"])

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        # If the user didn't specify anything, try to use flash_attention_2 if available.
        # Otherwise we fall back to the default SDPA -> Eager from the super() method.
        # ModernBert's FA2 implementation correctly handles non-fp16/bf16 dtypes, we don't
        # need the FA2 warning for non-fp16/bf16 dtypes so we set fp16 for the FA2 check.
        if config._attn_implementation_internal is None and not getattr(config, "is_causal", False):
            config._attn_implementation_internal = "flash_attention_2"
            try:
                return cls._check_and_enable_flash_attn_2(
                    config,
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    hard_check_only=False,
                    check_device_map=check_device_map,
                )
            except (ValueError, ImportError):
                config._attn_implementation_internal = None
        return super()._autoset_attn_implementation(
            config,
            use_flash_attention_2=use_flash_attention_2,
            torch_dtype=torch.float16,
            device_map=device_map,
            check_device_map=check_device_map,
        )

    def _maybe_set_compile(self):
        if self.config.reference_compile is False:
            return

        if hasattr(self, "hf_device_map") and len(self.hf_device_map) > 1:
            if self.config.reference_compile:
                logger.warning_once(
                    "If `accelerate` split the model across devices, `torch.compile` will not work. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "mps":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.mps` device is not supported. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.device.type == "cpu":
            if self.config.reference_compile:
                logger.warning_once(
                    "Compiling the model with `torch.compile` and using a `torch.cpu` device is not supported. "
                    "Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        if self.config.reference_compile is None:
            self.config.reference_compile = is_triton_available()

    def resize_token_embeddings(self, *args, **kwargs):
        model_embeds = super().resize_token_embeddings(*args, **kwargs)

        if self.config.reference_compile in {True, None}:
            if self.config.reference_compile:
                logger.warning_once(
                    "Resizing token embeddings with `torch.compile` is not supported. Falling back to non-compiled mode."
                )
            self.config.reference_compile = False

        return model_embeds


MODERNBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. With Flash Attention 2.0, padding will be ignored
            by default should you provide it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        repad (`bool`, *optional*):
            Whether or not to repad the output tensors (hidden states, attentions, etc.)
"""


@add_start_docstrings(
    "The bare ModernBert Model outputting raw hidden-states without any specific head on top.",
    MODERNBERT_START_DOCSTRING,
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

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repad: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if use_cache and self.config._attn_implementation == "flash_attention_2":
            raise ValueError("use_cache is not supported with flash attention 2, please switch to eager")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        if self.config._attn_implementation == "flash_attention_2":
            output_attentions = False

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if not (self.config.is_causal or past_key_values is not None):
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        if past_key_values is not None:
            # Extend attention mask for past key values
            past_length = past_key_values[0][0].size(-2)
            attention_mask = torch.ones((batch_size, past_length + 1), device=device, dtype=attention_mask.dtype)

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask
                    )
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.size(1) :]

            attention_mask, sliding_window_mask = self._update_attention_mask(
                attention_mask, output_attentions=output_attentions
            )

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    sliding_window_mask,
                    position_ids,
                    cu_seqlens,
                    max_seqlen,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    output_attentions=output_attentions,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions and len(layer_outputs) > 1:
                if use_cache:
                    all_self_attentions = all_self_attentions + (layer_outputs[-2],)
                else:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if repad:
            hidden_states = _pad_modernbert_output(
                inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len
            )
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    _pad_modernbert_output(inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len)
                    for hs in all_hidden_states
                )
            if all_self_attentions is not None:
                all_self_attentions = tuple(
                    _pad_modernbert_output(inputs=attn, indices=indices, batch=batch_size, seqlen=seq_len)
                    for attn in all_self_attentions
                )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attentions] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _update_attention_mask(self, attention_mask: torch.Tensor, output_attentions: bool) -> torch.Tensor:
        if output_attentions:
            if self.config._attn_implementation == "flash_attention_2":
                logger.warning_once(
                    "Outputting attentions is only supported with the eager or sdpa attention implementations, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`.'
                    " Setting `output_attentions=False`."
                )

        global_attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

        # Create position indices
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        # Calculate distance between positions
        distance = torch.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (
            (distance <= self.config.local_attention // 2).unsqueeze(0).unsqueeze(0).to(attention_mask.device)
        )
        # Combine with existing mask
        sliding_window_mask = global_attention_mask.masked_fill(window_mask.logical_not(), torch.finfo(self.dtype).min)

        return global_attention_mask, sliding_window_mask


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


@add_start_docstrings(
    "The ModernBert Model with a decoder head on top that is used for masked language modeling.",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertForMaskedLM(ModernBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
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
        self._maybe_set_compile()

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        if self.config._attn_implementation == "flash_attention_2":
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_modernbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    "The ModernBert Model with a sequence classification head on top that performs pooling.",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertForSequenceClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
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
        self._maybe_set_compile()
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape if input_ids is not None else inputs_embeds.shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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

        if self.config.is_causal:
            # we need to use EOS / no intermediate pooling
            nonpooled_output = self.head(last_hidden_state)
            nonpooled_output = self.drop(nonpooled_output)
            classifier_output = self.classifier(nonpooled_output)
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            logits = classifier_output[torch.arange(batch_size, device=classifier_output.device), sequence_lengths]
        else:
            if self.config.classifier_pooling == "cls":
                last_hidden_state = last_hidden_state[:, 0]
            elif self.config.classifier_pooling == "mean":
                last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                    dim=1, keepdim=True
                )

            pooled_output = self.head(last_hidden_state)
            pooled_output = self.drop(pooled_output)
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


@add_start_docstrings(
    "The ModernBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertForTokenClassification(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
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
        self._maybe_set_compile()
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
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

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """The ModernBert model usedfor Causal Language Modeling (CLM) tasks. Note that there are two methods
    of using the model: (1) with a causal head on top, as a standard decoder-only model and (2) as an
    encoder-only model using psudo-causal generation from iterative masking. The iterative method was
    proposed in "BERTs are Generative In-Context Learners" by David Samuel.""",
    MODERNBERT_START_DOCSTRING,
)
class ModernBertForCausalLM(ModernBertPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.model = ModernBertModel(config)

        if self.config.is_causal:
            self.lm_head = ModernBertPredictionHead(config)
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
        else:
            # using the psuedo-casual method. NOTE: be sure your tokenizer has `is_causal` set to True
            self.head = ModernBertPredictionHead(config)
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        head = self.lm_head if self.config.is_causal else self.head
        return self.decoder(head(output))

    def _prepare_masked_input_for_position(
        self,
        input_ids: torch.LongTensor,
        position: int,
    ) -> torch.LongTensor:
        """Masks current position and two future tokens as per paper."""
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Create new tensor with extra space for masks and EOS
        new_seq_len = seq_len + self.config.num_future_masks + 1  # Add space for masks and EOS
        masked_input = torch.zeros((batch_size, new_seq_len), dtype=input_ids.dtype, device=input_ids.device)

        # Copy original sequence
        masked_input[:, :seq_len] = input_ids

        # Mask current position and future tokens
        for i in range(self.config.num_future_masks + 1):
            mask_pos = position + i
            if mask_pos < new_seq_len:
                masked_input[:, mask_pos] = self.config.mask_token_id

        # Add EOS token at the end
        masked_input[:, -1] = self.config.eos_token_id if self.config.eos_token_id else self.config.sep_token_id

        return masked_input

    @add_start_docstrings_to_model_forward(MODERNBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        if self.config.is_causal:
            return self._forward_causal(
                input_ids,
                attention_mask,
                sliding_window_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                indices,
                cu_seqlens,
                max_seqlen,
                batch_size,
                seq_len,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                **kwargs,
            )
        else:
            return self._forward_pseudo_casual(
                input_ids,
                attention_mask,
                sliding_window_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                indices,
                cu_seqlens,
                max_seqlen,
                batch_size,
                seq_len,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                **kwargs,
            )

    def _forward_causal(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            # NOTE: by moving thet unpadding into ModernBertModel, we lose a bit of speed from not unpadding through the head
            # but the tradeoff is a much cleaner implementation since we can assume everything is padded here
            repad = True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            repad=repad,
        )
        last_hidden_state = outputs[0]

        lm_logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.lm_head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _forward_pseudo_casual(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_single_token_logits: Optional[bool] = False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape if input_ids is not None else inputs_embeds.shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if labels is not None:
            raise ValueError("Training is not supported for pseudo-causal generation")

        # For generation or single token prediction
        if return_single_token_logits:
            # For batched generation, all sequences will have same length, otherwise raise an error
            if self.config.mask_token_id in input_ids.any():
                raise ValueError("Mask token found in input_ids during generation")

            # Prepare masked input for the batch
            masked_input = self._prepare_masked_input_for_position(input_ids, seq_len)
            attention_mask = torch.ones_like(masked_input)

            outputs = self.model(input_ids=masked_input, attention_mask=attention_mask, **kwargs)

            output_logits = (
                self.compiled_head(outputs.last_hidden_state)
                if self.config.reference_compile
                else self.decoder(self.head(outputs.last_hidden_state))
            )

            # Get just the next token logits but reshape to match expected format
            new_logit = output_logits[:, seq_len, :].unsqueeze(1)  # Shape: (batch_size, 1, vocab_size)
            seq_len = input_ids.shape[1]

            # Create zero logits for all positions except the last
            zero_logits = torch.zeros(
                (batch_size, seq_len - 2, self.config.vocab_size), device=new_logit.device, dtype=new_logit.dtype
            )

            # Concat to get proper shape while ensuring only the last position has real logits, even with batches
            logits = torch.cat([zero_logits, new_logit], dim=1)

        else:
            # NOTE: must turn every token into a instance, since it can only predict logits for one token at a time
            # This means the batch size will explode. If you get OOM errors, try reducing the batch size to one
            batched_inputs = torch.zeros(
                (
                    batch_size * (seq_len + self.config.num_future_masks + 1),
                    (seq_len + self.config.num_future_masks + 1),
                ),
                dtype=input_ids.dtype,
                device=device,
            )

            for pos in range(seq_len):
                start_idx = pos * batch_size
                end_idx = (pos + 1) * batch_size
                batched_inputs[start_idx:end_idx] = self._prepare_masked_input_for_position(input_ids, pos)

            batched_attention_mask = torch.triu(torch.ones_like(batched_inputs), diagonal=1)

            outputs = self.model(input_ids=batched_inputs, attention_mask=batched_attention_mask, **kwargs)
            output_logits = (
                self.compiled_head(outputs.last_hidden_state)
                if self.config.reference_compile
                else self.decoder(self.head(outputs.last_hidden_state))
            )
            all_logits = torch.zeros_like(output_logits)

            for pos in range(seq_len):
                start_idx = pos * batch_size
                end_idx = (pos + 1) * batch_size
                all_logits[:, pos, :] = output_logits[start_idx:end_idx, pos, :]

            logits = all_logits

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,  # can't re-use key value cache cuz bidirectional
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorders the cache for beam search during generation."""
        # If we don't use cache, just return None
        if past_key_values is None:
            return None

        if self.config._attn_implementation != "eager":
            raise ValueError("Cache usage is only supported for eager attention")

        # If we do use cache, reorder it
        reordered_past = ()
        for layer_past in past_key_values:
            # Reorder each layer's cache according to beam_idx
            reordered_layer_past = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device)) if past_state is not None else None
                for past_state in layer_past
            )
            reordered_past = reordered_past + (reordered_layer_past,)
        return reordered_past


__all__ = [
    "ModernBertConfig",
    "ModernBertModel",
    "ModernBertPreTrainedModel",
    "ModernBertForMaskedLM",
    "ModernBertForSequenceClassification",
    "ModernBertForTokenClassification",
    "ModernBertForCausalLM",
]
