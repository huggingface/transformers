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
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, is_flash_attn_2_available, logging
from ...utils.generic import is_flash_attention_requested
from ...utils.import_utils import is_triton_available
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding, apply_rotary_pos_emb


if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
else:
    RotaryEmbedding = object


logger = logging.get_logger(__name__)


class ModernBertConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ModernBertModel`]. It is used to instantiate an ModernBert
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ModernBERT-base.
    e.g. [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

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
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        local_attention (`int`, *optional*, defaults to 128):
            The window size for local attention.
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
    default_theta = {"global": 160_000.0, "local": 10_000.0}

    def __init__(
        self,
        vocab_size: int | None = 50368,
        hidden_size: int | None = 768,
        intermediate_size: int | None = 1152,
        num_hidden_layers: int | None = 22,
        num_attention_heads: int | None = 12,
        hidden_activation: str | None = "gelu",
        max_position_embeddings: int | None = 8192,
        initializer_range: float | None = 0.02,
        initializer_cutoff_factor: float | None = 2.0,
        norm_eps: int | None = 1e-5,
        norm_bias: bool | None = False,
        pad_token_id: int | None = 50283,
        eos_token_id: int | None = 50282,
        bos_token_id: int | None = 50281,
        cls_token_id: int | None = 50281,
        sep_token_id: int | None = 50282,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        layer_types: list[str] | None = None,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        local_attention: int | None = 128,
        embedding_dropout: float | None = 0.0,
        mlp_bias: bool | None = False,
        mlp_dropout: float | None = 0.0,
        decoder_bias: bool | None = True,
        classifier_pooling: Literal["cls", "mean"] = "cls",
        classifier_dropout: float | None = 0.0,
        classifier_bias: bool | None = False,
        classifier_activation: str | None = "gelu",
        deterministic_flash_attn: bool | None = False,
        sparse_prediction: bool | None = False,
        sparse_pred_ignore_index: int | None = -100,
        reference_compile: bool | None = None,
        repad_logits_with_grad: bool | None = False,
        **kwargs,
    ):
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
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.local_attention = local_attention
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

        if self.classifier_pooling not in ["cls", "mean"]:
            raise ValueError(
                f'Invalid value for `classifier_pooling`, should be either "cls" or "mean", but is {self.classifier_pooling}.'
            )

        self.layer_types = layer_types

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self.global_attn_every_n_layers = kwargs.get("global_attn_every_n_layers", 3)

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool(i % self.global_attn_every_n_layers) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        self.rope_parameters = rope_parameters
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation=None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`. If we find `rope_parameters`
        # as arg in the inputs, we can safely assume that it is in the new format. New naming used -> new format
        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)
            self.rope_parameters["sliding_attention"].update(rope_scaling)
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("global_rope_theta", self.default_theta["global"])
        )
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("local_rope_theta", self.default_theta["local"])
        )

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs

    def to_dict(self):
        output = super().to_dict()
        output.pop("reference_compile", None)
        return output


def _unpad_modernbert_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None, torch.Tensor | None]:
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
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
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
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
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
        max_seqlen: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        max_seqlen: if max_seqlen, device, and dtype are provided, we precompute the cos_sin_cache
            up to max_seqlen. If the max_seqlen, device, or dtype during training/inference differ,
            the cos_sin_cache will be recomputed during the forward pass.
        """
        super().__init__(dim=dim, base=base, device=device, interleaved=False)
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        self, input_ids: torch.LongTensor | None = None, inputs_embeds: torch.Tensor | None = None
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


class ModernBertRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: ModernBertConfig, device=None):
        super().__init__(config, device)

    @staticmethod
    def compute_default_rope_parameters(
        config: ModernBertConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        return super().compute_default_rope_parameters(config, device, seq_len, layer_type)


def eager_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    position_embeddings: torch.Tensor,
    output_attentions: bool | None = False,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = position_embeddings
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def flash_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    rotary_emb: ModernBertUnpaddedRotaryEmbedding,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    **_kwargs,
) -> tuple[torch.Tensor]:
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
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
        )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
        )
    return (attn.view(bs, dim),)


def sdpa_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: torch.LongTensor | None,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    position_embeddings: torch.Tensor,
    **_kwargs,
) -> tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = position_embeddings
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=attention_mask,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


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

    def __init__(self, config: ModernBertConfig, layer_id: int | None = None):
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
        layer_type = config.layer_types[layer_id]

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
            max_position_embeddings = config.local_attention
        else:
            self.local_attention = (-1, -1)
            max_position_embeddings = config.max_position_embeddings

        if is_flash_attention_requested(config):
            rope_parameters_dict = (
                self.config.rope_parameters[layer_type] if layer_type is not None else self.config.rope_parameters
            )
            rope_theta = rope_parameters_dict["rope_theta"]
            self.rotary_emb = ModernBertUnpaddedRotaryEmbedding(
                dim=self.head_dim, max_seqlen=max_position_embeddings, base=rope_theta
            )
        else:
            self.rotary_emb = None

        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if is_flash_attention_requested(self.config):
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
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


class ModernBertEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ModernBertConfig, layer_id: int | None = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)
        self.attention_type = config.layer_types[layer_id]

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        position_embeddings: torch.Tensor | None = None,
        output_attentions: bool | None = False,
    ) -> torch.Tensor:
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_outputs[0]
        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        hidden_states = hidden_states + mlp_output

        return (hidden_states,) + attn_outputs[1:]  # add attentions if outputted


@auto_docstring
class ModernBertPreTrainedModel(PreTrainedModel):
    config: ModernBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ModernBertEmbeddings", "ModernBertEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = False

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    init.zeros_(module.bias)

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
        elif isinstance(
            module,
            (
                ModernBertForSequenceClassification,
                ModernBertForMultipleChoice,
                ModernBertForTokenClassification,
                ModernBertForQuestionAnswering,
            ),
        ):
            init_weight(module.classifier, stds["final_out"])
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, ModernBertRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, ModernBertUnpaddedRotaryEmbedding):
            inv_freq = module._compute_inv_freq()
            init.copy_(module.inv_freq, inv_freq)

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False
    ) -> str:
        """
        Checks and dispatches to hhe requested attention implementation.
        """
        # If the user didn't specify anything, try to use flash_attention_2 if available.
        # Otherwise we fall back to the default SDPA -> Eager from the super() method.
        # ModernBert's FA2 implementation correctly handles non-fp16/bf16 dtypes, we don't
        # need the FA2 warning for non-fp16/bf16 dtypes so we set fp16 for the FA2 check.

        try:
            attn_implementation = (
                "flash_attention_2"
                if attn_implementation is None and self._flash_attn_2_can_dispatch()
                else attn_implementation
            )
        except (ValueError, ImportError):
            pass
        return super()._check_and_adjust_attn_implementation(
            attn_implementation=attn_implementation, is_init_check=is_init_check
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


@auto_docstring
class ModernBertModel(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.rotary_emb = ModernBertRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutput:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
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
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

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

        repad = False
        if is_flash_attention_requested(self.config):
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
            if position_ids is None:
                position_ids = indices.unsqueeze(0)
        else:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            attention_mask, sliding_window_mask = self._update_attention_mask(
                attention_mask, output_attentions=output_attentions
            )

        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
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
        # If the attention implementation is FA2 and there is no need for repadding, there might still be the batch
        # dimension missing
        elif (
            is_flash_attention_requested(self.config)
            and all_hidden_states is not None
            and all_hidden_states[-1].dim() == 2
        ):
            hidden_states = hidden_states.unsqueeze(0)
            all_hidden_states = tuple(hs.unsqueeze(0) for hs in all_hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _update_attention_mask(self, attention_mask: torch.Tensor, output_attentions: bool) -> torch.Tensor:
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


@auto_docstring(
    custom_intro="""
    The ModernBert Model with a decoder head on top that is used for masked language modeling.
    """
)
class ModernBertForMaskedLM(ModernBertPreTrainedModel):
    _tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}

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

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | MaskedLMOutput:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
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
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        if is_flash_attention_requested(self.config):
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
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        if is_flash_attention_requested(self.config):
            # Logits padding
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_modernbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)
            # Hidden states padding
            if getattr(outputs, "hidden_states", None) is not None:
                padded_hidden_states = []
                for hs in outputs.hidden_states:
                    if hs.dim() == 3 and hs.shape[0] == 1:
                        hs = hs.squeeze(0)
                    padded_hidden_states.append(
                        _pad_modernbert_output(inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len)
                    )
                outputs.hidden_states = tuple(padded_hidden_states)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernBert Model with a sequence classification head on top that performs pooling.
    """
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

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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


@auto_docstring(
    custom_intro="""
    The ModernBert Model with a token classification head on top, e.g. for Named Entity Recognition (NER) tasks.
    """
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

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | TokenClassifierOutput:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
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
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

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


@auto_docstring
class ModernBertForQuestionAnswering(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        start_positions: torch.Tensor | None = None,
        end_positions: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | QuestionAnsweringModelOutput:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
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
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
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

        last_hidden_state = self.head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The ModernBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.
    """
)
class ModernBertForMultipleChoice(ModernBertPreTrainedModel):
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config

        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.drop = torch.nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        batch_size: int | None = None,
        seq_len: int | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | MultipleChoiceModelOutput:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors.
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
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        self._maybe_set_compile()

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
        last_hidden_state = outputs[0]  # shape (num_choices, seq_len, hidden_size)

        # If classifier_pooling is "cls", isolate the <cls> token
        if self.config.classifier_pooling == "cls":
            indices_0 = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
            # for left or right padding, <cls> is the first non-pad token
            if attention_mask is not None:
                cls_mask = attention_mask.argmax(dim=-1).to(last_hidden_state.device)
            # if no pad, <cls> is the first token
            else:
                cls_mask = torch.tensor(0, dtype=torch.long, device=last_hidden_state.device)
            # extract the <cls> token for the logits
            last_hidden_state = last_hidden_state[indices_0, cls_mask]

        # If classifier_pooling is "mean", pool the hidden states by averaging over the sequence length
        elif self.config.classifier_pooling == "mean":
            num_non_pad_tokens = attention_mask.sum(dim=1, keepdim=True)
            last_hidden_state = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / num_non_pad_tokens

        pooled_output = self.head(last_hidden_state)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ModernBertConfig",
    "ModernBertModel",
    "ModernBertPreTrainedModel",
    "ModernBertForMaskedLM",
    "ModernBertForSequenceClassification",
    "ModernBertForTokenClassification",
    "ModernBertForQuestionAnswering",
    "ModernBertForMultipleChoice",
]
