# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...utils.import_utils import is_einops_available
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertAttention,
    BertCrossAttention,
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertIntermediate,
    BertLayer,
    BertLMHeadModel,
    BertLMPredictionHead,
    BertModel,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertOutput,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertPreTrainingHeads,
    BertSelfAttention,
    BertSelfOutput,
)


def _check_einops_available():
    if not is_einops_available():
        raise ImportError(
            "NomicBERT requires the `einops` library. "
            "Please install it with `pip install einops` or "
            "`pip install transformers[torch]`."
        )


class NomicBertConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`NomicBertModel`]. It is used to instantiate an NomicBERT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [nomic-ai/nomic-bert-2048](https://huggingface.co/nomic-ai/nomic-bert-2048).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        rotary_emb_fraction (`float`, *optional*, defaults to 0.0):
            Fraction of the hidden size used for rotary embeddings.
        rotary_emb_base (`int`, *optional*, defaults to 10000):
            Base for the rotary embeddings.
        rotary_emb_scale_base (`float`, *optional*):
            Scale base for the rotary embeddings.
        rotary_emb_interleaved (`bool`, *optional*, defaults to `False`):
            Whether to use interleaved rotary embeddings.
        type_vocab_size (`int`, *optional*, defaults to 2):
            The size of the token type (segment) vocabulary. Used to distinguish different portions of the input,
            such as sentence A and sentence B in pairwise classification tasks.
        pad_vocab_size_multiple (`int`, *optional*, defaults to 1):
            pads the vocabulary size to a multiple (e.g. 8, 64, 128)
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the input and output word embeddings. If set to `True`, the same embedding matrix
            is used for both input embeddings and output logits.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        pad_token_id (`int`, *optional*, defaults to 0):
            The token ID used for padding.

    ```python
    >>> from transformers import NomicBertModel, NomicBertConfig

    >>> # Initializing a NomicBERT 2048 style configuration
    >>> configuration = NomicBertConfig()

    >>> # Initializing a model from the NomicBERT 2048 style configuration
    >>> model = NomicBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "nomic_bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_cache=True,
        classifier_dropout=None,
        rotary_emb_fraction=0.0,
        rotary_emb_base=10_000,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        type_vocab_size=2,
        pad_vocab_size_multiple=1,
        tie_word_embeddings=True,
        max_position_embeddings=2048,
        pad_token_id=0,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            classifier_dropout=classifier_dropout,
            type_vocab_size=type_vocab_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.rotary_emb_fraction = rotary_emb_fraction
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.rotary_emb_interleaved = rotary_emb_interleaved
        self.pad_vocab_size_multiple = pad_vocab_size_multiple


class NomicBertEmbeddings(BertEmbeddings):
    """
    NomicBERT embeddings adapted from BertEmbeddings.
    Overrides embedding layer only if vocab padding is required.
    """

    def __init__(self, config):
        super().__init__(config)

        if getattr(config, "pad_vocab_size_multiple", None) and config.pad_vocab_size_multiple > 1:
            padded_vocab_size = self._round_to_multiple(config.vocab_size, config.pad_vocab_size_multiple)
            self.word_embeddings = nn.Embedding(
                padded_vocab_size,
                config.hidden_size,
                padding_idx=config.pad_token_id,
            )

    def _round_to_multiple(self, value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple


class NomicBertSelfAttention(BertSelfAttention):
    """
    Custom Self-Attention mechanism for NomicBERT.
    Key Difference: Replaces standard BERT absolute position embeddings with
    Rotary Positional Embeddings (RoPE) applied directly to Q and K.
    """

    def __init__(self, config, position_embedding_type=None, is_causal=False, layer_idx=None):
        super().__init__(config, position_embedding_type=position_embedding_type)

        self.layer_idx = layer_idx
        self.is_causal = is_causal

        rotary_dim = int(self.attention_head_size * config.rotary_emb_fraction)
        # Initialize the RoPE module.
        if rotary_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                dim=rotary_dim,
                base=config.rotary_emb_base,
                scale_base=config.rotary_emb_scale_base,
                interleaved=config.rotary_emb_interleaved,
            )
        else:
            self.rotary_emb = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        position_ids=None,
        **kwargs,
    ):
        # Let BERT do QKV projection
        batch_size, seq_len, hidden_size = hidden_states.size()
        num_heads = self.num_attention_heads
        head_size = hidden_size // num_heads

        query_layer = self.query(hidden_states).view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
        key_layer = self.key(hidden_states).view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)
        value_layer = self.value(hidden_states).view(batch_size, seq_len, num_heads, head_size).permute(0, 2, 1, 3)

        # Calculate RoPE offset
        seq_len_offset = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                # New DynamicCache path
                seq_len_offset = past_key_values.get_seq_length(self.layer_idx)
            else:
                # Legacy tuple path
                seq_len_offset = past_key_values[0].shape[2]

        # Rotate Q and K here to encode relative positions.
        if self.rotary_emb is not None:
            q_rot = query_layer[..., : self.rotary_emb.dim]
            k_rot = key_layer[..., : self.rotary_emb.dim]

            # Use position_ids if available (fixes left-padding), else fallback to offset
            rope_offset = seq_len_offset
            if position_ids is not None:
                rope_offset = position_ids

            q_rot, k_rot = self.rotary_emb(q_rot, k_rot, seqlen_offset=rope_offset)

            query_layer = torch.cat([q_rot, query_layer[..., self.rotary_emb.dim :]], dim=-1)
            key_layer = torch.cat([k_rot, key_layer[..., self.rotary_emb.dim :]], dim=-1)

        if self.is_decoder or past_key_values is not None:
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    # DynamicCache handles concatenation internally and returns the full sequence
                    key_layer, value_layer = past_key_values.update(key_layer, value_layer, self.layer_idx)
                else:
                    # Legacy tuple logic (manual concatenation)
                    if past_key_values is not None:
                        key_layer = torch.cat([past_key_values[0], key_layer], dim=2)
                        value_layer = torch.cat([past_key_values[1], value_layer], dim=2)

                    past_key_values = (key_layer, value_layer)

        # Calculate Attention Scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # Scale scores by sqrt(d_model) to stabilize gradients
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply mask if present
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize to Probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Calculate Weighted Sum (Context)
        context_layer = torch.matmul(attention_probs, value_layer)
        # Re-assemble Heads (Standard BERT Logic)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Flatten 'Heads' and 'HeadDim' back into a single 'Hidden' dimension
        context_layer = context_layer.view(batch_size, seq_len, hidden_size)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder or past_key_values is not None:
            outputs = outputs + (past_key_values,)

        return outputs


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) module for applying rotary embeddings to query and key tensors.
    RoPE encodes relative positional information by rotating the query and key vectors in the complex plane.
    """

    def __init__(
        self,
        dim: int,
        base=10000,
        scale_base=None,
        interleaved=False,
        device=None,
    ):
        """
        Initialize RotaryEmbedding.

        Args:
            dim: The dimension of the rotary embeddings.
            base: The base frequency for computing inverse frequencies. Defaults to 10000.
            scale_base: Optional scaling base for the rotary embeddings. If provided, enables scaled rotary embeddings.
            interleaved: If True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style). Defaults to False.
            device: Optional device to initialize buffers on.
        """
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0
        self.base = float(base)
        # Generate and save the inverse frequency buffer (non trainable)
        # This computes 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _rotate_half(self, x, interleaved=False):
        """
        Rotate half of the input tensor for rotary embedding computation.

        For non-interleaved (GPT-NeoX style): splits tensor into two halves and rotates them.
        For interleaved (GPT-J style): rotates pairs of even and odd dimensions.

        Args:
            x: Input tensor of shape (..., headdim)
            interleaved: Whether to use interleaved rotation pattern

        Returns:
            Rotated tensor of the same shape as input
        """
        if not interleaved:
            # GPT-NeoX style
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        else:
            # GPT-J style
            x1, x2 = x[..., ::2], x[..., 1::2]
            _check_einops_available()
            from einops import rearrange

            return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    def apply_rotary_emb(self, x, cos, sin, offset=0, interleaved=False):
        """
        Apply rotary embeddings to input tensor.

        The rotary embedding is applied as: x_rot = x * cos + rotate_half(x) * sin
        Only the first `rotary_dim` dimensions are rotated; the rest remain unchanged.

        Args:
            x: Input tensor of shape (batch_size, seqlen, nheads, headdim)
            cos: Cosine values of shape (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
            sin: Sine values of shape (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
            offset: Offset for slicing cos/sin tensors. Used for KV cache scenarios.
            interleaved: Whether to use interleaved rotation pattern

        Returns:
            Tensor with rotary embeddings applied to the first `rotary_dim` dimensions
        """
        ro_dim = cos.shape[-1] * 2
        assert ro_dim <= x.shape[-1]
        cos, sin = (
            cos[offset : offset + x.shape[2]],
            sin[offset : offset + x.shape[2]],
        )

        _check_einops_available()
        from einops import repeat

        cos = repeat(cos, "s d -> 1 1 s (2 d)" if not interleaved else "s d -> 1 1 s (d 2)")
        sin = repeat(sin, "s d -> 1 1 s (2 d)" if not interleaved else "s d -> 1 1 s (d 2)")
        return torch.cat(
            [x[..., :ro_dim] * cos + self._rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
            dim=-1,
        )

    def _compute_inv_freq(self, device=None):
        """
        Compute inverse frequencies for rotary embeddings.

        Computes 1 / (base^(2i/dim)) for i in [0, dim/2), which are the frequencies
        used to generate the rotary embeddings.

        Args:
            device: Optional device to create the tensor on

        Returns:
            Tensor of shape (dim // 2,) containing inverse frequencies
        """
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        """
        Update the cached cosine and sine values for rotary embeddings.

        The cache is recomputed when:
        - The sequence length increases beyond the cached length
        - The device changes (e.g., during tracing)
        - The dtype changes
        - Switching from inference to training mode

        Args:
            seqlen: Target sequence length for the cache
            device: Device to create tensors on
            dtype: Data type for the output cos/sin tensors
        """
        # Reset the cache if sequence length has changed, device changed, dtype changed,
        # or switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)
            # Use fp32 for computation to maintain precision
            # inv_freq multiplied with t produces large values, and using bf16/fp16
            # would lose significant precision and change cos/sin outputs
            if self.inv_freq.dtype != torch.float32:
                inv_freq = self._compute_inv_freq(device=device)
            else:
                inv_freq = self.inv_freq
            # Compute frequencies: outer product of positions and inverse frequencies
            # Shape: (seqlen, dim // 2)
            # Note: Using torch.outer instead of einsum to avoid AMP converting fp32 to fp16
            freqs = torch.outer(t, inv_freq)

            # Compute cosine and sine, then convert to target dtype
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch_size, seqlen, nheads, headdim)
            k: Key tensor of shape (batch_size, seqlen, nheads, headdim)
            seqlen_offset: Offset for sequence positions. Can be:
                - An integer: all sequences are shifted by this amount (common in KV cache scenarios)
                - A tensor of shape (batch_size,): each sequence has its own offset
                Defaults to 0.
            max_seqlen: Maximum sequence length for cache update. If provided and seqlen_offset
                is a tensor, updates the cache up to this length. Useful for batched inference
                with variable-length sequences.

        Returns:
            Tuple of (q_rot, k_rot) with rotary embeddings applied. Both tensors have the same
            shape as the input q and k tensors.
        """
        seqlen = q.shape[2]
        if seqlen > self._seq_len_cached:
            self._update_cos_sin_cache(seqlen, device=q.device, dtype=q.dtype)
        elif max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=q.device, dtype=q.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=q.device, dtype=q.dtype)

        q_rot = self.apply_rotary_emb(q, self._cos_cached, self._sin_cached, seqlen_offset, self.interleaved)
        k_rot = self.apply_rotary_emb(k, self._cos_cached, self._sin_cached, seqlen_offset, self.interleaved)
        return q_rot, k_rot


class NomicBertCrossAttention(BertCrossAttention):
    pass


class NomicBertSelfOutput(BertSelfOutput):
    pass


class NomicBertAttention(BertAttention):
    """
    This module overrides the standard `BertAttention` to incorporate Rotary Positional Embeddings (RoPE)
    via `NomicBertSelfAttention` and `position_ids` propagation. It handles the specific
    initialization requirements of NomicBERT while maintaining compatibility with the
    Transformers library build system.
    """

    def __init__(self, config, position_embedding_type=None, layer_idx=None, is_cross_attention=False):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.self = NomicBertSelfAttention(
            config, position_embedding_type=position_embedding_type, layer_idx=layer_idx
        )

        self.is_cross_attention = is_cross_attention

        # Explicitly define attention_class to satisfy the linter/build system
        attention_class = NomicBertCrossAttention if is_cross_attention else NomicBertSelfAttention

        if is_cross_attention:
            self.self = attention_class(config, position_embedding_type=position_embedding_type)
        else:
            self.self = attention_class(config, position_embedding_type=position_embedding_type, layer_idx=layer_idx)

        self.output = NomicBertSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        position_ids=None,
    ):
        """
        Forward pass for the NomicBERT Attention layer.

        Args:
            hidden_states (`torch.Tensor`):
                Input hidden states of shape `(batch_size, seq_len, hidden_size)`.
            attention_mask (`torch.FloatTensor`, *optional*):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in `[0, 1]`: 1 for tokens that are **not masked**, 0 for masked tokens.
            head_mask (`torch.FloatTensor`, *optional*):
                Mask to nullify selected heads of the self-attention modules.
                Mask values selected in `[0, 1]`: 1 indicates the head is **not masked**, 0 indicates the head is **masked**.
            encoder_hidden_states (`torch.FloatTensor`, *optional*):
                Hidden states of the encoder (for cross-attention).
            encoder_attention_mask (`torch.FloatTensor`, *optional*):
                Mask to avoid performing attention on the encoder outputs (for cross-attention).
            past_key_values (`Tuple[Tuple[torch.FloatTensor]]`, *optional*):
                Cached key and value states from previous steps for fast decoding.
            output_attentions (`bool`, *optional*):
                Whether to return the attention probabilities.
            position_ids (`torch.LongTensor`, *optional*):
                Indices of positions of each input sequence token in the position embeddings.
                Required for accurate Rotary Embedding calculations during generation.

        Returns:
            `Tuple[torch.Tensor]`:
                A tuple containing:
                - **attention_output** (`torch.Tensor`): The output of the attention layer.
                - **attention_probs** (`torch.Tensor`, *optional*): Returned if `output_attentions=True`.
                - **past_key_values** (`Tuple`, *optional*): Returned if `is_decoder=True` or `past_key_values` were passed.
        """
        # Call SelfAttention
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            output_attentions,
            position_ids=position_ids,
        )

        # Process context layer (always index 0)
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class NomicBertIntermediate(BertIntermediate):
    """
    NomicBERT Intermediate layer.
    Replaces standard BERT GELU with SwiGLU (Swish-Gated Linear Unit).

    Standard BERT: Activation(Linear(x))
    NomicBERT:     SiLU(Gate(x)) * Value(x)
    """

    def __init__(self, config):
        super().__init__(config)
        # Add the Gate Layer
        # SwiGLU needs a second parallel layer for the gate.
        self.dense_gate = nn.Linear(config.hidden_size, config.intermediate_size)

        # Force Activation to SiLU (Swish)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = nn.SiLU()
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Compute the Gate (Project + Swish Activation)
        gate_output = self.intermediate_act_fn(self.dense_gate(hidden_states))

        # Compute the Value (Project Linear)
        value_output = self.dense(hidden_states)

        # Element-wise Multiplication (Gating)
        intermediate_output = gate_output * value_output

        return intermediate_output


class NomicBertOutput(BertOutput):
    pass


class NomicBertLayer(BertLayer):
    """
    NomicBERT Layer.
    Overrides standard BERT components to incorporate:
    Rotary Positional Embeddings (RoPE) in the Attention mechanism.
    And SwiGLU activation in the Intermediate layer.
    """

    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.attention = NomicBertAttention(config, layer_idx=layer_idx)
        self.intermediate = NomicBertIntermediate(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        position_ids=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class NomicBertEncoder(BertEncoder):
    """
    NomicBERT Encoder.
    Inherits from BertEncoder but allows for custom layer classes (like NomicBertLayer)
    to be passed during initialization via kwargs.
    """

    def __init__(self, config, layer_class=None, **kwargs):
        super().__init__(config, **kwargs)

        # Use NomicBertLayer by default if not specified
        layer_class = layer_class or NomicBertLayer

        # Re-initialize self.layer with the correct index passed to each layer
        self.layer = nn.ModuleList([layer_class(config, layer_idx=i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        if use_cache and isinstance(past_key_values, Cache):
            next_decoder_cache = past_key_values

        for i, layer in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = None

            if past_key_values is not None and not isinstance(past_key_values, Cache):
                past_key_value = past_key_values[i]

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                position_ids=position_ids,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                expected_len = 1 + (1 if output_attentions else 0) + 1
                if len(layer_outputs) == expected_len:
                    cache_to_add = layer_outputs[-1]
                    if isinstance(cache_to_add, Cache):
                        next_decoder_cache = cache_to_add
                    else:
                        next_decoder_cache += (cache_to_add,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class NomicBertPooler(BertPooler):
    pass


class NomicBertPredictionHeadTransform(BertPredictionHeadTransform):
    pass


class NomicBertLMPredictionHead(BertLMPredictionHead):
    pass


class NomicBertOnlyMLMHead(BertOnlyMLMHead):
    pass


class NomicBertOnlyNSPHead(BertOnlyNSPHead):
    pass


class NomicBertPreTrainingHeads(BertPreTrainingHeads):
    pass


class NomicBertPreTrainedModel(BertPreTrainedModel):
    pass


class NomicBertForPreTrainingOutput(BertForPreTrainingOutput):
    pass


class NomicBertModel(BertModel):
    """
    NomicBERT Model transformer outputting raw hidden-states without any specific head on top.
    It overrides the embeddings, encoder, and pooler to use the NomicBERT-specific implementations.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.embeddings = NomicBertEmbeddings(config)
        self.encoder = NomicBertEncoder(config, layer_class=NomicBertLayer)
        self.pooler = NomicBertPooler(config) if add_pooling_layer else None

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`
            token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence token in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.
            head_mask (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`
            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`):
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
                blocks) that can be used to speed up sequential decoding.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

        Returns:
            [`~modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions`] or `tuple(torch.FloatTensor)`:
            A [`~modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions`] (if `return_dict=True` is passed or the `config.use_return_dict=True`) or a tuple of `torch.FloatTensor` comprising various elements depending on the configuration (`NomicBertConfig`) and inputs.

            - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
            - **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
            - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            - **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            - **past_key_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- Tuple of `torch.FloatTensor` of length `config.n_layers`, with each tuple containing the cached key and value states of the self-attention blocks.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Generate position_ids
        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device).expand(input_shape)
            else:
                position_ids = self.embeddings.create_position_ids_from_input_ids(
                    input_ids, padding_idx=self.config.pad_token_id, past_key_values_length=0
                )

        if attention_mask is None:
            past_length = 0
            if past_key_values is not None:
                if not isinstance(past_key_values, Cache):
                    past_length = past_key_values[0][0].shape[2]
            attention_mask = torch.ones(((batch_size, seq_length + past_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class NomicBertForPreTraining(BertForPreTraining):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"


class NomicBertLMHeadModel(BertLMHeadModel):
    pass


class NomicBertForMaskedLM(BertForMaskedLM):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"


class NomicBertForNextSentencePrediction(BertForNextSentencePrediction):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"


class NomicBertForSequenceClassification(BertForSequenceClassification):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"


class NomicBertForMultipleChoice(BertForMultipleChoice):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"


class NomicBertForTokenClassification(BertForTokenClassification):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"


class NomicBertForQuestionAnswering(BertForQuestionAnswering):
    config_class = NomicBertConfig
    base_model_prefix = "nomic_bert"


__all__ = [
    "NomicBertConfig",
    "NomicBertForMaskedLM",
    "NomicBertForMultipleChoice",
    "NomicBertForNextSentencePrediction",
    "NomicBertForPreTraining",
    "NomicBertForQuestionAnswering",
    "NomicBertForSequenceClassification",
    "NomicBertForTokenClassification",
    "NomicBertLayer",
    "NomicBertLMHeadModel",
    "NomicBertModel",
    "NomicBertPreTrainedModel",
]
