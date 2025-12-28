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

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

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

class NomicBertConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`NomicBertModel`]. It is used to instantiate an NomicBERT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the [nomic-ai/nomic-bert-2048](https://huggingface.co/nomic-ai/nomic-bert-2048).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        rotary_emb_fraction (`float`, *optional*, defaults to 0.0):
            Fraction of the hidden size used for rotary embeddings.
        use_flash_attn (`bool`, *optional*, defaults to `False`):
            Whether to use Flash Attention.
        rotary_emb_base (`int`, *optional*, defaults to 10,000):
            Base for the rotary embeddings.
        rotary_emb_scale_base (`float`, *optional*, defaults to `None`):
            Scale base for the rotary embeddings.
        rotary_emb_interleaved (`bool`, *optional*, defaults to `False`):
            Whether to use interleaved rotary embeddings.
        use_rms_norm (`bool`, *optional*, defaults to `False`):
            Whether to use RMSNorm instead of LayerNorm.
        type_vocab_size (`int`, *optional*, defaults to 2):
            The size of the token type (segment) vocabulary. Used to distinguish different portions of the input,
            such as sentence A and sentence B in pairwise classification tasks.
        pad_vocab_size_multiple (`int`, *optional*, defaults to 1):
            pads the vocabulary size to a multiple (e.g. 8, 64, 128)
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the input and output word embeddings. If set to `True`, the same embedding matrix
            is used for both input embeddings and output logits.
        rotary_scaling_factor (`float`, *optional*, defaults to `None`):
            Scaling factor for the rotary embeddings (used for linear scaling).
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).

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
        rotary_emb_fraction=0.0,
        use_flash_attn=False,
        rotary_emb_base=10_000,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        use_rms_norm=False,
        type_vocab_size=2,
        pad_vocab_size_multiple=1,
        tie_word_embeddings=True,
        rotary_scaling_factor=None,
        max_position_embeddings=2048,
        **kwargs,
    ):
        super().__init__(
            type_vocab_size=type_vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            max_position_embeddings=max_position_embeddings,
            **kwargs
            )

        self.rotary_emb_fraction = rotary_emb_fraction
        self.use_flash_attn = use_flash_attn
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scale_base = rotary_emb_scale_base
        self.rotary_emb_interleaved = rotary_emb_interleaved
        self.use_rms_norm = use_rms_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.rotary_scaling_factor = rotary_scaling_factor


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
    
    def _round_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple



class NomicBertSelfAttention(BertSelfAttention):
    """
    Custom Self-Attention mechanism for NomicBERT.
    Key Difference: Replaces standard BERT absolute position embeddings with
    Rotary Positional Embeddings (RoPE) applied directly to Q and K.
    """
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type=position_embedding_type)

        # Initialize the RoPE module.
        self.rotary_emb = RotaryEmbedding(
            dim=int(self.attention_head_size * config.rotary_emb_fraction),
            base=config.rotary_emb_base,
            scale_base=config.rotary_emb_scale_base,
            interleaved=config.rotary_emb_interleaved,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # Let BERT do QKV projection
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer   = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Rotate Q and K here to encode relative positions.
        query_layer, key_layer = self.rotary_emb(query_layer, key_layer)

        # Calculate Attention Scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # Scale scores by sqrt(d_model) to stabilize gradients
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply mask if present
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize to Probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Calculate Weighted Sum (Context)
        context_layer = torch.matmul(attention_probs, value_layer)
        # Re-assemble Heads (Standard BERT Logic)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Flatten 'Heads' and 'HeadDim' back into a single 'Hidden' dimension
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    pass


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
    def __init__(self, config):
        super().__init__(config)
        self.attention.self = NomicBertSelfAttention(config)
        self.intermediate = NomicBertIntermediate(config)


class NomicBertEncoder(BertEncoder):
    pass


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
    pass


class NomicBertForPreTraining(BertForPreTraining):
    pass


class NomicBertLMHeadModel(BertLMHeadModel):
    pass


class NomicBertForMaskedLM(BertForMaskedLM):
    pass


class NomicBertForNextSentencePrediction(BertForNextSentencePrediction):
    pass


class NomicBertForSequenceClassification(BertForSequenceClassification):
    pass


class NomicBertForMultipleChoice(BertForMultipleChoice):
    pass


class NomicBertForTokenClassification(BertForTokenClassification):
    pass


class NomicBertForQuestionAnswering(BertForQuestionAnswering):
    pass


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

