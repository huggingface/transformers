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
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...integrations import use_kernel_func_from_hub, use_kernelized_func
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters, dynamic_rope_update
from ...utils.generic import maybe_autocast
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertAttention,
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
    defaults will yield a similar configuration to that of the [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

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
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
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
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
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
        self.rope_parameters = rope_parameters


class NomicBertEmbeddings(BertEmbeddings):
    pass

class NomicBertRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: NomicBertConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: NomicBertConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
@use_kernelized_func(apply_rotary_pos_emb)
class NomicBertSelfAttention(BertSelfAttention):
    """
    Custom Self-Attention mechanism for NomicBERT.
    Key Difference: Replaces standard BERT absolute position embeddings with
    Rotary Positional Embeddings (RoPE) applied directly to Q and K.
    """

    def __init__(self, config, position_embedding_type=None, is_causal=False, layer_idx=None):
        super().__init__(config, position_embedding_type=position_embedding_type)

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
        position_embeddings=None,
        cache_position = None,
        **kwargs,
    ):
        batch_size, seq_len, hidden_size = hidden_states.size()
        # Let BERT do QKV projection
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Apply Rotary Position Embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        # Fallback
        elif position_ids is not None and isinstance(position_ids, tuple) and len(position_ids) == 2:
            cos, sin = position_ids
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        # Handle KV Cache
        if past_key_values is not None:
            if not isinstance(past_key_values, Cache):
                key_layer, value_layer = past_key_values.update(key_layer, value_layer, self.layer_idx)
            else:
                # Update DynamicCache
                cache_kwargs = {}
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

                key_layer, value_layer = past_key_values.update(
                    key_layer, value_layer, self.layer_idx, cache_kwargs
                )


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

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Calculate Weighted Sum (Context)
        context_layer = torch.matmul(attention_probs, value_layer)
        # Re-assemble Heads (Standard BERT Logic)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Flatten 'Heads' and 'HeadDim' back into a single 'Hidden' dimension
        new_context_layer_shape = context_layer.size()[:-2] + (hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder or past_key_values is not None:
            outputs = outputs + (past_key_values,)

        return outputs


class NomicBertSelfOutput(BertSelfOutput):
    pass


class NomicBertAttention(BertAttention):
    def __init__(
        self, config, position_embedding_type=None, layer_idx=None, is_cross_attention=False
    ):
        super().__init__(config, position_embedding_type=position_embedding_type)

        self.self = NomicBertSelfAttention(
            config, position_embedding_type=position_embedding_type, layer_idx=layer_idx
        )

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
            past_key_values (`Cache`, *optional*):
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
                - **past_key_values** (`Cache`, *optional*): Returned if `is_decoder=True` or `past_key_values` were passed.
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

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.gradient_checkpointing = False

        # Re-initialize self.layer with the correct index passed to each layer
        self.layer = nn.ModuleList([NomicBertLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])

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

        # DynamicCache is updated in place by attention layers
        if self.gradient_checkpointing and self.training:
            use_cache = False

        next_decoder_cache = past_key_values if use_cache else None

        for i, layer in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            past_key_value = None

            if past_key_values is not None:
                if not isinstance(past_key_values, Cache):
                    raise ValueError("NomicBert only supports Cache-based past_key_values")
                past_key_value = past_key_values

            if self.gradient_checkpointing and self.training:
                if not hidden_states.requires_grad:
                    hidden_states.requires_grad_(True)

                def custom_forward(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    position_ids,
                ):
                    # We manually reconstruct the call to layer() ensuring arguments match layer.forward signature
                    return layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        None,  # past_key_value is always None during checkpointing
                        output_attentions,
                        position_ids,
                    )

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    position_ids,
                    use_reentrant=False,  # Optional: Recommended for modern PyTorch
                )

            else:
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
    supports_gradient_checkpointing = True


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

    def _check_past_key_values_for_generate(self, past_key_values):
        if isinstance(past_key_values, Cache):
            return
        if past_key_values is None:
            return
        raise ValueError("NomicBert only supports Cache-based past_key_values during generation.")

    def get_head_mask(
        self, head_mask: torch.Tensor | None, num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> torch.Tensor:
        """
        Prepare the head mask if needed.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def _reorder_cache(self, past_key_values, beam_idx):
        # HF may pass legacy tuple caches here
        if past_key_values is None:
            return None

        # Convert legacy tuple -> Cache if needed
        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # Reorder in-place and return Cache
        past_key_values.reorder_cache(beam_idx)
        return past_key_values

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
            past_key_values (`Cache` , *optional*):
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

        if self.training:
            use_cache = False

        if not self.config.is_decoder:
            use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

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
        past_key_values_length = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            past_key_values_length = past_key_values.get_seq_length()

        if position_ids is None:
            if inputs_embeds is not None:
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = self.embeddings.create_position_ids_from_input_ids(
                    input_ids, padding_idx=self.config.pad_token_id, past_key_values_length=past_key_values_length
                )

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                if self.embeddings.token_type_ids.shape[1] < seq_length:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
                else:
                    buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                    token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=0,
        )

        extended_attention_mask = self.get_extended_attention_mask(attention_mask[:, -seq_length:], input_shape)
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
