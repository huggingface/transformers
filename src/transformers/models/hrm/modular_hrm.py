# coding=utf-8
# Copyright 2025 Sapient Inc. All rights reserved.
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
"""
PyTorch HRM (Hierarchical Reasoning Model) implementation.

The Hierarchical Reasoning Model (HRM) was proposed in "Hierarchical Reasoning Model" by Guan Wang, Jin Li,
Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori.

HRM is a novel recurrent neural network architecture designed for sequential reasoning tasks. It employs
hierarchical and multi-timescale processing with Adaptive Computation Time (ACT), achieving strong performance
on tasks like Sudoku solving, maze navigation, and abstract reasoning with minimal training data.

This implementation follows the architecture described in the paper: https://arxiv.org/abs/2506.21734

## Modular Design Notes

HRM follows the modular transformers pattern by inheriting reusable components from Llama where appropriate:
- `apply_rotary_pos_emb`, `rotate_half`: Inherited from Llama (identical implementations)
- `LlamaRotaryEmbedding`: Available for import but not used (see HrmRotaryEmbedding below)

However, HRM has several unique architectural components that cannot inherit from existing models:

### Custom Components (cannot inherit):

1. **HrmLinear, HrmEmbedding**: Use custom truncated normal initialization specific to HRM's training dynamics.
   Cannot inherit from standard nn.Linear/nn.Embedding due to initialization requirements.

2. **HrmRotaryEmbedding**: Precomputes cos/sin for all positions (simpler than LlamaRotaryEmbedding which
   supports dynamic rope scaling). HRM doesn't need rope_scaling or position_ids in forward().

3. **HrmAttention**: Uses fused QKV projection and custom initialization. While similar to standard attention,
   the projection layout and initialization make inheritance from LlamaAttention impractical.

4. **HrmCarry, HrmInner, HrmReasoningModule, HrmBlock**: Core hierarchical reasoning architecture unique to HRM.
   These implement the two-level (H/L) hierarchical processing and carry state mechanism.

5. **HrmModel, HrmForCausalLM**: Use carry state instead of KV cache, implement ACT halting mechanism,
   and have unique forward passes that cannot inherit from standard transformer models.

This design balances code reuse (via inherited functions) with the need to preserve HRM's novel architecture.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ..falcon_mamba.modeling_falcon_mamba import rms_forward
from ..llama.modeling_llama import apply_rotary_pos_emb


logger = logging.get_logger(__name__)


class HrmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HrmModel`]. It is used to instantiate a HRM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the HRM base model.

    The Hierarchical Reasoning Model (HRM) is a novel recurrent neural network architecture for sequential reasoning
    tasks featuring:
    - Two-level hierarchical processing inspired by human cognition
    - High-level (H) module: Slow, abstract planning and reasoning
    - Low-level (L) module: Fast, detailed computations
    - Adaptive Computation Time (ACT) mechanism with Q-learning based halting

    This model was introduced in the paper "Hierarchical Reasoning Model" by Guan Wang, Jin Li, Yuhao Sun,
    Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi Yadkori.
    For more details, see: https://arxiv.org/abs/2506.21734

    This model was contributed by [zbloss](https://huggingface.co/zbloss). The original code can be found
    [zbloss/HRM-sudoku-extreme](https://huggingface.co/zbloss/HRM-sudoku-extreme). Checkpoints for this model can be found
    on the Hugging Face Hub at [zbloss/HRM-sudoku-extreme](https://huggingface.co/zbloss/HRM-sudoku-extreme).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
            vocab_size (`int`, *optional*, defaults to 11):
                Vocabulary size of the HRM model. Defines the number of different tokens that can be represented by the
                `inputs_ids` passed when calling [`HrmModel`]. For reasoning tasks like Sudoku, this is typically 11
                (digits 0-9 plus a padding token).
            hidden_size (`int`, *optional*, defaults to 512):
                Dimension of the hidden representations and embeddings.
            num_hidden_layers (`int`, *optional*, defaults to 4):
                Number of transformer layers in both high-level (H) and low-level (L) modules.
                Use `h_layers` and `l_layers` to set them independently.
            h_layers (`int`, *optional*):
                Number of transformer layers in the high-level (H) module for abstract planning.
                If not specified, defaults to `num_hidden_layers`.
            l_layers (`int`, *optional*):
                Number of transformer layers in the low-level (L) module for detailed computations.
                If not specified, defaults to `num_hidden_layers`.
            num_attention_heads (`int`, *optional*, defaults to 8):
                Number of attention heads for each attention layer in the transformer blocks.
            intermediate_size (`int`, *optional*):
                Dimension of the MLP representations. If not specified, defaults to `hidden_size * expansion`.
            expansion (`float`, *optional*, defaults to 4.0):
                MLP expansion ratio for SwiGLU feed-forward layers. Used to calculate `intermediate_size`
                if not explicitly provided.
            max_position_embeddings (`int`, *optional*, defaults to 81):
                The maximum sequence length that this model might ever be used with. For Sudoku, this is 81 (9x9 grid).
                For ARC tasks, this can be up to 900 (30x30 grid).
            h_cycles (`int`, *optional*, defaults to 2):
                Number of high-level reasoning cycles per forward pass. Controls the depth of abstract planning.
            l_cycles (`int`, *optional*, defaults to 2):
                Number of low-level computation cycles per high-level cycle. Controls granularity of detailed processing.
            pos_encodings (`str`, *optional*, defaults to `"rope"`):
                Type of positional encoding to use. Options are "rope" (Rotary Position Embeddings) or "learned".
            rope_theta (`float`, *optional*, defaults to 10000.0):
                The base period of the RoPE embeddings. Only used when `pos_encodings="rope"`.
            rms_norm_eps (`float`, *optional*, defaults to 1e-05):
                The epsilon used by the RMS normalization layers for numerical stability.
            puzzle_emb_ndim (`int`, *optional*, defaults to 0):
                Dimension of per-puzzle sparse embeddings. Set to 0 to disable puzzle-specific embeddings.
                When > 0, each unique puzzle gets a learned embedding of this dimension.
            num_puzzle_identifiers (`int`, *optional*, defaults to 1):
                Total number of unique puzzle types/IDs for which to learn separate embeddings.
                Only used when `puzzle_emb_ndim > 0`.
            halt_max_steps (`int`, *optional*, defaults to 16):
                Maximum number of computation steps before forcing the ACT mechanism to halt.
                Controls the computational budget per sequence.
            halt_exploration_prob (`float`, *optional*, defaults to 0.1):
                Probability of exploration during ACT training. Used for Q-learning based adaptive halting.
            dtype (`str` or `torch.dtype`, *optional*, defaults to `"bfloat16"`):
                The dtype of the model's forward pass computations. Can be "bfloat16", "float32", or "float16".
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            use_cache (`bool`, *optional*, defaults to `False`):
                Whether or not the model should return the state for recurrent computation.
                HRM uses a unique state system for hierarchical processing.

    Example:
    ```python
    >>> from transformers import HrmConfig, HrmModel

    >>> # Initializing a HRM configuration for Sudoku solving
    >>> configuration = HrmConfig(
    ...     vocab_size=11,  # 0-9 digits + padding
    ...     hidden_size=512,
    ...     num_hidden_layers=4,
    ...     max_position_embeddings=81,  # 9x9 grid
    ...     h_cycles=2,
    ...     l_cycles=2,
    ... )

    >>> # Initializing a model from the configuration
    >>> model = HrmModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "hrm"

    def __init__(
        self,
        vocab_size=11,
        hidden_size=512,
        num_hidden_layers=4,
        h_layers=None,
        l_layers=None,
        num_attention_heads=8,
        intermediate_size=None,
        expansion=4.0,
        max_position_embeddings=81,
        h_cycles=2,
        l_cycles=2,
        pos_encodings="rope",
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        dtype="bfloat16",
        initializer_range=0.02,
        use_cache=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.h_layers = h_layers if h_layers is not None else num_hidden_layers
        self.l_layers = l_layers if l_layers is not None else num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.expansion = expansion
        self.intermediate_size = intermediate_size if intermediate_size is not None else int(hidden_size * expansion)
        self.max_position_embeddings = max_position_embeddings
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.pos_encodings = pos_encodings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.puzzle_emb_ndim = puzzle_emb_ndim
        self.num_puzzle_identifiers = num_puzzle_identifiers
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        super().__init__(dtype=dtype, **kwargs)


@dataclass
class HrmModelOutput(ModelOutput):
    """
    Output type of [`HrmModel`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        q_halt_logits (`torch.FloatTensor` of shape `(batch_size,)`):
            Q-values for halting in the Adaptive Computation Time mechanism.
        q_continue_logits (`torch.FloatTensor` of shape `(batch_size,)`):
            Q-values for continuing in the Adaptive Computation Time mechanism.
        carry (`HrmState`, *optional*):
            Model state for recurrent computation, containing hidden states and halting information.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
    """

    logits: torch.FloatTensor = None
    q_halt_logits: torch.FloatTensor = None
    q_continue_logits: torch.FloatTensor = None
    carry: Optional["HrmState"] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class HrmCausalLMOutput(ModelOutput):
    """
    Output type of [`HrmForCausalLM`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        q_halt_logits (`torch.FloatTensor` of shape `(batch_size,)`):
            Q-values for halting in the Adaptive Computation Time mechanism.
        q_continue_logits (`torch.FloatTensor` of shape `(batch_size,)`):
            Q-values for continuing in the Adaptive Computation Time mechanism.
        carry (`HrmState`, *optional*):
            Model state for recurrent computation, containing hidden states and halting information.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    q_halt_logits: torch.FloatTensor = None
    q_continue_logits: torch.FloatTensor = None
    carry: Optional["HrmState"] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


def truncated_normal_init_(
    tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
) -> torch.Tensor:
    """Initialize tensor with truncated normal distribution (JAX-style)."""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor




class HrmLinear(nn.Module):
    """
    Linear layer with automatic type casting for mixed-precision training.

    Note: Cannot inherit from nn.Linear due to custom truncated normal initialization
    that is critical to HRM's training dynamics. This custom initialization differs from
    standard PyTorch initialization and is applied during __init__.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(
            truncated_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


class HrmEmbedding(nn.Embedding):
    """Embedding layer with type casting for mixed-precision training."""

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        # Initialize parent without calling its __init__ to avoid double initialization
        nn.Module.__init__(self)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self.cast_to = cast_to
        self.weight = nn.Parameter(truncated_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight.to(self.cast_to))


class HrmRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) precomputation.

    Note: Does not inherit from LlamaRotaryEmbedding because HRM uses a simpler approach.
    HRM precomputes cos/sin for all positions at initialization and doesn't need:
    - Dynamic rope scaling (rope_scaling config)
    - Position-dependent forward passes (position_ids parameter)
    - Rope type variations (default/linear/dynamic/yarn/longrope)
    This simpler design is sufficient for HRM's fixed-length reasoning tasks.
    """

    def __init__(self, dim=None, max_position_embeddings=None, base=10000.0, device=None, config=None):
        super().__init__()
        # Support both direct parameters and config-based initialization
        if config is not None:
            dim = config.hidden_size // config.num_attention_heads
            max_position_embeddings = config.max_position_embeddings
            base = config.rope_theta

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - position_ids parameter is accepted for compatibility but not used."""
        return self.cos_cached, self.sin_cached


class HrmAttention(nn.Module):
    """
    Multi-head self-attention with FlashAttention and eager fallback.

    Note: Does not inherit from LlamaAttention due to architectural differences:
    - Uses fused QKV projection (single linear layer) instead of separate Q, K, V projections
    - Custom truncated normal initialization via HrmLinear
    - Simpler interface without cache_position, past_key_values, or output_attentions
    These differences make inheritance from LlamaAttention impractical while still reusing
    the core rope functions (apply_rotary_pos_emb, rotate_half).
    """

    def __init__(self, config: HrmConfig, causal: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.output_size = self.head_dim * config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_attention_heads
        self.causal = causal

        self.qkv_proj = HrmLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = HrmLinear(self.output_size, self.hidden_size, bias=False)

    def _eager_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch fallback attention when FlashAttention is not available."""
        batch, seq_len, num_heads, head_dim = query.shape
        # Transpose to (batch, num_heads, seq_len, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)

        # Apply causal mask if needed
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax and apply to values
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)

        # Transpose back to (batch, seq_len, num_heads, head_dim)
        return output.transpose(1, 2).contiguous()

    def forward(
        self, hidden_states: torch.Tensor, cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            # Slice RoPE embeddings to match actual sequence length
            cos = cos[:seq_len, :].unsqueeze(0)  # Add batch dimension: [1, seq_len, head_dim]
            sin = sin[:seq_len, :].unsqueeze(0)  # Add batch dimension: [1, seq_len, head_dim]
            # HRM uses (batch, seq_len, heads, head_dim) so unsqueeze_dim=2
            query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=2)

        # Try FlashAttention first, fall back to eager if not available
        try:
            attn_output = _flash_attention_forward(
                query_states=query,
                key_states=key,
                value_states=value,
                attention_mask=None,  # HRM doesn't use attention masks
                query_length=seq_len,
                is_causal=self.causal,
                dropout=0.0,  # HRM doesn't use attention dropout
            )
        except (ValueError, ImportError):
            # Fall back to eager attention if Flash Attention is not available
            attn_output = self._eager_attention(query, key, value)

        attn_output = attn_output.view(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class HrmSwiGLU(nn.Module):
    """SwiGLU feed-forward network layer."""

    def __init__(self, config: HrmConfig):
        super().__init__()
        # Use intermediate_size if provided, otherwise calculate from expansion
        if config.intermediate_size is not None:
            inter = config.intermediate_size
        else:
            inter = int(config.expansion * config.hidden_size * 2 / 3)
            # Round up to multiple of 256 for efficiency
            inter = ((inter + 255) // 256) * 256
        self.gate_up_proj = HrmLinear(config.hidden_size, inter * 2, bias=False)
        self.down_proj = HrmLinear(inter, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class HrmBlock(nn.Module):
    """Single transformer block with post-normalization."""

    def __init__(self, config: HrmConfig):
        super().__init__()
        self.self_attn = HrmAttention(config, causal=False)
        self.mlp = HrmSwiGLU(config)
        self.norm_eps = config.rms_norm_eps

    def forward(
        self, hidden_states: torch.Tensor, cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        # Post-normalization
        hidden_states = rms_forward(
            hidden_states + self.self_attn(hidden_states, cos_sin=cos_sin), variance_epsilon=self.norm_eps
        )
        hidden_states = rms_forward(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HrmReasoningModule(nn.Module):
    """Reasoning module for H-level or L-level processing."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
        return hidden_states


@dataclass
class HrmInnerState:
    """Internal state for the hierarchical reasoning modules."""

    high_level_state: torch.Tensor  # High-level hidden state for abstract reasoning
    low_level_state: torch.Tensor  # Low-level hidden state for detailed computations


@dataclass
class HrmState:
    """Complete state for HRM with ACT mechanism."""

    inner_state: HrmInnerState
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: dict


class HrmInner(nn.Module):
    """Core inner model implementing hierarchical reasoning with dual recurrence."""

    def __init__(self, config: HrmConfig):
        super().__init__()
        self.config = config
        self.forward_dtype = (
            getattr(torch, config.torch_dtype) if isinstance(config.torch_dtype, str) else config.torch_dtype
        )

        # Embeddings
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = HrmEmbedding(config.vocab_size, config.hidden_size, embed_init_std, self.forward_dtype)

        # Output heads
        self.lm_head = HrmLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = HrmLinear(config.hidden_size, 2, bias=True)

        # Puzzle embeddings (sparse, optional)
        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)  # ceil div
        if config.puzzle_emb_ndim > 0:
            # Simplified version without CastedSparseEmbedding
            self.puzzle_emb = nn.Embedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim)
            nn.init.zeros_(self.puzzle_emb.weight)

        # Positional encodings - always initialize appropriate type based on config
        self.pos_encodings = config.pos_encodings
        if self.pos_encodings == "rope":
            self.rotary_emb = HrmRotaryEmbedding(
                dim=config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif self.pos_encodings == "learned":
            self.embed_pos = HrmEmbedding(
                config.max_position_embeddings + self.puzzle_emb_len,
                config.hidden_size,
                embed_init_std,
                self.forward_dtype,
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {config.pos_encodings}")

        # Reasoning modules
        self.high_level_module = HrmReasoningModule(layers=[HrmBlock(config) for _ in range(config.h_layers)])
        self.low_level_module = HrmReasoningModule(layers=[HrmBlock(config) for _ in range(config.l_layers)])

        # Initial states
        self.register_buffer(
            "high_level_init_state",
            truncated_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.register_buffer(
            "low_level_init_state",
            truncated_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

    def _init_q_head_weights(self):
        """Apply special initialization to Q head for ACT halting mechanism.

        The Q head needs zero weights and strong negative bias (-5) to encourage
        initial exploration during Adaptive Computation Time learning.
        This is called after standard weight initialization.
        """
        self.q_head.weight.data.zero_()
        self.q_head.bias.data.fill_(-5.0)

    def _get_rotary_embeddings(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get rotary embeddings if using RoPE, otherwise return None."""
        if self.pos_encodings == "rope":
            return self.rotary_emb()
        return None

    def _input_embeddings(
        self, input: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute input embeddings with token, puzzle, and position encoding."""
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0 and puzzle_identifiers is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # Scale factor is 1/sqrt(2) to maintain variance after adding two embeddings
            embedding = math.sqrt(0.5) * (embedding + self.embed_pos.weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_state(self, batch_size: int, sequence_length: int, device: torch.device) -> HrmInnerState:
        """Create uninitialized state tensors.

        Args:
            batch_size (int): Batch size
            sequence_length (int): Sequence length (without puzzle embedding positions)
            device (torch.device): Device to create tensors on

        Returns:
            HrmInnerState: Uninitialized inner state
        """
        return HrmInnerState(
            high_level_state=torch.empty(
                batch_size,
                sequence_length + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            low_level_state=torch.empty(
                batch_size,
                sequence_length + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_state(self, reset_flag: torch.Tensor, state: HrmInnerState) -> HrmInnerState:
        """Reset state to learned initial values based on flags."""
        device = state.high_level_state.device
        high_init = self.high_level_init_state.to(device)
        low_init = self.low_level_init_state.to(device)
        return HrmInnerState(
            high_level_state=torch.where(reset_flag.view(-1, 1, 1), high_init, state.high_level_state),
            low_level_state=torch.where(reset_flag.view(-1, 1, 1), low_init, state.low_level_state),
        )

    def forward(
        self, state: HrmInnerState, batch: dict
    ) -> tuple[HrmInnerState, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Execute hierarchical reasoning forward pass with 1-step gradient."""
        # Get rotary embeddings if using RoPE (None for learned positional embeddings)
        cos_sin = self._get_rotary_embeddings()

        # Input encoding
        input_embeddings = self._input_embeddings(batch["input_ids"], batch.get("puzzle_identifiers"))

        # Forward iterations without gradients (for computational efficiency)
        with torch.no_grad():
            high = state.high_level_state
            low = state.low_level_state

            for high_cycle_idx in range(self.config.h_cycles):
                for low_cycle_idx in range(self.config.l_cycles):
                    # Skip the last L-level update (will be done with gradients)
                    if not ((high_cycle_idx == self.config.h_cycles - 1) and (low_cycle_idx == self.config.l_cycles - 1)):
                        low = self.low_level_module(low, high + input_embeddings, cos_sin=cos_sin)
                # Skip the last H-level update (will be done with gradients)
                if high_cycle_idx != self.config.h_cycles - 1:
                    high = self.high_level_module(high, low, cos_sin=cos_sin)

        # Final iteration with 1-step gradient for backpropagation
        low = self.low_level_module(low, high + input_embeddings, cos_sin=cos_sin)
        high = self.high_level_module(high, low, cos_sin=cos_sin)

        # Prepare outputs
        new_state = HrmInnerState(high_level_state=high.detach(), low_level_state=low.detach())
        output = self.lm_head(high)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(high[:, 0]).to(torch.float32)

        return new_state, output, (q_logits[..., 0], q_logits[..., 1])


class HrmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HrmConfig
    base_model_prefix = "hrm"
    supports_gradient_checkpointing = False
    _no_split_modules = ["HrmBlock"]
    _skip_keys_device_placement = ["high_level_init_state", "low_level_init_state"]

    def _init_weights(self, module):
        """Initialize the weights using truncated normal initialization."""
        std = self.config.initializer_range
        if isinstance(module, HrmLinear):
            # Reinitialize HrmLinear weights
            truncated_normal_init_(module.weight, std=std / (module.weight.shape[1] ** 0.5))
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, HrmEmbedding):
            # Reinitialize HrmEmbedding weights
            truncated_normal_init_(module.weight, std=std)
        elif isinstance(module, nn.Embedding):
            # For puzzle embeddings
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            # For any standard linear layers
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()


class HrmModel(HrmPreTrainedModel):
    """Hierarchical Reasoning Model with Adaptive Computation Time."""

    def __init__(self, config: HrmConfig):
        super().__init__(config)
        self.config = config
        self.inner = HrmInner(config)
        self.post_init()
        # Apply special Q head initialization after standard weight init
        self.inner._init_q_head_weights()

    def get_input_embeddings(self):
        return self.inner.embed_tokens

    def set_input_embeddings(self, value):
        self.inner.embed_tokens = value

    def initial_state(self, batch: dict) -> HrmState:
        """Initialize state for a new batch."""
        batch_size = batch["input_ids"].shape[0]
        sequence_length = batch["input_ids"].shape[1]
        device = batch["input_ids"].device
        return HrmState(
            inner_state=self.inner.empty_state(batch_size, sequence_length, device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        puzzle_identifiers: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        state: Optional[HrmState] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Execute one ACT step with halting logic."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        batch = {"input_ids": input_ids}
        if puzzle_identifiers is not None:
            batch["puzzle_identifiers"] = puzzle_identifiers
        if labels is not None:
            batch["labels"] = labels

        # Initialize state if not provided
        if state is None:
            state = self.initial_state(batch)

        # Update data and state
        new_inner_state = self.inner.reset_state(state.halted, state.inner_state)
        new_steps = torch.where(state.halted, torch.zeros_like(state.steps), state.steps)

        # Handle size mismatch during generation - update state data with batch data
        # Use scatter operations for efficient batch-wise updates
        new_current_data = {}

        # Separate keys into common and state-only for batch processing
        common_keys = [k for k in state.current_data.keys() if k in batch]
        state_only_keys = [k for k in state.current_data.keys() if k not in batch]

        # Get halted indices once for reuse across all keys
        halted_indices = state.halted.nonzero(as_tuple=False).squeeze(-1)

        # Process common keys using scatter operations
        for k in common_keys:
            batch_val = batch[k]
            state_val = state.current_data[k]

            # During generation, input_ids may be a single token while state has full sequence
            # Pad batch to match state size in this expected case
            if batch_val.shape[1] < state_val.shape[1]:
                pad_size = state_val.shape[1] - batch_val.shape[1]
                batch_val = F.pad(batch_val, (0, pad_size), value=0)

            # Use scatter operation to update halted sequences
            # Clone state and scatter batch values at halted positions (dim 0 is batch dimension)
            result = state_val.clone()
            if halted_indices.numel() > 0:
                result.index_copy_(0, halted_indices, batch_val.index_select(0, halted_indices))

            new_current_data[k] = result

        # Keep existing state data for keys not in current batch
        for k in state_only_keys:
            new_current_data[k] = state.current_data[k]

        # Forward inner model
        new_inner_state, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_state, new_current_data)

        # Halting logic
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and self.config.halt_max_steps > 1:
                halted = halted | (q_halt_logits > q_continue_logits)
                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits.float()) < self.config.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

        new_state = HrmState(new_inner_state, new_steps, halted, new_current_data)

        # Prepare hidden states and attentions if requested
        all_hidden_states = None
        all_attentions = None

        if output_hidden_states:
            # Return embeddings + H-level + L-level states to match transformers convention
            # Standard format: (embeddings, *layer_outputs)
            # For HRM: embeddings would be the input embeddings, then H and L level outputs
            # Note: We strip puzzle embedding positions to match input sequence length
            # During generation, only return hidden states for the input tokens (not the full carry)
            embeddings = self.inner._input_embeddings(
                new_current_data["input_ids"], new_current_data.get("puzzle_identifiers")
            )
            puzzle_len = self.inner.puzzle_emb_len

            # Get the sequence length from input_ids (which may be just 1 token during generation)
            seq_len = input_ids.shape[1]
            all_hidden_states = (
                embeddings[:, puzzle_len : puzzle_len + seq_len] if puzzle_len > 0 else embeddings[:, :seq_len],
                new_inner_state.high_level_state[:, puzzle_len : puzzle_len + seq_len]
                if puzzle_len > 0
                else new_inner_state.high_level_state[:, :seq_len],
                new_inner_state.low_level_state[:, puzzle_len : puzzle_len + seq_len]
                if puzzle_len > 0
                else new_inner_state.low_level_state[:, :seq_len],
            )

        # Note: HRM doesn't explicitly compute attention weights in the current implementation
        # so attentions will be None for now
        if output_attentions:
            all_attentions = ()  # Empty tuple as HRM doesn't expose attention weights

        if not return_dict:
            return (logits, q_halt_logits, q_continue_logits, new_state, all_hidden_states, all_attentions)

        return HrmModelOutput(
            logits=logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=q_continue_logits,
            carry=new_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class HrmForCausalLM(HrmPreTrainedModel, GenerationMixin):
    def __init__(self, config: HrmConfig):
        super().__init__(config)
        self.model = HrmModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return None  # HRM doesn't tie output embeddings

    def set_output_embeddings(self, new_embeddings):
        pass  # HRM doesn't tie output embeddings

    def prepare_inputs_for_generation(self, input_ids, state=None, **kwargs):
        """Prepare inputs for generation, handling dynamic sequence lengths."""
        # During generation, only pass the last token if state exists
        if state is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "state": state, **kwargs}

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        """Update model kwargs for next generation step."""
        model_kwargs["state"] = outputs.carry
        return model_kwargs

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        puzzle_identifiers: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        state: Optional[HrmState] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            puzzle_identifiers=puzzle_identifiers,
            labels=labels,
            state=state,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            **kwargs,
        )

        # Extract outputs
        if return_dict:
            logits = outputs.logits
            q_halt_logits = outputs.q_halt_logits
            q_continue_logits = outputs.q_continue_logits
            state_out = outputs.carry
            hidden_states = outputs.hidden_states
            attentions = outputs.attentions
        else:
            logits, q_halt_logits, q_continue_logits, state_out, hidden_states, attentions = outputs

        loss = None
        if labels is not None:
            # Standard cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            return (loss, logits, q_halt_logits, q_continue_logits, state_out, hidden_states, attentions)

        return HrmCausalLMOutput(
            loss=loss,
            logits=logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=q_continue_logits,
            carry=state_out,
            hidden_states=hidden_states,
            attentions=attentions,
        )


__all__ = ["HrmConfig", "HrmModel", "HrmForCausalLM", "HrmPreTrainedModel"]
