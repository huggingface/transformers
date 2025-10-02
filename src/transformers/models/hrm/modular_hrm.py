# coding=utf-8
# Copyright 2025 The HRM Team and The HuggingFace Inc. team. All rights reserved.
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
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import _flash_attention_forward
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging


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
                Whether or not the model should return the carry state for recurrent computation.
                HRM uses a unique carry state system for hierarchical processing.

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
        carry (`HrmCarry`, *optional*):
            Carry state for recurrent computation, containing hidden states and halting information.
    """

    logits: torch.FloatTensor = None
    q_halt_logits: torch.FloatTensor = None
    q_continue_logits: torch.FloatTensor = None
    carry: Optional["HrmCarry"] = None


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
        carry (`HrmCarry`, *optional*):
            Carry state for recurrent computation, containing hidden states and halting information.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    q_halt_logits: torch.FloatTensor = None
    q_continue_logits: torch.FloatTensor = None
    carry: Optional["HrmCarry"] = None


# Helper functions
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to query and key tensors."""
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """Root Mean Square Layer Normalization."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


# Model components
class HrmLinear(nn.Module):
    """Linear layer with automatic type casting for mixed-precision training."""

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


class HrmEmbedding(nn.Module):
    """Embedding layer with type casting for mixed-precision training."""

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            truncated_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class HrmRotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) precomputation."""

    def __init__(self, dim, max_position_embeddings, base=10000.0, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached, self.sin_cached


class HrmAttention(nn.Module):
    """Multi-head self-attention with FlashAttention and eager fallback."""

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
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

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
        hidden_states = rms_norm(
            hidden_states + self.self_attn(hidden_states, cos_sin=cos_sin), variance_epsilon=self.norm_eps
        )
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
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
class HrmInnerCarry:
    """Internal state for the hierarchical reasoning modules."""

    z_H: torch.Tensor  # High-level hidden state
    z_L: torch.Tensor  # Low-level hidden state


@dataclass
class HrmCarry:
    """Complete carry state for HRM with ACT mechanism."""

    inner_carry: HrmInnerCarry
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

        # Positional encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = HrmRotaryEmbedding(
                dim=config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings + self.puzzle_emb_len,
                base=config.rope_theta,
            )
        elif config.pos_encodings == "learned":
            self.embed_pos = HrmEmbedding(
                config.max_position_embeddings + self.puzzle_emb_len,
                config.hidden_size,
                embed_init_std,
                self.forward_dtype,
            )
        else:
            raise ValueError(f"Unknown pos_encodings: {config.pos_encodings}")

        # Reasoning modules
        self.H_level = HrmReasoningModule(layers=[HrmBlock(config) for _ in range(config.h_layers)])
        self.L_level = HrmReasoningModule(layers=[HrmBlock(config) for _ in range(config.l_layers)])

        # Initial states
        self.register_buffer(
            "H_init",
            truncated_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.register_buffer(
            "L_init",
            truncated_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

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
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: torch.device) -> HrmInnerCarry:
        """Create uninitialized carry state tensors."""
        return HrmInnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.max_position_embeddings + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.max_position_embeddings + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HrmInnerCarry) -> HrmInnerCarry:
        """Reset carry state to learned initial values based on flags."""
        device = carry.z_H.device
        H_init = self.H_init.to(device)
        L_init = self.L_init.to(device)
        return HrmInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L),
        )

    def forward(
        self, carry: HrmInnerCarry, batch: dict
    ) -> tuple[HrmInnerCarry, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Execute hierarchical reasoning forward pass with 1-step gradient."""
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        # Input encoding
        input_embeddings = self._input_embeddings(batch["input_ids"], batch.get("puzzle_identifiers"))

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            for _H_step in range(self.config.h_cycles):
                for _L_step in range(self.config.l_cycles):
                    if not ((_H_step == self.config.h_cycles - 1) and (_L_step == self.config.l_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
                if _H_step != self.config.h_cycles - 1:
                    z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        # Outputs
        new_carry = HrmInnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HrmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = HrmConfig
    base_model_prefix = "hrm"
    supports_gradient_checkpointing = False
    _no_split_modules = ["HrmBlock"]
    _skip_keys_device_placement = ["H_init", "L_init"]

    def _init_weights(self, module):
        """Initialize the weights - already handled by HRM's custom init."""
        pass  # HRM uses custom initialization in the model itself


class HrmModel(HrmPreTrainedModel):
    """Hierarchical Reasoning Model with Adaptive Computation Time."""

    def __init__(self, config: HrmConfig):
        super().__init__(config)
        self.config = config
        self.inner = HrmInner(config)
        self.post_init()

    def initial_carry(self, batch: dict) -> HrmCarry:
        """Initialize carry state for a new batch."""
        batch_size = batch["input_ids"].shape[0]
        device = batch["input_ids"].device
        return HrmCarry(
            inner_carry=self.inner.empty_carry(batch_size, device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        puzzle_identifiers: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        carry: Optional[HrmCarry] = None,
        **kwargs,
    ):
        """Execute one ACT step with halting logic."""
        batch = {"input_ids": input_ids}
        if puzzle_identifiers is not None:
            batch["puzzle_identifiers"] = puzzle_identifiers
        if labels is not None:
            batch["labels"] = labels

        # Initialize carry if not provided
        if carry is None:
            carry = self.initial_carry(batch)

        # Update data and carry
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

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

        new_carry = HrmCarry(new_inner_carry, new_steps, halted, new_current_data)

        return HrmModelOutput(
            logits=logits,
            q_halt_logits=q_halt_logits,
            q_continue_logits=q_continue_logits,
            carry=new_carry,
        )


class HrmForCausalLM(HrmPreTrainedModel):
    """HRM Model with a language modeling head."""

    def __init__(self, config: HrmConfig):
        super().__init__(config)
        self.model = HrmModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        puzzle_identifiers: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        carry: Optional[HrmCarry] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids, puzzle_identifiers=puzzle_identifiers, labels=labels, carry=carry, **kwargs
        )

        loss = None
        if labels is not None:
            # Standard cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return HrmCausalLMOutput(
            loss=loss,
            logits=outputs.logits,
            q_halt_logits=outputs.q_halt_logits,
            q_continue_logits=outputs.q_continue_logits,
            carry=outputs.carry,
        )

    def generate(self, input_ids: torch.LongTensor, max_new_tokens: int = 50, **kwargs):
        """Simple greedy generation for HRM."""
        carry = self.model.initial_carry({"input_ids": input_ids})

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=input_ids, carry=carry, **kwargs)
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            carry = outputs.carry

            # Check if all sequences halted
            if carry.halted.all():
                break

        return input_ids


__all__ = ["HrmConfig", "HrmModel", "HrmForCausalLM", "HrmPreTrainedModel"]
