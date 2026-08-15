# Copyright 2026 H Company and the HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask, sliding_window_bidirectional_overlay
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, MaskedLMOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    is_torchdynamo_compiling,
    logging,
    torch_compilable_check,
)
from ...utils.generic import can_return_tuple, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import eager_attention_forward, repeat_kv
from .configuration_neomme import NeoMMEConfig


logger = logging.get_logger(__name__)


def parameter_free_rms_norm(hidden_states: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS normalization without a learnable weight."""
    return F.rms_norm(hidden_states, (hidden_states.shape[-1],), eps=eps)


def get_rotary_dim(config: NeoMMEConfig, layer_type: str) -> int:
    """Number of head dimensions that carry position for `layer_type`."""
    partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
    return int(config.head_dim * partial_rotary_factor)


def apply_interleaved_rotary_pos_emb(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int, unsqueeze_dim: int = 2
) -> torch.Tensor:
    """Apply interleaved rotary position embeddings to query or key states.

    NeoMME rotates even/odd dimension pairs rather than the first/second half split
    used by most models in `transformers`.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, num_heads, head_dim)`):
            Query or key states, heads-last.
        cos (`torch.Tensor` of shape `(batch_size, seq_len, rotary_dim // 2)`):
            Cosine of the interleaved two-axis angles.
        sin (`torch.Tensor` of shape `(batch_size, seq_len, rotary_dim // 2)`):
            Sine of the interleaved two-axis angles.
        rotary_dim (`int`):
            Number of leading head dims to rotate; the rest pass through unchanged.
        unsqueeze_dim (`int`, *optional*, defaults to 2):
            Dimension `cos`/`sin` are unsqueezed at to broadcast over the head axis.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_states = hidden_states[..., :rotary_dim].float()
    pass_through = hidden_states[..., rotary_dim:].float()
    first, second = rotary_states[..., 0::2], rotary_states[..., 1::2]
    rotated = torch.stack([first * cos - second * sin, first * sin + second * cos], dim=-1).flatten(-2)
    return torch.cat([rotated, pass_through], dim=-1).to(hidden_states.dtype)


def unmask_empty_rows(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Unmask fully masked query rows to avoid NaNs in attention."""
    if not torch.is_tensor(attention_mask):
        return attention_mask
    if attention_mask.dtype == torch.bool:
        return attention_mask | (~attention_mask).all(-1, keepdim=True)
    return attention_mask.masked_fill(
        (attention_mask <= torch.finfo(attention_mask.dtype).min).all(-1, keepdim=True), 0.0
    )


class NeoMMEEmbeddings(nn.Module):
    """Factorized (ALBERT-style) token embeddings: `vocab_size -> embedding_rank -> hidden_size`."""

    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_rank)
        self.embedding_projection = nn.Linear(config.embedding_rank, config.hidden_size, bias=False)

    def forward(
        self, input_ids: torch.LongTensor | None = None, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        return self.embedding_projection(inputs_embeds)

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the tied factorized embedding for MLM logits."""
        projected = hidden_states @ self.embedding_projection.weight  # (batch_size, sequence_length, embedding_rank)
        return projected @ self.word_embeddings.weight.t()  # (batch_size, sequence_length, vocab_size)


class NeoMMEValueEmbeddings(nn.Embedding):
    """Per-token value embeddings added to the first and last global attention layers."""


class NeoMMEPatchEmbeddings(nn.Module):
    """Patch stem that maps flattened image patches to hidden size."""

    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.patch_dim)
        self.projection = nn.Sequential(
            nn.Linear(config.patch_dim, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(pixel_values))


class NeoMMERotaryEmbedding(nn.Module):
    """Two-axis interleaved M-RoPE with per-layer-type frequency spectra."""

    def __init__(self, config: NeoMMEConfig, device=None):
        super().__init__()
        self.config = config
        self.layer_types = sorted(set(config.layer_types))
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_init_fns: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}
        self.rope_type: dict[str, str] = {}
        for layer_type in self.layer_types:
            rope_type = config.rope_parameters[layer_type]["rope_type"]
            self.rope_type[layer_type] = rope_type
            self.rope_init_fns[layer_type] = (
                self.compute_default_rope_parameters if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]
            )
            inv_freq, attention_scaling = self.rope_init_fns[layer_type](config, device=device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            # `dynamic_rope_update` restores the unscaled spectrum from this copy when a sequence shrinks
            # back inside the original context, so it must survive the buffer being overwritten.
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: NeoMMEConfig,
        device: torch.device | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Default inverse frequencies for a layer type."""
        rotary_dim = get_rotary_dim(config, layer_type)
        theta = config.rope_parameters[layer_type]["rope_theta"]
        inv_freq = theta ** -(torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim)
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.LongTensor, layer_type: str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin from two-axis `position_ids` of shape `(2, batch, seq_len)`."""
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(2, -1, -1)
        elif position_ids.dim() != 3 or position_ids.shape[0] != 2:
            # Otherwise a `(3, B, L)` or `(B, L, 2)` tensor indexes as if it were axis-major and silently
            # encodes the wrong positions, or dies inside the module on an opaque IndexError.
            raise ValueError(
                f"position_ids must be (2, batch_size, sequence_length) with the M-RoPE axis leading, or "
                f"(batch_size, sequence_length) to use one axis for both; got {tuple(position_ids.shape)}."
            )
        inv_freq = getattr(self, f"{layer_type}_inv_freq")  # (rotary_dim // 2,)
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        row_angles = (
            position_ids[0].float().unsqueeze(-1) * inv_freq[0::2]
        )  # (batch_size, sequence_length, rotary_dim // 4)
        column_angles = (
            position_ids[1].float().unsqueeze(-1) * inv_freq[1::2]
        )  # (batch_size, sequence_length, rotary_dim // 4)
        angles = torch.stack([row_angles, column_angles], dim=-1).flatten(
            -2
        )  # (batch_size, sequence_length, rotary_dim // 2)
        return angles.cos() * attention_scaling, angles.sin() * attention_scaling


class NeoMMEAttention(nn.Module):
    """Bidirectional grouped-query attention with QK-norm, M-RoPE, and a sigmoid output gate.

    QK-norm runs before rotary embedding, value embeddings are added after rotation, and
    exclusive self-attention is applied before the output gate.
    """

    def __init__(self, config: NeoMMEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.norm_eps = config.norm_eps
        self.rotary_dim = get_rotary_dim(config, self.attention_type)
        self.is_causal = False

        # `config.layer_window_sizes` is a HALF-width (`abs(i - j) <= window`). The flash-attention path
        # builds an inclusive symmetric band of `sliding_window - 1` per side, hence the `+ 1`.
        window = config.layer_window_sizes[layer_idx]
        self.sliding_window = None if window is None else window + 1

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        # Fused K/V projection: rows `[:num_key_value_heads * head_dim]` are K, the rest are V.
        self.kv_proj = nn.Linear(config.hidden_size, 2 * config.num_key_value_heads * config.head_dim, bias=False)
        self.output_gate = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        # Exclusive Self-Attention: zero-init, so `tanh(alpha) == 0` makes it an exact no-op at step 0.
        self.alpha = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        value_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]  # (batch_size, sequence_length)

        query_states = self.q_proj(hidden_states).view(
            *input_shape, self.num_attention_heads, self.head_dim
        )  # (batch_size, sequence_length, num_attention_heads, head_dim)
        query_states = parameter_free_rms_norm(query_states, self.norm_eps)
        key_states, value_states = (
            self.kv_proj(hidden_states).view(*input_shape, 2, self.num_key_value_heads, self.head_dim).unbind(-3)
        )  # K is index 0, V is index 1: (batch_size, sequence_length, 2, num_key_value_heads, head_dim)
        key_states = parameter_free_rms_norm(key_states, self.norm_eps)

        cos, sin = position_embeddings
        query_states = apply_interleaved_rotary_pos_emb(query_states, cos, sin, self.rotary_dim)
        key_states = apply_interleaved_rotary_pos_emb(key_states, cos, sin, self.rotary_dim)
        if value_embeds is not None:
            value_states = value_states + value_embeds.view(*input_shape, self.num_key_value_heads, self.head_dim)

        query_states = query_states.transpose(1, 2)  # -> (batch_size, num_attention_heads, sequence_length, head_dim)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )  # attn_output is heads-last: (batch_size, sequence_length, num_attention_heads, head_dim)

        attn_output = self._exclusive_self_attention(attn_output, value_states)
        attn_output = attn_output.reshape(
            *input_shape, -1
        )  # (batch_size, sequence_length, num_attention_heads * head_dim)
        gated_output = attn_output * torch.sigmoid(self.output_gate(hidden_states))
        return self.o_proj(gated_output), attn_weights  # (batch_size, sequence_length, hidden_size)

    def _exclusive_self_attention(self, attn_output: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        """Exclusive self-attention correction along the value direction."""
        value_states = repeat_kv(
            value_states, self.num_key_value_groups
        )  # (batch_size, num_attention_heads, sequence_length, head_dim)
        value_states = value_states.transpose(1, 2)  # (batch_size, sequence_length, num_attention_heads, head_dim)
        value_unit = F.normalize(value_states.float(), dim=-1).to(attn_output.dtype)
        projection = (attn_output * value_unit).sum(
            -1, keepdim=True
        )  # (batch_size, sequence_length, num_attention_heads, 1)
        scale = torch.tanh(self.alpha).to(attn_output.dtype).view(1, 1, self.num_attention_heads, 1)
        return attn_output - (scale * projection) * value_unit


class NeoMMEMLP(nn.Module):
    """Squared ReLU feed-forward block."""

    def __init__(self, config: NeoMMEConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(hidden_states)).square())


class NeoMMEEncoderLayer(GradientCheckpointingLayer):
    """Pre-norm encoder layer with initial-state mixing and muP depth scaling."""

    def __init__(self, config: NeoMMEConfig, layer_idx: int):
        super().__init__()
        self.self_attn = NeoMMEAttention(config, layer_idx)
        self.mlp = NeoMMEMLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.norm_eps = config.norm_eps
        # muP depth transfer, a forward-time constant that is never folded into the weights.
        self.residual_scale = (2 * config.num_hidden_layers) ** -0.5
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        initial_hidden_states: torch.Tensor,
        value_embeds: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # `hidden_states`, `initial_hidden_states` and `value_embeds` are the arguments that carry
        # gradients, so they MUST stay positional: reentrant gradient checkpointing only re-attaches the
        # graph for positional inputs, and a shared grad-carrying keyword argument would be backwarded
        # through twice (the value-embedding table feeds two layers).
        mixed_states = self.lambdas[0] * hidden_states + self.lambdas[1] * initial_hidden_states
        attn_output, _ = self.self_attn(
            parameter_free_rms_norm(mixed_states, self.norm_eps),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            value_embeds=value_embeds,
            **kwargs,
        )
        hidden_states = hidden_states + self.residual_scale * attn_output
        mlp_output = self.mlp(parameter_free_rms_norm(hidden_states, self.norm_eps))
        return hidden_states + self.residual_scale * mlp_output


@auto_docstring
class NeoMMEPreTrainedModel(PreTrainedModel):
    config: NeoMMEConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["NeoMMEEncoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    _can_record_outputs = {
        "hidden_states": NeoMMEEncoderLayer,
        "attentions": NeoMMEAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        # Generic init for whatever the branches below do not name. Safe for the tensors NeoMME needs born
        # at exactly zero: `apply` visits children before parents, so the parent-level zeroing runs last.
        super()._init_weights(module)
        std = self.config.initializer_range

        if isinstance(module, NeoMMEEmbeddings):
            # The factorized table is scaled by the RANK, not `initializer_range`, so the tied decode
            # logits stay O(1) at init (otherwise the initial cross-entropy is ~250 instead of ln V).
            init.normal_(module.word_embeddings.weight, mean=0.0, std=self.config.embedding_rank**-0.5)
            init.normal_(module.embedding_projection.weight, mean=0.0, std=std)
        elif isinstance(module, NeoMMEValueEmbeddings):
            init.zeros_(module.weight)
        elif isinstance(module, NeoMMEPatchEmbeddings):
            for submodule in module.projection.modules():
                if isinstance(submodule, nn.Linear):
                    init.normal_(submodule.weight, mean=0.0, std=std)
                    if submodule.bias is not None:
                        init.zeros_(submodule.bias)
        elif isinstance(module, NeoMMEAttention):
            init.normal_(module.q_proj.weight, mean=0.0, std=std)
            init.normal_(module.kv_proj.weight, mean=0.0, std=std)
            init.normal_(module.output_gate.weight, mean=0.0, std=std)
            init.zeros_(module.o_proj.weight)  # residual branch starts as an exact no-op
            init.zeros_(module.alpha)
        elif isinstance(module, NeoMMEMLP):
            init.normal_(module.up_proj.weight, mean=0.0, std=std)
            init.zeros_(module.down_proj.weight)
        elif isinstance(module, NeoMMEEncoderLayer):
            init.copy_(module.lambdas, torch.tensor([1.0, 0.0]))
        elif isinstance(module, NeoMMEForRetrieval):
            init.normal_(module.embedding_proj_layer.weight, mean=0.0, std=std)
        elif isinstance(module, NeoMMERotaryEmbedding):
            for layer_type in module.layer_types:
                inv_freq, _ = module.rope_init_fns[layer_type](module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                init.zeros_(module.bias)

    def _resize_token_embeddings(
        self, new_num_tokens: int, pad_to_multiple_of: int | None = None, mean_resizing: bool = True
    ) -> nn.Embedding:
        """Resize word and value embedding tables together."""
        word_embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        backbone = getattr(self, self.base_model_prefix, self)
        if getattr(backbone, "value_embeddings", None) is None:
            return word_embeddings

        resized = self._get_resized_embeddings(
            backbone.value_embeddings, word_embeddings.weight.shape[0], mean_resizing=mean_resizing
        )
        # `_get_resized_embeddings` returns a plain `nn.Embedding`; the marker subclass is what the init
        # dispatch and the conversion script match on, so keep it.
        backbone.value_embeddings = NeoMMEValueEmbeddings(resized.num_embeddings, resized.embedding_dim)
        backbone.value_embeddings.weight = resized.weight
        return word_embeddings


@auto_docstring(
    custom_intro="""
    The bare NeoMME model. It encodes text tokens and image patches with one bidirectional Transformer.
    """
)
class NeoMMEModel(NeoMMEPreTrainedModel):
    def __init__(self, config: NeoMMEConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = NeoMMEEmbeddings(config)
        self.patch_embeddings = NeoMMEPatchEmbeddings(config)
        self.rotary_emb = NeoMMERotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [NeoMMEEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        global_layers = [i for i, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"]
        self.value_embeddings = (
            NeoMMEValueEmbeddings(config.vocab_size, config.num_key_value_heads * config.head_dim)
            if config.use_value_embeds and global_layers
            else None
        )
        self.value_embedding_layers = (
            {global_layers[0], global_layers[-1]} if self.value_embeddings is not None else set()
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        position_ids (`torch.LongTensor` of shape `(2, batch_size, sequence_length)` or `(batch_size, sequence_length)`, *optional*):
            Positions for the input tokens. [`NeoMMEProcessor`] returns two-axis positions for document images. A
            one-axis position tensor is used for text inputs.
        pixel_values (`torch.Tensor` of shape `(num_patches, 3 * patch_size ** 2)`, *optional*):
            Flattened image patches returned by [`NeoMMEProcessor`]. The model places these patches at image
            placeholders in `input_ids`.
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embedding_rank)`, *optional*):
            Token embeddings before projection to `hidden_size`. Use `input_ids` for image inputs because the model
            needs the image placeholders to place `pixel_values`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is not None and self.value_embeddings is not None:
            logger.warning_once("inputs_embeds cannot apply value embeddings without token ids")

        hidden_states = self.embeddings(
            input_ids=input_ids, inputs_embeds=inputs_embeds
        )  # (batch_size, sequence_length, hidden_size)
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("`pixel_values` requires `input_ids` to locate image placeholder tokens.")
            hidden_states = self._scatter_patch_embeddings(input_ids, hidden_states, pixel_values)

        batch_size, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            # One axis: `NeoMMERotaryEmbedding` expands it onto both, which is what text-only inputs want.
            position_ids = torch.arange(seq_len, device=hidden_states.device).expand(batch_size, -1)

        # `initial_hidden_states` is captured HERE — after the patch scatter and the first norm — and every
        # layer mixes it back in through its `lambdas`.
        hidden_states = initial_hidden_states = parameter_free_rms_norm(hidden_states, self.config.norm_eps)

        attention_masks = self._build_attention_masks(hidden_states, attention_mask)
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }
        # Value embeddings are a per-token table lookup, so they need the ids themselves: an
        # `inputs_embeds`-only call runs without them.
        value_embeds = None
        if self.value_embeddings is not None and input_ids is not None:
            value_embeds = self.value_embeddings(
                input_ids
            )  # (batch_size, sequence_length, num_key_value_heads * head_dim)

        for layer_idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                initial_hidden_states,
                value_embeds if layer_idx in self.value_embedding_layers else None,
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                attention_mask=attention_masks[layer_idx],
                **kwargs,
            )

        # The final norm is part of the backbone; heads must NOT re-norm.
        hidden_states = parameter_free_rms_norm(hidden_states, self.config.norm_eps)
        return BaseModelOutput(last_hidden_state=hidden_states)

    def _scatter_patch_embeddings(
        self, input_ids: torch.LongTensor, hidden_states: torch.Tensor, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        """Scatter patch embeddings into image placeholder tokens."""
        if pixel_values.shape[-1] != self.config.patch_dim:
            raise ValueError(
                f"pixel_values has patch width {pixel_values.shape[-1]} but the model expects "
                f"{self.config.patch_dim} (= 3 * patch_size ** 2 with patch_size={self.config.patch_size})"
            )
        previous_ids = F.pad(input_ids[:, :-1], (1, 0), value=self.config.pad_token_id or 0)  # ids shifted right
        image_mask = (
            (input_ids == self.config.image_token_id) & (previous_ids != self.config.document_token_id)
        ).unsqueeze(-1)

        if not is_torchdynamo_compiling():
            num_image_tokens = image_mask.sum()
            torch_compilable_check(
                num_image_tokens == pixel_values.shape[0],
                lambda: (
                    f"Got {pixel_values.shape[0]} image patches for {int(num_image_tokens)} image placeholder tokens"
                ),
            )
        patch_embeds = self.patch_embeddings(pixel_values.to(hidden_states.dtype))  # (num_patches, hidden_size)
        return hidden_states.masked_scatter(image_mask, patch_embeds)

    def _build_attention_masks(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> list[torch.Tensor | None]:
        """Build one attention mask per layer."""
        if isinstance(attention_mask, dict):
            return [attention_mask[layer_type] for layer_type in self.config.layer_types]

        mask_kwargs = {"config": self.config, "inputs_embeds": hidden_states, "attention_mask": attention_mask}
        masks: dict[int | None, torch.Tensor | None] = {}
        for window in set(self.config.layer_window_sizes):
            if window is None:
                masks[window] = create_bidirectional_mask(**mask_kwargs)
            else:
                masks[window] = unmask_empty_rows(
                    create_bidirectional_mask(
                        **mask_kwargs, and_mask_function=sliding_window_bidirectional_overlay(window)
                    )
                )
        return [masks[window] for window in self.config.layer_window_sizes]


@auto_docstring(
    custom_intro="""
    The NeoMME model with a masked language modeling head.
    """
)
class NeoMMEForMaskedLM(NeoMMEPreTrainedModel):
    def __init__(self, config: NeoMMEConfig):
        super().__init__(config)
        self.config = config
        self.model = NeoMMEModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        """The decode is tied through the factorized embedding; there is no separate output layer."""
        return None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for the masked-language-modeling loss. Indices should be in `[0, ..., config.vocab_size - 1]`
            or `-100`; only tokens with a label different from `-100` contribute.
        """
        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.model.embeddings.decode(outputs.last_hidden_state)  # (batch_size, sequence_length, vocab_size)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, **kwargs)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Output type for [`NeoMMEForRetrieval`].
    """
)
@dataclass
class NeoMMEForRetrievalOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
        Retrieval loss. This value is always `None`.
    embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_dim)`, *optional*):
        Normalized token embeddings for late-interaction retrieval. Padding rows are zeroed. Use
        [`NeoMMEProcessor.score_retrieval`] to score these embeddings with MaxSim.
    dense_embeddings (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
        A normalized mean-pooled embedding for each input. You can request a Matryoshka prefix with `dense_dim`.
        Use [`NeoMMEProcessor.score_retrieval`] to score dense embeddings with cosine similarity.
    """

    loss: torch.FloatTensor | None = None
    embeddings: torch.FloatTensor | None = None
    dense_embeddings: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring(
    custom_intro="""
    The NeoMME model with multi-vector and dense retrieval heads. One forward pass can return token embeddings for
    MaxSim scoring and mean-pooled embeddings for cosine similarity.
    """
)
class NeoMMEForRetrieval(NeoMMEPreTrainedModel):
    def __init__(self, config: NeoMMEConfig):
        super().__init__(config)
        self.config = config
        self.model = NeoMMEModel(config)
        self.embedding_proj_layer = nn.Linear(config.hidden_size, config.embedding_dim, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings.word_embeddings = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_multivector: bool = True,
        output_dense: bool = True,
        dense_dim: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> NeoMMEForRetrievalOutput:
        r"""
        output_multivector (`bool`, *optional*, defaults to `True`):
            Whether to return token embeddings for late-interaction retrieval.
        output_dense (`bool`, *optional*, defaults to `True`):
            Whether to return one mean-pooled dense embedding per input.
        dense_dim (`int`, *optional*):
            Width of the Matryoshka prefix to return for dense embeddings. The model truncates the pooled vector
            before normalizing it.
        """
        if not (output_multivector or output_dense):
            raise ValueError("At least one of `output_multivector` or `output_dense` must be True")

        outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)

        return NeoMMEForRetrievalOutput(
            embeddings=self._multivector(hidden_states, attention_mask) if output_multivector else None,
            dense_embeddings=self._dense(hidden_states, attention_mask, dense_dim) if output_dense else None,
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_multivector_embeddings(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Return normalized token embeddings for late-interaction retrieval.

        The method returns the `embeddings` output and does not compute dense embeddings. Padding rows are zeroed.
        """
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            output_dense=False,
            **kwargs,
        ).embeddings

    def get_dense_embeddings(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        dense_dim: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Return normalized mean-pooled embeddings for dense retrieval.

        The method returns the `dense_embeddings` output and does not compute token embeddings. Set `dense_dim` to
        truncate the pooled vector before normalization.
        """
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            inputs_embeds=inputs_embeds,
            output_multivector=False,
            dense_dim=dense_dim,
            **kwargs,
        ).dense_embeddings

    def _multivector(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        proj_dtype = self.embedding_proj_layer.weight.dtype
        embeddings = self.embedding_proj_layer(
            hidden_states.to(proj_dtype)
        )  # (batch_size, sequence_length, embedding_dim)
        embeddings = F.normalize(embeddings, dim=-1)
        # Overwrite padding rows rather than multiply them out, so a non-finite value cannot survive.
        return embeddings.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

    def _dense(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, dense_dim: int | None = None
    ) -> torch.Tensor:
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.shape).to(hidden_states.dtype)
        pooled = (hidden_states * expanded_mask).sum(1) / expanded_mask.sum(1).clamp_min(1e-9)
        if dense_dim is None:
            return F.normalize(pooled, dim=-1)

        # Slicing would quietly accept a negative or out-of-range width and return a differently-sized
        # vector, which downstream cosine scoring cannot detect.
        if not 0 < dense_dim <= pooled.shape[-1]:
            raise ValueError(f"dense_dim must be in 1..{pooled.shape[-1]} (the pooled width), got {dense_dim}")
        return F.normalize(pooled[..., :dense_dim], dim=-1)


__all__ = [
    "NeoMMEForMaskedLM",
    "NeoMMEForRetrieval",
    "NeoMMEModel",
    "NeoMMEPreTrainedModel",
]
