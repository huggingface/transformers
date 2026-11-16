# Copyright 2024 Mistral and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Pixtral model."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ...activations import ACT2FN
from ...integrations import use_kernel_forward_from_hub, use_kernelized_func
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import is_flash_attention_requested, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_pixtral import PixtralVisionConfig


logger = logging.get_logger(__name__)


# Adapted from transformers.models.gemma4.modeling_gemma4.Gemma4VisionRotaryEmbedding with interleaved freqs
class PixtralVisionRotaryEmbedding(nn.Module):
    """
    Simple axial 2D rope with same freqs used for H and W grids. The freqs are
    pre-computed using `head-dim//4` which is later used to concat H and W positions.
    The final angles rotate over the whole head dim, no partial rotation involved.
    """

    def __init__(self, config: PixtralVisionConfig):
        super().__init__()
        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_axial_rope_parameters
        if self.rope_type != "axial":
            raise ValueError(f"{self.__class__.__name__} supports only axial rope, but requested {self.rope_type}")
        inv_freq, self.attention_scaling = rope_init_fn(self.config)

        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
        self.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_axial_rope_parameters(config: PixtralVisionConfig, **kwargs) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the axial RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        inv_freq_2d = torch.cat([inv_freq[0::2], inv_freq[1::2]])

        return inv_freq_2d, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        freqs = position_ids[..., None].float() * self.inv_freq.to(device=x.device, dtype=torch.float)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling

        cos = self.recomposition_frequencies(cos)
        sin = self.recomposition_frequencies(sin)
        return cos.to(x.dtype), sin.to(x.dtype)

    def recomposition_frequencies(self, freq):
        """
        Recompose the frequencies into the final spatial layout used per each grid.
        """
        # block-concat grids as H-W-H-W
        freq_h, freq_w = (m[:, i % 2] for i, m in enumerate(freq.chunk(2, dim=-1)))
        freq_hw = torch.cat([freq_h, freq_w], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_forward_from_hub("rotary_pos_emb")
# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
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


# Copied from transformers.models.siglip.modeling_siglip.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@use_kernelized_func(apply_rotary_pos_emb)
class PixtralAttention(nn.Module):
    """
    Multi-headed attention compatible with ALL_ATTENTION_FUNCTIONS.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = False

        self.scaling = self.head_dim**-0.5
        self.is_causal = False

        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Pixtral
class PixtralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Pixtral
class PixtralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        PixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PixtralAttentionLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.feed_forward = PixtralMLP(config)
        self.attention = PixtralAttention(config)
        self.ffn_norm = PixtralRMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.Tensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
        """
        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class PixtralTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            self.layers.append(PixtralAttentionLayer(config))
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Args:
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embeddings which serve as input to the Transformer.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


@auto_docstring
class PixtralPreTrainedModel(PreTrainedModel):
    config: PixtralVisionConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _supports_attention_backend = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _no_split_modules = ["PixtralAttentionLayer"]
    _can_record_outputs = {
        "hidden_states": PixtralAttentionLayer,
        "attentions": PixtralAttention,
    }


def generate_block_attention_mask(patch_image_index, tensor):
    """Additive mask letting each patch attend to the patches of its own image only."""
    same_image = patch_image_index[:, None] == patch_image_index[None, :]
    block_mask = torch.zeros(same_image.shape, dtype=tensor.dtype, device=tensor.device)
    block_mask = block_mask.masked_fill(~same_image, torch.finfo(tensor.dtype).min)
    return block_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)


@auto_docstring
class PixtralVisionModel(PixtralPreTrainedModel):
    base_model_prefix = "vision_encoder"

    def __init__(self, config: PixtralVisionConfig):
        super().__init__(config)
        self.config = config
        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.patch_size = config.patch_size
        self.ln_pre = PixtralRMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralTransformer(config)
        self.patch_positional_embedding = PixtralVisionRotaryEmbedding(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.patch_conv

    def get_data_dependent_inputs(
        self, pixel_values: torch.Tensor, image_sizes: torch.Tensor | list | None = None, **kwargs
    ) -> dict[str, Any]:
        r"""
        The inputs the forward computes from `image_sizes` with data-dependent operations (host syncs, loops over
        images): the image each unpadded patch belongs to (`patch_image_index`) and its 2D `position_ids` within that
        image, plus the attention's `cu_seqlens` and `max_seqlen` with flash attention. Passed back to the forward,
        they keep `torch.compile` and `torch.export` free of those operations. Returned as they are when already in
        `kwargs`; only the config and the shape of `pixel_values` are read.

        Args:
            pixel_values (`torch.Tensor` of shape `(num_images, num_channels, height, width)`):
                The padded images, whose height and width bound each image's patch grid.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)` or `list`, *optional*):
                The height and width of each image, before padding. All of the padded size when not given.
        """
        names = ("patch_image_index", "position_ids")
        if is_flash_attention_requested(self.config):
            names += ("cu_seqlens", "max_seqlen")
        if all(kwargs.get(name) is not None for name in names):
            return {name: kwargs[name] for name in names}

        batch_size, _, height, width = pixel_values.shape
        if image_sizes is None:
            image_sizes = [(height, width)] * batch_size
        # Slicing the padded grid keeps each image's patch count bounded by it
        rows, columns = torch.meshgrid(
            torch.arange(height // self.patch_size, device=pixel_values.device),
            torch.arange(width // self.patch_size, device=pixel_values.device),
            indexing="ij",
        )
        patch_image_index, position_ids = [], []
        for index, size in enumerate(image_sizes):
            image_rows = rows[: size[0] // self.patch_size, : size[1] // self.patch_size].flatten()
            image_columns = columns[: size[0] // self.patch_size, : size[1] // self.patch_size].flatten()
            position_ids.append(torch.stack([image_rows, image_columns], dim=-1))
            patch_image_index.append(torch.full_like(image_rows, index))
        inputs = {"patch_image_index": torch.cat(patch_image_index), "position_ids": torch.cat(position_ids)}

        if is_flash_attention_requested(self.config):
            sequence_lengths = [image_position_ids.shape[0] for image_position_ids in position_ids]
            cu_seqlens = torch.tensor([0, *sequence_lengths], device=pixel_values.device, dtype=torch.int32)
            inputs["cu_seqlens"] = cu_seqlens.cumsum(dim=0, dtype=torch.int32)
            inputs["max_seqlen"] = max(sequence_lengths)
        return inputs

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        data_dependent_inputs = self.get_data_dependent_inputs(pixel_values, image_sizes, **kwargs)
        kwargs = {key: value for key, value in kwargs.items() if key not in data_dependent_inputs}

        # pass images through initial convolution independently
        target_dtype = self.patch_conv.weight.dtype
        patch_embeds = self.patch_conv(pixel_values.to(dtype=target_dtype))

        # gather each image's unpadded patches into a single sequence
        patch_image_index = data_dependent_inputs["patch_image_index"]
        position_ids = data_dependent_inputs["position_ids"]
        patch_embeds = patch_embeds.permute(0, 2, 3, 1)[patch_image_index, position_ids[:, 0], position_ids[:, 1]]
        patch_embeds = self.ln_pre(patch_embeds.unsqueeze(0))
        position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

        if is_flash_attention_requested(self.config):
            # Axial RoPE positions have two coordinates per patch, so they cannot
            # describe the packed image boundaries expected by Flash Attention.
            cu_seqlens = data_dependent_inputs["cu_seqlens"]
            max_seqlen = data_dependent_inputs["max_seqlen"]
            kwargs.update(
                cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens, max_length_q=max_seqlen, max_length_k=max_seqlen
            )
            attention_mask = None
        else:
            attention_mask = generate_block_attention_mask(patch_image_index, patch_embeds)

        return self.transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )


__all__ = ["PixtralVisionModel", "PixtralPreTrainedModel"]
