# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Idefics2 model."""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from ..auto import AutoModel
from .configuration_idefics2 import Idefics2Config, Idefics2PerceiverConfig, Idefics2VisionConfig


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Idefics2 model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """
)
class Idefics2BaseModelOutputWithPast(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        Sequence of hidden-states at the output of the last layer of the model.
        If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
        hidden_size)` is output.
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
        `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
        input) to speed up sequential decoding.
    image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
        sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Idefics2 causal language model (or autoregressive) outputs.
    """
)
# Copied from transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast with Idefics->Idefics2
class Idefics2CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
        sequence_length, hidden_size)`.

        image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[tuple[torch.FloatTensor]] = None


class Idefics2VisionEmbeddings(nn.Module):
    """
    This is a modified version of `siglip.modelign_siglip.SiglipVisionEmbeddings` to enable images of variable
    resolution.

    The modifications are adapted from [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://huggingface.co/papers/2307.06304)
    which allows treating images in their native aspect ratio and without the need to resize them to the same
    fixed size. In particular, we start from the original pre-trained SigLIP model
    (which uses images of fixed-size square images) and adapt it by training on images of variable resolutions.
    """

    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, _, max_im_h, max_im_w = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(
            1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side, device=pixel_values.device
        )
        position_ids = torch.full(
            size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0, device=pixel_values.device
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()
            nb_patches_w = p_attn_mask[0].sum()

            h_indices = torch.arange(nb_patches_h, device=position_ids.device, dtype=pixel_values.dtype)
            w_indices = torch.arange(nb_patches_w, device=position_ids.device, dtype=pixel_values.dtype)

            fractional_coords_h = h_indices / nb_patches_h * (1 - 1e-6)
            fractional_coords_w = w_indices / nb_patches_w * (1 - 1e-6)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids

        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.siglip.modeling_siglip.SiglipAttention with Siglip->Idefics2Vision
class Idefics2VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Ignore copy
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.siglip.modeling_siglip.SiglipMLP with Siglip->Idefics2Vision
class Idefics2VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Idefics2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.siglip.modeling_siglip.SiglipMultiheadAttentionPoolingHead with Siglip->Idefics2
class Idefics2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Ignore copy
        self.mlp = Idefics2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            output_size=config.hidden_size,
        )

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class Idefics2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Idefics2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Idefics2VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Idefics2VisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    @auto_docstring
    # Copied from transformers.models.siglip.modeling_siglip.SiglipEncoderLayer.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# Copied from transformers.models.siglip.modeling_siglip.SiglipEncoder with Siglip->Idefics2
class Idefics2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Idefics2EncoderLayer`].

    Args:
        config: Idefics2Config
    """

    def __init__(self, config: Idefics2Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Idefics2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    @auto_docstring
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                **kwargs,
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


@auto_docstring
class Idefics2PreTrainedModel(PreTrainedModel):
    config: Idefics2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Idefics2VisionAttention", "Idefics2MLP", "Idefics2PerceiverLayer", "Idefics2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _supports_attention_backend = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, Idefics2RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()  # native torch init
        elif isinstance(module, Idefics2MultiheadAttentionPoolingHead):
            module.probe.data.normal_()
        elif isinstance(module, Idefics2PerceiverResampler):
            module.latents.data.fill_(1.0)


@auto_docstring(
    custom_intro="""
    Idefics2 vision encoder model that returnss raw image embeddings.
    """
)
class Idefics2VisionTransformer(Idefics2PreTrainedModel):
    config: Idefics2VisionConfig
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "hidden_states": Idefics2EncoderLayer,
        "attentions": Idefics2VisionAttention,
    }

    def __init__(self, config: Idefics2VisionConfig):
        super().__init__(config)
        embed_dim = config.hidden_size

        self.config = config
        self.embeddings = Idefics2VisionEmbeddings(config)
        self.encoder = Idefics2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        patch_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_patches_height, num_patches_width)`, *optional*):
            The attention mask for the patches.
        """
        batch_size = pixel_values.size(0)
        if patch_attention_mask is None:
            patch_size = self.config.patch_size
            patch_attention_mask = torch.ones(
                (
                    batch_size,
                    pixel_values.size(2) // patch_size,
                    pixel_values.size(3) // patch_size,
                )
            )
            patch_attention_mask = patch_attention_mask.to(dtype=torch.bool, device=pixel_values.device)

        hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)

        patch_attention_mask = patch_attention_mask.view(batch_size, -1)
        # The call to `_upad_input` in `_flash_attention_forward` is expensive
        # So when the `patch_attention_mask` is full of 1s (i.e. attending to the whole sequence),
        # avoiding passing the attention_mask, which is equivalent to attending to the full sequence
        if not torch.any(~patch_attention_mask):
            patch_attention_mask = None
        elif self.config._attn_implementation != "flash_attention_2":
            patch_attention_mask = _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)

        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=patch_attention_mask,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Idefics2
class Idefics2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Idefics2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()
        self.config = config
        self.layer_idx = None
        self.hidden_size = config.hidden_size
        self.num_heads = config.resampler_n_heads
        self.head_dim = config.resampler_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.is_causal = False

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            latents (`torch.Tensor`): Tensor of shape [bsz, n_latents, embed_dim] representing fixed length latents to compress to.
            context (`torch.Tensor`): Tensor of shape [bsz, seq, embed_dim] representing long-form context to resample.
            attention_mask (`torch.Tensor`, *optional*): Tensor of shape [bsz, 1, seq, n_latents] representing attention mask.
            position_ids (`torch.LongTensor`, *optional*): Tensor of shape [bsz, seq] representing position indices of each input token.
            past_key_values (`Cache`, *optional*): Tuple of tensors containing cached key and value states.
            output_attentions (`bool`, *optional*, defaults to `False`): Whether to return attention weights.
            use_cache (`bool`, *optional*, defaults to `False`): Whether to use past_key_values for caching.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        hidden_states = torch.concat([context, latents], dim=-2)

        queries = self.q_proj(latents)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_values = getattr(self, "past_key_values", past_key_values)

        if past_key_values is not None:
            keys, values = past_key_values.update(keys, values, self.layer_idx)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            queries,
            keys,
            values,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_latents = config.resampler_n_latents
        self.depth = config.resampler_depth
        self.rms_norm_eps = config.rms_norm_eps

        self.input_latents_norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = Idefics2PerceiverAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.mlp = Idefics2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 4,
            output_size=config.hidden_size,
            hidden_act=config.hidden_act,
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
        """
        residual = latents

        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)

        latents, _ = self.self_attn(
            latents=latents,
            context=context,
            attention_mask=attention_mask,
            **kwargs,
        )
        latents = residual + latents
        residual = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents

        return latents


@auto_docstring(
    custom_intro="""
    Idefics2 perceiver resampler model that performs `depth` blocks of cross-attention with a fixed
    """
)
class Idefics2PerceiverResampler(Idefics2PreTrainedModel):
    config: Idefics2PerceiverConfig
    _supports_sdpa = True
    _supports_flash_attention_2 = True
    _supports_flex_attn = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.n_latents = config.resampler_n_latents
        self.depth = config.resampler_depth
        self.rms_norm_eps = config.rms_norm_eps

        # Create Latents for Perceiver
        self.latents = nn.Parameter(torch.ones(self.n_latents, self.hidden_size))

        # Create Transformer Blocks
        self.layers = nn.ModuleList([Idefics2PerceiverLayer(config, idx) for idx in range(self.depth)])
        self.norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    @auto_docstring
    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        context (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`):
            Input to the layer.
        """
        # seq embed -> bsz seq embed
        latents = self.latents.unsqueeze(0).expand((context.shape[0], *self.latents.size()))

        latent_attention_mask = torch.ones(
            (attention_mask.size(0), latents.size(1)), dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, latent_attention_mask], dim=-1)
        attention_mask = (
            _prepare_4d_attention_mask(attention_mask, latents.dtype, tgt_len=self.n_latents)
            if self.config._attn_implementation != "flash_attention_2"
            else attention_mask
        )

        compressed_context = latents
        for perceiver_layer in self.layers:
            compressed_context = perceiver_layer(
                compressed_context,
                context,
                attention_mask=attention_mask,
                position_ids=None,
                **kwargs,
            )

        compressed_context = self.norm(compressed_context)

        return compressed_context


class Idefics2Connector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.modality_projection = Idefics2MLP(
            hidden_size=config.vision_config.hidden_size,
            intermediate_size=config.text_config.intermediate_size,
            output_size=config.text_config.hidden_size,
            hidden_act=config.text_config.hidden_act,
        )
        self.perceiver_resampler = Idefics2PerceiverResampler._from_config(config.perceiver_config)

    def forward(self, image_hidden_states, attention_mask):
        image_hidden_states = self.modality_projection(image_hidden_states)
        image_hidden_states = self.perceiver_resampler(context=image_hidden_states, attention_mask=attention_mask)
        return image_hidden_states


@auto_docstring(
    custom_intro="""
    Idefics2 model consisting of a SIGLIP vision encoder and Mistral language decoder
    """
)
class Idefics2Model(Idefics2PreTrainedModel):
    def __init__(self, config: Idefics2Config):
        super().__init__(config)
        self.padding_idx = self.config.text_config.pad_token_id
        self.vocab_size = self.config.text_config.vocab_size

        self.vision_model = Idefics2VisionTransformer._from_config(config.vision_config)
        self.connector = Idefics2Connector(config)
        self.text_model = AutoModel.from_config(config.text_config)

        self.image_seq_len = config.perceiver_config.resampler_n_latents
        self.image_token_id = self.config.image_token_id

        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings.

        This is useful for lora when using gradient checkpointing.
        c.f. https://github.com/huggingface/peft/issues/1402#issuecomment-1913675032

        Override to set output.requires_grad = True for both the decoder's and vision model's embeddings.
        """

        def get_lowest_module(module):
            if len(list(module.children())) == 0:
                # If the module has no children, it is a leaf module (e.g., Linear, Conv2d, etc.)
                return module
            else:
                # Recursively call the function on each child module
                return get_lowest_module(list(module.children())[0])

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = get_lowest_module(self.vision_model).register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.Tensor],
        image_hidden_states: Optional[torch.Tensor],
    ):
        """
        This method aims at merging the token embeddings with the image hidden states into one single sequence of vectors that are fed to the transformer LM.
        The merging happens as follows:
        - The text token sequence is: `tok_1 tok_2 tok_3 <fake_token_around_image> <image> <image> ... <image> <fake_token_around_image> tok_4`.
        - We get the image hidden states for the image through the vision encoder (and potentially the perceiver), and that hidden state is then projected into the text embedding space.
        We thus have a sequence of image hidden states of size (1, image_seq_len, hidden_dim), where 1 is for batch_size of 1 image and hidden_dim is the hidden_dim of the LM transformer.
        - The merging happens so that we obtain the following sequence: `vector_tok_1 vector_tok_2 vector_tok_3 vector_fake_tok_around_image {sequence of image_seq_len image hidden states} vector_fake_toke_around_image vector_tok_4`. That sequence is fed to the LM.
        - To fit the format of that sequence, `input_ids`, `input_embeds`, `attention_mask` are all 3 adapted to insert the image hidden states.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_hidden_states = image_hidden_states.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_hidden_states)
        return inputs_embeds

    def get_image_features(
        self, pixel_values: torch.FloatTensor, pixel_attention_mask: Optional[torch.LongTensor] = None
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            pixel_attention_mask (`torch.LongTensor`, *optional*):
                The attention mask indicating padded regions in the image.
        """
        batch_size, num_images, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
        pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        # Handle the vision attention mask
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                dtype=torch.bool,
                device=pixel_values.device,
            )
        else:
            # Remove padding images from the mask/pP p
            pixel_attention_mask = pixel_attention_mask.view(batch_size * num_images, *pixel_attention_mask.shape[2:])
            pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) == patch_size * patch_size).bool()
        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
        image_hidden_states = image_hidden_states.last_hidden_state

        # Modality projection & resampling
        image_hidden_states = self.connector(
            image_hidden_states, attention_mask=patch_attention_mask.view(pixel_values.size(0), -1)
        )
        image_hidden_states = image_hidden_states.view(-1, image_hidden_states.shape[-1])
        return image_hidden_states

    @can_return_tuple
    @auto_docstring(
        custom_intro="""
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.

        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, Idefics2BaseModelOutputWithPast]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection and perceiver resampling.
        """

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        kwargs["return_dict"] = True
        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Idefics2BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


@auto_docstring(
    custom_intro="""
    The Idefics2 Model with a language modeling head. It is made up a SigLIP vision encoder, with a language modeling head on top.
    """
)
class Idefics2ForConditionalGeneration(Idefics2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Idefics2Model(config)
        self.image_token_id = self.config.image_token_id

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.vocab_size = config.text_config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    def enable_input_require_grads(self):
        """
        Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
        the model weights fixed.
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._text_require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
        self._vision_require_grads_hook = self.model.vision_model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )

    def disable_input_require_grads(self):
        self._text_require_grads_hook.remove()
        self._vision_require_grads_hook.remove()

    def get_input_embeddings(self):
        return self.model.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.text_model.set_input_embeddings(value)

    def get_image_features(
        self, pixel_values: torch.FloatTensor, pixel_attention_mask: Optional[torch.LongTensor] = None
    ):
        return self.model.get_image_features(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Idefics2CausalLMOutputWithPast]:
        r"""
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*):
            Mask to avoid performing attention on padding pixel indices.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection and perceiver resampling.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `Idefics2ForConditionalGeneration`).
            Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
            computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from transformers import AutoProcessor, AutoModelForVision2Seq
        >>> from transformers.image_utils import load_image

        >>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
        >>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
        >>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
        >>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

        >>> processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
        >>> model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b-base", device_map="auto")

        >>> BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        >>> EOS_WORDS_IDS = [processor.tokenizer.eos_token_id]

        >>> # Create inputs
        >>> prompts = [
        ...   "<image>In this image, we can see the city of New York, and more specifically the Statue of Liberty.<image>In this image,",
        ...   "In which city is that bridge located?<image>",
        ... ]
        >>> images = [[image1, image2], [image3]]
        >>> inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to("cuda")

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=20)
        >>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        >>> print(generated_texts)
        ['In this image, we can see the city of New York, and more specifically the Statue of Liberty. In this image, we can see the city of New York, and more specifically the Statue of Liberty.\n\n', 'In which city is that bridge located?\n\nThe bridge is located in the city of Pittsburgh, Pennsylvania.\n\n\nThe bridge is']
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Idefics2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        pixel_attention_mask=None,
        image_hidden_states=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- there are mutually exclusive inputs (if the logic to make `image_hidden_states` take
        # precedence is moved to the model, we can remove this fn)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if image_hidden_states is not None or cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_attention_mask"] = None

        return model_inputs


__all__ = ["Idefics2ForConditionalGeneration", "Idefics2PreTrainedModel", "Idefics2Model"]
