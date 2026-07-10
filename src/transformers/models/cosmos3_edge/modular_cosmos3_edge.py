# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Cosmos3 Edge reasoner model."""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.generic import is_flash_attention_requested, maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_cu_seqlens
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
from ..siglip2.modeling_siglip2 import (
    Siglip2Attention,
    Siglip2Encoder,
    Siglip2EncoderLayer,
    Siglip2VisionEmbeddings,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeTextConfig(LlamaConfig):
    model_type = "cosmos3_edge_text"
    base_config_key = "text_config"
    default_theta = 100_000_000.0
    ignore_keys_at_rope_validation = {"mrope_section"}
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    vocab_size: int = 131072
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False
    hidden_act: str = "relu2"
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    rope_parameters: dict | None = None
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 11

    def convert_rope_params_to_dict(self, **kwargs):
        mrope_section = kwargs.pop("mrope_section", None)
        kwargs = super().convert_rope_params_to_dict(**kwargs)
        self.rope_parameters.setdefault("mrope_section", mrope_section or [24, 20, 20])
        return kwargs

    def validate_architecture(self):
        super().validate_architecture()
        mrope_section = self.rope_parameters["mrope_section"]
        if len(mrope_section) != 3 or sum(mrope_section) != self.head_dim // 2:
            raise ValueError(
                "`rope_parameters.mrope_section` must contain three sections whose sum equals half of `head_dim`, "
                f"got {mrope_section} for head_dim={self.head_dim}."
            )


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeVisionConfig(Siglip2VisionConfig):
    r"""
    num_patches (`int`, *optional*, defaults to 256):
        Number of patches in the learned reference positional-embedding grid.
    """

    model_type = "cosmos3_edge_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_patches: int = 256
    spatial_merge_size: int = 2


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeProjectorConfig(PreTrainedConfig):
    r"""
    input_hidden_size (`int`, *optional*, defaults to 1152):
        Hidden size produced by the vision encoder before spatial merging.
    merger_intermediate_size (`int`, *optional*, defaults to 11520):
        Intermediate size of the projector MLP.
    out_hidden_size (`int`, *optional*, defaults to 2048):
        Hidden size projected into the language model.
    use_postshuffle_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization after spatial patches are grouped.
    """

    model_type = "cosmos3_edge_projector"

    input_hidden_size: int = 1152
    merger_intermediate_size: int = 11520
    out_hidden_size: int = 2048
    spatial_merge_size: int = 2
    use_postshuffle_norm: bool = False


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeConfig(PreTrainedConfig):
    r"""
    projector_config (`dict` or [`Cosmos3EdgeProjectorConfig`], *optional*):
        Configuration of the vision-to-language patch merger.

    Example:

    ```python
    >>> from transformers import Cosmos3EdgeConfig, Cosmos3EdgeForConditionalGeneration

    >>> configuration = Cosmos3EdgeConfig()
    >>> model = Cosmos3EdgeForConditionalGeneration(configuration)
    ```
    """

    model_type = "cosmos3_edge"
    sub_configs = {
        "text_config": Cosmos3EdgeTextConfig,
        "vision_config": Cosmos3EdgeVisionConfig,
        "projector_config": Cosmos3EdgeProjectorConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: Cosmos3EdgeTextConfig | dict | None = None
    vision_config: Cosmos3EdgeVisionConfig | dict | None = None
    projector_config: Cosmos3EdgeProjectorConfig | dict | None = None
    image_token_id: int = 19
    video_token_id: int = 18
    vision_start_token_id: int = 20
    vision_end_token_id: int = 21
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Cosmos3EdgeTextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Cosmos3EdgeTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Cosmos3EdgeVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Cosmos3EdgeVisionConfig(**self.vision_config)

        if self.projector_config is None:
            self.projector_config = Cosmos3EdgeProjectorConfig(
                input_hidden_size=self.vision_config.hidden_size,
                out_hidden_size=self.text_config.hidden_size,
                spatial_merge_size=self.vision_config.spatial_merge_size,
            )
        elif isinstance(self.projector_config, dict):
            self.projector_config = Cosmos3EdgeProjectorConfig(**self.projector_config)

        if not isinstance(self.text_config, Cosmos3EdgeTextConfig):
            raise TypeError("`text_config` must be a `Cosmos3EdgeTextConfig` or a dictionary.")
        if not isinstance(self.vision_config, Cosmos3EdgeVisionConfig):
            raise TypeError("`vision_config` must be a `Cosmos3EdgeVisionConfig` or a dictionary.")
        if not isinstance(self.projector_config, Cosmos3EdgeProjectorConfig):
            raise TypeError("`projector_config` must be a `Cosmos3EdgeProjectorConfig` or a dictionary.")

        super().__post_init__(**kwargs)


class Cosmos3EdgeTextRotaryEmbedding(LlamaRotaryEmbedding):
    """Interleaved M-RoPE used for Cosmos3 Edge text and visual tokens."""

    def __init__(self, config: Cosmos3EdgeTextConfig, device=None):
        super().__init__(config, device=device)
        self.mrope_section = config.rope_parameters["mrope_section"]

    @staticmethod
    def compute_default_rope_parameters(
        config: Cosmos3EdgeTextConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @staticmethod
    def apply_interleaved_mrope(freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            freqs_t[..., slice(offset, length, 3)] = freqs[dim, ..., slice(offset, length, 3)]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Cosmos3EdgeTextAttention(LlamaAttention):
    """Dense GQA attention with Cosmos3 Edge M-RoPE."""


class Cosmos3EdgeTextMLP(nn.Module):
    """The dense two-projection ReLU-squared MLP used by Cosmos3 Edge."""

    def __init__(self, config: Cosmos3EdgeTextConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class Cosmos3EdgeTextRMSNorm(LlamaRMSNorm):
    pass


class Cosmos3EdgeTextDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Cosmos3EdgeTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Cosmos3EdgeTextAttention(config, layer_idx)
        self.mlp = Cosmos3EdgeTextMLP(config)
        self.input_layernorm = Cosmos3EdgeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Cosmos3EdgeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


_COSMOS3_EDGE_DROPPED_GENERATOR_KEYS = [
    r"^time_embedder\.",
    r"^proj_(?:in|out)\.",
    r"^norm_moe_gen\.",
    r"^layers\.\d+\.input_layernorm_moe_gen\.",
    r"^layers\.\d+\.post_attention_layernorm_moe_gen\.",
    r"^layers\.\d+\.mlp_moe_gen\.",
    r"^layers\.\d+\.self_attn\.(?:add_[qkv]_proj|to_add_out|norm_added_[qk])\.",
]


class Cosmos3EdgePreTrainedModel(LlamaPreTrainedModel):
    config_class = Cosmos3EdgeConfig
    input_modalities = ("image", "video", "text")
    _no_split_modules = ["Cosmos3EdgeTextDecoderLayer"]
    _can_record_outputs = {
        "hidden_states": Cosmos3EdgeTextDecoderLayer,
        "attentions": Cosmos3EdgeTextAttention,
    }
    _keys_to_ignore_on_load_unexpected = _COSMOS3_EDGE_DROPPED_GENERATOR_KEYS


class Cosmos3EdgeTextModel(Cosmos3EdgePreTrainedModel):
    config_class = Cosmos3EdgeTextConfig
    base_model_prefix = "language_model"

    def __init__(self, config: Cosmos3EdgeTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Cosmos3EdgeTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Cosmos3EdgeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Cosmos3EdgeTextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @merge_with_config_defaults
    @capture_outputs
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids[0],
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids[0],
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Cosmos3EdgeVisionEmbeddings(Siglip2VisionEmbeddings):
    """SigLIP2 patch and learned-position embeddings for packed Edge vision inputs."""

    def forward(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        source_dtype = positional_embeddings.dtype
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.float()

        position_chunks = []
        merge_size = self.config.spatial_merge_size
        for temporal, height, width in grid_thw.tolist():
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized_embeddings = resized_embeddings.squeeze(0).permute(1, 2, 0).to(source_dtype)
            resized_embeddings = resized_embeddings.reshape(
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                -1,
            )
            resized_embeddings = resized_embeddings.permute(0, 2, 1, 3, 4).reshape(height * width, -1)
            position_chunks.append(resized_embeddings.repeat(temporal, 1))

        position_embeddings = torch.cat(position_chunks, dim=0)
        if patch_embeds.shape[0] != position_embeddings.shape[0]:
            raise ValueError(
                "The packed visual patch count does not match `grid_thw`: "
                f"got {patch_embeds.shape[0]} patches for {position_embeddings.shape[0]} expected patches."
            )
        return patch_embeds + position_embeddings


class Cosmos3EdgeVisionAttention(Siglip2Attention):
    """Packed non-causal SigLIP2 attention with one sequence per image or video frame."""

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, **kwargs) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).reshape(sequence_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(sequence_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(sequence_length, self.num_heads, self.head_dim)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            max_sequence_length = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scale,
                dropout=0.0 if not self.training else self.dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_sequence_length,
                max_length_k=max_sequence_length,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_output = torch.cat(
                [
                    attention_interface(
                        self,
                        query,
                        key,
                        value,
                        attention_mask=None,
                        scaling=self.scale,
                        dropout=0.0 if not self.training else self.dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for query, key, value in zip(*splits)
                ],
                dim=1,
            )

        return self.out_proj(attn_output.reshape(sequence_length, -1).contiguous())


class Cosmos3EdgeVisionEncoderLayer(Siglip2EncoderLayer):
    def __init__(self, config: Cosmos3EdgeVisionConfig):
        super().__init__(config)
        self.self_attn = Cosmos3EdgeVisionAttention(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.IntTensor` of shape `(num_sequences + 1,)`):
            Cumulative patch counts that delimit each image or video frame in the packed sequence.
        """
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), cu_seqlens=cu_seqlens, **kwargs
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class Cosmos3EdgeEncoder(Siglip2Encoder):
    def __init__(self, config: Cosmos3EdgeVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Cosmos3EdgeVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor, cu_seqlens: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.IntTensor` of shape `(num_sequences + 1,)`):
            Cumulative patch counts that delimit each image or video frame in the packed sequence.
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens=cu_seqlens, **kwargs)
        return hidden_states


class Cosmos3EdgeVisionModel(Cosmos3EdgePreTrainedModel):
    """Packed variable-resolution SigLIP2 vision tower used by Cosmos3 Edge."""

    config_class = Cosmos3EdgeVisionConfig
    base_model_prefix = "visual"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    _no_split_modules = ["Cosmos3EdgeVisionEncoderLayer"]

    def __init__(self, config: Cosmos3EdgeVisionConfig):
        super().__init__(config)
        self.embeddings = Cosmos3EdgeVisionEmbeddings(config)
        self.encoder = Cosmos3EdgeEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    @auto_docstring
    def forward(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor, **kwargs) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height, and width patch-grid dimensions for each packed image or video.
        """
        cu_seqlens = get_vision_cu_seqlens(grid_thw, kwargs=kwargs)
        hidden_states = self.embeddings(pixel_values, grid_thw)
        hidden_states = self.encoder(hidden_states, cu_seqlens=cu_seqlens, **kwargs)
        return self.post_layernorm(hidden_states)


class Cosmos3EdgePatchMerger(nn.Module):
    def __init__(self, config: Cosmos3EdgeProjectorConfig):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.input_hidden_size = config.input_hidden_size
        self.hidden_size = config.input_hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = config.use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if self.use_postshuffle_norm else config.input_hidden_size,
            eps=1e-6,
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, config.merger_intermediate_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(config.merger_intermediate_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.reshape(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states).reshape(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Cosmos3EdgeModel(Cosmos3EdgePreTrainedModel):
    accepts_loss_kwargs = False

    def __init__(self, config: Cosmos3EdgeConfig):
        super().__init__(config)
        text_config = config.text_config
        vision_config = config.vision_config
        projector_config = config.projector_config
        self.visual = Cosmos3EdgeVisionModel._from_config(vision_config)
        self.projector = Cosmos3EdgePatchMerger(projector_config)
        self.language_model = Cosmos3EdgeTextModel._from_config(text_config)
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_decoder(self):
        return self.language_model

    def set_decoder(self, decoder):
        self.language_model = decoder

    def _get_visual_features(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor):
        vision_hidden_states = self.visual(pixel_values, grid_thw=grid_thw)
        projector_input = vision_hidden_states.reshape(
            -1, self.projector.spatial_merge_size**2, self.projector.input_hidden_size
        )
        projected_hidden_states = self.projector(projector_input)
        split_sizes = (grid_thw.prod(dim=-1) // self.projector.spatial_merge_size**2).tolist()
        return torch.split(projected_hidden_states, split_sizes)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor, **kwargs):
        return self._get_visual_features(pixel_values, image_grid_thw)

    def get_video_features(self, pixel_values_videos: torch.FloatTensor, video_grid_thw: torch.LongTensor, **kwargs):
        return self._get_visual_features(pixel_values_videos, video_grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        image_mask = input_ids == self.config.image_token_id
        video_mask = input_ids == self.config.video_token_id
        expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        expanded_video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if image_features is not None and expanded_image_mask.sum() != image_features.numel():
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {image_mask.sum()}, features: {image_features.shape[0]}."
            )
        if video_features is not None and expanded_video_mask.sum() != video_features.numel():
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens: {video_mask.sum()}, features: {video_features.shape[0]}."
            )
        return expanded_image_mask, expanded_video_mask

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
            video_grid_thw[:, 0] = 1

        merge_size = self.projector.spatial_merge_size
        rope_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        modality_groups = {1: 0, 2: 0}

        for batch_index, token_types in enumerate(mm_token_type_ids):
            if attention_mask is not None:
                token_types = token_types[attention_mask[batch_index].bool()]

            current_position = 0
            position_chunks = []
            for modality_type, group in itertools.groupby(token_types.tolist()):
                group_length = len(list(group))
                if modality_type == 0:
                    positions = torch.arange(group_length, device=input_ids.device).view(1, -1).expand(3, -1)
                elif modality_type in grid_iters:
                    modality_groups[modality_type] += 1
                    if grid_iters[modality_type] is None:
                        modality = "image" if modality_type == 1 else "video"
                        raise ValueError(f"{modality.capitalize()} tokens were supplied without a matching grid.")
                    try:
                        temporal, height, width = next(grid_iters[modality_type]).tolist()
                    except StopIteration as error:
                        modality = "image" if modality_type == 1 else "video"
                        raise ValueError(f"More {modality} token groups were supplied than matching grids.") from error

                    grid_height, grid_width = height // merge_size, width // merge_size
                    temporal_ids = torch.arange(temporal, device=input_ids.device).view(-1, 1, 1)
                    height_ids = torch.arange(grid_height, device=input_ids.device).view(1, -1, 1)
                    width_ids = torch.arange(grid_width, device=input_ids.device).view(1, 1, -1)
                    positions = torch.stack(
                        torch.broadcast_tensors(temporal_ids, height_ids, width_ids), dim=0
                    ).reshape(3, -1)
                    if positions.shape[-1] != group_length:
                        raise ValueError("Visual token counts do not match the supplied image/video grids.")
                else:
                    raise ValueError(f"Unsupported multimodal token type: {modality_type}.")

                position_chunks.append(positions + current_position)
                current_position = int(position_chunks[-1].max()) + 1

            sample_position_ids = torch.cat(position_chunks, dim=1) if position_chunks else input_ids.new_empty((3, 0))
            if sample_position_ids.shape[-1] != token_types.shape[-1]:
                raise ValueError("Visual token counts do not match the supplied image/video grids.")
            if attention_mask is None:
                position_ids[:, batch_index] = sample_position_ids
            else:
                position_ids[:, batch_index, attention_mask[batch_index].bool()] = sample_position_ids
            rope_deltas.append(current_position - token_types.shape[-1] if token_types.shape[-1] else 0)

        if image_grid_thw is not None and modality_groups[1] != len(image_grid_thw):
            raise ValueError("The number of image grids does not match the number of image token groups.")
        if video_grid_thw is not None and modality_groups[2] != len(video_grid_thw):
            raise ValueError("The number of video grids does not match the number of video token groups.")
        return position_ids, torch.tensor(rope_deltas, dtype=torch.long, device=input_ids.device).unsqueeze(1)

    def compute_3d_position_ids(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.LongTensor:
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
                "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
                "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )
        if (
            input_ids is not None
            and mm_token_type_ids is not None
            and has_multimodal
            and (self.rope_deltas is None or past_length == 0)
        ):
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            return position_ids

        batch_size, sequence_length = inputs_embeds.shape[:2]
        if attention_mask is not None:
            text_position_ids = attention_mask.long().cumsum(-1) - 1
            text_position_ids.masked_fill_(attention_mask == 0, 0)
            text_position_ids = text_position_ids[:, -sequence_length:]
        else:
            text_position_ids = torch.arange(past_length, past_length + sequence_length, device=inputs_embeds.device)
            text_position_ids = text_position_ids.unsqueeze(0).expand(batch_size, -1)
        position_ids = text_position_ids.unsqueeze(0).expand(3, -1, -1)
        if self.rope_deltas is not None and (past_length or input_ids is None):
            position_ids = position_ids + self.rope_deltas.to(device=inputs_embeds.device).unsqueeze(0)
        return position_ids

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height, and width patch-grid dimensions for each image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height, and width patch-grid dimensions for each video.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` must be set when `inputs_embeds` is not provided.")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            if input_ids is None or image_grid_thw is None:
                raise ValueError("`input_ids` and `image_grid_thw` are required when `pixel_values` is provided.")
            image_features = torch.cat(self.get_image_features(pixel_values, image_grid_thw), dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_features)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        if pixel_values_videos is not None:
            if input_ids is None or video_grid_thw is None:
                raise ValueError(
                    "`input_ids` and `video_grid_thw` are required when `pixel_values_videos` is provided."
                )
            video_features = torch.cat(self.get_video_features(pixel_values_videos, video_grid_thw), dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, video_features=video_features)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Cosmos3EdgeForConditionalGeneration(Cosmos3EdgePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {}
    accepts_loss_kwargs = False

    def __init__(self, config: Cosmos3EdgeConfig):
        super().__init__(config)
        text_config = config.text_config
        if not isinstance(text_config, Cosmos3EdgeTextConfig):
            raise TypeError("`text_config` must be a `Cosmos3EdgeTextConfig`.")
        self.model = Cosmos3EdgeModel(config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model.get_decoder()

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_image_features(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)

    def get_video_features(self, *args, **kwargs):
        return self.model.get_video_features(*args, **kwargs)

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height, and width patch-grid dimensions for each image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height, and width patch-grid dimensions for each video.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        cache = model_kwargs.get("past_key_values")
        if cache is not None and cache.get_seq_length() and self.model.rope_deltas is not None:
            return text_positions.unsqueeze(0).expand(3, -1, -1) + self.model.rope_deltas.unsqueeze(0)

        input_ids = model_kwargs.get("input_ids", inputs_tensor)
        has_multimodal = (
            model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None
        )
        if input_ids is not None and has_multimodal:
            if model_kwargs.get("mm_token_type_ids") is None:
                raise ValueError(
                    "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
                    "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
                    "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
                )
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                mm_token_type_ids=model_kwargs["mm_token_type_ids"],
                image_grid_thw=model_kwargs.get("image_grid_thw"),
                video_grid_thw=model_kwargs.get("video_grid_thw"),
                attention_mask=model_kwargs.get("attention_mask"),
            )
            self.model.rope_deltas = rope_deltas
            return position_ids

        self.model.rope_deltas = torch.zeros(inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device)
        return text_positions.unsqueeze(0).expand(3, -1, -1)


__all__ = [
    "Cosmos3EdgeConfig",
    "Cosmos3EdgeTextConfig",
    "Cosmos3EdgeVisionConfig",
    "Cosmos3EdgeProjectorConfig",
    "Cosmos3EdgeModel",
    "Cosmos3EdgeTextModel",
    "Cosmos3EdgeVisionModel",
    "Cosmos3EdgeForConditionalGeneration",
    "Cosmos3EdgePreTrainedModel",
]
