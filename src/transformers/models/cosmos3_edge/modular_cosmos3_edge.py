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

from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torch.nn.utils.rnn import pad_sequence

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import auto_docstring, can_return_tuple, logging
from ...utils.generic import maybe_autocast
from ..llama.modeling_llama import LlamaRotaryEmbedding
from ..nemotron_h.modeling_nemotron_h import (
    NemotronHMLP,
    NemotronHRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
from ..siglip2.modeling_siglip2 import Siglip2Encoder, Siglip2VisionEmbeddings


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeTextConfig(PreTrainedConfig):
    r"""
    Configuration for the dense Nemotron-derived text tower used by Cosmos3 Edge.

    Args:
        num_logits_to_keep (`int`, *optional*, defaults to 1):
            Number of final token logits to compute. Set to `None` to compute logits for every token.
        rope_theta (`float` or `int`, *optional*, defaults to 100000000.0):
            Base period used by rotary position embeddings.
        mrope_section (`list[int]` or `tuple[int, int, int]`, *optional*, defaults to `(24, 20, 20)`):
            Per-axis rotary-frequency sections for temporal, height, and width position IDs.
    """

    model_type = "cosmos3_edge_text"
    ignore_keys_at_rope_validation = {"mrope_section"}

    vocab_size: int = 131072
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 56
    layers_block_type: list[str] | None = None
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    head_dim: int = 128
    max_position_embeddings: int = 131072
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False
    mlp_hidden_act: str = "relu2"
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    hidden_dropout: float | int = 0.0
    use_cache: bool = True
    num_logits_to_keep: int = 1
    rope_theta: float | int = 100000000.0
    rope_parameters: dict | None = None
    mrope_section: list[int] | tuple[int, int, int] = (24, 20, 20)
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 11

    def __post_init__(self, **kwargs):
        legacy_pattern = kwargs.pop("hybrid_override_pattern", None)
        if self.layers_block_type is None:
            if legacy_pattern is not None:
                mapping = {"*": "full_attention", "-": "mlp"}
                self.layers_block_type = [mapping[layer] for layer in legacy_pattern]
            else:
                self.layers_block_type = ["full_attention", "mlp"] * 28

        invalid_layer_types = set(self.layers_block_type) - {"full_attention", "mlp"}
        if invalid_layer_types:
            raise ValueError(
                f"Cosmos3 Edge only supports `full_attention` and `mlp` layers, got {sorted(invalid_layer_types)}."
            )
        self.num_hidden_layers = len(self.layers_block_type)
        self.rope_theta = float(self.rope_theta)

        self.rope_parameters = dict(self.rope_parameters or {})
        self.rope_parameters.setdefault("rope_type", "default")
        self.rope_parameters.setdefault("rope_theta", self.rope_theta)
        self.rope_parameters.setdefault("mrope_section", list(self.mrope_section))

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)

    def validate_layer_type(self):
        """Validate Edge's alternating attention and dense MLP physical layers."""
        if not isinstance(self.layers_block_type, list):
            raise ValueError("`layers_block_type` must be a list of strings.")
        if len(self.layers_block_type) != self.num_hidden_layers:
            raise ValueError("`num_hidden_layers` must equal the number of `layers_block_type` entries.")
        invalid_layer_types = set(self.layers_block_type) - {"full_attention", "mlp"}
        if invalid_layer_types:
            raise ValueError(
                f"Cosmos3 Edge only supports `full_attention` and `mlp` layers, got {sorted(invalid_layer_types)}."
            )


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeVisionConfig(Siglip2VisionConfig):
    r"""
    Configuration for the SigLIP2 vision tower used by Cosmos3 Edge.

    Args:
        num_patches (`int`, *optional*, defaults to 256):
            Number of patches in the learned reference positional-embedding grid.
        vision_use_head (`bool`, *optional*, defaults to `False`):
            Whether to construct the SigLIP2 pooling head. Cosmos3 Edge does not use this head.
    """

    model_type = "cosmos3_edge_vision"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    num_patches: int = 256
    spatial_merge_size: int = 2
    vision_use_head: bool = False


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeProjectorConfig(PreTrainedConfig):
    r"""
    Configuration for the Cosmos3 Edge spatial patch merger.

    Args:
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

    def __post_init__(self, **kwargs):
        legacy_intermediate_size = kwargs.pop("merger_intermedia", None)
        if legacy_intermediate_size is not None:
            self.merger_intermediate_size = legacy_intermediate_size
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeConfig(PreTrainedConfig):
    r"""
    Args:
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
            text_config = dict(self.text_config)
            text_config.pop("model_type", None)
            self.text_config = Cosmos3EdgeTextConfig(**text_config)

        if self.vision_config is None:
            self.vision_config = Cosmos3EdgeVisionConfig()
        elif isinstance(self.vision_config, dict):
            vision_config = dict(self.vision_config)
            vision_config.pop("model_type", None)
            self.vision_config = Cosmos3EdgeVisionConfig(**vision_config)

        if self.projector_config is None:
            self.projector_config = Cosmos3EdgeProjectorConfig(
                input_hidden_size=self.vision_config.hidden_size,
                out_hidden_size=self.text_config.hidden_size,
                spatial_merge_size=self.vision_config.spatial_merge_size,
            )
        elif isinstance(self.projector_config, dict):
            projector_config = dict(self.projector_config)
            projector_config.pop("model_type", None)
            self.projector_config = Cosmos3EdgeProjectorConfig(**projector_config)

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
        if config.rope_parameters is None:
            raise ValueError("`rope_parameters` must be set for Cosmos3 Edge M-RoPE.")
        super().__init__(config, device=device)
        self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])

    @staticmethod
    def compute_default_rope_parameters(
        config: Cosmos3EdgeTextConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        if config is None or config.rope_parameters is None:
            raise ValueError("`config.rope_parameters` must be set to initialize M-RoPE.")

        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim or config.hidden_size // config.num_attention_heads
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


class Cosmos3EdgeTextAttention(nn.Module):
    """Dense GQA attention with Cosmos3 Edge M-RoPE."""

    def __init__(self, config: Cosmos3EdgeTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class Cosmos3EdgeTextMLP(NemotronHMLP):
    pass


class Cosmos3EdgeTextRMSNorm(NemotronHRMSNorm):
    pass


class Cosmos3EdgeTextLayer(nn.Module):
    def __init__(self, config: Cosmos3EdgeTextConfig, layer_idx: int):
        super().__init__()
        self.norm = Cosmos3EdgeTextRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.block_type = config.layers_block_type[layer_idx]
        if self.block_type == "full_attention":
            self.mixer = Cosmos3EdgeTextAttention(config, layer_idx)
        else:
            self.mixer = Cosmos3EdgeTextMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.block_type == "full_attention":
            hidden_states, _ = self.mixer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            hidden_states = self.mixer(hidden_states)
        return residual + hidden_states


class Cosmos3EdgeTextModel(PreTrainedModel):
    config_class = Cosmos3EdgeTextConfig
    base_model_prefix = "language_model"
    _no_split_modules = ["Cosmos3EdgeTextLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = True

    def __init__(self, config: Cosmos3EdgeTextConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Cosmos3EdgeTextLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm_f = Cosmos3EdgeTextRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.rotary_emb = Cosmos3EdgeTextRotaryEmbedding(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` must be set when `inputs_embeds` is not provided.")
            inputs_embeds = self.embeddings(input_ids)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        batch_size, sequence_length = inputs_embeds.shape[:2]
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = (
                torch.arange(past_seen_tokens, past_seen_tokens + sequence_length, device=inputs_embeds.device)
                .view(1, 1, -1)
                .expand(3, batch_size, -1)
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        mask_position_ids = position_ids[0]
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=mask_position_ids,
        )
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        hidden_states = self.norm_f(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Cosmos3EdgeVisionEmbeddings(Siglip2VisionEmbeddings):
    pass


class Cosmos3EdgeEncoder(Siglip2Encoder):
    pass


class Cosmos3EdgeVisionModel(nn.Module):
    """SigLIP2 vision tower accepting the packed patches emitted by the Edge processor."""

    def __init__(self, config: Cosmos3EdgeVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Cosmos3EdgeVisionEmbeddings(config)
        self.encoder = Cosmos3EdgeEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor, **kwargs) -> torch.Tensor:
        if grid_thw is None:
            raise ValueError("`grid_thw` is required for Cosmos3 Edge visual inputs.")
        frame_spatial_shapes = torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], dim=0)
        frame_lengths = frame_spatial_shapes.prod(dim=-1)
        if int(frame_lengths.sum()) != pixel_values.shape[0]:
            raise ValueError(
                "The packed visual patch count does not match `grid_thw`: "
                f"got {pixel_values.shape[0]} patches for {int(frame_lengths.sum())} expected patches."
            )

        max_length = int(frame_lengths.max())
        padded_pixel_values = pixel_values.new_zeros(
            (frame_spatial_shapes.shape[0], max_length, pixel_values.shape[-1])
        )
        pixel_attention_mask = torch.zeros(
            (frame_spatial_shapes.shape[0], max_length), dtype=torch.long, device=pixel_values.device
        )
        offset = 0
        for frame_idx, frame_length in enumerate(frame_lengths.tolist()):
            padded_pixel_values[frame_idx, :frame_length] = pixel_values[offset : offset + frame_length]
            pixel_attention_mask[frame_idx, :frame_length] = 1
            offset += frame_length

        hidden_states = self.embeddings(padded_pixel_values, frame_spatial_shapes)
        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=pixel_attention_mask,
        )
        outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = self.post_layernorm(outputs.last_hidden_state)
        return torch.cat(
            [hidden_states[idx, :frame_length] for idx, frame_length in enumerate(frame_lengths.tolist())],
            dim=0,
        )


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


@dataclass
class Cosmos3EdgeModelOutputWithPast(BaseModelOutputWithPast):
    rope_deltas: torch.LongTensor | None = None


_COSMOS3_EDGE_DROPPED_GENERATOR_KEYS = [
    # The unified Diffusers shards also contain Generator-only modules. The
    # reasoner text weights share the same physical blocks and are remapped by
    # `conversion_mapping.py`; only these disjoint Generator parameters are
    # intentionally ignored by the Transformers reasoner.
    r"^time_embedder\.",
    r"^proj_(?:in|out)\.",
    r"^norm_moe_gen\.",
    r"^layers\.\d+\.input_layernorm_moe_gen\.",
    r"^layers\.\d+\.post_attention_layernorm_moe_gen\.",
    r"^layers\.\d+\.mlp_moe_gen\.",
    r"^layers\.\d+\.self_attn\.(?:add_[qkv]_proj|to_add_out|norm_added_[qk])\.",
]


class Cosmos3EdgePreTrainedModel(PreTrainedModel):
    config_class = Cosmos3EdgeConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    _no_split_modules = ["Cosmos3EdgeTextLayer", "Cosmos3EdgeEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _keys_to_ignore_on_load_unexpected = _COSMOS3_EDGE_DROPPED_GENERATOR_KEYS


class Cosmos3EdgeModel(Cosmos3EdgePreTrainedModel):
    accepts_loss_kwargs = False

    def __init__(self, config: Cosmos3EdgeConfig):
        super().__init__(config)
        text_config = config.text_config
        vision_config = config.vision_config
        projector_config = config.projector_config
        if not isinstance(text_config, Cosmos3EdgeTextConfig):
            raise TypeError("`text_config` must be a `Cosmos3EdgeTextConfig`.")
        if not isinstance(vision_config, Cosmos3EdgeVisionConfig):
            raise TypeError("`vision_config` must be a `Cosmos3EdgeVisionConfig`.")
        if not isinstance(projector_config, Cosmos3EdgeProjectorConfig):
            raise TypeError("`projector_config` must be a `Cosmos3EdgeProjectorConfig`.")
        # The packed vision tower is intentionally a lightweight `nn.Module`
        # rather than a standalone `PreTrainedModel`, so propagate the root
        # attention backend that `_from_config` would normally initialize.
        vision_config._attn_implementation = config._attn_implementation
        self.visual = Cosmos3EdgeVisionModel(vision_config)
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

    def _patch_merge(self, image_embeds: torch.Tensor, grid_thw: torch.LongTensor):
        merge_size = self.projector.spatial_merge_size
        hidden_size = image_embeds.shape[-1]
        merged_embeds = []
        merged_grid = []
        offset = 0
        for temporal, height, width in grid_thw.tolist():
            if height % merge_size or width % merge_size:
                raise ValueError(
                    f"Visual grid ({height}, {width}) must be divisible by spatial merge size {merge_size}."
                )
            num_patches = temporal * height * width
            media_embeds = image_embeds[offset : offset + num_patches]
            offset += num_patches
            media_embeds = media_embeds.reshape(temporal, height, width, hidden_size)
            media_embeds = media_embeds.reshape(
                temporal,
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                hidden_size,
            )
            media_embeds = media_embeds.permute(0, 1, 3, 2, 4, 5).reshape(-1, merge_size**2 * hidden_size)
            merged_embeds.append(media_embeds)
            merged_grid.append((temporal, height // merge_size, width // merge_size))
        if offset != image_embeds.shape[0]:
            raise ValueError("`grid_thw` does not account for every visual patch.")
        return torch.cat(merged_embeds, dim=0), torch.tensor(merged_grid, device=grid_thw.device, dtype=grid_thw.dtype)

    def _get_visual_features(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor):
        vision_hidden_states = self.visual(
            pixel_values.to(dtype=self.visual.embeddings.patch_embedding.weight.dtype), grid_thw=grid_thw
        )
        merged_hidden_states, merged_grid_thw = self._patch_merge(vision_hidden_states, grid_thw)
        projector_input = merged_hidden_states.reshape(
            -1, self.projector.spatial_merge_size**2, self.projector.input_hidden_size
        )
        projected_hidden_states = self.projector(projector_input)
        split_sizes = merged_grid_thw.prod(dim=-1).tolist()
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
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
            video_grid_thw[:, 0] = 1

        image_index = 0
        video_index = 0
        all_position_ids = []
        rope_deltas = []
        merge_size = self.projector.spatial_merge_size

        for batch_index in range(input_ids.shape[0]):
            ids = input_ids[batch_index]
            if attention_mask is not None:
                ids = ids[attention_mask[batch_index].to(dtype=torch.bool)]
            ids_list = ids.tolist()
            vision_starts = (ids == self.config.vision_start_token_id).nonzero(as_tuple=False).flatten()
            vision_tokens = ids[vision_starts + 1] if len(vision_starts) else ids.new_empty(0)
            remaining_images = int((vision_tokens == self.config.image_token_id).sum())
            remaining_videos = int((vision_tokens == self.config.video_token_id).sum())

            position_chunks = []
            start = 0
            for _ in range(remaining_images + remaining_videos):
                image_end = (
                    ids_list.index(self.config.image_token_id, start)
                    if remaining_images and self.config.image_token_id in ids_list[start:]
                    else len(ids_list) + 1
                )
                video_end = (
                    ids_list.index(self.config.video_token_id, start)
                    if remaining_videos and self.config.video_token_id in ids_list[start:]
                    else len(ids_list) + 1
                )
                if image_end < video_end:
                    if image_grid_thw is None:
                        raise ValueError("Image tokens were supplied without `image_grid_thw`.")
                    temporal, height, width = image_grid_thw[image_index].tolist()
                    image_index += 1
                    remaining_images -= 1
                    end = image_end
                else:
                    if video_grid_thw is None:
                        raise ValueError("Video tokens were supplied without `video_grid_thw`.")
                    temporal, height, width = video_grid_thw[video_index].tolist()
                    video_index += 1
                    remaining_videos -= 1
                    end = video_end

                base = int(position_chunks[-1].max()) + 1 if position_chunks else 0
                text_length = end - start
                if text_length:
                    position_chunks.append(
                        torch.arange(text_length, device=ids.device).view(1, -1).expand(3, -1) + base
                    )
                visual_base = base + text_length
                grid_height = height // merge_size
                grid_width = width // merge_size
                temporal_ids = (
                    torch.arange(temporal, device=ids.device)
                    .view(-1, 1)
                    .expand(-1, grid_height * grid_width)
                    .flatten()
                )
                height_ids = (
                    torch.arange(grid_height, device=ids.device)
                    .view(1, -1, 1)
                    .expand(temporal, -1, grid_width)
                    .flatten()
                )
                width_ids = (
                    torch.arange(grid_width, device=ids.device)
                    .view(1, 1, -1)
                    .expand(temporal, grid_height, -1)
                    .flatten()
                )
                position_chunks.append(torch.stack((temporal_ids, height_ids, width_ids)) + visual_base)
                start = end + temporal * grid_height * grid_width

            if start < len(ids_list):
                base = int(position_chunks[-1].max()) + 1 if position_chunks else 0
                position_chunks.append(
                    torch.arange(len(ids_list) - start, device=ids.device).view(1, -1).expand(3, -1) + base
                )

            position_ids = torch.cat(position_chunks, dim=1) if position_chunks else ids.new_empty((3, 0))
            if position_ids.shape[-1] != len(ids_list):
                raise ValueError("Visual token counts do not match the supplied image/video grids.")
            all_position_ids.append(position_ids)
            rope_deltas.append(position_ids.max() + 1 - len(ids_list) if len(ids_list) else 0)

        return (
            pad_sequence(all_position_ids, batch_first=False, padding_value=1),
            torch.as_tensor(rope_deltas, dtype=torch.long, device=input_ids.device).unsqueeze(1),
        )

    def compute_3d_position_ids(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.LongTensor:
        past_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        if (
            input_ids is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
            and (self.rope_deltas is None or past_length == 0)
        ):
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids,
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
        if self.rope_deltas is None:
            self.rope_deltas = torch.zeros(batch_size, 1, dtype=torch.long, device=inputs_embeds.device)
        elif past_length:
            position_ids = position_ids + self.rope_deltas.to(device=inputs_embeds.device).unsqueeze(0)
        return position_ids

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
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | Cosmos3EdgeModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("`input_ids` must be set when `inputs_embeds` is not provided.")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        assert inputs_embeds is not None

        if pixel_values is not None:
            if input_ids is None or image_grid_thw is None:
                raise ValueError("`input_ids` and `image_grid_thw` are required when `pixel_values` is provided.")
            image_features = torch.cat(self.get_image_features(pixel_values, image_grid_thw), dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_features)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            assert inputs_embeds is not None
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
            assert inputs_embeds is not None

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
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
        return Cosmos3EdgeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


@dataclass
class Cosmos3EdgeCausalLMOutputWithPast(CausalLMOutputWithPast):
    rope_deltas: torch.LongTensor | None = None


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
        logits_to_keep: int | torch.Tensor = 0,
        use_cache: bool | None = None,
        **kwargs,
    ) -> tuple | Cosmos3EdgeCausalLMOutputWithPast:
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

        return Cosmos3EdgeCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
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
        is_first_iteration=False,
        **kwargs,
    ):
        kwargs["logits_to_keep"] = self.config.text_config.num_logits_to_keep
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
        if input_ids is not None and (
            model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None
        ):
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
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
    "Cosmos3EdgeForConditionalGeneration",
    "Cosmos3EdgePreTrainedModel",
]
