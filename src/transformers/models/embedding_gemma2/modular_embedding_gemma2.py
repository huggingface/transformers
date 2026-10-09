# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from typing import Any, Literal

import numpy as np
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...audio_utils import AudioInput, is_valid_audio
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_utils import ImageInput, make_nested_list_of_images
from ...integrations import use_kernelized_func
from ...masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
    torch_compilable_check,
)
from ...video_processing_utils import VideoMetadata
from ...video_utils import VideoInput, is_valid_video, make_batched_videos
from ..auto import AutoConfig
from ..gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3MLP, Gemma3TextModel, apply_rotary_pos_emb
from ..gemma4.modeling_gemma4 import (
    Gemma4AudioModelOutput,
    Gemma4Model,
    Gemma4MultimodalEmbedder,
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    eager_attention_forward,
)
from ..gemma4.processing_gemma4 import Gemma4Processor, Gemma4ProcessorKwargs
from ..gemma4.video_processing_gemma4 import (
    Gemma4VideoProcessor,
    Gemma4VideoProcessorKwargs,
    get_aspect_ratio_preserving_size,  # noqa: F401
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/embeddinggemma-2")
@strict
class EmbeddingGemma2TextConfig(PreTrainedConfig):
    r"""
    sliding_window (`int`, *optional*, defaults to 512):
        Inclusive radius of the bidirectional sliding window: a `sliding_attention` layer attends to
        every position with `abs(q_idx - kv_idx) <= sliding_window`. It is a radius and not the
        one-sided width the causal Gemma models configure, because the mask here is symmetric, so
        the default is half of the 1024-wide window the reference implementation states.
    hidden_size_per_layer_input (`int`, *optional*, defaults to 512):
        Dimensionality of the per-layer (PLE) residual signal. EmbeddingGemma 2 uses
        *projection-only* PLE: the signal is derived from `inputs_embeds` alone, with no
        auxiliary token lookup table.
    embedding_dim (`int`, *optional*, defaults to 768):
        Dimensionality of the sentence embedding produced by `embedding_projection`.
    """

    model_type = "embedding_gemma2_text"
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.v_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 262_144
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 262_144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    sliding_window: int = 512
    layer_types: list[str] | None = None
    hidden_size_per_layer_input: int = 512
    embedding_dim: int = 768

    def __post_init__(self, **kwargs):
        # `sliding_window_pattern`, `global_head_dim` and `num_global_key_value_heads` are builder-only
        # kwargs, as in Gemma 4: they shape `layer_types` and `per_layer_config` and are not kept on
        # the config, which exposes the derived values instead.
        sliding_window_pattern = kwargs.pop("sliding_window_pattern", 6)
        if sliding_window_pattern < 1:
            raise ValueError(f"`sliding_window_pattern` must be a positive integer, got {sliding_window_pattern}.")

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        if self.layer_types and (last_layer_type := self.layer_types[-1]) != "full_attention":
            logger.warning(
                f"Last layer must use `full_attention`, but got `{last_layer_type}`. Forcing last layer to `full_attention`."
            )
            self.layer_types[-1] = "full_attention"

        default_rope_params: dict[Literal["full_attention", "sliding_attention"] : dict[str, Any]] = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
        }
        if self.rope_parameters is None:
            self.rope_parameters = default_rope_params

        # Full-attention layers are wider, hence the separate head dimension and kv head count.
        global_head_dim = kwargs.pop("global_head_dim", 512)
        num_global_key_value_heads = kwargs.pop("num_global_key_value_heads", 1)
        if "per_layer_config" not in kwargs:
            kwargs["per_layer_config"] = {
                layer_idx: {"head_dim": global_head_dim, "num_key_value_heads": num_global_key_value_heads}
                for layer_idx, layer_type in enumerate(self.layer_types)
                if layer_type == "full_attention"
            }

        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        # No need to handle BC for new models, because they have no old-format `rope_scaling`
        return kwargs


@auto_docstring(checkpoint="google/embeddinggemma-2")
@strict
class EmbeddingGemma2Config(PreTrainedConfig):
    r"""
    text_config (`EmbeddingGemma2TextConfig`, *optional*):
        Configuration of the text backbone.
    vision_config (`PreTrainedConfig` or `dict`, *optional*):
        Configuration of the vision tower. Reused verbatim from Gemma 4; the tower itself is
        resolved at runtime through `AutoModel`.
    audio_config (`PreTrainedConfig` or `dict`, *optional*):
        Configuration of the audio tower. Reused verbatim from Gemma 4; the tower itself is
        resolved at runtime through `AutoModel`.
    boi_token_id (`int`, *optional*, defaults to 255999):
        The begin-of-image token index to wrap the image prompt.
    eoi_token_id (`int`, *optional*, defaults to 258882):
        The end-of-image token index to wrap the image prompt.
    boa_token_id (`int`, *optional*, defaults to 256000):
        The begin-of-audio token index to wrap the audio prompt.
    eoa_token_index (`int`, *optional*, defaults to 258883):
        The end-of-audio token index to wrap the audio prompt.
    """

    model_type = "embedding_gemma2"
    sub_configs_defaults = {
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="gemma4_vision", optional=True),
        "text_config": SubConfigSpec(config_class=EmbeddingGemma2TextConfig),
        "audio_config": SubConfigSpec(config_class=AutoConfig, model_type="gemma4_audio", optional=True),
    }

    text_config: EmbeddingGemma2TextConfig | dict[str, Any] | None = None
    vision_config: PreTrainedConfig | dict[str, Any] | None = None
    audio_config: PreTrainedConfig | dict[str, Any] | None = None
    boi_token_id: int | None = 255_999
    eoi_token_id: int | None = 258_882
    image_token_id: int | None = 258_880
    video_token_id: int | None = 258_884
    boa_token_id: int | None = 256_000
    eoa_token_index: int | None = 258_883
    audio_token_id: int | None = 258_881
    initializer_range: float | None = 0.02


class EmbeddingGemma2RMSNorm(Gemma4RMSNorm):
    pass


class EmbeddingGemma2RotaryEmbedding(Gemma4TextRotaryEmbedding):
    pass


class EmbeddingGemma2TextScaledWordEmbedding(Gemma4TextScaledWordEmbedding):
    pass


class EmbeddingGemma2MLP(Gemma3MLP):
    pass


class EmbeddingGemma2TextPLE(nn.Module):
    """Produces the per-layer embeddings (PLE) that every decoder layer is gated with.

    EmbeddingGemma 2 uses *projection-only* PLE: the signal is derived from `inputs_embeds` alone,
    with no token-identity lookup and no blending scale.
    """

    def __init__(self, config: EmbeddingGemma2TextConfig):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size,
            config.num_hidden_layers * config.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_model_projection_scale = config.hidden_size**-0.5
        self.per_layer_projection_norm = EmbeddingGemma2RMSNorm(
            config.hidden_size_per_layer_input, eps=config.rms_norm_eps
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return self.per_layer_projection_norm(per_layer_projection)


class EmbeddingGemma2TextPLEBlock(nn.Module):
    """Mixes one layer's slice of the per-layer embeddings into the residual stream.

    The third residual sub-block of a decoder layer, after attention and the MLP. The defining
    operation is the elementwise gate against `per_layer_input`; the two linears are a bottleneck
    down to `hidden_size_per_layer_input` and back.
    """

    def __init__(self, config: EmbeddingGemma2TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        self.act_fn = ACT2FN[config.hidden_activation]
        self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
        self.post_per_layer_input_norm = EmbeddingGemma2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.per_layer_input_gate(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = self.per_layer_projection(hidden_states)
        hidden_states = self.post_per_layer_input_norm(hidden_states)
        return residual + hidden_states


@use_kernelized_func(apply_rotary_pos_emb)
class EmbeddingGemma2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper.

    EmbeddingGemma 2 is an encoder: attention is bidirectional on every layer and no key-value
    state is carried between calls.
    """

    def __init__(self, config: EmbeddingGemma2TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # +1 is needed because flash attention sets inclusive boundaries (see modeling_flash_attention_utils.py)
        self.sliding_window = (
            config.sliding_window + 1 if config.layer_types[layer_idx] == "sliding_attention" else None
        )

        layer_config = config.per_layer_config[layer_idx]
        self.head_dim = layer_config.head_dim
        self.num_key_value_groups = config.num_attention_heads // layer_config.num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, layer_config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, layer_config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = EmbeddingGemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = EmbeddingGemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = EmbeddingGemma2RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class EmbeddingGemma2EncoderLayer(Gemma3DecoderLayer):
    """A single transformer block. Named an *encoder* layer because attention is bidirectional."""

    def __init__(self, config: EmbeddingGemma2TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_scalar = nn.Buffer(torch.ones(1))
        self.ple_block = EmbeddingGemma2TextPLEBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = self.ple_block(hidden_states, per_layer_input)

        hidden_states *= self.layer_scalar
        return hidden_states


class EmbeddingGemma2PreTrainedModel(Gemma4PreTrainedModel):
    config: EmbeddingGemma2Config
    _no_split_modules = ["EmbeddingGemma2EncoderLayer"]
    # Deletes the inherited attribute: both entries were KV-cache related, which we do not have.
    _skip_keys_device_placement = AttributeError()

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, EmbeddingGemma2RotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_config = module.config.per_layer_config[layer_type]
                curr_inv_freq, _ = rope_init_fn(rope_config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, EmbeddingGemma2TextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, EmbeddingGemma2EncoderLayer):
            init.ones_(module.layer_scalar)

    def get_per_layer_input_embeddings(self):
        raise AttributeError("Deleted: EmbeddingGemma 2 has no `embed_tokens_per_layer` table.")

    def set_per_layer_input_embeddings(self, value):  # trf-ignore: TRF033
        raise AttributeError("Deleted: no `embed_tokens_per_layer` table.")

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        raise AttributeError("Deleted: no `embed_tokens_per_layer` table to resize alongside.")

    def _resize_per_layer_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None, mean_resizing=True):
        raise AttributeError("Deleted: no `embed_tokens_per_layer` table to resize.")


@auto_docstring(
    custom_intro="""
    The EmbeddingGemma 2 text backbone. It owns the `embedding_projection` that maps the final hidden states
    down to `config.embedding_dim`, so that the composite model needs no `forward` override.
    """
)
class EmbeddingGemma2TextModel(Gemma3TextModel):
    _can_record_outputs = {
        "hidden_states": EmbeddingGemma2EncoderLayer,
        "attentions": EmbeddingGemma2Attention,
    }

    def __init__(self, config: EmbeddingGemma2TextConfig):
        super().__init__(config)

        # bfloat16 rounding turns sqrt(512)=22.6274 into 22.625; see https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = EmbeddingGemma2TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [EmbeddingGemma2EncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.unique_layer_types = set(self.config.layer_types)
        self.ple = EmbeddingGemma2TextPLE(config)

        # Projecting per token is equivalent to projecting after mean pooling
        self.embedding_projection = nn.Linear(config.hidden_size, config.embedding_dim, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        per_layer_inputs = self.ple(inputs_embeds)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        # It may already have been prepared as a mapping by the caller
        if not isinstance(attention_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
            attention_mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                per_layer_inputs[:, :, i, :],
                attention_mask=attention_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.embedding_projection(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


class EmbeddingGemma2MultimodalEmbedder(Gemma4MultimodalEmbedder):
    def __init__(
        self,
        multimodal_config: PreTrainedConfig,
        text_config: EmbeddingGemma2TextConfig,
    ):
        super().__init__(multimodal_config, text_config)


class EmbeddingGemma2AudioModelOutput(Gemma4AudioModelOutput):
    pass


@auto_docstring(
    custom_intro="""
    Base class for EmbeddingGemma2 outputs, with hidden states and attentions.
    """
)
@dataclass
class EmbeddingGemma2ModelOutput(BaseModelOutput):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    audio_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        audio_hidden_states of the model produced by the audio encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None

    audio_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    The EmbeddingGemma 2 model: a vision backbone, an audio backbone and a text backbone whose final hidden
    states are projected to `config.text_config.embedding_dim`. Intended to be wrapped by SentenceTransformers'
    mean pooling and normalization.
    """
)
class EmbeddingGemma2Model(Gemma4Model):
    config: EmbeddingGemma2Config
    _keys_to_ignore_on_load_unexpected = [
        r"(^|\.)vision_tower\.",
        r"(^|\.)embed_vision\.",
        r"(^|\.)audio_tower\.",
        r"(^|\.)embed_audio\.",
    ]

    def __init__(self, config: EmbeddingGemma2Config):
        super().__init__(config)
        # Drop inherited `vocab_size_per_layer_input` (no token-identity PLE in EmbeddingGemma 2)
        del self.vocab_size_per_layer_input
        self.post_init()

    @can_return_tuple
    @auto_docstring(custom_intro="Projects the last hidden state from the vision model into language model space.")
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            The patch positions as (x, y) coordinates in the image. Padding patches are indicated by (-1, -1).
        """
        if self.vision_tower is None:
            raise ValueError(
                "Image features were requested, but the model was initialized without a vision_config. "
                "Cannot process images without a vision tower and vision embedder."
            )
        return super().get_image_features(pixel_values=pixel_values, image_position_ids=image_position_ids, **kwargs)

    @can_return_tuple
    @auto_docstring(custom_intro="Projects the last hidden state from the vision encoder into language model space.")
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_position_ids: torch.LongTensor | None = None,
        num_frames_per_video: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(total_num_frames, max_patches, patch_pixels)`):
            The frames of every video in the batch, concatenated along the frame axis rather than stacked
            on a separate video axis, so that videos of different lengths can be batched together.
        video_position_ids (`torch.LongTensor` of shape `(total_num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        num_frames_per_video (`torch.LongTensor` of shape `(num_videos,)`):
            Number of frames belonging to each video, used to split the flat frame sequence back per video.
        """
        if self.vision_tower is None:
            raise ValueError(
                "Video features were requested, but the model was initialized without a vision_config. "
                "Cannot process video without a vision tower and vision embedder."
            )
        return super().get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_position_ids=video_position_ids,
            num_frames_per_video=num_frames_per_video,
            **kwargs,
        )

    def get_per_layer_input_embeddings(self):
        raise AttributeError("Deleted: EmbeddingGemma 2 has no `embed_tokens_per_layer` table.")

    def set_per_layer_input_embeddings(self, value):  # trf-ignore: TRF033
        raise AttributeError("Deleted: EmbeddingGemma 2 has no `embed_tokens_per_layer` table.")

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        num_frames_per_video: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EmbeddingGemma2ModelOutput:
        r"""
        input_features_mask (`torch.FloatTensor` of shape `(num_images, seq_length)`):
            The attention mask for the input audio.
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        video_position_ids (`torch.LongTensor` of shape `(total_num_frames, max_patches, 2)`, *optional*):
            2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        num_frames_per_video (`torch.LongTensor` of shape `(num_videos,)`, *optional*):
            Number of frames belonging to each video. Required whenever `pixel_values_videos` is passed,
            since the frames of all videos are concatenated along a single axis.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds)
        multimodal_mask = image_mask | video_mask | audio_mask

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        if inputs_embeds is None:
            llm_input_ids = torch.where(multimodal_mask, self.config.text_config.pad_token_id, input_ids)
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_position_ids, return_dict=True).pooler_output
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_image_tokens = image_mask.sum()
            image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features:"
                f" {image_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device), image_features.to(inputs_embeds.device)
            )

        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos, video_position_ids, num_frames_per_video, return_dict=True
            ).pooler_output
            video_features = torch.cat(video_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            # Confirm the number of soft tokens from the vision tower matches the number of slots in the embeddings.
            n_video_tokens = video_mask.sum()
            video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features:"
                f" {video_features.shape[0]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device), video_features.to(inputs_embeds.device)
            )

        # Merge text and audio
        if input_features is not None and input_features_mask is not None:
            audio_output = self.get_audio_features(input_features, input_features_mask, return_dict=True)
            audio_features = audio_output.pooler_output
            audio_mask_from_encoder = audio_output.attention_mask  # True = valid

            # Keep only real audio soft tokens, mirroring the vision encoder's padding stripping.
            audio_features = audio_features[audio_mask_from_encoder.to(audio_features.device)]

            n_audio_tokens = audio_mask.sum()
            audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            torch_compilable_check(
                inputs_embeds[audio_mask].numel() == audio_features.numel(),
                f"Audio features and audio tokens do not match, tokens: {n_audio_tokens}, features:"
                f" {audio_features.shape[0] * audio_features.shape[1]}",
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                audio_mask.to(inputs_embeds.device), audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )

        return EmbeddingGemma2ModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            audio_hidden_states=audio_features if input_features is not None else None,
        )


class EmbeddingGemma2VideoProcessorKwargs(Gemma4VideoProcessorKwargs):
    """
    patch_size (`int`, *optional*):
        Size of each image patch in pixels.
    max_soft_tokens (`int`, *optional*):
        Maximum number of soft (vision) tokens per video frame.
        Must be one of {70, 140, 280, 560, 1120}.
    pooling_kernel_size (`int`, *optional*):
        Spatial pooling kernel size applied after patchification.
    add_timestamps (`bool`, *optional*):
        Whether to prefix each frame in the video placeholder expansion with its `mm:ss` timestamp.
        Requires `VideoMetadata` with a valid `fps`, since timestamps cannot be inferred from
        already-decoded frames.
    max_frames (`int`, *optional*):
        The maximum number of frames to sample. If set, the sampled indices will
        be uniformly re-sampled to fit the budget.
    overflow_strategy (`str`, *optional*):
        The strategy used to cut the total number of sampled frames down to fit into the budget.
        Can be set only to "uniform" or "truncate". Applied after FPS-based sampling, and on its
        own when FPS-based sampling is off or not applicable.
    """

    add_timestamps: bool
    max_frames: int | None
    overflow_strategy: str | None


class EmbeddingGemma2VideoProcessor(Gemma4VideoProcessor):
    # unlike Gemma4 - by default sample 1 fps uniformly
    fps = 1
    max_frames = 32
    overflow_strategy = "uniform"
    add_timestamps = False
    num_frames = AttributeError()

    valid_kwargs = EmbeddingGemma2VideoProcessorKwargs

    def sample_frames(
        self,
        metadata: VideoMetadata,
        fps: int | float | None = None,
        max_frames: int | None = None,
        overflow_strategy: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        if kwargs.get("num_frames") is not None:
            raise ValueError(
                f"Sampling with `num_frames` is not supported for {self.__class__.__name__}. "
                "Please use `fps` and `max_frames` to control video sampling."
            )

        # 1) Sample to match the target `fps` if it is set, otherwise keep the whole video.
        # A decoded array carries no frame rate, and neither `fps` nor `duration` can be inferred
        # from one, so rate-based sampling is simply not applicable to that input. Skip it rather
        # than guess a source rate: a wrong guess silently discards frames.
        if fps is not None and (metadata.fps is None or metadata.duration is None):
            logger.warning_once(
                "Asked to sample uniformly with `fps`, but the video metadata has no `fps` or `duration`. "
                "Keeping every frame and applying only the `max_frames` budget. Pass a `VideoMetadata` "
                "object with a valid `fps` and `duration` to sample at a target frame rate."
            )
            fps = None

        if fps is None:
            indices = np.arange(metadata.total_num_frames, dtype=int)
        else:
            step = metadata.fps / fps  # native frames per sampled frame
            num_sampled = max(1, int(metadata.duration * fps))
            indices = np.array(
                [min(metadata.total_num_frames - 1, int(i * step)) for i in range(num_sampled)], dtype=int
            )

        # 2) Cap total number of frames to `max_frames` checking the input `overflow_strategy`
        if overflow_strategy is not None:
            if max_frames is None:
                raise ValueError(
                    f"You must pass `max_frames` when requesting an overflow_strategy={overflow_strategy}!"
                )

            # If video is too short, do no accidentally pad inputs when trying to re-sample
            if len(indices) <= max_frames:
                pass
            elif overflow_strategy == "truncate":
                indices = indices[:max_frames]
            elif overflow_strategy == "uniform":
                linspace_idx = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
                indices = np.array([indices[i] for i in linspace_idx], dtype=int)
            else:
                raise ValueError(
                    f"You passed `overflow_strategy={overflow_strategy}` but expected one of ['truncate', 'uniform']"
                )

        return indices


class EmbeddingGemma2ProcessorKwargs(Gemma4ProcessorKwargs):
    images_kwargs = AttributeError()

    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
        "images_kwargs": {
            "do_convert_rgb": True,
        },
        "audio_kwargs": {},
        "videos_kwargs": {"return_metadata": True},
    }


class EmbeddingGemma2Processor(Gemma4Processor):
    valid_processor_kwargs = EmbeddingGemma2ProcessorKwargs

    def model_input_names(self):
        raise AttributeError("Deleted: Gemma 4 appended `mm_token_type_ids`, which we no longer produce.")

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs,
    ):
        # When `text` is None, record per-sample counts for `audio` and `videos` before
        # `ProcessorMixin.prepare_inputs_layout` runs `make_list_of_audio` and `make_batched_videos`,
        # which flatten 2D nested per-sample lists into a 1D list of total items and discard sample
        # boundaries. `images` does not need this because `make_nested_list_of_images` preserves the
        # outer per-sample list structure.
        if not text:
            audio_per_sample = (
                [len(el) if isinstance(el, (list, tuple)) and not is_valid_audio(el) else 1 for el in audio]
                if isinstance(audio, (list, tuple)) and not is_valid_audio(audio)
                else None
            )
            videos_per_sample = (
                [
                    len(make_batched_videos(el)) if not (isinstance(el, (list, tuple)) and not el) else 0
                    for el in videos
                ]
                if isinstance(videos, (list, tuple)) and not is_valid_video(videos)
                else None
            )

        images, text, videos, audio = ProcessorMixin.prepare_inputs_layout(
            self, images=images, text=text, videos=videos, audio=audio, **kwargs
        )

        # Model requires nested struct
        if images is not None:
            images = make_nested_list_of_images(images)

        # Normalize videos so len(videos) gives the number of videos, not frames
        if videos is not None:
            videos = make_batched_videos(videos)

        if not text:
            modality_counts = []
            if images is not None:
                modality_counts.append((self.image_token, [len(image_list) for image_list in images]))
            if videos is not None:
                modality_counts.append((self.video_token, videos_per_sample or [1] * len(videos)))
            if audio is not None:
                modality_counts.append((self.audio_token, audio_per_sample or [1] * len(audio)))

            if modality_counts:
                batch_sizes = {len(counts) for _, counts in modality_counts}
                if len(batch_sizes) > 1:
                    raise ValueError(
                        f"Received inconsistently sized modality batches when `text` is None: "
                        f"{[len(counts) for _, counts in modality_counts]}."
                    )
                batch_size = batch_sizes.pop()
                text = [
                    " ".join(token for token, counts in modality_counts for _ in range(counts[sample_idx]))
                    for sample_idx in range(batch_size)
                ]

        return images, text, videos, audio

    def validate_inputs(
        self,
        images: ImageInput | list[ImageInput] | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        ProcessorMixin.validate_inputs(self, images=images, text=text, **kwargs)

        # Unlike Gemma 4, any single modality on its own is a valid embedding input.
        if text is None and images is None and videos is None and audio is None:
            raise ValueError("You must provide at least one of `text`, `images`, `videos`, or `audio`.")

        if audio is not None and (self.audio_token is None or self.boa_token is None or self.eoa_token is None):
            raise ValueError("Audio inputs were provided, but the tokenizer does not have an `audio_token` defined.")

        if text is not None:
            n_images_in_text = [sample.count(self.image_token) for sample in text]
            if images is not None:
                if len(images) != len(text):
                    raise ValueError(
                        f"Received inconsistently sized batches of images ({len(images)}) and text ({len(text)})."
                    )

                n_images_in_images = [len(sublist) for sublist in images]
                if n_images_in_text != n_images_in_images:
                    raise ValueError(
                        f"The total number of {self.image_token} tokens in the prompts should be the same as the number of images passed."
                        f" Found {n_images_in_text} {self.image_token} tokens and {n_images_in_images} images per sample."
                    )
            elif any(n_images_in_text):
                raise ValueError(
                    f"Found {sum(n_images_in_text)} {self.image_token} tokens in the text but no images were passed."
                )

            n_videos_in_text = [sample.count(self.video_token) for sample in text]
            if videos is not None:
                if sum(n_videos_in_text) != len(videos):
                    raise ValueError(
                        f"The total number of {self.video_token} tokens in the prompts should be the same as the number of videos passed."
                        f" Found {sum(n_videos_in_text)} {self.video_token} tokens and {len(videos)} videos."
                    )
            elif any(n_videos_in_text):
                raise ValueError(
                    f"Found {sum(n_videos_in_text)} {self.video_token} tokens in the text but no videos were passed."
                )

            if self.audio_token is not None:
                n_audio_in_text = [sample.count(self.audio_token) for sample in text]
                if audio is not None:
                    n_audio_passed = len(audio) if isinstance(audio, (list, tuple)) else 1
                    if sum(n_audio_in_text) != n_audio_passed:
                        raise ValueError(
                            f"The total number of {self.audio_token} tokens in the prompts should be the same as the number of audio inputs passed."
                            f" Found {sum(n_audio_in_text)} {self.audio_token} tokens and {n_audio_passed} audio inputs."
                        )
                elif any(n_audio_in_text):
                    raise ValueError(
                        f"Found {sum(n_audio_in_text)} {self.audio_token} tokens in the text but no audio inputs were passed."
                    )

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        num_soft_tokens = video_inputs["num_soft_tokens_per_video"][video_idx]
        add_timestamps = kwargs.get("add_timestamps", self.video_processor.add_timestamps)

        # Visual-only mode: one block per frame, no timestamps
        if not add_timestamps:
            # `pixel_values_videos` is a flat frame sequence, so its leading axis indexes frames, not videos
            num_frames = int(video_inputs["num_frames_per_video"][video_idx])
            frame_str = f"{self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}"
            return "".join([frame_str] * num_frames)

        metadata = video_inputs["video_metadata"][video_idx]

        if metadata.fps is None:
            raise ValueError(
                "Asked to build a prompt with frame timestamps, but no `fps` was provided in video metadata. "
                "The capture rate of already-decoded frames cannot be inferred. Please pass a `VideoMetadata` "
                "object with a valid `fps`, or set `add_timestamps=False`."
            )

        # mm:ss format for timestamps
        timestamp_str = [f"{int(seconds // 60):02d}:{int(seconds % 60):02d}" for seconds in metadata.timestamps]
        return " ".join(
            [f"{t} {self.boi_token}{self.video_token * num_soft_tokens}{self.eoi_token}" for t in timestamp_str]
        )


__all__ = [
    "EmbeddingGemma2Config",
    "EmbeddingGemma2Model",
    "EmbeddingGemma2PreTrainedModel",
    "EmbeddingGemma2Processor",
    "EmbeddingGemma2TextConfig",
    "EmbeddingGemma2TextModel",
    "EmbeddingGemma2VideoProcessor",
]
