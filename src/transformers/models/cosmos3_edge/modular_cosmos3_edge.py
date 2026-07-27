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

import math
import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...image_processing_backends import PilBackend, TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, ImageInput, PILImageResampling, SizeDict, get_image_size
from ...masking_utils import create_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, Unpack, VideosKwargs
from ...utils import (
    TensorType,
    add_start_docstrings,
    auto_docstring,
    can_return_tuple,
    is_torchvision_available,
    logging,
    torch_compilable_check,
)
from ...utils.generic import (
    accepts_precomputed_kwargs,
    get_max_seqlen,
    is_flash_attention_requested,
    maybe_autocast,
    merge_with_config_defaults,
)
from ...utils.output_capturing import capture_outputs
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos
from ...vision_utils import get_vision_attention_seqlens
from ..clip.modeling_clip import CLIPMLP
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from ..qwen2_vl.image_processing_qwen2_vl import smart_resize
from ..qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
    TransformersKwargs,
)
from ..qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionPatchMerger
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from ..siglip2.configuration_siglip2 import Siglip2VisionConfig
from ..siglip2.modeling_siglip2 import (
    Siglip2Attention,
    Siglip2Encoder,
    Siglip2EncoderLayer,
    Siglip2VisionEmbeddings,
)


logger = logging.get_logger(__name__)


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


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
        "layers.*.mlp.fc1": "colwise",
        "layers.*.mlp.fc2": "rowwise",
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

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": self.default_theta,
                "mrope_section": [24, 20, 20],
            }
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        rope_type = self.rope_parameters["rope_type"]
        if rope_type != "default":
            raise ValueError(f"Cosmos3 Edge requires `rope_type='default'`, got {rope_type!r}.")

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
class Cosmos3EdgeConfig(PreTrainedConfig):
    r"""
    projector_hidden_size (`int`, *optional*, defaults to 11520):
        Intermediate hidden size of the vision-to-language projector MLP.

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
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: Cosmos3EdgeTextConfig | dict | None = None
    vision_config: Cosmos3EdgeVisionConfig | dict | None = None
    projector_hidden_size: int = 11520
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

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if not isinstance(self.text_config, Cosmos3EdgeTextConfig):
            raise TypeError("`text_config` must be a `Cosmos3EdgeTextConfig` or a dictionary.")
        if not isinstance(self.vision_config, Cosmos3EdgeVisionConfig):
            raise TypeError("`vision_config` must be a `Cosmos3EdgeVisionConfig` or a dictionary.")


class Cosmos3EdgeTextRotaryEmbedding(LlamaRotaryEmbedding):
    """Interleaved M-RoPE used for Cosmos3 Edge text and visual tokens."""

    @staticmethod
    def compute_default_rope_parameters(
        config: Cosmos3EdgeTextConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Construct an axis-aware inverse-frequency matrix for interleaved temporal, height, and width RoPE."""
        base = config.rope_parameters["rope_theta"]
        dim = config.head_dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )

        indices = torch.arange(inv_freq.shape[0], device=device)
        mrope_section = config.rope_parameters["mrope_section"]
        height_mask = (indices % 3 == 1) & (indices < mrope_section[1] * 3)
        width_mask = (indices % 3 == 2) & (indices < mrope_section[2] * 3)
        temporal_mask = ~(height_mask | width_mask)
        inv_freq = torch.stack(
            (
                inv_freq * temporal_mask,
                inv_freq * height_mask,
                inv_freq * width_mask,
            )
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        position_ids = position_ids.permute(1, 2, 0).float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = position_ids.float() @ self.inv_freq.float().to(x.device)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Cosmos3EdgeTextAttention(LlamaAttention):
    """Dense GQA attention with Cosmos3 Edge M-RoPE."""


class Cosmos3EdgeTextMLP(CLIPMLP):
    """The dense two-projection ReLU-squared MLP used by Cosmos3 Edge."""

    def __init__(self, config: Cosmos3EdgeTextConfig):
        super().__init__(config)
        # CLIPMLP has the same fc1 -> activation -> fc2 structure, but Edge checkpoints omit MLP biases.
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)


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
    r"^(?:model\.language_model\.)?action_modality_embed$",
    r"^(?:model\.language_model\.)?action_proj_(?:in|out)\.(?:bias|fc)\.weight$",
    r"^(?:model\.language_model\.)?time_embedder\.",
    r"^(?:model\.language_model\.)?proj_(?:in|out)\.",
    r"^(?:model\.language_model\.)?norm_moe_gen\.",
    r"^(?:model\.language_model\.)?layers\.\d+\.input_layernorm_moe_gen\.",
    r"^(?:model\.language_model\.)?layers\.\d+\.post_attention_layernorm_moe_gen\.",
    r"^(?:model\.language_model\.)?layers\.\d+\.mlp_moe_gen\.",
    r"^(?:model\.language_model\.)?layers\.\d+\.self_attn\.(?:add_[qkv]_proj|to_add_out|norm_added_[qk]|k_norm_und_for_gen)\.",
]


class Cosmos3EdgeVisionEmbeddings(Siglip2VisionEmbeddings):
    """SigLIP2 patch and learned-position embeddings for packed Edge vision inputs."""

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        grid_thw: torch.LongTensor,
        spatial_merge_size: int,
    ) -> torch.Tensor:
        """Resize a learned reference grid independently for every packed image or video frame.

        Args:
            positional_embeddings (`torch.Tensor`):
                Learned positional embeddings of shape `(height, width, hidden_size)`.
            grid_thw (`torch.LongTensor`):
                Temporal, height, and width patch-grid dimensions for every packed image or video.
            spatial_merge_size (`int`):
                Number of patches merged along each spatial axis by the projector.

        Returns:
            `torch.Tensor`: Positional embeddings packed in the same block-major order as the input patches.
        """
        # The checkpoint uses a learned square reference grid, interpolated independently for every packed frame.
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
        source_dtype = positional_embeddings.dtype
        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU.
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.float()

        position_chunks = []
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
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            # Preserve the processor's block-major 2x2 patch order before the projector groups adjacent patches.
            resized_embeddings = resized_embeddings.transpose(1, 2).reshape(height * width, -1)
            position_chunks.append(resized_embeddings.repeat(temporal, 1))

        return torch.cat(position_chunks, dim=0)

    def forward(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor) -> torch.Tensor:
        # Apply patch embeddings to already patchified pixel values.
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get image-specific positional embeddings in the packed block-major order expected by the checkpoint.
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        position_embeddings = self.resize_positional_embeddings(
            positional_embeddings, grid_thw, self.config.spatial_merge_size
        )
        torch_compilable_check(
            patch_embeds.shape[0] == position_embeddings.shape[0],
            "The packed visual patch count does not match `grid_thw`.",
        )
        return patch_embeds + position_embeddings


class Cosmos3EdgeVisionAttention(Siglip2Attention):
    """Packed non-causal SigLIP2 attention with one sequence per image or video frame."""

    def __init__(self, config: Cosmos3EdgeVisionConfig):
        super().__init__(config)
        self.num_key_value_groups = 1

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int | None = None, **kwargs
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).reshape(sequence_length, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(sequence_length, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(sequence_length, -1, self.head_dim)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
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
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
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
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Cosmos3EdgeEncoder(Siglip2Encoder):
    def __init__(self, config: Cosmos3EdgeVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Cosmos3EdgeVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds: torch.Tensor, grid_thw: torch.LongTensor, **kwargs) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height, and width patch-grid dimensions for every packed image or video.
        """
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                **kwargs,
            )

        return hidden_states


class Cosmos3EdgePreTrainedModel(Qwen2VLPreTrainedModel):
    config_class = Cosmos3EdgeConfig
    input_modalities = ("image", "video", "text")
    # Packed, variable-length visual inputs use Python-level per-grid reshaping and cannot be compiled fullgraph.
    _can_compile_fullgraph = False
    _no_split_modules = ["Cosmos3EdgeTextDecoderLayer", "Cosmos3EdgeVisionEncoderLayer"]
    _can_record_outputs = {
        "hidden_states": [Cosmos3EdgeTextDecoderLayer, Cosmos3EdgeVisionEncoderLayer],
        "attentions": Cosmos3EdgeTextAttention,
    }
    _keys_to_ignore_on_load_unexpected = _COSMOS3_EDGE_DROPPED_GENERATOR_KEYS


class Cosmos3EdgeTextModel(LlamaModel, Cosmos3EdgePreTrainedModel):
    config_class = Cosmos3EdgeTextConfig
    input_modalities = ("text",)

    def __init__(self, config: Cosmos3EdgeTextConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Cosmos3EdgeTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Cosmos3EdgeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Cosmos3EdgeTextRotaryEmbedding(config)

    @merge_with_config_defaults
    @capture_outputs
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output.
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1:
            position_ids = position_ids.expand(3, inputs_embeds.shape[0], -1)

        text_position_ids = position_ids[0]
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Cosmos3EdgeVisionModel(Cosmos3EdgePreTrainedModel):
    """Packed variable-resolution SigLIP2 vision tower used by Cosmos3 Edge."""

    config_class = Cosmos3EdgeVisionConfig
    main_input_name = "pixel_values"
    input_modalities = ("image", "video")

    def __init__(self, config: Cosmos3EdgeVisionConfig):
        super().__init__(config)
        self.embeddings = Cosmos3EdgeVisionEmbeddings(config)
        self.encoder = Cosmos3EdgeEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor, **kwargs
    ) -> BaseModelOutputWithPooling:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height, and width patch-grid dimensions for each packed image or video.
        """
        hidden_states = self.embeddings(pixel_values, grid_thw)
        hidden_states = self.encoder(hidden_states, grid_thw=grid_thw, **kwargs)
        last_hidden_state = self.post_layernorm(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)


class Cosmos3EdgePatchMerger(Qwen3_5MoeVisionPatchMerger):
    def __init__(self, config: Cosmos3EdgeConfig):
        nn.Module.__init__(self)
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.input_hidden_size = config.vision_config.hidden_size
        self.hidden_size = self.input_hidden_size * self.spatial_merge_size**2
        self.use_postshuffle_norm = False
        self.norm = nn.LayerNorm(self.input_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, config.projector_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(config.projector_hidden_size, config.text_config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.spatial_merge_size**2, self.input_hidden_size)
        return super().forward(x)


class Cosmos3EdgeModel(Qwen2VLModel, Cosmos3EdgePreTrainedModel):
    config_class = Cosmos3EdgeConfig
    accepts_loss_kwargs = False

    def __init__(self, config: Cosmos3EdgeConfig):
        # Qwen2VLModel's constructor instantiates its Qwen-specific submodels. Cosmos3 Edge uses the same
        # multimodal API, but its checkpoint has distinct packed vision and Llama-derived text components.
        Cosmos3EdgePreTrainedModel.__init__(self, config)
        self.visual = Cosmos3EdgeVisionModel._from_config(config.vision_config)
        self.projector = Cosmos3EdgePatchMerger(config)
        self.language_model = Cosmos3EdgeTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, num_channels * patch_size * patch_size)`):
            Packed image patches.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height, and width dimensions of every packed image patch grid.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_embeds = self.projector(vision_outputs.last_hidden_state)
        split_sizes = (image_grid_thw.prod(-1) // self.projector.spatial_merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(image_embeds, split_sizes)

        return vision_outputs

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, num_channels * patch_size * patch_size)`):
            Packed video-frame patches.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height, and width dimensions of every packed video patch grids.
        """
        # Video frames use the same vision tower and projector path as images.
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **super_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Edge's processor emits one timestamped visual span per frame, so split each video's temporal grid before
        # applying Qwen2-VL's common multimodal position-index routine.
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
            video_grid_thw[:, 0] = 1

        return super().get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            **super_kwargs,
        )

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
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height, and width of the feature grid for each image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height, and width of the feature grid for each video.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, **kwargs).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, **kwargs).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Cosmos3EdgeForConditionalGeneration(Qwen2VLForConditionalGeneration, Cosmos3EdgePreTrainedModel):
    config_class = Cosmos3EdgeConfig
    _tied_weights_keys = {}
    accepts_loss_kwargs = False

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Qwen2-VL exposes four axes (text plus three visual axes). Edge's interleaved M-RoPE consumes the three
        # visual axes directly, so start from the common 2D text positions rather than Qwen2-VL's four-axis helper.
        text_positions = GenerationMixin._prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs)

        # Early exit in case we are continuing generation from past kv.
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            position_ids = text_positions[None, ...] + self.model.rope_deltas
            return position_ids

        # Otherwise compute 3D position ids for vision tokens.
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        has_multimodal_inputs = (
            model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None
        )
        if is_input_ids and has_multimodal_inputs and model_kwargs.get("mm_token_type_ids") is None:
            raise ValueError(
                "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
                "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
                "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )

        if is_input_ids and model_kwargs.get("mm_token_type_ids") is not None and has_multimodal_inputs:
            model_kwargs = {key: value for key, value in model_kwargs.items() if key != "input_ids"}
            position_ids, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
            self.model.rope_deltas = rope_deltas
        else:
            position_ids = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        return position_ids

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Video placeholders are emitted once per frame, while `video_grid_thw` has one row per source video.
        # `_get_image_nums_and_video_nums` cannot do this conversion because it does not receive `video_grid_thw`,
        # so convert frame-span counts back to source-video counts here before repeating packed tensors for beams.
        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw")
            video_grid_thw = model_kwargs.get("video_grid_thw")
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds")
            )

            if video_grid_thw is not None:
                cumulative_frame_counts = torch.cumsum(video_grid_thw[:, 0], dim=0)
                cumulative_token_video_counts = torch.cumsum(video_nums, dim=0)
                video_boundary_indices = torch.searchsorted(cumulative_frame_counts, cumulative_token_video_counts)
                video_nums = torch.diff(torch.cat([-video_boundary_indices.new_ones(1), video_boundary_indices]))

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                return torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)

            for key in dict_to_expand:
                if key == "pixel_values":
                    samples = torch.split(image_grid_thw, list(image_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(image_nums), repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key, value in dict_to_expand.items():
                if key == "position_ids" and value.ndim == 3:
                    dict_to_expand[key] = value.repeat_interleave(expand_size, dim=1)
                elif value is not None and isinstance(value, torch.Tensor) and key not in visual_keys:
                    dict_to_expand[key] = value.repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), and the loss is only computed for tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height, and width of the feature grid for each image.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height, and width of the feature grid for each video.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

        >>> model = Cosmos3EdgeForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        >>> inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=1024)
        >>> generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        >>> output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> print(output_text)
        ```
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Processor implementations live in the modular source so the fast/PIL/video/generated modules stay synchronized.


class Cosmos3EdgeImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `16`):
        Spatial patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to `2`):
        Number of adjacent patches merged along each spatial axis by the projector.
    """

    patch_size: int
    merge_size: int


@auto_docstring
class Cosmos3EdgeImageProcessor(TorchvisionBackend):
    """Dynamically resize images and return packed, unpadded SigLIP2 patches."""

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 256 * 256, "longest_edge": 4096 * 4096}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    patch_size = 16
    merge_size = 2
    valid_kwargs = Cosmos3EdgeImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Cosmos3EdgeImageProcessorKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (self.size.shortest_edge is None or self.size.longest_edge is None):
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")

    def _standardize_kwargs(self, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if size.shortest_edge is None or size.longest_edge is None:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        return kwargs

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Cosmos3EdgeImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        pixel_values = []
        image_grids = []

        for image in images:
            height, width = image.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                image = self.resize(
                    image=image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            else:
                resized_height, resized_width = height, width
                patch_group_size = patch_size * merge_size
                if resized_height % patch_group_size or resized_width % patch_group_size:
                    raise ValueError(
                        "Images must have dimensions divisible by `patch_size * merge_size` when `do_resize=False`, "
                        f"got height={resized_height}, width={resized_width}, patch_size={patch_size}, "
                        f"merge_size={merge_size}."
                    )

            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            channels = image.shape[0]
            grid_height, grid_width = resized_height // patch_size, resized_width // patch_size
            patches = image.reshape(
                channels,
                grid_height // merge_size,
                merge_size,
                patch_size,
                grid_width // merge_size,
                merge_size,
                patch_size,
            )
            # The projector expects block-major patches with HWC values within each flattened patch:
            # (group_h, group_w, merge_h, merge_w, patch_h, patch_w, channel).
            patches = patches.permute(1, 4, 2, 5, 3, 6, 0).reshape(grid_height * grid_width, -1)

            pixel_values.append(patches)
            image_grids.append((1, grid_height, grid_width))

        return BatchFeature(
            data={
                "pixel_values": torch.cat(pixel_values, dim=0),
                "image_grid_thw": torch.tensor(image_grids, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """Return the number of pre-projector vision patches for an image size."""
        images_kwargs = images_kwargs or {}
        size = images_kwargs.get("size", self.size)
        if isinstance(size, SizeDict):
            min_pixels, max_pixels = size.shortest_edge, size.longest_edge
        else:
            min_pixels, max_pixels = size["shortest_edge"], size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)


class Cosmos3EdgeImageProcessorPil(PilBackend):
    r"""
    Dynamically resize images and return packed, unpadded SigLIP2 patches.

    patch_size (`int`, *optional*, defaults to `16`):
        Spatial patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to `2`):
        Number of adjacent patches merged along each spatial axis by the projector.
    """

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 256 * 256, "longest_edge": 4096 * 4096}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    patch_size = 16
    merge_size = 2
    valid_kwargs = Cosmos3EdgeImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Cosmos3EdgeImageProcessorKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (self.size.shortest_edge is None or self.size.longest_edge is None):
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")

    def _standardize_kwargs(self, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if size.shortest_edge is None or size.longest_edge is None:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        return kwargs

    def preprocess(self, images: ImageInput, **kwargs: Unpack[Cosmos3EdgeImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        pixel_values = []
        image_grids = []

        for image in images:
            height, width = image.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                image = self.resize(
                    image=image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            else:
                resized_height, resized_width = height, width
                patch_group_size = patch_size * merge_size
                if resized_height % patch_group_size or resized_width % patch_group_size:
                    raise ValueError(
                        "Images must have dimensions divisible by `patch_size * merge_size` when `do_resize=False`, "
                        f"got height={resized_height}, width={resized_width}, patch_size={patch_size}, "
                        f"merge_size={merge_size}."
                    )

            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            channels = image.shape[0]
            grid_height, grid_width = resized_height // patch_size, resized_width // patch_size
            patches = image.reshape(
                channels,
                grid_height // merge_size,
                merge_size,
                patch_size,
                grid_width // merge_size,
                merge_size,
                patch_size,
            )
            # The projector expects block-major patches with HWC values within each flattened patch:
            # (group_h, group_w, merge_h, merge_w, patch_h, patch_w, channel).
            patches = patches.transpose(1, 4, 2, 5, 3, 6, 0).reshape(grid_height * grid_width, -1)

            pixel_values.append(patches)
            image_grids.append((1, grid_height, grid_width))

        return BatchFeature(
            data={
                "pixel_values": np.concatenate(pixel_values, axis=0),
                "image_grid_thw": np.asarray(image_grids, dtype=np.int64),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """Return the number of pre-projector vision patches for an image size."""
        images_kwargs = images_kwargs or {}
        size = images_kwargs.get("size", self.size)
        if isinstance(size, SizeDict):
            min_pixels, max_pixels = size.shortest_edge, size.longest_edge
        else:
            min_pixels, max_pixels = size["shortest_edge"], size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)


def smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 1,
    factor: int = 32,
    min_pixels: int = 64 * 64,
    max_pixels: int = 24 * 1024 * 1024,
) -> tuple[int, int]:
    """Resize video frames while keeping the packed patch grid valid for Cosmos3 Edge.

    This follows Qwen3-VL's video resize function. Cosmos3 Edge changes the defaults because it processes every
    sampled frame independently (`temporal_factor=1`) and uses the checkpoint's pixel budget.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Cosmos3EdgeVideoProcessorInitKwargs(VideosKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `16`):
        Spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to `1`):
        Temporal patch size. Cosmos3 Edge processes every sampled frame independently, so only `1` is supported.
    merge_size (`int`, *optional*, defaults to `2`):
        Number of adjacent patches merged along each spatial axis by the projector.
    min_frames (`int`, *optional*, defaults to `4`):
        Minimum number of frames sampled from a video.
    max_frames (`int`, *optional*, defaults to `768`):
        Maximum number of frames sampled from a video.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int


@add_start_docstrings(
    "Constructs a video processor that dynamically resizes and packs Cosmos3 Edge video frames.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 16):
            Spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            Temporal patch size of the vision encoder. Cosmos3 Edge processes each sampled frame independently.
        merge_size (`int`, *optional*, defaults to 2):
            Spatial merge size applied by the vision projector.
    """,
)
class Cosmos3EdgeVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 64 * 64, "longest_edge": 24 * 1024 * 1024}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    fps = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = True
    valid_kwargs = Cosmos3EdgeVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Cosmos3EdgeVideoProcessorInitKwargs]):
        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else dict(size)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        if kwargs.get("temporal_patch_size", self.temporal_patch_size) != 1:
            raise ValueError("Cosmos3 Edge only supports `temporal_patch_size=1`.")
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(self, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if size.shortest_edge is None or size.longest_edge is None:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        if kwargs.get("temporal_patch_size", self.temporal_patch_size) != 1:
            raise ValueError("Cosmos3 Edge only supports `temporal_patch_size=1`.")
        return kwargs

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Uniformly sample frames with the checkpoint's default two frames per second policy."""
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        total_num_frames = metadata.total_num_frames
        fps = self.fps if fps is None else fps
        if num_frames is None and fps is not None:
            if metadata.fps is None:
                metadata.fps = 24
                logger.warning_once(
                    "Cosmos3 Edge samples video frames using fps, but input video metadata did not provide an fps. "
                    "Defaulting to fps=24. Pass `video_metadata` for accurate timestamps."
                )
            num_frames = int(total_num_frames / metadata.fps * fps)
            num_frames = min(max(num_frames, self.min_frames), self.max_frames, total_num_frames)

        if num_frames is None:
            num_frames = min(max(total_num_frames, self.min_frames), self.max_frames)

        return np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            batch_size, num_frames, channels, height, width = stacked_videos.shape
            if do_resize:
                resized_height, resized_width = smart_resize_video(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.reshape(batch_size * num_frames, channels, height, width)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.reshape(
                    batch_size, num_frames, channels, resized_height, resized_width
                )
            resized_videos_grouped[shape] = stacked_videos

        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}

        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            patch_group_size = patch_size * merge_size
            if resized_height % patch_group_size or resized_width % patch_group_size:
                raise ValueError(
                    "Video frames must have dimensions divisible by `patch_size * merge_size`, got "
                    f"height={resized_height}, width={resized_width}, patch_size={patch_size}, "
                    f"merge_size={merge_size}."
                )
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            batch_size, grid_t, channels = stacked_videos.shape[:3]
            grid_height, grid_width = resized_height // patch_size, resized_width // patch_size

            patches = stacked_videos.reshape(
                batch_size,
                grid_t,
                channels,
                grid_height // merge_size,
                merge_size,
                patch_size,
                grid_width // merge_size,
                merge_size,
                patch_size,
            )
            # Preserve time-major, block-major patches with HWC values within each flattened patch:
            # (batch, time, group_h, group_w, merge_h, merge_w, patch_h, patch_w, channel).
            patches = patches.permute(0, 1, 3, 6, 4, 7, 5, 8, 2)
            processed_videos_grouped[shape] = patches.reshape(
                batch_size, grid_t * grid_height * grid_width, channels * patch_size * patch_size
            )
            processed_grids[shape] = [[grid_t, grid_height, grid_width]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        return BatchFeature(
            data={
                "pixel_values_videos": torch.cat(processed_videos, dim=0),
                "video_grid_thw": torch.tensor(processed_grids, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs: dict | None = None
    ) -> int:
        """Return the number of pre-projector vision patches for a video size."""
        videos_kwargs = videos_kwargs or {}
        size = videos_kwargs.get("size", self.size)
        if isinstance(size, SizeDict):
            min_pixels, max_pixels = size.shortest_edge, size.longest_edge
        else:
            min_pixels, max_pixels = size["shortest_edge"], size["longest_edge"]
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        resized_height, resized_width = smart_resize_video(
            num_frames=num_frames,
            height=height,
            width=width,
            temporal_factor=temporal_patch_size,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_t = math.ceil(num_frames / temporal_patch_size)
        return grid_t * (resized_height // patch_size) * (resized_width // patch_size)


@auto_docstring
class Cosmos3EdgeProcessor(Qwen3VLProcessor):
    """Construct a Cosmos3 Edge multimodal prompt from image, video, and text inputs."""

    valid_processor_kwargs = ProcessingKwargs

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        """Expand an image placeholder to one text token per projected 2×2 patch group."""
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = int(image_inputs["image_grid_thw"][image_idx].prod()) // merge_length
        return self.image_token * num_image_tokens

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        """Expand a video into timestamped, frame-level vision segments."""
        grid_thw = video_inputs["video_grid_thw"][video_idx]
        merge_length = self.video_processor.merge_size**2
        num_tokens_per_frame = int(grid_thw[1:].prod()) // merge_length
        metadata = video_inputs["video_metadata"][video_idx]

        if metadata.fps is None:
            logger.warning_once(
                "Cosmos3 Edge requires frame timestamps to construct prompts, but the input video's fps could not "
                "be inferred. Defaulting to fps=24. Pass `video_metadata` for accurate timestamps."
            )
        metadata.fps = 24 if metadata.fps is None else metadata.fps

        timestamps = self._calculate_timestamps(
            metadata.frames_indices,
            metadata.fps,
            merge_size=self.video_processor.temporal_patch_size,
        )

        return "".join(
            f"<{timestamp:.1f} seconds>{self.vision_start_token}"
            f"{self.video_token * num_tokens_per_frame}{self.vision_end_token}"
            for timestamp in timestamps
        )

    @staticmethod
    def _calculate_timestamps(
        indices: list[int] | np.ndarray,
        video_fps: float,
        merge_size: int = 1,
    ) -> list[float]:
        """Compute one timestamp per temporal patch, using the center frame's time."""
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        return [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]

    def get_text_with_replacements(
        self,
        text: list[str],
        images_replacements: list[str] = [],
        videos_replacements: list[str] = [],
        audio_replacements: list[str] = [],
    ) -> tuple[list[str], list[dict]]:
        """Replace placeholders while treating the template's full video wrapper as one unit.

        The Edge chat template emits ``<|vision_start|><|video_pad|><|vision_end|>``. Each video must become a
        separate timestamped vision segment for every frame, so replacing only ``<|video_pad|>`` would leave an
        invalid outer vision wrapper around all frames.
        """
        token_groups = []
        if images_replacements:
            token_groups.append(f"(?P<image>{re.escape(self.image_token)})")
        if videos_replacements:
            video_wrapper = re.escape(self.vision_start_token + self.video_token + self.vision_end_token)
            token_groups.append(f"(?P<video>{video_wrapper}|{re.escape(self.video_token)})")
        if audio_replacements and getattr(self, "audio_token", None) is not None:
            token_groups.append(f"(?P<audio>{re.escape(self.audio_token)})")
        if not token_groups:
            return text, []

        replacements = {
            "image": iter(images_replacements),
            "video": iter(videos_replacements),
            "audio": iter(audio_replacements),
        }
        pattern = "|".join(token_groups)
        batch_replacement_offsets = []

        for batch_index, sample in enumerate(text):
            last_end = 0
            offset = 0
            expanded_sample = []
            replacement_offsets = []
            for match in re.finditer(pattern, sample):
                start, end = match.span()
                expanded_sample.append(sample[last_end:start])
                modality = match.lastgroup
                replacement = next(replacements[modality])
                start_with_offset = start + offset
                replacement_offsets.append(
                    {
                        "type": modality,
                        "span": (start, end),
                        "new_span": (start_with_offset, start_with_offset + len(replacement)),
                        "text": match.group(),
                        "replacement": replacement,
                    }
                )
                expanded_sample.append(replacement)
                offset += len(replacement) - (end - start)
                last_end = end

            expanded_sample.append(sample[last_end:])
            text[batch_index] = "".join(expanded_sample)
            batch_replacement_offsets.append(replacement_offsets)
        return text, batch_replacement_offsets

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """Compute placeholder counts for serving frameworks without materializing pixels."""
        vision_data = {}
        images_kwargs = dict(kwargs.get("images_kwargs", kwargs))
        videos_kwargs = dict(kwargs.get("videos_kwargs", kwargs))

        if image_sizes is not None:
            merge_size = images_kwargs.get("merge_size", self.image_processor.merge_size)
            num_image_patches = [
                self.image_processor.get_number_of_image_patches(height, width, images_kwargs)
                for height, width in image_sizes
            ]
            vision_data["num_image_patches"] = num_image_patches
            vision_data["num_image_tokens"] = [num_patches // merge_size**2 for num_patches in num_image_patches]

        if video_sizes is not None:
            merge_size = videos_kwargs.get("merge_size", self.video_processor.merge_size)
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(num_frames, height, width, videos_kwargs)
                for num_frames, height, width in video_sizes
            ]
            vision_data["num_video_tokens"] = [num_patches // merge_size**2 for num_patches in num_video_patches]

        return MultiModalData(**vision_data)

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]


__all__ = [
    "Cosmos3EdgeConfig",
    "Cosmos3EdgeTextConfig",
    "Cosmos3EdgeVisionConfig",
    "Cosmos3EdgeModel",
    "Cosmos3EdgeTextModel",
    "Cosmos3EdgeVisionModel",
    "Cosmos3EdgeForConditionalGeneration",
    "Cosmos3EdgePreTrainedModel",
    "Cosmos3EdgeImageProcessor",
    "Cosmos3EdgeImageProcessorPil",
    "Cosmos3EdgeVideoProcessor",
    "Cosmos3EdgeProcessor",
]
