# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_int
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..codegen.modeling_codegen import create_sinusoidal_positions
from ..gemma2.modeling_gemma2 import eager_attention_forward
from ..qwen3_next.modeling_qwen3_next import l2norm
from ..siglip.configuration_siglip import SiglipConfig, SiglipTextConfig
from ..t5.tokenization_t5 import T5Tokenizer
from ..vivit.configuration_vivit import VivitConfig
from ..vivit.modeling_vivit import (
    VivitAttention,
    VivitEmbeddings,
    VivitLayer,
    VivitTubeletEmbeddings,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/videoprism-base-f16r288")
@strict
class VideoPrismVisionConfig(VivitConfig):
    r"""
    num_frames (`int`, *optional*, defaults to 16):
        The number of frames in the input video.
    tubelet_size (`List[int]`, *optional*, defaults to `[1, 18, 18]`):
        The size of the tubelet patch.
    num_spatial_layers (`int`, *optional*, defaults to 12):
        Number of spatial transformer blocks.
    num_temporal_layers (`int`, *optional*, defaults to 4):
        Number of temporal transformer blocks.
    attn_logit_softcapping (`float`, *optional*, defaults to 50.0):
        Softcapping constant for attention logits.
    num_auxiliary_layers (`int`, *optional*, defaults to 2):
        Number of auxiliary layers. This is used in the VideoPrismVideoModel that is a part of VideoPrismClipModel.
    apply_l2norm (`bool`, *optional*, defaults to `True`):
        Whether to apply L2 normalization to the output. This is used in the VideoPrismVideoModel that is a part of VideoPrismClipModel.
    """

    model_type = "videoprism_vision_model"
    base_config_key = "vision_config"
    image_size: int | list[int] | tuple[int, int] = 288
    num_frames: int = 16
    tubelet_size: list[int] | tuple[int, ...] = (1, 18, 18)
    num_spatial_layers: int = 12
    num_temporal_layers: int = 4
    hidden_act: str = "gelu_python"
    attn_logit_softcapping: float = 50.0
    num_auxiliary_layers: int = 2
    apply_l2norm: bool = True
    num_hidden_layers = AttributeError()
    pooler_act = AttributeError()
    pooler_output_size = AttributeError()

    def __post_init__(self, **kwargs):
        raise AttributeError("Not used here")


@auto_docstring(checkpoint="google/videoprism-lvt-base-f16r288")
@strict
class VideoPrismTextConfig(SiglipTextConfig):
    r"""
    apply_l2norm (`bool`, *optional*, defaults to `True`):
        Whether to apply L2 normalization to the output of VideoPrismTextEncoder.
    attn_logit_softcapping (`float`, *optional*, defaults to 50.0):
        Softcapping constant for attention logits.
    """

    hidden_act: str = "relu"
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    attention_probs_dropout_prob: float | int = 0.0
    apply_l2norm: bool = True
    qkv_bias: bool = True
    hidden_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    attn_logit_softcapping: float = 50.0
    attention_dropout = AttributeError()
    projection_size = AttributeError()

    def __post_init__(self, **kwargs):
        raise AttributeError("Not used here")


@auto_docstring(
    checkpoint="google/videoprism-lvt-base-f16r288",
    custom_intro="""
    This is the configuration class to store the configuration of a [`VideoPrismClipModel`]. It is used to instantiate a
    VideoPrismClipModel according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the VideoPrism
    [google/videoprism-lvt-base-f16r288](https://huggingface.co/google/videoprism-lvt-base-f16r288) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """,
)
@strict
class VideoPrismConfig(SiglipConfig):
    r"""
    Example:

    ```python
    >>> from transformers import VideoPrismClipModel, VideoPrismConfig

    >>> # Initializing a VideoPrismConfig with default values
    >>> configuration = VideoPrismConfig()

    >>> # Initializing a VideoPrismClipModel with the configuration
    >>> model = VideoPrismClipModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    initializer_factor = AttributeError()


class VideoPrismTokenizer(T5Tokenizer):
    r"""
    Constructs a VideoPrism tokenizer, which is essentially a T5 tokenizer without its postprocessor
    (appending an EOS token at the end of the sequence).

    This tokenizer inherits from [`T5Tokenizer`] which contains most of the main methods. Users should refer to this
    superclass for more information regarding those methods.
    """

    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        _spm_precompiled_charsmap=None,
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs,
    ):
        super().__init__(
            vocab=vocab,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            _spm_precompiled_charsmap=_spm_precompiled_charsmap,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        # VideoPrism does not append an EOS token by default
        self._tokenizer.post_processor = None


class VideoPrismProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "truncation": True,
            "max_length": 64,
        },
        "video_kwargs": {
            "size": {"height": 288, "width": 288},
            "do_normalize": False,
            "do_sample_frames": True,
        },
    }


@auto_docstring
class VideoPrismProcessor(ProcessorMixin):
    valid_processor_kwargs = VideoPrismProcessorKwargs

    def __init__(self, video_processor=None, tokenizer=None):
        super().__init__(video_processor, tokenizer)


@auto_docstring(custom_intro="""Base class for model outputs that include spatial and temporal states.""")
@dataclass
class BaseModelOutputWithSpatialAndTemporalStates(ModelOutput):
    r"""
    temporal_hidden_state (`torch.FloatTensor`, *optional*):
        The last hidden state of the temporal encoder, typically of shape
        `(batch_size * num_patches, num_frames, hidden_size)`.
    spatial_hidden_state (`torch.FloatTensor`, *optional*):
        The last hidden state of the spatial encoder, typically of shape
        `(batch_size * num_frames, num_patches, hidden_size)`.
    """

    last_hidden_state: torch.FloatTensor
    temporal_hidden_state: torch.FloatTensor | None = None
    spatial_hidden_state: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""Base class for VideoPrismClipModel outputs.""",
)
@dataclass
class VideoPrismClipOutput(ModelOutput):
    r"""
    logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
        The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text
        similarity scores.
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
        The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video
        similarity scores.
    video_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
        The video embeddings obtained by applying the projection layer to the pooled output of [`VideoPrismVideoModel`].
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`VideoPrismTextModel`].
    video_model_output (`BaseModelOutputWithPooling`):
        The output of [`VideoPrismVideoModel`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`VideoPrismTextModel`].
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for video-text similarity.
    """

    logits_per_video: torch.FloatTensor | None = None
    logits_per_text: torch.FloatTensor | None = None
    video_embeds: torch.FloatTensor | None = None
    text_embeds: torch.FloatTensor | None = None
    video_model_output: BaseModelOutputWithPooling = None
    text_model_output: BaseModelOutputWithPooling = None
    loss: torch.FloatTensor | None = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "video_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class VideoPrismTubeletEmbeddings(VivitTubeletEmbeddings):
    """
    VideoPrism Tubelet Embeddings.

    The authors of Videoprism use the Factorized Encoder architecture, i.e. "Model 2", introduced in the VIVIT paper (https://huggingface.co/papers/2103.15691).
    This differs from Vivit by using a convolution of `tubelet_size=(1, 18, 18)`, which is essntially a 2d convolution in the spatial dimension.
    The temporal dimension is also merged with the `batch_size` in order to make sure the image embeddings have no temporal component, unlike Vivit.
    """

    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        del self.num_patches
        self.pos_emb_shape = [self.image_size[0] // tubelet_size[1], self.image_size[1] // tubelet_size[2]]  # noqa: F821
        self.num_patches = self.pos_emb_shape[0] * self.pos_emb_shape[1]

    def forward(self, pixel_values_videos: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = pixel_values_videos.shape
        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}). Set interpolate_pos_encoding=True to automatically resize the model position embeddings."
            )
        # permute to (batch_size, num_channels, num_frames, height, width)
        pixel_values_videos = pixel_values_videos.transpose(1, 2)
        hidden_states = self.projection(pixel_values_videos)
        # flatten the spatial part and permute to (batch_size, num_frames, num_patches, hidden_dim)
        hidden_states = hidden_states.flatten(3).permute(0, 2, 3, 1)
        # combine batch and time dimension
        batch_size, num_frames, num_patches, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size * num_frames, num_patches, hidden_size)

        return hidden_states


class VideoPrismSpatialEmbeddings(VivitEmbeddings):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        del self.cls_token
        del self.image_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))  # noqa: F821

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        # The only difference from Vivit is that we don't have to account for a cls token
        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        dim = embeddings.shape[-1]

        num_row_patches = height // self.patch_size[0]
        num_col_patches = width // self.patch_size[1]

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = self.position_embeddings.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # This differs from Vivit by using bilinear mode instead of bicubic.
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(num_row_patches, num_col_patches),
            mode="bilinear",
            antialias=True,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        interpolate_pos_encoding: bool | None = False,
    ) -> torch.Tensor:
        batch, frames, channel, height, width = pixel_values_videos.shape
        embeddings = self.patch_embeddings(pixel_values_videos, interpolate_pos_encoding)
        # no cls token is added unlike Vivit
        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class VideoPrismTemporalEmbeddings(VivitEmbeddings):
    """
    VideoPrism Temporal Embeddings.

    Receives embeddings from spatial encoder, reshapes the hidden state to
    (batch_size * num_patches, num_frames, hidden_size) and adds positional embeddings.
    This module is only used in the VideoPrism architecture and not available in Vivit.
    """

    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        del self.cls_token
        del self.patch_embeddings
        del self.patch_size
        del num_patches  # noqa: F821
        del self.image_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.num_frames, config.hidden_size))

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        target_emb_length = embeddings.shape[1]
        source_emb_length = self.position_embeddings.shape[1]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and target_emb_length == source_emb_length:
            return self.position_embeddings

        source_emb = self.position_embeddings
        dim = embeddings.shape[-1]
        source_emb = source_emb.unsqueeze(1)
        source_emb = nn.functional.interpolate(
            source_emb,
            size=(target_emb_length, dim),
            mode="bilinear",
            antialias=True,
        )

        return source_emb.squeeze(1)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        input_shape: torch.Size,
        interpolate_pos_encoding: bool | None = False,
    ) -> torch.Tensor:
        if input_shape is not None:
            batch, frames, channel, height, width = input_shape
        _, features, dim = pixel_values_videos.shape
        hidden_states = pixel_values_videos.view(batch, frames, features, dim)
        hidden_states = hidden_states.transpose(2, 1)
        embeddings = hidden_states.reshape(batch * features, frames, dim)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class VideoPrismTextEmbeddings(nn.Module):
    def __init__(self, config: VideoPrismTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.register_buffer(
            "position_embedding", create_sinusoidal_positions(config.max_position_embeddings, config.hidden_size)
        )
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scale = config.hidden_size**0.5

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        if position_ids is None:
            position_ids = self.position_ids[:, : inputs_embeds.shape[1]]

        inputs_embeds = inputs_embeds * self.scale
        position_embeddings = self.position_embedding[position_ids].to(dtype=inputs_embeds.dtype)
        embeddings = inputs_embeds + position_embeddings

        cls_emb = self.cls_emb * self.scale
        cls_emb = cls_emb.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((embeddings, cls_emb), dim=1)
        return embeddings


class VideoPrismAttention(VivitAttention):
    def __init__(self, config: VideoPrismVisionConfig | VideoPrismTextConfig):
        super().__init__()
        self.num_key_value_groups = 1.0


class VideoPrismLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # a custom layernorm formula with gamma -> gamma + 1 is used in this model
        return F.layer_norm(hidden_states, self.normalized_shape, self.weight + 1, self.bias, self.eps)


class VideoPrismLayer(VivitLayer):
    def __init__(self, config: VideoPrismVisionConfig | VideoPrismTextConfig):
        super().__init__()
        self.layernorm_before = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)


@auto_docstring
class VideoPrismPreTrainedModel(PreTrainedModel):
    config: VideoPrismConfig
    base_model_prefix = "videoprism"
    main_input_name = "pixel_values_videos"
    input_modalities = ("video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "VideoPrismSpatialEmbeddings",
        "VideoPrismTemporalEmbeddings",
        "VideoPrismSpatialEncoder",
        "VideoPrismTemporalEncoder",
        "VideoPrismAuxiliaryEncoder",
        "VideoPrismTextEncoder",
        "VideoPrismMultiheadAttentionPoolingHead",
    ] # todo gotta change the class names
    # sdpa is disabled because it does not support attention capping 
    # used in eager and logits are too far off
    _supports_sdpa = False
    _supports_flash_attn = True
    _supports_attention_backend = True
    _supports_flex_attention = True
    _can_record_outputs = {
        "hidden_states": VideoPrismLayer,
        "attentions": VideoPrismAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            init.lecun_normal_(module.weight)

        elif isinstance(module, VideoPrismSpatialEmbeddings):
            init.lecun_normal_(module.position_embeddings)

        elif isinstance(module, VideoPrismTemporalEmbeddings):
            init.lecun_normal_(module.position_embeddings)

        elif isinstance(module, VideoPrismMultiheadAttentionPoolingHead):
            init.zeros_(module.per_dim_scale)
            init.lecun_normal_(module.pooling_attention_query)
            scale = module.scale.new_tensor(1.442695041 / (module.attention_head_size**0.5))
            init.copy_(module.scale, scale)

        elif isinstance(module, VideoPrismTextEmbeddings):
            position_embedding = create_sinusoidal_positions(
                module.config.max_position_embeddings, module.config.hidden_size
            ).to(device=module.position_embedding.device, dtype=module.position_embedding.dtype)
            init.copy_(module.position_embedding, position_embedding)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))

        elif isinstance(module, VideoPrismTextModel):
            init.normal_(module.embeddings.token_embedding.weight, std=module.config.hidden_size**-0.5)
            init.normal_(module.embeddings.cls_emb, std=module.config.hidden_size**-0.5)


@auto_docstring(
    custom_intro="""
    The bare VideoPrism vision encoder outputting raw hidden-states without any specific head on top. This model is the backbone encoder used in VideoPrismVideoModel.
    """
)
class VideoPrismVisionModel(VideoPrismPreTrainedModel):
    config: VideoPrismVisionConfig
    input_modalities = ("video",)
    base_model_prefix = "vision_model"
    _input_embed_layer = "patch_embedding"

    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.layernorm1 = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.spatial_embeddings = VideoPrismSpatialEmbeddings(config)
        self.temporal_embeddings = VideoPrismTemporalEmbeddings(config)
        self.spatial_layers = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_spatial_layers)])
        self.temporal_layers = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_temporal_layers)])
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.spatial_embeddings.patch_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.spatial_embeddings.patch_embeddings = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithSpatialAndTemporalStates:
        kwargs.setdefault("softcap", self.config.attn_logit_softcapping)
        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")

        input_shape = pixel_values_videos.shape

        # spatial
        spatial_embeds = self.spatial_embeddings(pixel_values_videos, interpolate_pos_encoding)
        spatial_hidden_states = spatial_embeds
        for spatial_layer in self.spatial_layers:
            spatial_hidden_states = spatial_layer(spatial_hidden_states, **kwargs)
        features = self.layernorm1(spatial_hidden_states)

        # temporal
        temporal_embeds = self.temporal_embeddings(features, input_shape, interpolate_pos_encoding)
        temporal_hidden_states = temporal_embeds
        for temporal_layer in self.temporal_layers:
            temporal_hidden_states = temporal_layer(temporal_hidden_states, **kwargs)
        features = self.layernorm2(temporal_hidden_states)

        # final reshape
        _, num_frames, dim = features.shape
        features = features.view(input_shape[0], -1, num_frames, dim).transpose(1, 2).contiguous()
        _, num_frames, num_patches, dim = features.shape
        features = features.view(input_shape[0], num_frames * num_patches, -1)

        return BaseModelOutputWithSpatialAndTemporalStates(
            last_hidden_state=features,
            temporal_hidden_state=temporal_hidden_states,
            spatial_hidden_state=spatial_hidden_states,
        )


class VideoPrismMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.intermediate_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.num_key_value_groups = 1.0
        # PerDimScale
        self.per_dim_scale = nn.Parameter(torch.zeros(self.attention_head_size))
        r_softplus_0 = 1.442695041
        scale = torch.tensor(r_softplus_0 / (self.attention_head_size**0.5))
        self.register_buffer("scale", scale)
        self.is_causal = False
        self.pooling_attention_query = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.q_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.qkv_bias)
        self.layernorm = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        kwargs.setdefault("is_causal", False)
        input_shape = hidden_states.shape[:-1]
        batch_size = input_shape[0]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query = self.pooling_attention_query.expand(batch_size, -1, -1)
        query_layer = self.q_proj(query).view(batch_size, 1, -1, self.attention_head_size).transpose(1, 2)

        softplus = nn.functional.softplus(self.per_dim_scale)
        scale = self.scale.to(query_layer.dtype) * softplus
        query_states = query_layer * scale.expand(*query_layer.shape)

        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=1.0, # 1.0 because head_size scaling has aleady been applied to query
            dropout=0.0 if not self.training else self.dropout_prob,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, 1, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        attn_output = self.layernorm(attn_output)
        return attn_output, attn_weights


@auto_docstring(
    custom_intro="""
    The bare VideoPrism text encoder outputting last hidden states without any specific head on top. This model is used in VideoPrismClipModel.
    """
)
class VideoPrismTextModel(VideoPrismPreTrainedModel):
    config: VideoPrismTextConfig
    input_modalities = ("text",)
    base_model_prefix = "text_model"
    main_input_name = "input_ids"
    _no_split_modules = ["VideoPrismTextEmbeddings", "VideoPrismLayer"]
    _input_embed_layer = "token_embedding"

    def __init__(self, config: VideoPrismTextConfig):
        super().__init__(config)
        self.embeddings = VideoPrismTextEmbeddings(config)
        self.layers = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_hidden_layers)])
        self.layernorm = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        kwargs.setdefault("softcap", self.config.attn_logit_softcapping)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)

        if attention_mask is not None:
            cls_padding = torch.ones(hidden_states.shape[0], 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, cls_padding), dim=1)
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
            )

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, **kwargs)
        hidden_states = self.layernorm(hidden_states)
        
        text_embeddings = hidden_states[:, -1]
        if self.config.apply_l2norm:
            text_embeddings = l2norm(text_embeddings, dim=-1)

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=text_embeddings)


@auto_docstring(
    custom_intro="""
    VideoPrism video model consisting of the vision encoder backbone with auxiliary encoder layers and an attention pooling head on top. This model is used in VideoPrismClipModel.
    """
)
class VideoPrismVideoModel(VideoPrismPreTrainedModel):
    config: VideoPrismVisionConfig

    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.vision_model = VideoPrismVisionModel._from_config(config)
        self.auxiliary_layers = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_auxiliary_layers)])
        self.head = VideoPrismMultiheadAttentionPoolingHead(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.vision_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.FloatTensor,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        kwargs.setdefault("softcap", self.config.attn_logit_softcapping)
        vision_model_outputs = self.vision_model(
            pixel_values_videos=pixel_values_videos, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        auxiliary_hidden_states = vision_model_outputs.last_hidden_state
        for layer in self.auxiliary_layers:
            auxiliary_hidden_states = layer(auxiliary_hidden_states, **kwargs)

        kwargs["softcap"] = None
        head_output = self.head(auxiliary_hidden_states, **kwargs)
        video_embeddings = head_output[0]
        if self.config.apply_l2norm:
            video_embeddings = l2norm(video_embeddings, dim=-1)

        return BaseModelOutputWithPooling(
            last_hidden_state=auxiliary_hidden_states,
            pooler_output=video_embeddings,
            hidden_states=getattr(vision_model_outputs, "hidden_states", None),
            attentions=getattr(vision_model_outputs, "attentions", None),
        )


@auto_docstring(
    custom_intro="""
    VideoPrism model for video-text contrastive learning. This model consists of a VideoPrismVideoModel and a VideoPrismTextModel, and computes similarity scores between video and text inputs.
    """
)
class VideoPrismClipModel(VideoPrismPreTrainedModel):
    def __init__(self, config: VideoPrismConfig):
        super().__init__(config)
        self.video_model = VideoPrismVideoModel._from_config(config.vision_config)
        self.text_model = VideoPrismTextModel._from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.text_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, VideoPrismClipModel

        >>> model = VideoPrismClipModel.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288")
        >>> tokenizer = AutoTokenizer.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288")

        >>> inputs = tokenizer(["a video of a cat.", "a video of a dog."], padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        return self.text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import VideoPrismProcessor, VideoPrismClipModel

        >>> model = VideoPrismClipModel.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288")
        >>> processor = VideoPrismProcessor.from_pretrained("MHRDYN7/videoprism-lvt-base-f16r288")

        >>> inputs = processor(videos="path/to/video.mp4", return_tensors="pt")
        >>> with torch.no_grad():
        ...     video_features = model.get_video_features(**inputs)
        ```"""
        return self.video_model(
            pixel_values_videos=pixel_values_videos,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.FloatTensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        temperature: float | None = None,
        return_loss: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VideoPrismClipOutput:
        r"""
        temperature (`float`, *optional*):
            A temperature scalar to scale the similarity scores. If not provided, no scaling is applied.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        """

        video_model_outputs = self.get_video_features(
            pixel_values_videos=pixel_values_videos, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        text_model_outputs = self.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        video_embeddings = video_model_outputs.pooler_output
        text_embeddings = text_model_outputs.pooler_output
        video_emb_dim = video_embeddings[0].shape[-1]
        text_emb_dim = text_embeddings[0].shape[-1]

        video_embeds = video_embeddings.reshape(-1, video_emb_dim)
        text_embeds = text_embeddings.reshape(-1, text_emb_dim)
        similarity_matrix = torch.matmul(video_embeds, text_embeds.T)

        if temperature is not None:
            similarity_matrix /= temperature

        logits_per_video = torch.exp(similarity_matrix)
        logits_per_text = logits_per_video.T
        logits_per_video = logits_per_video / torch.sum(logits_per_video, dim=0, keepdims=True)
        logits_per_text = logits_per_text / torch.sum(logits_per_text, dim=0, keepdims=True)

        # adopted from siglip
        loss = None
        if return_loss:
            # Adapted from https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/trainers/proj/image_text/siglip.py#L287
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()

        return VideoPrismClipOutput(
            logits_per_video=logits_per_video,
            logits_per_text=logits_per_text,
            video_embeds=video_embeds,
            text_embeds=text_embeds,
            video_model_output=video_model_outputs,
            text_model_output=text_model_outputs,
            loss=loss,
        )


@auto_docstring(
    custom_intro="""
    VideoPrism Model transformer with a video classification head on top (a linear layer on top of the attention pooler).
    """
)
class VideoPrismForVideoClassification(VideoPrismPreTrainedModel):
    config: VideoPrismVisionConfig
    input_modalities = ("video",)
    base_model_prefix = "vision_model"
    _input_embed_layer = "patch_embedding"

    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.vision_model = VideoPrismVisionModel._from_config(config)
        self.head = VideoPrismMultiheadAttentionPoolingHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.vision_model.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        kwargs.setdefault("softcap", self.config.attn_logit_softcapping)
        vision_model_outputs = self.vision_model(
            pixel_values_videos=pixel_values_videos, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        sequence_output = vision_model_outputs.last_hidden_state
        kwargs["softcap"] = None
        pooled_output = self.head(sequence_output, **kwargs)[0]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=vision_model_outputs.last_hidden_state,
        )


__all__ = [
    "VideoPrismVisionConfig",
    "VideoPrismTextConfig",
    "VideoPrismConfig",
    "VideoPrismVisionModel",
    "VideoPrismPreTrainedModel",
    "VideoPrismVideoModel",
    "VideoPrismTextModel",
    "VideoPrismClipModel",
    "VideoPrismForVideoClassification",
    "VideoPrismTokenizer",
    "VideoPrismProcessor",
]
