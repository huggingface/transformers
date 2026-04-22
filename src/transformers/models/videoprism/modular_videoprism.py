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
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
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
    VivitEncoder,
    VivitLayer,
    VivitSelfAttention,
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
    num_channels: int = 3
    num_spatial_layers: int = 12
    num_temporal_layers: int = 4
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_python"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-06
    qkv_bias: bool = True
    attn_logit_softcapping: float = 50.0
    num_auxiliary_layers: int = 2
    apply_l2norm: bool = True


@auto_docstring(checkpoint="google/videoprism-lvt-base-f16r288")
@strict
class VideoPrismTextConfig(SiglipTextConfig):
    r"""
    apply_l2norm (`bool`, *optional*, defaults to `True`):
        Whether to apply L2 normalization to the output of VideoPrismTextEncoder.
    attn_logit_softcapping (`float`, *optional*, defaults to 50.0):
        Softcapping constant for attention logits.
    """

    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 64
    hidden_act: str = "relu"
    layer_norm_eps: float = 1e-6
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
        },
    }


@auto_docstring
class VideoPrismProcessor(ProcessorMixin):
    valid_processor_kwargs = VideoPrismProcessorKwargs

    def __init__(self, video_processor=None, tokenizer=None):
        super().__init__(video_processor, tokenizer)


@dataclass
@auto_docstring(custom_intro="""Base class for model outputs that include spatial and temporal states.""")
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


@dataclass
@auto_docstring(custom_intro="""Base class for VideoPrismVideoModel outputs.""")
class VideoPrismVideoOutput(ModelOutput):
    r"""
    video_last_hidden_state (`torch.FloatTensor`):
        The pooled video embeddings after the attention pooling head, typically of shape
        `(batch_size, 1, hidden_size)`.
    auxiliary_output (`BaseModelOutput`, *optional*):
        The output of the auxiliary encoder. Its `last_hidden_state` is typically of shape
        `(batch_size, num_patches * num_frames, hidden_size)`.
    attention_pooling_output (`tuple(torch.FloatTensor, torch.FloatTensor)`, *optional*):
        The output tuple of [`VideoPrismMultiheadAttentionPoolingHead`] containing the pooled tensor of shape
        `(batch_size, 1, hidden_size)` and the attention probabilities of shape
        `(batch_size, num_attention_heads, 1, num_patches * num_frames)`.
    """

    video_last_hidden_state: torch.FloatTensor
    auxiliary_output: BaseModelOutput | None = None
    attention_pooling_output: tuple[torch.FloatTensor, torch.FloatTensor] | None = None


@dataclass
@auto_docstring(
    custom_intro="""Base class for VideoPrismClipModel outputs.""",
)
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
    video_model_output (`VideoPrismVideoOutput`):
        The output of the [`VideoPrismVideoModel`].
    text_model_output (`BaseModelOutput`):
        The output of the [`VideoPrismTextModel`].
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for video-text similarity.
    """

    logits_per_video: torch.FloatTensor | None = None
    logits_per_text: torch.FloatTensor | None = None
    video_embeds: torch.FloatTensor | None = None
    text_embeds: torch.FloatTensor | None = None
    video_model_output: VideoPrismVideoOutput = None
    text_model_output: BaseModelOutput = None
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
        self.image_size = (
            config.image_size if isinstance(config.image_size, tuple) else (config.image_size, config.image_size)
        )
        self.pos_emb_shape = [self.image_size[0] // self.patch_size[1], self.image_size[1] // self.patch_size[2]]
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
    """
    VideoPrism Spatial Embeddings.

    Creates embeddings from a video using VideoPrismSpatialTubeletEmbeddings and adds positional embeddings.
    This module differs from Vivit model
    """

    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        del self.cls_token
        self.tubelet_size = config.tubelet_size
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_embeddings.num_patches, config.hidden_size))

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

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

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.config.num_frames, config.hidden_size))

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


class VideoPrismSelfAttention(VivitSelfAttention):
    def __init__(self, config: VideoPrismVisionConfig | VideoPrismTextConfig):
        super().__init__(config)
        self.num_key_value_groups = 1.0
        self.attn_logit_softcapping = self.config.attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)

        query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output, attn_weights


class VideoPrismAttention(VivitAttention):
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, attention_mask, **kwargs)
        output = self.output(self_attn_output, hidden_states)
        return output


class VideoPrismLayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # a custom layernorm formula with gamma -> gamma + 1 is used in this model
        return F.layer_norm(hidden_states, self.normalized_shape, self.weight + 1, self.bias, self.eps)


class VideoPrismLayer(VivitLayer):
    def __init__(self, config: VideoPrismVisionConfig | VideoPrismTextConfig):
        super().__init__(config)
        del self.chunk_size_feed_forward
        del self.seq_len_dim
        self.layernorm_after = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_before = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, attention_mask, **kwargs)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in VideoPrism, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        return layer_output


class VideoPrismSpatialEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_spatial_layers)])


class VideoPrismTemporalEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_temporal_layers)])


class VideoPrismAuxiliaryEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_auxiliary_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, **kwargs)

        return BaseModelOutput(last_hidden_state=hidden_states)


class VideoPrismTextEncoder(VivitEncoder):
    def __init__(self, config: VideoPrismTextConfig):
        super().__init__(config)
        self.layer = nn.ModuleList([VideoPrismLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.is_causal = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask, **kwargs)

        return BaseModelOutput(last_hidden_state=hidden_states)


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
    ]
    _supports_sdpa = False
    _supports_flash_attn = True
    _supports_attention_backend = True
    _supports_flex_attention = True
    _can_record_outputs = {
        "hidden_states": VideoPrismLayer,
        "attentions": VideoPrismSelfAttention,
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
            scale = module.scale.new_tensor(1.442695041 / (module.dim**0.5))
            init.copy_(module.scale, scale)

        elif isinstance(module, VideoPrismTextEmbeddings):
            position_embedding = create_sinusoidal_positions(
                module.config.max_position_embeddings, module.config.hidden_size
            ).to(device=module.position_embedding.device, dtype=module.position_embedding.dtype)
            init.copy_(module.position_embedding, position_embedding)
            init.copy_(module.position_ids, torch.arange(module.position_ids.shape[-1]).expand((1, -1)))

        elif isinstance(module, VideoPrismTextModel):
            init.normal_(module.embeddings.token_embedding.weight, std=module.config.hidden_size**-0.5)
            init.normal_(module.cls_emb, std=module.config.hidden_size**-0.5)


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
        self.spatial_encoder = VideoPrismSpatialEncoder(config)
        self.temporal_encoder = VideoPrismTemporalEncoder(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.spatial_embeddings.patch_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.spatial_embeddings.patch_embeddings = value

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithSpatialAndTemporalStates:
        if pixel_values_videos is None:
            raise ValueError("You have to specify pixel_values_videos")

        input_shape = pixel_values_videos.shape

        # spatial
        spatial_embeds = self.spatial_embeddings(pixel_values_videos, interpolate_pos_encoding)
        spatial_encoder_outputs: BaseModelOutput = self.spatial_encoder(hidden_states=spatial_embeds, **kwargs)
        spatial_sequence_output = spatial_encoder_outputs.last_hidden_state
        features = self.layernorm1(spatial_sequence_output)

        # temporal
        temporal_embeds = self.temporal_embeddings(features, input_shape, interpolate_pos_encoding)
        temporal_encoder_outputs: BaseModelOutput = self.temporal_encoder(hidden_states=temporal_embeds, **kwargs)
        temporal_sequence_output = temporal_encoder_outputs.last_hidden_state
        features = self.layernorm2(temporal_sequence_output)

        # final reshape
        _, num_frames, dim = features.shape
        features = features.view(input_shape[0], -1, num_frames, dim).transpose(1, 2).contiguous()
        _, num_frames, num_patches, dim = features.shape
        features = features.view(input_shape[0], num_frames * num_patches, -1)

        return BaseModelOutputWithSpatialAndTemporalStates(
            last_hidden_state=features,
            temporal_hidden_state=temporal_sequence_output,
            spatial_hidden_state=spatial_sequence_output,
        )


class VideoPrismMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = self.config.num_attention_heads
        self.attention_head_size = int(self.config.intermediate_size / self.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = self.config.attention_probs_dropout_prob
        self.num_key_value_groups = 1.0
        # PerDimScale
        self.dim = int(self.config.intermediate_size / self.config.num_attention_heads)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.dim))
        r_softplus_0 = 1.442695041
        scale = torch.tensor(r_softplus_0 / (self.dim**0.5))
        self.register_buffer("scale", scale)
        self.is_causal = False
        self.pooling_attention_query = nn.Parameter(torch.zeros(1, 1, self.config.hidden_size))
        self.query = nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=self.config.qkv_bias)
        self.key = nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=self.config.qkv_bias)
        self.value = nn.Linear(self.config.hidden_size, self.config.intermediate_size, bias=self.config.qkv_bias)
        self.projection = nn.Linear(self.config.intermediate_size, self.config.hidden_size, bias=self.config.qkv_bias)
        self.layernorm = VideoPrismLayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.dim = int(self.config.intermediate_size / self.config.num_attention_heads)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size, seq_length, hidden_size = hidden_states.shape
        query = self.pooling_attention_query.expand(batch_size, -1, -1)
        query_layer = (
            self.query(query).view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        )
        softplus = nn.functional.softplus(self.per_dim_scale)
        scale = self.scale.to(query_layer.dtype) * softplus
        query_layer = query_layer * scale.expand(*query_layer.shape)

        key_layer = (
            self.key(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        value_layer = (
            self.value(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            is_causal=self.is_causal,
            scaling=1.0,
            dropout=0.0 if not self.training else self.dropout_prob,
            softcap=None,
            **kwargs,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        outputs = self.projection(context_layer)
        outputs = self.layernorm(outputs)
        return (outputs, attention_probs)


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

        inputs_embeds = inputs_embeds * self.config.hidden_size**0.5
        position_embeddings = self.position_embedding[position_ids].to(dtype=inputs_embeds.dtype)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


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
        self.config = config
        self.embeddings = VideoPrismTextEmbeddings(self.config)
        self.text_encoder = VideoPrismTextEncoder(self.config)
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.layernorm = VideoPrismLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.normalize = config.apply_l2norm
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        batch_size, seq_len, dim = hidden_states.shape
        cls_emb = self.cls_emb * (self.config.hidden_size**0.5)
        cls_emb = cls_emb.expand(hidden_states.shape[0], -1, -1)
        features = torch.cat((hidden_states, cls_emb), dim=1)

        if attention_mask is not None:
            cls_padding = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, cls_padding), dim=1)
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=features,
                attention_mask=attention_mask,
                past_key_values=None,
            )

        text_encoder_output = self.text_encoder(features, attention_mask)
        features = text_encoder_output.last_hidden_state
        features = self.layernorm(features)
        text_embeddings = features[:, -1]

        if self.normalize:
            text_embeddings = l2norm(text_embeddings, dim=-1)

        return BaseModelOutput(
            last_hidden_state=text_embeddings,
        )


@auto_docstring(
    custom_intro="""
    VideoPrism video model consisting of the vision encoder backbone with auxiliary encoder layers and an attention pooling head on top. This model is used in VideoPrismClipModel.
    """
)
class VideoPrismVideoModel(VideoPrismPreTrainedModel):
    config: VideoPrismVisionConfig

    def __init__(self, config: VideoPrismVisionConfig):
        super().__init__(config)
        self.backbone = VideoPrismVisionModel._from_config(config)
        self.auxiliary_encoder = VideoPrismAuxiliaryEncoder(config)
        self.contrastive_vision_pooler = VideoPrismMultiheadAttentionPoolingHead(config)
        self.normalize = config.apply_l2norm
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.spatial_embeddings.patch_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.backbone.spatial_embeddings.patch_embeddings = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.FloatTensor,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> VideoPrismVideoOutput:
        backbone_outputs = self.backbone(
            pixel_values_videos=pixel_values_videos, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        video_features = backbone_outputs.last_hidden_state
        auxiliary_output = self.auxiliary_encoder(video_features)
        auxiliary_output_features = auxiliary_output.last_hidden_state
        contrastive_vision_pooler_output = self.contrastive_vision_pooler(auxiliary_output_features, **kwargs)
        video_embeddings = contrastive_vision_pooler_output[0]
        if self.normalize:
            video_embeddings = l2norm(video_embeddings, dim=-1)

        return VideoPrismVideoOutput(
            video_last_hidden_state=video_embeddings,
            auxiliary_output=auxiliary_output,
            attention_pooling_output=contrastive_vision_pooler_output,
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

        video_model_outputs = self.video_model(
            pixel_values_videos=pixel_values_videos, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        text_model_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        video_embeddings = video_model_outputs.video_last_hidden_state
        text_embeddings = text_model_outputs.last_hidden_state
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
        if not isinstance(config, VideoPrismVisionConfig):
            raise TypeError(
                f"`config` is expected to be of type `VideoPrismVisionConfig` but is of type {type(config)}."
            )
        super().__init__(config)
        self.config = config
        self.encoder = VideoPrismVisionModel._from_config(self.config)
        self.contrastive_vision_pooler = VideoPrismMultiheadAttentionPoolingHead(self.config)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.spatial_embeddings.patch_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.encoder.spatial_embeddings.patch_embeddings = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values_videos: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        encoder_outputs = self.encoder(
            pixel_values_videos=pixel_values_videos, interpolate_pos_encoding=interpolate_pos_encoding, **kwargs
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.contrastive_vision_pooler(sequence_output, **kwargs)[0]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.last_hidden_state,
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
