# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from huggingface_hub.dataclasses import strict
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, slice_logits_to_keep
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)
from ...utils.generic import accepts_precomputed_kwargs, is_flash_attention_requested, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ...vision_utils import get_max_seqlen, get_vision_attention_seqlens, get_vision_position_ids
from ..auto import AutoConfig, AutoModel
from ..lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig
from ..lfm2_vl.modeling_lfm2_vl import (
    Lfm2VlForConditionalGeneration,
    Lfm2VlMultiModalProjector,
    Lfm2VlPreTrainedModel,
)
from ..lfm3.modeling_lfm3 import Lfm3MLP, Lfm3RMSNorm, create_masks_by_layer_type
from ..llama.modeling_llama import eager_attention_forward
from ..qwen2_vl.modeling_qwen2_vl import Qwen2VLModel, Qwen2VLVisionRotaryEmbedding, apply_rotary_pos_emb_vision
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from ..qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor


@auto_docstring(checkpoint="LiquidAI/LFM3-VL-1.9B-Preview")
@strict
class Lfm3VlVisionConfig(PreTrainedConfig):
    r"""
    layer_types (`list[str]`, *optional*):
        Per-layer operator type, each one of `"full_attention"` or `"conv"`. Defaults to two gated convolution
        layers followed by alternating attention and convolution layers.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial size of each image patch.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The number of video frames packed into one patch.
    spatial_merge_size (`int`, *optional*, defaults to 2):
        The side of the square block of patches merged into one language token.
    conv_kernel_size (`int`, *optional*, defaults to 3):
        The kernel size of the depthwise 2D convolution layers.
    norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the RMS normalization layers.
    """

    model_type = "lfm3_vl_vision"
    base_config_key = "vision_config"
    default_theta = 1_000_000.0
    default_rope_type = "axial"
    _attention_layer_types = ("full_attention",)

    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    layer_types: list[str] | None = None
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["conv" if i < 2 or i % 2 else "full_attention" for i in range(self.num_hidden_layers)]
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of ``@strict``-powered validation."""
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"`layer_types` has {len(self.layer_types)} entries but `num_hidden_layers` is {self.num_hidden_layers}."
            )
        allowed_layer_types = (*self._attention_layer_types, "conv")
        invalid_layer_types = [layer_type for layer_type in self.layer_types if layer_type not in allowed_layer_types]
        if invalid_layer_types:
            raise ValueError(f"`layer_types` entries must be one of {allowed_layer_types}; got {invalid_layer_types}.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`hidden_size` must be a multiple of `num_attention_heads`.")


@auto_docstring(checkpoint="LiquidAI/LFM3-VL-1.9B-Preview")
@strict
class Lfm3VlConfig(Lfm2VlConfig):
    r"""
    video_token_id (`int`, *optional*, defaults to 125029):
        The id of the placeholder token for video patches.
    vision_start_token_id (`int`, *optional*, defaults to 125009):
        The id of the token opening an image or a video frame.
    vision_end_token_id (`int`, *optional*, defaults to 125010):
        The id of the token closing an image or a video frame.
    projector_layer_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon of the layer normalization applied to the vision features before projection.
    """

    model_type = "lfm3_vl"
    sub_configs_defaults = {
        "vision_config": Lfm3VlVisionConfig,
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm3"),
    }

    image_token_id: int = 125_030
    video_token_id: int = 125_029
    vision_start_token_id: int = 125_009
    vision_end_token_id: int = 125_010
    projector_hidden_size: int = 4096
    projector_layer_norm_eps: float = 1e-6
    projector_use_layernorm = AttributeError()
    downsample_factor = AttributeError()


class Lfm3VlRMSNorm(Lfm3RMSNorm):
    pass


class Lfm3VlMLP(Lfm3MLP):
    pass


class Lfm3VlVisionRotaryEmbedding(Qwen2VLVisionRotaryEmbedding):
    @staticmethod
    def compute_axial_rope_parameters(config: Lfm3VlVisionConfig, **kwargs) -> tuple[torch.Tensor, float]:
        """Inverse frequencies of the axial RoPE over half the head dimension per axis, with an attention scaling of 1."""
        base = config.rope_parameters["rope_theta"]
        spatial_dim = config.hidden_size // config.num_attention_heads // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq, 1.0


class Lfm3VlVisionAttention(nn.Module):
    """Bidirectional grouped-query attention over the patches of one frame, with RMSNorm on the query and key heads."""

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_layernorm = Lfm3VlRMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm3VlRMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q_layernorm(self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim))
        key_states = self.k_layernorm(
            self.k_proj(hidden_states).view(seq_length, self.num_key_value_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: process each frame separately
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]
            attn_outputs, attn_weights = [], []
            for q, k, v in zip(*splits):
                output, weights = attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0,
                    is_causal=False,
                    **kwargs,
                )
                attn_outputs.append(output)
                attn_weights.append(weights)
            attn_output = torch.cat(attn_outputs, dim=1)
            # frames of different shapes have differently sized maps: one entry per frame, or None under sdpa
            attn_weights = None if attn_weights[0] is None else tuple(attn_weights)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


def get_vision_frame_shapes(grid_thw: torch.Tensor) -> list[list[int]]:
    # the convolution reshapes each frame to its grid, which needs host-side sizes
    return grid_thw[:, 1:].repeat_interleave(grid_thw[:, 0], dim=0).tolist()


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 2D convolution over the patch grid of each frame."""

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.spatial_conv = nn.Conv2d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, frame_shapes: list[list[int]]):
        B, C, x = self.in_proj(hidden_states).chunk(3, dim=-1)
        gated = B * x
        merge = self.spatial_merge_size
        # patches arrive in merge-block order: raster each frame to its grid, convolve, and go back
        conv_outputs = []
        lengths = [height * width for height, width in frame_shapes]
        for frame, (height, width) in zip(torch.split(gated, lengths, dim=0), frame_shapes):
            frame = frame.view(height // merge, width // merge, merge, merge, -1)
            frame = frame.permute(4, 0, 2, 1, 3).reshape(1, -1, height, width)
            frame = self.spatial_conv(frame)
            frame = frame.view(-1, height // merge, merge, width // merge, merge).permute(1, 3, 2, 4, 0)
            conv_outputs.append(frame.reshape(height * width, -1))
        return self.out_proj(C * torch.cat(conv_outputs, dim=0))


class Lfm3VlVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: Lfm3VlVisionConfig, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm3VlVisionAttention(config)
        else:
            self.conv = Lfm3VlVisionConv2d(config)
        self.feed_forward = Lfm3VlMLP(config)
        self.operator_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        frame_shapes: list[list[int]],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_attention_layer:
            operator_output, _ = self.self_attn(
                self.operator_norm(hidden_states),
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            operator_output = self.conv(self.operator_norm(hidden_states), frame_shapes)
        hidden_states = hidden_states + operator_output
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


class Lfm3VlPreTrainedModel(Lfm2VlPreTrainedModel):
    input_modalities = ("image", "video", "text")
    # the cached decode step traces without graph breaks, like the text model
    _can_compile_fullgraph = True
    _no_split_modules = ["Lfm3VlVisionLayer"]

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        return create_masks_by_layer_type(config, inputs_embeds, attention_mask, past_key_values, position_ids)


@auto_docstring(
    custom_intro="""
    The LFM3-VL vision encoder: a stack of gated convolution and bidirectional attention layers over the patches of
    each image or video frame, embedded by a linear patchifier.
    """
)
class Lfm3VlVisionModel(Lfm3VlPreTrainedModel):
    config: Lfm3VlVisionConfig
    input_modalities = ("image", "video")
    _input_embed_layer = "patchifier"
    _can_record_outputs = {
        "hidden_states": Lfm3VlVisionLayer,
        "attentions": OutputRecorder(Lfm3VlVisionAttention, index=1),
    }

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.patchifier = nn.Linear(patch_dim, config.hidden_size, bias=False)
        self.embedding_norm = Lfm3VlRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_pos_emb = Lfm3VlVisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Lfm3VlVisionLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Unpack[TransformersKwargs]
    ) -> BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image or video, grouped by merge block.
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each image or video.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        frame_shapes = get_vision_frame_shapes(grid_thw)

        hidden_states = self.patchifier(pixel_values.to(self.patchifier.weight.dtype))
        hidden_states = self.embedding_norm(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                frame_shapes=frame_shapes,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class Lfm3VlMultiModalProjector(Lfm2VlMultiModalProjector):
    """Merges each block of `spatial_merge_size**2` vision tokens, normalized before the merge, into one language token."""

    def __init__(self, config: Lfm3VlConfig):
        nn.Module.__init__(self)
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.projector_layer_norm_eps)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * config.vision_config.spatial_merge_size**2,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )

    def forward(self, image_features: torch.Tensor):
        hidden_states = self.layer_norm(image_features).reshape(-1, self.linear_1.in_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_unshuffle(self, hidden_states: torch.Tensor):
        raise AttributeError("Not needed for Lfm3Vl")


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Vl outputs, with hidden states and attentions.
    """
)
@dataclass
class Lfm3VlModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_image_tokens, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_tokens, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None
    video_hidden_states: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Vl causal language model (or autoregressive) outputs.
    """
)
@dataclass
class Lfm3VlCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_image_tokens, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_tokens, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None
    video_hidden_states: torch.FloatTensor | None = None


class Lfm3VlModel(Qwen2VLModel):
    accepts_loss_kwargs = AttributeError()

    def __init__(self, config: Lfm3VlConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = Lfm3VlMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        del self.visual
        del self.rope_deltas

    def get_vision_position_ids(self):
        raise AttributeError("Lfm3Vl uses 1D positions")

    def get_rope_index(self):
        raise AttributeError("Lfm3Vl uses 1D positions")

    def compute_3d_position_ids(self):
        raise AttributeError("Lfm3Vl uses 1D positions")

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of the patch grid of each image.
        """
        image_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_features = self.multi_modal_projector(image_outputs.last_hidden_state)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        image_outputs.pooler_output = torch.split(image_features, split_sizes)
        return image_outputs

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains video last hidden states from the vision tower and apply multimodal projection."
    )
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3VlModelOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each image.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be supplied together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be supplied together")
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        image_features = None
        if mm_encoder_outputs.get("image") is not None:
            image_features = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        video_features = None
        if mm_encoder_outputs.get("video") is not None:
            video_features = torch.cat(mm_encoder_outputs["video"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Lfm3VlModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
            video_hidden_states=video_features,
        )


class Lfm3VlForConditionalGeneration(Lfm2VlForConditionalGeneration):
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of the patch grid of each image.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3VlCausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each image.
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`, *optional*):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width dimensions of the patch grid of each video.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> model = AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM3-VL-1.9B-Preview")
        >>> processor = AutoProcessor.from_pretrained("LiquidAI/LFM3-VL-1.9B-Preview")

        >>> conversation = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
        ...             {"type": "text", "text": "What is in this image?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ... )

        >>> outputs = model.generate(**inputs, max_new_tokens=45)
        >>> processor.batch_decode(outputs, skip_special_tokens=True)[0]
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs[0]
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Lfm3VlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )


class Lfm3VlVideoProcessor(Qwen3VLVideoProcessor):
    size = {"shortest_edge": 4_096, "longest_edge": 25_165_824}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    min_frames = 4
    max_frames = 768
    cap_pixels_per_frame = False
    fps = 2.0


class Lfm3VlProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class Lfm3VlProcessor(Qwen3VLProcessor):
    valid_processor_kwargs = Lfm3VlProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", "<|image|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video|>")
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(self.vision_end_token)
        ProcessorMixin.__init__(self, image_processor, tokenizer, video_processor, chat_template=chat_template)

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.image_processor.model_input_names
                + self.video_processor.model_input_names
            )
        )

    def _process_videos(self, videos, **kwargs):
        # the frame timestamps need the metadata whatever the caller's `return_metadata`; `__call__` drops it from
        # the output unless it was asked for
        kwargs["return_metadata"] = True
        return ProcessorMixin._process_videos(self, videos, **kwargs)

    def replace_video_token(self, video_inputs: dict, video_idx: int, **kwargs) -> str:
        metadata = video_inputs["video_metadata"][video_idx]
        if metadata.frames_indices is None:
            raise ValueError("Frame timestamps need the sampled frame indices in `video_metadata`")
        return super().replace_video_token(video_inputs, video_idx, **kwargs)


__all__ = [
    "Lfm3VlConfig",
    "Lfm3VlVisionConfig",
    "Lfm3VlForConditionalGeneration",
    "Lfm3VlModel",
    "Lfm3VlPreTrainedModel",
    "Lfm3VlVisionModel",
    "Lfm3VlVideoProcessor",
    "Lfm3VlProcessor",
]
