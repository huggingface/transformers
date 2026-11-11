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
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import is_flash_attention_requested
from ...vision_utils import get_max_seqlen, get_vision_attention_seqlens, get_vision_position_ids
from ..auto import AutoConfig, AutoModel
from ..lfm2.modeling_lfm2 import Lfm2DecoderLayer
from ..lfm2_vl.modeling_lfm2_vl import Lfm2VlPreTrainedModel
from ..lfm3.modeling_lfm3 import Lfm3MLP, Lfm3RMSNorm
from ..llama.modeling_llama import eager_attention_forward
from ..qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    apply_rotary_pos_emb_vision,
)
from ..qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from ..qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention, Qwen3VLVisionRotaryEmbedding
from ..qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


@auto_docstring(checkpoint="LiquidAI/LFM3-VL-2B")
@strict
class Lfm3VlVisionConfig(Qwen3VLVisionConfig):
    r"""
    conv_kernel_size (`int`, *optional*, defaults to 3):
        Kernel size of the depthwise convolution over the patch grid of each frame.
    """

    model_type = "lfm3_vl_vision"
    default_theta = 1_000_000.0
    attribute_map = AttributeError()

    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    layer_types: list[str] | None = None
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    depth = AttributeError()
    hidden_act = AttributeError()
    num_heads = AttributeError()
    out_hidden_size = AttributeError()
    num_position_embeddings = AttributeError()
    deepstack_visual_indexes = AttributeError()

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = ["conv" if i < 2 or i % 2 else "full_attention" for i in range(self.num_hidden_layers)]
        PreTrainedConfig.__post_init__(self, **kwargs)


@auto_docstring(checkpoint="LiquidAI/LFM3-VL-2B")
@strict
class Lfm3VlConfig(PreTrainedConfig):
    model_type = "lfm3_vl"
    sub_configs_defaults = {
        "vision_config": Lfm3VlVisionConfig,
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="lfm3"),
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 125_030
    video_token_id: int = 125_029
    vision_start_token_id: int = 125_009
    vision_end_token_id: int = 125_010
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 4096
    projector_bias: bool = True
    tie_word_embeddings: bool = True


class Lfm3VlVisionRMSNorm(Lfm3RMSNorm):
    pass


class Lfm3VlVisionMLP(Lfm3MLP):
    pass


class Lfm3VlVisionRotaryEmbedding(Qwen3VLVisionRotaryEmbedding):
    pass


class Lfm3VlVisionAttention(Qwen3VLVisionAttention):
    """Bidirectional grouped-query attention over the patches of one frame, with RMSNorm on the query and key heads."""

    def __init__(self, config: Lfm3VlVisionConfig, layer_idx: int):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        del self.qkv
        del self.proj
        self.q_proj = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)
        self.q_norm = Lfm3VlVisionRMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = Lfm3VlVisionRMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # diff -> separate grouped-query projections with RMSNorm on the query and key heads instead of a fused qkv
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
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
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs, attn_weights = [], []
            for q, k, v in zip(*splits):
                attn_output, attn_weight = attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=attention_mask,
                    scaling=self.scaling,
                    dropout=0.0,
                    is_causal=False,
                    **kwargs,
                )
                attn_outputs.append(attn_output)
                attn_weights.append(attn_weight)

            attn_output = torch.cat(attn_outputs, dim=1)
            # frames of different shapes have differently sized maps: one entry per frame, or None under sdpa
            attn_weights = None if attn_weights[0] is None else tuple(attn_weights)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Lfm3VlVisionShortConv(nn.Module):
    """Gated depthwise 2D convolution over the patch grid of each frame."""

    def __init__(self, config: Lfm3VlVisionConfig, layer_idx: int):
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

    def forward(self, hidden_states: torch.Tensor, grid_thw: list[list[int]], **kwargs):
        input_gate, output_gate, conv_input = self.in_proj(hidden_states).chunk(3, dim=-1)
        gated = input_gate * conv_input
        merge = self.spatial_merge_size
        # patches arrive in merge-block order: raster the frames of each image or video to their grid, convolve them
        # as one batch (the frames of an item share a shape), and go back
        conv_outputs = []
        channels = gated.shape[-1]
        lengths = [temporal * height * width for temporal, height, width in grid_thw]
        for frames, (temporal, height, width) in zip(torch.split(gated, lengths, dim=0), grid_thw):
            frames = frames.view(temporal, height // merge, width // merge, merge, merge, channels)
            frames = frames.permute(0, 5, 1, 3, 2, 4).reshape(temporal, channels, height, width)
            frames = self.spatial_conv(frames)
            frames = frames.view(temporal, channels, height // merge, merge, width // merge, merge)
            conv_outputs.append(frames.permute(0, 2, 4, 3, 5, 1).reshape(temporal * height * width, channels))
        hidden_states = torch.cat(conv_outputs, dim=0)
        hidden_states = output_gate * hidden_states
        return self.out_proj(hidden_states)


class Lfm3VlVisionLayer(Lfm2DecoderLayer):
    pass


class Lfm3VlPreTrainedModel(Lfm2VlPreTrainedModel):
    input_modalities = ("image", "video", "text")
    # the cached decode step traces without graph breaks, like the text model
    _can_compile_fullgraph = True
    _no_split_modules = ["Lfm3VlVisionLayer"]


@auto_docstring(
    custom_intro="""
    The LFM3-VL vision encoder: a stack of gated convolution and bidirectional attention layers over the patches of
    each image or video frame, embedded by a linear patch embedding.
    """
)
class Lfm3VlVisionModel(Qwen2VisionTransformerPretrainedModel):
    _no_split_modules = ["Lfm3VlVisionLayer"]
    _can_record_outputs = {
        "hidden_states": Lfm3VlVisionLayer,
        "attentions": Lfm3VlVisionAttention,
    }

    def __init__(self, config: Lfm3VlVisionConfig):
        super().__init__(config)
        self.patch_embed = nn.Linear(
            config.in_channels * config.temporal_patch_size * config.patch_size**2, config.hidden_size, bias=False
        )
        self.embedding_norm = Lfm3VlVisionRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.layers = nn.ModuleList(
            [Lfm3VlVisionLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        del self.blocks
        del self.merger

    def get_dtype(self):
        raise AttributeError("Not needed for Lfm3Vl")

    def get_device(self):
        raise AttributeError("Not needed for Lfm3Vl")

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
        # diff -> the convolution reshapes each frame to its grid, which needs host-side sizes
        grid_thw_list = grid_thw.tolist()  # trf-ignore: TRF056

        hidden_states = self.patch_embed(pixel_values)
        hidden_states = self.embedding_norm(hidden_states)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                grid_thw=grid_thw_list,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class Lfm3VlMultiModalProjector(nn.Module):
    """Merges each block of `spatial_merge_size**2` vision tokens, normalized before the merge, into one language token."""

    def __init__(self, config: Lfm3VlConfig):
        super().__init__()
        self.merged_hidden_size = config.vision_config.hidden_size * config.vision_config.spatial_merge_size**2
        self.layer_norm = nn.LayerNorm(config.vision_config.hidden_size, eps=1e-6)
        self.linear_1 = nn.Linear(self.merged_hidden_size, config.projector_hidden_size, bias=config.projector_bias)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size, config.text_config.hidden_size, bias=config.projector_bias
        )

    def forward(self, image_features: torch.Tensor):
        # normalize each vision token before merging the blocks
        hidden_states = self.layer_norm(image_features).reshape(-1, self.merged_hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Vl outputs, with hidden states and attentions.
    """
)
@dataclass
class Lfm3VlModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    image_pooler_output (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_image_tokens, hidden_size)`.
        Image features produced by the vision encoder and projected to the language model hidden size.
    video_pooler_output (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_tokens, hidden_size)`.
        Video features produced by the vision encoder and projected to the language model hidden size.
    """

    image_pooler_output: torch.FloatTensor | None = None
    video_pooler_output: torch.FloatTensor | None = None


@auto_docstring(
    custom_intro="""
    Base class for Lfm3Vl causal language model (or autoregressive) outputs.
    """
)
@dataclass
class Lfm3VlCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    image_pooler_output (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_image_tokens, hidden_size)`.
        Image features produced by the vision encoder and projected to the language model hidden size.
    video_pooler_output (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_tokens, hidden_size)`.
        Video features produced by the vision encoder and projected to the language model hidden size.
    """

    image_pooler_output: torch.FloatTensor | None = None
    video_pooler_output: torch.FloatTensor | None = None


class Lfm3VlModel(Qwen2VLModel):
    def __init__(self, config: Lfm3VlConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = Lfm3VlMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        del self.visual
        del self.rope_deltas

    def get_vision_position_ids(self):
        raise AttributeError("Lfm3Vl uses 2D vision and 1D text positions")

    def get_rope_index(self):
        raise AttributeError("Lfm3Vl uses 2D vision and 1D text positions")

    def compute_3d_position_ids(self):
        raise AttributeError("Lfm3Vl uses 2D vision and 1D text positions")

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        image_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_features = self.multi_modal_projector(image_outputs.last_hidden_state)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_tower.spatial_merge_size**2).tolist()
        image_outputs.pooler_output = torch.split(image_features, split_sizes)
        return image_outputs

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Lfm3VlModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )
        if mm_encoder_outputs.get("video") is None and pixel_values_videos is not None:
            mm_encoder_outputs["video"] = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        if mm_encoder_outputs.get("image") is not None:
            image_features = torch.cat(mm_encoder_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

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
            cache=cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        return Lfm3VlModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_pooler_output=image_features if mm_encoder_outputs.get("image") is not None else None,
            video_pooler_output=video_features if mm_encoder_outputs.get("video") is not None else None,
        )


@auto_docstring(
    custom_intro="""
    The LFM3_VL model which consists of a vision backbone and a language model.
    """
)
class Lfm3VlForConditionalGeneration(Qwen2VLForConditionalGeneration):
    @auto_docstring
    def get_image_features(self, **super_kwargs) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every image, grouped by merge block.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of the patch grid of each image.
        """
        return super().get_image_features(**super_kwargs)

    @auto_docstring
    def get_video_features(self, **super_kwargs) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(num_patches, in_channels * temporal_patch_size * patch_size**2)`):
            The flattened patches of every video, grouped by merge block.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`):
            The temporal, height and width dimensions of the patch grid of each video.
        """
        return super().get_video_features(**super_kwargs)

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
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
        Example:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText

        >>> model = AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM3-VL-2B")
        >>> processor = AutoProcessor.from_pretrained("LiquidAI/LFM3-VL-2B")

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
        ... ).to(model.device, model.dtype)

        >>> outputs = model.generate(**inputs, max_new_tokens=45)
        >>> processor.batch_decode(outputs, skip_special_tokens=True)[0]
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
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
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_pooler_output=outputs.image_pooler_output,
            video_pooler_output=outputs.video_pooler_output,
        )

    def _prepare_position_ids_for_generation(self):
        raise AttributeError("Lfm3Vl uses 2D vision and 1D text positions")


@auto_docstring
class Lfm3VlProcessor(Qwen3VLProcessor):
    return_mm_token_type_ids = False

    def model_input_names(self):
        raise AttributeError("Lfm3Vl doesn't need to override it")


__all__ = [
    "Lfm3VlConfig",
    "Lfm3VlVisionConfig",
    "Lfm3VlForConditionalGeneration",
    "Lfm3VlModel",
    "Lfm3VlPreTrainedModel",
    "Lfm3VlVisionModel",
    "Lfm3VlProcessor",
]
