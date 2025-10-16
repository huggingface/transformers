# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
    valid_images,
    validate_preprocess_arguments,
)
from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import check_model_inputs
from ...video_utils import (
    VideoInput,
    group_videos_by_shape,
    make_batched_videos,
    reorder_videos,
)
from ..auto import CONFIG_MAPPING, AutoConfig
from ..auto.modeling_auto import AutoModel
from ..qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorKwargs, smart_resize
from ..qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLImageProcessorFast,
)
from ..qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
    TransformersKwargs,
    VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from ..qwen2_vl.processing_qwen2_vl import (
    Qwen2VLProcessor,
    Qwen2VLProcessorKwargs,
)
from ..qwen2_vl.video_processing_qwen2_vl import (
    Qwen2VLVideoProcessor,
    Qwen2VLVideoProcessorInitKwargs,
)
from ..siglip.configuration_siglip import SiglipVisionConfig
from ..siglip.modeling_siglip import (
    SiglipAttention,
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipMLP,
)


logger = logging.get_logger(__name__)


class VideoLlama3VisionConfig(SiglipVisionConfig):
    """
    This is the configuration class to store the configuration of a [`VideoLlama3VisionModel`]. It is used to instantiate a
    VideoLLaMA3 vision encoder model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    VideoLLaMA3-2B [lkhl/VideoLLaMA3-2B-Image-HF](https://huggingface.co/lkhl/VideoLLaMA3-2B-Image-HF).

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    """

    model_type = "video_llama_3_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.initializer_range = initializer_range
        del self.image_size


class VideoLlama3Config(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a [`VideoLlama3Model`]. It is used to instantiate a
    VideoLLaMA3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    VideoLLaMA3-2B [lkhl/VideoLLaMA3-2B-Image-HF](https://huggingface.co/lkhl/VideoLLaMA3-2B-Image-HF).

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `VideoLlama3VisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index to encode the image prompt.
    """

    model_type = "video_llama_3"
    sub_configs = {"vision_config": VideoLlama3VisionConfig, "text_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif isinstance(vision_config, PreTrainedConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            raise ValueError(
                f"vision_config must be of type `dict` or `PreTrainedConfig`, but got {type(vision_config)}."
            )

        if isinstance(text_config, dict):
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif isinstance(text_config, PreTrainedConfig):
            self.text_config = text_config
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"]()
        else:
            raise ValueError(f"text_config must be of type `dict` or `PreTrainedConfig`, but got {type(text_config)}.")

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)


class VideoLlama3VisionRotaryEmbedding(VisionRotaryEmbedding):
    def forward(self, grid_thw, merge_sizes) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        for (t, h, w), merge_size in zip(grid_thw, merge_sizes):
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_thw = grid_thw[:, 1:].max()

        seq = torch.arange(max_grid_thw, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        rotary_pos_emb_full = torch.outer(seq, self.inv_freq)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        return (emb.cos(), emb.sin())


class VideoLlama3VisionEmbeddings(nn.Module):
    def __init__(self, config: VideoLlama3VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, self.config.num_channels, self.patch_size, self.patch_size)
        patch_embeds = self.patch_embedding(hidden_states)
        embeddings = patch_embeds.view(-1, self.embed_dim)
        return embeddings


class VideoLlama3VisionMLP(SiglipMLP):
    pass


class VideoLlama3VisionAttention(SiglipAttention):
    def __init__(self, config):
        super().__init__(config)
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        del self.scale
        del self.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input to the layer of shape `(seq_len, embed_dim)`.
            cu_seqlens (`torch.Tensor` of shape `(num_images_or_videos + 1,)`):
                The cumulative sequence lengths of each image or video feature.
            position_embeddings (`tuple(torch.Tensor, torch.Tensor)` of shape `(num_patches, head_dim // 2)`):
                The cosine and sine position embeddings for vision attention.
        """
        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
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
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )
                attn_outputs.append(attn_output)
                attn_weights.append(attn_weight)

            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class VideoLlama3VisionEncoderLayer(SiglipEncoderLayer):
    def __init__(self, config: VideoLlama3VisionConfig):
        super().__init__(config)
        self.self_attn = VideoLlama3VisionAttention(config=config)
        self.mlp = VideoLlama3VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        cu_seqlens (`torch.Tensor` of shape `(num_images_or_videos + 1,)`):
            The cumulative sequence lengths of each image or video feature.
        position_embeddings (`tuple(torch.Tensor, torch.Tensor)` of shape `(num_patches, head_dim // 2)`):
            The cosine and sine position embeddings for vision attention.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VideoLlama3VisionEncoder(SiglipEncoder):
    def __init__(self, config: VideoLlama3VisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([VideoLlama3VisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        cu_seqlens (`torch.Tensor` of shape `(num_images_or_videos + 1,)`):
            The cumulative sequence lengths of each image or video feature.
        position_embeddings (`tuple(torch.Tensor, torch.Tensor)` of shape `(num_patches, head_dim // 2)`):
            The cosine and sine position embeddings for vision attention.
        """
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


class VideoLlama3PreTrainedModel(Qwen2VLPreTrainedModel):
    config: VideoLlama3Config
    _no_split_modules = ["VideoLlama3VisionEncoderLayer"]


class VideoLlama3VisionModel(VideoLlama3PreTrainedModel):
    config: VideoLlama3VisionConfig
    main_input_name = "pixel_values"
    input_modalities = "image"
    _can_record_outputs = {
        "hidden_states": VideoLlama3VisionEncoderLayer,
        "attentions": VideoLlama3VisionAttention,
    }

    def __init__(self, config: VideoLlama3VisionConfig):
        super().__init__(config)
        head_dim = config.hidden_size // config.num_attention_heads

        self.rotary_pos_emb = VideoLlama3VisionRotaryEmbedding(head_dim // 2)
        self.embeddings = VideoLlama3VisionEmbeddings(config)
        self.encoder = VideoLlama3VisionEncoder(config)
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def get_input_embeddings(self) -> VideoLlama3VisionEmbeddings:
        return self.embeddings.patch_embedding

    def pixel_unshuffle(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        merge_sizes: torch.Tensor,
    ):
        hidden_states_chunks = hidden_states.split(grid_thw.prod(dim=1).tolist(), dim=0)
        outputs = []

        for hidden_states, (t, h, w), merge_size in zip(hidden_states_chunks, grid_thw, merge_sizes):
            c = hidden_states.shape[-1]
            hidden_states = hidden_states.view(t, h // merge_size, w // merge_size, merge_size, merge_size, c).permute(
                0, 1, 3, 2, 4, 5
            )
            hidden_states = hidden_states.reshape(t, h, w, c).permute(0, 3, 1, 2)
            hidden_states = torch.nn.functional.interpolate(
                hidden_states, size=(h // merge_size, w // merge_size), mode="bilinear"
            )
            hidden_states = hidden_states.permute(0, 2, 3, 1).view(-1, c)
            outputs.append(hidden_states)

        return torch.cat(outputs, dim=0)

    @check_model_inputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        merge_sizes: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
        merge_sizes (`torch.Tensor` of shape `(num_images_or_videos,)`):
            The spatial downsampling ratio of each image or video feature.
        """
        position_embeddings = self.rotary_pos_emb(grid_thw, merge_sizes)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = self.embeddings(pixel_values.type(self.dtype))
        encoder_outputs: BaseModelOutput = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        last_hidden_state = self.pixel_unshuffle(last_hidden_state, grid_thw, merge_sizes)

        return BaseModelOutput(last_hidden_state=last_hidden_state)


class VideoLlama3Projector(nn.Module):
    def __init__(self, config: VideoLlama3Config) -> None:
        super().__init__()
        in_hidden_size = config.vision_config.hidden_size
        out_hidden_size = config.text_config.hidden_size
        self.readout = nn.Sequential(
            nn.Linear(in_hidden_size, out_hidden_size),
            nn.GELU(),
            nn.Linear(out_hidden_size, out_hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.readout(hidden_states)
        return hidden_states


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for VideoLLaMA3 outputs, with hidden states and attentions.
    """
)
class VideoLlama3ModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_images_features, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_features, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    video_hidden_states: Optional[torch.FloatTensor] = None


class VideoLlama3Model(Qwen2VLModel):
    _checkpoint_conversion_mapping = {}
    _can_compile_fullgraph = False

    def __init__(self, config: VideoLlama3Config):
        PreTrainedModel.__init__(self, config)
        self.vision_model = AutoModel.from_config(config.vision_config)
        self.projector = VideoLlama3Projector(config)
        self.language_model = AutoModel.from_config(config.text_config)

        self.post_init()

    def get_rope_index(self):
        raise AttributeError("Not needed for VideoLLaMA3")

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        video_merge_sizes: torch.LongTensor,
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
                The spatial downsampling ratio of each video feature.
        """
        return self.get_image_features(pixel_values_videos, video_grid_thw, video_merge_sizes)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
        image_merge_sizes: torch.LongTensor,
    ):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            image_merge_sizes (`torch.Tensor` of shape `(num_images,)`):
                The spatial downsampling ratio of each image feature.
        """
        image_embeds = self.vision_model(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
            merge_sizes=image_merge_sizes,
            return_dict=True,
        ).last_hidden_state
        image_embeds = self.projector(image_embeds)

        split_sizes = image_grid_thw.prod(dim=1) // (image_merge_sizes**2)
        image_embeds = torch.split(image_embeds, split_sizes.tolist())

        return image_embeds

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, VideoLlama3ModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        image_merge_sizes (`torch.Tensor` of shape `(num_images,)`):
            The spatial downsampling ratio of each image feature.
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`):
            The temporal, height and width of feature shape of each video before vision encoder.
        video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
            The spatial downsampling ratio of each video feature.
        video_compression_mask (`torch.BoolTensor` of shape `(num_video_features,)`, *optional*):
            The mask to indicate which video features are kept after token compression.
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, image_merge_sizes)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        video_embeds = None
        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, video_merge_sizes)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            if video_compression_mask is not None:
                video_embeds = video_embeds[video_compression_mask.to(video_embeds.device)]
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return VideoLlama3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_embeds,
            video_hidden_states=video_embeds,
        )


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for VideoLLaMA3 causal language model (or autoregressive) outputs.
    """
)
class VideoLlama3CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_images_features, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(num_video_features, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    video_hidden_states: Optional[torch.FloatTensor] = None


class VideoLlama3ForConditionalGeneration(Qwen2VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {}
    _can_compile_fullgraph = False

    def __init__(self, config: VideoLlama3Config):
        super().__init__(config)  # just to add type hint on config

    def visual(self):
        raise AttributeError("Not needed for VideoLLaMA3")

    @property
    def vision_model(self):
        return self.model.vision_model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, VideoLlama3CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        image_merge_sizes (`torch.Tensor` of shape `(num_images,)`):
            The spatial downsampling ratio of each image feature.
        video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`):
            The temporal, height and width of feature shape of each video before vision encoder.
        video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
            The spatial downsampling ratio of each video feature.
        video_compression_mask (`torch.BoolTensor` of shape `(num_video_features,)`, *optional*):
            The mask to indicate which video features are kept after token compression.
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_merge_sizes=image_merge_sizes,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            video_merge_sizes=video_merge_sizes,
            video_compression_mask=video_compression_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return VideoLlama3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_merge_sizes=image_merge_sizes,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            video_merge_sizes=video_merge_sizes,
            video_compression_mask=video_compression_mask,
            use_cache=use_cache,
            **kwargs,
        )

        if model_inputs["cache_position"][0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_merge_sizes: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        video_merge_sizes: Optional[torch.LongTensor] = None,
        video_compression_mask: Optional[torch.BoolTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id

        if inputs_embeds is not None:
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        if image_grid_thw is not None:
            num_image_features = image_grid_thw.prod(dim=1) // (image_merge_sizes**2)
        else:
            num_image_features = []

        if video_grid_thw is not None:
            num_video_features = video_grid_thw.prod(dim=1) // (video_merge_sizes**2)
            if video_compression_mask is not None:
                num_video_features = video_compression_mask.split(num_video_features.tolist())
                num_video_features = [mask.sum() for mask in num_video_features]
        else:
            num_video_features = []

        image_nums, video_nums = [], []
        start_image_idx, start_video_idx = 0, 0

        for num_image_tokens, num_video_tokens in zip(image_mask.sum(dim=1), video_mask.sum(dim=1)):
            cu_num_features = 0
            image_idx = start_image_idx
            while image_idx < len(num_image_features) and cu_num_features < num_image_tokens:
                cu_num_features += num_image_features[image_idx]
                image_idx += 1
            assert cu_num_features == num_image_tokens, (
                "The number of image tokens does not match the number of image features."
            )
            image_nums.append(image_idx - start_image_idx)
            start_image_idx = image_idx

            cu_num_features = 0
            video_idx = start_video_idx
            while video_idx < len(num_video_features) and cu_num_features < num_video_tokens:
                cu_num_features += num_video_features[video_idx]
                video_idx += 1
            assert cu_num_features == num_video_tokens, (
                "The number of video tokens does not match the number of video features."
            )
            video_nums.append(video_idx - start_video_idx)
            start_video_idx = video_idx

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = [
            "pixel_values",
            "image_grid_thw",
            "image_merge_sizes",
            "pixel_values_videos",
            "video_grid_thw",
            "video_merge_sizes",
            "video_compression_mask",
        ]

        def _repeat_interleave_samples(x, lengths, repeat_times):
            samples = torch.split(x, lengths)
            repeat_args = [repeat_times] + [1] * (x.dim() - 1)
            result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
            return result

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            video_merge_sizes = model_kwargs.get("video_merge_sizes", None)
            video_compression_mask = model_kwargs.get("video_compression_mask", None)

            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids,
                inputs_embeds=model_kwargs.get("inputs_embeds", None),
                image_grid_thw=image_grid_thw,
                image_merge_sizes=model_kwargs.get("image_merge_sizes", None),
                video_grid_thw=video_grid_thw,
                video_merge_sizes=video_merge_sizes,
                video_compression_mask=video_compression_mask,
            )
            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_merge_sizes":
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_compression_mask":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    merge_sizes = torch.split(video_merge_sizes, list(video_nums))
                    lengths = [
                        (torch.prod(sample, dim=1) // merge_size**2).sum()
                        for sample, merge_size in zip(samples, merge_sizes)
                    ]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_merge_sizes":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )

            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
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


class VideoLlama3ProcessorKwargs(Qwen2VLProcessorKwargs):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class VideoLlama3Processor(Qwen2VLProcessor):
    r"""
    Constructs a VideoLLaMA3 processor which wraps a VideoLLaMA3 image processor and a Qwen2 tokenizer into a single processor.
    [`VideoLlama3Processor`] offers all the functionalities of [`VideoLlama3ImageProcessor`] and [`Qwen2Tokenizer`]. See the
    [`~VideoLlama3Processor.__call__`] and [`~VideoLlama3Processor.decode`] for more information.
    Args:
        image_processor ([`VideoLlama3ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2Tokenizer`], *optional*):
            The tokenizer is a required input.
        video_processor ([`VideoLlama3VideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
    """

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[VideoLlama3ProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            VideoLlama3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
            image_merge_sizes = image_inputs["image_merge_sizes"]
        else:
            image_grid_thw = image_merge_sizes = []

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            num_video_tokens = [
                grid_thw.prod() // merge_size**2
                for grid_thw, merge_size in zip(videos_inputs["video_grid_thw"], videos_inputs["video_merge_sizes"])
            ]
            video_compression_masks = videos_inputs["video_compression_mask"].split(num_video_tokens)
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
            timestamps = []
            for metadata in video_metadata:
                if metadata.fps is None:
                    logger.warning_once(
                        "VideoLLaMA4 requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                        "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                        "Defaulting to `fps=1`. Please provide `video_metadata` for more accurate results."
                    )
                metadata.fps = 1 if metadata.fps is None else metadata.fps
                timestamps.append(metadata.timestamps)
        else:
            video_compression_masks = timestamps = []

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if images is not None:
            image_index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[image_index].prod() // (image_merge_sizes[image_index] ** 2)
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    image_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            video_index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    frame_compression_masks = video_compression_masks[video_index].split(
                        len(video_compression_masks[video_index]) // len(timestamps[video_index])
                    )
                    num_frame_tokens = [x.sum() for x in frame_compression_masks]
                    frame_prompts = [
                        f"Time {t:.1f}s:" + "<|placeholder|>" * n
                        for n, t in zip(num_frame_tokens, timestamps[video_index])
                    ]
                    text[i] = text[i].replace(self.video_token, ",".join(frame_prompts), 1)
                    video_index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)


class VideoLlama3ImageProcessorKwargs(Qwen2VLImageProcessorKwargs):
    pass


class VideoLlama3ImageProcessor(Qwen2VLImageProcessor):
    r"""
    Constructs a VideoLLaMA3 image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        size (`dict[str, int]`, *optional*, defaults to `{"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}`):
            Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 1):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "image_merge_sizes"]
    valid_kwargs = VideoLlama3ImageProcessorKwargs

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        merge_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            **kwargs,
        )

        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

        if self.temporal_patch_size != 1:
            raise ValueError("`temporal_patch_size` must be 1 for VideoLLaMA3")

    def preprocess(
        self,
        images: ImageInput,
        videos: Optional[VideoInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            min_pixels (`int`, *optional*, defaults to `self.min_pixels`):
                The min pixels of the image to resize the image.
            max_pixels (`int`, *optional*, defaults to `self.max_pixels`):
                The max pixels of the image to resize the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        data = {}
        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            data.update(
                {
                    "pixel_values": np.array(pixel_values),
                    "image_grid_thw": np.array(vision_grid_thws),
                    "image_merge_sizes": np.array([merge_size] * len(vision_grid_thws)),
                }
            )

        return BatchFeature(data=data, tensor_type=return_tensors)


class VideoLlama3ImageProcessorFast(Qwen2VLImageProcessorFast):
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    temporal_patch_size = 1
    merge_size = 1
    valid_kwargs = VideoLlama3ImageProcessorKwargs
    model_input_names = [
        "pixel_values",
        "image_grid_thw",
        "image_merge_sizes",
        "pixel_values_videos",
        "video_grid_thw",
        "video_merge_sizes",
    ]

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        videos: VideoInput,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Optional[Union[str, "torch.device"]] = None,
        **kwargs: Unpack[VideoLlama3ImageProcessorKwargs],
    ) -> BatchFeature:
        # Prepare input images
        batch_feature = BatchFeature()
        if images is not None:
            if kwargs["temporal_patch_size"] != 1:
                raise ValueError("`temporal_patch_size` must be 1 for VideoLLaMA3")
            images = self._prepare_image_like_inputs(
                images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
            )
            batch_feature = self._preprocess(images, **kwargs)
            batch_feature["image_merge_sizes"] = torch.tensor(
                [kwargs["merge_size"]] * batch_feature.image_grid_thw.size(0),
                dtype=batch_feature.image_grid_thw.dtype,
                device=batch_feature.image_grid_thw.device,
            )
        if videos is not None:
            logger.warning(
                "`VideoLlama3ImageProcessorFast` works only with image inputs and doesn't process videos anymore. "
                "This is a deprecated behavior and will be removed in v5.0. "
                "Your videos should be forwarded to `VideoLlama3VideoProcessor`. "
            )
            # Can't change _prepare_images_structure to work with videos because it also needs to work with images.
            videos = make_batched_videos(videos)
            videos = [
                torch.stack(self._prepare_image_like_inputs(video, do_convert_rgb, input_data_format, device))
                for video in videos
            ]
            video_outputs = self._preprocess(videos, **kwargs)
            batch_feature.update(
                {"pixel_values_videos": video_outputs.pixel_values, "video_grid_thw": video_outputs.image_grid_thw}
            )
            batch_feature["video_merge_sizes"] = torch.tensor(
                [kwargs["merge_size"]] * video_outputs.image_grid_thw.size(0),
                dtype=video_outputs.image_grid_thw.dtype,
                device=video_outputs.image_grid_thw.device,
            )
        return batch_feature


class VideoLlama3VideoProcessorInitKwargs(Qwen2VLVideoProcessorInitKwargs):
    use_token_compression: Optional[bool]


class VideoLlama3VideoProcessor(Qwen2VLVideoProcessor):
    use_token_compression = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    temporal_patch_size = 1
    max_frames = 180
    return_metadata = True
    valid_kwargs = VideoLlama3VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw", "video_merge_sizes", "video_compression_mask"]

    def _get_compression_mask(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        video_merge_sizes: torch.LongTensor,
        threshold: Optional[float] = 0.1,
        min_tokens: Optional[int] = 1,
    ) -> torch.BoolTensor:
        """
        Get the compression mask for video tokens based on pixel differences.
        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            video_merge_sizes (`torch.Tensor` of shape `(num_videos,)`):
                The spatial downsampling ratio of each video feature.
            threshold (`float`, *optional*, defaults to 0.1):
                The threshold to determine whether a token should be kept based on pixel differences.
            min_tokens (`int`, *optional*, defaults to 1):
                The minimum number of tokens to keep for each frame.
        """
        videos = pixel_values_videos.split(video_grid_thw.prod(dim=1).tolist(), dim=0)
        compression_masks = []

        for images, grid_size, merge_size in zip(videos, video_grid_thw, video_merge_sizes):
            t, h, w = grid_size
            if t == 1:
                num_tokens = images.size(0) // (merge_size**2)
                compression_masks.append(torch.ones((num_tokens,), dtype=torch.bool, device=images.device))
            else:
                # NOTE: video token compressor
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = images[1:] - images[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = pixel_diff > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())

        return torch.cat(compression_masks)

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        use_token_compression: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional["torch.Tensor"] = None,
        **kwargs,
    ):
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels // shape[0],
                )
                stacked_videos = self.resize(
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        video_merge_sizes = torch.tensor([merge_size] * video_grid_thw.size(0)).to(video_grid_thw)

        if use_token_compression:
            video_compression_mask = self._get_compression_mask(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_merge_sizes=video_merge_sizes,
            )
        else:
            num_video_tokens = video_grid_thw.prod(-1).sum() // (merge_size**2)
            video_compression_mask = torch.ones(
                (num_video_tokens,), dtype=torch.bool, device=pixel_values_videos.device
            )

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
                "video_merge_sizes": video_merge_sizes,
                "video_compression_mask": video_compression_mask,
            },
            tensor_type=return_tensors,
        )


__all__ = [
    "VideoLlama3VisionConfig",
    "VideoLlama3Config",
    "VideoLlama3VisionModel",
    "VideoLlama3PreTrainedModel",
    "VideoLlama3Model",
    "VideoLlama3ForConditionalGeneration",
    "VideoLlama3Processor",
    "VideoLlama3ImageProcessor",
    "VideoLlama3ImageProcessorFast",
    "VideoLlama3VideoProcessor",
]
