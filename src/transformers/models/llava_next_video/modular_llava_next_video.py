# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

import math
from typing import Optional, Union

import torch
from torch import nn

from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextModelOutputWithPast,
    LlavaNextMultiModalProjector,
    TransformersKwargs,
    image_size_to_num_patches,
)

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import is_torchdynamo_compiling, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class LlavaNextVideoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaNextVideoForConditionalGeneration`]. It is used to instantiate an
    Llava-NeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [llava-hf/LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf)
    model.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 32001):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        multimodal_projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`. If `"default"`, the CLS token is removed from the vision features.
            If `"full"`, the full vision features are used.
        vision_feature_layer (`Union[int, list[int]]`, *optional*, defaults to -2):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features.
        image_grid_pinpoints (`List`, *optional*, defaults to `[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]`):
            A list of possible resolutions to use for processing high resolution images. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        video_token_index (`int`, *optional*, defaults to 32000):
            The video token index to encode the image prompt.
        spatial_pool_mode (`str`, *optional*, defaults to `"average"`):
            Pooling mode to use for videos. Can be "average", "max" or "conv".
        spatial_pool_stride (`int`, *optional*, defaults to 2):
            Stride used in the pooling layer for videos.
        image_seq_length (`int`, *optional*, defaults to 576):
            Sequence length of one image embedding.
        video_seq_length (`int`, *optional*, defaults to 288):
            Sequence length of one video embedding.

    Example:

    ```python
    >>> from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> configuration = LlavaNextVideoConfig(vision_config, text_config)

    >>> model = LlavaNextVideoForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_next_video"
    attribute_map = {
        "image_token_id": "image_token_index",
        "video_token_id": "video_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=32001,
        projector_hidden_act="gelu",
        multimodal_projector_bias=True,
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_grid_pinpoints=None,
        tie_word_embeddings=False,
        video_token_index=32000,
        spatial_pool_mode="average",
        spatial_pool_stride=2,
        image_seq_length=576,
        video_seq_length=288,
        **kwargs,
    ):
        self.video_token_index = video_token_index
        self.spatial_pool_mode = spatial_pool_mode
        self.spatial_pool_stride = spatial_pool_stride
        self.image_seq_length = image_seq_length
        self.video_seq_length = video_seq_length
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.multimodal_projector_bias = multimodal_projector_bias

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        image_grid_pinpoints = (
            image_grid_pinpoints
            if image_grid_pinpoints is not None
            else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
        )
        self.image_grid_pinpoints = image_grid_pinpoints

        if isinstance(vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class LlavaNextVideoModelOutputWithPast(LlavaNextModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor`  of size `(batch_size * num_frames, num_videos, sequence_length, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    video_hidden_states: Optional[torch.FloatTensor] = None


class LlavaNextVideoCausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
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
        A `torch.FloatTensor` of size (batch_size * num_patches, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor`  of size `(batch_size * num_frames, num_videos, sequence_length, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    video_hidden_states: Optional[torch.FloatTensor] = None


class LlavaNextVideoPooler(nn.Module):
    def __init__(self, config):
        super().__init__()

        mode = config.spatial_pool_mode
        stride = config.spatial_pool_stride
        out_channels = getattr(config, "spatial_pool_out_channels", config.vision_config.hidden_size)
        self.image_size = (config.vision_config.image_size // config.vision_config.patch_size) ** 2

        if mode == "average":
            self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        elif mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        elif mode == "conv":
            self.pool = nn.Conv2d(
                in_channels=config.vision_config.hidden_size,
                out_channels=out_channels,
                kernel_size=stride,
                stride=stride,
            )
        else:
            raise ValueError(f"Unknown pooling mode: {mode}. Has to be one of [`average`, `max`, `conv`]")

    def forward(self, image_features):
        ori_width = int(math.sqrt(image_features.shape[1] * self.image_size // self.image_size))
        ori_height = int(ori_width * self.image_size // self.image_size)

        batch_size, _, dim = image_features.shape
        image_features_spatial = image_features.view(batch_size, ori_height, ori_height, dim).permute(0, 3, 1, 2)
        image_features_spatial_pool = self.pool(image_features_spatial)

        return image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()


class LlavaNextVideoMultiModalProjector(LlavaNextMultiModalProjector):
    pass


class LlavaNextVideoModel(LlavaNextModel):
    def __init__(self, config: LlavaNextVideoConfig, **super_kwargs):
        super().__init__(config, **super_kwargs)
        self.vision_resampler = LlavaNextVideoPooler(config)
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_layer (`Union[int, list[int]]`, *optional*):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (list[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_features.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)

        image_features, feature_lens = self.pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy,
            image_newline=self.image_newline,
        )
        return image_features

    def get_video_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ):
        """
        Obtains video last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_frames, channels, height, width)`)
               The tensors corresponding to the input video.
            vision_feature_layer (`Union[int, list[int]]`, *optiona;*):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            video_features (list[`torch.Tensor`]): List of video feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_videos, video_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        batch_size, frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values, output_hidden_states=True)

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_video_features = video_features.hidden_states[vision_feature_layer]
        else:
            hs_pool = [video_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_video_features = torch.cat(hs_pool, dim=-1)

        if vision_feature_select_strategy == "default":
            selected_video_features = selected_video_features[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_video_features = selected_video_features

        # Same as image features except that video has pooling layer
        video_features = self.vision_resampler(selected_video_features)
        video_features = self.multi_modal_projector(video_features)
        video_features = torch.split(video_features, frames, dim=0)
        return video_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, LlavaNextVideoModelOutputWithPast]:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, image_size, image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`LlavaNextVideoVideoProcessor.__call__`] for details. [`LlavaProcessor`] uses
            [`LlavaNextVideoVideoProcessor`] for processing videos.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        self.vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=self.vision_feature_layer,
                vision_feature_select_strategy=self.vision_feature_select_strategy,
            )
            image_features = torch.cat(image_features, dim=0)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id

            n_image_tokens = (special_image_mask).sum()
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_features = image_features.shape[0]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if pixel_values_videos is not None and pixel_values_videos.size(0) > 0:
            video_features = self.get_video_features(
                pixel_values_videos,
                vision_feature_layer=self.vision_feature_layer,
                vision_feature_select_strategy=self.vision_feature_select_strategy,
            )
            video_features = [feature.flatten(0, 1) for feature in video_features]
            video_feature_lens = [feature.size(0) for feature in video_features]
            video_features = torch.cat(video_features, dim=0)
            video_feature_lens = torch.tensor(video_feature_lens, dtype=torch.long, device=video_features.device)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.video_token_id

            n_video_tokens = (special_image_mask).sum()
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != video_features.numel():
                n_video_features = video_features.shape[0]
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, video_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return LlavaNextVideoModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            video_hidden_states=video_features if pixel_values_videos is not None else None,
        )


class LlavaNextVideoForConditionalGeneration(LlavaNextForConditionalGeneration):
    def get_video_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ):
        return self.model.get_video_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaNextVideoCausalLMOutputWithPast]:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, image_size, image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`LlavaNextVideoVideoProcessor.__call__`] for details. [`LlavaProcessor`] uses
            [`LlavaNextVideoVideoProcessor`] for processing videos.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> import av
        >>> from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`list[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

        >>> model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", device_map="auto")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")

        >>> prompt = "USER: <video>\nWhy is this video funny? ASSISTANT:"
        >>> video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
        >>> container = av.open(video_path)

        >>> # sample uniformly 8 frames from the video (model was trained with 32 frames per video, but this video is short)
        >>> total_frames = container.streams.video[0].frames
        >>> indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        >>> clip = read_video_pyav(container, indices)
        >>> inputs_video = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)

        >>> # load an image to generate from an image
        >>> prompt = "USER:<image>\nWhat is shown in this image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs_image = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        >>> # Generate from video
        >>> generate_ids = model.generate(**inputs_video, max_length=50)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:\nWhy is this video funny? ASSISTANT: The humor in this video comes from the unexpected and endearing sight of a baby wearing glasses and (...)"

        >>> # Generate from image
        >>> generate_ids = model.generate(**inputs_image, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER: \nWhat's the content of the image? ASSISTANT: The image shows a red stop sign on a pole, with a traditional Chinese archway (...)"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            image_sizes=image_sizes,
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

        return LlavaNextVideoCausalLMOutputWithPast(
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
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_sizes=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- extra custom processing

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["image_sizes"] = image_sizes

        return model_inputs


__all__ = [
    "LlavaNextVideoConfig",
    "LlavaNextVideoForConditionalGeneration",
    "LlavaNextVideoModel",
    "LlavaNextVideoPreTrainedModel",  # noqa: F822
]
