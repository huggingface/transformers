# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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

from typing import Optional, Union

import torch
import torch.nn as nn

from ...cache_utils import Cache
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    get_size_dict,
    group_images_by_shape,
    reorder_images,
)
from ...image_transforms import convert_to_rgb, to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...processing_utils import Unpack
from ...tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)
from ...utils import (
    TensorType,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    filter_out_non_signature_kwargs,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..deepseek_vl.configuration_deepseek_vl import DeepseekVLConfig
from ..deepseek_vl.image_processing_deepseek_vl import DeepseekVLImageProcessor
from ..deepseek_vl.image_processing_deepseek_vl_fast import DeepseekVLImageProcessorFast
from ..deepseek_vl.modeling_deepseek_vl import (
    DeepseekVLForConditionalGeneration,
    DeepseekVLModel,
    DeepseekVLPreTrainedModel,
)
from ..deepseek_vl.processing_deepseek_vl import DeepseekVLProcessor, DeepseekVLProcessorKwargs
from ..idefics.modeling_idefics import IdeficsBaseModelOutputWithPast, IdeficsCausalLMOutputWithPast
from ..sam.modeling_sam import SamLayerNorm, SamVisionNeck


if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F

    from ...image_utils import pil_torch_interpolation_mapping
elif is_torchvision_available():
    from torchvision.transforms import functional as F

    from ...image_utils import pil_torch_interpolation_mapping


logger = logging.get_logger(__name__)


DEEPSEEK_VL_COMMON_CUSTOM_ARGS = r"""
    high_res_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size), *optional*):
        The tensors corresponding to the input images. Pixel values can be obtained using
        [`AutoImageProcessor`].
"""


class DeepseekVLHybridConfig(DeepseekVLConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekVLHybridModel`]. It is used to instantiate a
    DeepseekVLHybrid model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DeepseekVLHybrid
    [deepseek-community/deepseek-vl-7b-chat](https://huggingface.co/deepseek-community/deepseek-vl-7b-chat) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SiglipVisionConfig`):
            The config object or dictionary of the vision backbone.
        high_res_vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SamVisionConfig`):
            The config object or dictionary of the high resolution vision backbone.
        image_token_id (`int`, *optional*, defaults to 100015):
            The index representing image tokens in the model's token vocabulary.

    Example:

    ```python
    >>> from transformers import DeepseekVLHybridConfig, DeepseekVLHybridModel

    >>> # Initializing a DeepseekVLHybrid deepseek-community/deepseek-vl-7b-chat style configuration
    >>> configuration = DeepseekVLHybridConfig()

    >>> # Initializing a model (with random weights) from the deepseek-community/deepseek-vl-7b-chat style configuration
    >>> model = DeepseekVLHybridModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "deepseek_vl_hybrid"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig, "high_res_vision_config": AutoConfig}

    def __init__(
        self,
        text_config: AutoConfig = None,
        vision_config: AutoConfig = None,
        high_res_vision_config: AutoConfig = None,
        image_token_id: int = 100015,
        **kwargs,
    ):
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            **kwargs,
        )

        if high_res_vision_config is None:
            high_res_vision_config = {}
            logger.info("`high_res_vision_config` is `None`. Initializing the `SamVisionConfig` with default values.")

        if isinstance(high_res_vision_config, dict):
            high_res_vision_config["model_type"] = high_res_vision_config.get("model_type", "sam_vision_model")
            high_res_vision_config = CONFIG_MAPPING[high_res_vision_config["model_type"]](**high_res_vision_config)

        self.high_res_vision_config = high_res_vision_config


class DeepseekVLHybridBaseModelOutputWithPast(IdeficsBaseModelOutputWithPast):
    pass


class DeepseekVLHybridCausalLMOutputWithPast(IdeficsCausalLMOutputWithPast):
    pass


class DeepseekVLHybridLayerNorm(SamLayerNorm):
    pass


class DeepseekVLSamVisionNeck(SamVisionNeck):
    def __init__(self, config):
        super().__init__(config)


class DeepseekVLSamVisionProj(nn.Module):
    def __init__(self, config, output_size: int = 24):
        super().__init__()
        self.config = config
        self.output_size = output_size

        self.conv1 = nn.Conv2d(
            config.output_channels, config.output_channels * 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            config.output_channels * 2, config.output_channels * 4, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # interpolate Sam encodings to match Siglip encodings
        features = torch.nn.functional.interpolate(
            features,
            size=(4 * self.output_size, 4 * self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        features = self.conv1(features)
        features = self.conv2(features)
        return features


class DeepseekVLHybridAligner(nn.Module):
    def __init__(self, config: DeepseekVLHybridConfig):
        super().__init__()

        in_channels = config.vision_config.hidden_size
        high_res_in_channels = config.high_res_vision_config.output_channels * 4
        out_channels = config.text_config.hidden_size

        self.vision_proj = nn.Linear(in_channels, out_channels // 2)
        self.high_res_vision_proj = nn.Linear(high_res_in_channels, out_channels // 2)

        self.act = nn.GELU()
        self.proj = nn.Linear(out_channels, out_channels)

    def forward(
        self,
        vision_encodings: torch.Tensor,
        high_res_vision_encodings: torch.Tensor,
    ) -> torch.Tensor:
        vision_encodings = self.vision_proj(vision_encodings)
        high_res_vision_encodings = self.high_res_vision_proj(high_res_vision_encodings)

        encodings = torch.concat([high_res_vision_encodings, vision_encodings], dim=-1)
        encodings = self.act(encodings)
        encodings = self.proj(encodings)

        return encodings


class DeepseekVLHybridPreTrainedModel(DeepseekVLPreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DeepseekVLHybridLayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, DeepseekVLHybridModel):
            module.high_res_vision_alpha.data.zero_()


class DeepseekVLHybridModel(DeepseekVLModel):
    def __init__(self, config):
        self.output_size = config.vision_config.image_size // config.vision_config.patch_size
        self.global_attn_index = config.high_res_vision_config.global_attn_indexes[0]

        self.high_res_vision_model = AutoModel.from_config(config.high_res_vision_config)
        self.high_res_vision_neck = DeepseekVLSamVisionNeck(config.high_res_vision_config)
        self.high_res_vision_proj = DeepseekVLSamVisionProj(
            config.high_res_vision_config, output_size=self.output_size
        )
        self.high_res_vision_alpha = nn.Parameter(torch.zeros(1))

        super().__init__(config)

    def get_low_res_image_features(self, pixel_values):
        output = self.vision_model(pixel_values)
        output = output[0]
        return output

    def get_high_res_image_features(self, pixel_values):
        output = self.high_res_vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = output.last_hidden_state
        last_hidden_state = self.high_res_vision_proj(last_hidden_state)

        hidden_states = output.hidden_states
        global_hidden_state = hidden_states[self.global_attn_index + 1]  # +1 for embedding layer
        global_hidden_state = self.high_res_vision_neck(global_hidden_state)
        global_hidden_state = self.high_res_vision_proj(global_hidden_state)

        output = last_hidden_state + global_hidden_state * self.high_res_vision_alpha

        # batch_size, hidden_size, height, width -> batch_size, seq_len, hidden_size
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.shape[0], -1, output.shape[-1])

        return output

    def get_image_features(self, pixel_values, high_res_pixel_values):
        vision_encodings = self.get_low_res_image_features(pixel_values)
        high_res_vision_encodings = self.get_high_res_image_features(high_res_pixel_values)
        images_embeds = self.aligner(vision_encodings, high_res_vision_encodings)
        return images_embeds

    @can_return_tuple
    @auto_docstring(custom_args=DEEPSEEK_VL_COMMON_CUSTOM_ARGS)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        high_res_pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and high_res_pixel_values is None:
            raise ValueError("Both pixel_values and high_res_pixel_values should be specified at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            if input_ids is None:
                image_attention_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                image_attention_mask = image_attention_mask.all(-1)
            else:
                image_attention_mask = input_ids == self.config.image_token_id

            image_attention_mask = image_attention_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = self.get_image_features(pixel_values, high_res_pixel_values)
            image_features = image_embeds.reshape(-1, inputs_embeds.shape[-1])
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_attention_mask, image_features)

        lm_output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        return DeepseekVLHybridBaseModelOutputWithPast(
            last_hidden_state=lm_output.last_hidden_state,
            past_key_values=lm_output.past_key_values,
            hidden_states=lm_output.hidden_states,
            attentions=lm_output.attentions,
            image_hidden_states=image_embeds if pixel_values is not None else None,
        )


class DeepseekVLHybridForConditionalGeneration(DeepseekVLForConditionalGeneration):
    @can_return_tuple
    @auto_docstring(custom_args=DEEPSEEK_VL_COMMON_CUSTOM_ARGS)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        high_res_pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            high_res_pixel_values=high_res_pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return DeepseekVLHybridCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        high_res_pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values
            model_inputs["high_res_pixel_values"] = high_res_pixel_values

        return model_inputs


class DeepseekVLHybridImageProcessor(DeepseekVLImageProcessor):
    r"""
    Constructs a DEEPSEEK_VL_HYBRID image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        high_res_size (`dict`, *optional*, defaults to `{"height": 1024, "width": 1024}`):
            Size of the high resolution output image after resizing. Can be overridden by the `high_res_size` parameter in the `preprocess`
            method.
        min_size (`int`, *optional*, defaults to 14):
            The minimum allowed size for the resized image. Ensures that neither the height nor width
            falls below this value after resizing.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        high_res_resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `high_res_resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        high_res_image_mean (`float` or `list[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            Mean to use if normalizing the high resolution image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `high_res_image_mean` parameter in the `preprocess` method.
        high_res_image_std (`float` or `list[float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
            Standard deviation to use if normalizing the high resolution image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `high_res_image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values", "high_res_pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        high_res_size: Optional[dict[str, int]] = None,
        min_size: int = 14,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        high_res_resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        high_res_image_mean: Optional[Union[float, list[float]]] = None,
        high_res_image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        **kwargs,
    ) -> None:
        high_res_size = high_res_size if high_res_size is not None else {"height": 1024, "width": 1024}
        high_res_size = get_size_dict(high_res_size, default_to_square=True)

        self.high_res_size = high_res_size
        self.high_res_image_mean = high_res_image_mean if high_res_image_mean is not None else OPENAI_CLIP_MEAN
        self.high_res_image_std = high_res_image_std if high_res_image_std is not None else OPENAI_CLIP_STD

        self.resample = resample
        self.high_res_resample = high_res_resample

        super().__init__(
            do_resize=do_resize,
            size=size,
            min_size=min_size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )

        if high_res_image_mean is None:
            self.high_res_background_color = (127, 127, 127)
        else:
            self.high_res_background_color = tuple(int(x * 255) for x in high_res_image_mean)

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        high_res_size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        high_res_resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        high_res_image_mean: Optional[Union[float, list[float]]] = None,
        high_res_image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            high_res_size (`Dict[str, int]`, *optional*, defaults to `self.high_res_size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the high resolution output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            high_res_resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BICUBIC`. Only has
                an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            high_res_image_mean (`float` or `List[float]`, *optional*, defaults to `self.high_res_image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            high_res_image_std (`float` or `List[float]`, *optional*, defaults to `self.high_res_image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
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
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        high_res_resample = high_res_resample if high_res_resample is not None else self.high_res_resample
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        high_res_image_mean = high_res_image_mean if high_res_image_mean is not None else self.high_res_image_mean
        high_res_image_std = high_res_image_std if high_res_image_std is not None else self.high_res_image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)
        high_res_size = high_res_size if high_res_size is not None else self.high_res_size
        high_res_size_dict = get_size_dict(high_res_size)

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        all_images = []
        all_high_res_images = []
        for image in images:
            # high_res_image: resize (high) -> rescale -> normalize (high)
            # low_res_image:  resize (high) -> rescale -> resize (low) -> normalize (low)
            high_res_image = image
            if do_resize:
                high_res_image = self.resize(
                    image=high_res_image,
                    size=high_res_size_dict,
                    background_color=self.high_res_background_color,
                    resample=high_res_resample,
                    input_data_format=input_data_format,
                )
                image = self.resize(
                    image=high_res_image,
                    size=size_dict,
                    background_color=self.background_color,
                    resample=resample,
                    input_data_format=input_data_format,
                )

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                high_res_image = self.rescale(
                    image=high_res_image, scale=rescale_factor, input_data_format=input_data_format
                )

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )
                high_res_image = self.normalize(
                    image=high_res_image,
                    mean=high_res_image_mean,
                    std=high_res_image_std,
                    input_data_format=input_data_format,
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            high_res_image = to_channel_dimension_format(
                high_res_image, data_format, input_channel_dim=input_data_format
            )

            all_images.append(image)
            all_high_res_images.append(high_res_image)

        data = {"pixel_values": all_images, "high_res_pixel_values": all_high_res_images}
        return BatchFeature(data=data, tensor_type=return_tensors)


class DeepseekVLHybridFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    min_size (`int`, *optional*, defaults to 14):
        The minimum allowed size for the resized image. Ensures that neither the height nor width
        falls below this value after resizing.
     high_res_size (`dict`, *optional*, defaults to `{"height": 1024, "width": 1024}`):
        Size of the high resolution output image after resizing. Can be overridden by the `high_res_size` parameter in the `preprocess`
        method.
    high_res_resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
        Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
        overridden by the `high_res_resample` parameter in the `preprocess` method.
    high_res_image_mean (`float` or `list[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
        Mean to use if normalizing the high resolution image. This is a float or list of floats the length of the number of
        channels in the image. Can be overridden by the `high_res_image_mean` parameter in the `preprocess` method.
    high_res_image_std (`float` or `list[float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
        Standard deviation to use if normalizing the high resolution image. This is a float or list of floats the length of the
        number of channels in the image. Can be overridden by the `high_res_image_std` parameter in the `preprocess` method.
    """

    min_size: int
    high_res_size: dict
    high_res_resample: "PILImageResampling"
    high_res_image_mean: list[float]
    high_res_image_std: list[float]


class DeepseekVLHybridImageProcessorFast(DeepseekVLImageProcessorFast):
    high_res_image_mean = OPENAI_CLIP_MEAN
    high_res_image_std = OPENAI_CLIP_STD
    high_res_size = {"height": 1024, "width": 1024}
    high_res_resample = PILImageResampling.BICUBIC
    model_input_names = ["pixel_values", "high_res_pixel_values"]

    def __init__(self, **kwargs: Unpack[DeepseekVLHybridFastImageProcessorKwargs]):
        if kwargs.get("image_mean") is None:
            background_color = (127, 127, 127)
        else:
            background_color = tuple([int(x * 255) for x in kwargs.get("image_mean")])
        if kwargs.get("high_res_image_mean") is None:
            high_res_background_color = (127, 127, 127)
        else:
            high_res_background_color = tuple(int(x * 255) for x in kwargs.get("high_res_image_mean"))
        BaseImageProcessorFast.__init__(self, **kwargs)
        self.background_color = tuple(background_color)
        self.high_res_background_color = tuple(high_res_background_color)

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        high_res_size: Optional[SizeDict] = None,
        default_to_square: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        high_res_image_mean: Optional[Union[float, list[float]]] = None,
        high_res_image_std: Optional[Union[float, list[float]]] = None,
        data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if kwargs is None:
            kwargs = {}
        if size is not None:
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if high_res_size is not None:
            high_res_size = SizeDict(**get_size_dict(size=high_res_size, default_to_square=default_to_square))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)
        if isinstance(high_res_image_mean, list):
            high_res_image_mean = tuple(high_res_image_mean)
        if isinstance(high_res_image_std, list):
            high_res_image_std = tuple(high_res_image_std)
        if data_format is None:
            data_format = ChannelDimension.FIRST

        high_res_resample = kwargs.pop("high_res_resample")
        kwargs["high_res_interpolation"] = (
            pil_torch_interpolation_mapping[high_res_resample]
            if isinstance(high_res_resample, (int, PILImageResampling))
            else high_res_resample
        )

        low_res_resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[low_res_resample]
            if isinstance(low_res_resample, (int, PILImageResampling))
            else low_res_resample
        )

        kwargs["size"] = size
        kwargs["high_res_size"] = high_res_size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        kwargs["high_res_image_mean"] = high_res_image_mean
        kwargs["high_res_image_std"] = high_res_image_std
        kwargs["data_format"] = data_format

        return kwargs

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        high_res_size: SizeDict,
        min_size: int,
        interpolation: Optional["F.InterpolationMode"],
        high_res_interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        high_res_image_mean: Optional[Union[float, list[float]]],
        high_res_image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        do_pad: bool = True,
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        high_res_resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_high_res_images = self.resize(
                    image=stacked_images, size=high_res_size, min_size=min_size, interpolation=high_res_interpolation
                )
            high_res_resized_images_grouped[shape] = stacked_high_res_images
        high_res_resized_images = reorder_images(high_res_resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_high_res_images, grouped_high_res_images_index = group_images_by_shape(
            high_res_resized_images, disable_grouping=disable_grouping
        )
        high_res_padded_images = {}
        high_res_processed_images_grouped = {}
        for shape, stacked_high_res_images in grouped_high_res_images.items():
            if do_pad:
                stacked_high_res_images = self.pad_to_square(
                    stacked_high_res_images, background_color=self.high_res_background_color
                )
                high_res_padded_images[shape] = stacked_high_res_images
            # Fused rescale and normalize
            stacked_high_res_images = self.rescale_and_normalize(
                stacked_high_res_images,
                do_rescale,
                rescale_factor,
                do_normalize,
                high_res_image_mean,
                high_res_image_std,
            )
            high_res_processed_images_grouped[shape] = stacked_high_res_images
        high_res_processed_images = reorder_images(high_res_processed_images_grouped, grouped_high_res_images_index)
        high_res_processed_images = (
            torch.stack(high_res_processed_images, dim=0) if return_tensors else high_res_processed_images
        )

        resized_images_grouped = {}
        for shape, stacked_high_res_padded_images in high_res_padded_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_high_res_padded_images, size=size, min_size=min_size, interpolation=interpolation
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_high_res_images_index)

        grouped_resized_images, grouped_resized_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping
        )
        processed_images_grouped = {}
        for shape, stacked_images in grouped_resized_images.items():
            if do_pad:
                stacked_images = self.pad_to_square(stacked_images, background_color=self.background_color)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_resized_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(
            data={"pixel_values": processed_images, "high_res_pixel_values": high_res_processed_images},
            tensor_type=return_tensors,
        )


class DeepseekVLHybridProcessorKwargs(DeepseekVLProcessorKwargs):
    pass


class DeepseekVLHybridProcessor(DeepseekVLProcessor):
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        images: ImageInput = None,
        **kwargs: Unpack[DeepseekVLHybridProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        DeepseekVLHybridImageProcessor's [`~DeepseekVLHybridImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
            `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
            `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            DeepseekVLHybridProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = []
        one_img_tokens = self.image_token * self.num_image_tokens
        for prompt in text:
            prompt = prompt.replace(self.image_token, one_img_tokens)
            prompt_strings.append(prompt)

        data = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        # process images if pixel_values are provided
        if images is not None:
            inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            data["pixel_values"] = inputs["pixel_values"]
            data["high_res_pixel_values"] = inputs["high_res_pixel_values"]

        return BatchFeature(data=data)


__all__ = [
    "DeepseekVLHybridConfig",
    "DeepseekVLHybridPreTrainedModel",
    "DeepseekVLHybridModel",
    "DeepseekVLHybridForConditionalGeneration",
    "DeepseekVLHybridImageProcessor",
    "DeepseekVLHybridImageProcessorFast",
    "DeepseekVLHybridProcessor",
]
