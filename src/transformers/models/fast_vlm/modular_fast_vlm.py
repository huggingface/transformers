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

import math
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from torch.nn.functional import adaptive_avg_pool2d

from ...activations import ACT2FN
from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING
from ..llava.configuration_llava import LlavaConfig
from ..llava.modeling_llava import (
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaMultiModalProjector,
    LlavaPreTrainedModel,
)


class FastVlmConfig(LlavaConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastVLMForConditionalGeneration`]. It is used to instantiate a
    FastVLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield the same configurationa as the one of FastVLM-7B.

    e.g. [KamilaMila/FastVLM-7B](https://huggingface.co/KamilaMila/FastVLM-7B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `TimmWrapperConfig` for `fastvit_mci3`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"full"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can only be `"full"`.
        vision_feature_layer (`Union[int, list[int]]`, *optional*, defaults to -1):
            The index of the layer to select the vision feature. If multiple indices are provided,
            the vision feature of the corresponding indices will be concatenated to form the
            vision features. Must be negative.
        image_seq_length (`int`, *optional*, defaults to 256):
            Sequence length of one image embedding.
        multimodal_projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.

    Example:

    ```python
    >>> from transformers import FastVlmForConditionalGeneration, FastVlmConfig

    >>> # Initializing a FastVLM-7B style configuration
    >>> configuration = FastVlmConfig()

    >>> # Initializing a model from the FastVLM-7B style configuration
    >>> model = FastVlmForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fast_vlm"

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=151646,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="full",
        vision_feature_layer=-1,
        image_seq_length=256,
        multimodal_projector_bias=True,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length
        if math.isqrt(image_seq_length) ** 2 != image_seq_length:
            raise ValueError(f"Inavalid image_seq_length: {image_seq_length}. It needs to be a perfect square.")

        if vision_feature_select_strategy != "full":
            raise ValueError(
                "Only vision_feature_select_strategy='full' supported in FastVLM!"
                f"Got: {vision_feature_select_strategy}"
            )

        if any(
            layer >= 0
            for layer in (
                vision_feature_layer if isinstance(vision_feature_layer, Iterable) else [vision_feature_layer]
            )
        ):
            raise ValueError(f"Only negative layer values are supported. Got {vision_feature_layer}")

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if isinstance(vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "timm_wrapper")
            vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            vision_config = CONFIG_MAPPING["timm_wrapper"](
                architecture="fastvit_mci3",
                do_pooling=True,
                global_pool="avg",
                hidden_size=3072,
                initializer_range=0.02,
                model_args={"inference_mode": True},
            )

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "qwen2")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"](
                hidden_size=3584,
                vocab_size=152128,
                intermediate_size=18944,
                num_attention_heads=28,
                num_key_value_heads=4,
                num_hidden_layers=28,
            )

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias

        PretrainedConfig.__init__(**kwargs)


class FastVlmMultiModalProjector(LlavaMultiModalProjector):
    def __init__(self, config: FastVlmConfig):
        nn.Module.__init__()
        if isinstance(config.vision_feature_layer, int):
            layers = [config.vision_feature_layer]
        else:
            layers = config.vision_feature_layer
        #  different layers have different hidden sizes that are concatenated
        total_hidden_size = 0
        for layer in layers:
            total_hidden_size += config.vision_config.hidden_size // (2 ** (-layer - 1))
        self.linear_1 = nn.Linear(
            total_hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )


class FastVlmPreTrainedModel(LlavaPreTrainedModel):
    pass


class FastVlmModel(LlavaModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: FastVlmConfig):
        # Timm models don't support this way of setting attention mode so we set the vision config to eager while keeping the language part
        # the same as the user requested
        config.vision_config._attn_implementation = "eager"
        super().__init__(config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, list[int]]`, *optional*):
                The index/indices of the layer to select the vision feature. Must be negative.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Only "full" supported.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # only this value makes sense in FastVLM (we can't have a CLS token in conv layers)
        if vision_feature_select_strategy != "full":
            raise ValueError(
                f"Unexpected select feature strategy: {vision_feature_select_strategy}, Only 'full' is supported in FastVLM."
            )

        if any(
            layer >= 0
            for layer in (
                vision_feature_layer if isinstance(vision_feature_layer, Iterable) else [vision_feature_layer]
            )
        ):
            raise ValueError(f"Only negative layer values are supported. Got {vision_feature_layer}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory-efficient at all
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

        # since the vision tower is hybrid in FastVLM, its output needs to be handled differently from Llava
        desired_shape = math.isqrt(self.config.image_seq_length)
        if isinstance(vision_feature_layer, int):
            if vision_feature_layer == -1:
                selected_image_feature = image_outputs.last_hidden_state
            else:
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer + 1]
            selected_image_feature = adaptive_avg_pool2d(selected_image_feature, (desired_shape, desired_shape))
        else:
            hs_pool = []
            for layer_idx in vision_feature_layer:
                if layer_idx == -1:
                    selected_image_feature = image_outputs.last_hidden_state
                else:
                    selected_image_feature = image_outputs.hidden_states[layer_idx + 1]
                selected_image_feature = adaptive_avg_pool2d(selected_image_feature, (desired_shape, desired_shape))
                hs_pool.append(selected_image_feature)
            selected_image_feature = torch.cat(hs_pool, dim=-3)

        selected_image_feature = selected_image_feature.flatten(2).permute(0, 2, 1)
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = list(image_features)
        return image_features

    def forward(self, **super_kwargs):
        r"""
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone. Can only be `"full"`.

        vision_feature_layer (`Union[int, list[int], NoneType]`, *optional*):
            The index of the layer to select the vision feature. If multiple indices are provided, the vision feature of the
            corresponding indices will be concatenated to form the vision features. Must be negative.
        """
        super().forward(**super_kwargs)


class FastVlmForConditionalGeneration(LlavaForConditionalGeneration):
    def forward(self, **super_kwargs):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone. Can only be `"full"`.

        vision_feature_layer (`Union[int, list[int], NoneType]`, *optional*):
            The index of the layer to select the vision feature. If multiple indices are provided, the vision feature of the
            corresponding indices will be concatenated to form the vision features. Must be negative.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, FastVlmForConditionalGeneration

        >>> model = FastVlmForConditionalGeneration.from_pretrained("KamilaMila/FastVLM-7B")
        >>> processor = AutoProcessor.from_pretrained("KamilaMila/FastVLM-7B")

        >>> prompt = "<|im_start|>user\n<image>\nWhat's the content of the image?<|im_end|>\n<|im_start|>assistant\n"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""
        super().forward(**super_kwargs)


__all__ = ["FastVlmForConditionalGeneration", "FastVlmModel", "FastVlmPreTrainedModel", "FastVlmConfig"]
