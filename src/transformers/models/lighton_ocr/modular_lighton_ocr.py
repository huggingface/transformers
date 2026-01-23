# Copyright 2026 The LightOn Team and The HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import numpy as np
import torch
from torch import nn

from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration,
    Mistral3Model,
    Mistral3ModelOutputWithPast,
    Mistral3MultiModalProjector,
)
from ..pixtral.image_processing_pixtral import get_resize_output_image_size


class LightOnOcrConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LightOnOcrForConditionalGeneration`]. It is used to instantiate a
    LightOnOcr model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration with the defaults will yield
    a similar configuration to that of the LightOnOcr [lightonocr-hf/lightonocr-9b](https://huggingface.co/lightonocr-hf/lightonocr-9b) architecture.

    Args:
        spatial_merge_size (`int`, *optional*, defaults to 2):
            The size of spatial merging for image patches.
        image_token_id (`int`, *optional*, defaults to 151655):
            The id of the image token in the vocabulary.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        vision_config (`dict` or `LightOnOcrVisionConfig`, *optional*):
            Custom vision configuration or dictionary with vision configuration values.
        text_config (`dict` or `LightOnOcrTextConfig`, *optional*):
            Custom text configuration or dictionary with text configuration values.

    Example:

    ```python
    >>> from transformers import LightOnOcrConfig, LightOnOcrForConditionalGeneration

    >>> # Initializing a LightOnOcr configuration
    >>> configuration = LightOnOcrConfig()

    >>> # Initializing a model from the configuration
    >>> model = LightOnOcrForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "lighton_ocr"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        spatial_merge_size: int = 2,
        image_token_id: int = 151655,
        tie_word_embeddings: bool = True,
        vision_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings

        if vision_config is None:
            self.vision_config = CONFIG_MAPPING["pixtral"](
                attention_dropout=0,
                head_dim=64,
                hidden_act="silu",
                hidden_size=1024,
                image_size=1540,
                initializer_range=0.02,
                intermediate_size=4096,
                model_type="pixtral",
                num_attention_heads=16,
                num_channels=3,
                num_hidden_layers=24,
                patch_size=14,
                rope_theta=10000,
            )
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        else:
            vision_config["model_type"] = vision_config.get("model_type", "pixtral")
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)

        if text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"](
                attention_dropout=0,
                head_dim=128,
                hidden_act="silu",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=3072,
                max_position_embeddings=40960,
                num_attention_heads=16,
                num_hidden_layers=28,
                num_key_value_heads=8,
                rms_norm_eps=1e-6,
                rope_theta=1000000,
                sliding_window=None,
                use_cache=True,
                vocab_size=151936,
            )
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        else:
            text_config["model_type"] = text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)

        super().__init__(**kwargs)


class LightOnOcrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


class LightOnOcrProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        chat_template=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        # Calculate effective patch size for image processing
        self.effective_patch_size = patch_size * spatial_merge_size

        # Get special tokens and IDs directly from tokenizer attributes
        self.image_token = tokenizer.image_token
        self.image_break_token = tokenizer.image_break_token
        self.image_end_token = tokenizer.image_end_token
        self.image_token_id = tokenizer.image_token_id
        self.image_break_token_id = tokenizer.image_break_token_id
        self.image_end_token_id = tokenizer.image_end_token_id

        self.image_ids = [self.image_token_id, self.image_break_token_id, self.image_end_token_id]

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[LightOnOcrProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You must provide either text or images")
        output_kwargs = self._merge_kwargs(
            LightOnOcrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        if image_inputs.get("pixel_values") is not None:
            image_sizes_iter = iter(image_inputs["image_sizes"])
            prompt_strings = []

            for sample in text:
                replace_strings = []

                while self.image_token in sample:
                    image_height, image_width = next(image_sizes_iter)
                    num_height_tokens = image_height // self.effective_patch_size
                    num_width_tokens = image_width // self.effective_patch_size
                    num_patches = num_height_tokens * num_width_tokens

                    replace_str = self.image_token * num_patches
                    replace_strings.append(replace_str)

                    sample = sample.replace(self.image_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    replace_str = replace_strings.pop(0)
                    sample = sample.replace("<placeholder>", replace_str, 1)

                prompt_strings.append(sample)
        else:
            prompt_strings = text

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[np.isin(array_ids, self.image_ids)] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = LightOnOcrProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            size = images_kwargs.get("size", None) or self.image_processor.size

            num_image_tokens = []
            for height, width in image_sizes:
                resized_height, resized_width = get_resize_output_image_size(
                    np.zeros((height, width, 3)),
                    size=(size["longest_edge"], size["longest_edge"]),
                    patch_size=(self.effective_patch_size, self.effective_patch_size),
                )
                num_height_tokens = resized_height // self.effective_patch_size
                num_width_tokens = resized_width // self.effective_patch_size
                num_image_tokens.append(num_width_tokens * num_height_tokens)

            num_image_patches = [1] * len(image_sizes)
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)


class LightOnOcrMultiModalProjector(Mistral3MultiModalProjector):
    def __init__(self, config: LightOnOcrConfig):
        self.config = config
        super().__init__()
        self.act = nn.GELU()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)
        del self.num_feature_layers


class LightOnOcrModelOutputWithPast(Mistral3ModelOutputWithPast):
    pass


class LightOnOcrModel(Mistral3Model):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: LightOnOcrConfig):
        PreTrainedModel.__init__(self, config)
        self.vision_encoder = AutoModel.from_config(config.vision_config)
        self.vision_projection = LightOnOcrMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def get_image_features(self, pixel_values: torch.Tensor, image_sizes: torch.Tensor | list):
        """
        Obtains image features from the vision encoder and projection.

        Args:
            pixel_values: Image tensors
            image_sizes: Tensor or list of (height, width) pairs for each image

        Returns:
            List of image feature tensors, one per image
        """
        visual_features = self.vision_encoder(pixel_values, image_sizes=image_sizes).last_hidden_state

        image_features = self.vision_projection(visual_features.squeeze(0), image_sizes)

        # Split features per image based on the effective patch size
        downsample_ratio = self.config.vision_config.patch_size * self.config.spatial_merge_size
        split_sizes = [(height // downsample_ratio) * (width // downsample_ratio) for height, width in image_sizes]
        image_features = torch.split(image_features, split_sizes)

        return image_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        image_sizes: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | LightOnOcrModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values=pixel_values, image_sizes=image_sizes)
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Ensure position_ids has batch dimension for flash attention compatibility
        if position_ids is not None and position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

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

        return LightOnOcrModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class LightOnOcrForConditionalGeneration(Mistral3ForConditionalGeneration):
    _checkpoint_conversion_mapping = {}

    def get_image_features(self, pixel_values: torch.FloatTensor, image_sizes: torch.Tensor, **kwargs):
        return self.model.get_image_features(pixel_values=pixel_values, image_sizes=image_sizes)


__all__ = [
    "LightOnOcrPreTrainedModel",  # noqa
    "LightOnOcrForConditionalGeneration",
    "LightOnOcrModel",
    "LightOnOcrConfig",
    "LightOnOcrProcessor",
]
