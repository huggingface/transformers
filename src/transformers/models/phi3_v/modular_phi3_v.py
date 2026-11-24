# Copyright 2026 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
from dataclasses import fields

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...activations import ACT2FN
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ImageInput, PILImageResampling, SizeDict
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessorMixin, Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig, AutoModel
from ..llava_next.configuration_llava_next import LlavaNextConfig
from ..llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextMultiModalProjector,
    LlavaNextPreTrainedModel,
)


@auto_docstring(checkpoint="microsoft/Phi-3.5-vision-instruct")
@strict
class Phi3VConfig(LlavaNextConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Phi3VForConditionalGeneration, Phi3VConfig

    >>> # Initializing a Phi-3.5-vision style configuration
    >>> configuration = Phi3VConfig()

    >>> # Initializing a model from the configuration
    >>> model = Phi3VForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phi3_v"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="phi3"),
        "vision_config": SubConfigSpec(
            config_class=AutoConfig,
            model_type="clip_vision_model",
            init_kwargs={
                "intermediate_size": 4096,
                "hidden_size": 1024,
                "patch_size": 14,
                "image_size": 336,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "projection_dim": 768,
            },
        ),
    }

    image_token_index: int = 32044
    image_grid_pinpoints = AttributeError()
    image_seq_length = AttributeError()

    def __post_init__(self, **kwargs):
        # Hub configs are flat Phi-3 configs with the vision settings hardcoded in the remote code. They also force
        # `flash_attention_2`, which would override `attn_implementation`
        if self.text_config is None:
            for key in ("_attn_implementation", "embd_layer", "img_processor"):
                kwargs.pop(key, None)
            text_keys = {field.name for field in fields(CONFIG_MAPPING["phi3"])} | {"rope_scaling", "rope_theta"}
            self.text_config = {key: kwargs.pop(key) for key in list(kwargs) if key in text_keys}
        PreTrainedConfig.__post_init__(self, **kwargs)


class Phi3VMultiModalProjector(LlavaNextMultiModalProjector):
    def __init__(self, config: Phi3VConfig):
        nn.Module.__init__(self)
        # Projects 2x2 merged vision patches
        self.linear_1 = nn.Linear(
            4 * config.vision_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )


class Phi3VPreTrainedModel(LlavaNextPreTrainedModel):
    # VLM2Vec-Full stores a copy of the tied input embeddings
    _keys_to_ignore_on_load_unexpected = [r"vision_embed_tokens\.wte\."]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Phi3VModel):
            init.zeros_(module.image_newline)
            init.zeros_(module.image_separator)


@auto_docstring(
    custom_intro="""
    The Phi-3-vision model which consists of a vision backbone and a language model without language modeling head.
    """
)
class Phi3VModel(LlavaNextModel):
    def __init__(self, config: Phi3VConfig):
        PreTrainedModel.__init__(self, config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = Phi3VMultiModalProjector(config)
        # Shaped like the Hub checkpoints' `sub_GN` and `glb_GN`
        self.image_newline = nn.Parameter(torch.zeros(1, 1, 1, 4 * config.vision_config.hidden_size))
        self.image_separator = nn.Parameter(torch.zeros(1, 1, 4 * config.vision_config.hidden_size))
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def pack_image_features(self, **super_kwargs):
        raise AttributeError("Not needed for Phi3V")

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.Tensor` of shape `(batch_size, num_crops, channels, height, width)`)
            The global image followed by the crops of each image, padded to the largest number of crops.
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
            Size (height, width) of each image after resizing and padding to a multiple of the crop size.
        vision_feature_layer (`int`, *optional*):
            The index of the layer to select the vision feature.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            `"default"` removes the CLS token.
        """
        crop_size = self.config.vision_config.image_size
        grids = [(height // crop_size, width // crop_size) for height, width in image_sizes.tolist()]
        num_crops = [rows * cols + 1 for rows, cols in grids]
        pixel_values = torch.cat([pixels[:num] for pixels, num in zip(pixel_values, num_crops)])

        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        image_features = image_outputs.hidden_states[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            image_features = image_features[:, 1:]

        # Merge each 2x2 block of patches into one token: (crops, side, side, 4 * hidden_size)
        side = crop_size // self.config.vision_config.patch_size // 2
        image_features = image_features.view(-1, side, 2, side, 2, image_features.shape[-1])
        image_features = image_features.permute(0, 1, 3, 2, 4, 5).flatten(3)
        newline = self.image_newline[0].to(image_features)
        separator = self.image_separator[0].to(image_features)

        packed_features = []
        for features, (rows, cols) in zip(torch.split(image_features, num_crops), grids):
            sub_image = (
                features[1:].view(rows, cols, side, side, -1).transpose(1, 2).reshape(rows * side, cols * side, -1)
            )
            sub_image = torch.cat([sub_image, newline.expand(rows * side, -1, -1)], dim=1)
            global_image = torch.cat([features[0], newline.expand(side, -1, -1)], dim=1)
            image_tokens = torch.cat([sub_image.flatten(0, 1), separator, global_image.flatten(0, 1)])
            packed_features.append(self.multi_modal_projector(image_tokens))
        image_outputs.pooler_output = packed_features

        return image_outputs


@auto_docstring(
    custom_intro="""
    The Phi-3-vision model which consists of a vision backbone and a language model.
    """
)
class Phi3VForConditionalGeneration(LlavaNextForConditionalGeneration):
    def get_image_features(self, **super_kwargs):
        r"""
        pixel_values (`torch.Tensor` of shape `(batch_size, num_crops, channels, height, width)`)
            The global image followed by the crops of each image, padded to the largest number of crops.
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
            Size (height, width) of each image after resizing and padding to a multiple of the crop size.
        vision_feature_layer (`int`, *optional*):
            The index of the layer to select the vision feature.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            `"default"` removes the CLS token.
        """
        return super().get_image_features(**super_kwargs)

    def pack_image_features(self, **super_kwargs):
        raise AttributeError("Not needed for Phi3V")

    def forward(self, **super_kwargs):
        r"""
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            `"default"` removes the CLS token.
        """
        return super().forward(**super_kwargs)


class Phi3VImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    num_crops (`int`, *optional*, defaults to `self.num_crops`):
        Maximum number of crops per image. The image is resized to fit this many crops of `size` before cropping.
    """

    num_crops: int


@auto_docstring
class Phi3VImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    size = {"height": 336, "width": 336}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    num_crops = 16
    valid_kwargs = Phi3VImageProcessorKwargs
    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(self, **kwargs: Unpack[Phi3VImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Phi3VImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def get_resized_size(self, height: int, width: int, num_crops: int, crop_size: int) -> tuple[int, int]:
        """Largest size with the same aspect ratio whose long side is a multiple of `crop_size` that fits `num_crops`."""
        long_side, short_side = max(height, width), min(height, width)
        ratio = long_side / short_side
        scale = 1
        while (scale + 1) * math.ceil((scale + 1) / ratio) <= num_crops:
            scale += 1
        new_long_side = scale * crop_size
        new_short_side = int(new_long_side / ratio)
        return (new_long_side, new_short_side) if height > width else (new_short_side, new_long_side)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        num_crops: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        crop_size = size.height
        processed_images = []
        image_sizes = []
        for image in images:
            height, width = self.get_resized_size(*image.shape[-2:], num_crops, crop_size)
            # Bilinear as in the original processor; `resample` is used for the global image
            image = tvF.resize(image, [height, width], interpolation=tvF.InterpolationMode.BILINEAR)
            # Pad the short side with white to a multiple of the crop size, centered
            pad_height = math.ceil(height / crop_size) * crop_size - height
            pad_width = math.ceil(width / crop_size) * crop_size - width
            padding = [pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2]
            image = tvF.pad(image, padding, fill=255)
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            global_image = self.resize(image, size, resample=resample, antialias=False)
            height, width = image.shape[-2:]
            crops = image.view(-1, height // crop_size, crop_size, width // crop_size, crop_size)
            crops = crops.permute(1, 3, 0, 2, 4).flatten(0, 1)
            processed_images.append(torch.cat([global_image[None], crops]))
            image_sizes.append([height, width])

        max_crops = max(len(crops) for crops in processed_images)
        pixel_values = torch.stack(
            [F.pad(crops, (0, 0, 0, 0, 0, 0, 0, max_crops - len(crops))) for crops in processed_images]
        )
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_sizes": image_sizes}, tensor_type=return_tensors
        )


@auto_docstring
class Phi3VProcessor(ProcessorMixin):
    def __init__(self, image_processor, tokenizer, chat_template=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", "<|image|>")
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def _get_num_image_tokens(self, height: int, width: int) -> int:
        crop_size = self.image_processor.size["height"]
        rows, cols = math.ceil(height / crop_size), math.ceil(width / crop_size)
        side = crop_size // 28  # 14x14 patches, merged 2x2
        # Each row of crops and of the global image ends with a newline, and a separator precedes the global image
        return rows * side * (cols * side + 1) + 1 + side * (side + 1)

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        num_image_tokens = self._get_num_image_tokens(*image_inputs["image_sizes"][image_idx])
        # The original processor tokenized the text after each image separately, starting it with BOS
        return self.image_token * num_image_tokens + self.tokenizer.bos_token

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
            images_kwargs = self._merge_kwargs(**kwargs)["images_kwargs"]
            num_crops = images_kwargs.get("num_crops", self.image_processor.num_crops)
            crop_size = self.image_processor.size["height"]
            num_image_tokens = [
                self._get_num_image_tokens(*self.image_processor.get_resized_size(*image_size, num_crops, crop_size))
                for image_size in image_sizes
            ]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": [1] * len(image_sizes)})
        return MultiModalData(**vision_data)


__all__ = [
    "Phi3VConfig",
    "Phi3VPreTrainedModel",
    "Phi3VModel",
    "Phi3VForConditionalGeneration",
    "Phi3VImageProcessor",
    "Phi3VProcessor",
]
