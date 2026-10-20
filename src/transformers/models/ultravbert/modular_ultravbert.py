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

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torch.nn import CrossEntropyLoss
from torchvision.transforms.v2 import functional as tvF

from ... import initialization as init
from ...configuration_utils import SubConfigSpec
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TensorType, TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple
from ..auto import AutoConfig
from ..idefics3.image_processing_idefics3 import (
    MAX_IMAGE_SIZE,
    Idefics3ImageProcessor,
    Idefics3ImageProcessorKwargs,
    _resize_output_size_rescale_to_max_len,
    _resize_output_size_scale_below_upper_bound,
)
from ..idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from ..modernvbert.configuration_modernvbert import ModernVBertConfig
from ..modernvbert.modeling_modernvbert import (
    ModernVBertBaseModelOutput,
    ModernVBertConnector,
    ModernVBertForMaskedLM,
    ModernVBertForSequenceClassification,
    ModernVBertForTokenClassification,
    ModernVBertMaskedLMOutput,
    ModernVBertModel,
    ModernVBertPreTrainedModel,
)
from ..ultrabert.modeling_ultrabert import UltrabertPredictionHead, UltrabertRMSNorm


@auto_docstring(checkpoint="athrael-soju/UltraVBERT")
@strict
class UltraVBertConfig(ModernVBertConfig):
    r"""
    pixel_shuffle_factor (`int`, *optional*, defaults to 4):
        Side of the square of vision patches that the connector merges into one image token.
    initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
        The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
    classifier_pooling (`Literal["cls", "mean"]`, *optional*, defaults to `"cls"`):
        The pooling strategy to use for classification tasks.
    """

    model_type = "ultravbert"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="ultrabert"),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="siglip_vision_model"),
    }

    image_token_id: int = 49191
    classifier_bias = AttributeError()


def get_capped_longest_edge(height: int, width: int, longest_edge: int, tile_size: int, max_image_tiles: int) -> int:
    for edge in range(longest_edge, tile_size - 1, -tile_size):
        resized_height, resized_width = _resize_output_size_rescale_to_max_len(height, width, max_len=edge)
        resized_height, resized_width = _resize_output_size_scale_below_upper_bound(
            resized_height, resized_width, max_len=MAX_IMAGE_SIZE
        )
        aspect_ratio = resized_width / resized_height
        if resized_width >= resized_height:
            cols = math.ceil(resized_width / tile_size)
            rows = math.ceil(int(cols * tile_size / aspect_ratio) / tile_size)
        else:
            rows = math.ceil(resized_height / tile_size)
            cols = math.ceil(int(rows * tile_size * aspect_ratio) / tile_size)
        if rows * cols <= max_image_tiles:
            return edge
    return tile_size


class UltraVBertImageProcessorKwargs(Idefics3ImageProcessorKwargs, total=False):
    r"""
    do_image_splitting (`bool`, *optional*, defaults to `True`):
        Whether to split the image into sub-images concatenated with the original image. They are split into patches
        such that each patch has a size of `max_image_size["height"]` x `max_image_size["width"]`.
    max_image_size (`Dict`, *optional*, defaults to `{"longest_edge": 364}`):
        Maximum resolution of the patches of images accepted by the model. This is a dictionary containing the key "longest_edge".
    return_row_col_info (`bool`, *optional*, defaults to `False`):
        Whether to return the row and column information of the images.
    max_image_tiles (`int`, *optional*):
        Largest number of tiles an image splits into, the global image excluded. An image that would split into more
        is resized to a longest edge lowered in steps of `max_image_size["longest_edge"]` until it fits.
    """

    max_image_tiles: int | None


class UltraVBertImageProcessor(Idefics3ImageProcessor):
    max_image_tiles = None
    valid_kwargs = UltraVBertImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[UltraVBertImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[UltraVBertImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        do_image_splitting: bool | None,
        max_image_size: dict[str, int] | None,
        return_row_col_info: bool | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        max_image_tiles: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if do_resize and do_image_splitting and max_image_tiles is not None:
            tile_size = max_image_size["longest_edge"]
            images = [
                [
                    self.resize(
                        image,
                        SizeDict(
                            longest_edge=get_capped_longest_edge(
                                *image.shape[-2:], size.longest_edge, tile_size, max_image_tiles
                            )
                        ),
                        resample=resample,
                    )
                    for image in batch_images
                ]
                for batch_images in images
            ]
            do_resize = False
        return super()._preprocess(
            images,
            do_resize,
            size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            do_pad,
            do_image_splitting,
            max_image_size,
            return_row_col_info,
            disable_grouping,
            return_tensors,
            **kwargs,
        )

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> tuple[int, int, int]:
        images_kwargs = dict(images_kwargs or {})
        max_image_tiles = images_kwargs.get("max_image_tiles", self.max_image_tiles)
        if max_image_tiles is not None:
            size = images_kwargs.get("size", self.size)
            tile_size = images_kwargs.get("max_image_size", self.max_image_size)["longest_edge"]
            longest_edge = get_capped_longest_edge(height, width, size["longest_edge"], tile_size, max_image_tiles)
            images_kwargs["size"] = {"longest_edge": longest_edge}
        return super().get_number_of_image_patches(height, width, images_kwargs)


class UltraVBertImageProcessorPil(Idefics3ImageProcessorPil):
    max_image_tiles = None
    valid_kwargs = UltraVBertImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[UltraVBertImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[UltraVBertImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[list[np.ndarray]],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        do_image_splitting: bool | None,
        max_image_size: dict[str, int] | None,
        return_row_col_info: bool | None,
        return_tensors: str | TensorType | None,
        max_image_tiles: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if do_resize and do_image_splitting and max_image_tiles is not None:
            tile_size = max_image_size["longest_edge"]
            images = [
                [
                    self.resize(
                        image,
                        SizeDict(
                            longest_edge=get_capped_longest_edge(
                                *image.shape[-2:], size.longest_edge, tile_size, max_image_tiles
                            )
                        ),
                        resample=resample,
                    )
                    for image in batch_images
                ]
                for batch_images in images
            ]
            do_resize = False
        return super()._preprocess(
            images,
            do_resize,
            size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            do_pad,
            do_image_splitting,
            max_image_size,
            return_row_col_info,
            return_tensors,
            **kwargs,
        )

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> tuple[int, int, int]:
        images_kwargs = dict(images_kwargs or {})
        max_image_tiles = images_kwargs.get("max_image_tiles", self.max_image_tiles)
        if max_image_tiles is not None:
            size = images_kwargs.get("size", self.size)
            tile_size = images_kwargs.get("max_image_size", self.max_image_size)["longest_edge"]
            longest_edge = get_capped_longest_edge(height, width, size["longest_edge"], tile_size, max_image_tiles)
            images_kwargs["size"] = {"longest_edge": longest_edge}
        return super().get_number_of_image_patches(height, width, images_kwargs)


@auto_docstring
@dataclass
class UltraVBertBaseModelOutput(ModernVBertBaseModelOutput):
    pass


@auto_docstring
@dataclass
class UltraVBertMaskedLMOutput(ModernVBertMaskedLMOutput):
    pass


class UltraVBertRMSNorm(UltrabertRMSNorm):
    pass


class UltraVBertConnector(ModernVBertConnector):
    def __init__(self, config):
        super().__init__(config)
        # UltraBERT applies no norm to its input embeddings, so this norm sets the image tokens' scale.
        self.norm = UltraVBertRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.pixel_shuffle_factor)
        return self.norm(self.modality_projection(image_hidden_states))


@auto_docstring
class UltraVBertPreTrainedModel(ModernVBertPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)

        def init_weight(module: nn.Module, std: float):
            cutoff_factor = getattr(self.config, "initializer_cutoff_factor", 2.0)
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.bias is not None:
                    init.zeros_(module.bias)

        if isinstance(module, UltraVBertConnector):
            out_std = self.config.initializer_range / math.sqrt(2.0 * self.config.text_config.num_hidden_layers)
            init_weight(module.modality_projection, out_std)
        elif isinstance(module, UltraVBertForMaskedLM):
            out_std = self.config.initializer_range / math.sqrt(2.0 * self.config.text_config.num_hidden_layers)
            init_weight(module.head.decoder, out_std)
        elif isinstance(module, (UltraVBertForSequenceClassification, UltraVBertForTokenClassification)):
            final_out_std = self.config.initializer_range / math.sqrt(self.config.text_config.hidden_size)
            init_weight(module.classifier, final_out_std)


@auto_docstring
class UltraVBertModel(ModernVBertModel):
    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | UltraVBertBaseModelOutput:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )


class UltraVBertPredictionHead(UltrabertPredictionHead):
    pass


@auto_docstring
class UltraVBertForMaskedLM(ModernVBertForMaskedLM):
    _tied_weights_keys = {"head.decoder.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.head = UltraVBertPredictionHead(config.text_config)
        del self.projection_head
        del self.lm_head

    def get_output_embeddings(self):
        return self.head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.head.decoder = new_embeddings

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | UltraVBertMaskedLMOutput:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        logits = self.head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits.view(-1, self.vocab_size), labels.view(-1))

        return UltraVBertMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


class UltraVBertForSequenceClassification(ModernVBertForSequenceClassification):
    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.head = nn.Identity()

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SequenceClassifierOutput:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            labels=labels,
            **kwargs,
        )


class UltraVBertForTokenClassification(ModernVBertForTokenClassification):
    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.head = nn.Identity()

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TokenClassifierOutput:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        image_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The hidden states of the image encoder after modality projection.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            labels=labels,
            **kwargs,
        )


__all__ = [
    "UltraVBertConfig",
    "UltraVBertImageProcessor",
    "UltraVBertImageProcessorPil",
    "UltraVBertPreTrainedModel",
    "UltraVBertModel",
    "UltraVBertForMaskedLM",
    "UltraVBertForSequenceClassification",
    "UltraVBertForTokenClassification",
]
