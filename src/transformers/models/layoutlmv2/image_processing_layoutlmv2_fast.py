# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for LayoutLMv2."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import BaseImageProcessorFast, BatchFeature, DefaultFastImageProcessorKwargs
from ...image_transforms import ChannelDimension, group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
    requires_backends,
)
from .image_processing_layoutlmv2 import apply_tesseract


logger = logging.get_logger(__name__)


class LayoutLMv2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
        apply_ocr (`bool`, *optional*, defaults to `True`):
            Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
            the `apply_ocr` parameter in the `preprocess` method.
        ocr_lang (`str`, *optional*):
            The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
            used. Can be overridden by the `ocr_lang` parameter in the `preprocess` method.
        tesseract_config (`str`, *optional*):
            Any additional custom configuration flags that are forwarded to the `config` parameter when calling
            Tesseract. For example: '--psm 6'. Can be overridden by the `tesseract_config` parameter in the
            `preprocess` method.
    """

    apply_ocr: Optional[bool]
    ocr_lang: Optional[str]
    tesseract_config: Optional[str]


@auto_docstring
class LayoutLMv2ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    size = {"height": 224, "width": 224}
    rescale_factor = None
    do_resize = True
    apply_ocr = True
    ocr_lang = None
    tesseract_config = ""
    valid_kwargs = LayoutLMv2FastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[LayoutLMv2FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[LayoutLMv2FastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        apply_ocr: bool,
        ocr_lang: Optional[str],
        tesseract_config: Optional[str],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        # Tesseract OCR to get words + normalized bounding boxes
        if apply_ocr:
            requires_backends(self, "pytesseract")
            words_batch = []
            boxes_batch = []
            for image in images:
                if image.is_cuda:
                    logger.warning_once(
                        "apply_ocr can only be performed on cpu. Tensors will be transferred to cpu before processing."
                    )
                words, boxes = apply_tesseract(
                    image.cpu(), ocr_lang, tesseract_config, input_data_format=ChannelDimension.FIRST
                )
                words_batch.append(words)
                boxes_batch.append(boxes)

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # flip color channels from RGB to BGR (as Detectron2 requires this)
            stacked_images = stacked_images.flip(1)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        data = BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

        if apply_ocr:
            data["words"] = words_batch
            data["boxes"] = boxes_batch

        return data


__all__ = ["LayoutLMv2ImageProcessorFast"]
