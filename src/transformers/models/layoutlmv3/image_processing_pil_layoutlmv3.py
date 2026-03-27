# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for LayoutLMv3."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, requires_backends
from ...utils.import_utils import requires
from .image_processing_layoutlmv3 import LayoutLMv3ImageProcessorKwargs, apply_tesseract


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class LayoutLMv3ImageProcessorPil(PilBackend):
    valid_kwargs = LayoutLMv3ImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    apply_ocr = True
    ocr_lang = None
    tesseract_config = ""

    def __init__(self, **kwargs: Unpack[LayoutLMv3ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[LayoutLMv3ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        apply_ocr: bool = True,
        ocr_lang: str | None = None,
        tesseract_config: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        # Tesseract OCR to get words + normalized bounding boxes
        if apply_ocr:
            requires_backends(self, "pytesseract")
            words_batch = []
            boxes_batch = []
            for image in images:
                words, boxes = apply_tesseract(
                    image, ocr_lang, tesseract_config, input_data_format=ChannelDimension.FIRST
                )
                words_batch.append(words)
                boxes_batch.append(boxes)

        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image=image, crop_size=crop_size)
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor)
            if do_normalize:
                image = self.normalize(image=image, mean=image_mean, std=image_std)
            processed_images.append(image)

        data = BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

        if apply_ocr:
            data["words"] = words_batch
            data["boxes"] = boxes_batch

        return data


__all__ = ["LayoutLMv3ImageProcessorPil"]
