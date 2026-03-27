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
"""Image processor class for LayoutLMv2."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import flip_channel_order
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, requires_backends
from ...utils.import_utils import requires
from .image_processing_layoutlmv2 import LayoutLMv2ImageProcessorKwargs, apply_tesseract


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class LayoutLMv2ImageProcessorPil(PilBackend):
    valid_kwargs = LayoutLMv2ImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    size = {"height": 224, "width": 224}
    rescale_factor = None
    do_resize = True
    apply_ocr = True
    ocr_lang = None
    tesseract_config = ""

    def __init__(self, **kwargs: Unpack[LayoutLMv2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[LayoutLMv2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
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

            # flip color channels from RGB to BGR (as Detectron2 requires this)
            image = flip_channel_order(image, input_data_format=ChannelDimension.FIRST)

            processed_images.append(image)

        data = BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

        if apply_ocr:
            data["words"] = words_batch
            data["boxes"] = boxes_batch

        return data


__all__ = ["LayoutLMv2ImageProcessorPil"]
