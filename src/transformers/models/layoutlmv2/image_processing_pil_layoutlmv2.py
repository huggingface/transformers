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

from typing import TYPE_CHECKING

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import flip_channel_order, to_pil_image
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, requires_backends


if TYPE_CHECKING:
    pass

try:
    import pytesseract
except ImportError:
    pytesseract = None


# Adapted from transformers.models.layoutlmv2.image_processing_layoutlmv2.LayoutLMv2ImageProcessorKwargs
class LayoutLMv2ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    apply_ocr (`bool`, *optional*, defaults to `self.apply_ocr`):
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

    apply_ocr: bool
    ocr_lang: str | None
    tesseract_config: str | None


# Adapted from transformers.models.layoutlmv2.image_processing_layoutlmv2.normalize_box
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


# Adapted from transformers.models.layoutlmv2.image_processing_layoutlmv2.apply_tesseract
def apply_tesseract(
    image: np.ndarray,
    lang: str | None,
    tesseract_config: str | None = None,
    input_data_format: str | ChannelDimension | None = None,
):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    requires_backends(apply_tesseract, ["pytesseract"])

    # Convert torch tensor to numpy if needed
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    elif not isinstance(image, np.ndarray):
        image = np.array(image)

    tesseract_config = tesseract_config if tesseract_config is not None else ""

    # apply OCR
    pil_image = to_pil_image(image, input_data_format=input_data_format)
    image_width, image_height = pil_image.size
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type="dict", config=tesseract_config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # filter empty words and corresponding coordinates
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # turn coordinates into (left, top, left+width, top+height) format
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]
        actual_boxes.append(actual_box)

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    return words, normalized_boxes


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
        resample: "PILImageResampling | None",
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
