# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""
Feature extractor class for LayoutLMv2.
"""

from typing import List, Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import ImageFeatureExtractionMixin, is_torch_tensor
from ...utils import TensorType, is_pytesseract_available, logging, requires_backends


# soft dependency
if is_pytesseract_available():
    import pytesseract

logger = logging.get_logger(__name__)

ImageInput = Union[
    Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
]


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def apply_tesseract(image: Image.Image, lang: Optional[str], tesseract_config: Optional[str]):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""

    # apply OCR
    data = pytesseract.image_to_data(image, lang=lang, output_type="dict", config=tesseract_config)
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

    image_width, image_height = image.size

    # finally, normalize the bounding boxes
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    return words, normalized_boxes


class LayoutLMv2FeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a LayoutLMv2 feature extractor. This can be used to resize document images to the same size, as well as
    to apply OCR on them in order to get a list of words and normalized bounding boxes.

    This feature extractor inherits from [`~feature_extraction_utils.PreTrainedFeatureExtractor`] which contains most
    of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int` or `Tuple(int)`, *optional*, defaults to 224):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if `do_resize` is
            set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        apply_ocr (`bool`, *optional*, defaults to `True`):
            Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes.
        ocr_lang (`str`, *optional*):
            The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
            used.
        tesseract_config (`str`, *optional*):
            Any additional custom configuration flags that are forwarded to the `config` parameter when calling
            Tesseract. For example: '--psm 6'.

            <Tip>

            LayoutLMv2FeatureExtractor uses Google's Tesseract OCR engine under the hood.

            </Tip>"""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Image.BILINEAR,
        apply_ocr=True,
        ocr_lang=None,
        tesseract_config="",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.apply_ocr = apply_ocr
        self.ocr_lang = ocr_lang
        self.tesseract_config = tesseract_config

    def __call__(
        self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
            - **words** -- Optional words as identified by Tesseract OCR (only when [`LayoutLMv2FeatureExtractor`] was
              initialized with `apply_ocr` set to `True`).
            - **boxes** -- Optional bounding boxes as identified by Tesseract OCR, normalized based on the image size
              (only when [`LayoutLMv2FeatureExtractor`] was initialized with `apply_ocr` set to `True`).

        Examples:

        ```python
        >>> from transformers import LayoutLMv2FeatureExtractor
        >>> from PIL import Image

        >>> image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")

        >>> # option 1: with apply_ocr=True (default)
        >>> feature_extractor = LayoutLMv2FeatureExtractor()
        >>> encoding = feature_extractor(image, return_tensors="pt")
        >>> print(encoding.keys())
        >>> # dict_keys(['pixel_values', 'words', 'boxes'])

        >>> # option 2: with apply_ocr=False
        >>> feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
        >>> encoding = feature_extractor(image, return_tensors="pt")
        >>> print(encoding.keys())
        >>> # dict_keys(['pixel_values'])
        ```"""

        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples), "
                f"but is of type {type(images)}."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # Tesseract OCR to get words + normalized bounding boxes
        if self.apply_ocr:
            requires_backends(self, "pytesseract")
            words_batch = []
            boxes_batch = []
            for image in images:
                words, boxes = apply_tesseract(self.to_pil_image(image), self.ocr_lang, self.tesseract_config)
                words_batch.append(words)
                boxes_batch.append(boxes)

        # transformations (resizing)
        if self.do_resize and self.size is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]

        images = [self.to_numpy_array(image, rescale=False) for image in images]
        # flip color channels from RGB to BGR (as Detectron2 requires this)
        images = [image[::-1, :, :] for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        if self.apply_ocr:
            encoded_inputs["words"] = words_batch
            encoded_inputs["boxes"] = boxes_batch

        return encoded_inputs
