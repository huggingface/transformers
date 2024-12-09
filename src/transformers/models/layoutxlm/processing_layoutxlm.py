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
Processor class for LayoutXLM.
"""

import sys
import warnings
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, TextKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class LayoutXLMTextKwargs(TextKwargs, total=False):
    boxes: Optional[Union[List[List[int]], List[List[List[int]]]]]
    word_labels: Optional[Union[List[int], List[List[int]]]]


class LayoutXLMImagesKwargs(ImagesKwargs, total=False):
    apply_ocr: bool
    ocr_lang: Optional[str]
    tesseract_config: Optional[str]


class LayoutXLMProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: LayoutXLMTextKwargs
    images_kwargs: LayoutXLMImagesKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {
            "apply_ocr": True,
        },
    }


class LayoutXLMProcessor(ProcessorMixin):
    r"""
    Constructs a LayoutXLM processor which combines a LayoutXLM image processor and a LayoutXLM tokenizer into a single
    processor.

    [`LayoutXLMProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv2ImageProcessor`] to resize document images to a fixed size, and optionally applies OCR to
    get words and normalized bounding boxes. These are then provided to [`LayoutXLMTokenizer`] or
    [`LayoutXLMTokenizerFast`], which turns the words and bounding boxes into token-level `input_ids`,
    `attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
    into token-level `labels` for token classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`LayoutLMv2ImageProcessor`, *optional*):
            An instance of [`LayoutLMv2ImageProcessor`]. The image processor is a required input.
        tokenizer (`LayoutXLMTokenizer` or `LayoutXLMTokenizerFast`, *optional*):
            An instance of [`LayoutXLMTokenizer`] or [`LayoutXLMTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LayoutLMv2ImageProcessor"
    tokenizer_class = ("LayoutXLMTokenizer", "LayoutXLMTokenizerFast")
    optional_call_args = ["text_pair", "boxes", "word_labels"]

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: ImageInput,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        # The following is to capture `text_pair`, `boxes`, `word_labels` arguments that may be passed as a positional argument.
        # See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details.
        # This behavior is only needed for backward compatibility and will be removed in future versions.
        *args,
        audio=None,
        videos=None,
        **kwargs: Unpack[LayoutXLMProcessorKwargs],
    ) -> BatchFeature:
        """
        This method first forwards the `images` argument to [`~LayoutLMv2ImagePrpcessor.__call__`]. In case
        [`LayoutLMv2ImageProcessor`] was initialized with `apply_ocr` set to `True`, it passes the obtained words and
        bounding boxes along with the additional arguments to [`~LayoutXLMTokenizer.__call__`] and returns the output,
        together with resized `images`. In case [`LayoutLMv2ImagePrpcessor`] was initialized with `apply_ocr` set to
        `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along with the additional
        arguments to [`~LayoutXLMTokenizer.__call__`] and returns the output, together with resized `images``.

        Please refer to the docstring of the above two methods for more information.

        Args:
            images (`ImageInput`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **image** -- Pixel values to be fed to a model.
            - **bbox** -- Bounding boxes of the words in the image.
        """
        output_kwargs = self._merge_kwargs(
            LayoutXLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
            **self.prepare_and_validate_optional_call_args(*args),
        )

        text_pair = output_kwargs["text_kwargs"].pop("text_pair", None)
        boxes = output_kwargs["text_kwargs"].pop("boxes", None)
        word_labels = output_kwargs["text_kwargs"].pop("word_labels", None)
        apply_ocr = output_kwargs["images_kwargs"].get("apply_ocr", self.image_processor.apply_ocr)

        # verify input
        if apply_ocr and (boxes is not None):
            raise ValueError(
                "You cannot provide bounding boxes "
                "if you initialized the image processor with apply_ocr set to True."
            )

        if apply_ocr and (word_labels is not None):
            raise ValueError(
                "You cannot provide word labels if you initialized the image processor with apply_ocr set to True."
            )

        if (
            output_kwargs["text_kwargs"]["return_overflowing_tokens"]
            and not output_kwargs["text_kwargs"]["return_offsets_mapping"]
        ):
            raise ValueError("You cannot return overflowing tokens without returning the offsets mapping.")

        # first, apply the image processor
        features = self.image_processor(images=images, **output_kwargs["images_kwargs"])

        # second, apply the tokenizer
        if text is not None and apply_ocr and text_pair is None:
            if isinstance(text, str):
                text = [text]  # add batch dimension (as the image processor always adds a batch dimension)
            text_pair = features["words"]

        if text is None:
            if not hasattr(features, "words"):
                raise ValueError("You need to provide `text` or set `apply_ocr` to `True`")
            text = features["words"]
        if boxes is None:
            if not hasattr(features, "boxes"):
                raise ValueError("You need to provide `boxes` or set `apply_ocr` to `True`")
            boxes = features["boxes"]

        encoded_inputs = self.tokenizer(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            **output_kwargs["text_kwargs"],
        )

        # add pixel values
        images = features.pop("pixel_values")
        if output_kwargs["text_kwargs"]["return_overflowing_tokens"]:
            images = self.get_overflowing_images(images, encoded_inputs["overflow_to_sample_mapping"])
        encoded_inputs["image"] = images

        return BatchFeature(
            data=dict(**encoded_inputs), tensor_type=output_kwargs["common_kwargs"].get("return_tensors")
        )

    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # in case there's an overflow, ensure each `input_ids` sample is mapped to its corresponding image
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        return images_with_overflow

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "image"]

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
