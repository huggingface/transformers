# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
Image/Text processor class for FLAVA
"""

import sys
import warnings
from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class FlavaImagesKwargs(ImagesKwargs, total=False):
    return_image_mask: Optional[bool]
    return_codebook_pixels: Optional[bool]


class FlavaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: FlavaImagesKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "verbose": True,
        },
    }


class FlavaProcessor(ProcessorMixin):
    r"""
    Constructs a FLAVA processor which wraps a FLAVA image processor and a FLAVA tokenizer into a single processor.

    [`FlavaProcessor`] offers all the functionalities of [`FlavaImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~FlavaProcessor.__call__`] and [`~FlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`FlavaImageProcessor`], *optional*): The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*): The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FlavaImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
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
        self.current_processor = self.image_processor

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        **kwargs: Unpack[FlavaProcessorKwargs],
    ):
        """
        This method uses [`FlavaImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        output_kwargs = self._merge_kwargs(
            FlavaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is not None:
            encoding = self.tokenizer(text=text, **output_kwargs["text_kwargs"])
        if images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])

        if text is not None and images is not None:
            encoding.update(image_features)
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(
                data=dict(**image_features), tensor_type=output_kwargs["common_kwargs"]["return_tensors"]
            )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

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
