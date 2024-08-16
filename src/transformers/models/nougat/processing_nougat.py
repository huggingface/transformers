# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for Nougat.
"""

import sys
import warnings
from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, TextKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class NougatTextKwargs(TextKwargs, total=False):
    text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]
    text_target: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]
    text_pair_target: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]


class NougatImagesKwargs(ImagesKwargs, total=False):
    do_crop_margin: Optional[bool]
    do_thumbnail: Optional[bool]
    do_align_long_axis: Optional[bool]


class NougatProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: NougatTextKwargs
    images_kwargs: NougatImagesKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "is_split_into_words": False,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "verbose": True,
        },
        "images_kwargs": {
            "data_format": "channels_first",
        },
    }


class NougatProcessor(ProcessorMixin):
    r"""
    Constructs a Nougat processor which wraps a Nougat image processor and a Nougat tokenizer into a single processor.

    [`NougatProcessor`] offers all the functionalities of [`NougatImageProcessor`] and [`NougatTokenizerFast`]. See the
    [`~NougatProcessor.__call__`] and [`~NougatProcessor.decode`] for more information.

    Args:
        image_processor ([`NougatImageProcessor`]):
            An instance of [`NougatImageProcessor`]. The image processor is a required input.
        tokenizer ([`NougatTokenizerFast`]):
            An instance of [`NougatTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        backwards_compatibility_placeholder_arg=None,
        **kwargs: Unpack[NougatProcessorKwargs],
    ):
        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            NougatProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # Temporary fix for "paddding_side" in init_kwargs
        _ = output_kwargs["text_kwargs"].pop("padding_side", None)

        # For backwards compatibility, we reuse `audio` as `text_pair`
        # in case downstream users passed it as a positional argument
        if output_kwargs["text_kwargs"].get("text_pair") is not None and audio is not None:
            raise ValueError(
                "You cannot provide `text_pair` as a positional argument and as a keyword argument at the same time."
                "Please provide it only as a keyword argument (i.e. `text_pair=...`)."
            )
        if "text_pair" not in output_kwargs["text_kwargs"]:
            warnings.warn(
                "No `text_pair` kwarg was detected. The use of `text_pair` as an argument without specifying it explicitely as `text_pair=` will be deprecated in future versions."
            )
            if audio is not None:
                output_kwargs["text_kwargs"]["text_pair"] = audio

        # For backwards compatibility, we reuse `videos` as `text_target`
        # in case downstream users passed it as a positional argument
        if output_kwargs["text_kwargs"].get("text_target") is not None and videos is not None:
            raise ValueError(
                "You cannot provide `text_target` as a positional argument and as a keyword argument at the same time."
                "Please provide it only as a keyword argument (i.e. `text_target=...`)."
            )
        if "text_target" not in output_kwargs["text_kwargs"]:
            warnings.warn(
                "No `text_target` kwarg was detected. The use of `text_target` as an argument without specifying it explicitely as `text_target=` will be deprecated in future versions."
            )
            if videos is not None:
                output_kwargs["text_kwargs"]["text_target"] = videos

        # For backwards compatibility, we reuse `backwards_compatibility_placeholder_arg` as `text_pair_target`
        # in case downstream users passed it as a positional argument
        if (
            output_kwargs["text_kwargs"].get("text_pair_target") is not None
            and backwards_compatibility_placeholder_arg is not None
        ):
            raise ValueError(
                "You cannot provide `text_pair_target` as a positional argument and as a keyword argument at the same time."
                "Please provide it only as a keyword argument (i.e. `text_pair_target=...`)."
            )
        if "text_pair_target" not in output_kwargs["text_kwargs"]:
            warnings.warn(
                "No `text_pair_target` kwarg was detected. The use of `text_pair_target` as an argument without specifying it explicitely as `text_pair_target=` will be deprecated in future versions."
            )
            if backwards_compatibility_placeholder_arg is not None:
                output_kwargs["text_kwargs"]["text_pair_target"] = backwards_compatibility_placeholder_arg

        text_pair = output_kwargs["text_kwargs"].pop("text_pair", None)
        text_target = output_kwargs["text_kwargs"].pop("text_target", None)
        text_pair_target = output_kwargs["text_kwargs"].pop("text_pair_target", None)

        if images is not None:
            inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        if text is not None:
            encodings = self.tokenizer(
                text,
                text_pair=text_pair,
                text_target=text_target,
                text_pair_target=text_pair_target,
                **output_kwargs["text_kwargs"],
            )

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_generation(self, *args, **kwargs):
        """
        This method forwards all its arguments to NougatTokenizer's [`~PreTrainedTokenizer.post_process_generation`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.post_process_generation(*args, **kwargs)
