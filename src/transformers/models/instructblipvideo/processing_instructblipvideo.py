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
Processor class for InstructBLIP. Largely copy of Blip2Processor with addition of a tokenizer for the Q-Former.
"""

import os
from typing import Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import (
    AddedToken,
    PreTokenizedInput,
    TextInput,
)
from ...utils import logging
from ...video_utils import VideoInput
from ..auto import AutoTokenizer


logger = logging.get_logger(__name__)


class InstructBlipVideoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {},
    }


class InstructBlipVideoProcessor(ProcessorMixin):
    r"""
    Constructs an InstructBLIPVideo processor which wraps a InstructBLIP image processor and a LLaMa/T5 tokenizer into a single
    processor.

    [`InstructBlipVideoProcessor`] offers all the functionalities of [`InstructBlipVideoImageProcessor`] and [`AutoTokenizer`]. See the
    docstring of [`~InstructBlipVideoProcessor.__call__`] and [`~InstructBlipVideoProcessor.decode`] for more information.

    Args:
        video_processor (`InstructBlipVideoVideoProcessor`):
            An instance of [`InstructBlipVideoVideoProcessor`]. The video processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
        qformer_tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
        num_query_tokens (`int`, *optional*):
            Number of tokens used by the Qformer as queries, should be same as in model's config.
    """

    attributes = ["video_processor", "tokenizer", "qformer_tokenizer"]
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"
    qformer_tokenizer_class = "AutoTokenizer"

    def __init__(self, video_processor, tokenizer, qformer_tokenizer, num_query_tokens=None, **kwargs):
        if not hasattr(tokenizer, "video_token"):
            self.video_token = "<video>"
            tokenizer.add_tokens([AddedToken(self.video_token, normalized=False, special=True)], special_tokens=True)
        else:
            self.video_token = tokenizer.video_token
        self.num_query_tokens = num_query_tokens
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        super().__init__(video_processor, tokenizer, qformer_tokenizer)

    def __call__(
        self,
        images: VideoInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[InstructBlipVideoProcessorKwargs],
    ) -> BatchFeature:
        """
        This method uses [`InstructBlipVideoImageProcessor.__call__`] method to prepare image(s) or video(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of images or text.")

        output_kwargs = self._merge_kwargs(
            InstructBlipVideoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # BC for explicit return_tensors
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["return_mm_token_type_ids"].pop("return_mm_token_type_ids", None)
        max_length = output_kwargs["text_kwargs"].pop("max_length", None)
        if max_length is not None:
            output_kwargs["text_kwargs"]["max_length"] = max_length - self.num_query_tokens

        encoding = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            qformer_text_encoding = self.qformer_tokenizer(text=text, **output_kwargs["text_kwargs"])
            encoding["qformer_input_ids"] = qformer_text_encoding.pop("input_ids")
            encoding["qformer_attention_mask"] = qformer_text_encoding.pop("attention_mask")

            text_encoding = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

            if images is not None:
                video_tokens = self.video_token * self.num_query_tokens * 4
                output_kwargs["text_kwargs"]["add_special_tokens"] = False
                output_kwargs["text_kwargs"]["padding"] = False
                output_kwargs["text_kwargs"]["truncation"] = False
                video_text_encoding = self.tokenizer(video_tokens, **output_kwargs["text_kwargs"])
                for k in text_encoding:
                    text_encoding[k] = [video_text_encoding[k] + sample for sample in text_encoding[k]]
            encoding.update(text_encoding)

        if images is not None:
            image_encoding = self.video_processor(images, **output_kwargs["images_kwargs"])
            encoding.update(image_encoding)

        if text is not None and return_mm_token_type_ids:
            array_ids = np.array(encoding["input_ids"])
            mm_token_type_ids = np.zeros_like(array_ids)
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            encoding["mm_token_type_ids"] = mm_token_type_ids.tolist()

        encoding = BatchFeature(encoding, tensor_type=return_tensors)
        return encoding

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        qformer_input_names = ["qformer_input_ids", "qformer_attention_mask"]
        return tokenizer_input_names + video_processor_input_names + qformer_input_names

    # overwrite to save the Q-Former tokenizer in a separate folder
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        qformer_tokenizer_path = os.path.join(save_directory, "qformer_tokenizer")
        self.qformer_tokenizer.save_pretrained(qformer_tokenizer_path)

        # We modify the attributes so that only the tokenizer and image processor are saved in the main folder
        qformer_present = "qformer_tokenizer" in self.attributes
        if qformer_present:
            self.attributes.remove("qformer_tokenizer")

        outputs = super().save_pretrained(save_directory, **kwargs)

        if qformer_present:
            self.attributes += ["qformer_tokenizer"]
        return outputs

    # overwrite to load the Q-Former tokenizer from a separate folder
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if return_unused_kwargs a tuple is returned where the second element is 'unused_kwargs'
        if isinstance(processor, tuple):
            processor = processor[0]
        qformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="qformer_tokenizer")
        processor.qformer_tokenizer = qformer_tokenizer
        return processor


__all__ = ["InstructBlipVideoProcessor"]
