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
import sys
import warnings
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import VideoInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import AddedToken, PreTokenizedInput, TextInput
from ...utils import logging
from ..auto import AutoTokenizer


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


logger = logging.get_logger(__name__)


class InstructBlipVideoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "truncation": None,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
    }


class InstructBlipVideoProcessor(ProcessorMixin):
    r"""
    Constructs an InstructBLIPVideo processor which wraps a InstructBLIP image processor and a LLaMa/T5 tokenizer into a single
    processor.

    [`InstructBlipVideoProcessor`] offers all the functionalities of [`InstructBlipVideoImageProcessor`] and [`AutoTokenizer`]. See the
    docstring of [`~InstructBlipVideoProcessor.__call__`] and [`~InstructBlipVideoProcessor.decode`] for more information.

    Args:
        image_processor (`InstructBlipVideoImageProcessor`):
            An instance of [`InstructBlipVideoImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
        qformer_tokenizer (`AutoTokenizer`, *optional*):
            An instance of ['PreTrainedTokenizer`]. The Q-Former tokenizer is a required input.
        num_query_tokens (`int`, *optional*):
            Number of tokens used by the Qformer as queries, should be same as in model's config.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["num_query_tokens"]
    image_processor_class = "InstructBlipVideoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer=None, num_query_tokens=None, **kwargs):
        # add QFormer tokenizer
        self.qformer_tokenizer = qformer_tokenizer
        self.video_token = AddedToken("<video>", normalized=False, special=True)
        tokenizer.add_tokens([self.video_token], special_tokens=True)
        self.num_query_tokens = num_query_tokens
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: Optional[VideoInput] = None,  # Keeping this here for backwards compatibility
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        videos: Optional[VideoInput] = None,
        audio=None,
        **kwargs: Unpack[InstructBlipVideoProcessorKwargs],
    ) -> BatchFeature:
        """
        This method uses [`InstructBlipVideoImageProcessor.__call__`] method to prepare image(s) or video(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Args:
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`ImageInput`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            videos (`VideoInput`, *optional*):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            -- **qformer_input_ids** - List of token ids from the Q-Former tokenizer to be fed to a model. Returned when `text` is not `None`.
            -- **qformer_attention_mask** - List of indices specifying which tokens from the Q-Former tokenizer should be attended to by the model. Returned when `text` is not `None`.
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is not None:
            warnings.warn(
                "The `images` argument is deprecated and will be removed in future versions, use `videos` instead.",
                FutureWarning,
            )
        if images is not None and videos is not None:
            raise ValueError(
                "You cannot provide both `images` and `videos` at the same time. Please pass video data as `videos=...` instead."
            )
        if images is not None and videos is None:
            videos = images

        output_kwargs = self._merge_kwargs(
            InstructBlipVideoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        encoding = BatchFeature()

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            _text_encoding = self.tokenizer(
                text=text,
                **{
                    **output_kwargs["text_kwargs"],
                    "return_tensors": None,  # required to concatenate below
                },
            )

            # if we know how many query tokens, expand text inside processor. We need this hacky manipulation
            # because BLIP expects image tokens to be at the beginning even before BOS token
            if self.num_query_tokens is not None and videos is not None:
                text_encoding = {}
                video_tokens = (
                    self.video_token.content * self.num_query_tokens * 4
                )  # InstrucBLIP works with 4 frames only
                video_token_encoding = self.tokenizer([video_tokens], add_special_tokens=False, return_tensors=None)
                for k in _text_encoding:
                    text_encoding[k] = [
                        img_encoding + txt_encoding
                        for img_encoding, txt_encoding in zip(video_token_encoding[k], _text_encoding[k])
                    ]
            else:
                text_encoding = _text_encoding
                if videos is not None:
                    logger.warning_once(
                        "Expanding inputs for video tokens in InstructBLIPVideo should be done in processing. "
                        "Please follow instruction here (https://gist.github.com/zucchini-nlp/65f22892b054dc0d68228af56fbeaac2) to update your InstructBLIPVideo model. "
                        "Using processors without these attributes in the config is deprecated and will throw an error in v4.47."
                    )

            # cast to desired return tensors type after concatenating
            text_encoding = BatchFeature(
                text_encoding, tensor_type=output_kwargs["common_kwargs"].get("return_tensors")
            )
            encoding.update(text_encoding)
            qformer_text_encoding = self.qformer_tokenizer(text=text, **output_kwargs["text_kwargs"])
            encoding["qformer_input_ids"] = qformer_text_encoding.pop("input_ids")
            encoding["qformer_attention_mask"] = qformer_text_encoding.pop("attention_mask")

        if videos is not None:
            image_encoding = self.image_processor(videos, **output_kwargs["images_kwargs"])
            encoding.update(image_encoding)

        return encoding

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # overwrite to save the Q-Former tokenizer in a separate folder
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        qformer_tokenizer_path = os.path.join(save_directory, "qformer_tokenizer")
        self.qformer_tokenizer.save_pretrained(qformer_tokenizer_path)
        return super().save_pretrained(save_directory, **kwargs)

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
