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

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import (
    AddedToken,
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from ...utils import TensorType, logging
from ...video_utils import VideoInput


logger = logging.get_logger(__name__)


class InstructBlipVideoProcessor(ProcessorMixin):
    r"""
    Constructs an InstructBLIPVideo processor which wraps a InstructBLIP image processor and a LLaMa/T5 tokenizer into a single
    processor.

    [`InstructBlipVideoProcessor`] offers all the functionalities of [`InstructBlipVideoVideoProcessor`] and [`AutoTokenizer`]. See the
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

    def __init__(self, video_processor, tokenizer, qformer_tokenizer, num_query_tokens=None, **kwargs):
        if not hasattr(tokenizer, "video_token"):
            self.video_token = AddedToken("<video>", normalized=False, special=True)
            tokenizer.add_tokens([self.video_token], special_tokens=True)
        else:
            self.video_token = tokenizer.video_token
        self.num_query_tokens = num_query_tokens
        super().__init__(video_processor, tokenizer, qformer_tokenizer)

    def __call__(
        self,
        images: Optional[VideoInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        This method uses [`InstructBlipVideoVideoProcessor.__call__`] method to prepare image(s) or video(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of images or text.")

        encoding = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not isinstance(text, list) and not isinstance(text[0], str):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

            qformer_text_encoding = self.qformer_tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            encoding["qformer_input_ids"] = qformer_text_encoding.pop("input_ids")
            encoding["qformer_attention_mask"] = qformer_text_encoding.pop("attention_mask")

            # We need this hacky manipulation because BLIP expects image tokens to be at the beginning even before BOS token
            # InstrucBLIP works with 4 frames only
            if max_length is not None:
                max_length -= self.num_query_tokens
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=None,  # required to concatenate below
                **kwargs,
            )

            if images is not None:
                video_tokens = self.video_token.content * self.num_query_tokens * 4
                video_text_encoding = self.tokenizer(
                    video_tokens,
                    add_special_tokens=False,  # required to concatenate below
                    return_attention_mask=return_attention_mask,
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_mask,
                    return_offsets_mapping=return_offsets_mapping,
                    return_token_type_ids=return_token_type_ids,
                    return_length=return_length,
                    return_tensors=None,
                )
                for k in text_encoding:
                    text_encoding[k] = [video_text_encoding[k] + sample for sample in text_encoding[k]]
            encoding.update(text_encoding)

        if images is not None:
            image_encoding = self.video_processor(images, return_tensors=return_tensors)
            encoding.update(image_encoding)

        encoding = BatchFeature(encoding, tensor_type=return_tensors)
        return encoding

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        qformer_input_names = ["qformer_input_ids", "qformer_attention_mask"]
        return tokenizer_input_names + video_processor_input_names + qformer_input_names


__all__ = ["InstructBlipVideoProcessor"]
