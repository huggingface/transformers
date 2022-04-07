# coding=utf-8
# Copyright 2021 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Union

import numpy as np

from transformers.data.data_collator import DataCollatorForWholeWordMask, tolist

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class FLAVAProcessor(ProcessorMixin):
    r"""
    Constructs a FLAVA processor which wraps a FLAVA feature extractor and a FLAVA tokenizer into a single processor.

    [`FLAVAProcessor`] offers all the functionalities of [`FLAVAFeatureExtractor`] and [`FLAVATokenizerFast`]. See the
    [`~FLAVAProcessor.__call__`] and [`~FLAVAProcessor.decode`] for more information.

    Args:
        feature_extractor ([`FLAVAFeatureExtractor`]):
            The feature extractor is a required input.
        tokenizer ([`FLAVATokenizerFast`]):
            The tokenizer is a required input.
    """
    feature_extractor_class = "FLAVAFeatureExtractor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, feature_extractor, tokenizer, mlm_probability=0.15):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self.text_masker = DataCollatorForWholeWordMask(tokenizer, mlm=True, mlm_probability=mlm_probability)

    def __call__(
        self,
        images: Optional[
            Union[
                "Image.Image",  # noqa
                np.ndarray,
                "torch.Tensor",  # noqa
                List["Image.Image"],  # noqa
                List[np.ndarray],
                List["torch.Tensor"],  # noqa
            ]
        ] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_masks: Optional[bool] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ):
        """
        This method uses [`FLAVAFeatureExtractor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information. Other special args are mentioned
        below:

        Args:
            return_mask (`bool`, *optional*, defaults to None):
                If True, the processor will return `bool_masked_pos` suggesting masks for image's patch version and
                `input_ids_masked` and `mlm_labels` for MLM.
        """

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask or return_masks,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        if images is not None:
            image_features = self.feature_extractor(
                images, return_masks=return_masks, return_tensors=return_tensors, **kwargs
            )

        if return_masks and text is not None:
            batch_masked = self.text_masker(tolist(encoding["input_ids"]), return_tensors=return_tensors)
            encoding["input_ids_masked"] = batch_masked["input_ids"]
            encoding["mlm_labels"] = batch_masked["labels"]
            encoding.pop("special_tokens_mask")

        if text is not None and images is not None:
            encoding.update(image_features)
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to FLAVATokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to FLAVATokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
