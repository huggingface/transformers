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
Processor class for LayoutLMv2.
"""
from typing import List, Optional, Union

from ...file_utils import TensorType
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from .feature_extraction_layoutlmv2 import LayoutLMv2FeatureExtractor
from .tokenization_layoutlmv2 import LayoutLMv2Tokenizer
from .tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast


class LayoutLMv2Processor:
    r"""
    Constructs a LayoutLMv2 processor which combines a LayoutLMv2 feature extractor and a LayoutLMv2 tokenizer into a
    single processor.

    [`LayoutLMv2Processor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv2FeatureExtractor`] to resize document images to a fixed size, and
    optionally applies OCR to get words and normalized bounding boxes. These are then provided to
    [`LayoutLMv2Tokenizer`] or [`LayoutLMv2TokenizerFast`], which turns the words
    and bounding boxes into token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`.
    Optionally, one can provide integer `word_labels`, which are turned into token-level `labels` for token
    classification tasks (such as FUNSD, CORD).

    Args:
        feature_extractor (`LayoutLMv2FeatureExtractor`):
            An instance of [`LayoutLMv2FeatureExtractor`]. The feature extractor is a required
            input.
        tokenizer (`LayoutLMv2Tokenizer` or `LayoutLMv2TokenizerFast`):
            An instance of [`LayoutLMv2Tokenizer`] or
            [`LayoutLMv2TokenizerFast`]. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, LayoutLMv2FeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {LayoutLMv2FeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)):
            raise ValueError(
                f"`tokenizer` has to be of type {LayoutLMv2Tokenizer.__class__} or {LayoutLMv2TokenizerFast.__class__}, but is {type(tokenizer)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def save_pretrained(self, save_directory):
        """
        Save a LayoutLMv2 feature_extractor object and LayoutLMv2 tokenizer object to the directory `save_directory`,
        so that it can be re-loaded using the [`~LayoutLMv2Processor.from_pretrained`] class method.

        <Tip>

        This class method is simply calling
        [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizer.save_pretrained`]. Please refer to the
        docstrings of the methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, use_fast=True, **kwargs):
        r"""
        Instantiate a [`LayoutLMv2Processor`] from a pretrained LayoutLMv2 processor.

        <Tip>

        This class method is simply calling LayoutLMv2FeatureExtractor's
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
        LayoutLMv2TokenizerFast's
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`]. Please refer to the
        docstrings of the methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.

            use_fast (`bool`, *optional*, defaults to `True`):
                Whether or not to instantiate a fast tokenizer.

            **kwargs
                Additional keyword arguments passed along to both [`SequenceFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        feature_extractor = LayoutLMv2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if use_fast:
            tokenizer = LayoutLMv2TokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            tokenizer = LayoutLMv2Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        This method first forwards the `images` argument to
        [`~LayoutLMv2FeatureExtractor.__call__`]. In case [`LayoutLMv2FeatureExtractor`] was
        initialized with `apply_ocr` set to `True`, it passes the obtained words and bounding boxes along with
        the additional arguments to [`~LayoutLMv2Tokenizer.__call__`] and returns the output, together
        with resized `images`. In case [`LayoutLMv2FeatureExtractor`] was initialized with `apply_ocr`
        set to `False`, it passes the words (`text`/``text_pair`) and `boxes` specified by the user along with the additional arguments to [`~LayoutLMv2Tokenizer.__call__`] and returns the output, together with resized `images``.

        Please refer to the docstring of the above two methods for more information.
        """
        # verify input
        if self.feature_extractor.apply_ocr and (boxes is not None):
            raise ValueError(
                "You cannot provide bounding boxes "
                "if you initialized the feature extractor with apply_ocr set to True."
            )

        if self.feature_extractor.apply_ocr and (word_labels is not None):
            raise ValueError(
                "You cannot provide word labels "
                "if you initialized the feature extractor with apply_ocr set to True."
            )

        # first, apply the feature extractor
        features = self.feature_extractor(images=images, return_tensors=return_tensors)

        # second, apply the tokenizer
        if text is not None and self.feature_extractor.apply_ocr and text_pair is None:
            if isinstance(text, str):
                text = [text]  # add batch dimension (as the feature extractor always adds a batch dimension)
            text_pair = features["words"]

        encoded_inputs = self.tokenizer(
            text=text if text is not None else features["words"],
            text_pair=text_pair if text_pair is not None else None,
            boxes=boxes if boxes is not None else features["boxes"],
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )

        # add pixel values
        encoded_inputs["image"] = features.pop("pixel_values")

        return encoded_inputs
