# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Processor class for ViLT.
"""

from typing import List, Optional, Union

from transformers import BertTokenizerFast

from ...file_utils import TensorType
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from .feature_extraction_vilt import ViltFeatureExtractor


class ViltProcessor:
    r"""
    Constructs a ViLT processor which wraps a BERT tokenizer and ViLT feature extractor into a single processor.

    [`ViltProcessor`] offers all the functionalities of [`ViltFeatureExtractor`] and [`BertTokenizerFast`]. See the
    docstring of [`~ViltProcessor.__call__`] and [`~ViltProcessor.decode`] for more information.

    Args:
        feature_extractor (`ViltFeatureExtractor`):
            An instance of [`ViltFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`BertTokenizerFast`):
            An instance of ['BertTokenizerFast`]. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, ViltFeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {ViltFeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, BertTokenizerFast):
            raise ValueError(f"`tokenizer` has to be of type {BertTokenizerFast.__class__}, but is {type(tokenizer)}")

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a ViLT feature_extractor object and BERT tokenizer object to the directory `save_directory`, so that it
        can be re-loaded using the [`~ViltProcessor.from_pretrained`] class method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizer.save_pretrained`]. Please refer to the docstrings of the methods
        above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a [`ViltProcessor`] from a pretrained ViLT processor.

        <Tip>

        This class method is simply calling ViltFeatureExtractor's
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and BertTokenizerFast's
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`]. Please refer to the docstrings of the methods
        above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both [`SequenceFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        feature_extractor = ViltFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
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
        This method uses [`ViltFeatureExtractor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
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
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )
        # add pixel_values + pixel_mask
        encoding_feature_extractor = self.feature_extractor(images, return_tensors=return_tensors)
        encoding.update(encoding_feature_extractor)

        return encoding

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
