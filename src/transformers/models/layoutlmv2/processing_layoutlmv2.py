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
Processor class for LayoutLMv2
"""
from .feature_extraction_layoutlmv2 import LayoutLMv2FeatureExtractor
from .tokenization_layoutlmv2 import LayoutLMv2Tokenizer


class LayoutLMv2Processor:
    r"""
    Constructs a LayoutLMv2 processor which combines a LayoutLMv2 feature extractor and a LayoutLMv2 tokenizer into a
    single processor.

    :class:`~transformers.LayoutLMv2Processor` offers all the functionalities you need to prepare data for the model.

    It first uses :class:`~transformers.LayoutLMv2FeatureExtractor` to apply OCR on document images to get words and
    normalized bounding boxes. These are then provided to :class:`~transformers.LayoutLMv2Tokenizer`, which turns the
    words and bounding boxes into token-level :obj:`input_ids`, :obj:`attention_mask`, :obj:`token_type_ids`,
    :obj:`token_type_ids`.

    Args:
        feature_extractor (:obj:`LayoutLMv2FeatureExtractor`):
            An instance of :class:`~transformers.LayoutLMv2FeatureExtractor`. The feature extractor is a required
            input.
        tokenizer (:obj:`LayoutLMv2Tokenizer`):
            An instance of :class:`~transformers.LayoutLMv2Tokenizer`. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, LayoutLMv2FeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {LayoutLMv2FeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, LayoutLMv2Tokenizer):
            raise ValueError(
                f"`tokenizer` has to be of type {LayoutLMv2Tokenizer.__class__}, but is {type(tokenizer)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def save_pretrained(self, save_directory):
        """
        Save a LayoutLMv2 feature_extractor object and LayoutLMv2 tokenizer object to the directory ``save_directory``,
        so that it can be re-loaded using the :func:`~transformers.LayoutLMv2Processor.from_pretrained` class method.

        .. note::

            This class method is simply calling
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained` and
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.save_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """

        self.feature_extractor.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a :class:`~transformers.LayoutLMv2Processor` from a pretrained LayoutLMv2 processor.

        .. note::

            This class method is simply calling LayoutLMv2FeatureExtractor's
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.from_pretrained` and
            LayoutLMv2Tokenizer's :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
            Please refer to the docstrings of the methods above for more information.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :meth:`~transformers.SequenceFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/preprocessor_config.json``.
            **kwargs
                Additional keyword arguments passed along to both :class:`~transformers.SequenceFeatureExtractor` and
                :class:`~transformers.PreTrainedTokenizer`
        """
        feature_extractor = LayoutLMv2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = LayoutLMv2Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        This method first forwards all its arguments to LayoutLMv2FeatureExtractor's
        :meth:`~transformers.LayoutLMv2FeatureExtractor.__call__`. Next, it passes the obtained features to
        :meth:`~transformers.LayoutLMv2Processor.__call__` and returns the output. Please refer to the docstring of the
        above two methods for more information.
        """
        # first, apply the feature extractor
        features = self.feature_extractor(*args, **kwargs)

        pixel_values = features.pop("pixel_values")

        # second, apply the tokenizer
        # encoded_inputs = self.tokenizer(**features, *args, **kwargs)

        encoded_inputs["image"] = pixel_values

        return encoded_inputs
