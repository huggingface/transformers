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
Image/Text processor class for Clip
"""
from contextlib import contextmanager

from .feature_extraction_clip import ClipFeatureExtractor
from .tokenization_clip import ClipTokenizer


class ClipProcessor:
    r"""
    Constructs a Clip processor which wraps a Clip feature extractor and a Clip tokenizer into a single processor.

    :class:`~transformers.ClipProcessor` offers all the functionalities of :class:`~transformers.ClipFeatureExtractor`
    and :class:`~transformers.ClipTokenizer`. See the :meth:`~transformers.ClipProcessor.__call__` and
    :meth:`~transformers.ClipProcessor.decode` for more information.

    Args:
        feature_extractor (:obj:`ClipFeatureExtractor`):
            An instance of :class:`~transformers.ClipFeatureExtractor`. The feature extractor is a required input.
        tokenizer (:obj:`ClipTokenizer`):
            An instance of :class:`~transformers.ClipTokenizer`. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, ClipFeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {ClipFeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, ClipTokenizer):
            raise ValueError(f"`tokenizer` has to be of type {ClipTokenizer.__class__}, but is {type(tokenizer)}")

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a Clip feature extractor object and Clip tokenizer object to the directory ``save_directory``, so that it
        can be re-loaded using the :func:`~transformers.ClipProcessor.from_pretrained` class method.

        .. note::

            This class method is simply calling :meth:`~transformers.PreTrainedFeatureExtractor.save_pretrained` and
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
        Instantiate a :class:`~transformers.ClipProcessor` from a pretrained Clip processor.

        .. note::

            This class method is simply calling ClipFeatureExtractor's
            :meth:`~transformers.PreTrainedFeatureExtractor.from_pretrained` and ClipTokenizer's
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :meth:`~transformers.PreTrainedFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/feature_extraction_config.json``.
            **kwargs
                Additional keyword arguments passed along to both :class:`~transformers.PreTrainedFeatureExtractor` and
                :class:`~transformers.PreTrainedTokenizer`
        """
        feature_extractor = ClipFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = ClipTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to ClipFeatureExtractor's
        :meth:`~transformers.ClipFeatureExtractor.__call__` and returns its output. If used in the context
        :meth:`~transformers.ClipProcessor.as_target_processor` this method forwards all its arguments to
        ClipTokenizer's :meth:`~transformers.ClipTokenizer.__call__`. Please refer to the doctsring of the above two
        methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ClipTokenizer's
        :meth:`~transformers.PreTrainedTokenizer.batch_decode`. Please refer to the docstring of this method for more
        information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to ClipTokenizer's :meth:`~transformers.PreTrainedTokenizer.decode`.
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning Clip.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
