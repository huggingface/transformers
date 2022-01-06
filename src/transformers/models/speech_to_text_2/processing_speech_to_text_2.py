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
Speech processor class for Speech2Text2
"""
from contextlib import contextmanager

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ..auto.feature_extraction_auto import AutoFeatureExtractor
from .tokenization_speech_to_text_2 import Speech2Text2Tokenizer


class Speech2Text2Processor:
    r"""
    Constructs a Speech2Text2 processor which wraps a Speech2Text2 feature extractor and a Speech2Text2 tokenizer into
    a single processor.

    [`Speech2Text2Processor`] offers all the functionalities of [`AutoFeatureExtractor`] and [`Speech2Text2Tokenizer`].
    See the [`~Speech2Text2Processor.__call__`] and [`~Speech2Text2Processor.decode`] for more information.

    Args:
        feature_extractor (`AutoFeatureExtractor`):
            An instance of [`AutoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Speech2Text2Tokenizer`):
            An instance of [`Speech2Text2Tokenizer`]. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, SequenceFeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {SequenceFeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, Speech2Text2Tokenizer):
            raise ValueError(
                f"`tokenizer` has to be of type {Speech2Text2Tokenizer.__class__}, but is {type(tokenizer)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a Speech2Text2 feature extractor object and Speech2Text2 tokenizer object to the directory
        `save_directory`, so that it can be re-loaded using the [`~Speech2Text2Processor.from_pretrained`] class
        method.

        <Tip>

        This class method is simply calling [`~PreTrainedFeatureExtractor.save_pretrained`] and
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
        Instantiate a [`Speech2Text2Processor`] from a pretrained Speech2Text2 processor.

        <Tip>

        This class method is simply calling AutoFeatureExtractor's [`~PreTrainedFeatureExtractor.from_pretrained`] and
        Speech2Text2Tokenizer's [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`]. Please refer to the
        docstrings of the methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~PreTrainedFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both [`PreTrainedFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Speech2Text2Tokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor's
        [`~AutoFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Speech2Text2Processor.as_target_processor`] this method forwards all its arguments to
        Speech2Text2Tokenizer's [`~Speech2Text2Tokenizer.__call__`]. Please refer to the doctsring of the above two
        methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2Text2Tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2Text2Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Speech2Text2.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
