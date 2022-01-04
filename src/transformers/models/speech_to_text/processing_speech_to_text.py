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
Speech processor class for Speech2Text
"""
from contextlib import contextmanager

from .feature_extraction_speech_to_text import Speech2TextFeatureExtractor
from .tokenization_speech_to_text import Speech2TextTokenizer


class Speech2TextProcessor:
    r"""
    Constructs a Speech2Text processor which wraps a Speech2Text feature extractor and a Speech2Text tokenizer into a
    single processor.

    [`Speech2TextProcessor`] offers all the functionalities of [`Speech2TextFeatureExtractor`] and
    [`Speech2TextTokenizer`]. See the [`~Speech2TextProcessor.__call__`] and [`~Speech2TextProcessor.decode`] for more
    information.

    Args:
        feature_extractor (`Speech2TextFeatureExtractor`):
            An instance of [`Speech2TextFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Speech2TextTokenizer`):
            An instance of [`Speech2TextTokenizer`]. The tokenizer is a required input.
    """

    def __init__(self, feature_extractor, tokenizer):
        if not isinstance(feature_extractor, Speech2TextFeatureExtractor):
            raise ValueError(
                f"`feature_extractor` has to be of type {Speech2TextFeatureExtractor.__class__}, but is {type(feature_extractor)}"
            )
        if not isinstance(tokenizer, Speech2TextTokenizer):
            raise ValueError(
                f"`tokenizer` has to be of type {Speech2TextTokenizer.__class__}, but is {type(tokenizer)}"
            )

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor

    def save_pretrained(self, save_directory):
        """
        Save a Speech2Text feature extractor object and Speech2Text tokenizer object to the directory `save_directory`,
        so that it can be re-loaded using the [`~Speech2TextProcessor.from_pretrained`] class method.

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
        self.feature_extractor._set_processor_class(self.__class__.__name__)
        self.feature_extractor.save_pretrained(save_directory)

        self.tokenizer._set_processor_class(self.__class__.__name__)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a [`Speech2TextProcessor`] from a pretrained Speech2Text processor.

        <Tip>

        This class method is simply calling Speech2TextFeatureExtractor's
        [`~PreTrainedFeatureExtractor.from_pretrained`] and Speech2TextTokenizer's
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
                  [`~PreTrainedFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both [`PreTrainedFeatureExtractor`] and
                [`PreTrainedTokenizer`]
        """
        feature_extractor = Speech2TextFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = Speech2TextTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Speech2TextFeatureExtractor's
        [`~Speech2TextFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Speech2TextProcessor.as_target_processor`] this method forwards all its arguments to Speech2TextTokenizer's
        [`~Speech2TextTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        return self.current_processor(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2TextTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2TextTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Speech2Text.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
