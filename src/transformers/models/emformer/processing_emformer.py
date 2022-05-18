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
Speech processor class for Emformer
"""
import warnings
from contextlib import contextmanager

from ...processing_utils import ProcessorMixin
from .feature_extraction_emformer import EmformerFeatureExtractor
from .tokenization_emformer import EmformerTokenizer


class EmformerProcessor(ProcessorMixin):
    r"""
    Constructs an Emformer processor which wraps an Emformer feature extractor and an Emformer target tokenizer into a
    single processor.

    [`EmformerProcessor`] offers all the functionalities of [`EmformerFeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~EmformerProcessor.__call__`] and [`~EmformerProcessor.decode`] for more information.

    Args:
        feature_extractor (`EmformerFeatureExtractor`):
            An instance of [`EmformerFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "EmformerFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'EmformerTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            feature_extractor = EmformerFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = EmformerTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to EmformerFeatureExtractor's
        [`~EmformerFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~EmformerProcessor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.__call__`]. Please refer to the docstring of the above two methods for more information.
        """
        return self.current_processor(*args, **kwargs)

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to EmformerFeatureExtractor's
        [`~EmformerFeatureExtractor.pad`] and returns its output. If used in the context
        [`~EmformerProcessor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.pad`]. Please refer to the docstring of the above two methods for more information.
        """
        return self.current_processor.pad(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Emformer.
        """
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
