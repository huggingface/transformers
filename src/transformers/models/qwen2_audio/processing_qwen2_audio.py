# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for Qwen2Audio.
"""

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin


class Qwen2AudioProcessor(ProcessorMixin):
    r"""
    Constructs a Qwen2Audio processor which wraps a Qwen2Audio feature extractor and a Qwen2Audio tokenizer into a single processor.

    [`Qwen2AudioProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2AudioProcessor.__call__`] and [`~Qwen2AudioProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None):
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, *args, **kwargs) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the doctsring
        of the above two methods for more information.
        """

        audios = kwargs.pop("audios", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audios = args[0]
            args = args[1:]

        if audios is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")
        text_inputs = self.tokenizer(text, **kwargs)

        kwargs.pop("padding", None)
        if audios is not None:
            inputs = self.feature_extractor(
                audios, *args, sampling_rate=sampling_rate, return_attention_mask=True, padding="max_length", **kwargs
            )
            inputs["feature_attention_mask"] = inputs.pop(
                "attention_mask"
            )  # rename attention_mask to prevent conflicts later on
            text_inputs.update(inputs)

        return BatchFeature(data={**text_inputs})

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))
