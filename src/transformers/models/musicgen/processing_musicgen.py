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
Text/audio processor class for MusicGen
"""


from ...processing_utils import ProcessorMixin


class MusicgenProcessor(ProcessorMixin):
    r"""
    Constructs a MusicGen processor which wraps an EnCodec feature extractor and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`EncodecFeatureExtractor`] and [`TTokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`EncodecFeatureExtractor`):
            An instance of [`EncodecFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    """
    feature_extractor_class = "EncodecFeatureExtractor"
    tokenizer_class = "T5Tokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to EncodecFeatureExtractor's [`~EncodecFeatureExtractor.__call__`] and the `text`
        argument to [`~T5Tokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if text is not None:
            inputs = self.tokenizer(text, **kwargs)

        if audio is not None:
            audio_inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)

        if audio is None:
            return inputs

        elif text is None:
            return audio_inputs

        else:
            inputs["input_values"] = audio_inputs["input_values"]
            if "padding_mask" in audio_inputs:
                inputs["padding_mask"] = audio_inputs["padding_mask"]
            return inputs
