# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional

from ...utils import is_soundfile_available, is_torch_available


if is_torch_available():
    pass

if is_soundfile_available():
    pass

from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack


class KyutaiSpeechToTextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


class KyutaiSpeechToTextProcessor(ProcessorMixin):
    r"""
    Constructs a Moshi ASR processor which wraps [`EncodecFeatureExtractor`] and
    [`PreTrainedTokenizerFast`] into a single processor that inherits both the audio feature extraction and
    tokenizer functionalities. See the [`~KyutaiSpeechToTextProcessor.__call__`] for more
    information.
    """

    feature_extractor_class = "EncodecFeatureExtractor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __call__(
        self,
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[KyutaiSpeechToTextProcessorKwargs],
    ):
        # TODO: @eustlb, add doc

        output_kwargs = self._merge_kwargs(
            KyutaiSpeechToTextProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # TODO: ensure audio in correct format

        inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])

        return inputs

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to KyutaiSpeechToTextTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to KyutaiSpeechToTextTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)


__all__ = ["KyutaiSpeechToTextProcessor"]
