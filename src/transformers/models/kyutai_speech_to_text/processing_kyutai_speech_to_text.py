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


from ...processing_utils import ProcessingKwargs, ProcessorMixin


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

    feature_extractor_class = "KyutaiSpeechToTextFeatureExtractor"
    tokenizer_class = "PreTrainedTokenizerFast"
    valid_processor_kwargs = KyutaiSpeechToTextProcessorKwargs


__all__ = ["KyutaiSpeechToTextProcessor"]
