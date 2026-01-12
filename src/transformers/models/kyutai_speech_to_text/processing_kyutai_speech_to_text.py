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
from ...utils import auto_docstring


class KyutaiSpeechToTextProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "audio_kwargs": {
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class KyutaiSpeechToTextProcessor(ProcessorMixin):
    valid_processor_kwargs = KyutaiSpeechToTextProcessorKwargs

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)


__all__ = ["KyutaiSpeechToTextProcessor"]
