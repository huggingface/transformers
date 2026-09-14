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
Speech processor class for Whisper
"""

from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring


@auto_docstring
class WhisperProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    @auto_docstring
    def __call__(self, *args, **kwargs):
        audio = kwargs.pop("audio", None)
        text = kwargs.pop("text", None)

        # for BC
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        outputs = super().__call__(audio=audio, text=text, **kwargs)
        if text is not None:
            outputs["labels"] = outputs["input_ids"]
        return outputs

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)


__all__ = ["WhisperProcessor"]
