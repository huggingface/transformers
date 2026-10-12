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
Processor class for CLVP
"""

from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring
class ClvpProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(self, *args, text=None, audio=None, **kwargs):
        raw_speech = kwargs.pop("raw_speech", None)
        if raw_speech is not None:
            logger.warning(
                "Using `raw_speech` keyword argument is deprecated when calling ClvpProcessor, instead use `audio`."
            )
            audio = raw_speech

        # Merge first so that both flat (BC) and nested modality kwargs are resolved into structured kwargs. Injecting a
        # nested `audio_kwargs` before merging would switch `_merge_kwargs` into nested-dict mode and silently drop any
        # flat kwargs the user passed.
        merged_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )

        # The CLVP model relies on the *text* attention mask. When both text and audio are provided, prevent the
        # feature extractor's audio attention mask from overriding the tokenizer's attention mask in the merged output.
        if audio is not None and text is not None:
            merged_kwargs["audio_kwargs"]["return_attention_mask"] = False

        return super().__call__(*args, text=text, audio=audio, **merged_kwargs)


__all__ = ["ClvpProcessor"]
