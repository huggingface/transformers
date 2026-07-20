# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import AudioInput
from ...utils import auto_docstring, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


@requires(backends=("torch",))
@auto_docstring
class GraniteSpeechNarProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        audio: AudioInput,
        **kwargs,
    ) -> BatchFeature:
        return self.feature_extractor(audio, **kwargs)


__all__ = ["GraniteSpeechNarProcessor"]
