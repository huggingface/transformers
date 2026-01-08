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
Audio/Text processor class for SeamlessM4T
"""

from typing import Optional, Union

from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class SeamlessM4TTextKwargs(TextKwargs):
    """
    src_lang (`str`, *optional*):
        The source language code for the input text (e.g., "eng" for English, "fra" for French). This is used
        to set the language token at the beginning of the input sequence, which helps the model understand the
        input language for translation or transcription tasks.
    tgt_lang (`str`, *optional*):
        The target language code for the output (e.g., "eng" for English, "fra" for French). This is used to
        specify the desired output language for translation tasks. The model will generate text in this language.
    """

    src_lang: Optional[str]
    tgt_lang: Optional[str]


class SeamlessM4TProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: SeamlessM4TTextKwargs
    _defaults = {}


@auto_docstring
class SeamlessM4TProcessor(ProcessorMixin):
    valid_processor_kwargs = SeamlessM4TProcessorKwargs

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        r"""
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **input_features** -- Audio input features to be fed to a model. Returned when `audios` is not `None`.
        """
        if text is not None and audio is not None:
            raise ValueError(
                "Text and audios are mututally exclusive when passed to `SeamlessM4T`. Specify one or another."
            )
        return super().__call__(text=text, audio=audio, **kwargs)


__all__ = ["SeamlessM4TProcessor"]
