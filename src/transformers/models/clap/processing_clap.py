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
Audio/Text processor class for CLAP
"""

from typing import Optional, Union

from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ...utils.deprecation import deprecate_kwarg


logger = logging.get_logger(__name__)


class ClapProcessor(ProcessorMixin):
    r"""
    Constructs a CLAP processor which wraps a CLAP feature extractor and a RoBerta tokenizer into a single processor.

    [`ClapProcessor`] offers all the functionalities of [`ClapFeatureExtractor`] and [`RobertaTokenizerFast`]. See the
    [`~ClapProcessor.__call__`] and [`~ClapProcessor.decode`] for more information.

    Args:
        feature_extractor ([`ClapFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`RobertaTokenizerFast`]):
            The tokenizer is a required input.
    """

    feature_extractor_class = "ClapFeatureExtractor"
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @deprecate_kwarg("audios", version="v4.59.0", new_name="audio")
    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        audios: Optional[AudioInput] = None,
        audio: Optional[AudioInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        """
        Forwards the `audio` and `sampling_rate` arguments to [`~ClapFeatureExtractor.__call__`] and the `text`
        argument to [`~RobertaTokenizerFast.__call__`]. Please refer to the docstring of the above two methods for more
        information.
        """
        # The `deprecate_kwarg` will not work if the inputs are passed as arguments, so we check
        # again that the correct naming is used
        if audios is not None and audio is None:
            logger.warning(
                "Using `audios` keyword argument is deprecated when calling ClapProcessor, instead use `audio`."
            )
            audio = audios

        return super().__call__(text=text, audio=audio, **kwargs)


__all__ = ["ClapProcessor"]
