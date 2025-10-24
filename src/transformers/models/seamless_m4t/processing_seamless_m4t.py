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
Audio/Text processor class for SeamlessM4T
"""

from typing import Optional, Union

from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
from ...utils.deprecation import deprecate_kwarg


logger = logging.get_logger(__name__)


class SeamlessM4TTextKwargs(TextKwargs):
    src_lang: Optional[str]
    tgt_lang: Optional[str]


class SeamlessM4TProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: SeamlessM4TTextKwargs
    _defaults = {}


class SeamlessM4TProcessor(ProcessorMixin):
    r"""
    Constructs a SeamlessM4T processor which wraps a SeamlessM4T feature extractor and a SeamlessM4T tokenizer into a
    single processor.

    [`SeamlessM4TProcessor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and
    [`SeamlessM4TTokenizerFast`]. See the [`~SeamlessM4TProcessor.__call__`] and [`~SeamlessM4TProcessor.decode`] for
    more information.

    Args:
        feature_extractor ([`SeamlessM4TFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`SeamlessM4TTokenizerFast`]):
            The tokenizer is a required input.
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = ("SeamlessM4TTokenizer", "SeamlessM4TTokenizerFast")
    valid_processor_kwargs = SeamlessM4TProcessorKwargs

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
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to SeamlessM4TTokenizerFast's [`~SeamlessM4TTokenizerFast.__call__`] if `text` is not
        `None` to encode the text. To prepare the audio(s), this method forwards the `audios` and `kwargs` arguments to
        SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.__call__`] if `audios` is not `None`. Please refer
        to the docstring of the above two methods for more information.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audios (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
                and T the sample length of the audio.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **input_features** -- Audio input features to be fed to a model. Returned when `audios` is not `None`.
        """
        if text is not None and audios is not None:
            raise ValueError(
                "Text and audios are mututally exclusive when passed to `SeamlessM4T`. Specify one or another."
            )
        if audio is None and audios is not None:
            logger.warning(
                "Passing `audios` as keyword argument is deprecated and will be removed in v4.63, please pass `audio` instead."
            )
            audio = audios
        return super().__call__(text=text, audio=audio, **kwargs)


__all__ = ["SeamlessM4TProcessor"]
