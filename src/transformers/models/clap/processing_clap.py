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

import sys
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput


if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack


class ClapProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


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

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audios: Optional[AudioInput] = None,
        images=None,
        videos=None,
        **kwargs: Unpack[ClapProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to RobertaTokenizerFast's [`~RobertaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
        ClapFeatureExtractor's [`~ClapFeatureExtractor.__call__`] if `audios` is not `None`. Please refer to the
        doctsring of the above two methods for more information.

        Args:
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`AudioInput`, *optional*):
                The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
                and T the sample length of the audio.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **audio_features** -- Audio features to be fed to a model. Returned when `audios` is not `None`.
        """

        if text is None and audios is None:
            raise ValueError("You have to specify either text or audios. Both cannot be none.")

        output_kwargs = self._merge_kwargs(
            ClapProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        data = {}
        if text is not None:
            text_features = self.tokenizer(text, **output_kwargs["text_kwargs"])
            data.update(text_features)
        if audios is not None:
            audio_features = self.feature_extractor(audios, **output_kwargs["audio_kwargs"])
            data.update(audio_features)
        return BatchFeature(data=data, tensor_type=output_kwargs["common_kwargs"].get("return_tensors"))

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))
