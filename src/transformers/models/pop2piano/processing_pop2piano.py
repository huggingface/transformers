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
""" Processor class for Pop2Piano."""

import os
from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import TensorType


class Pop2PianoProcessor(ProcessorMixin):
    r"""
    Constructs an Pop2Piano processor which wraps a Pop2Piano Feature Extractor and Pop2Piano Tokenizer into a single
    processor.

    [`Pop2PianoProcessor`] offers all the functionalities of [`Pop2PianoFeatureExtractor`] and [`Pop2PianoTokenizer`].
    See the docstring of [`~Pop2PianoProcessor.__call__`] and [`~Pop2PianoProcessor.decode`] for more information.

    Args:
        feature_extractor (`Pop2PianoFeatureExtractor`):
            An instance of [`Pop2PianoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Pop2PianoTokenizer`):
            An instance of ['Pop2PianoTokenizer`]. The tokenizer is a required input.
    """
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "Pop2PianoFeatureExtractor"
    tokenizer_class = "Pop2PianoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    def __call__(
        self,
        audio: Union[np.ndarray, List[float], List[np.ndarray]] = None,
        sampling_rate: Union[int, List[int]] = None,
        steps_per_beat: int = 2,
        do_infer_resample: Optional[bool] = True,
        return_attention_mask: Optional[bool] = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        token_ids: Union[List, TensorType] = None,
        feature_extractor_output: BatchFeature = None,
        return_midi: bool = True,
        **kwargs,
    ) -> BatchFeature:
        """
        This method uses [`Pop2PianoFeatureExtractor.__call__`] method to prepare log-mel-spectrograms for the model,
        and [`Pop2PianoTokenizer.__call__`] to prepare pretty_midi objects from the model outputs.

        Please refer to the docstring of the above two methods for more information.
        """

        # Since Feature Extractor needs both audio and sampling_rate and tokenizer needs both token_ids and
        # feature_extractor_output, we must check for both.
        if (audio is None and sampling_rate is None) and (token_ids is None and feature_extractor_output is None):
            raise ValueError(
                "You have to specify at least audios and sampling_rate in order to use feature extractor or "
                "token_ids along with feature_extractor_output to use the tokenizer part."
            )

        encoding = BatchFeature()

        if audio is not None and sampling_rate is not None:
            extracted_features = self.feature_extractor(
                audio=audio,
                sampling_rate=sampling_rate,
                steps_per_beat=steps_per_beat,
                do_infer_resample=do_infer_resample,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors,
            )
            encoding.update(extracted_features)

        if token_ids is not None and feature_extractor_output is not None:
            tokenizer_outputs = self.tokenizer(
                token_ids=token_ids,
                feature_extractor_output=feature_extractor_output,
                return_midi=return_midi,
            )
            encoding.update(tokenizer_outputs)

        return encoding

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(*args)
