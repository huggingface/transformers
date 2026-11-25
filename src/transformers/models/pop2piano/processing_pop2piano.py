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
"""Processor class for Pop2Piano."""

import os

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_python import BatchEncoding, PaddingStrategy, TruncationStrategy
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires


@requires(backends=("essentia", "librosa", "pretty_midi", "scipy", "torch"))
@auto_docstring
class Pop2PianoProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        audio: np.ndarray | list[float] | list[np.ndarray] = None,
        sampling_rate: int | list[int] | None = None,
        steps_per_beat: int = 2,
        resample: bool | None = True,
        notes: list | TensorType = None,
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy = None,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> BatchFeature | BatchEncoding:
        # Since Feature Extractor needs both audio and sampling_rate and tokenizer needs both token_ids and
        # feature_extractor_output, we must check for both.
        r"""
        sampling_rate (`int` or `list[int]`, *optional*):
            The sampling rate of the input audio in Hz. This should match the sampling rate used by the feature
            extractor. If not provided, the default sampling rate from the processor configuration will be used.
        steps_per_beat (`int`, *optional*, defaults to `2`):
            The number of time steps per musical beat. This parameter controls the temporal resolution of the
            musical representation. A higher value provides finer temporal granularity but increases the sequence
            length. Used when processing audio to extract musical features.
        notes (`list` or `TensorType`, *optional*):
            Pre-extracted musical notes in MIDI format. When provided, the processor skips audio feature extraction
            and directly processes the notes through the tokenizer. Each note should be represented as a list or
            tensor containing pitch, velocity, and timing information.
        """
        if (audio is None and sampling_rate is None) and (notes is None):
            raise ValueError(
                "You have to specify at least audios and sampling_rate in order to use feature extractor or "
                "notes to use the tokenizer part."
            )

        if audio is not None and sampling_rate is not None:
            inputs = self.feature_extractor(
                audio=audio,
                sampling_rate=sampling_rate,
                steps_per_beat=steps_per_beat,
                resample=resample,
                **kwargs,
            )
        if notes is not None:
            encoded_token_ids = self.tokenizer(
                notes=notes,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                verbose=verbose,
                **kwargs,
            )

        if notes is None:
            return inputs

        elif audio is None or sampling_rate is None:
            return encoded_token_ids

        else:
            inputs["token_ids"] = encoded_token_ids["token_ids"]
            return inputs

    def batch_decode(
        self,
        token_ids,
        feature_extractor_output: BatchFeature,
        return_midi: bool = True,
    ) -> BatchEncoding:
        """
        This method uses [`Pop2PianoTokenizer.batch_decode`] method to convert model generated token_ids to midi_notes.

        Please refer to the docstring of the above two methods for more information.
        """

        return self.tokenizer.batch_decode(
            token_ids=token_ids, feature_extractor_output=feature_extractor_output, return_midi=return_midi
        )

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(*args)


__all__ = ["Pop2PianoProcessor"]
