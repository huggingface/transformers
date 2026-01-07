# coding=utf-8
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

from typing import Optional, Union

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class VibeVoiceFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a VibeVoice feature extractor.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The number of channels.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used for padding.
        normalize_audio (`bool`, *optional*, defaults to `True`):
            Whether to normalize audio to a target dB FS.
        target_dB_FS (`float`, *optional*, defaults to -25):
            Target dB FS for normalization.
        eps (`float`, *optional*, defaults to 1e-06):

    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=24000,
        padding_value=0.0,
        normalize_audio=True,
        target_dB_FS=-25,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.normalize_audio = normalize_audio
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: Optional[int] = None,
        padding: Optional[Union[bool, str, PaddingStrategy]] = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = True,
        max_length: Optional[int] = None,
    ) -> BatchFeature:
        """
        Main method to prepare audio for the VibeVoice model.

        Args:
            audio: Audio input(s) to process. Can be:
                - np.ndarray: Audio array
                - List[float]: Audio as list of floats
                - List[np.ndarray]: Batch of audio arrays
                - List[list[float]]: Batch of audio as lists of floats
            sampling_rate (int, optional): Sampling rate of the input audio
            return_tensors (str, optional): Return format ('pt' for PyTorch, 'np' for NumPy)

        Returns:
            dict: Processed audio inputs with keys:
                - input_features: Audio tensor(s) ready for the model
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        # Ensure batch
        audio = make_list_of_audio(audio)

        # Ensure numpy float arrays and mono
        for idx, example in enumerate(audio):
            example = np.asarray(example, dtype=np.float32)
            if example.ndim != 1:
                raise ValueError(f"Audio should be mono, got shape: {example.shape}")
            audio[idx] = example

        if self.normalize_audio:
            for idx, example in enumerate(audio):
                rms = np.sqrt(np.mean(example**2))
                scalar = 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
                example = example * scalar
                max_val = np.max(np.abs(example))
                if max_val > 1.0:
                    example = example / (max_val + self.eps)
                audio[idx] = example

        output_values = BatchFeature({"input_features": audio})
        if padding:
            output_values = self.pad(
                output_values,
                max_length=max_length,
                return_tensors=return_tensors,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                padding=padding,
            )
            if return_attention_mask:
                output_values["input_features_mask"] = output_values.pop("attention_mask")

        # add channel dimension if missing
        if output_values["input_features"].ndim == 2:
            output_values["input_features"] = output_values["input_features"][:, None, :]

        return output_values


__all__ = ["VibeVoiceFeatureExtractor"]
