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
    """
    Feature extractor for VibeVoice acoustic tokenizer models.

    This feature extractor handles audio preprocessing for VibeVoice models, including:
    - Audio format conversion (stereo to mono)
    - Optional audio normalization
    - Streaming support for infinite-length audio
    
    Args:
        sampling_rate (int, optional): Expected sampling rate. Defaults to 24000.
        normalize_audio (bool, optional): Whether to normalize audio. Defaults to True.
        target_dB_FS (float, optional): Target dB FS for normalization. Defaults to -25.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.
    """
    model_input_names = ["input_features"]

    def __init__(
        self,
        sampling_rate: int = 24000,
        normalize_audio: bool = True,
        target_dB_FS: float = -25,
        eps: float = 1e-6,
        feature_size: int = 1,
        padding_value: float = 0.0,
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
    ) -> BatchFeature:
        """
        Process audio for VibeVoice models.
        
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

        # Ensure numpy arrays and mono
        for idx, example in enumerate(audio):
            if not isinstance(example, np.ndarray):
                example = np.asarray(example, dtype=np.float32)
            elif example.dtype != np.float32:
                example = example.astype(np.float32)
            if example.ndim == 1:
                audio[idx] = example  # Already mono
            elif example.ndim == 2:
                # Convert to mono by averaging across channels (works for any channel dimension)
                audio[idx] = np.mean(example, axis=0 if example.shape[0] <= 2 else 1)
            else:
                raise ValueError(f"Audio should be 1D or 2D, got shape: {example.shape}")

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
                return_tensors=return_tensors,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask
            )
            output_values["input_features_mask"] = output_values.pop("attention_mask")

        return output_values


__all__ = ["VibeVoiceFeatureExtractor"]
