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
"""Feature extractor class for VibeVoice."""

from typing import Any, Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...utils import logging, TensorType, PaddingStrategy


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
    model_input_names = ["audio"]

    def __init__(
        self,
        sampling_rate: int = 24000,
        normalize_audio: bool = True,
        target_dB_FS: float = -25,
        eps: float = 1e-6,
        n_channels: int = 1,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(feature_size=n_channels, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.target_dB_FS = target_dB_FS
        self.eps = eps

        # Save config
        self.feature_extractor_dict = {
            "sampling_rate": sampling_rate,
            "normalize_audio": normalize_audio,
            "target_dB_FS": target_dB_FS,
            "eps": eps,
        }

    def __call__(
        self,
        audio: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
        sampling_rate: Optional[int] = None,
        padding: Optional[Union[bool, str, PaddingStrategy]] = True,
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

        is_batched = bool(
            isinstance(audio, (list, tuple)) and (isinstance(audio[0], (np.ndarray, tuple, list)))
        )

        # Ensure numpy, float32 arrays
        if is_batched:
            audio = [np.asarray(a, dtype=np.float32) for a in audio]
        elif not is_batched and not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        elif isinstance(audio, np.ndarray) and audio.dtype is np.dtype(np.float64):
            audio = audio.astype(np.float32)

        # Ensure batch
        if not is_batched:
            audio = [np.asarray(audio)]

        # Ensure mono
        for idx, example in enumerate(audio):
            if len(example.shape) == 2:
                if example.shape[0] == 2:  # (2, time)
                    audio[idx] = np.mean(example, axis=0)
                elif example.shape[1] == 2:  # (time, 2)
                    audio[idx] = np.mean(example, axis=1)
                else:
                    # If one dimension is 1, squeeze it
                    if example.shape[0] == 1:
                        audio[idx] = example.squeeze(0)
                    elif example.shape[1] == 1:
                        audio[idx] = example.squeeze(1)
                    else:
                        raise ValueError(f"Unexpected audio shape: {example.shape}")
            elif len(example.shape) != 1:
                raise ValueError(f"Audio should be 1D or 2D, got shape: {example.shape}")

        # Normalize
        if self.normalize_audio:
            audio = [normalize_audio(a, self.target_dB_FS, self.eps) for a in audio]

        output_values = BatchFeature({"audio": audio})
        if padding:
            output_values = self.pad(
                output_values,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask
            )
            output_values["padding_mask"] = output_values.pop("attention_mask")

        return output_values

    # Override to_dict method for configuration saving
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the object to a dict containing all attributes needed for serialization.
        """
        return self.feature_extractor_dict
    

def normalize_audio(audio: np.ndarray, target_dB_FS: float = -25, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize audio to target dB FS level and avoid clipping.
    
    Args:
        audio (np.ndarray): Input audio signal
        target_dB_FS (float): Target dB FS level for the audio. Default: -25
        eps (float): Small value to avoid division by zero. Default: 1e-6
        
    Returns:
        np.ndarray: Normalized audio signal
    """
    # Adjust to target dB FS
    rms = np.sqrt(np.mean(audio**2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    audio = audio * scalar

    # Avoid clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / (max_val + eps)

    return audio


__all__ = ["VibeVoiceFeatureExtractor"]
