# Copyright 2026 the HuggingFace Team. All rights reserved.
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


import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...image_processing_utils import BatchFeature
from ...utils import (
    TensorType,
    is_torch_available,
)


if is_torch_available():
    import torch


class Gemma4UnifiedAudioFeatureExtractor(SequenceFeatureExtractor):
    """Encoder-free audio feature extractor that chunks raw waveform into frames.

    Unlike the standard Gemma4 audio feature extractor which computes mel spectrograms,
    this unified version simply chunks raw 16 kHz audio into fixed-length frames
    of `audio_samples_per_token` samples each. Each frame becomes a single audio
    soft token with the raw waveform samples as its features.

    Args:
        feature_size (`int`, *optional*, defaults to 640):
            The feature dimension of the extracted features (samples per token).
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio.
        audio_samples_per_token (`int`, *optional*, defaults to 640):
            Number of raw audio samples per output token. At 16 kHz, 640 samples = 40ms.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 640,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        audio_samples_per_token: int = 640,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.audio_samples_per_token = audio_samples_per_token

    def _extract_waveform_features(
        self,
        waveform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chunk a raw waveform into fixed-length frames.

        Each frame of `audio_samples_per_token` samples becomes one audio soft token.
        The waveform is zero-padded to be evenly divisible by the frame size.

        Args:
            waveform: 1-D array of raw audio samples.

        Returns:
            features: (num_tokens, audio_samples_per_token) array of waveform frames.
            mask: (num_tokens,) boolean array, True for all valid tokens.
        """
        # Pad waveform to be evenly divisible by samples_per_token
        pad_len = (-len(waveform)) % self.audio_samples_per_token
        if pad_len:
            waveform = np.pad(waveform, (0, pad_len))

        num_tokens = len(waveform) // self.audio_samples_per_token
        features = waveform.reshape(num_tokens, self.audio_samples_per_token).astype(np.float32)

        # All tokens are valid (padding is within the last frame, not creating extra frames)
        mask = np.ones(num_tokens, dtype=bool)
        return features, mask

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str = "longest",
        max_length: int | None = None,
        truncation: bool = True,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Chunk raw audio waveforms into fixed-length frames for the unified model.

        Args:
            raw_speech:
                The raw audio waveform(s) to process.
            padding (`str`, *optional*, defaults to `"longest"`):
                Padding strategy for batches with different lengths.
            max_length (`int`, *optional*):
                Maximum number of tokens to produce per audio.
            truncation (`bool`, *optional*, defaults to `True`):
                Whether to truncate audio above `max_length` tokens.
            return_tensors (`str`, *optional*):
                The type of tensors to return.
        """
        # Normalize input to list of 1-D arrays
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        elif not isinstance(raw_speech, (list, tuple)):
            raw_speech = [np.asarray(raw_speech)]
        else:
            raw_speech = [np.asarray(s) for s in raw_speech]

        # Extract features for each waveform
        all_features = [{"input_features": self._extract_waveform_features(waveform)[0]} for waveform in raw_speech]

        # Delegate padding and truncation to the parent class
        padded_inputs = self.pad(
            all_features,
            padding=padding,
            max_length=max_length,
            truncation=truncation and max_length is not None,
            return_attention_mask=True,
            return_tensors=return_tensors,
        )

        # Rename attention_mask → input_features_mask.
        # pad() produces int32 (0/1); downstream code expects a boolean mask for indexing.
        mask = padded_inputs.pop("attention_mask")
        if is_torch_available() and isinstance(mask, torch.Tensor):
            mask = mask.bool()
        else:
            mask = np.asarray(mask, dtype=bool)
        padded_inputs["input_features_mask"] = mask

        return padded_inputs


__all__ = ["Gemma4UnifiedAudioFeatureExtractor"]
