# Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor for Fun-ASR-Nano (mel-spectrogram with LFR)."""

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import logging


logger = logging.get_logger(__name__)


class FunAsrNanoFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Fun-ASR-Nano feature extractor.

    This feature extractor extracts mel-spectrogram features and applies Low Frame Rate (LFR)
    processing by stacking consecutive frames and subsampling.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            Number of mel frequency bins.
        sampling_rate (`int`, *optional*, defaults to 16000):
            Sampling rate of the audio.
        frame_length (`int`, *optional*, defaults to 25):
            Frame length in milliseconds for STFT.
        frame_shift (`int`, *optional*, defaults to 10):
            Frame shift (hop length) in milliseconds for STFT.
        lfr_m (`int`, *optional*, defaults to 7):
            Number of consecutive frames to stack (LFR stacking factor).
        lfr_n (`int`, *optional*, defaults to 6):
            Subsampling stride for LFR (take every lfr_n-th stacked frame).
        window (`str`, *optional*, defaults to `"hamming"`):
            Window function for STFT.
        preemphasis (`float`, *optional*, defaults to 0.97):
            Pre-emphasis coefficient.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used for padding shorter sequences.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return an attention mask.

    Example:

    ```python
    >>> import numpy as np
    >>> from transformers import FunAsrNanoFeatureExtractor

    >>> feature_extractor = FunAsrNanoFeatureExtractor()
    >>> audio = np.random.randn(16000)  # 1 second of audio at 16kHz
    >>> features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    >>> features.input_features.shape  # (1, num_frames_after_lfr, 560)
    ```
    """

    model_input_names = ["input_features", "feature_lengths"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        window: str = "hamming",
        preemphasis: float = 0.97,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.window = window
        self.preemphasis = preemphasis
        self.n_fft = int(self.sampling_rate * self.frame_length / 1000)
        self.hop_length = int(self.sampling_rate * self.frame_shift / 1000)

    def _extract_fbank_features(self, waveform: np.ndarray) -> np.ndarray:
        """Extract mel-filterbank features from a single waveform.

        Args:
            waveform: 1D numpy array of audio samples.

        Returns:
            Mel-filterbank features of shape (num_frames, feature_size).
        """
        import torch
        import torchaudio

        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)

        # Extract fbank features using torchaudio
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform_tensor,
            num_mel_bins=self.feature_size,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            sample_frequency=self.sampling_rate,
            window_type=self.window,
        )

        return fbank.numpy()

    def _apply_lfr(self, features: np.ndarray) -> np.ndarray:
        """Apply Low Frame Rate (LFR) by stacking and subsampling frames.

        Args:
            features: Mel features of shape (num_frames, feature_size).

        Returns:
            LFR features of shape (num_lfr_frames, feature_size * lfr_m).
        """
        num_frames, feature_dim = features.shape
        lfr_m = self.lfr_m
        lfr_n = self.lfr_n

        # Pad features if necessary
        left_pad = (lfr_m - 1) // 2
        right_pad = lfr_m - 1 - left_pad

        # Repeat first and last frames for padding
        padded = np.concatenate(
            [
                np.tile(features[0:1], (left_pad, 1)),
                features,
                np.tile(features[-1:], (right_pad, 1)),
            ],
            axis=0,
        )

        # Stack frames
        num_output_frames = (num_frames + lfr_n - 1) // lfr_n
        lfr_features = []
        for i in range(num_output_frames):
            center = i * lfr_n
            start = center
            end = start + lfr_m
            if end <= len(padded):
                stacked = padded[start:end].reshape(-1)
            else:
                # Handle edge case
                chunk = padded[start:]
                pad_frames = lfr_m - len(chunk)
                chunk = np.concatenate([chunk, np.tile(chunk[-1:], (pad_frames, 1))], axis=0)
                stacked = chunk.reshape(-1)
            lfr_features.append(stacked)

        return np.array(lfr_features)

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
        padding: bool | str = True,
        max_length: int | None = None,
        truncation: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Extract features from raw audio waveforms.

        Args:
            raw_speech: Raw audio waveform(s). Can be a single array or a list of arrays.
            sampling_rate: Sampling rate of the input audio.
            return_tensors: Type of tensors to return ("pt", "np", "tf").
            padding: Whether to pad sequences to the same length.
            max_length: Maximum sequence length for padding/truncation.
            truncation: Whether to truncate sequences longer than max_length.

        Returns:
            BatchFeature with `input_features` and `feature_lengths`.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling rate {self.sampling_rate}, got {sampling_rate}. "
                f"Please resample your audio to {self.sampling_rate} Hz."
            )

        # Handle single input
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim == 1:
            raw_speech = [raw_speech]
        elif isinstance(raw_speech, list) and isinstance(raw_speech[0], (float, int)):
            raw_speech = [np.array(raw_speech, dtype=np.float32)]
        elif isinstance(raw_speech, list) and isinstance(raw_speech[0], np.ndarray):
            pass
        else:
            raw_speech = [np.array(s, dtype=np.float32) for s in raw_speech]

        # Extract features for each audio
        all_features = []
        feature_lengths = []
        for waveform in raw_speech:
            if isinstance(waveform, list):
                waveform = np.array(waveform, dtype=np.float32)

            # Extract mel features
            fbank = self._extract_fbank_features(waveform)

            # Apply LFR
            lfr_features = self._apply_lfr(fbank)

            all_features.append(lfr_features)
            feature_lengths.append(len(lfr_features))

        # Pad to max length
        max_feat_len = max(feature_lengths)
        padded_features = []
        for feat in all_features:
            if len(feat) < max_feat_len:
                pad_width = ((0, max_feat_len - len(feat)), (0, 0))
                feat = np.pad(feat, pad_width, mode="constant", constant_values=self.padding_value)
            padded_features.append(feat)

        padded_features = np.array(padded_features, dtype=np.float32)
        feature_lengths = np.array(feature_lengths, dtype=np.int64)

        encoded_inputs = {
            "input_features": padded_features,
            "feature_lengths": feature_lengths,
        }

        return BatchFeature(encoded_inputs, tensor_type=return_tensors)


__all__ = ["FunAsrNanoFeatureExtractor"]
