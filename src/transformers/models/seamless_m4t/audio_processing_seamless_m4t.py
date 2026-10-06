# Copyright 2026 The HuggingFace Inc. team.
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

from typing import Annotated

from ...audio_processing_backends import TorchAudioBackend
from ...processing_utils import AudioKwargs
from ...utils.type_validators import strictly_positive


class SeamlessM4tAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    stride (`int`, *optional*, defaults to 2):
        Number of consecutive mel frames stacked into each output frame.
    """

    stride: Annotated[int, strictly_positive]


class SeamlessM4tAudioProcessorMixin:
    sampling_rate = 16000
    spectrogram_config = {
        "stft_config": {
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "window_fn": "povey",
            "power": 2.0,
            "center": False,
            "periodic": False,
            "fft_dtype": "complex64",
        },
        "mel_scale_config": {
            "n_mels": 80,
            "f_min": 20.0,
            "f_max": 8000.0,
            "mel_scale": "kaldi",
            "triangularize_in_mel_space": True,
        },
        "log_mode": "log",
        "preemphasis": 0.97,
        "remove_dc_offset": True,
        "waveform_scale": 32768.0,
        "mel_floor": 1.192092955078125e-07,
        "computation_dtype": "float64",
        "transpose_features": True,
    }

    stride = 2
    valid_kwargs = SeamlessM4tAudioProcessorKwargs

    def _compute_batched_features(self, audio, *, audio_ranges, spectrogram_config, stride, padding_value, **kwargs):
        features, frame_counts = super()._compute_batched_features(
            audio,
            audio_ranges=audio_ranges,
            spectrogram_config=spectrogram_config,
            padding_value=padding_value,
            **kwargs,
        )
        # The legacy extractor normalized each clip, padded the features, then padded the frame axis to an
        # even count (its `pad_to_multiple_of=2`) so the stride stacking below keeps the last frame.
        features = self._mask_padded_frames(
            self._normalize_utterances(features, frame_counts), frame_counts, padding_value=padding_value
        )
        if features.shape[1] % stride:
            features = self._pad_axis(features, 0, -features.shape[1] % stride, axis=1, value=padding_value)
        return features, frame_counts

    def _normalize_utterances(self, features, frame_counts):
        """Zero-mean, unit-variance each clip over its own frames (unbiased variance, `1e-7` inside the root)."""
        features, mean, variance = self._frame_moments(features, frame_counts, ddof=1)
        return self._astype((features - mean) / self._sqrt(variance + 1e-7), "float32")

    def _padded_frame_count(self, padded_length, spectrogram_config, *, stride, **kwargs) -> int:
        count = super()._padded_frame_count(padded_length, spectrogram_config, **kwargs)
        return count - count % -stride

    def _finalize_output(self, output, feature_ranges=None, *, stride, **kwargs):
        features = output["audio_features"]
        batch_size, num_frames, num_channels = features.shape

        remainder = num_frames % stride
        if remainder != 0:
            features = features[:, : num_frames - remainder, :]
            num_frames = num_frames - remainder

        output["audio_features"] = features.reshape(batch_size, num_frames // stride, num_channels * stride)

        if "audio_features_mask" in output:
            mask = output["audio_features_mask"]
            if remainder != 0:
                mask = mask[:, :num_frames]
            output["audio_features_mask"] = mask[:, stride - 1 :: stride]

        return output


class SeamlessM4tAudioProcessor(SeamlessM4tAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["SeamlessM4tAudioProcessor"]
