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

from ...audio_processing_backends import TorchAudioBackend
from ...processing_utils import AudioKwargs


class SpeechToTextAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    normalize_means (`bool`, *optional*, defaults to `True`):
        Whether to mean-normalize the extracted features per utterance.
    normalize_vars (`bool`, *optional*, defaults to `True`):
        Whether to variance-normalize the extracted features per utterance.
    do_ceptral_normalize (`bool`, *optional*, defaults to `True`):
        Whether to apply utterance-level cepstral mean and variance normalization at all. The
        legacy extractor gates the whole CMVN block on this, above `normalize_means`/
        `normalize_vars`; without it a checkpoint that disables normalization was normalized anyway.
    """

    normalize_means: bool
    normalize_vars: bool
    do_ceptral_normalize: bool


class SpeechToTextAudioProcessorMixin:
    do_ceptral_normalize = True
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
            "left_align_fft": True,
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
        "mel_floor": 1.192092955078125e-07,
        "waveform_scale": 32768.0,
        "transpose_features": True,  # kaldi's (time, n_mels) orientation
    }

    normalize_means = True
    normalize_vars = True
    valid_kwargs = SpeechToTextAudioProcessorKwargs

    def _compute_batched_features(
        self,
        audio,
        *,
        audio_ranges,
        spectrogram_config,
        do_ceptral_normalize,
        normalize_means,
        normalize_vars,
        padding_value,
        **kwargs,
    ):
        features, frame_counts = super()._compute_batched_features(
            audio,
            audio_ranges=audio_ranges,
            spectrogram_config=spectrogram_config,
            padding_value=padding_value,
            **kwargs,
        )
        if do_ceptral_normalize:
            features = self._utterance_cmvn(
                features, frame_counts, normalize_means=normalize_means, normalize_vars=normalize_vars
            )
        # the legacy extractor padded the features, not the audio
        return self._mask_padded_frames(features, frame_counts, padding_value=padding_value), frame_counts

    def _utterance_cmvn(self, features, frame_counts, *, normalize_means, normalize_vars):
        """Cepstral mean and variance normalization of each clip over its own frames (biased std)."""
        features, mean, variance = self._frame_moments(features, frame_counts, ddof=0)
        if normalize_means:
            features = features - mean
        if normalize_vars:
            features = features / self._sqrt(variance)
        return self._astype(features, "float32")


class SpeechToTextAudioProcessor(SpeechToTextAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["SpeechToTextAudioProcessor"]
