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

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_processing_utils import AudioKwargs
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


def _qwen3_asr_chunk_length_to_max_length(value, config_dict):
    # Legacy Qwen3 ASR hub configs store `chunk_length=30` (seconds); the new API uses
    # `max_length` in samples. Translate using the sampling rate (already translated to
    # `sample_rate` by the base mapping by the time this runs, or still the legacy key
    # if not yet processed).
    sample_rate = config_dict.get("sample_rate") or config_dict.get("sampling_rate") or 16000
    config_dict.setdefault("max_length", value * sample_rate)


class Qwen3ASRAudioKwargs(AudioKwargs, total=False):
    n_window: int | None


class Qwen3ASRAudioProcessorNumpy(NumpyAudioBackend):
    """NumPy sibling of [`Qwen3ASRAudioProcessor`]. Required to produce bit-exact outputs
    against the torch sibling (ADR 0001).

    Whisper-style 128-bin log-mel features with three Qwen3-ASR-specific twists:

    - clips shorter than ``min_length`` samples are zero-padded up to it (and counted as
      valid in the padding mask, matching the original Qwen3-ASR library),
    - the padding mask lives on the mel-frame axis (sample mask strided by ``hop_length``),
    - the mel time axis (features and mask) is right-padded to a multiple of
      ``2 * n_window`` frames, as required by ``Qwen3ASREncoder``'s chunked attention.
    """

    sample_rate = 16000
    force_mono = True
    padding = "max_length"
    max_length = 480000  # 30 seconds at 16000 Hz
    min_length = 8000
    n_window = 50
    valid_kwargs = Qwen3ASRAudioKwargs

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=400,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            mel_scale="slaney",
            norm="slaney",
            computation_dtype="float64",
        ),
        log_mode="log10",
        # NOTE: the legacy FE's `skip_last_frame` is implemented in `_extract_spectrogram`
        # (pre-mel) rather than via the config flag (post-mel): the mel matmul is
        # shape-sensitive at 1 ulp (BLAS blocking), so the center frame must be dropped
        # before the projection to stay bit-exact with the legacy FE.
        clip_max_offset=8.0,
        post_log_shift=4.0,
        post_log_scale=0.25,
    )

    legacy_field_mapping = {
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
        "chunk_length": _qwen3_asr_chunk_length_to_max_length,
        "n_samples": "max_length",
        "nb_max_frames": None,
    }

    def __init__(self, min_length: int = 8000, **kwargs):
        super().__init__(**kwargs)
        self.min_length = min_length

    def _process_audio(self, audio_el):
        audio_el = super()._process_audio(audio_el)
        # Zero-pad clips shorter than `min_length`, matching the original Qwen3-ASR library.
        # Done before batch padding so the padded samples count as valid in the mask
        # (as original: do not adjust the mask, it hurts performance on AMI).
        if self.min_length and audio_el.shape[-1] < self.min_length:
            audio_el = self._pad_single(audio_el, self.min_length)
        return audio_el

    def _extract_spectrogram(self, audio, *, spectrogram_config, **kwargs):
        features = super()._extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
        # Drop the trailing center frame *before* the mel projection, like the legacy FE
        # (`stft[..., :-1]`); see the NOTE on `spectrogram_config`.
        return features[..., :-1]

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # `filters_first` matmul order with mel_floor clamp, matching the torch sibling.
        return np.maximum(spectrogram_config.mel_floor, np.matmul(self.mel_filters.T, features))

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        hop_length = spectrogram_config.stft_config.hop_length
        if include_center_frame:
            # Mask width over the padded batch: the legacy FE strides the sample-level mask
            # by hop_length and trims the tail column when it doesn't divide evenly, i.e.
            # padded_length // hop_length — the feature width after `skip_last_frame`.
            return audio_lengths // hop_length
        # Per-utterance valid frames: strided sample-mask indices 0, hop, 2*hop, ... below
        # the valid length, i.e. ceil(length / hop_length).
        return (audio_lengths + hop_length - 1) // hop_length

    def _postprocess_output(self, output, audio_ranges=None, n_window=None, **kwargs):
        # Right-pad the mel time axis (features and mask) to a multiple of `2 * n_window`
        # (needed by `Qwen3ASREncoder`). `n_window=0` disables this padding.
        if n_window is None:
            n_window = self.n_window
        multiple = 2 * n_window if n_window else 0
        if multiple > 1:
            features = output["audio_features"]
            remainder = features.shape[-1] % multiple
            if remainder:
                padded_length = features.shape[-1] + multiple - remainder
                output["audio_features"] = self._pad_single(features, padded_length)
                if "audio_features_mask" in output:
                    output["audio_features_mask"] = self._pad_single(output["audio_features_mask"], padded_length)
        return output


__all__ = ["Qwen3ASRAudioProcessorNumpy"]
