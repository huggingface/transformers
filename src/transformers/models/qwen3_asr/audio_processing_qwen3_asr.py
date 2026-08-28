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

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...processing_utils import AudioKwargs


class Qwen3ASRAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    min_length (`int`, *optional*, defaults to 8000):
        Minimum waveform length in samples; shorter audio is padded up to this length.
    n_window (`int`, *optional*, defaults to 50):
        Encoder window size. Features are padded so their frame count is a multiple of
        `2 * n_window`.
    """

    min_length: int
    n_window: int | None


class Qwen3ASRAudioProcessorMixin:
    force_mono = True
    max_length = 480000
    padding = "max_length"
    sampling_rate = 16000
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
        clip_max_offset=8.0,
        post_log_shift=4.0,
        post_log_scale=0.25,
        # legacy masks by striding the sample mask -> boundary-straddling frames count as valid
        count_partial_frames=True,
    )

    min_length = 8000
    n_window = 50
    valid_kwargs = Qwen3ASRAudioProcessorKwargs

    def _extract_spectrogram(self, audio, *, spectrogram_config, **kwargs):
        features = super()._extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
        return features[..., :-1]

    def _get_mask_width(self, padded_length, spectrogram_config) -> int:
        # The legacy FE strides the sample-level mask by hop_length and trims the tail column
        return int(padded_length // spectrogram_config.stft_config.hop_length)

    def _postprocess_output(self, output, audio_ranges=None, n_window=None, **kwargs):
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

    def _process_audio(self, audio_el):
        audio_el = super()._process_audio(audio_el)
        if self.min_length and audio_el.shape[-1] < self.min_length:
            audio_el = self._pad_single(audio_el, self.min_length)
        return audio_el


class Qwen3ASRAudioProcessor(Qwen3ASRAudioProcessorMixin, TorchAudioBackend):
    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.to(device=features.device)
        return torch.clamp(torch.matmul(mel_filters.T, features), min=spectrogram_config.mel_floor)


__all__ = ["Qwen3ASRAudioProcessor"]
