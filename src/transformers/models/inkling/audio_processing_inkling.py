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

import math

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, _clamp_min


class InklingAudioProcessorMixin:
    sampling_rate = 16000
    force_mono = True
    model_input_names = ["input_features", "input_features_mask"]
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=1600,
            hop_length=800,
            win_length=1600,
            window_fn="hann_window",
            power=1.0,
            center=False,
            periodic=True,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            norm="slaney",
            mel_scale="slaney",
        ),
        log_mode="log10",
        mel_floor=1e-10,
        transpose_features=True,
        # `_stft` left-pads by `n_fft - hop`, so frame k is centred on sample k*hop and counts as
        # valid whenever that centre lies in the real audio, even if its window reaches padding.
        count_partial_frames=True,
    )

    def _stft(self, audio, *, spectrogram_config, audio_ranges=None, **kwargs):
        # Inkling's fixed framing: left-pad (n_fft - hop) and right-pad up to a hop multiple, center=False.
        stft_cfg = spectrogram_config.stft_config
        hop, n_fft = stft_cfg.hop_length, stft_cfg.n_fft
        right_pad = math.ceil(audio.shape[-1] / hop) * hop - audio.shape[-1]
        audio = self._pad_axis(audio, max(n_fft - hop, 0), right_pad, axis=-1)
        return super()._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        magnitudes = _clamp_min(stft_out.real**2 + stft_out.imag**2, 1e-10) ** 0.5
        return magnitudes**power if power != 1.0 else magnitudes

    def _get_mask_width(self, padded_length, spectrogram_config) -> int:
        # Inkling right-pads to a whole number of hops, so it emits ceil(length / hop) frames.
        hop = spectrogram_config.stft_config.hop_length
        return int((padded_length + hop - 1) // hop)

    def _postprocess_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        # No normalization; zero padded frames and emit the legacy keys the model consumes.
        # The mask is named `input_features_mask` so it doesn't collide with a text `attention_mask`.
        features = output.pop("audio_features")
        mask = output.pop("audio_features_mask", None)
        if mask is not None:
            # the mask is 0/1, so the numpy int-promotion round-trip is exact
            features = self._astype(features * mask[..., None], "float32")
            output["input_features_mask"] = mask
        output["input_features"] = features
        return output


class InklingAudioProcessor(InklingAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["InklingAudioProcessor"]
