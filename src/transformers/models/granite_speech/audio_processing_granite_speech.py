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

import math

import numpy as np

from ...audio_processing_base import BatchFeature
from ...audio_processing_utils import BaseAudioProcessor
from ...processing_utils import AudioKwargs


class GraniteSpeechAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    projector_window_size (`int`, *optional*, defaults to 15):
        Window size, in mel frames, consumed by the audio projector.
    projector_downsample_rate (`int`, *optional*, defaults to 5):
        Factor by which the audio projector downsamples its input.
    """

    projector_window_size: int
    projector_downsample_rate: int


class GraniteSpeechAudioProcessor(BaseAudioProcessor):
    sampling_rate = 16000
    # The workflow builds a mandatory projector mask rather than the default frame mask.
    extra_model_input_names = ["audio_features_mask", "audio_embed_sizes"]
    return_padding_mask = False

    # Native pipeline, bit-equal to the upstream FE's `torchaudio.transforms.MelSpectrogram`
    # + log10 + Whisper-style max-clip/rescale (ADR 0004 post-log fields).
    spectrogram_config = {
        "stft_config": {
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "power": 2.0,
        },
        "mel_scale_config": {"n_mels": 80},
        "log_mode": "log10",
        "mel_floor": 1e-10,
        "floor_below_peak": 8.0,
        "log_shift": 4.0,
        "log_scale": 0.25,
    }

    projector_window_size = 15
    projector_downsample_rate = 5
    valid_kwargs = GraniteSpeechAudioProcessorKwargs

    legacy_field_mapping = {"return_attention_mask": None, "return_padding_mask": None}

    def _preprocess(
        self,
        audio,
        *,
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        padding_side,
        padding_value,
        spectrogram_config,
        projector_window_size,
        projector_downsample_rate,
        return_tensors,
        **kwargs,
    ):
        """Pad waveforms, pair mel frames, and derive the projector mask in one workflow."""
        padded, ranges = self.pad(
            audio,
            padding,
            max_length,
            truncation,
            pad_to_multiple_of,
            padding_side=padding_side,
            padding_value=padding_value,
        )
        logmel = self.ops.swapaxes(
            self.spectrogram(
                self.ops.stack(padded), spectrogram_config=spectrogram_config, audio_ranges=ranges, **kwargs
            ),
            -1,
            -2,
        )
        if logmel.shape[1] % 2 == 1:
            logmel = logmel[:, :-1]
        features = self.ops.reshape(logmel, (logmel.shape[0], -1, 2 * logmel.shape[-1]))
        hop_length = spectrogram_config.stft_config.hop_length
        effective_window_size = projector_window_size // projector_downsample_rate
        sizes = []
        for start, end in ranges:
            mel_length = (end - start) // hop_length + 1
            nblocks = math.ceil((mel_length // 2) / projector_window_size)
            sizes.append(nblocks * effective_window_size)
        return BatchFeature(
            {
                "audio_features": features,
                "audio_embed_sizes": sizes,
                "audio_features_mask": self._embed_mask(sizes, features),
            },
            tensor_type=return_tensors,
        )

    def _embed_mask(self, sizes, like):
        """Boolean `(batch, max(sizes))` mask of each clip's projector outputs, on `like`'s device."""
        lengths = self.ops.asarray(np.asarray(sizes, dtype=np.int64), like=like)
        return self.ops.arange(max(sizes), like=like)[None, :] < lengths[:, None]


__all__ = ["GraniteSpeechAudioProcessor"]
