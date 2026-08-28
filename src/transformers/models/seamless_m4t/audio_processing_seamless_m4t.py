# Copyright 2025 The HuggingFace Inc. team.
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
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...processing_utils import AudioKwargs


class SeamlessM4tAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    stride (`int`, *optional*, defaults to 2):
        Number of consecutive mel frames stacked into each output frame.
    """

    stride: int


class SeamlessM4tAudioProcessorMixin:
    do_batch_spectrogram = False
    force_mono = True
    pad_to_multiple_of = 2
    sampling_rate = 16000
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn="povey",
            power=2.0,
            center=False,
            periodic=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=20.0,
            f_max=8000.0,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        ),
        log_mode="log",
        preemphasis=0.97,
        remove_dc_offset=True,
        waveform_scale=32768.0,
        mel_floor=1.192092955078125e-07,
        computation_dtype="float64",
    )

    stride = 2
    valid_kwargs = SeamlessM4tAudioProcessorKwargs


class SeamlessM4tAudioProcessor(SeamlessM4tAudioProcessorMixin, TorchAudioBackend):
    def extract_spectrogram(self, audio, **kwargs):
        features = []
        for waveform in audio:
            waveform = waveform.squeeze()
            f = super().extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)
            features.append(f[0].transpose(-2, -1))
        return features

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg, audio_dtype=None):
        spec = super()._window_and_fft(frames, window, frame_length, n_fft, stft_cfg, audio_dtype=audio_dtype)
        # The legacy FE stores FFT frames in a complex64 buffer before taking float64
        # magnitudes (`np.abs(spectrogram, dtype=np.float64) ** power`); quantize then upcast
        return spec.to(torch.complex64).to(torch.complex128)

    def _postprocess_features(self, features, feature_lengths):
        # bit-exact with the legacy FE: numpy reductions use pairwise summation, whose
        # accumulation order differs from torch's float32 `mean`/`var`. The legacy features are
        normalized = []
        for f in features:
            x = np.asfortranarray(f.detach().cpu().numpy())
            x = (x - np.expand_dims(x.mean(0), 0)) / np.sqrt(np.expand_dims(x.var(0, ddof=1), 0) + 1e-7)
            normalized.append(torch.from_numpy(x))
        return normalized

    def _postprocess_output(self, output, feature_ranges=None, **kwargs):
        features = output["audio_features"]
        batch_size, num_frames, num_channels = features.shape

        remainder = num_frames % self.stride
        if remainder != 0:
            features = features[:, : num_frames - remainder, :]
            num_frames = num_frames - remainder

        output["audio_features"] = features.reshape(batch_size, num_frames // self.stride, num_channels * self.stride)

        if "audio_features_mask" in output:
            mask = output["audio_features_mask"]
            if remainder != 0:
                mask = mask[:, :num_frames]
            indices = torch.arange(0, num_frames)
            output["audio_features_mask"] = mask[:, indices % self.stride == 1]

        return output


__all__ = ["SeamlessM4tAudioProcessor"]
