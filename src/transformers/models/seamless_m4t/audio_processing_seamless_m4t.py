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
from .audio_processing_numpy_seamless_m4t import SeamlessM4tAudioProcessorNumpy


class SeamlessM4tAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    do_batch_spectrogram = False
    stride = 2
    pad_to_multiple_of = 2  # Align feature padding to stride


    spectrogram_config = SeamlessM4tAudioProcessorNumpy.spectrogram_config
    waveform_scale = 32768.0

    def extract_spectrogram(self, audio, **kwargs):
        # Per-waveform fbank extraction returning (time, n_mels)
        features = []
        for waveform in audio:
            waveform = waveform.squeeze() * self.waveform_scale
            f = super().extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)
            features.append(f[0].transpose(-2, -1))
        return features

    # Mel filters: the base dispatcher resolves the top-level `computation_dtype="float64"`
    # into a float64 torch-native kaldi-exact build, which is bit-identical to the legacy
    # FE's float64 numpy build (the float32 mel-space cancellation that motivated the old
    # numpy-built override only appears in the default float32 kaldi path).

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg, audio_dtype=None):
        spec = super()._window_and_fft(frames, window, frame_length, n_fft, stft_cfg, audio_dtype=audio_dtype)
        # The legacy FE stores FFT frames in a complex64 buffer before taking float64
        # magnitudes (`np.abs(spectrogram, dtype=np.float64) ** power`); quantize then upcast
        # so `_compute_magnitudes` sees the same values.
        return spec.to(torch.complex64).to(torch.complex128)

    def _postprocess_features(self, features, feature_lengths):
        # Per-utterance mean/variance normalization (before padding). Computed in numpy to stay
        # bit-exact with the legacy FE: numpy reductions use pairwise summation, whose
        # accumulation order differs from torch's float32 `mean`/`var`. The legacy features are
        # F-contiguous (a `.T` view of the (n_mels, time) spectrogram) and numpy's accumulation
        # order depends on memory layout, so match it with `asfortranarray`.
        normalized = []
        for f in features:
            x = np.asfortranarray(f.detach().cpu().numpy())
            x = (x - np.expand_dims(x.mean(0), 0)) / np.sqrt(np.expand_dims(x.var(0, ddof=1), 0) + 1e-7)
            normalized.append(torch.from_numpy(x))
        return normalized

    def _postprocess_output(self, output, feature_ranges=None, **kwargs):
        features = output["audio_features"]  # (batch, num_frames, num_channels)
        batch_size, num_frames, num_channels = features.shape

        # Stride concatenation
        remainder = num_frames % self.stride
        if remainder != 0:
            features = features[:, :num_frames - remainder, :]
            num_frames = num_frames - remainder

        output["audio_features"] = features.reshape(batch_size, num_frames // self.stride, num_channels * self.stride)

        # Adjust mask for stride
        if "audio_features_mask" in output:
            mask = output["audio_features_mask"]
            if remainder != 0:
                mask = mask[:, :num_frames]
            indices = torch.arange(0, num_frames)
            output["audio_features_mask"] = mask[:, indices % self.stride == 1]

        return output


__all__ = ["SeamlessM4tAudioProcessor"]
