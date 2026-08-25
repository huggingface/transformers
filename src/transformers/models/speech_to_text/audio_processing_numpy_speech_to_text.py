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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class SpeechToTextAudioProcessorNumpy(NumpyAudioBackend):
    """NumPy sibling of [`SpeechToTextAudioProcessor`]. Per-waveform kaldi-style fbank
    features from the native pipeline described by `spectrogram_config` (no torchaudio
    dependency), followed by per-utterance CMVN on the padded batch.

    The legacy FE calls `torchaudio.compliance.kaldi.fbank`, so bit-exactness with it is a
    torch-only property (the torch sibling has it). numpy's FFT and window differ from torch's
    in the last float32 ulp, which `log` amplifies in near-silent mel bins."""

    sampling_rate = 16000
    force_mono = True
    do_batch_spectrogram = False

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn="povey",
            power=2.0,
            center=False,
            periodic=False,
            left_align_fft=True,
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
        mel_floor=1.192092955078125e-07,
        waveform_scale=32768.0,
        transpose_features=True,  # kaldi's (time, n_mels) orientation
    )

    def __init__(self, normalize_means=True, normalize_vars=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars

    @staticmethod
    def utterance_cmvn(x, input_length, normalize_means=True, normalize_vars=True, padding_value=0.0):
        if normalize_means:
            mean = x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        if normalize_vars:
            std = x[:input_length].std(axis=0)
            x = np.divide(x, std)
        if input_length < x.shape[0]:
            x[input_length:] = padding_value
        return x.astype(np.float32)

    def _postprocess_output(self, output, feature_ranges=None, **kwargs):
        # Apply utterance CMVN normalization on the padded, stacked features
        features = output["audio_features"]  # (batch, time, n_mels)
        normalized = []
        for i, (start, end) in enumerate(feature_ranges):
            length = end - start
            normalized.append(
                self.utterance_cmvn(features[i], length, self.normalize_means, self.normalize_vars, self.padding_value)
            )
        output["audio_features"] = np.stack(normalized)
        return output


__all__ = ["SpeechToTextAudioProcessorNumpy"]
