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
from .audio_processing_numpy_speech_to_text import SpeechToTextAudioProcessorNumpy


class SpeechToTextAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`SpeechToTextAudioProcessorNumpy`]. Per-waveform kaldi fbank features
    followed by per-utterance CMVN on the padded batch."""

    sample_rate = 16000
    force_mono = True
    do_batch_spectrogram = False

    # Single source of truth for the config lives on the numpy sibling (importable without torch).
    spectrogram_config = SpeechToTextAudioProcessorNumpy.spectrogram_config

    def __init__(self, normalize_means=True, normalize_vars=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars

    def extract_spectrogram(self, audio, **kwargs):
        # Native kaldi-exact pipeline (bit-equal to `torchaudio.compliance.kaldi.fbank`),
        # transposed to kaldi's (time, n_mels) orientation expected downstream.
        features = super().extract_spectrogram(audio, **kwargs)
        return [f.transpose(-2, -1) for f in features]

    @staticmethod
    def utterance_cmvn(x, input_length, normalize_means=True, normalize_vars=True, padding_value=0.0):
        # CMVN is computed in numpy to stay bit-exact with the legacy feature extractor
        # and the numpy sibling: numpy reductions use pairwise summation, whose
        # accumulation order differs from torch's `mean`/`std` (~1e-5 drift in float32).
        x = x.detach().cpu().numpy()
        if normalize_means:
            mean = x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        if normalize_vars:
            std = x[:input_length].std(axis=0)
            x = np.divide(x, std)
        if input_length < x.shape[0]:
            if not (normalize_means or normalize_vars):
                x = x.copy()  # don't mutate the caller's tensor through the numpy view
            x[input_length:] = padding_value
        return torch.from_numpy(x.astype(np.float32))

    def _postprocess_output(self, output, feature_ranges=None, **kwargs):
        # Apply utterance CMVN normalization on the padded, stacked features
        features = output["audio_features"]  # (batch, time, n_mels)
        normalized = []
        for i, (start, end) in enumerate(feature_ranges):
            length = end - start
            normalized.append(
                self.utterance_cmvn(features[i], length, self.normalize_means, self.normalize_vars, self.padding_value)
            )
        output["audio_features"] = torch.stack(normalized)
        return output


__all__ = ["SpeechToTextAudioProcessor"]
