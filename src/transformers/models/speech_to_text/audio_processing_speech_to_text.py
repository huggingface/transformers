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


class SpeechToTextAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    normalize_means (`bool`, *optional*, defaults to `True`):
        Whether to mean-normalize the extracted features per utterance.
    normalize_vars (`bool`, *optional*, defaults to `True`):
        Whether to variance-normalize the extracted features per utterance.
    """

    normalize_means: bool
    normalize_vars: bool


class SpeechToTextAudioProcessorMixin:
    do_batch_spectrogram = False
    force_mono = True
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

    normalize_means = True
    normalize_vars = True
    valid_kwargs = SpeechToTextAudioProcessorKwargs


class SpeechToTextAudioProcessor(SpeechToTextAudioProcessorMixin, TorchAudioBackend):
    @staticmethod
    def utterance_cmvn(x, input_length, normalize_means=True, normalize_vars=True, padding_value=0.0):
        # CMVN is computed in numpy to stay bit-exact with the legacy feature extractor
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
                x = x.copy()
            x[input_length:] = padding_value
        return torch.from_numpy(x.astype(np.float32))

    def _postprocess_output(self, output, feature_ranges=None, **kwargs):
        features = output["audio_features"]
        normalized = []
        for i, (start, end) in enumerate(feature_ranges):
            length = end - start
            normalized.append(
                self.utterance_cmvn(features[i], length, self.normalize_means, self.normalize_vars, self.padding_value)
            )
        output["audio_features"] = torch.stack(normalized)
        return output


__all__ = ["SpeechToTextAudioProcessor"]
