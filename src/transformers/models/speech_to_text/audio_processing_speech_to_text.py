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

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, spectrogram, window_function
from ...feature_extraction_utils import BatchFeature
from ...utils import is_speech_available


if is_speech_available():
    import torch
    import torchaudio.compliance.kaldi as ta_kaldi


class SpeechToTextAudioProcessor(NumpyAudioBackend):
    sample_rate = 16000
    force_mono = True

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
        mel_floor=1.192092955078125e-07,
        waveform_scale=32768.0,
    )

    def __init__(self, normalize_means=True, normalize_vars=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars
        if not is_speech_available():
            self.window = window_function(400, "povey", periodic=False)

    def _extract_fbank_features(self, waveform):
        waveform = waveform * (2**15)  # Kaldi compliance
        if is_speech_available():
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            features = ta_kaldi.fbank(waveform_tensor, num_mel_bins=80, sample_frequency=self.sample_rate)
            return features.numpy()
        else:
            waveform = np.squeeze(waveform)
            return spectrogram(
                waveform,
                self.window,
                frame_length=400,
                hop_length=160,
                fft_length=512,
                power=2.0,
                center=False,
                preemphasis=0.97,
                mel_filters=self.mel_filters,
                log_mel="log",
                mel_floor=1.192092955078125e-07,
                remove_dc_offset=True,
            ).T

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

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Extract features from raw (unpadded) audio, then pad at feature level
        features = [self._extract_fbank_features(waveform) for waveform in audio]
        lengths = [f.shape[0] for f in features]

        # Pad features to longest
        max_len = max(lengths)
        padded = []
        for f in features:
            if f.shape[0] < max_len:
                f = np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode="constant", constant_values=0.0)
            padded.append(f)

        # Utterance CMVN normalization
        normalized = [
            self.utterance_cmvn(f, length, self.normalize_means, self.normalize_vars, self.padding_value)
            for f, length in zip(padded, lengths)
        ]

        stacked = np.stack(normalized, axis=0)
        data = {"audio_features": stacked}

        if self.return_attention_mask:
            attention_mask = np.zeros((len(lengths), max_len), dtype=np.int32)
            for i, length in enumerate(lengths):
                attention_mask[i, :length] = 1
            data["audio_features_mask"] = attention_mask

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["SpeechToTextAudioProcessor"]
