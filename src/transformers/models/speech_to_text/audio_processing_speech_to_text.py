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
from ...audio_processing_base import make_legacy_audio_processor_alias
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class SpeechToTextAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`SpeechToTextAudioProcessorNumpy`]. Per-waveform kaldi fbank features
    followed by per-utterance CMVN on the padded batch."""

    sample_rate = 16000
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
    )
    waveform_scale = 32768.0

    def __init__(self, normalize_means=True, normalize_vars=True, **kwargs):
        super().__init__(**kwargs)
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars

    def _torch_kaldi_fbank(self, waveform):
        """Extract kaldi-compatible fbank features for a single waveform, returning a torch tensor.

        Uses `torchaudio.compliance.kaldi.fbank` when available; otherwise falls back to the
        base spectrogram pipeline. Mirrors the numpy sibling's `_kaldi_fbank` helper.
        """
        from ...utils import is_speech_available

        if is_speech_available():
            import torchaudio.compliance.kaldi as ta_kaldi

            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            return ta_kaldi.fbank(waveform, num_mel_bins=80, sample_frequency=self.sample_rate)

        # Fallback: use the base STFT pipeline on a single waveform.
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        waveform = waveform.squeeze()
        features = self.extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)
        return features[0].transpose(-2, -1)

    def _extract_fbank_features(self, waveform):
        """Extract log-mel filterbank features for a single waveform."""
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        waveform = waveform * self.waveform_scale
        return self._torch_kaldi_fbank(waveform)

    def extract_spectrogram(self, audio, **kwargs):
        # Per-waveform fbank extraction returning (time, n_mels)
        return [self._extract_fbank_features(waveform) for waveform in audio]

    @staticmethod
    def utterance_cmvn(x, input_length, normalize_means=True, normalize_vars=True, padding_value=0.0):
        if normalize_means:
            mean = x[:input_length].mean(dim=0)
            x = x - mean
        if normalize_vars:
            # numpy's `std` defaults to ddof=0; torch's defaults to unbiased=True (ddof=1).
            # Match the numpy sibling (ADR 0001).
            std = x[:input_length].std(dim=0, unbiased=False)
            x = x / std
        if input_length < x.shape[0]:
            x = x.clone()
            x[input_length:] = padding_value
        return x.to(torch.float32)

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


Speech2TextFeatureExtractor = make_legacy_audio_processor_alias(SpeechToTextAudioProcessor, "Speech2TextFeatureExtractor")


__all__ = ["SpeechToTextAudioProcessor", "Speech2TextFeatureExtractor"]
