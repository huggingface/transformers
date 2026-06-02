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

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_processing_base import make_legacy_audio_processor_alias
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class SeamlessM4tAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    do_batch_spectrogram = False
    stride = 2
    pad_to_multiple_of = 2  # Align feature padding to stride

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
        computation_dtype="float64",
    )
    waveform_scale = 32768.0

    def extract_spectrogram(self, audio, **kwargs):
        # Per-waveform fbank extraction returning (time, n_mels)
        features = []
        for waveform in audio:
            waveform = waveform.squeeze() * self.waveform_scale
            f = super().extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)
            features.append(f[0].transpose(-2, -1))
        return features

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        # `computation_dtype="float64"` upcasts the magnitudes but the kaldi-exact mel filter
        # bank is built in float32; cast the filters to match the features dtype, mirroring the
        # numpy sibling's `mel_filters.astype(features.dtype)` pattern.
        mel_filters = self.mel_filters.to(device=features.device, dtype=features.dtype)
        if spectrogram_config.mel_scale_config.matmul_order == "features_first":
            mel_spec = torch.matmul(features.transpose(-2, -1), mel_filters)
        else:
            mel_spec = torch.nn.functional.linear(features.transpose(-2, -1), mel_filters.T).transpose(-2, -1)
        return torch.clamp(mel_spec, min=spectrogram_config.mel_floor)

    def _postprocess_features(self, features, feature_lengths):
        # Per-utterance mean/variance normalization (before padding). `torch.var(unbiased=True)`
        # matches numpy `var(ddof=1)`.
        normalized = []
        for f in features:
            mean = f.mean(dim=0).unsqueeze(0)
            var = f.var(dim=0, unbiased=True).unsqueeze(0)
            normalized.append((f - mean) / torch.sqrt(var + 1e-7))
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


SeamlessM4TFeatureExtractor = make_legacy_audio_processor_alias(
    SeamlessM4tAudioProcessor, "SeamlessM4TFeatureExtractor"
)


__all__ = ["SeamlessM4tAudioProcessor", "SeamlessM4TFeatureExtractor"]
