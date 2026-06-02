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


class AudioSpectrogramTransformerAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    return_padding_mask = False
    do_batch_spectrogram = False

    max_length_frames = 1024
    do_normalize = True

    # AudioSet normalization constants
    ast_mean = -4.2677393
    ast_std = 4.5689974

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn="hann_window",
            power=2.0,
            center=False,
            periodic=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
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

    def extract_spectrogram(self, audio, **kwargs):
        import torchaudio.compliance.kaldi as ta_kaldi

        features = []
        for waveform in audio:
            # `torchaudio.compliance.kaldi.fbank` expects a 2D (channel, time) tensor and
            # returns (time, num_mel_bins). This matches the numpy sibling which routes
            # through the same `ta_kaldi.fbank` via `_kaldi_fbank` after a from_numpy bridge.
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            features.append(
                ta_kaldi.fbank(
                    waveform,
                    num_mel_bins=128,
                    sample_frequency=self.sample_rate,
                    window_type="hanning",
                )
            )
        return features

    def _pad_features(self, features, padding, max_length, truncation, pad_to_multiple_of):
        # Always pad/truncate to max_length_frames regardless of caller's padding args
        return super()._pad_features(features, "max_length", self.max_length_frames, True, pad_to_multiple_of)

    def _postprocess_output(self, output, **kwargs):
        # Rename to audio_values (AST convention) and apply AudioSet normalization
        features = output.pop("audio_features")
        if self.do_normalize:
            features = (features - self.ast_mean) / (self.ast_std * 2)
        output["audio_values"] = features
        return output


AudioSpectrogramTransformerFeatureExtractor = make_legacy_audio_processor_alias(
    AudioSpectrogramTransformerAudioProcessor, "AudioSpectrogramTransformerFeatureExtractor"
)


__all__ = [
    "AudioSpectrogramTransformerAudioProcessor",
    "AudioSpectrogramTransformerFeatureExtractor",
]
