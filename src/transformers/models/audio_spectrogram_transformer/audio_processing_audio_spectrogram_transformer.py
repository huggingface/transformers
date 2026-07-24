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

from ...audio_processing_backends import TorchAudioBackend
from .audio_processing_numpy_audio_spectrogram_transformer import AudioSpectrogramTransformerAudioProcessorNumpy


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


    spectrogram_config = AudioSpectrogramTransformerAudioProcessorNumpy.spectrogram_config

    def extract_spectrogram(self, audio, **kwargs):
        # Native kaldi-exact pipeline (bit-equal to `torchaudio.compliance.kaldi.fbank`),
        # transposed to kaldi's (time, num_mel_bins) orientation expected downstream.
        features = super().extract_spectrogram(audio, **kwargs)
        return [f.transpose(-2, -1) for f in features]

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


__all__ = ["AudioSpectrogramTransformerAudioProcessor"]
