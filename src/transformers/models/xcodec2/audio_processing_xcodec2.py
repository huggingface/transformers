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

import torch

from ...audio_processing_backends import TorchAudioBackend
from .audio_processing_numpy_xcodec2 import Xcodec2AudioProcessorNumpy


class Xcodec2AudioProcessor(TorchAudioBackend):
    """Dual-output codec processor: raw padded audio for the acoustic encoder
    (`audio_values`) plus per-utterance kaldi fbank features for the semantic encoder
    (`audio_features`), computed in `_postprocess_output` from the padded audio batch.
    The fbank path is bit-exact against `torchaudio.compliance.kaldi.fbank` as used by the
    legacy `Xcodec2FeatureExtractor`."""

    sample_rate = 16000
    force_mono = True
    add_channel_dim = True
    padding_value = 0.0
    # One acoustic-encoder frame = `hop_length` audio samples (product of downsampling ratios)
    hop_length = 320
    pad_to_multiple_of = 320
    # Semantic features: pairs of consecutive fbank frames are concatenated (stride 2)
    stride = 2
    # Mel frames are padded with 1.0 (the legacy FE's `padding_value`), unlike the raw audio
    feature_padding_value = 1.0
    # Semantic features are derived from the padded audio in `_postprocess_output`, not via
    # the base spectrogram path
    do_extract_spectrogram = False

    spectrogram_config = Xcodec2AudioProcessorNumpy.spectrogram_config
    legacy_field_mapping = Xcodec2AudioProcessorNumpy.legacy_field_mapping

    def _process_audio(self, audio_el):
        # The legacy FE appends one zero sample to every waveform before padding
        audio_el = super()._process_audio(audio_el)
        return torch.nn.functional.pad(audio_el, (0, 1))

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        audio_values = output["audio_values"]  # (batch, 1, padded_length)
        padded_length = audio_values.shape[-1]
        half_hop = self.hop_length // 2

        # Per-utterance fbank on the valid (hop-aligned) slice of the padded audio,
        # normalized with per-utterance mean/variance before mel-frame padding
        features = []
        for i, (start, end) in enumerate(audio_ranges):
            orig_length = end - start
            valid_length = min((orig_length + self.hop_length - 1) // self.hop_length * self.hop_length, padded_length)
            waveform = torch.nn.functional.pad(audio_values[i, 0, :valid_length], (half_hop, half_hop))
            f = self.extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)[0].transpose(-2, -1)
            f = (f - f.mean(0)) / torch.sqrt(f.var(0, unbiased=True) + 1e-7)
            features.append(f)

        # Pad mel frames to the longest utterance (aligned to `stride`) with `feature_padding_value`
        frame_lengths = [f.shape[0] for f in features]
        max_frames = max(frame_lengths)
        if max_frames % self.stride:
            max_frames += self.stride - max_frames % self.stride
        batch = torch.stack(
            [
                torch.nn.functional.pad(f, (0, 0, 0, max_frames - f.shape[0]), value=self.feature_padding_value)
                for f in features
            ]
        )
        mask = self._get_mask([(0, length) for length in frame_lengths], max_frames)

        # Stride concatenation: (batch, frames, n_mels) -> (batch, frames // stride, n_mels * stride)
        batch_size, num_frames, num_mel_bins = batch.shape
        output["audio_features"] = batch.reshape(batch_size, num_frames // self.stride, num_mel_bins * self.stride)
        output["audio_features_mask"] = (
            mask.reshape(batch_size, num_frames // self.stride, self.stride).min(dim=-1).values
        )
        return output


__all__ = ["Xcodec2AudioProcessor"]
