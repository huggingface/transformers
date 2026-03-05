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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...feature_extraction_utils import BatchFeature


class AudioSpectrogramTransformerAudioProcessor(NumpyAudioBackend):
    sample_rate = 16000
    force_mono = True
    max_length_frames = 1024
    transpose_features = True

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

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # Extract spectrogram via generic config-based API
        features = self.extract_spectrogram(audio, spectrogram_config=self.spectrogram_config)

        # Generic extract_spectrogram returns (n_mels, frames); transpose to (frames, n_mels)
        features = [f.T for f in features]

        # Pad or truncate to max_length_frames
        padded = []
        for fbank in features:
            n_frames = fbank.shape[0]
            if n_frames < self.max_length_frames:
                pad_amount = self.max_length_frames - n_frames
                fbank = np.pad(fbank, ((0, pad_amount), (0, 0)), mode="constant", constant_values=0.0)
            elif n_frames > self.max_length_frames:
                fbank = fbank[: self.max_length_frames, :]
            padded.append(fbank)

        # Normalize with AudioSet stats
        normalized = [(f - self.ast_mean) / (self.ast_std * 2) for f in padded]

        # Stack into batch
        output_key = self.model_input_names[0]
        stacked = np.stack(normalized, axis=0) if return_tensors else normalized
        return BatchFeature(data={output_key: stacked}, tensor_type=return_tensors)


__all__ = ["AudioSpectrogramTransformerAudioProcessor"]
