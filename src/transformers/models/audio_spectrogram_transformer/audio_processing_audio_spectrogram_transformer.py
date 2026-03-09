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
    do_extract_spectrogram = True

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
        features = super().extract_spectrogram(audio, **kwargs)

        # (n_mels, frames) -> (frames, n_mels)
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
        return [(f - self.ast_mean) / (self.ast_std * 2) for f in padded]

    def _preprocess(self, audio, **kwargs):
        output = super()._preprocess(audio, **kwargs)
        # TODO: it is wrongly named input_values in the original feature extractor
        return BatchFeature({"audio_values": output["audio_features"]})



__all__ = ["AudioSpectrogramTransformerAudioProcessor"]
