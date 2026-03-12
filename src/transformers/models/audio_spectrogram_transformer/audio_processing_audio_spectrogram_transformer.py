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

from ...audio_processing_backends import NumpyAudioBackend, TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...feature_extraction_utils import BatchFeature
from ...utils import is_torch_available


class AudioSpectrogramTransformerAudioProcessor(NumpyAudioBackend if not is_torch_available() else TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    return_attention_mask = False

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

    def extract_spectrogram(self, audio, *, spectrogram_config=None, **kwargs):
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = [audio[i] for i in range(audio.shape[0])]
        elif hasattr(audio, 'dim') and audio.dim() > 1:
            audio = [audio[i] for i in range(audio.shape[0])]
        elif not isinstance(audio, list):
            audio = [audio]

        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config
        features = super().extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
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
        if self.do_normalize:
            padded = [(f - self.ast_mean) / (self.ast_std * 2) for f in padded]

        return np.stack(padded, axis=0)

    def _preprocess(self, audio, padding, max_length, truncation, pad_to_multiple_of, return_tensors, **kwargs):
        # AST does all processing at the feature level in extract_spectrogram
        features = self.extract_spectrogram(audio, spectrogram_config=self.spectrogram_config)
        return BatchFeature({"audio_values": features}, tensor_type=return_tensors)


__all__ = ["AudioSpectrogramTransformerAudioProcessor"]
