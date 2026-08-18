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
from .audio_processing_numpy_cohere_asr import CohereAsrAudioProcessorMixin


class CohereAsrAudioProcessor(CohereAsrAudioProcessorMixin, TorchAudioBackend):
    """Torch sibling of [`CohereAsrAudioProcessorNumpy`]: energy-based long-audio chunking,
    deterministic dither, waveform preemphasis, ``log(mel @ |X|^2 + 2^-24)`` features with
    per-utterance mean/variance normalization.

    Every numeric quirk of the legacy FE is declared in the shared ``spectrogram_config``
    (librosa filter rounding, plain `mel_filters @ magnitudes` matmul, `view_as_real`
    magnitude form, (batch, frames, mels) layout); only the deterministic dither below needs
    a torch-specific implementation, since its RNG cannot be reproduced with numpy's.
    """

    def _apply_dither(self, audio, audio_ranges=None):
        """Deterministic per-utterance dither: each row is seeded by its valid sample count,
        so dither is invariant to batch composition (matches the legacy FE)."""
        if self.dither <= 0 or audio_ranges is None:
            return audio
        audio = audio.clone()
        generator = torch.Generator(device=audio.device)
        for i, (start, end) in enumerate(audio_ranges):
            valid_samples = min(end - start, audio.shape[1])
            if valid_samples <= 0:
                continue
            generator.manual_seed(valid_samples)
            noise = torch.randn(valid_samples, dtype=audio.dtype, device=audio.device, generator=generator)
            audio[i, :valid_samples] = audio[i, :valid_samples] + self.dither * noise
        return audio


__all__ = ["CohereAsrAudioProcessor"]
