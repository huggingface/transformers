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
from .audio_processing_numpy_gemma4_unified import Gemma4UnifiedAudioProcessorNumpy


class Gemma4UnifiedAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`Gemma4UnifiedAudioProcessorNumpy`]. Encoder-free audio processor
    that chunks raw 16 kHz waveforms into fixed-length frames:

    1. Each waveform is zero-padded to a multiple of ``audio_samples_per_token`` samples
       (the padded tail stays inside the last frame — it never creates an extra one).
    2. The waveform is reshaped to ``(num_tokens, audio_samples_per_token)``; each frame
       of raw samples becomes one audio soft token (640 samples = 40 ms at 16 kHz).
    3. Frames are padded across the batch at the token level; the mask marks every token
       of a waveform valid, including the final partially-padded frame.

    Unlike the standard Gemma4 audio processor there is no mel spectrogram stage."""

    sample_rate = 16000
    force_mono = True
    padding = "longest"
    padding_value = 0.0

    # Encoder-free raw-waveform chunking: no STFT/mel stage; see the numpy sibling.
    do_extract_spectrogram = True
    do_batch_spectrogram = False

    audio_samples_per_token = Gemma4UnifiedAudioProcessorNumpy.audio_samples_per_token
    legacy_field_mapping = Gemma4UnifiedAudioProcessorNumpy.legacy_field_mapping

    def __init__(self, audio_samples_per_token: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if audio_samples_per_token is not None:
            self.audio_samples_per_token = audio_samples_per_token

    def extract_spectrogram(self, audio, **kwargs):
        # Not a spectrogram: chunk each raw waveform into (num_tokens, samples_per_token)
        # frames. Each frame becomes one audio soft token.
        return [self._chunk_waveform(waveform) for waveform in audio]

    def _chunk_waveform(self, waveform):
        """Chunk a 1-D waveform into fixed-length frames of `audio_samples_per_token`
        samples, zero-padding the tail so the last (partial) frame is kept."""
        pad_len = (-waveform.shape[-1]) % self.audio_samples_per_token
        if pad_len:
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        num_tokens = waveform.shape[-1] // self.audio_samples_per_token
        return waveform.reshape(num_tokens, self.audio_samples_per_token).to(torch.float32)


__all__ = ["Gemma4UnifiedAudioProcessor"]
