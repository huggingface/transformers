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

import numpy as np

from ...audio_processing_backends import NumpyAudioBackend


def _gemma4_unified_feature_size_to_samples_per_token(value, config_dict):
    # Legacy configs carry the frame size both as `feature_size` and
    # `audio_samples_per_token`; keep the modern key if it is already present.
    config_dict.setdefault("audio_samples_per_token", value)


class Gemma4UnifiedAudioProcessorNumpy(NumpyAudioBackend):
    """NumPy sibling of [`Gemma4UnifiedAudioProcessor`]. Bit-exact to the torch sibling
    (ADR 0001): the pipeline is pure chunking with no floating-point arithmetic. See
    [`Gemma4UnifiedAudioProcessor`] for the full pipeline description."""

    sample_rate = 16000
    force_mono = True
    padding = "longest"
    padding_value = 0.0

    # Encoder-free raw-waveform chunking: there is no STFT/mel stage. Features are
    # extracted per waveform (`extract_spectrogram` is fully overridden, as sanctioned
    # for non-spectrogram models) and padded at the token level, matching the legacy
    # extractor's feature-level `pad()`.
    do_extract_spectrogram = True
    do_batch_spectrogram = False

    audio_samples_per_token = 640

    legacy_field_mapping = {
        "feature_size": _gemma4_unified_feature_size_to_samples_per_token,
    }

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
            waveform = np.pad(waveform, (0, pad_len))
        num_tokens = waveform.shape[-1] // self.audio_samples_per_token
        return waveform.reshape(num_tokens, self.audio_samples_per_token).astype(np.float32)


__all__ = ["Gemma4UnifiedAudioProcessorNumpy"]
