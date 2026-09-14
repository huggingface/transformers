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
from ...processing_utils import AudioKwargs


class VibevoiceAcousticTokenizerAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    target_dB_FS (`int`, *optional*, defaults to -25):
        Loudness, in dBFS, the waveform is normalized to.
    eps (`float`, *optional*, defaults to 1e-06):
        Small constant guarding the loudness-normalization division.
    normalize_audio (`bool`, *optional*, defaults to `True`):
        Whether to loudness-normalize the waveform at all. The legacy extractor gates the whole
        normalization on this; without it a checkpoint that disables it was normalized anyway.
    """

    target_dB_FS: int
    eps: float
    normalize_audio: bool


class VibevoiceAcousticTokenizerAudioProcessorMixin:
    sampling_rate = 24000
    add_channel_dim = True

    target_dB_FS = -25
    eps = 1e-6
    normalize_audio = True
    # Not in the legacy extractor's signature (feature_size, sampling_rate, padding_value,
    # normalize_audio, target_dB_FS, eps, pad_to_multiple_of): `db_normalize` is superseded by
    # `normalize_audio`, and `speech_tok_compress_ratio` is a modeling ratio, not a feature one.
    legacy_field_mapping = {"db_normalize": None, "speech_tok_compress_ratio": None}
    valid_kwargs = VibevoiceAcousticTokenizerAudioProcessorKwargs

    def _downmix_to_mono(self, audio_el, *, normalize_audio, target_dB_FS, eps, **kwargs):
        audio_el = super()._downmix_to_mono(audio_el, **kwargs)
        if not normalize_audio:
            return audio_el
        rms = (audio_el**2).mean() ** 0.5
        audio_el = audio_el * (10 ** (target_dB_FS / 20) / (rms + eps))
        max_val = abs(audio_el).max()
        if max_val > 1.0:
            audio_el = audio_el / (max_val + eps)
        return audio_el


class VibevoiceAcousticTokenizerAudioProcessor(VibevoiceAcousticTokenizerAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["VibevoiceAcousticTokenizerAudioProcessor"]
