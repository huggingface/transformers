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

from typing import Annotated

from ...audio_processing_backends import TorchAudioBackend
from ...audio_processing_base import BatchFeature
from ...processing_utils import AudioKwargs
from ...utils.type_validators import positive_int


def _gemma4_unified_feature_size_to_samples_per_token(value, config_dict):
    # Legacy configs carry the frame size both as `feature_size` and
    config_dict.setdefault("audio_samples_per_token", value)


_gemma4_unified_feature_size_to_samples_per_token.legacy_target = "audio_samples_per_token"


class Gemma4UnifiedAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    audio_samples_per_token (`int`, *optional*, defaults to 640):
        Number of waveform samples represented by a single audio token.
    """

    audio_samples_per_token: Annotated[int, positive_int]


class Gemma4UnifiedAudioProcessorMixin:
    model_input_names = ["audio_features", "audio_features_mask"]
    legacy_field_mapping = {
        "feature_size": _gemma4_unified_feature_size_to_samples_per_token,
    }
    padding = "longest"
    sampling_rate = 16000

    audio_samples_per_token = 640
    valid_kwargs = Gemma4UnifiedAudioProcessorKwargs

    def _validate_preprocess_kwargs(self, *, padding_side, **kwargs):
        super()._validate_preprocess_kwargs(**kwargs)
        if padding_side != "right":
            raise ValueError("Gemma4 Unified pads token frames on the right.")

    def _preprocess(
        self,
        audio,
        *,
        audio_samples_per_token,
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        padding_value,
        return_padding_mask,
        return_tensors,
        **kwargs,
    ):
        """Frame raw waveforms into tokens, then pad in tokens rather than waveform samples."""
        features = [
            self._chunk_waveform(waveform, audio_samples_per_token=audio_samples_per_token) for waveform in audio
        ]
        features, ranges = self._pad_features(
            features, padding, max_length, truncation, pad_to_multiple_of, padding_value=padding_value
        )
        output = {"audio_features": self._stack_features(features)}
        if return_padding_mask:
            output["audio_features_mask"] = self._get_mask(ranges, features[0].shape[0])
        return BatchFeature(output, tensor_type=return_tensors)

    def _chunk_waveform(self, waveform, *, audio_samples_per_token):
        pad_len = (-waveform.shape[-1]) % audio_samples_per_token
        if pad_len:
            waveform = self._pad_axis(waveform, 0, pad_len, axis=-1)
        return self._astype(waveform.reshape(-1, audio_samples_per_token), "float32")


class Gemma4UnifiedAudioProcessor(Gemma4UnifiedAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["Gemma4UnifiedAudioProcessor"]
