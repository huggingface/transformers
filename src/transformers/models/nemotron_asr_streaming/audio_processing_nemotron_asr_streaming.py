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

from ..parakeet.audio_processing_parakeet import ParakeetAudioProcessor


class NemotronAsrStreamingAudioProcessorMixin:
    """Parakeet's STFT + mel + preemphasis + log pipeline without its per-utterance mean/variance
    normalization: padded frames are zeroed. Output keys stay canonical (`audio_features` /
    `audio_features_mask`); consumers still reading `input_features` or `attention_mask` get them
    through the deprecated-key alias."""

    def _finalize_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        features = output.pop("audio_features")
        mask = output.pop("audio_features_mask", None)
        if mask is not None:
            features = features * self._astype(mask[..., None], self._dtype_name(features))
            output["audio_features_mask"] = mask
        output["audio_features"] = features
        return output


class NemotronAsrStreamingAudioProcessor(NemotronAsrStreamingAudioProcessorMixin, ParakeetAudioProcessor):
    def _dtype_name(self, x):
        return str(x.dtype).removeprefix("torch.")


__all__ = ["NemotronAsrStreamingAudioProcessor"]
