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


class NemotronAsrStreamingAudioProcessor(ParakeetAudioProcessor):
    """Audio processor for NemotronAsrStreaming.

    The STFT + mel + preemphasis + log pipeline is identical to Parakeet's
    (`n_fft=512`, `hop_length=160`, `win_length=400`, `power=2.0`,
    `pad_mode="constant"`, `periodic=False`, slaney mel, `preemphasis=0.97`,
    `log_mode="log"`, `mel_floor=2**-24`), including the librosa-bit-exact
    `_standard_mel_banks`. Unlike Parakeet, NemotronAsrStreaming never applies
    per-utterance mean/variance normalization — it only zeroes the padded frames
    and emits the legacy output keys the model consumes (`input_features` /
    `attention_mask`).
    """

    model_input_names = ["input_features", "attention_mask"]

    def _postprocess_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        # No per-utterance mean/var normalization (unlike Parakeet's CMVN). Zero the padded
        # frames via the mask, then rename to the legacy keys the model's forward expects.
        features = output.pop("audio_features")
        mask = output.pop("audio_features_mask", None)
        if mask is not None:
            features = features * mask.unsqueeze(-1).to(features.dtype)
            output["attention_mask"] = mask
        output["input_features"] = features
        return output


__all__ = ["NemotronAsrStreamingAudioProcessor"]
