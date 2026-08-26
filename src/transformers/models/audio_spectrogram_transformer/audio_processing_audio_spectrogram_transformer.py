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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class AudioSpectrogramTransformerAudioProcessorMixin:
    ast_mean = -4.2677393
    ast_std = 4.5689974
    do_batch_spectrogram = False
    do_normalize = True
    force_mono = True
    # The legacy FE saved `feature_size=1` (a raw-audio default) and kept the real mel count in
    legacy_field_mapping = {"feature_size": None}
    max_length_frames = 1024
    model_input_names = ["audio_values"]
    return_padding_mask = False
    sampling_rate = 16000
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn="hann_window",
            power=2.0,
            center=False,
            periodic=False,
            left_align_fft=True,
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
        transpose_features=True,  # kaldi's (time, num_mel_bins) orientation
    )

    def _pad_features(self, features, padding, max_length, truncation, pad_to_multiple_of):
        return super()._pad_features(features, "max_length", self.max_length_frames, True, pad_to_multiple_of)

    def _postprocess_output(self, output, **kwargs):
        features = output.pop("audio_features")
        if self.do_normalize:
            features = (features - self.ast_mean) / (self.ast_std * 2)
        output["audio_values"] = features
        return output


class AudioSpectrogramTransformerAudioProcessor(AudioSpectrogramTransformerAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["AudioSpectrogramTransformerAudioProcessor"]
