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
from ...audio_utils import _array_namespace
from ...processing_utils import AudioKwargs


class Xcodec2AudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    hop_length (`int`, *optional*, defaults to 320):
        Codec frame size, in samples. Distinct from the STFT hop in `spectrogram_config`.
    stride (`int`, *optional*, defaults to 2):
        Number of mel frames stacked into each output frame.
    feature_padding_value (`float`, *optional*, defaults to 1.0):
        Value used to pad the extracted features.
    """

    hop_length: int
    stride: int
    feature_padding_value: float | int


class Xcodec2AudioProcessorMixin:
    add_channel_dim = True
    do_extract_spectrogram = False
    # Two consumers: the acoustic encoder takes the zero-padded waveform (`audio_values`), the
    # semantic encoder a kaldi fbank of it, computed per clip in `_finalize_output` and padded with
    # 1.0 (the legacy FE's `padding_value`). Legacy hub configs also carry the fbank geometry as
    # flat keys, which are fixed here.
    legacy_field_mapping = {
        "feature_size": None,
        "frame_length": None,
        "frame_shift": None,
        "num_mel_bins": None,
        "hop_length": None,
        "padding_value": "feature_padding_value",
    }
    model_input_names = ["audio_features", "audio_features_mask", "audio_values", "audio_values_mask"]
    pad_to_multiple_of = 320
    sampling_rate = 16000
    spectrogram_config = {
        "stft_config": {
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "window_fn": "povey",
            "power": 2.0,
            "center": False,
            "periodic": False,
            "left_align_fft": True,
        },
        "mel_scale_config": {
            "n_mels": 80,
            "f_min": 20.0,
            "f_max": 8000.0,
            "mel_scale": "kaldi",
            "triangularize_in_mel_space": True,
        },
        "log_mode": "log",
        "preemphasis": 0.97,
        "remove_dc_offset": True,
        "mel_floor": 1.192092955078125e-07,
        "waveform_scale": 32768.0,
        "transpose_features": True,  # kaldi's (time, n_mels) orientation
    }

    hop_length = 320
    stride = 2
    feature_padding_value = 1.0
    valid_kwargs = Xcodec2AudioProcessorKwargs

    def _downmix_to_mono(self, audio_el):
        # the legacy FE appends one zero sample to every waveform before padding
        return self._pad_axis(super()._downmix_to_mono(audio_el), 0, 1, axis=-1)

    def _pad_semantic_waveform(self, waveform):
        # half a codec hop of zeros on both sides, so the fbank frames line up with the codec frames
        half_hop = self.hop_length // 2
        return self._pad_axis(waveform, half_hop, half_hop, axis=-1)

    def _pad_feature_single(self, feature, max_length):
        return self._pad_axis(feature, 0, max_length - feature.shape[0], axis=0, value=self.feature_padding_value)

    def _finalize_output(self, output, audio_ranges=None, **kwargs):
        audio_values = output["audio_values"]
        padded_length = audio_values.shape[-1]

        features = []
        for i, (start, end) in enumerate(audio_ranges):
            # the fbank sees each clip rounded up to whole hops, not the batch-padded length
            valid_length = min((end - start + self.hop_length - 1) // self.hop_length * self.hop_length, padded_length)
            waveform = self._pad_semantic_waveform(audio_values[i, 0, :valid_length])
            features.append(self._standardize_frames(self.compute_features([waveform])[0]))

        features, frame_ranges = self._pad_features(features, "longest", None, False, self.stride)
        batch = self._stack(features)
        mask = self._get_mask(frame_ranges, batch.shape[1])

        batch_size, num_frames, num_mel_bins = batch.shape
        output["audio_features"] = batch.reshape(batch_size, num_frames // self.stride, num_mel_bins * self.stride)
        stride_groups = mask.reshape(batch_size, num_frames // self.stride, self.stride)
        output["audio_features_mask"] = _array_namespace(mask).amin(stride_groups, -1)
        return output


class Xcodec2AudioProcessor(Xcodec2AudioProcessorMixin, TorchAudioBackend):
    def _standardize_frames(self, features):
        return (features - features.mean(0)) / torch.sqrt(features.var(0, unbiased=True) + 1e-7)


__all__ = ["Xcodec2AudioProcessor"]
