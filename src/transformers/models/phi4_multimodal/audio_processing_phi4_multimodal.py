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

from ...audio_processing_backends import TorchAudioBackend
from ...processing_utils import AudioKwargs


class Phi4MultimodalAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    audio_compression_rate (`int`, *optional*, defaults to 8):
        Factor by which the audio encoder compresses the mel frames.
    audio_downsample_rate (`int`, *optional*, defaults to 1):
        Factor by which the audio projector downsamples the encoder output.
    audio_feat_stride (`int`, *optional*, defaults to 1):
        Stride applied to the extracted audio features.
    """

    audio_compression_rate: int
    audio_downsample_rate: int
    audio_feat_stride: int


class Phi4MultimodalAudioProcessorMixin:
    sampling_rate = 16000
    extra_model_input_names = ["audio_embed_sizes"]
    # Kaldi-style fbank on int16-scaled samples: per-frame preemphasis, Hamming window, power
    # spectrum, Kaldi mel bank, floor at 1 then natural log. The legacy extractor batched this
    # recipe and zeroed one partial frame past each clip's valid count; that frame is outside
    # `audio_features_mask` and is not reproduced (ADR 0001, 2026-10-06 amendment).
    spectrogram_config = {
        "stft_config": {
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "window_fn": "hamming_window",
            "periodic": False,
            "center": False,
            "power": 2.0,
            "window_dtype": "float64",
        },
        "waveform_scale": 32768.0,
        "preemphasis": 0.97,
        "preemphasis_mode": "per_frame",
        "mel_scale_config": {
            "n_mels": 80,
            "f_min": 0,
            "f_max": 7690,
            "mel_scale": "kaldi",
            "triangularize_in_mel_space": True,
            "matmul_order": "features_first",
            "computation_dtype": "float64",
        },
        "mel_floor": 1.0,
        "log_mode": "log",
    }

    audio_compression_rate = 8
    audio_downsample_rate = 1
    audio_feat_stride = 1
    valid_kwargs = Phi4MultimodalAudioProcessorKwargs
    # `audio_embed_sizes` is computed from spectrogram frames, so extraction is not optional.
    frozen_options = ("do_extract_spectrogram",)

    def _compute_audio_embed_size(self, audio_frames, *, audio_compression_rate, audio_downsample_rate):
        integer = audio_frames // audio_compression_rate
        result = integer + (audio_frames % audio_compression_rate > 0)

        integer = result // audio_downsample_rate
        return integer + (result % audio_downsample_rate > 0)

    def _finalize_output(
        self,
        output,
        audio_ranges=None,
        feature_ranges=None,
        *,
        spectrogram_config,
        audio_feat_stride,
        audio_compression_rate,
        audio_downsample_rate,
        **kwargs,
    ):
        """Add `audio_embed_sizes`: how many encoder tokens each clip's valid frames become."""
        frame_counts = self._valid_frame_counts(self._lengths_from_ranges(audio_ranges), spectrogram_config)
        feature_lengths = self._as_backend_array(frame_counts, like=output["audio_features"]) * audio_feat_stride
        output["audio_embed_sizes"] = self._compute_audio_embed_size(
            feature_lengths,
            audio_compression_rate=audio_compression_rate,
            audio_downsample_rate=audio_downsample_rate,
        )
        return output


class Phi4MultimodalAudioProcessor(Phi4MultimodalAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["Phi4MultimodalAudioProcessor"]
