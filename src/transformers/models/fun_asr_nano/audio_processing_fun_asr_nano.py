# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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


class FunAsrNanoAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    num_frames_lfr (`int`, *optional*, defaults to 7):
        Number of consecutive mel frames stacked into each low-frame-rate frame.
    stride_lfr (`int`, *optional*, defaults to 6):
        Subsampling stride between low-frame-rate frames.
    """

    num_frames_lfr: int
    stride_lfr: int


def _ms_to_samples(target):
    """`torchaudio.compliance.kaldi.fbank` takes its frame geometry in **milliseconds**.

    The base mapping sends `frame_length` to `win_length` as if it were samples, which turned this
    checkpoint's `frame_length: 25` into a 25-sample window instead of 400. Convert against the
    config's own `sampling_rate`, as kaldi does internally.
    """

    def apply(value, config_dict):
        sampling_rate = config_dict.get("sampling_rate") or FunAsrNanoAudioProcessorMixin.sampling_rate
        stft_config = config_dict.setdefault("spectrogram_config", {}).setdefault("stft_config", {})
        stft_config[target] = int(value * sampling_rate / 1000)

    apply.legacy_target = f"spectrogram_config.stft_config.{target}"
    return apply


class FunAsrNanoAudioProcessorMixin:
    sampling_rate = 16000
    legacy_field_mapping = {
        "frame_length": _ms_to_samples("win_length"),
        "frame_shift": _ms_to_samples("hop_length"),
    }
    # `torchaudio.compliance.kaldi.fbank` geometry: 25 ms frames, 10 ms shift, 80 mel bins,
    # hamming window (FunASR's front-end uses hamming where most kaldi callers use povey).
    # Unlike XCodec2 the legacy extractor feeds unit-scale audio straight in, so there is no
    # int16 waveform scaling here.
    spectrogram_config = {
        "stft_config": {
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "window_fn": "hamming",
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
        "transpose_features": True,
    }

    num_frames_lfr = 7
    stride_lfr = 6
    valid_kwargs = FunAsrNanoAudioProcessorKwargs

    def _compute_batched_features(
        self, audio, *, audio_ranges, spectrogram_config, num_frames_lfr, stride_lfr, padding_value, **kwargs
    ):
        # The legacy extractor ran its fbank clip by clip, an artefact of
        # `torchaudio.compliance.kaldi.fbank` taking one utterance. Batched, the mel projection accumulates
        # over the batch's frame count and moves the features by float32 rounding: the exception recorded
        # in ADR 0001's 2026-09-30 amendment.
        features, frame_counts = super()._compute_batched_features(
            audio,
            audio_ranges=audio_ranges,
            spectrogram_config=spectrogram_config,
            padding_value=padding_value,
            **kwargs,
        )
        return self._apply_lfr(
            features, frame_counts, num_frames_lfr=num_frames_lfr, stride_lfr=stride_lfr, padding_value=padding_value
        )

    def _apply_lfr(self, features, frame_counts, *, num_frames_lfr, stride_lfr, padding_value):
        """Low frame rate: stack `num_frames_lfr` mel frames, hop by `stride_lfr`, per clip.

        A window reaching past a clip's edge repeats that clip's first or last real frame rather than
        zero-padding, so it never mixes audio with silence or with the batch padding. LFR frames past a
        clip's own count are `padding_value`, as the legacy extractor's feature-level padding left them.
        """
        batch_size, num_frames, num_mels = features.shape
        num_output_frames = -(-num_frames // stride_lfr)
        counts = self._as_backend_array(frame_counts, like=features)
        windows = self._arange(num_output_frames, like=features)[None, :, None] * stride_lfr
        windows = windows + self._arange(num_frames_lfr, like=features)[None, None, :] - (num_frames_lfr - 1) // 2
        # clamp each source frame to the clip's own `[0, count - 1]`
        sources = -self._maximum(-self._clamp_min(windows, 0), 1 - counts[:, None, None])
        stacked = features[self._arange(batch_size, like=features)[:, None, None], sources]
        stacked = stacked.reshape(batch_size, num_output_frames, num_frames_lfr * num_mels)
        lfr_counts = -(-frame_counts // stride_lfr)
        return self._mask_padded_frames(stacked, lfr_counts, padding_value=padding_value), lfr_counts

    def _padded_frame_count(self, padded_length, spectrogram_config, *, stride_lfr, **kwargs) -> int:
        return -(-super()._padded_frame_count(padded_length, spectrogram_config, **kwargs) // stride_lfr)

    def _finalize_output(self, output, **kwargs):
        if "audio_features_mask" in output:
            output["audio_features_mask"] = self._astype(output["audio_features_mask"], "int64")
        return output


class FunAsrNanoAudioProcessor(FunAsrNanoAudioProcessorMixin, TorchAudioBackend):
    pass


__all__ = ["FunAsrNanoAudioProcessor"]
