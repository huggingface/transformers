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

import torch

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


class FunAsrNanoAudioProcessorMixin:
    sampling_rate = 16000
    # Keep each clip unpadded through the fbank so the mel projection remains bit-exact
    # with the legacy extractor. The base then pads the LFR features and builds their mask.
    do_batch_spectrogram = False

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

    def _finalize_features(self, features, feature_lengths):
        return [self._apply_lfr(feature) for feature in features]

    def _finalize_output(self, output, **kwargs):
        if "audio_features_mask" in output:
            output["audio_features_mask"] = self._astype(output["audio_features_mask"], "int64")
        return output


class FunAsrNanoAudioProcessor(FunAsrNanoAudioProcessorMixin, TorchAudioBackend):
    def _apply_lfr(self, features):
        """Low frame rate: stack `num_frames_lfr` mel frames, hop by `stride_lfr`.

        Edges are handled by repeating the first and last frame rather than zero-padding, so a
        stacked window never mixes real audio with silence.
        """
        num_input_frames = features.shape[0]
        left_pad = (self.num_frames_lfr - 1) // 2
        right_pad = self.num_frames_lfr - 1 - left_pad
        padded = torch.cat([features[0:1].expand(left_pad, -1), features, features[-1:].expand(right_pad, -1)], dim=0)
        num_output_frames = -(-num_input_frames // self.stride_lfr)
        required = (num_output_frames - 1) * self.stride_lfr + self.num_frames_lfr
        if required > padded.shape[0]:
            padded = torch.cat([padded, padded[-1:].expand(required - padded.shape[0], -1)], dim=0)
        windows = padded.unfold(0, self.num_frames_lfr, self.stride_lfr).transpose(1, 2)
        return windows.reshape(num_output_frames, -1)


__all__ = ["FunAsrNanoAudioProcessor"]
