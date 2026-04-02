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

import torch

from spectrograms import numpy_mel_spectrogram as _np_spec

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...feature_extraction_utils import BatchFeature


class Phi4MultimodalAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    audio_compression_rate = 8
    audio_downsample_rate = 1
    audio_feat_stride = 1
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn="hamming_window",
            periodic=False,
            center=False,
            power=2.0,
            window_dtype="float64",
        ),
        preemphasis=0.97,
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=0,
            f_max=7690,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
            matmul_order="features_first",
        ),
        mel_floor=1.0,
        log_mode="log",
    )

    def _mel_filter_bank(self, spectrogram_config):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        mel_filters_np = _np_spec.mel_filter_bank(
            num_frequency_bins=1 + stft_cfg.n_fft // 2,
            num_mel_filters=mel_cfg.n_mels,
            min_frequency=mel_cfg.f_min,
            max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
            sampling_rate=self.sample_rate,
            norm=mel_cfg.norm,
            mel_scale=mel_cfg.mel_scale,
            triangularize_in_mel_space=mel_cfg.triangularize_in_mel_space,
        )
        return torch.from_numpy(mel_filters_np).to(torch.float32)

    def _apply_frame_processing(self, frames, *, spectrogram_config, audio_ranges=None, **kwargs):
        # Mask frames that overlap the boundary between real audio and padding
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length or stft_cfg.n_fft
        hop_length = stft_cfg.hop_length or win_length // 2
        batch_size = frames.shape[0]

        if audio_ranges is not None and batch_size > 1:
            audio_lengths_t = torch.tensor([end - start for start, end in audio_ranges])
            to_mask_idxs = torch.arange(batch_size)[audio_lengths_t != audio_lengths_t.max()]
            if to_mask_idxs.numel() > 0:
                frames = frames.clone()
                down = (audio_lengths_t[to_mask_idxs] - win_length) // hop_length + 1
                up = audio_lengths_t[to_mask_idxs] // hop_length - 1
                offset = down.min()
                max_idx = up.max()

                mask_range = torch.arange(max_idx - offset).expand(to_mask_idxs.shape[0], -1)
                mask = ((down - offset).unsqueeze(1) <= mask_range) & (mask_range < (up - offset).unsqueeze(1))
                mask = mask.unsqueeze(-1).expand(-1, -1, win_length)

                masked_frames = frames[to_mask_idxs, offset:max_idx].masked_fill_(mask, 0)
                frames[to_mask_idxs, offset:max_idx] = masked_frames

        frames_prev = torch.roll(frames, 1, dims=-1)
        frames_prev[..., 0] = frames_prev[..., 1]
        return (frames - spectrogram_config.preemphasis * frames_prev) * 32768

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg):
        frames = frames * window
        if frame_length < n_fft:
            frames = torch.nn.functional.pad(frames, (0, n_fft - frame_length))
        # Cast to complex64 before abs() to match the FE's precision path
        spec = torch.fft.rfft(frames, n=n_fft).to(torch.complex64)
        if stft_cfg.normalized:
            spec = spec / window.pow(2.0).sum().sqrt()
        return spec.transpose(-2, -1)

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        win_length = spectrogram_config.stft_config.win_length or spectrogram_config.stft_config.n_fft
        hop_length = spectrogram_config.stft_config.hop_length or win_length // 2
        return (audio_lengths - win_length) // hop_length + 1

    def _compute_audio_embed_size(self, audio_frames):
        integer = audio_frames // self.audio_compression_rate
        remainder = audio_frames % self.audio_compression_rate
        result = integer + (remainder > 0).to(integer.dtype)

        integer = result // self.audio_downsample_rate
        remainder = result % self.audio_downsample_rate
        result = integer + (remainder > 0).to(integer.dtype)

        return result

    def _preprocess(
        self,
        audio,
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        spectrogram_config=None,
        do_extract_spectrogram=None,
        **kwargs,
    ) -> BatchFeature:
        output = super()._preprocess(
            audio,
            padding,
            max_length,
            truncation,
            pad_to_multiple_of,
            return_tensors,
            spectrogram_config=spectrogram_config,
            do_extract_spectrogram=do_extract_spectrogram,
            **kwargs,
        )

        feature_lengths = output["audio_features_mask"].sum(dim=-1)
        feature_lengths = feature_lengths * self.audio_feat_stride
        output["audio_embed_sizes"] = self._compute_audio_embed_size(feature_lengths)

        return output


__all__ = ["Phi4MultimodalAudioProcessor"]
