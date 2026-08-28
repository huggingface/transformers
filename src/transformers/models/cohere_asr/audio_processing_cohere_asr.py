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

import numpy as np
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, _array_namespace
from ...processing_utils import AudioKwargs


class CohereAsrAudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    max_audio_clip_s (`float`, *optional*, defaults to 35.0):
        Maximum length, in seconds, of a single audio chunk. Longer audio is split into chunks.
    overlap_chunk_second (`float`, *optional*, defaults to 5.0):
        Width, in seconds, of the window at the end of each chunk searched for a split point, so a
        boundary lands in silence rather than mid-word.
    min_energy_window_samples (`int`, *optional*, defaults to 1600):
        Size, in samples, of the non-overlapping windows scanned when locating the quietest split point.
    """

    max_audio_clip_s: float
    overlap_chunk_second: float
    min_energy_window_samples: int


class CohereAsrAudioProcessorMixin:
    sampling_rate = 16000
    force_mono = True
    padding = "longest"
    dither: float = 1e-5

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            hop_length=160,
            win_length=400,
            window_fn="hann_window",
            power=2.0,
            pad_mode="constant",
            periodic=False,
            magnitude_mode="sqrt_sum_squares",
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=0.0,
            norm="slaney",
            mel_scale="slaney",
            matmul_order="filters_first_matmul",
            bank_rounding="librosa",
        ),
        preemphasis=0.97,
        preemphasis_mode="waveform",
        log_mode="log",
        mel_floor=0.0,
        pre_log_offset=2**-24,
        transpose_features=True,
    )

    skip_tensor_conversion = ["audio_chunk_index"]
    extra_model_input_names = ["audio_chunk_index"]

    max_audio_clip_s: float = 35.0
    overlap_chunk_second: float = 5.0
    min_energy_window_samples: int = 1600
    valid_kwargs = CohereAsrAudioProcessorKwargs

    def _apply_dither(self, audio, audio_ranges=None):
        if self.dither <= 0 or audio_ranges is None:
            return audio
        noise = _array_namespace(audio).zeros_like(audio)
        for i, (start, end) in enumerate(audio_ranges):
            valid_samples = min(end - start, audio.shape[1])
            if valid_samples > 0:
                noise[i, :valid_samples] = self._seeded_noise(valid_samples, valid_samples, audio)
        return audio + self.dither * noise

    def _seeded_noise(self, length, seed, like):
        raise NotImplementedError

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output
        audio_lengths = np.asarray([end - start for start, end in audio_ranges])
        feature_lengths = self._get_valid_feature_lengths(audio_lengths, self.spectrogram_config)

        features = output["audio_features"]
        xp = _array_namespace(output["audio_features"])
        lengths = self._astype(self._as_backend_array(np.asarray(feature_lengths)), "float32")
        mask = (xp.arange(features.shape[1])[None, :] < lengths[:, None])[..., None]
        masked = features * mask
        mean = (masked.sum(axis=1) / lengths[:, None])[:, None, :]
        variance = (((masked - mean) ** 2) * mask).sum(axis=1) / (lengths - 1)[:, None]
        std = xp.sqrt(variance)[:, None, :]
        audio_features = (features - mean) / (std + 1e-5) * mask

        output["audio_features"] = audio_features
        return output

    def _preprocess_audio_like_inputs(self, audio, *args, sampling_rate=None, **kwargs):
        prepared = self._prepare_audio_like_inputs(audio=audio, sampling_rate=sampling_rate)
        chunked, audio_chunk_index = self._split_audio_chunks(prepared)
        result = self._preprocess(chunked, *args, **kwargs)
        result["audio_chunk_index"] = audio_chunk_index
        return result

    def _split_audio_chunks(self, prepared_audio):
        """Split audio longer than ``max_audio_clip_s - overlap_chunk_second`` at the
        quietest window. Returns (chunks, [(sample_idx, chunk_idx or None)])."""
        threshold_s = max(0.0, self.max_audio_clip_s - self.overlap_chunk_second)
        chunked, chunk_index = [], []
        for sample_idx, waveform in enumerate(prepared_audio):
            if waveform.shape[0] / self.sampling_rate <= threshold_s:
                chunked.append(waveform)
                chunk_index.append((sample_idx, None))
                continue
            for chunk_idx, chunk in enumerate(self._split_single_audio(waveform)):
                chunked.append(chunk)
                chunk_index.append((sample_idx, chunk_idx))
        return chunked, chunk_index

    def _split_single_audio(self, waveform):
        """Cut into ``max_audio_clip_s`` chunks, each ending at the quietest window within the
        final ``overlap_chunk_second`` so the boundary lands in silence rather than mid-word."""
        chunk_size = max(1, int(round(self.max_audio_clip_s * self.sampling_rate)))
        context = max(1, int(round(self.overlap_chunk_second * self.sampling_rate)))
        total = waveform.shape[0]
        chunks, start = [], 0
        while start + chunk_size < total:
            split = self._find_split_point_energy(
                waveform, max(start, start + chunk_size - context), start + chunk_size
            )
            split = max(start + 1, min(split, total))
            chunks.append(waveform[start:split])
            start = split
        chunks.append(waveform[start:total])
        return chunks

    def _find_split_point_energy(self, waveform, start_idx: int, end_idx: int) -> int:
        """Start of the quietest non-overlapping ``min_energy_window_samples`` window in
        ``waveform[start_idx:end_idx]``, or the midpoint when the span is too short to scan."""
        segment = waveform[start_idx:end_idx]
        size = self.min_energy_window_samples
        if segment.shape[0] <= size:
            return (start_idx + end_idx) // 2
        xp = _array_namespace(segment)
        offsets = xp.arange(0, segment.shape[0] - size, size)
        windows = segment[offsets[:, None] + xp.arange(size)]
        return int(start_idx + offsets[int((windows * windows).mean(-1).argmin())])


class CohereAsrAudioProcessor(CohereAsrAudioProcessorMixin, TorchAudioBackend):
    def _seeded_noise(self, length, seed, like):
        generator = torch.Generator(device=like.device).manual_seed(seed)
        return torch.randn(length, dtype=like.dtype, device=like.device, generator=generator)


__all__ = ["CohereAsrAudioProcessor"]
