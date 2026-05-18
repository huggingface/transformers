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

from ...audio_processing_backends import NumpyAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


EPSILON = 1e-5


class CohereAsrAudioProcessorNumpy(NumpyAudioBackend):
    """NumPy sibling of [`CohereAsrAudioProcessor`]. Bit-exact to the torch sibling within
    the float32 noise floor (ADR 0001) when ``dither=0`` — the deterministic torch-RNG dither
    cannot be reproduced bit-exactly with numpy's RNG, so the parity test disables it. See
    [`CohereAsrAudioProcessor`] for the full pipeline description."""

    sample_rate = 16000
    force_mono = True
    padding = "longest"

    dither: float = 1e-5
    max_audio_clip_s: float = 35.0
    overlap_chunk_second: float = 5.0
    min_energy_window_samples: int = 1600

    legacy_field_mapping = {
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
    }

    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            hop_length=160,
            win_length=400,
            window_fn="hann_window",
            power=2.0,
            pad_mode="constant",
            periodic=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=0.0,
            norm="slaney",
            mel_scale="slaney",
        ),
        preemphasis=0.97,
        log_mode="log",
        mel_floor=2**-24,
    )

    def __init__(
        self,
        dither: float | None = None,
        max_audio_clip_s: float | None = None,
        overlap_chunk_second: float | None = None,
        min_energy_window_samples: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dither is not None:
            self.dither = dither
        if max_audio_clip_s is not None:
            self.max_audio_clip_s = max_audio_clip_s
        if overlap_chunk_second is not None:
            self.overlap_chunk_second = overlap_chunk_second
        if min_energy_window_samples is not None:
            self.min_energy_window_samples = min_energy_window_samples

    def _mel_filter_bank(self, spectrogram_config):
        """Replicate librosa's per-band float32 accumulation pattern for bit-exact FE parity."""
        from ...audio_utils import hertz_to_mel, mel_to_hertz

        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        n_fft = stft_cfg.n_fft
        n_mels = mel_cfg.n_mels
        f_min = mel_cfg.f_min
        f_max = mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2

        mel_min = hertz_to_mel(f_min, mel_scale=mel_cfg.mel_scale)
        mel_max = hertz_to_mel(f_max, mel_scale=mel_cfg.mel_scale)
        mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
        filter_freqs = mel_to_hertz(mel_pts.copy(), mel_scale=mel_cfg.mel_scale)
        fft_freqs = np.linspace(0, self.sample_rate / 2, 1 + n_fft // 2)

        fdiff = np.diff(filter_freqs)
        ramps = np.subtract.outer(filter_freqs, fft_freqs)

        weights = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
        for i in range(n_mels):
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        if mel_cfg.norm == "slaney":
            enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
            weights *= enorm[:, np.newaxis]

        return weights.T.astype(np.float32)

    def _needs_manual_framing(self, spectrogram_config):
        # Preemphasis is handled waveform-level in `_stft`; no per-frame processing needed.
        return spectrogram_config.remove_dc_offset or spectrogram_config.stft_config.left_align_fft

    def _apply_dither(self, audio, audio_lengths):
        """Deterministic per-utterance dither. Numpy RNG state differs from torch's, so this
        path is NOT bit-exact across backends — the parity test fixture sets ``dither=0`` to
        avoid the divergence (ADR 0001)."""
        if self.dither <= 0:
            return audio
        for i in range(audio.shape[0]):
            valid_samples = min(int(audio_lengths[i]), audio.shape[1])
            if valid_samples <= 0:
                continue
            rng = np.random.RandomState(valid_samples)
            noise = rng.standard_normal(valid_samples).astype(audio.dtype)
            audio[i, :valid_samples] = audio[i, :valid_samples] + self.dither * noise
        return audio

    def _stft(self, audio, *, spectrogram_config, audio_ranges=None, **kwargs):
        audio_lengths = (
            np.asarray([end - start for start, end in audio_ranges]) if audio_ranges is not None else None
        )

        if audio_lengths is not None:
            audio = self._apply_dither(audio.copy(), audio_lengths)

        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            audio = np.concatenate([audio[:, :1], audio[:, 1:] - preemphasis * audio[:, :-1]], axis=1)
            if audio_lengths is not None:
                timemask = np.expand_dims(np.arange(audio.shape[-1]), axis=0) < np.expand_dims(audio_lengths, axis=1)
                audio = np.where(timemask, audio, 0.0).astype(audio.dtype, copy=False)

        return super()._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        magnitudes = np.abs(stft_out)
        if power != 1.0:
            magnitudes = magnitudes ** power
        return magnitudes

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        return np.matmul(self.mel_filters.T, features)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        features = np.log(features + spectrogram_config.mel_floor)
        return np.transpose(features, axes=(0, 2, 1))

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output

        features = output["audio_features"]
        stft_cfg = self.spectrogram_config.stft_config
        audio_lengths = np.asarray([end - start for start, end in audio_ranges])
        features_lengths = np.floor_divide(
            audio_lengths + stft_cfg.n_fft // 2 * 2 - stft_cfg.n_fft, stft_cfg.hop_length
        )
        attention_mask = np.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = np.expand_dims(attention_mask, axis=-1)
        features_lengths_f = features_lengths.astype(features.dtype)
        mel_masked = features * mask
        mean = np.expand_dims(mel_masked.sum(axis=1) / np.expand_dims(features_lengths_f, axis=-1), axis=1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(axis=1) / np.expand_dims(
            features_lengths_f - 1, axis=-1
        )
        std = np.expand_dims(np.sqrt(variance), axis=1)
        output["audio_features"] = (features - mean) / (std + EPSILON) * mask
        return output

    # ── Long-audio chunking (mirrors the torch sibling) ─────────────────────────────────

    def _preprocess_audio_like_inputs(self, audio, *args, sample_rate=None, **kwargs):
        prepared = self._prepare_audio_like_inputs(audio=audio, sample_rate=sample_rate)
        chunked, audio_chunk_index = self._split_audio_chunks(prepared)
        result = self._preprocess(chunked, *args, **kwargs)
        return_tensors = kwargs.get("return_tensors")
        result["audio_chunk_index"] = self._encode_chunk_index(audio_chunk_index, return_tensors)
        return result

    def _encode_chunk_index(self, audio_chunk_index, return_tensors):
        encoded = [[s, -1 if c is None else c] for s, c in audio_chunk_index]
        if return_tensors == "pt":
            import torch

            return torch.tensor(encoded, dtype=torch.long)
        return np.asarray(encoded, dtype=np.int64)

    def _split_audio_chunks(self, prepared_audio):
        fast_path_threshold_s = max(0.0, self.max_audio_clip_s - self.overlap_chunk_second)
        chunked: list = []
        audio_chunk_index: list[tuple[int, int | None]] = []
        for sample_idx, waveform in enumerate(prepared_audio):
            duration_s = waveform.shape[0] / self.sample_rate
            if duration_s <= fast_path_threshold_s:
                chunked.append(waveform)
                audio_chunk_index.append((sample_idx, None))
            else:
                chunks = self._split_single_audio(waveform)
                for chunk_idx, chunk in enumerate(chunks):
                    chunked.append(chunk)
                    audio_chunk_index.append((sample_idx, chunk_idx))
        return chunked, audio_chunk_index

    def _split_single_audio(self, waveform):
        chunk_size = max(1, int(round(self.max_audio_clip_s * self.sample_rate)))
        boundary_context_size = max(1, int(round(self.overlap_chunk_second * self.sample_rate)))
        total_samples = waveform.shape[0]
        if total_samples <= chunk_size:
            return [waveform]

        chunks_meta: list[tuple[int, int]] = []
        idx = 0
        while idx < total_samples:
            if idx + chunk_size >= total_samples:
                chunks_meta.append((idx, total_samples))
                break
            search_start = max(idx, idx + chunk_size - boundary_context_size)
            search_end = min(idx + chunk_size, total_samples)
            split_point = self._find_split_point_energy(waveform, search_start, search_end)
            split_point = max(idx + 1, min(split_point, total_samples))
            chunks_meta.append((idx, split_point))
            idx = split_point

        return [waveform[start:end] for start, end in chunks_meta if end > start]

    def _find_split_point_energy(self, waveform, start_idx: int, end_idx: int) -> int:
        segment = waveform[start_idx:end_idx]
        if segment.shape[0] <= self.min_energy_window_samples:
            return (start_idx + end_idx) // 2

        min_energy = float("inf")
        quietest_idx = start_idx
        upper = segment.shape[0] - self.min_energy_window_samples
        for i in range(0, upper, self.min_energy_window_samples):
            window = segment[i : i + self.min_energy_window_samples]
            energy = float(np.sqrt(np.mean(window * window)))
            if energy < min_energy:
                min_energy = energy
                quietest_idx = start_idx + i
        return quietest_idx


__all__ = ["CohereAsrAudioProcessorNumpy"]
