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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, _array_namespace


EPSILON = 1e-5


class CohereAsrAudioProcessorMixin:
    """Backend-agnostic Cohere-ASR logic shared by the numpy and torch siblings; only the
    RNG-dependent dither and the torch mel/magnitude leaves live in the sibling classes."""

    sampling_rate = 16000
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
        preemphasis_mode="waveform",
        log_mode="log",
        mel_floor=0.0,  # no clamp; the log guard is pre_log_offset
        pre_log_offset=2**-24,
    )

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # transpose to (batch, frames, mels)
        features = super()._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)
        return features.swapaxes(-2, -1)

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output
        stft_cfg = self.spectrogram_config.stft_config
        feature_lengths = [
            (end - start + stft_cfg.n_fft // 2 * 2 - stft_cfg.n_fft) // stft_cfg.hop_length
            for start, end in audio_ranges
        ]
        output["audio_features"] = self._masked_mean_var_normalize(
            output["audio_features"], feature_lengths, epsilon=EPSILON
        )
        return output

    def _preprocess_audio_like_inputs(self, audio, *args, sampling_rate=None, **kwargs):
        # long-audio chunking (1 audio → N chunks) happens before padding/extraction
        prepared = self._prepare_audio_like_inputs(audio=audio, sampling_rate=sampling_rate)
        chunked, audio_chunk_index = self._split_audio_chunks(prepared)
        result = self._preprocess(chunked, *args, **kwargs)
        result["audio_chunk_index"] = self._encode_chunk_index(audio_chunk_index, kwargs.get("return_tensors"))
        return result

    def _encode_chunk_index(self, audio_chunk_index, return_tensors):
        # integer-encode so it survives `convert_to_tensors`; no-chunking marker None -> -1
        encoded = [[s, -1 if c is None else c] for s, c in audio_chunk_index]
        if return_tensors == "pt":
            import torch

            return torch.tensor(encoded, dtype=torch.long)
        return np.asarray(encoded, dtype=np.int64)

    def _split_audio_chunks(self, prepared_audio):
        """Split audio longer than ``max_audio_clip_s - overlap_chunk_second`` at the
        quietest window. Returns (chunks, [(sample_idx, chunk_idx or None)])."""
        fast_path_threshold_s = max(0.0, self.max_audio_clip_s - self.overlap_chunk_second)
        chunked: list = []
        audio_chunk_index: list[tuple[int, int | None]] = []
        for sample_idx, waveform in enumerate(prepared_audio):
            duration_s = waveform.shape[0] / self.sampling_rate
            if duration_s <= fast_path_threshold_s:
                chunked.append(waveform)
                audio_chunk_index.append((sample_idx, None))
            else:
                for chunk_idx, chunk in enumerate(self._split_single_audio(waveform)):
                    chunked.append(chunk)
                    audio_chunk_index.append((sample_idx, chunk_idx))
        return chunked, audio_chunk_index

    def _split_single_audio(self, waveform):
        chunk_size = max(1, int(round(self.max_audio_clip_s * self.sampling_rate)))
        boundary_context_size = max(1, int(round(self.overlap_chunk_second * self.sampling_rate)))
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

        xp = _array_namespace(segment)
        min_energy = float("inf")
        quietest_idx = start_idx
        upper = segment.shape[0] - self.min_energy_window_samples
        for i in range(0, upper, self.min_energy_window_samples):
            window = segment[i : i + self.min_energy_window_samples]
            energy = float(xp.sqrt(xp.mean(window * window)))
            if energy < min_energy:
                min_energy = energy
                quietest_idx = start_idx + i
        return quietest_idx


class CohereAsrAudioProcessorNumpy(CohereAsrAudioProcessorMixin, NumpyAudioBackend):
    """NumPy sibling of [`CohereAsrAudioProcessor`]. Bit-exact to the torch sibling within
    the float32 noise floor when ``dither=0`` — the deterministic torch-RNG dither cannot
    be reproduced bit-exactly with numpy's RNG, so the parity test disables it. See
    [`CohereAsrAudioProcessor`] for the full pipeline description."""

    def _apply_dither(self, audio, audio_ranges=None):
        """Deterministic per-utterance dither, seeded by valid sample count. Numpy and torch
        RNGs differ, so the parity fixture sets ``dither=0``. Runs before `waveform_scale`
        and waveform preemphasis in the base `_stft` (ordering is load-bearing)."""
        if self.dither <= 0 or audio_ranges is None:
            return audio
        audio = audio.copy()
        for i, (start, end) in enumerate(audio_ranges):
            valid_samples = min(end - start, audio.shape[1])
            if valid_samples <= 0:
                continue
            rng = np.random.RandomState(valid_samples)
            noise = rng.standard_normal(valid_samples).astype(audio.dtype)
            audio[i, :valid_samples] = audio[i, :valid_samples] + self.dither * noise
        return audio


__all__ = ["CohereAsrAudioProcessorNumpy"]
