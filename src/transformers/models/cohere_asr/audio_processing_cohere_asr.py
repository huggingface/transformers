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
from ...audio_processing_base import make_legacy_audio_processor_alias
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


EPSILON = 1e-5


class CohereAsrAudioProcessor(TorchAudioBackend):
    """Torch sibling of [`CohereAsrAudioProcessorNumpy`]. Cohere-ASR preprocessing pipeline:

    1. Energy-based chunking of long-form audio (legacy ``max_audio_clip_s`` split at quietest
       window inside the boundary search region).
    2. Deterministic, batch-invariant dithering — each utterance's RNG is seeded by its
       valid sample count so identical waveforms produce identical features regardless of
       batch composition.
    3. Waveform-level preemphasis with explicit zeroing of padded samples (mirrors the
       Parakeet pattern in [ADR 0005](docs/adr/0005-hook-surface-boundary.md)).
    4. Native ``torch.stft`` with ``pad_mode="constant"`` and a non-periodic Hann window.
    5. ``log(mel @ |X|^2 + 2^-24)`` mel-filterbank projection (librosa-style per-band
       float32 truncation in ``_mel_filter_bank``).
    6. Per-utterance mean/variance normalization on the padded batch, applied in
       ``_postprocess_output`` (ADR 0005)."""

    sample_rate = 16000
    force_mono = True
    padding = "longest"

    # Cohere-ASR-specific kwargs (not part of SpectrogramConfig).
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

    # ── Mel filter bank (librosa-bit-exact, shared with the numpy sibling) ────────────────

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

        return torch.from_numpy(weights.T).to(torch.float32)

    # ── STFT pipeline ────────────────────────────────────────────────────────────────────

    def _needs_manual_framing(self, spectrogram_config):
        # Preemphasis is handled waveform-level in `_stft` (see Parakeet), so no per-frame
        # processing is needed.
        return spectrogram_config.remove_dc_offset or spectrogram_config.stft_config.left_align_fft

    def _apply_dither(self, audio, audio_lengths):
        """Deterministic per-utterance dither: each row is seeded by its valid sample count
        so dither is invariant to batch composition (matches the legacy FE)."""
        if self.dither <= 0:
            return audio
        generator = torch.Generator(device=audio.device)
        for i in range(audio.shape[0]):
            valid_samples = min(int(audio_lengths[i].item()), audio.shape[1])
            if valid_samples <= 0:
                continue
            generator.manual_seed(valid_samples)
            noise = torch.randn(valid_samples, dtype=audio.dtype, device=audio.device, generator=generator)
            audio[i, :valid_samples] = audio[i, :valid_samples] + self.dither * noise
        return audio

    def _stft(self, audio, *, spectrogram_config, audio_ranges=None, **kwargs):
        audio_lengths = (
            torch.tensor([end - start for start, end in audio_ranges], device=audio.device)
            if audio_ranges is not None
            else None
        )

        # 1. Dither
        if audio_lengths is not None:
            audio = self._apply_dither(audio.clone(), audio_lengths)

        # 2. Waveform-level preemphasis with masking to zero out padding
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None:
            audio = torch.cat([audio[:, :1], audio[:, 1:] - preemphasis * audio[:, :-1]], dim=1)
            if audio_lengths is not None:
                timemask = torch.arange(audio.shape[-1], device=audio.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
                audio = audio.masked_fill(~timemask, 0.0)

        return super()._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        # Match the legacy view_as_real + sqrt(real²+imag²) ** power pattern. `abs() ** power`
        # is numerically equivalent within FFT-library noise.
        magnitudes = torch.view_as_real(stft_out)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))
        if power != 1.0:
            magnitudes = magnitudes.pow(power)
        return magnitudes

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        return torch.matmul(self.mel_filters.T, features)

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Per-utterance mean/var normalization is in `_postprocess_output` per ADR 0005;
        # this hook is pointwise. The legacy form is `log(x + guard)`, not `log(clamp(x, guard))`.
        features = torch.log(features + spectrogram_config.mel_floor)
        return features.permute(0, 2, 1)

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output

        features = output["audio_features"]
        stft_cfg = self.spectrogram_config.stft_config
        audio_lengths = torch.tensor([end - start for start, end in audio_ranges])
        features_lengths = torch.floor_divide(
            audio_lengths + stft_cfg.n_fft // 2 * 2 - stft_cfg.n_fft, stft_cfg.hop_length
        )
        attention_mask = torch.arange(features.shape[1])[None, :] < features_lengths[:, None]
        mask = attention_mask.unsqueeze(-1)
        mel_masked = features * mask
        mean = (mel_masked.sum(dim=1) / features_lengths.unsqueeze(-1)).unsqueeze(1)
        variance = ((mel_masked - mean) ** 2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(1)
        output["audio_features"] = (features - mean) / (std + EPSILON) * mask
        return output

    # ── Long-audio chunking (cohere_asr-specific, doesn't fit existing hooks) ────────────
    #
    # The legacy FE splits long audio at the quietest sample-window inside the boundary
    # search region. This is a model-specific pre-extraction transform (1 audio → N chunks);
    # it doesn't fit `_normalize_magnitude` (pointwise) or `_postprocess_output` (post-batch).
    # Implemented as an override of `_preprocess_audio_like_inputs` so the chunking happens
    # on the prepared (per-item) audio list before padding/extraction.

    def _preprocess_audio_like_inputs(self, audio, *args, sample_rate=None, **kwargs):
        prepared = self._prepare_audio_like_inputs(audio=audio, sample_rate=sample_rate)
        chunked, audio_chunk_index = self._split_audio_chunks(prepared)
        result = self._preprocess(chunked, *args, **kwargs)
        # Materialise `audio_chunk_index` as an integer tensor so it survives both
        # `convert_to_tensors` and cross-backend parity comparison. `chunk_idx is None`
        # (the fast-path marker) becomes the sentinel -1.
        return_tensors = kwargs.get("return_tensors")
        result["audio_chunk_index"] = self._encode_chunk_index(audio_chunk_index, return_tensors)
        return result

    def _encode_chunk_index(self, audio_chunk_index, return_tensors):
        encoded = [[s, -1 if c is None else c] for s, c in audio_chunk_index]
        if return_tensors == "pt":
            return torch.tensor(encoded, dtype=torch.long)
        return np.asarray(encoded, dtype=np.int64)

    def _split_audio_chunks(self, prepared_audio):
        """Energy-based chunking: each audio whose duration exceeds
        ``max_audio_clip_s - overlap_chunk_second`` is split at the quietest window inside
        the boundary search region. Returns (list of chunks, list of (sample_idx, chunk_idx)).
        ``chunk_idx`` is ``None`` for audio that took the fast (no-split) path."""
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
            energy = torch.sqrt(torch.mean(window * window)).item()
            if energy < min_energy:
                min_energy = energy
                quietest_idx = start_idx + i
        return quietest_idx


CohereAsrFeatureExtractor = make_legacy_audio_processor_alias(
    CohereAsrAudioProcessor, "CohereAsrFeatureExtractor"
)


__all__ = ["CohereAsrAudioProcessor", "CohereAsrFeatureExtractor"]
