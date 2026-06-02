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

import math
import warnings

import numpy as np
import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_processing_base import make_legacy_audio_processor_alias
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig, mel_filter_bank


def _gemma4_frame_length_ms_to_win_length(value, config_dict):
    sr = config_dict.get("sample_rate") or config_dict.get("sampling_rate") or 16000
    spec = config_dict.setdefault("spectrogram_config", {})
    stft = spec.setdefault("stft_config", {})
    stft.setdefault("win_length", int(round(sr * value / 1000.0)))


def _gemma4_hop_length_ms_to_hop_length(value, config_dict):
    sr = config_dict.get("sample_rate") or config_dict.get("sampling_rate") or 16000
    spec = config_dict.setdefault("spectrogram_config", {})
    stft = spec.setdefault("stft_config", {})
    stft.setdefault("hop_length", int(round(sr * value / 1000.0)))


class Gemma4AudioProcessor(TorchAudioBackend):
    """Torch sibling of [`Gemma4AudioProcessorNumpy`]. Universal-Speech-Models-style mel
    feature extractor (https://huggingface.co/papers/2303.01037):

    1. Optional dither + input scaling.
    2. Semicausal time padding (``pad_left = win_length // 2`` zeros prepended) so the
       first STFT frame is centered at t=0 (matches ``sl.STFT(time_padding='semicausal')``).
    3. Unfold framing at ``win_length + 1`` samples so HTK-flavor preemphasis can be applied
       per-frame before reducing to ``win_length``.
    4. Periodic Hann window + ``rfft(n=fft_length)`` with implicit right-padding.
    5. ``log(|X| @ mel_filters + mel_floor)`` (HTK mel, no slaney norm).
    6. Optional per-bin mean / stddev normalization.
    7. Frame-aware mask: a mel frame is valid only when every sample in its analysis window
       is real audio (we check the last sample's attention_mask value)."""

    sample_rate = 16000
    force_mono = True
    padding = "longest"
    padding_value = 0.0
    max_length = 480_000
    truncation = True
    pad_to_multiple_of = 128

    # Universal-Speech preemphasis flavour — fixed by the model.
    preemphasis_htk_flavor: bool = True
    fft_overdrive: bool = False
    dither: float = 0.0
    input_scale_factor: float = 1.0

    legacy_field_mapping = {
        "feature_size": "spectrogram_config.mel_scale_config.n_mels",
        "frame_length_ms": _gemma4_frame_length_ms_to_win_length,
        "hop_length_ms": _gemma4_hop_length_ms_to_hop_length,
        "min_frequency": "spectrogram_config.mel_scale_config.f_min",
        "max_frequency": "spectrogram_config.mel_scale_config.f_max",
    }

    # NB: ``n_fft`` is set to 512 (= 2 ** ceil(log2(320))) for the default win_length=320.
    # When the loaded config uses a different ``win_length`` or sets ``fft_overdrive``, the
    # ``__init__`` below recomputes ``n_fft`` and rebuilds the spectrogram_config.
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=320,  # 20 ms at 16 kHz
            hop_length=160,  # 10 ms at 16 kHz
            window_fn="hann_window",
            power=1.0,
            center=False,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
            mel_scale="htk",
            matmul_order="features_first",
        ),
        preemphasis=0.0,
        mel_floor=1e-3,
        log_mode="log",
    )

    def __init__(
        self,
        preemphasis_htk_flavor: bool | None = None,
        fft_overdrive: bool | None = None,
        dither: float | None = None,
        input_scale_factor: float | None = None,
        per_bin_mean=None,
        per_bin_stddev=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if preemphasis_htk_flavor is not None:
            self.preemphasis_htk_flavor = preemphasis_htk_flavor
        if fft_overdrive is not None:
            self.fft_overdrive = fft_overdrive
        if dither is not None:
            self.dither = dither
        if input_scale_factor is not None:
            self.input_scale_factor = input_scale_factor

        # If the loaded config has a non-default `win_length` or `fft_overdrive`, recompute
        # `n_fft` to match the legacy formula and rebuild mel_filters.
        self._maybe_rebuild_for_win_length()

        n_mels = self.spectrogram_config.mel_scale_config.n_mels
        self.per_bin_mean = (
            torch.as_tensor(per_bin_mean).reshape(1, 1, n_mels) if per_bin_mean is not None else None
        )
        self.per_bin_stddev = (
            torch.as_tensor(per_bin_stddev).reshape(1, 1, n_mels) if per_bin_stddev is not None else None
        )

        # Manual periodic-Hann window over win_length — match the numpy sibling so the
        # `frames * window` multiplication is bit-equivalent across backends.
        win_length = self.spectrogram_config.stft_config.win_length
        hann_arange = np.arange(win_length, dtype=np.float32)
        window_np = (0.5 * (1 - np.cos(2 * np.pi * hann_arange / win_length))).astype(np.float32)
        self.window = torch.from_numpy(window_np)

    def _maybe_rebuild_for_win_length(self):
        from dataclasses import replace

        stft_cfg = self.spectrogram_config.stft_config
        win_length = stft_cfg.win_length
        expected_n_fft = 2 ** math.ceil(math.log2(win_length))
        if self.fft_overdrive:
            expected_n_fft *= 2
        if stft_cfg.n_fft != expected_n_fft:
            self.spectrogram_config = replace(
                self.spectrogram_config,
                stft_config=replace(stft_cfg, n_fft=expected_n_fft),
            )
            # Recompute mel filters with the new n_fft.
            self.mel_filters = self._mel_filter_bank(self.spectrogram_config)

    # ── Mel filter bank — uses the HTK helper, no slaney norm ───────────────────────────

    def _mel_filter_bank(self, spectrogram_config):
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mel = mel_filter_bank(
                num_frequency_bins=stft_cfg.n_fft // 2 + 1,
                num_mel_filters=mel_cfg.n_mels,
                min_frequency=mel_cfg.f_min,
                max_frequency=mel_cfg.f_max if mel_cfg.f_max is not None else self.sample_rate / 2,
                sampling_rate=self.sample_rate,
                norm=None,
                mel_scale=mel_cfg.mel_scale,
            )
        # mel_filter_bank returns numpy; the legacy multiplies as numpy float64 then casts
        # the result to float32 (`prepared_speech.astype(np.float32)`). Keep float64 here
        # for the matmul to match.
        return torch.from_numpy(mel)

    # ── STFT pipeline ────────────────────────────────────────────────────────────────────

    def _stft(self, audio, *, spectrogram_config, **kwargs):
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length
        hop_length = stft_cfg.hop_length
        pad_left = win_length // 2
        frame_size_for_unfold = win_length + 1

        # 1. Dither (note: legacy uses np.random.randn — not deterministic; the parity
        # fixture disables dither so this branch is exercised only in production code).
        if self.dither > 0.0:
            noise = torch.randn(audio.shape, dtype=audio.dtype, device=audio.device)
            audio = audio + self.dither * noise

        # 2. Input scaling
        if self.input_scale_factor != 1.0:
            audio = audio * self.input_scale_factor

        # 3. Semicausal time padding: zeros prepended only on the left
        audio = torch.nn.functional.pad(audio, (pad_left, 0), mode="constant", value=0.0)

        # 4. Unfold with size = win_length + 1, then HTK preemphasis (or slice off last sample)
        frames = audio.unfold(-1, frame_size_for_unfold, hop_length)
        frames = self._apply_htk_frame_processing(frames, spectrogram_config)

        # 5. Window + rfft
        window = self.window.to(device=audio.device, dtype=frames.dtype)
        frames = frames * window
        stft = torch.fft.rfft(frames, n=stft_cfg.n_fft, dim=-1)

        # 6. Magnitude (power=1.0). Transpose to (..., freq, num_frames) for the base
        # `_apply_mel_scale` (it transposes back internally for features_first matmul).
        return stft.abs().transpose(-2, -1)

    def _apply_htk_frame_processing(self, frames, spectrogram_config):
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None and preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first = frames[..., :1] * (1.0 - preemphasis)
                rest = frames[..., 1:-1] - preemphasis * frames[..., :-2]
                return torch.cat([first, rest], dim=-1)
            return frames[..., 1:] - preemphasis * frames[..., :-1]
        return frames[..., :-1]

    def _normalize_magnitude(self, features, *, spectrogram_config, **kwargs):
        # Legacy uses `log(mel_spec + mel_floor)`, not `log(clamp(mel_spec, mel_floor))`.
        # `_apply_mel_scale` (base) clamps to `mel_floor` already; we have to recompute the
        # raw value. Easier: override `_apply_mel_scale` below to skip the clamp.
        features = torch.log(features)
        return features.to(torch.float32)

    def _apply_mel_scale(self, features, *, spectrogram_config, **kwargs):
        mel_filters = self.mel_filters.to(device=features.device, dtype=features.dtype)
        # features shape (..., freq, num_frames). features_first means matmul as
        # (..., num_frames, freq) @ (freq, n_mels) → (..., num_frames, n_mels).
        mel_spec = torch.matmul(features.transpose(-2, -1), mel_filters)
        # Legacy: `np.log(mel_spec + mel_floor)`. Done in `_normalize_magnitude` after this.
        return mel_spec + spectrogram_config.mel_floor

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        if audio_ranges is None or "audio_features" not in output:
            return output

        features = output["audio_features"]
        mask = output.get("audio_features_mask")

        # Per-bin normalization
        if self.per_bin_mean is not None:
            mean = self.per_bin_mean.to(device=features.device, dtype=features.dtype)
            features = features - mean
        if self.per_bin_stddev is not None:
            stddev = self.per_bin_stddev.to(device=features.device, dtype=features.dtype)
            features = features / stddev

        # Multiply features by mask (legacy `speech * mask[..., None]`)
        if mask is not None:
            features = features * mask.to(features.dtype).unsqueeze(-1)

        output["audio_features"] = features
        return output

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length
        hop_length = stft_cfg.hop_length
        pad_left = win_length // 2
        frame_size_for_unfold = win_length + 1
        # num_mel_frames = (L + pad_left - frame_size_for_unfold) // hop + 1
        lengths = (audio_lengths + pad_left - frame_size_for_unfold) // hop_length + 1
        if isinstance(lengths, np.ndarray):
            lengths = np.maximum(0, lengths)
        elif isinstance(lengths, torch.Tensor):
            lengths = torch.clamp(lengths, min=0)
        else:
            lengths = max(0, int(lengths))
        return lengths


Gemma4AudioFeatureExtractor = make_legacy_audio_processor_alias(
    Gemma4AudioProcessor, "Gemma4AudioFeatureExtractor"
)


__all__ = ["Gemma4AudioProcessor", "Gemma4AudioFeatureExtractor"]
