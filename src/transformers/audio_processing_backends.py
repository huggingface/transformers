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
from typing import Unpack

import numpy as np

from .audio_processing_utils import BaseAudioProcessor
from .audio_utils import (
    _create_triangular_filter_bank,
    hertz_to_mel,
    mel_to_hertz,
)
from .processing_utils import AudioKwargs
from .utils import is_torch_available, logging, requires_backends


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


class NumpyAudioBackend(BaseAudioProcessor):
    """NumPy backend for portable CPU-only audio processing."""

    def __init__(self, *args, **kwargs: Unpack[AudioKwargs]):
        super().__init__(*args, **kwargs)
        self._set_attributes(**kwargs)

    @property
    def backend(self) -> str:
        return "numpy"

    # ── Backend array-API primitives ─────────────────────────────────────

    def _amax_over_features(self, x):
        if x.ndim > 2:
            return x.max(axis=tuple(range(1, x.ndim)), keepdims=True)
        return x.max()

    # ── STFT pipeline ─────────────────────────────────────────────────────

    def _preemphasize_waveform(self, audio, preemphasis, audio_ranges=None):
        out = audio.copy()
        out[..., 1:] = audio[..., 1:] - preemphasis * audio[..., :-1]  # first sample unchanged
        if audio_ranges is not None:
            lengths = np.asarray([end - start for start, end in audio_ranges])
            mask = np.arange(out.shape[-1])[None, :] < lengths[:, None]
            out = np.where(mask, out, 0.0).astype(audio.dtype, copy=False)
        return out

    def _dither_waveform(self, audio, audio_ranges=None, *, dither):
        return audio + (dither * np.random.randn(*audio.shape)).astype(audio.dtype)

    # ── Mel scale & normalization ─────────────────────────────────────────
    #
    # The base `_mel_filter_bank` dispatcher (audio_processing_utils) resolves geometry
    # and dtype; the three leaves below own the numerical construction. Each backend's
    # leaves deliberately implement their own ecosystem's rounding pattern: these numpy
    # leaves are bit-exact against librosa and the legacy numpy feature extractors, the
    # torch leaves against torchaudio / torchaudio.compliance.kaldi. The two backends are
    # numerically equivalent but NOT bit-identical.

    @staticmethod
    def _np_triangular_banks(fft_freqs, filter_freqs, computation_dtype):
        """Triangular bank with the numpy ecosystem's dtype policy.

        With no computation dtype, replicate librosa's per-band float32 rounding:
        slopes computed in float64 with each band's column cast to float32 on
        assignment (librosa assigns rows into a float32-initialized array, which
        rounds differently than casting a float64 matrix at the end). With a dtype,
        plain full-precision construction cast to that dtype.
        """
        if computation_dtype is None:
            num_frequency_bins = fft_freqs.shape[0]
            num_mel_filters = filter_freqs.shape[0] - 2
            filter_diff = np.diff(filter_freqs)
            ramps = np.subtract.outer(filter_freqs, fft_freqs)  # (num_mel_filters+2, num_frequency_bins)
            mel_filters = np.zeros((num_frequency_bins, num_mel_filters), dtype=np.float32)
            for i in range(num_mel_filters):
                lower = -ramps[i] / filter_diff[i]
                upper = ramps[i + 2] / filter_diff[i + 1]
                mel_filters[:, i] = np.maximum(0, np.minimum(lower, upper)).astype(np.float32)
            return mel_filters
        return _create_triangular_filter_bank(fft_freqs, filter_freqs).astype(np.dtype(computation_dtype), copy=False)

    def _kaldi_exact_mel_banks(
        self,
        num_mel_filters,
        num_frequency_bins,
        min_frequency,
        max_frequency,
        sampling_rate,
        n_fft,
        mel_cfg,
        computation_dtype,
    ):
        """Mel-space triangularization, numpy-ecosystem semantics.

        Bit-exact against the legacy numpy feature extractors (e.g. SeamlessM4T):
        ``hertz_to_mel(mel_scale=...)`` on float64 bin frequencies and filter edges from
        ``np.linspace`` in mel space. Deliberately NOT torchaudio's hardcoded
        ``1127 * log`` float32 ``get_mel_banks`` arithmetic — the torch leaf owns that
        rounding pattern; the two leaves are numerically equivalent, not bit-identical.
        """
        mel_min = hertz_to_mel(min_frequency, mel_scale=mel_cfg.mel_scale)
        mel_max = hertz_to_mel(max_frequency, mel_scale=mel_cfg.mel_scale)
        filter_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
        fft_bin_width = sampling_rate / n_fft
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_cfg.mel_scale)
        return self._np_triangular_banks(fft_freqs, filter_freqs, computation_dtype)

    def _kaldi_mel_banks_with_zero_bands(
        self,
        num_mel_filters,
        num_frequency_bins,
        min_frequency,
        max_frequency,
        sampling_rate,
        n_fft,
        mel_cfg,
        computation_dtype,
    ):
        """Mel-space triangularization (numpy-ecosystem semantics, see
        `_kaldi_exact_mel_banks`) with the lowest ``bands_to_zero`` bins zeroed."""
        mel_min = hertz_to_mel(min_frequency, mel_scale=mel_cfg.mel_scale)
        mel_max = hertz_to_mel(max_frequency, mel_scale=mel_cfg.mel_scale)
        filter_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
        fft_bin_width = sampling_rate / n_fft
        fft_freqs = hertz_to_mel(
            fft_bin_width * np.arange(mel_cfg.bands_to_zero, num_frequency_bins), mel_scale=mel_cfg.mel_scale
        )
        mel_filters = self._np_triangular_banks(fft_freqs, filter_freqs, computation_dtype)
        if mel_cfg.bands_to_zero > 0:
            mel_filters = np.pad(mel_filters, ((mel_cfg.bands_to_zero, 0), (0, 0)))
        return mel_filters

    def _standard_mel_banks(
        self,
        num_mel_filters,
        num_frequency_bins,
        min_frequency,
        max_frequency,
        sampling_rate,
        n_fft,
        mel_cfg,
        computation_dtype,
    ):
        """Standard triangular mel filter bank, numpy-ecosystem semantics.

        Bit-exact against librosa's filters (and the legacy numpy feature extractors
        built on them): FFT bin frequencies always use the float64
        ``linspace(0, sr // 2, bins)`` form regardless of ``frequency_bin_mode``, and
        the slaney area-norm is applied after the dtype policy (i.e. after the per-band
        float32 cast when no computation dtype is set — librosa's rounding order).
        The torch leaf implements torchaudio's rounding instead; the two leaves are
        numerically equivalent, not bit-identical.
        """
        mel_min = hertz_to_mel(min_frequency, mel_scale=mel_cfg.mel_scale)
        mel_max = hertz_to_mel(max_frequency, mel_scale=mel_cfg.mel_scale)
        mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
        filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_cfg.mel_scale)
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
        mel_filters = self._np_triangular_banks(fft_freqs, filter_freqs, computation_dtype)
        if mel_cfg.norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
            mel_filters *= np.expand_dims(enorm, 0)
        if mel_cfg.bands_to_zero > 0:
            mel_filters = np.pad(mel_filters, ((mel_cfg.bands_to_zero, 0), (0, 0)))
        return mel_filters

    def _cast_mel_filters_to_default_float(self, mel_filters):
        return mel_filters.astype(np.float32, copy=False)

    def _project_to_mel(self, features, *, spectrogram_config, mel_filters=None, **kwargs):
        mel_filters = self.mel_filters if mel_filters is None else mel_filters
        mel_filters = mel_filters.astype(features.dtype, copy=False)
        if spectrogram_config.mel_scale_config.matmul_order == "features_first":
            mel_spec = np.matmul(features.swapaxes(-2, -1), mel_filters)
        else:
            mel_spec = np.matmul(mel_filters.T, features)
        return np.maximum(spectrogram_config.mel_floor, mel_spec)


class TorchAudioBackend(BaseAudioProcessor):
    """Torch backend for audio processing."""

    def __init__(self, *args, **kwargs: Unpack[AudioKwargs]):
        super().__init__(*args, **kwargs)
        self._set_attributes(**kwargs)

    @property
    def backend(self) -> str:
        return "torch"

    # ── Backend array-API primitives ─────────────────────────────────────

    _transfer_chunk_bytes = 8 * 2**20

    @staticmethod
    def _host_to_device(values, device, dtype=None):
        """Copy host data (lengths, ranges, filter banks) to `device` without blocking the host.

        A copy from pageable memory waits for every kernel already queued on the stream, so each one
        stalled the Python thread until the GPU drained -- including a model forward still running
        from the previous batch, which kept preprocessing from overlapping with inference. Staging
        through pinned memory makes the copy asynchronous.
        """
        tensor = torch.as_tensor(values, dtype=dtype)
        device = torch.device(device)
        if tensor.device == device:
            return tensor
        if device.type != "cuda" or tensor.device.type != "cpu":
            return tensor.to(device)
        return tensor.pin_memory().to(device, non_blocking=True)

    def _to_device(self, audio, device):
        if device is None:
            return audio
        device = torch.device(device)
        if all(audio_el.device == device for audio_el in audio):
            return audio
        if (
            device.type != "cuda"
            or len(audio) == 1
            or any(audio_el.device.type != "cpu" or audio_el.ndim != 1 for audio_el in audio)
            or len({audio_el.dtype for audio_el in audio}) > 1
        ):
            return [audio_el.to(device) for audio_el in audio]
        # Stage through pinned memory in ~8 MB chunks, so packing chunk k+1 on the host overlaps the DMA
        # of chunk k and only the last chunk's copy is exposed. Per-waveform pageable copies each blocked
        # the host, which made the transfer ~4x slower than the bytes it moves. The waveforms come back
        # as views of one device buffer.
        lengths = [audio_el.shape[-1] for audio_el in audio]
        flat = torch.empty(sum(lengths), dtype=audio[0].dtype, device=device)
        chunk_samples = self._transfer_chunk_bytes // flat.element_size()
        offset = start = pending = 0
        for end, length in enumerate(lengths, start=1):
            pending += length
            if pending < chunk_samples and end < len(audio):
                continue
            host = torch.cat(audio[start:end], out=torch.empty(pending, dtype=flat.dtype, pin_memory=True))
            flat[offset : offset + pending].copy_(host, non_blocking=True)
            offset, start, pending = offset + pending, end, 0
        return list(flat.split(lengths))

    def _pad_waveforms(self, audio, max_length, *, padding_side, padding_value, **kwargs):
        # Batched assembly only pays off on an accelerator: on CPU the per-item pads are cheaper than a
        # scatter, and a single waveform has nothing to amortize.
        if (
            len(audio) == 1
            or audio[0].device.type == "cpu"
            or padding_side not in ("left", "right")
            or any(audio_el.ndim != 1 or audio_el.device != audio[0].device for audio_el in audio)
            or len({audio_el.dtype for audio_el in audio}) > 1
            or any(audio_el.shape[-1] > max_length for audio_el in audio)
        ):
            return super()._pad_waveforms(
                audio, max_length, padding_side=padding_side, padding_value=padding_value, **kwargs
            )
        # One allocation and one scatter for the whole batch: per-waveform `pad` paid an allocation, a
        # fill and a copy per item, and the Python loop over rows kept the host busy while the device idled.
        # `masked_scatter_` fills the mask in row-major order, which is exactly the concatenation order
        # for either padding side, so the result is a pure copy -- bit-identical to per-item padding.
        width = max_length
        device = audio[0].device
        lengths = self._host_to_device([audio_el.shape[-1] for audio_el in audio], device)
        positions = torch.arange(width, device=device)
        mask = positions >= width - lengths[:, None] if padding_side == "left" else positions < lengths[:, None]
        batch = torch.full((len(audio), width), padding_value, dtype=audio[0].dtype, device=device)
        return list(batch.masked_scatter_(mask, torch.cat(audio)))

    def _amax_over_features(self, x):
        return x.amax(dim=(-2, -1), keepdim=True)

    # ── STFT pipeline ─────────────────────────────────────────────────────

    def _preemphasize_waveform(self, audio, preemphasis, audio_ranges=None):
        audio = torch.cat([audio[..., :1], audio[..., 1:] - preemphasis * audio[..., :-1]], dim=-1)
        if audio_ranges is not None:
            lengths = self._host_to_device([end - start for start, end in audio_ranges], audio.device)
            mask = torch.arange(audio.shape[-1], device=audio.device).unsqueeze(0) < lengths.unsqueeze(1)
            audio = audio.masked_fill(~mask, 0.0)
        return audio

    def _dither_waveform(self, audio, audio_ranges=None, *, dither):
        noise = torch.randn(audio.shape, dtype=audio.dtype, device=audio.device)
        return audio + dither * noise

    # ── Mel scale & normalization ─────────────────────────────────────────
    #
    # The base `_mel_filter_bank` dispatcher (audio_processing_utils) resolves geometry
    # and dtype; the three leaves below own the numerical construction. Each backend's
    # leaves deliberately implement their own ecosystem's rounding pattern: these torch
    # leaves are bit-exact against torchaudio / torchaudio.compliance.kaldi, the numpy
    # leaves against librosa and the legacy numpy feature extractors. The two backends
    # are numerically equivalent but NOT bit-identical.

    @staticmethod
    def _kaldi_exact_mel_banks(
        num_mel_filters,
        num_frequency_bins,
        min_frequency,
        max_frequency,
        sampling_rate,
        n_fft,
        mel_cfg,
        computation_dtype,
    ):
        """Matches torchaudio.compliance.kaldi.get_mel_banks exactly.

        Hardcoded ``1127 * log`` kaldi mel scale and ``get_mel_banks``'s edge arithmetic,
        in torch's default float32 when no computation dtype is set. The numpy leaf owns
        the legacy numpy-FE construction instead (``hertz_to_mel`` + linspace in float64);
        the two leaves are numerically equivalent, not bit-identical.
        """
        dtype = getattr(torch, computation_dtype) if computation_dtype else None
        num_fft_bins = n_fft // 2
        fft_bin_width = sampling_rate / n_fft
        mel_low = 1127.0 * math.log(1.0 + min_frequency / 700.0)
        mel_high = 1127.0 * math.log(1.0 + max_frequency / 700.0)
        mel_delta = (mel_high - mel_low) / (num_mel_filters + 1)

        bin_idx = torch.arange(num_mel_filters, dtype=dtype).unsqueeze(1)
        left_mel = mel_low + bin_idx * mel_delta
        center_mel = mel_low + (bin_idx + 1.0) * mel_delta
        right_mel = mel_low + (bin_idx + 2.0) * mel_delta

        mel = 1127.0 * (1.0 + fft_bin_width * torch.arange(num_fft_bins, dtype=dtype) / 700.0).log()
        mel = mel.unsqueeze(0)

        up_slope = (mel - left_mel) / (center_mel - left_mel)
        down_slope = (right_mel - mel) / (right_mel - center_mel)
        banks = torch.max(torch.zeros(1, dtype=dtype), torch.min(up_slope, down_slope))
        banks = torch.nn.functional.pad(banks, (0, 1), mode="constant", value=0)
        return banks.T

    @staticmethod
    def _kaldi_mel_banks_with_zero_bands(
        num_mel_filters,
        num_frequency_bins,
        min_frequency,
        max_frequency,
        sampling_rate,
        n_fft,
        mel_cfg,
        computation_dtype,
    ):
        """Kaldi-style (triangularize in mel space) with optional zeroed low bands."""
        dtype = getattr(torch, computation_dtype) if computation_dtype else None
        mel_min = hertz_to_mel(min_frequency, mel_scale=mel_cfg.mel_scale)
        mel_max = hertz_to_mel(max_frequency, mel_scale=mel_cfg.mel_scale)
        filter_freqs = torch.linspace(mel_min, mel_max, num_mel_filters + 2, dtype=dtype)

        fft_bin_width = sampling_rate / n_fft
        hz_freqs = fft_bin_width * torch.arange(mel_cfg.bands_to_zero, num_frequency_bins, dtype=dtype)
        fft_freqs = hertz_to_mel(hz_freqs, mel_scale=mel_cfg.mel_scale)

        mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
        if mel_cfg.bands_to_zero > 0:
            mel_filters = torch.nn.functional.pad(mel_filters, (0, 0, mel_cfg.bands_to_zero, 0))
        return mel_filters

    @staticmethod
    def _standard_mel_banks(
        num_mel_filters,
        num_frequency_bins,
        min_frequency,
        max_frequency,
        sampling_rate,
        n_fft,
        mel_cfg,
        computation_dtype,
    ):
        """Standard (non-kaldi) triangular mel filter bank, torchaudio-ecosystem semantics.

        Bit-exact against ``torchaudio.functional.melscale_fbanks`` in the default-dtype
        case; the numpy leaf owns librosa's rounding pattern instead, so the two leaves are
        numerically equivalent but not bit-identical.

        ``bank_rounding="librosa"`` opts this leaf into librosa's pattern too, for models
        whose legacy extractor built its filters with ``librosa.filters.mel`` (Parakeet,
        Cohere-ASR): float64 ``linspace`` bins regardless of ``frequency_bin_mode``, cast to
        float32 *before* the slaney norm, then a second float32 rounding after it — the only
        order that reproduces those filters bit-exactly.
        """
        librosa_rounding = mel_cfg.bank_rounding == "librosa"
        if librosa_rounding:
            dtype = torch.float64
        else:
            dtype = getattr(torch, computation_dtype) if computation_dtype else None
        mel_min = hertz_to_mel(min_frequency, mel_scale=mel_cfg.mel_scale)
        mel_max = hertz_to_mel(max_frequency, mel_scale=mel_cfg.mel_scale)
        mel_freqs = torch.linspace(mel_min, mel_max, num_mel_filters + 2, dtype=dtype)
        filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_cfg.mel_scale)

        if mel_cfg.frequency_bin_mode == "rfft" and not librosa_rounding:
            fft_freqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
        else:
            fft_freqs = torch.linspace(0, sampling_rate // 2, num_frequency_bins)
        if dtype is not None:
            fft_freqs = fft_freqs.to(dtype)

        mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
        if librosa_rounding:
            mel_filters = mel_filters.to(torch.float32)

        if mel_cfg.norm == "slaney":
            enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
            mel_filters = mel_filters * enorm[None, :]
            if librosa_rounding:
                mel_filters = mel_filters.to(torch.float32)

        if mel_cfg.bands_to_zero > 0:
            mel_filters = torch.nn.functional.pad(mel_filters, (0, 0, mel_cfg.bands_to_zero, 0))
        return mel_filters

    def _cast_mel_filters_to_default_float(self, mel_filters):
        return mel_filters.to(torch.get_default_dtype())

    def _project_to_mel(self, features, *, spectrogram_config, mel_filters=None, **kwargs):
        # Match the filters to the feature dtype: unlike numpy, `torch.matmul` refuses mixed
        # dtypes, so float64 filters against float32 features would raise instead of promoting.
        mel_filters = self.mel_filters if mel_filters is None else mel_filters
        mel_filters = self._host_to_device(mel_filters.to(features.dtype), features.device)
        matmul_order = spectrogram_config.mel_scale_config.matmul_order
        if matmul_order == "features_first":
            mel_spec = torch.matmul(features.transpose(-2, -1), mel_filters)
        elif matmul_order == "filters_first_matmul":
            # legacy `mel_filters @ magnitudes`; differs from F.linear in the last ulp
            mel_spec = torch.matmul(mel_filters.T, features)
        else:
            # F.linear matches torchaudio's MelScale implementation exactly
            mel_spec = torch.nn.functional.linear(features.transpose(-2, -1), mel_filters.T).transpose(-2, -1)
        # Power/magnitude spectra and mel filters are non-negative, so a zero floor is
        # already guaranteed. Avoid dispatching a redundant pointwise CUDA kernel in the
        # common raw-mel path; nonzero floors retain their explicit clamp semantics.
        if spectrogram_config.mel_floor != 0.0:
            mel_spec = torch.clamp(mel_spec, min=spectrogram_config.mel_floor)
        return mel_spec


# ──────────────────────────────────────────────────────────────────────────────────────────────
# Op-table backends (docs/superpowers/specs/2026-10-06-audio-backend-ops-design.md).
#
# The classes above are the inheritance-era backends and still serve every model file; they are
# deleted in step 3 of the migration, when `NumpyOps` / `TorchOps` take their names. Until then the
# two generations coexist here so that each intermediate commit passes `./validate`.
# ──────────────────────────────────────────────────────────────────────────────────────────────


class AudioOps:
    """The contract an audio backend implements: array operations, nothing else.

    Every numerical stage of the pipeline lives in `BaseAudioProcessor`, written against these
    ops. A backend is a subclass implementing each *core* op below (every method of this class);
    it may also implement the names in `OPTIONAL_OPS`, which individual models require. It holds no
    model state. `NumpyOps` is the reference implementation: when this docstring and that code
    disagree, the code wins.

    Conventions:
    - `like` is an existing backend array whose *device* the new array should share (`randn` also
      takes its dtype from it). It may be `None`.
    - `dtype` is a numpy-style name: ``"float32"``, ``"float64"``, ``"int32"``, ``"int64"``,
      ``"bool"``, ``"complex64"``, ``"complex128"``. A backend resolves ``"float64"`` to its widest
      available float for the array's device (for example float32 on an Apple GPU) and
      ``"complex128"`` likewise (some backends have no complex128 at all). `dtype_name` reports the name an
      array actually has.
    - Axes follow numpy; ``axis=-1`` is the sample / frame axis. Reductions take `axis` as an int,
      a tuple of ints or `None` (all axes) and honour `keepdims`.
    - Ops return backend arrays, except `to_numpy` and `dtype_name`.
    - Binary ops (`maximum`, `matmul`, arithmetic) require operands of the *same* dtype; the base
      casts with `astype(x, dtype_name(y))` first. `masked_fill`'s mask broadcasts *to* `x`, never
      beyond it. `randn`'s `shape` is a tuple.
    - `log`, `log10`, `sqrt`, `pow`, `clamp_min` and `mean` take float input; `clamp_min`'s `value`
      may also be a 0-d array.

    Array protocol — what the base may do with a backend array besides calling ops:
    - arithmetic between two arrays of the same dtype, or between an array and a Python scalar
      *of the array's kind* (an int for int arrays, a float for float arrays, either for complex)
      on either side: ``+ - * /`` (no ``/`` on int arrays), and unary ``-``;
    - comparisons ``< <= > >= == !=`` giving bool arrays, and ``~ & |`` on bool arrays;
    - indexing with ints, slices with a positive step, `None` and `...` (views or copies; never
      rely on aliasing — torch rejects negative steps, so reversal is not part of the protocol;
      indexing away every axis may yield a scalar on numpy: keep one axis or use `keepdims=True`);
    - the attributes ``.shape`` and ``.ndim``.
    Nothing else: no item assignment (build with `masked_fill`, `concat`, `arange`), no `.dtype`
    (use `dtype_name`), no method-style reductions (use the ops).
    """

    name: str = ""

    # ── plumbing ───────────────────────────────────────────────────────────────────────────────

    def asarray(self, x, *, like=None):
        """Backend array from a numpy array, a sequence, or an existing backend array. May copy; never rely on aliasing."""
        raise NotImplementedError

    def to_numpy(self, x):
        """Host numpy array with the same values and dtype."""
        raise NotImplementedError

    def to_device(self, x, device):
        """`x` on `device` (a backend-specific designator). Backends without devices return `x`."""
        raise NotImplementedError

    def astype(self, x, dtype):
        """`x` cast to `dtype` (see the class docstring for the widest-float rule)."""
        raise NotImplementedError

    def dtype_name(self, x) -> str:
        """The numpy-style name of `x`'s dtype (``"float32"``, ``"complex64"``, ``"bool"``, ...)."""
        raise NotImplementedError

    def zeros(self, shape, *, dtype="float32", like=None):
        """Zeros of `shape` and `dtype`, on `like`'s device."""
        raise NotImplementedError

    def arange(self, n, *, like=None):
        """Integers ``0 .. n-1`` (int64), on `like`'s device."""
        raise NotImplementedError

    # ── assembly ───────────────────────────────────────────────────────────────────────────────

    def stack(self, seq):
        """Stack equally shaped arrays along a new leading axis."""
        raise NotImplementedError

    def concat(self, seq, axis=-1):
        """Concatenate along an existing axis."""
        raise NotImplementedError

    def pad(self, x, left, right, *, axis=-1, mode="constant", value=0.0):
        """Pad `axis` by `left` / `right` samples. `mode`: ``"constant"`` (with `value`),
        ``"reflect"`` (mirror without repeating the edge), ``"replicate"`` (repeat the edge).
        `reflect` requires `left` and `right` each at most `len - 1`."""
        raise NotImplementedError

    def masked_fill(self, x, mask, value):
        """`x` with `value` written where the boolean `mask` (broadcastable) is true."""
        raise NotImplementedError

    def swapaxes(self, x, axis1, axis2):
        """`x` with two axes exchanged (may be a view)."""
        raise NotImplementedError

    def contiguous(self, x):
        """Return `x` in row-major contiguous memory; a copy only if needed. Layout is observable: a
        GEMM over a strided operand accumulates in a different order, which is why a stage may
        require it."""
        raise NotImplementedError

    # ── elementwise ────────────────────────────────────────────────────────────────────────────

    def log(self, x):
        raise NotImplementedError

    def log10(self, x):
        raise NotImplementedError

    def sqrt(self, x):
        raise NotImplementedError

    def abs(self, x):
        """Magnitude; for complex input the modulus, in the matching real dtype."""
        raise NotImplementedError

    def real(self, x):
        """Real part of a complex array, in the matching real dtype."""
        raise NotImplementedError

    def imag(self, x):
        """Imaginary part of a complex array, in the matching real dtype."""
        raise NotImplementedError

    def maximum(self, x, y):
        """Elementwise maximum of two broadcastable arrays."""
        raise NotImplementedError

    def clamp_min(self, x, value):
        """Elementwise ``max(x, value)`` for a scalar `value`."""
        raise NotImplementedError

    def pow(self, x, exponent):
        """Elementwise ``x ** exponent`` for a Python scalar `exponent`. Float input."""
        raise NotImplementedError

    # ── reductions ─────────────────────────────────────────────────────────────────────────────

    def sum(self, x, axis, *, keepdims=False):
        """Sum over `axis` (int, tuple of ints, or `None` for all axes). `axis=()` is not allowed (likewise for `mean` and `amax`)."""
        raise NotImplementedError

    def mean(self, x, axis, *, keepdims=False):
        """Mean over `axis` (int, tuple of ints, or `None`; `axis=()` is not allowed). Float input only."""
        raise NotImplementedError

    def amax(self, x, axis, *, keepdims=False):
        """Maximum over `axis` (int, tuple of ints, or `None`; `axis=()` is not allowed)."""
        raise NotImplementedError

    # ── linear algebra ─────────────────────────────────────────────────────────────────────────

    def matmul(self, a, b):
        """Batched matrix product with numpy broadcasting over leading axes. Two 1-d operands may
        give a numpy scalar rather than an array; the base only uses 2-d (or higher) operands."""
        raise NotImplementedError

    # ── signal ─────────────────────────────────────────────────────────────────────────────────

    def window(self, name, length, *, periodic, dtype=None, like=None, **params):
        """Analysis window of `length` samples. `name` is one of ``"hann"``, ``"hamming"``,
        ``"boxcar"``, ``"povey"`` (Hann to the 0.85) or ``"blackman"``; `periodic` builds
        `length + 1` points and drops the last. `params` carries `blackman_coeff`, required for
        ``"blackman"``; other keys are ignored. `dtype` is the dtype name the window is *returned*
        in (only torch also evaluates in it). `dtype=None` is the one place a backend default
        differs: it means the backend's native precision for a window built for `like` -- `like`'s
        dtype on torch (what `torch.hann_window(..., dtype=audio.dtype)` did), float64 on numpy
        (what the numpy leaves did). `like` is a device hint, as for `zeros`: torch builds the
        window on `like`'s device (a change from the old CPU-then-`.to()` leaf, bit-identical on
        CPU); numpy and MLX ignore it. Each backend builds its own way, so float32 windows differ
        by ulps between backends; float64 ones agree to ~1e-16."""
        raise NotImplementedError

    def frame(self, x, frame_length, hop):
        """Overlapping frames of the last axis: ``(..., n_frames, frame_length)`` with
        ``n_frames = 1 + (len - frame_length) // hop``. May return a view.
        Read-only: callers multiply by a window into a new array and must not write into the frames.
        `x` must have at least `frame_length` samples."""
        raise NotImplementedError

    def rfft(self, x, n):
        """Real FFT of the last axis, `n` points, one-sided, unnormalised ("backward"). `n` shorter
        than the axis truncates, longer zero-pads. The output precision matches the input's
        (float32 → complex64, float64 → complex128 where the backend has it); callers wanting a
        wider FFT cast first."""
        raise NotImplementedError

    def resample(self, x, orig_sr, target_sr):
        """Resample the last axis from `orig_sr` to `target_sr` Hz with the backend's resampler,
        keeping the dtype. Sample rates are ints. Output length is ``len * target_sr // orig_sr``: resamplers disagree on
        the last sample at non-integer ratios, so every backend trims to that rule. Resamplers are
        different filters (soxr, torchaudio, scipy polyphase); they agree to ~3e-4 relative on
        band-limited audio, away from the edges."""
        raise NotImplementedError

    # ── randomness ─────────────────────────────────────────────────────────────────────────────

    def randn(self, shape, *, seed=None, like=None):
        """Standard normal samples of the tuple `shape`, in `like`'s dtype (float32 when `like` is
        `None`; `like` must be a float array) and on its device. An integer `seed` gives a private stream seeded with it; `None`
        draws from the library's global stream (so `torch.manual_seed` / `np.random.seed` apply).
        Streams differ between backends and between devices; only moments match."""
        raise NotImplementedError


CORE_OPS = tuple(name for name, value in vars(AudioOps).items() if callable(value) and not name.startswith("_"))

# Ops a backend may implement and a model may require (`required_ops`). No stub on `AudioOps`:
# their presence on a backend class means exactly "this backend implements it". The numpy
# implementation's docstring is the contract.
OPTIONAL_OPS = ("bilinear_resize",)


_AUDIO_BACKENDS: dict[str, type[AudioOps]] = {}


def missing_ops(backend_cls, names) -> list[str]:
    """Names in `names` that `backend_cls` (a backend class or instance) does not implement; a core
    op left as the contract's stub counts as missing."""
    if not isinstance(backend_cls, type):
        backend_cls = type(backend_cls)
    return [
        name
        for name in names
        if not callable(getattr(backend_cls, name, None))
        or getattr(backend_cls, name) is getattr(AudioOps, name, None)
    ]


def register_audio_backend(name: str, backend_cls: type[AudioOps]) -> None:
    """Make `backend_cls` loadable as ``backend=name``. Refused unless every core op is implemented."""
    if name in _AUDIO_BACKENDS:
        raise ValueError(f"Audio backend {name!r} is already registered.")
    missing = missing_ops(backend_cls, CORE_OPS)
    if missing:
        raise TypeError(
            f"Cannot register audio backend {name!r}: {backend_cls.__name__} is missing core ops {missing}."
        )
    own_name = vars(backend_cls).get("name")
    if own_name and own_name != name:
        raise ValueError(f"{backend_cls.__name__} is named {own_name!r}, cannot register it as {name!r}.")
    backend_cls.name = name
    _AUDIO_BACKENDS[name] = backend_cls


def get_audio_backend(name: str) -> type[AudioOps]:
    try:
        return _AUDIO_BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown audio backend {name!r}. Registered backends: {sorted(_AUDIO_BACKENDS)}. An outside "
            "backend registers itself when its module is imported (e.g. `import mlx_audio.transformers_backend`)."
        ) from None


def register_audio_op(backend: str, name: str):
    """Attach `fn` as op `name` on the registered backend `backend`, for code outside the library
    (remote-code processors, outside backends adding an optional op). In-library ops are plain
    methods in this module. Refuses to replace an existing op.

    The function receives no `self`: it is attached as a staticmethod, so it cannot call the
    backend's other ops. An op attached to a shipped class is inherited by every subclass of it."""

    def decorator(fn):
        backend_cls = get_audio_backend(backend)
        if name.startswith("_"):
            raise ValueError(f"{name!r} is not a valid op name.")
        if hasattr(backend_cls, name):
            raise ValueError(f"Op {name!r} already exists on audio backend {backend!r}.")
        setattr(backend_cls, name, staticmethod(fn))
        return fn

    return decorator


_NUMPY_PAD_MODES = {"constant": "constant", "reflect": "reflect", "replicate": "edge"}


def _check_axis(axis):
    """Shared by the three reductions of every backend: `axis=()` reduces nothing in numpy but
    everything in torch; neither is what a caller means, so every backend rejects it."""
    if axis == ():
        raise ValueError("axis=() is not allowed; pass an int, a tuple of ints or None.")


class NumpyOps(AudioOps):
    """The reference backend: every op, in numpy. No devices; `like` only matters for dtype."""

    name = "numpy"

    # ── plumbing ───────────────────────────────────────────────────────────────────────────────

    def asarray(self, x, *, like=None):
        return x if isinstance(x, np.ndarray) else np.asarray(x)

    def to_numpy(self, x):
        return np.asarray(x)

    def to_device(self, x, device):
        return x

    def astype(self, x, dtype):
        return x.astype(np.dtype(dtype), copy=False)

    def dtype_name(self, x):
        return x.dtype.name

    def zeros(self, shape, *, dtype="float32", like=None):
        return np.zeros(shape, dtype=np.dtype(dtype))

    def arange(self, n, *, like=None):
        return np.arange(n, dtype=np.int64)

    # ── assembly ───────────────────────────────────────────────────────────────────────────────

    def stack(self, seq):
        return np.stack(list(seq))

    def concat(self, seq, axis=-1):
        return np.concatenate(list(seq), axis=axis)

    def pad(self, x, left, right, *, axis=-1, mode="constant", value=0.0):
        width = [(0, 0)] * x.ndim
        width[axis] = (left, right)
        if mode == "constant":
            return np.pad(x, width, mode="constant", constant_values=value)
        if mode not in _NUMPY_PAD_MODES:
            raise ValueError(f"Unknown pad mode {mode!r}.")
        return np.pad(x, width, mode=_NUMPY_PAD_MODES[mode])

    def masked_fill(self, x, mask, value):
        return np.where(mask, np.asarray(value, dtype=x.dtype), x)

    def swapaxes(self, x, axis1, axis2):
        return np.swapaxes(x, axis1, axis2)

    def contiguous(self, x):
        return np.ascontiguousarray(x)

    # ── elementwise ────────────────────────────────────────────────────────────────────────────
    #
    # Every result goes through `np.asarray`: a ufunc on 0-d input returns a numpy *scalar*
    # (`np.float32`), without the `.shape` / `.ndim` semantics the protocol promises; the wrap
    # guarantees an ndarray. It does not fix promotion: on numpy 1.x a 0-d array still promotes
    # against a Python float (float32 * 2.0 → float64), which is why the base keeps a reduced axis
    # (`keepdims=True`) wherever a reduction feeds arithmetic.

    def log(self, x):
        return np.asarray(np.log(x))

    def log10(self, x):
        return np.asarray(np.log10(x))

    def sqrt(self, x):
        return np.asarray(np.sqrt(x))

    def abs(self, x):
        return np.asarray(np.abs(x))

    def real(self, x):
        return np.asarray(np.real(x))

    def imag(self, x):
        return np.asarray(np.imag(x))

    def maximum(self, x, y):
        return np.asarray(np.maximum(x, y))

    def clamp_min(self, x, value):
        return np.asarray(np.maximum(x, np.asarray(value, dtype=x.dtype)))

    def pow(self, x, exponent):
        # the operator, not `np.power`: numpy fast-paths a scalar 2 to `square` and 0.5 to `sqrt`, as
        # the legacy extractors' `** power` did, and the ufunc does not
        return np.asarray(x**exponent)

    # ── reductions ─────────────────────────────────────────────────────────────────────────────

    def sum(self, x, axis, *, keepdims=False):
        _check_axis(axis)
        return np.asarray(np.sum(x, axis=axis, keepdims=keepdims))

    def mean(self, x, axis, *, keepdims=False):
        _check_axis(axis)
        return np.asarray(np.mean(x, axis=axis, keepdims=keepdims))

    def amax(self, x, axis, *, keepdims=False):
        _check_axis(axis)
        return np.asarray(np.amax(x, axis=axis, keepdims=keepdims))

    # ── linear algebra ─────────────────────────────────────────────────────────────────────────

    def matmul(self, a, b):
        return np.matmul(a, b)

    # ── signal ─────────────────────────────────────────────────────────────────────────────────

    def window(self, name, length, *, periodic, dtype=None, like=None, **params):
        count = length + 1 if periodic else length
        fac = np.linspace(-np.pi, np.pi, count)
        if name == "hann":
            window = 0.5 + 0.5 * np.cos(fac)
        elif name == "hamming":
            window = 0.54 + 0.46 * np.cos(fac)
        elif name == "boxcar":
            window = np.ones(count)
        elif name == "povey":
            window = (0.5 + 0.5 * np.cos(fac)) ** 0.85
        elif name == "blackman":
            coeff = params["blackman_coeff"]
            indices = np.arange(count)
            angle = 2 * np.pi / (count - 1)
            window = coeff - 0.5 * np.cos(angle * indices) + (0.5 - coeff) * np.cos(2 * angle * indices)
        else:
            raise ValueError(f"Unknown window function {name!r}.")
        window = window[:length] if periodic else window
        return window.astype(np.dtype(dtype), copy=False) if dtype else window

    def frame(self, x, frame_length, hop):
        return np.lib.stride_tricks.sliding_window_view(x, frame_length, axis=-1)[..., ::hop, :]

    def rfft(self, x, n):
        # The contract keeps the input's precision; numpy 1.x computes a float32 FFT in float64 and
        # returns complex128, numpy 2 returns complex64 directly. The reference pins the *dtype*, and
        # values within the conformance tolerance.
        out = np.fft.rfft(x, n=n)
        return out.astype(np.complex64, copy=False) if x.dtype == np.float32 else out

    def resample(self, x, orig_sr, target_sr):
        requires_backends(self.resample, ["soxr"])
        import soxr

        if orig_sr == target_sr:
            return x
        # soxr wants (samples, channels); flatten leading axes to channels and restore them.
        leading = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1]).T
        out = soxr.resample(np.ascontiguousarray(flat), orig_sr, target_sr, quality="HQ").T
        out = out.reshape(*leading, out.shape[-1])
        return out[..., : x.shape[-1] * target_sr // orig_sr]  # the contract's length rule

    # ── randomness ─────────────────────────────────────────────────────────────────────────────

    def randn(self, shape, *, seed=None, like=None):
        dtype = like.dtype if like is not None else np.float32
        stream = np.random if seed is None else np.random.RandomState(seed)
        return stream.standard_normal(shape).astype(dtype, copy=False)

    # ── optional ops ───────────────────────────────────────────────────────────────────────────

    def bilinear_resize(self, x, size):
        """Bilinear resize of a 2-D ``(rows, cols)`` array to `size`, ``align_corners=False``
        (PyTorch / OpenCV convention): output pixel centres map to ``(i + 0.5) * in/out - 0.5``,
        clamped at 0, interpolating between the two nearest source rows and columns. Coordinates
        are computed in float32, as `torch.nn.functional.interpolate` does. Required by CLAP's
        fusion views (the shrink of a whole-clip mel to the encoder's frame count)."""
        out_rows, out_cols = size
        in_rows, in_cols = x.shape

        def coords(n_in, n_out):
            scale = np.float32(n_in) / np.float32(n_out)
            src = np.maximum((np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5), 0)
            i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
            i1 = np.minimum(i0 + 1, n_in - 1)
            return i0, i1, (src - i0).astype(x.dtype)

        r0, r1, wr = coords(in_rows, out_rows)
        c0, c1, wc = coords(in_cols, out_cols)
        top = x[r0][:, c0] * (1 - wc) + x[r0][:, c1] * wc
        bottom = x[r1][:, c0] * (1 - wc) + x[r1][:, c1] * wc
        return top * (1 - wr)[:, None] + bottom * wr[:, None]


register_audio_backend("numpy", NumpyOps)


if is_torch_available():

    class TorchOps(AudioOps):
        """Every op in torch. Tensors live wherever `like` / the input lives; `"float64"` resolves
        to float32 on MPS, the library's convention (`torch.float64 if device.type != "mps" else
        torch.float32`, as in ViT-MAE, RT-DETR and Whisper generation), applied once, here."""

        name = "torch"

        _DTYPES = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
            "complex64": torch.complex64,
            "complex128": torch.complex128,
        }
        _WIDEST_ON_MPS = {"float64": "float32", "complex128": "complex64"}

        @staticmethod
        def _resolve_dtype(dtype: str, device_type: str) -> str:
            if device_type == "mps":
                return TorchOps._WIDEST_ON_MPS.get(dtype, dtype)
            return dtype

        @classmethod
        def _torch_dtype(cls, dtype: str, device_type: str):
            name = cls._resolve_dtype(dtype, device_type)
            if name not in cls._DTYPES:
                raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(cls._DTYPES)}.")
            return cls._DTYPES[name]

        @staticmethod
        def _device_of(like):
            return like.device if isinstance(like, torch.Tensor) else torch.device("cpu")

        # ── plumbing ───────────────────────────────────────────────────────────────────────────

        def asarray(self, x, *, like=None):
            if isinstance(x, torch.Tensor):
                result = x
            elif isinstance(x, np.ndarray):
                # `from_numpy` cannot wrap read-only or negative-stride arrays: this copies read-only
                # or negative-stride input (no warning); everything else is shared without a copy.
                if x.flags.writeable and all(stride >= 0 for stride in x.strides):
                    result = torch.from_numpy(x)
                else:
                    result = torch.from_numpy(np.array(x))
            else:
                # Sequences (`list[list[float]]`) are a documented input type; go through numpy so
                # the torch backend accepts everything the numpy one does.
                result = torch.from_numpy(np.asarray(x))
            return self.to_device(result, like.device) if isinstance(like, torch.Tensor) else result

        def to_numpy(self, x):
            return x.detach().cpu().numpy()

        def to_device(self, x, device):
            device = torch.device(device)
            # The widest-float rule lives in `_resolve_dtype`; this is the transfer side of it.
            if device.type == "mps" and x.dtype in (torch.float64, torch.complex128):
                x = x.to(torch.float32 if x.dtype == torch.float64 else torch.complex64)
            if x.device == device:
                return x
            if device.type == "cuda" and x.device.type == "cpu":
                # Pinned staging makes the copy asynchronous, so preprocessing overlaps a model
                # forward still running on the stream.
                return x.pin_memory().to(device, non_blocking=True)
            return x.to(device)

        def astype(self, x, dtype):
            return x.to(self._torch_dtype(dtype, x.device.type))

        def dtype_name(self, x):
            return str(x.dtype).removeprefix("torch.")

        def zeros(self, shape, *, dtype="float32", like=None):
            device = self._device_of(like)
            return torch.zeros(shape, dtype=self._torch_dtype(dtype, device.type), device=device)

        def arange(self, n, *, like=None):
            return torch.arange(n, dtype=torch.int64, device=self._device_of(like))

        # ── assembly ───────────────────────────────────────────────────────────────────────────

        def stack(self, seq):
            return torch.stack(list(seq))

        def concat(self, seq, axis=-1):
            return torch.cat(list(seq), dim=axis)

        def pad(self, x, left, right, *, axis=-1, mode="constant", value=0.0):
            axis = axis % x.ndim
            if mode == "constant":
                spec = [0, 0] * (x.ndim - 1 - axis) + [left, right]
                return torch.nn.functional.pad(x, spec, "constant", value)
            if mode not in ("reflect", "replicate"):
                raise ValueError(f"Unknown pad mode {mode!r}.")
            # `reflect` / `replicate` pad only the last axis of a (batch, channel, length) tensor.
            # For a non-last axis this copies once in and once out and returns a non-contiguous tensor.
            moved = x.movedim(axis, -1)
            flat = moved.reshape(1, -1, moved.shape[-1])
            padded = torch.nn.functional.pad(flat, (left, right), mode=mode)
            return padded.reshape(*moved.shape[:-1], padded.shape[-1]).movedim(-1, axis)

        def masked_fill(self, x, mask, value):
            return x.masked_fill(mask, value)

        def swapaxes(self, x, axis1, axis2):
            return x.transpose(axis1, axis2)

        def contiguous(self, x):
            return x.contiguous()

        # ── elementwise ────────────────────────────────────────────────────────────────────────

        def log(self, x):
            return torch.log(x)

        def log10(self, x):
            return torch.log10(x)

        def sqrt(self, x):
            return torch.sqrt(x)

        def abs(self, x):
            return torch.abs(x)

        def real(self, x):
            return torch.real(x)

        def imag(self, x):
            return torch.imag(x)

        def maximum(self, x, y):
            return torch.maximum(x, y)

        def clamp_min(self, x, value):
            return torch.clamp(x, min=value)

        def pow(self, x, exponent):
            return torch.pow(x, exponent)

        # ── reductions ─────────────────────────────────────────────────────────────────────────

        @staticmethod
        def _dims(x, axis):
            return tuple(range(x.ndim)) if axis is None else axis

        def sum(self, x, axis, *, keepdims=False):
            _check_axis(axis)
            return torch.sum(x, dim=self._dims(x, axis), keepdim=keepdims)  # bool sums to int64, as numpy

        def mean(self, x, axis, *, keepdims=False):
            _check_axis(axis)
            return torch.mean(x, dim=self._dims(x, axis), keepdim=keepdims)

        def amax(self, x, axis, *, keepdims=False):
            _check_axis(axis)
            return torch.amax(x, dim=self._dims(x, axis), keepdim=keepdims)

        # ── linear algebra ─────────────────────────────────────────────────────────────────────

        def matmul(self, a, b):
            return torch.matmul(a, b)

        # ── signal ─────────────────────────────────────────────────────────────────────────────

        def window(self, name, length, *, periodic, dtype=None, like=None, **params):
            device = self._device_of(like)
            if dtype is not None:
                torch_dtype = self._torch_dtype(dtype, device.type)
            else:
                torch_dtype = like.dtype if isinstance(like, torch.Tensor) else torch.float32
            kwargs = {"dtype": torch_dtype, "device": device}
            if name == "hann":
                return torch.hann_window(length, periodic=periodic, **kwargs)
            if name == "hamming":
                return torch.hamming_window(length, periodic=periodic, **kwargs)
            if name == "boxcar":
                return torch.ones(length, **kwargs)
            if name == "povey":
                return torch.hann_window(length, periodic=periodic, **kwargs).pow(0.85)
            if name == "blackman":
                coeff = params["blackman_coeff"]
                angle = 2 * math.pi / (length if periodic else length - 1)
                indices = torch.arange(length, **kwargs)
                return coeff - 0.5 * torch.cos(angle * indices) + (0.5 - coeff) * torch.cos(2 * angle * indices)
            raise ValueError(f"Unknown window function {name!r}.")

        def frame(self, x, frame_length, hop):
            # A writable overlapping view; by contract callers never write into frames (numpy's view is read-only).
            return x.unfold(-1, frame_length, hop)

        def rfft(self, x, n):
            return torch.fft.rfft(x, n=n)

        def resample(self, x, orig_sr, target_sr):
            requires_backends(self.resample, ["torchaudio"])
            import torchaudio

            out = torchaudio.functional.resample(x, orig_freq=orig_sr, new_freq=target_sr)
            return out[..., : x.shape[-1] * target_sr // orig_sr]  # the contract's length rule

        # ── randomness ─────────────────────────────────────────────────────────────────────────

        def randn(self, shape, *, seed=None, like=None):
            # Streams differ by device as well as by backend: the same seed gives different numbers on CPU and CUDA.
            device = self._device_of(like)
            dtype = like.dtype if isinstance(like, torch.Tensor) else torch.float32
            if seed is None:
                return torch.randn(shape, dtype=dtype, device=device)  # the global stream (`torch.manual_seed`)
            generator = torch.Generator(device=device).manual_seed(seed)
            return torch.randn(shape, dtype=dtype, device=device, generator=generator)

        # ── optional ops ───────────────────────────────────────────────────────────────────────

        def bilinear_resize(self, x, size):
            resized = torch.nn.functional.interpolate(
                x[None, None], size=list(size), mode="bilinear", align_corners=False
            )
            return resized[0, 0]

    register_audio_backend("torch", TorchOps)
