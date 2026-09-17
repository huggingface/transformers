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
from .utils import (
    TensorType,
    is_kernels_available,
    is_speech_available,
    is_torch_available,
    logging,
    requires_backends,
)


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

    def _astype(self, x, dtype_name):
        return x.astype(np.dtype(dtype_name))

    def _amax_over_features(self, x):
        if x.ndim > 2:
            return x.max(axis=tuple(range(1, x.ndim)), keepdims=True)
        return x.max()

    def _zeros_int32(self, shape, *, like=None):
        return np.zeros(shape, dtype=np.int32)

    def _as_backend_array(self, x, *, like=None):
        return x if isinstance(x, np.ndarray) else np.asarray(x)

    def _arange(self, stop, *, like=None):
        return np.arange(stop)

    def _mean_axis0(self, x):
        return x.mean(axis=0)

    def _squeeze_axis0(self, x):
        return np.squeeze(x, axis=0)

    def _resample(self, audio, orig_sampling_rate: int, target_sampling_rate: int):
        requires_backends(self._resample, ["soxr"])
        import soxr

        return soxr.resample(audio, orig_sampling_rate, target_sampling_rate, quality="HQ")

    def _pad_axis(self, x, left, right, axis, value=0.0):
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (left, right)
        return np.pad(x, pad_width, mode="constant", constant_values=value)

    def _stack(self, seq):
        return np.stack(seq)

    def _insert_channel_dim(self, batch):
        return batch[:, np.newaxis, :]

    def _mean_last(self, x):
        return x.mean(axis=-1, keepdims=True)

    def _concat_last(self, parts):
        return np.concatenate(parts, axis=-1)

    # ── STFT pipeline ─────────────────────────────────────────────────────

    def _create_stft_window(self, win_length, stft_cfg, audio):
        if stft_cfg.window_fn == "hann_window_f32":
            # fixed USM float32 periodic Hann (bit-exact with the legacy Gemma3n extractor, which
            # builds it inline from a float32 arange -- under numpy scalar promotion the whole
            # cosine is then evaluated in single precision); ignores
            # `periodic`/`window_dtype`/`wkwargs`
            arange = np.arange(win_length, dtype=np.float32)
            return (0.5 * (1 - np.cos(2 * np.pi * arange / win_length))).astype(np.float32)
        if stft_cfg.window_fn == "hann_window_f64_as_f32":
            # Periodic Hann evaluated in float64 and stored as float32 -- what the shared
            # `window_function` util produces for a legacy extractor that does
            # `window_function(n).astype(np.float32)` (Gemma4). Distinct from
            # `hann_window_f32`, which evaluates the cosine itself in float32; the two differ
            # by ~2.4e-07 and that is enough to break bit-exact parity.
            return np.hanning(win_length + 1)[:-1].astype(np.float32)
        wkwargs = dict(stft_cfg.wkwargs or {})
        unsupported = set(wkwargs) - {"requires_grad"}
        if unsupported:
            raise ValueError(
                f"Unsupported window kwargs for the NumPy backend: {sorted(unsupported)}. "
                "Use the dedicated periodic and window_dtype fields for portable configuration."
            )
        if wkwargs.get("requires_grad"):
            raise ValueError("requires_grad=True is only meaningful for the Torch backend.")

        N = win_length + 1 if stft_cfg.periodic else win_length
        fac = np.linspace(-np.pi, np.pi, N)
        name = stft_cfg.window_fn
        if name in ("hann", "hann_window"):
            w = 0.5 + 0.5 * np.cos(fac)
        elif name in ("hamming", "hamming_window"):
            w = 0.54 + 0.46 * np.cos(fac)
        elif name == "boxcar":
            w = np.ones(N)
        elif name == "povey":
            w = (0.5 + 0.5 * np.cos(fac)) ** 0.85
        elif name in ("blackman", "blackman_window"):
            coeff = stft_cfg.blackman_coeff
            indices = np.arange(N)
            angle = 2 * np.pi / (N - 1)
            w = coeff - 0.5 * np.cos(angle * indices) + (0.5 - coeff) * np.cos(2 * angle * indices)
        else:
            raise ValueError(f"Unknown window function '{name}'")
        w = w[:win_length] if stft_cfg.periodic else w
        # Honour `window_dtype` as the torch leaf does. Without this the cosine's float64 result
        # was returned unconditionally, and `_frame_waveform`'s `result_type` promotion then
        # carried the whole framing/preemphasis path into float64 while torch stayed in float32.
        # Honour `window_dtype` as the torch leaf does. Left unread, the cosine's float64 result
        # was returned unconditionally and `_frame_waveform`'s `result_type` promotion carried
        # framing and preemphasis into float64 while torch stayed in float32 -- the same config
        # computed at two different precisions. The *fallback* deliberately stays float64 rather
        # than following the audio dtype as torch does: these numpy leaves are bit-exact against
        # the numpy legacy extractors, and narrowing it moves qwen3_asr and voxtral_realtime.
        return w.astype(np.dtype(stft_cfg.window_dtype), copy=False) if stft_cfg.window_dtype else w

    @staticmethod
    def _np_frame(x, frame_length, hop_length):
        """Create overlapping frames using stride tricks (replaces librosa.util.frame)."""
        n_frames = 1 + (x.shape[-1] - frame_length) // hop_length
        strides = x.strides[:-1] + (x.strides[-1] * hop_length, x.strides[-1])
        shape = x.shape[:-1] + (n_frames, frame_length)
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    def _frame_waveform(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        if stft_cfg.center == "left":
            # semicausal (USM/Gemma): zeros prepended only
            audio = self._pad_axis(audio, stft_cfg.win_length // 2, 0, axis=-1)
        elif stft_cfg.center:
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(frame_length // 2, frame_length // 2)]
            audio = np.pad(audio, pad_width, mode=stft_cfg.pad_mode)
        frames = self._np_frame(np.ascontiguousarray(audio), frame_length, hop_length)
        compute_dtype = np.result_type(audio.dtype, window.dtype)
        return frames.astype(compute_dtype, copy=False)

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

    def _stft_framed(self, frames, window, frame_length, n_fft, stft_cfg, audio_dtype=None):
        frames = frames * window
        fft = np.fft.rfft if stft_cfg.onesided else np.fft.fft
        spec = fft(frames, n=n_fft, axis=-1)
        if stft_cfg.fft_dtype in (None, "complex64"):
            # librosa contract: FFT output rounded through complex64
            spec = spec.astype(np.complex64)
        if stft_cfg.normalized in (True, "window"):
            spec = spec / np.sqrt(np.sum(window**2)).astype(spec.real.dtype)
        elif stft_cfg.normalized == "frame_length":
            spec = spec / np.sqrt(n_fft).astype(spec.real.dtype)
        elif stft_cfg.normalized is not False:
            raise ValueError(f"Invalid normalized value: {stft_cfg.normalized!r}")
        return np.moveaxis(spec, -1, -2)

    def _stft_native(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        # No numpy-native STFT exists; compose the manual framing + FFT leaves. This path
        # receives the center-padded window and frame_length == n_fft from
        # `_prepare_window_and_framing`, unlike the manual path (left-aligned window).
        # `fft_dtype` can't leak in here: `_waveform_to_spectrum` rejects it on native-STFT configurations.
        frames = self._frame_waveform(audio, window, frame_length, hop_length, n_fft, stft_cfg)
        return self._stft_framed(frames, window, frame_length, n_fft, stft_cfg)

    def _spectrum_magnitude(self, stft_out, power, spectrogram_config=None, **kwargs):
        # `computation_dtype` names the dtype the upstream FE took magnitudes in, as it does in the
        # torch leaf and in the mel-filter leaves below. It was previously read as a flag -- any
        # truthy value meant float64 -- so a config asking for float32 silently got float64 here
        # while torch honoured it, one field with two meanings.
        dtype = (
            np.dtype(spectrogram_config.computation_dtype)
            if spectrogram_config and spectrogram_config.computation_dtype
            else None
        )
        if spectrogram_config and spectrogram_config.stft_config.magnitude_mode == "sqrt_sum_squares":
            real = np.real(stft_out).astype(dtype, copy=False) if dtype else np.real(stft_out)
            imag = np.imag(stft_out).astype(dtype, copy=False) if dtype else np.imag(stft_out)
            magnitudes = np.sqrt(real**2 + imag**2)
            return magnitudes**power if power != 1.0 else magnitudes
        if dtype:
            return np.abs(stft_out, dtype=dtype) ** power
        return np.abs(stft_out) ** power

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

    # ── Kaldi fbank helper ────────────────────────────────────────────────

    def _kaldi_fbank(self, waveform, num_mel_bins, sample_frequency=None, **kwargs):
        """Extract kaldi-compatible fbank features using torchaudio (or fallback to base pipeline).

        Returns numpy array of shape (time, num_mel_bins).
        """
        if sample_frequency is None:
            sample_frequency = self.sampling_rate

        if is_speech_available():
            import torchaudio.compliance.kaldi as ta_kaldi

            waveform_tensor = torch.from_numpy(np.asarray(waveform)).unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform_tensor, num_mel_bins=num_mel_bins, sample_frequency=sample_frequency, **kwargs
            )
            return fbank.numpy()

        waveform = np.squeeze(waveform)
        features = self.spectrogram(waveform, spectrogram_config=self.spectrogram_config, dither=self.dither)
        return features.T


class TorchAudioBackend(BaseAudioProcessor):
    """Torch backend for audio processing."""

    def __init__(self, *args, **kwargs: Unpack[AudioKwargs]):
        super().__init__(*args, **kwargs)
        self._set_attributes(**kwargs)

    @property
    def backend(self) -> str:
        return "torch"

    # ── Backend array-API primitives ─────────────────────────────────────

    def _astype(self, x, dtype_name):
        return x.to(getattr(torch, dtype_name))

    def _prepare_waveform(self, audio_el, *, device=None, **kwargs):
        return audio_el.to(device=device) if device is not None else audio_el

    def _fused_cuda_incompatibility(
        self,
        audio,
        *,
        spectrogram_config,
        audio_ranges,
        padding_side,
        padding_value,
        return_tensors,
        dither,
        **kwargs,
    ):
        if not is_kernels_available():
            return "the `kernels` package is not installed or has an incompatible version"
        if self.feature_normalization not in (None, "per_feature_standardize"):
            return "feature_normalization must be None or 'per_feature_standardize'"

        config = spectrogram_config
        stft = config.stft_config
        mel = config.mel_scale_config
        numerical_hooks = (
            "_compute_spectrum",
            "_waveform_to_spectrum",
            "_spectrum_magnitude",
            "_project_to_mel",
            "_log_compress",
            "_shape_log_features",
            "_process_frames",
            "_stft_framed",
        )
        overridden_hooks = [
            name for name in numerical_hooks if getattr(type(self), name) is not getattr(TorchAudioBackend, name)
        ]
        unsupported = {
            "stft.pad": stft.pad != 0,
            "single_item_manual_framing": self._needs_manual_framing(config) and audio.shape[0] == 1,
            "stft.center": stft.center not in (False, True, "left"),
            "stft.pad_mode": stft.center is True and stft.pad_mode not in ("constant", "reflect"),
            "stft.power": stft.power not in (1.0, 2.0),
            "stft.normalized": stft.normalized is not False,
            "stft.onesided": not stft.onesided,
            "stft.extra_samples_per_frame": stft.extra_samples_per_frame not in (0, 1),
            "stft.fft_dtype": stft.fft_dtype not in (None, "float64", "native", "complex64"),
            "stft.window_dtype": stft.window_dtype is not None,
            "mel_scale_config": mel is None,
            "mel.computation_dtype": mel is not None and mel.computation_dtype is not None,
            "mel_filters": kwargs.get("mel_filters") is not None,
            "preemphasis_mode": config.preemphasis is not None
            and config.preemphasis_mode not in ("waveform", "per_frame", "htk_per_frame"),
            "mel_floor": config.mel_floor < 0.0,
            "log_mode": config.log_mode not in (None, "log", "log10", "dB"),
            "computation_dtype": config.computation_dtype not in (None, "float64"),
            "subtract_mean": config.subtract_mean,
            "combined_peak_rescaling": config.floor_below_peak is not None
            and (config.log_shift is not None or config.log_scale is not None),
            "log_reference": kwargs.get("reference", 1.0) != 1.0,
            "db_range": kwargs.get("db_range") is not None,
            "numerical_hooks": bool(overridden_hooks),
            "dither": dither < 0.0,
            "padding_side": padding_side != "right",
            "padding_value": padding_value != 0.0,
            "return_tensors": return_tensors not in (None, TensorType.PYTORCH),
            "audio_ranges": any(start != 0 for start, _ in audio_ranges),
            "audio": audio.device.type != "cuda" or audio.dtype != torch.float32 or not audio.is_contiguous(),
        }
        incompatible = [name for name, is_unsupported in unsupported.items() if is_unsupported]
        return f"incompatible setting(s): {', '.join(incompatible)}" if incompatible else None

    @staticmethod
    def _load_fused_cuda_kernel(*, required):
        from .integrations.hub_kernels import lazy_load_kernel

        try:
            kernel = lazy_load_kernel("parakeet-audio")
        except Exception as error:
            if required:
                raise ImportError("Failed to load `kernels-community/parakeet-audio` version 1.") from error
            logger.warning_once(
                "Could not load the fused CUDA audio kernel from the Hub; falling back to the eager Torch path. "
                f"Loader error: {error}"
            )
            return None
        if kernel is not None and not hasattr(kernel, "fused_log_mel"):
            if required:
                raise ImportError(
                    "The loaded `kernels-community/parakeet-audio` build does not expose `fused_log_mel`."
                )
            logger.warning_once(
                "The loaded fused CUDA audio kernel has an incompatible API; falling back to the eager Torch path."
            )
            return None
        if kernel is None and required:
            raise ImportError(
                "`use_fused_cuda=True` requires a compatible `kernels-community/parakeet-audio` version 1 build."
            )
        return kernel

    def _run_fused_cuda_kernel(self, kernel, audio, *, audio_ranges, spectrogram_config, standardize, **kwargs):
        lengths = torch.tensor([end - start for start, end in audio_ranges], dtype=torch.int64, device=audio.device)
        compute_dtype = (
            torch.float64
            if spectrogram_config.computation_dtype == "float64"
            or spectrogram_config.stft_config.fft_dtype in ("float64", "native")
            else audio.dtype
        )
        cache_key = (audio.device, compute_dtype, spectrogram_config)
        if getattr(self, "_cached_audio_kernel_inputs", (None,))[0] != cache_key:
            stft = spectrogram_config.stft_config
            needs_manual_framing = self._needs_manual_framing(spectrogram_config)
            compute_audio = audio.to(compute_dtype)
            window = self._create_stft_window(stft.win_length, stft, compute_audio)
            window, _ = self._prepare_window_and_framing(
                window,
                stft.win_length,
                stft.n_fft,
                needs_manual_framing=needs_manual_framing,
            )
            mel_filters = self._mel_filter_bank(spectrogram_config).to(device=audio.device, dtype=compute_dtype)
            self._cached_audio_kernel_inputs = (cache_key, window.contiguous(), mel_filters.contiguous())
        _, window, mel_filters = self._cached_audio_kernel_inputs

        dither = kwargs.get("dither", self.dither)
        kernel_audio = self._dither_waveform(audio, audio_ranges, dither=dither) if dither else audio
        if spectrogram_config.waveform_scale is not None:
            kernel_audio = kernel_audio * spectrogram_config.waveform_scale
        kernel_audio = kernel_audio.to(compute_dtype)
        mel_order = spectrogram_config.mel_scale_config.matmul_order
        output_time_major = (mel_order == "features_first") != spectrogram_config.transpose_features
        log_mode = spectrogram_config.log_mode
        mel_floor = spectrogram_config.mel_floor
        if log_mode == "dB":
            mel_floor = max(mel_floor, kwargs.get("min_value", 1e-10))
        features, feature_lengths = kernel.fused_log_mel(
            kernel_audio.contiguous(),
            lengths,
            window,
            mel_filters,
            hop_length=spectrogram_config.stft_config.hop_length,
            preemphasis=spectrogram_config.preemphasis or 0.0,
            log_offset=spectrogram_config.pre_log_offset or 0.0,
            standardize=standardize,
            normalization_eps=self.feature_normalization_eps,
            center=spectrogram_config.stft_config.center is True,
            reflect_padding=spectrogram_config.stft_config.pad_mode == "reflect",
            power=spectrogram_config.stft_config.power,
            log_mode=log_mode,
            mel_floor=mel_floor,
            log_multiplier=(10.0 if spectrogram_config.stft_config.power == 2.0 else 20.0)
            if log_mode == "dB"
            else 1.0,
            drop_last_frame=spectrogram_config.drop_last_frame,
            output_time_major=output_time_major,
            floor_below_peak=spectrogram_config.floor_below_peak,
            log_shift=spectrogram_config.log_shift,
            log_scale=spectrogram_config.log_scale,
            frame_length=(
                spectrogram_config.stft_config.win_length + spectrogram_config.stft_config.extra_samples_per_frame
                if self._needs_manual_framing(spectrogram_config)
                else spectrogram_config.stft_config.n_fft
            ),
            window_length=(
                spectrogram_config.stft_config.win_length
                if self._needs_manual_framing(spectrogram_config)
                else spectrogram_config.stft_config.n_fft
            ),
            center_left=spectrogram_config.stft_config.center == "left",
            remove_dc_offset=spectrogram_config.remove_dc_offset,
            preemphasis_mode=spectrogram_config.preemphasis_mode,
            extra_samples_per_frame=spectrogram_config.stft_config.extra_samples_per_frame,
            matmul_order=mel_order,
        )
        if not standardize:
            audio_lengths = np.asarray([end - start for start, end in audio_ranges])
            feature_lengths = torch.as_tensor(
                self._valid_frame_counts(audio_lengths, spectrogram_config), dtype=torch.int32, device=audio.device
            )
        return features, feature_lengths

    def spectrogram(self, audio, *, spectrogram_config, use_fused_cuda=None, **kwargs):
        if use_fused_cuda is False:
            return super().spectrogram(
                audio, spectrogram_config=spectrogram_config, use_fused_cuda=use_fused_cuda, **kwargs
            )

        if not isinstance(audio, torch.Tensor) or audio.device.type != "cuda" or audio.dtype != torch.float32:
            return super().spectrogram(
                audio, spectrogram_config=spectrogram_config, use_fused_cuda=use_fused_cuda, **kwargs
            )

        batched = audio.unsqueeze(0) if audio.ndim == 1 else audio
        audio_ranges = kwargs.get("audio_ranges")
        workflow_keys = {"audio_ranges", "padding_side", "padding_value", "return_tensors"}
        kernel_kwargs = {name: value for name, value in kwargs.items() if name not in workflow_keys}
        if audio_ranges is None:
            audio_ranges = [(0, batched.shape[-1])] * batched.shape[0]
        incompatibility = self._fused_cuda_incompatibility(
            batched,
            spectrogram_config=spectrogram_config,
            audio_ranges=audio_ranges,
            padding_side=kwargs.get("padding_side", "right"),
            padding_value=kwargs.get("padding_value", 0.0),
            return_tensors=kwargs.get("return_tensors", TensorType.PYTORCH),
            **kernel_kwargs,
        )
        if incompatibility is not None:
            if use_fused_cuda is True:
                raise ValueError(f"The fused CUDA audio kernel is unavailable because {incompatibility}.")
            return super().spectrogram(
                audio, spectrogram_config=spectrogram_config, use_fused_cuda=use_fused_cuda, **kwargs
            )
        kernel = self._load_fused_cuda_kernel(required=use_fused_cuda is True)
        if kernel is None:
            return super().spectrogram(
                audio, spectrogram_config=spectrogram_config, use_fused_cuda=use_fused_cuda, **kwargs
            )
        features, _ = self._run_fused_cuda_kernel(
            kernel,
            batched,
            audio_ranges=audio_ranges,
            spectrogram_config=spectrogram_config,
            standardize=False,
            **kernel_kwargs,
        )
        return features if audio.ndim > 1 else features[0]

    def _compute_batched_features(self, audio, *, audio_ranges, spectrogram_config, use_fused_cuda=None, **kwargs):
        if use_fused_cuda is False:
            return super()._compute_batched_features(
                audio,
                audio_ranges=audio_ranges,
                spectrogram_config=spectrogram_config,
                use_fused_cuda=use_fused_cuda,
                **kwargs,
            )

        incompatibility = self._fused_cuda_incompatibility(
            audio,
            spectrogram_config=spectrogram_config,
            audio_ranges=audio_ranges,
            **kwargs,
        )
        if incompatibility is not None:
            if use_fused_cuda is True:
                exception = ImportError if not is_kernels_available() else ValueError
                raise exception(f"The fused CUDA audio kernel is unavailable because {incompatibility}.")
            return super()._compute_batched_features(
                audio,
                audio_ranges=audio_ranges,
                spectrogram_config=spectrogram_config,
                use_fused_cuda=use_fused_cuda,
                **kwargs,
            )

        kernel = self._load_fused_cuda_kernel(required=use_fused_cuda is True)
        if kernel is None:
            return super()._compute_batched_features(
                audio,
                audio_ranges=audio_ranges,
                spectrogram_config=spectrogram_config,
                use_fused_cuda=use_fused_cuda,
                **kwargs,
            )
        return self._run_fused_cuda_kernel(
            kernel,
            audio,
            audio_ranges=audio_ranges,
            spectrogram_config=spectrogram_config,
            standardize=self.feature_normalization == "per_feature_standardize",
            **kwargs,
        )

    def _amax_over_features(self, x):
        return x.amax(dim=(-2, -1), keepdim=True)

    def _zeros_int32(self, shape, *, like=None):
        device = like.device if isinstance(like, torch.Tensor) else None
        return torch.zeros(shape, dtype=torch.int32, device=device)

    def _get_mask(self, ranges, padded_length, *, like=None):
        if not isinstance(like, torch.Tensor):
            return super()._get_mask(ranges, padded_length, like=like)
        range_tensor = torch.tensor(ranges, dtype=torch.int64, device=like.device)
        positions = torch.arange(padded_length, device=like.device)
        return ((positions >= range_tensor[:, :1]) & (positions < range_tensor[:, 1:])).to(torch.int32)

    def _get_mask_from_lengths(self, lengths, padded_length, *, like=None):
        if not isinstance(like, torch.Tensor):
            return super()._get_mask_from_lengths(lengths, padded_length, like=like)
        lengths = torch.as_tensor(lengths, dtype=torch.int64, device=like.device)
        positions = torch.arange(padded_length, device=like.device)
        return (positions < lengths[:, None]).to(torch.int32)

    def _as_backend_array(self, x, *, like=None):
        if isinstance(x, np.ndarray):
            result = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            result = x
        else:
            # Sequences (e.g. `list[list[float]]`) are a documented input type; convert through numpy
            # so the torch backend accepts everything the numpy one does.
            result = torch.from_numpy(np.asarray(x))
        return result.to(device=like.device) if isinstance(like, torch.Tensor) else result

    def _arange(self, stop, *, like=None):
        device = like.device if isinstance(like, torch.Tensor) else None
        return torch.arange(stop, device=device)

    def _mean_axis0(self, x):
        return x.mean(dim=0)

    def _squeeze_axis0(self, x):
        return x.squeeze(0)

    def _resample(self, audio, orig_sampling_rate: int, target_sampling_rate: int):
        requires_backends(self._resample, ["torchaudio"])
        import torchaudio

        return torchaudio.functional.resample(audio, orig_freq=orig_sampling_rate, new_freq=target_sampling_rate)

    def _pad_axis(self, x, left, right, axis, value=0.0):
        axis = axis % x.ndim
        pad = [0, 0] * (x.ndim - 1 - axis) + [left, right]
        return torch.nn.functional.pad(x, pad, "constant", value)

    def _stack(self, seq):
        return torch.stack(seq)

    def _insert_channel_dim(self, batch):
        return batch.unsqueeze(1)

    def _mean_last(self, x):
        return x.mean(dim=-1, keepdim=True)

    def _concat_last(self, parts):
        return torch.cat(parts, dim=-1)

    # ── STFT pipeline ─────────────────────────────────────────────────────

    def _create_stft_window(self, win_length, stft_cfg, audio):
        dtype = getattr(torch, stft_cfg.window_dtype) if stft_cfg.window_dtype else audio.dtype
        raw_wkwargs = dict(stft_cfg.wkwargs or {})
        unsupported = set(raw_wkwargs) - {"requires_grad"}
        if unsupported:
            raise ValueError(
                f"Unsupported window kwargs: {sorted(unsupported)}. "
                "Use the dedicated periodic and window_dtype fields for portable configuration."
            )
        wkwargs = {**raw_wkwargs, "dtype": dtype}
        name = stft_cfg.window_fn
        if name == "hann_window_f32":
            # numpy build + convert, so both backends' windows are bit-identical;
            # ignores `periodic`/`window_dtype`/`wkwargs`
            arange = np.arange(win_length, dtype=np.float32)
            window = torch.from_numpy((0.5 * (1 - np.cos(2 * np.pi * arange / win_length))).astype(np.float32))
            return window.to(device=audio.device)
        if name == "hann_window_f64_as_f32":
            # See the numpy leaf: periodic Hann evaluated in float64, stored as float32, matching
            # `window_function(n).astype(np.float32)`. Built through numpy so both backends'
            # windows stay bit-identical.
            window = torch.from_numpy(np.hanning(win_length + 1)[:-1].astype(np.float32))
            return window.to(device=audio.device)
        if name in ("hann", "hann_window"):
            window = torch.hann_window(win_length, periodic=stft_cfg.periodic, **wkwargs)
        elif name in ("hamming", "hamming_window"):
            window = torch.hamming_window(win_length, periodic=stft_cfg.periodic, **wkwargs)
        elif name == "boxcar":
            window = torch.ones(win_length, dtype=dtype)
        elif name == "povey":
            window = torch.hann_window(win_length, periodic=stft_cfg.periodic, **wkwargs).pow(0.85)
        elif name in ("blackman", "blackman_window"):
            denominator = win_length if stft_cfg.periodic else win_length - 1
            angle = 2 * math.pi / denominator
            indices = torch.arange(win_length, **wkwargs)
            coeff = stft_cfg.blackman_coeff
            window = coeff - 0.5 * torch.cos(angle * indices) + (0.5 - coeff) * torch.cos(2 * angle * indices)
        else:
            raise ValueError(f"Unknown window function '{name}'")
        return window.to(device=audio.device)

    def _frame_waveform(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        if stft_cfg.center == "left":
            pad_left = stft_cfg.win_length // 2
            audio = torch.nn.functional.pad(audio, (pad_left, 0), mode="constant", value=0.0)
        elif stft_cfg.center:
            audio = torch.nn.functional.pad(audio, (frame_length // 2, frame_length // 2), mode=stft_cfg.pad_mode)
        return audio.unfold(-1, frame_length, hop_length)

    def _preemphasize_waveform(self, audio, preemphasis, audio_ranges=None):
        audio = torch.cat([audio[..., :1], audio[..., 1:] - preemphasis * audio[..., :-1]], dim=-1)
        if audio_ranges is not None:
            lengths = torch.tensor([end - start for start, end in audio_ranges], device=audio.device)
            mask = torch.arange(audio.shape[-1], device=audio.device).unsqueeze(0) < lengths.unsqueeze(1)
            audio = audio.masked_fill(~mask, 0.0)
        return audio

    def _dither_waveform(self, audio, audio_ranges=None, *, dither):
        noise = torch.randn(audio.shape, dtype=audio.dtype, device=audio.device)
        return audio + dither * noise

    def _stft_framed(self, frames, window, frame_length, n_fft, stft_cfg, audio_dtype=None):
        frames = frames * window
        if stft_cfg.fft_dtype == "float64":
            frames = frames.to(torch.float64)  # mirrors numpy's rfft float64 promotion
        if frame_length < n_fft:
            frames = torch.nn.functional.pad(frames, (0, n_fft - frame_length))
        fft = torch.fft.rfft if stft_cfg.onesided else torch.fft.fft
        spec = self._round_through_complex64(fft(frames, n=n_fft), stft_cfg)
        if stft_cfg.normalized in (True, "window"):
            spec = spec / window.pow(2.0).sum().sqrt()
        elif stft_cfg.normalized == "frame_length":
            spec = spec / math.sqrt(n_fft)
        elif stft_cfg.normalized is not False:
            raise ValueError(f"Invalid normalized value: {stft_cfg.normalized!r}")
        return spec.transpose(-2, -1)

    def _stft_native(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        win_length = stft_cfg.win_length
        if audio.device.type == "cuda" and win_length < n_fft:
            # `torch.stft` performs this same pad -> unfold -> window -> FFT sequence,
            # but its generic wrapper adds material dispatch overhead on short GPU
            # workloads. Express the operations directly so CUDA reaches the cached
            # cuFFT plan without that wrapper; this is also the eager layout used by
            # native cuFFT audio frontends such as fast-gpu-asr. This specialization
            # pays off when the cached analysis window was padded to the larger FFT
            # size; full-window CUDA transforms and CPU transforms keep `torch.stft`,
            # which benchmarks faster for those cases.
            if stft_cfg.center:
                signal_dim = audio.ndim
                extended_shape = [1] * (3 - signal_dim) + list(audio.shape)
                pad = n_fft // 2
                audio = torch.nn.functional.pad(audio.view(extended_shape), (pad, pad), mode=stft_cfg.pad_mode)
                audio = audio.view(audio.shape[-signal_dim:])

            frames = audio.unfold(-1, n_fft, hop_length)
            frames = frames * window
            fft = torch.fft.rfft if stft_cfg.onesided else torch.fft.fft
            norm = "ortho" if stft_cfg.normalized == "frame_length" else "backward"
            stft_out = fft(frames, n=n_fft, norm=norm).transpose(-2, -1)
        else:
            stft_out = torch.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=frame_length,
                window=window,
                center=stft_cfg.center,
                pad_mode=stft_cfg.pad_mode,
                normalized=stft_cfg.normalized == "frame_length",
                onesided=stft_cfg.onesided,
                return_complex=True,
            )
        stft_out = self._round_through_complex64(stft_out, stft_cfg)
        if stft_cfg.normalized in (True, "window"):
            stft_out = stft_out / window.pow(2.0).sum().sqrt()
        elif stft_cfg.normalized not in (False, "frame_length"):
            raise ValueError(f"Invalid normalized value: {stft_cfg.normalized!r}")
        return stft_out

    @staticmethod
    def _round_through_complex64(spec, stft_cfg):
        # `fft_dtype="complex64"`: the FFT output is rounded through complex64 before the float64
        # magnitudes, as the legacy numpy `spectrogram()` did by writing into a complex64 buffer.
        if stft_cfg.fft_dtype == "complex64" and spec.dtype == torch.complex128:
            return spec.to(torch.complex64).to(torch.complex128)
        return spec

    def _cast_stft_output(self, magnitudes, spectrogram_config):
        if spectrogram_config.computation_dtype:
            return magnitudes
        return magnitudes.float()

    def _spectrum_magnitude(self, stft_out, power, spectrogram_config=None, **kwargs):
        # TODO(audio-processor): reinstate the contiguity fix once the perf/parity trade-off is
        # decided. `torch.stft` returns a non-contiguous tensor and `abs() ** power` keeps that
        # layout, so the downstream `mel_filters.T @ magnitudes` matmul can fall onto a slow
        # strided-GEMM path (~8x slower on a ROCm build). Upstream fixed that in the legacy
        # `WhisperFeatureExtractor` (#47351) and it was ported here in 55db94210b, adding
        # `.contiguous()` to both returns below.
        #
        # It was reverted because it is NOT value-neutral, contrary to what that commit claimed.
        # Contiguity preserves the values exactly but changes which GEMM kernel and blocking the
        # mel projection selects, which changes accumulation order and shifts float32 results by
        # ~1 ulp. That breaks bit-exact parity against the legacy numpy feature extractors for
        # `whisper`, `voxtral_realtime`, `audio_spectrogram_transformer` and `speech_to_text`
        # (the last amplifies 1 ulp to 4.77e-06 through its per-utterance CMVN).
        #
        # Bit-equality is the correctness guide for now; trading it for throughput is a separate,
        # explicitly-motivated decision. Do not re-add this without resolving that.
        #
        # Note: the `test-spectrogram` harness does NOT catch this class of change — its kaldi and
        # torchaudio families stay green because the reference libraries consume the same
        # contiguous layout we produce. Only the numpy legacy extractors pin the strided
        # accumulation order, so run the parity suite (`./validate`) for any layout change.
        if spectrogram_config and spectrogram_config.stft_config.magnitude_mode == "sqrt_sum_squares":
            # NeMo-derived form; differs from `abs()` in the last ulp
            magnitudes = torch.view_as_real(stft_out).pow(2).sum(-1).sqrt()
            return magnitudes.pow(power) if power != 1.0 else magnitudes
        return stft_out.abs() ** power

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
        mel_filters = mel_filters.to(device=features.device, dtype=features.dtype)
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
