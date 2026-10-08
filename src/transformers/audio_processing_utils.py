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
from dataclasses import fields
from typing import TYPE_CHECKING, Any, Unpack

import numpy as np
from huggingface_hub.dataclasses import validate_typed_dict

from .audio_processing_backends import _side_widths, get_audio_backend
from .audio_processing_base import AudioProcessingMixin, BatchFeature
from .audio_utils import (
    AudioInput,
    SpectrogramConfig,
    _create_triangular_filter_bank,
    hertz_to_mel,
    make_list_of_audio,
    mel_to_hertz,
)
from .processing_utils import AudioKwargs
from .tokenization_utils_base import TruncationStrategy
from .utils import PaddingStrategy, TensorType, is_torch_available, logging


if TYPE_CHECKING:
    import torch


logger = logging.get_logger(__name__)


# Config spellings of the window names `AudioOps.window` takes.
_WINDOW_ALIASES = {"hann_window": "hann", "hamming_window": "hamming", "blackman_window": "blackman"}
# Config spellings of `pad_mode` (numpy's `"edge"` included) mapped to the modes `AudioOps.pad` takes.
_PAD_MODES = {"constant": "constant", "reflect": "reflect", "edge": "replicate", "replicate": "replicate"}
# The complex dtype that carries a spectrum whose magnitudes are taken in the given float dtype.
_COMPLEX_OF = {"float32": "complex64", "float64": "complex128"}
# `MelScaleConfig.matmul_order`: two filters-first GEMM orders (torchaudio's / `F.linear`'s, and `filters.T @ x`)
# differ in the last ulp, so both are values; "features_first" leaves the frames axis first.
_MATMUL_ORDERS = ("filters_first", "filters_first_matmul", "features_first")


# ── Mel filter banks ──────────────────────────────────────────────────────────────────────────
#
# Banks are small and built once at init; `self.mel_filters` is a host numpy array on every backend
# (float32 unless a computation dtype says otherwise) and the projection converts it per call.
# `bank_rounding` names whose construction a bank reproduces: `"librosa"` is librosa's and the legacy
# numpy extractors' (float64 geometry on the host, rounded to float32 at the end); `"torchaudio"` is
# torchaudio's, evaluated op by op through the backend's own kernels (float32 by default) because
# torch's float32 `log`/`exp`/`linspace` are not correctly rounded: only the same kernels give the
# same bits, and `to_numpy` carries them to the host unchanged. The builders no longer reach
# `audio_utils`' torch branches (`hertz_to_mel`/`mel_to_hertz`): those are now public-API-only.


def _hertz_to_mel_ops(ops, freq, mel_scale, *, like):
    """`hertz_to_mel` on a backend array, torchaudio's arithmetic (`audio_utils.hertz_to_mel`'s torch branch).
    A transcription, not a call: `audio_utils` dispatches on the array type and imports torch in its branch,
    so it cannot run on `self.ops`."""
    if mel_scale == "htk":
        return 2595.0 * ops.log10(1.0 + freq / 700.0)
    if mel_scale == "kaldi":
        return 1127.0 * ops.log(1.0 + freq / 700.0)
    # slaney: the log step is evaluated as a float32 0-d array, as torchaudio's
    # `27.0 / torch.log(torch.tensor(6.4))`, then cast (exactly) to `freq`'s dtype so that binary ops
    # see one dtype; a float32 `freq` keeps the float32 step unchanged.
    linear = freq / (200.0 / 3.0)
    logstep = 27.0 / ops.log(ops.asarray(np.asarray(6.4, dtype=np.float32), like=like))
    logstep = ops.astype(logstep, ops.dtype_name(freq))
    log_branch = 15.0 + ops.log(ops.clamp_min(freq, 1000.0) / 1000.0) * logstep
    return ops.where(freq >= 1000.0, log_branch, linear)


def _mel_to_hertz_ops(ops, mels, mel_scale):
    """`mel_to_hertz` on a backend array, torchaudio's arithmetic (`audio_utils.mel_to_hertz`'s torch branch).
    A transcription, not a call: `audio_utils` dispatches on the array type and imports torch in its branch,
    so it cannot run on `self.ops`."""
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    if mel_scale == "kaldi":
        return 700.0 * (ops.exp(mels / 1127.0) - 1.0)
    linear = (200.0 / 3.0) * mels
    log_branch = 1000.0 * ops.exp((math.log(6.4) / 27.0) * (mels - 15.0))
    return ops.where(mels >= 15.0, log_branch, linear)


def _triangular_filter_bank_ops(ops, fft_freqs, filter_freqs):
    """`_create_triangular_filter_bank` on backend arrays: ``(num_frequency_bins, n_mels)``."""
    filter_diff = filter_freqs[1:] - filter_freqs[:-1]
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return ops.clamp_min(ops.minimum(down_slopes, up_slopes), 0.0)


def _triangular_banks_librosa(fft_freqs, filter_freqs, dtype):
    """librosa's dtype policy: float64 slopes, rounded to float32 at the end (or to `dtype`). librosa
    assigns each band's column into a float32 array; the arithmetic per element is the same as
    `_create_triangular_filter_bank`'s, so one cast at the end reproduces it bit for bit."""
    return _create_triangular_filter_bank(fft_freqs, filter_freqs).astype(np.dtype(dtype or "float32"), copy=False)


def _kaldi_banks_torchaudio(ops, n_mels, num_bins, f_min, f_max, sampling_rate, n_fft, dtype):
    """`torchaudio.compliance.kaldi.get_mel_banks`, op by op: hard-coded ``1127 * log`` in `dtype` (float32 by
    default), edges from `mel_low + k * mel_delta`, a zero Nyquist bin (padded before the transpose).
    `num_bins` is unused (the bins follow from `n_fft`); it keeps the builders' uniform signature."""
    dtype = dtype or "float32"
    bin_width = sampling_rate / n_fft
    mel_low = 1127.0 * math.log(1.0 + f_min / 700.0)
    mel_high = 1127.0 * math.log(1.0 + f_max / 700.0)
    mel_delta = (mel_high - mel_low) / (n_mels + 1)
    bin_idx = ops.astype(ops.arange(n_mels), dtype)[:, None]
    left = mel_low + bin_idx * mel_delta
    center = mel_low + (bin_idx + 1.0) * mel_delta
    right = mel_low + (bin_idx + 2.0) * mel_delta
    mel = 1127.0 * ops.log(1.0 + bin_width * ops.astype(ops.arange(n_fft // 2), dtype) / 700.0)
    mel = mel[None, :]
    up_slope = (mel - left) / (center - left)
    down_slope = (right - mel) / (right - center)
    banks = ops.maximum(ops.zeros((1,), dtype=dtype), ops.minimum(up_slope, down_slope))
    return ops.swapaxes(ops.pad(banks, 0, 1), 0, 1)


def _kaldi_banks_librosa(n_mels, num_bins, f_min, f_max, sampling_rate, n_fft, mel_cfg, dtype):
    """The legacy numpy extractors' construction, on the host: `hertz_to_mel` on float64 bin frequencies,
    filter edges from `np.linspace` in mel space, one float32 rounding at the end; the lowest
    `bands_to_zero` bins zeroed."""
    mel_min = hertz_to_mel(f_min, mel_scale=mel_cfg.mel_scale)
    mel_max = hertz_to_mel(f_max, mel_scale=mel_cfg.mel_scale)
    filter_freqs = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = sampling_rate / n_fft * np.arange(mel_cfg.bands_to_zero, num_bins)
    banks = _triangular_banks_librosa(hertz_to_mel(hz, mel_scale=mel_cfg.mel_scale), filter_freqs, dtype)
    return np.pad(banks, ((mel_cfg.bands_to_zero, 0), (0, 0))) if mel_cfg.bands_to_zero else banks


def _kaldi_banks_zero_bands_torchaudio(ops, n_mels, num_bins, f_min, f_max, sampling_rate, n_fft, mel_cfg, dtype):
    """Mel-space triangularization in `dtype` (float32 by default) with the lowest `bands_to_zero` bins
    zeroed: `linspace` edges and bin frequencies in `dtype`, `hertz_to_mel` on them, op by op."""
    dtype = dtype or "float32"
    mel_min = hertz_to_mel(f_min, mel_scale=mel_cfg.mel_scale)
    mel_max = hertz_to_mel(f_max, mel_scale=mel_cfg.mel_scale)
    filter_freqs = ops.linspace(mel_min, mel_max, n_mels + 2, dtype=dtype)
    hz = sampling_rate / n_fft * ops.astype(ops.arange(num_bins), dtype)[mel_cfg.bands_to_zero :]
    fft_freqs = _hertz_to_mel_ops(ops, hz, mel_cfg.mel_scale, like=hz)
    banks = _triangular_filter_bank_ops(ops, fft_freqs, filter_freqs)
    return ops.pad(banks, mel_cfg.bands_to_zero, 0, axis=0) if mel_cfg.bands_to_zero else banks


def _standard_banks(ops, n_mels, num_bins, f_min, f_max, sampling_rate, n_fft, mel_cfg, dtype):
    """Standard triangular bank. `bank_rounding="torchaudio"`: `melscale_fbanks`' order op by op (construction
    in `dtype`, float32 by default, slaney norm after in that dtype). `"librosa"`: `librosa.filters.mel`'s
    order on the host (float64 geometry, rounded to `dtype`, float32 by default, before the norm and again
    after it). `frequency_bin_mode` is honoured by both: `"rfft"` is `rfftfreq`'s bin spacing, `"linspace"`
    is `linspace(0, sr // 2, bins)`, the legacy numpy extractors' (and librosa's) form; the two differ
    when `sr // 2` is not a multiple of the bin width."""
    mel_min = hertz_to_mel(f_min, mel_scale=mel_cfg.mel_scale)
    mel_max = hertz_to_mel(f_max, mel_scale=mel_cfg.mel_scale)
    if mel_cfg.bank_rounding == "librosa":
        filter_freqs = mel_to_hertz(np.linspace(mel_min, mel_max, n_mels + 2), mel_scale=mel_cfg.mel_scale)
        if mel_cfg.frequency_bin_mode == "rfft":
            fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
        else:
            fft_freqs = np.linspace(0, sampling_rate // 2, num_bins)
        banks = _triangular_banks_librosa(fft_freqs, filter_freqs, dtype)
        if mel_cfg.norm == "slaney":
            enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
            banks = banks * enorm[None, :]
            banks = banks.astype(np.dtype(dtype or "float32"), copy=False)
        return np.pad(banks, ((mel_cfg.bands_to_zero, 0), (0, 0))) if mel_cfg.bands_to_zero else banks
    dtype = dtype or "float32"
    filter_freqs = _mel_to_hertz_ops(ops, ops.linspace(mel_min, mel_max, n_mels + 2, dtype=dtype), mel_cfg.mel_scale)
    if mel_cfg.frequency_bin_mode == "rfft":
        # `torch.fft.rfftfreq`: the integer bins scaled by `1 / (n * d)` in float32
        fft_freqs = ops.astype(ops.arange(num_bins), "float32") * (1.0 / (n_fft * (1.0 / sampling_rate)))
    else:
        fft_freqs = ops.linspace(0, sampling_rate // 2, num_bins, dtype="float32")
    fft_freqs = ops.astype(fft_freqs, dtype)
    banks = _triangular_filter_bank_ops(ops, fft_freqs, filter_freqs)
    if mel_cfg.norm == "slaney":
        enorm = 2.0 / (filter_freqs[2 : n_mels + 2] - filter_freqs[:n_mels])
        banks = banks * enorm[None, :]
    return ops.pad(banks, mel_cfg.bands_to_zero, 0, axis=0) if mel_cfg.bands_to_zero else banks


class BaseAudioProcessor(AudioProcessingMixin):
    """Base class of every audio processor: the whole pipeline (validation, resampling, padding, spectrogram,
    normalization, output), written once against an op table. A model subclasses it and overrides hooks.

    Args:
        sampling_rate (`int`, *optional*):
            The sampling rate the processor expects; required unless the subclass sets it as a class attribute.
        backend (`str`, *optional*):
            Which op table the processor computes with: `"torch"`, `"numpy"`, or a backend registered with
            [`register_audio_backend`] (e.g. `"mlx"` after `import mlx_audio.transformers_backend`). Defaults to
            `"torch"` when torch is importable, else `"numpy"`. A load-time choice: it is passed to the
            constructor (or to `from_pretrained` / `from_dict`), kept as `self.backend` with the op table in
            `self.ops`, and never serialized, so a saved processor loads on any backend. A model that calls ops
            beyond the core table lists them in `required_ops`; a backend lacking one is refused here.
        **kwargs ([`AudioKwargs`]):
            Values for the processor's options, validated once the attributes are set.
    """

    valid_kwargs = AudioKwargs
    padding = True
    padding_side = "right"
    padding_value = 0.0
    return_padding_mask = True

    dither: float = 0.0
    feature_normalization: str | None = None
    feature_normalization_eps: float = 1e-5

    # Declared options a call may not change: the configured value is the only legal value. For
    # a workflow that fixes an option its base pipeline otherwise exposes (a codec that only pads
    # on the right, a waveform model that never extracts a spectrogram). Declared, not coded —
    # see `_validate_preprocess_kwargs` and docs/adr/0014-declarative-kwargs-validation.md.
    frozen_options: tuple[str, ...] = ()

    # Keys `_finalize_output` adds on top of the derived ones, for `model_input_names`.
    extra_model_input_names: list[str] = []

    # Output keys left untouched by `convert_to_tensors`, for non-array metadata a model
    # returns alongside its features (e.g. Cohere-ASR's `audio_chunk_index`).
    skip_tensor_conversion: list[str] = []

    # Ops this model calls beyond the core table (`OPTIONAL_OPS`, or ops attached by `register_audio_op`).
    # Checked at construction, so a backend lacking one fails at `from_pretrained`, not at the first call.
    required_ops: tuple[str, ...] = ()

    def __init__(
        self,
        sampling_rate: int | None = None,
        *,
        backend: str | None = None,
        **kwargs: Unpack[AudioKwargs],
    ):
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate
        if self.sampling_rate is None:
            raise ValueError(
                f"`sampling_rate` must be set either as a class attribute on {self.__class__.__name__} "
                "or passed to __init__."
            )

        super().__init__(**kwargs)
        # The op table is a load-time choice: never serialized, not a `valid_kwargs` option.
        if backend is None:
            backend = "torch" if is_torch_available() else "numpy"
        if not isinstance(backend, str):
            raise TypeError(f"`backend` must be a string, got {type(backend).__name__}.")
        self.backend = backend
        self.ops = get_audio_backend(backend)()
        for op in self.required_ops:
            if not hasattr(self.ops, op):
                raise ValueError(f"{type(self).__name__} on backend {backend!r} is missing op {op!r}.")
        self._set_attributes(**kwargs)

    def _set_attributes(self, **kwargs):
        """Validate the resolved attributes and build the init-time constants (mel bank). Runs at the end of
        `__init__`."""
        super()._set_attributes(**kwargs)
        # Validate the *resolved* attribute set, not just what the caller passed: a bad value reaching us
        # from a class default or a hub config should fail here, not crash somewhere inside `_preprocess`.
        validate_typed_dict(self.valid_kwargs, {key: getattr(self, key) for key in self._valid_kwargs_names})
        if (
            self.spectrogram_config is not None
            and self.spectrogram_config.mel_scale_config is not None
            and not hasattr(self, "mel_filters")
        ):
            self.mel_filters = self._mel_filter_bank(self.spectrogram_config)

    @property
    def model_input_names(self) -> list[str]:
        """The keys `__call__` emits, i.e. the kwargs the model's `forward` accepts.

        Mirrors how `_standardize_kwargs` resolves `do_extract_spectrogram`. Declare
        `extra_model_input_names` for keys `_finalize_output` adds; shadow this with a plain
        list only when it *renames* the derived keys. A composite processor subtracts the ones
        it consumes itself via its `unused_input_names`.
        """
        extracts_spectrogram = getattr(self, "do_extract_spectrogram", self.spectrogram_config is not None)
        key = "audio_features" if extracts_spectrogram else "audio_values"
        names = [key, f"{key}_mask"] if self.return_padding_mask else [key]
        return names + self.extra_model_input_names

    def _standardize_kwargs(
        self,
        **kwargs,
    ) -> dict:
        if isinstance(kwargs.get("spectrogram_config"), dict):
            kwargs["spectrogram_config"] = SpectrogramConfig.from_dict(kwargs["spectrogram_config"])
        if kwargs.get("spectrogram_config") is not None and kwargs.get("do_extract_spectrogram") is None:
            kwargs["do_extract_spectrogram"] = True
        return kwargs

    def _validate_preprocess_kwargs(self, **kwargs):
        """Enforce `frozen_options` on the merged call kwargs. Final: models do not override this.

        Every other constraint a model has on its options is declared on its kwargs schema and
        checked by `validate_typed_dict` before this runs — a `Literal` for an enumerated choice,
        an `Annotated` validator such as `interval(min=0, exclude_min=True)` for a bound. A value
        derived from other options (`max_length` from a duration) is resolved in
        `_standardize_kwargs` or by the workflow that uses it, not validated here. Truncation
        without `max_length` is rejected by `pad`, where the resolved value is known.
        """
        for name in self.frozen_options:
            configured = getattr(self, name)
            if kwargs.get(name) != configured:
                raise ValueError(
                    f"`{name}` is fixed to {configured!r} for {type(self).__name__} and cannot be changed "
                    f"per call (got {kwargs.get(name)!r})."
                )

    def _serialize_value(self, key, value):
        if key == "spectrogram_config" and hasattr(value, "to_dict"):
            return self._serialize_spectrogram_config(value)
        return value

    def _serialize_spectrogram_config(self, config: SpectrogramConfig) -> dict:
        """`to_dict` plus the nested analogue of `_keep_in_dict`: a `None` that overrides a non-`None`
        class default is kept as an explicit `null`, otherwise `from_dict` merges the saved config onto
        the class default and silently restores the value the instance had turned off (CLAP's fusion
        bank has `norm: None` where the class says `"slaney"`)."""
        serialized = config.to_dict()
        default = getattr(type(self), "spectrogram_config", None)
        if default is None:
            return serialized
        if isinstance(default, dict):
            default = SpectrogramConfig.from_dict(default)
        for name in ("stft_config", "mel_scale_config"):
            nested, nested_default = getattr(config, name), getattr(default, name)
            if nested is None or nested_default is None:
                continue
            for f in fields(nested):
                if getattr(nested, f.name) is None and getattr(nested_default, f.name) is not None:
                    serialized.setdefault(name, {})[f.name] = None
        for f in fields(config):
            if f.name in ("stft_config", "mel_scale_config"):
                continue
            if getattr(config, f.name) is None and getattr(default, f.name) is not None:
                serialized[f.name] = None
        return serialized

    def __call__(self, audio: str | list[str] | AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
        is_audio_reference = isinstance(audio, str) or (
            isinstance(audio, (list, tuple)) and all(isinstance(audio_el, str) for audio_el in audio)
        )
        if not is_audio_reference and kwargs.get("sampling_rate") is None:
            logger.warning_once(
                f"`sampling_rate` was not provided. The audio arrays will be assumed to be sampled at "
                f"{self.sampling_rate} Hz, which can produce incorrect results if that assumption is wrong."
            )

        # `return_attention_mask` was the legacy spelling of `return_padding_mask`. Alias it rather than
        # letting it fall through to `**kwargs` unread — a silently ignored mask request is the exact
        # failure this contract removes. Removal target: v5.15, alongside the other legacy aliases.
        if "return_attention_mask" in kwargs:
            warnings.warn(
                "`return_attention_mask` is deprecated and will be removed in transformers v5.15. "
                "Use `return_padding_mask` instead.",
                FutureWarning,
                stacklevel=2,
            )
            kwargs.setdefault("return_padding_mask", kwargs.pop("return_attention_mask"))
        return super().preprocess(audio, *args, **kwargs)

    def _preprocess_like_inputs(self, audio: AudioInput, *args, **kwargs) -> BatchFeature:
        return self._preprocess_audio_like_inputs(audio, *args, **kwargs)

    def _preprocess_audio_like_inputs(
        self,
        audio: AudioInput,
        *args,
        sampling_rate: int | None = None,
        **kwargs: Unpack[AudioKwargs],
    ) -> BatchFeature:
        audio = self._prepare_audio_like_inputs(audio=audio, sampling_rate=sampling_rate, **kwargs)
        return self._preprocess(audio, *args, **kwargs)

    def _prepare_audio_like_inputs(
        self, audio: AudioInput, *args, sampling_rate: int | None = None, device=None, **kwargs
    ) -> list:
        audio, sampling_rate = self._prepare_audio_structure(audio, sampling_rate=sampling_rate)
        # Placement is a batch step, not a per-waveform hook: `_prepare_waveform` overrides need not
        # remember to forward `device`, and the whole batch crosses to the device in one transfer.
        audio = self._to_device([self._downmix_to_mono(audio_el) for audio_el in audio], device)
        # Resample before `_prepare_waveform`, so a hook that reasons in samples (a minimum length, a
        # reflect multiple, a fill to a clip length) sees the native rate; `_resample` sees a mono
        # waveform in the backend's own array type either way.
        if sampling_rate != self.sampling_rate:
            logger.warning_once(
                f"Resampling audio from {sampling_rate} Hz to {self.__class__.__name__}'s native sampling rate "
                f"of {self.sampling_rate} Hz. Pass audio already sampled at {self.sampling_rate} Hz to skip this."
            )
            audio = [self._resample(audio_el, sampling_rate, self.sampling_rate) for audio_el in audio]
        return [self._prepare_waveform(audio_el, **kwargs) for audio_el in audio]

    def _prepare_audio_structure(self, audio: AudioInput, sampling_rate: int | None = None) -> tuple[list, int]:
        """Resolve `audio` to a list of waveforms, plus the rate those waveforms are actually at.

        `sampling_rate` describes *passed arrays* only: it is the caller's assertion about data we cannot
        inspect. Audio referenced by URL or path is decoded at the native rate by construction, so a rate
        given alongside such input has nothing to describe and is ignored.
        """
        is_url_input = isinstance(audio, str) or (
            isinstance(audio, (list, tuple)) and all(isinstance(el, str) for el in audio)
        )

        if is_url_input:
            # `preprocess` setdefaults `sampling_rate` from `self.sampling_rate`, so only a caller who
            # passed a *different* rate is telling us something we cannot honour.
            if sampling_rate is not None and sampling_rate != self.sampling_rate:
                logger.warning_once(
                    f"Ignoring `sampling_rate={sampling_rate}`: audio given as a URL or path is decoded at "
                    f"{self.__class__.__name__}'s native sampling rate of {self.sampling_rate}. "
                    "`sampling_rate` describes audio passed as an array."
                )
            return make_list_of_audio(self.fetch_audio(audio)), self.sampling_rate

        # A lone 2-D array is one clip whose channels `_downmix_to_mono` averages. The legacy feature extractors
        # read the same array as a batch of mono clips, and the two are indistinguishable once both axes exceed 1.
        if getattr(audio, "ndim", None) == 2 and min(audio.shape) > 1:
            warnings.warn(
                f"Received a single 2-D array of shape {tuple(audio.shape)}: it is treated as one clip with "
                f"{min(audio.shape)} channels, averaged to mono. The legacy feature extractors read a 2-D array as a "
                "batch of mono clips; to process a batch, pass a list of 1-D arrays. Wrap a multi-channel clip in a "
                "list (`[audio]`) to silence this warning.",
                UserWarning,
            )

        return make_list_of_audio(audio), sampling_rate if sampling_rate is not None else self.sampling_rate

    def _resample(self, audio, orig_sampling_rate: int, target_sampling_rate: int):
        """Resample one mono waveform to ``target_sampling_rate``."""
        return self.ops.resample(audio, orig_sampling_rate, target_sampling_rate)

    def _preprocess(
        self,
        audio: list["np.ndarray"] | list["torch.Tensor"],
        padding: bool | str | PaddingStrategy | None,
        max_length: int | None,
        truncation: bool | str | TruncationStrategy | None,
        pad_to_multiple_of: int | None,
        return_tensors: str | TensorType | None,
        spectrogram_config: SpectrogramConfig | None,
        do_extract_spectrogram: bool | None,
        padding_side: str,
        padding_value: float,
        return_padding_mask: bool,
        **kwargs: Any,
    ) -> BatchFeature:
        # Pad the audio, then optionally extract a spectrogram on the padded batch.
        audio, audio_ranges = self.pad(
            audio,
            padding,
            max_length,
            truncation,
            pad_to_multiple_of,
            padding_side=padding_side,
            padding_value=padding_value,
            **kwargs,
        )
        padded_length = audio[0].shape[-1]
        batched = self._stack_waveforms(audio, padding=padding)

        if do_extract_spectrogram:
            features, feature_lengths = self._compute_batched_features(
                batched,
                spectrogram_config=spectrogram_config,
                audio_ranges=audio_ranges,
                padding=padding,
                max_length=max_length,
                truncation=truncation,
                pad_to_multiple_of=pad_to_multiple_of,
                padding_side=padding_side,
                padding_value=padding_value,
                return_tensors=return_tensors,
                **kwargs,
            )
            output = {"audio_features": features}
        else:
            output = {"audio_values": batched}

        if return_padding_mask:
            # Features live on the frame axis: map audio ranges → feature ranges via hop_length.
            if do_extract_spectrogram:
                mask_length = self._padded_frame_count(padded_length, spectrogram_config, **kwargs)
                output["audio_features_mask"] = self._get_mask_from_lengths(
                    feature_lengths, mask_length, like=next(iter(output.values()))
                )
            else:
                mask_ranges = audio_ranges
                mask_length = padded_length
                output["audio_values_mask"] = self._get_mask(
                    mask_ranges, mask_length, like=next(iter(output.values()))
                )

        output = self._finalize_output(
            output,
            audio_ranges=audio_ranges,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            spectrogram_config=spectrogram_config,
            padding_side=padding_side,
            padding_value=padding_value,
            return_padding_mask=return_padding_mask,
            **kwargs,
        )
        return BatchFeature(
            data=output, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion
        )

    def _compute_batched_features(
        self,
        audio,
        *,
        audio_ranges,
        spectrogram_config,
        **kwargs,
    ):
        """Compute the complete batched feature recipe before masks and output augmentation.

        Models whose recipe does not fit the portable pipeline override this seam (AST,
        SeamlessM4T, Fun-ASR, Speech2Text). The default composes the portable feature pipeline
        and the processor's declared batched normalization.
        """
        features = self.spectrogram(
            audio,
            spectrogram_config=spectrogram_config,
            audio_ranges=audio_ranges,
            **kwargs,
        )
        audio_lengths = self._lengths_from_ranges(audio_ranges)
        feature_lengths = self._valid_frame_counts(audio_lengths, spectrogram_config)
        if self.feature_normalization is None:
            return features, feature_lengths
        if self.feature_normalization != "per_feature_standardize":
            raise ValueError(f"Unknown feature normalization: {self.feature_normalization!r}")

        features = self._standardize_features(features, feature_lengths, eps=self.feature_normalization_eps)
        return features, feature_lengths

    def _finalize_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        """Hook: augment or modify the output dict after main processing.
        Override to add custom fields (e.g., audio_embed_sizes) or post-hoc normalization on the stacked/batched output.
        """
        return output

    def _mask_padded_frames(self, features, frame_counts, *, padding_value):
        """Set each clip's frames at or past its `frame_counts` entry to `padding_value`, on the time axis 1.

        For models whose legacy extractor padded the *features* rather than the audio: on the padded batch,
        those frames hold features of silence instead of the padding value.
        """
        counts = self.ops.asarray(np.asarray(frame_counts, dtype=np.int64), like=features)[:, None]
        valid = (self.ops.arange(features.shape[1], like=features)[None, :] < counts)[..., None]
        return self.ops.masked_fill(features, ~valid, padding_value)

    def _frame_moments(self, features, frame_counts, *, ddof):
        """Float64 mean and variance of each clip over its own frames, on the time axis 1 of the padded batch.

        Returns `(features, mean, variance)`, all float64, for per-clip normalization rounded once by the caller.
        Accumulating in float64 makes the result independent of the batch layout and of the reduction order.
        """
        features = self.ops.astype(features, "float64")
        counts = self.ops.asarray(np.asarray(frame_counts, dtype=np.int64), like=features)
        valid = (self.ops.arange(features.shape[1], like=features)[None, :] < counts[:, None])[..., None]
        counts = self.ops.astype(counts, "float64")[:, None, None]
        mean = self.ops.sum(self.ops.masked_fill(features, ~valid, 0.0), 1, keepdims=True) / counts
        centered = self.ops.masked_fill(features - mean, ~valid, 0.0)
        variance = self.ops.sum(centered * centered, 1, keepdims=True) / (counts - float(ddof))
        return features, mean, variance

    def _standardize_features(self, features, frame_counts, eps):
        """Zero-mean, unit-variance each utterance over its first `frame_counts` frames; padded frames come out zero.

        The NeMo recipe (Parakeet, Cohere-ASR): unbiased variance, `eps` added to the standard deviation.
        """
        counts = self.ops.asarray(np.asarray(frame_counts, dtype=np.int64), like=features)
        valid = (self.ops.arange(features.shape[1], like=features)[None, :] < counts[:, None])[..., None]
        # integer counts are exact in float32/float64 (below 2**24 frames)
        counts = self.ops.astype(counts, self.ops.dtype_name(features))[:, None, None]
        masked = self.ops.masked_fill(features, ~valid, 0.0)
        mean = self.ops.sum(masked, 1, keepdims=True) / counts
        centered = self.ops.masked_fill(masked - mean, ~valid, 0.0)
        variance = self.ops.sum(centered * centered, 1, keepdims=True) / (counts - 1.0)
        standardized = (features - mean) / (self.ops.sqrt(variance) + eps)
        return self.ops.masked_fill(standardized, ~valid, 0.0)

    def _resolve_padding_strategy(self, padding=False, max_length=None, *, padding_value):
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined"
                )

        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (padding_value is None):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy

    def pad(
        self,
        audio: list["np.ndarray"] | list["torch.Tensor"],
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        *,
        padding_side: str,
        padding_value: float,
        **kwargs,
    ) -> tuple[list, list[tuple[int, int]]]:
        padding_strategy = self._resolve_padding_strategy(
            padding=padding, max_length=max_length, padding_value=padding_value
        )

        if truncation:
            # Checked here, where the resolved value is known, rather than in `_validate_preprocess_kwargs`:
            # a workflow may derive `max_length` after validation ran (UnivNet, MusicGen Melody), and
            # `pad` is public and callable directly.
            if max_length is None:
                raise ValueError("When setting `truncation=True`, make sure that `max_length` is defined.")
            trunc_length = max_length
            if pad_to_multiple_of is not None and (trunc_length % pad_to_multiple_of != 0):
                trunc_length = ((trunc_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            audio = [self._truncate_waveform(audio_el, max_length=trunc_length, **kwargs) for audio_el in audio]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(audio_el.shape[-1] for audio_el in audio)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        actual_lengths = [audio_el.shape[-1] for audio_el in audio]

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            audio = self._pad_waveforms(
                audio, max_length=max_length, padding_side=padding_side, padding_value=padding_value, **kwargs
            )

        audio_ranges = []
        for i, length in enumerate(actual_lengths):
            padded_length = audio[i].shape[-1]
            if padding_side == "left":
                audio_ranges.append((padded_length - length, padded_length))
            else:
                audio_ranges.append((0, length))

        return audio, audio_ranges

    def _truncate_waveform(self, audio_el, max_length: int, **kwargs):
        return audio_el[..., :max_length] if audio_el.shape[-1] > max_length else audio_el

    def _pad_features(self, features, padding, max_length, truncation, pad_to_multiple_of, *, padding_value, **kwargs):
        padding_strategy = self._resolve_padding_strategy(
            padding=padding, max_length=max_length, padding_value=padding_value
        )
        if truncation and max_length is not None:
            features = [f[:max_length] for f in features]
        actual_lengths = [f.shape[0] for f in features]
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(actual_lengths)
            padding_strategy = PaddingStrategy.MAX_LENGTH
        if max_length is not None and pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        if padding_strategy == PaddingStrategy.MAX_LENGTH and max_length is not None:
            features = [
                f
                if f.shape[0] >= max_length
                else self._pad_feature_single(f, max_length, padding_value=padding_value, **kwargs)
                for f in features
            ]
        return features, [(0, length) for length in actual_lengths]

    def _pad_feature_single(self, feature, max_length, *, padding_value, **kwargs):
        """Right-pad one feature array/tensor along its first (time) axis with `padding_value`."""
        return self.ops.pad(feature, 0, max_length - feature.shape[0], axis=0, value=padding_value)

    def _downmix_to_mono(self, audio_el):
        """Average any channel axis away. To change the waveform itself, override `_prepare_waveform`.

        A 2-D waveform is one clip, in either layout: `(channels, samples)` as torchaudio loads it, or
        `(samples, channels)` as soundfile does. The longer axis is time; a square array is read as
        channels-first.
        """
        audio_el = self.ops.asarray(audio_el)
        if audio_el.ndim == 2 and audio_el.shape[0] > audio_el.shape[1]:
            audio_el = self.ops.swapaxes(audio_el, 0, 1)
        if audio_el.ndim > 1:
            # Multi-channel input is always averaged down to mono.
            audio_el = audio_el[0] if audio_el.shape[0] == 1 else self.ops.mean(audio_el, 0)
        return audio_el

    def _prepare_waveform(self, audio_el, **kwargs):
        """Hook: condition one mono waveform, before any resampling and before the batch is assembled.
        Override for level normalization, a length floor, or anything else the utterance needs on its own.
        """
        return audio_el

    def _to_device(self, audio: list, device) -> list:
        """Place the mono waveforms on `device`, as one batch transfer where the backend can."""
        return audio if device is None else self.ops.to_device_many(audio, device)

    def _pad_waveforms(
        self, audio: list, max_length: int, *, padding_side: str, padding_value: float, **kwargs
    ) -> list:
        """Pad every waveform to `max_length` on `padding_side`; one already that long passes unchanged."""
        return self.ops.pad_many(audio, max_length, side=padding_side, value=padding_value)

    def _stack_waveforms(self, audio, *, padding=None):
        self._validate_stackable(audio, padding=padding, item_name="waveforms")
        return self.ops.stack(audio)

    def _pad_waveform(
        self, audio, max_length: int, *, padding_side: str, padding_value: float, **kwargs
    ) -> AudioInput:
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio
        left, right = _side_widths(padding_side, max_length - current_length)
        return self.ops.pad(audio, left, right, axis=-1, value=padding_value)

    def _stack_features(self, features, *, padding=None):
        self._validate_stackable(features, padding=padding, item_name="features")
        return self.ops.stack(features)

    def _validate_stackable(self, items, *, padding, item_name):
        shapes = [tuple(item.shape) for item in items]
        if not shapes or all(shape == shapes[0] for shape in shapes[1:]):
            return

        if padding is False or padding == PaddingStrategy.DO_NOT_PAD:
            raise ValueError(
                f"Cannot create a batch from {item_name} with different shapes {shapes} when padding is disabled. "
                'Pass `padding="longest"`, use `padding="max_length"` with `truncation=True`, or provide a list '
                f"of {item_name} that already have the same shape."
            )
        raise ValueError(
            f"Cannot create a batch from {item_name} with different shapes {shapes}. "
            'Pass `padding="longest"`, use `padding="max_length"` with `truncation=True`, or provide inputs '
            "that produce the same shape."
        )

    def _get_mask(self, ranges, padded_length, *, like=None):
        """int32 `(len(ranges), padded_length)` with ones on `[start, end)` of each range."""
        positions = self.ops.arange(padded_length, like=like)[None, :]
        bounds = self.ops.asarray(np.asarray(ranges, dtype=np.int64).reshape(-1, 2), like=like)
        return self.ops.astype((positions >= bounds[:, :1]) & (positions < bounds[:, 1:]), "int32")

    def _get_mask_from_lengths(self, lengths, padded_length, *, like=None):
        """int32 `(len(lengths), padded_length)` with ones on `[0, length)` of each entry."""
        positions = self.ops.arange(padded_length, like=like)[None, :]
        lengths = self.ops.asarray(np.asarray(lengths, dtype=np.int64), like=like)
        return self.ops.astype(positions < lengths[:, None], "int32")

    # ── Spectrogram core ─────────────────────────────────────────────────

    def spectrogram(self, audio, *, spectrogram_config: SpectrogramConfig | None, **kwargs):
        """Compute a spectrogram without padding, mask construction, or model-specific output assembly.

        `audio` is one backend array with samples on the last axis. Leading batch dimensions
        are preserved. Output axes are (..., frequency/mel, frames), except
        `matmul_order="features_first"` projects to (..., frames, mel).
        `transpose_features=True` swaps the final two axes.
        Dtype and numerical stages come from `spectrogram_config`. Supply resolved options
        such as `dither` through kwargs; an optional `mel_filters` selects a call-local bank.
        Length metadata and padding policy belong to the calling workflow.
        """
        norm_kwargs = {k: v for k, v in kwargs.items() if k not in ("audio_ranges", "feature_ranges")}

        features = self._waveform_to_spectrum(audio, spectrogram_config=spectrogram_config, **kwargs)
        if spectrogram_config.mel_scale_config is not None:
            features = self._project_to_mel(features, spectrogram_config=spectrogram_config, **kwargs)
        features = self._log_compress(features, spectrogram_config=spectrogram_config, **norm_kwargs)

        return features

    def _waveform_to_spectrum(self, audio, *, spectrogram_config, dither, audio_ranges=None, **kwargs):
        """Convert waveforms with samples on the last axis to ``(..., frequency, frames)`` magnitudes."""
        self._validate_spectrogram_config(spectrogram_config)
        stft_cfg = spectrogram_config.stft_config
        audio = self._condition_waveform_for_stft(
            audio, spectrogram_config=spectrogram_config, dither=dither, audio_ranges=audio_ranges
        )
        window, frame_length = self._stft_window(audio, spectrogram_config=spectrogram_config)
        frames = self._frame_waveform(audio, frame_length + stft_cfg.extra_samples_per_frame, stft_cfg)
        frames = self._process_frames(
            frames, spectrogram_config=spectrogram_config, audio_ranges=audio_ranges, **kwargs
        )
        stft_out = self._stft_framed(frames, window, stft_cfg)
        return self._spectrum_magnitude(stft_out, spectrogram_config, audio_ranges=audio_ranges, **kwargs)

    def _validate_spectrogram_config(self, spectrogram_config: SpectrogramConfig) -> None:
        stft_cfg = spectrogram_config.stft_config
        if not stft_cfg.onesided:
            raise ValueError("onesided=False is not supported; no model needs the mirrored half of the spectrum.")
        if stft_cfg.extra_samples_per_frame:
            if stft_cfg.extra_samples_per_frame != 1:
                raise ValueError(
                    f"Only extra_samples_per_frame=1 is supported, got {stft_cfg.extra_samples_per_frame}."
                )
            if stft_cfg.center is True:
                raise ValueError(
                    "extra_samples_per_frame requires center=False or center='left', not symmetric centering."
                )
            if spectrogram_config.remove_dc_offset:
                raise ValueError("remove_dc_offset is not supported with extra_samples_per_frame.")
        elif spectrogram_config.preemphasis_mode == "htk_per_frame":
            raise ValueError("preemphasis_mode='htk_per_frame' requires extra_samples_per_frame=1.")
        if stft_cfg.fft_dtype not in (None, "float64", "complex64"):
            raise ValueError(f"fft_dtype must be None, 'float64' or 'complex64', got {stft_cfg.fft_dtype!r}.")
        if stft_cfg.pad_mode not in _PAD_MODES:
            raise ValueError(f"pad_mode must be one of {sorted(_PAD_MODES)}, got {stft_cfg.pad_mode!r}.")
        if stft_cfg.pad < 0:
            raise ValueError(f"pad must be non-negative, got {stft_cfg.pad}.")

    def _condition_waveform_for_stft(self, audio, *, spectrogram_config: SpectrogramConfig, dither, audio_ranges):
        """Apply waveform operations in their numerically significant order before STFT framing."""
        stft_cfg = spectrogram_config.stft_config
        if spectrogram_config.computation_dtype and stft_cfg.fft_dtype != "float64":
            # with `fft_dtype="float64"` the cast happens at the FFT boundary instead
            audio = self.ops.astype(audio, spectrogram_config.computation_dtype)
        if dither > 0:
            audio = self._dither_waveform(audio, audio_ranges, dither=dither)
        if spectrogram_config.waveform_scale is not None:
            audio = audio * spectrogram_config.waveform_scale
        if spectrogram_config.preemphasis is not None and spectrogram_config.preemphasis_mode == "waveform":
            audio = self._preemphasize_waveform(audio, spectrogram_config.preemphasis, audio_ranges)
        if stft_cfg.pad:
            audio = self.ops.pad(audio, stft_cfg.pad, stft_cfg.pad)
        return audio

    def _host_window(self, stft_cfg) -> np.ndarray | None:
        """The two named float32 recipes as host numpy arrays; `None` for every other window.

        `hann_window_f32` (Gemma3n) evaluates the cosine itself in float32 and
        `hann_window_f64_as_f32` (Gemma4) rounds a float64 one; both are kept for bit-exactness
        with their extractors, ignore `periodic` and `window_dtype`, and keep their float32 on
        every backend. Everything else is `self.ops.window`, built on the audio's device.
        """
        n = stft_cfg.win_length
        if stft_cfg.window_fn == "hann_window_f32":
            # a float32 arange under numpy scalar promotion evaluates the whole cosine in float32
            arange = np.arange(n, dtype=np.float32)
            return (0.5 * (1 - np.cos(2 * np.pi * arange / n))).astype(np.float32)
        if stft_cfg.window_fn == "hann_window_f64_as_f32":
            # `window_function(n).astype(np.float32)`: ~2.4e-07 away from `hann_window_f32`
            return np.hanning(n + 1)[:-1].astype(np.float32)
        return None

    @staticmethod
    def _check_wkwargs(stft_cfg):
        """`wkwargs` is kept for backwards compatibility; only `requires_grad=False` (a no-op) is
        accepted. Anything that would change the window has a portable field instead."""
        unsupported = [k for k, v in (stft_cfg.wkwargs or {}).items() if not (k == "requires_grad" and v is False)]
        if unsupported:
            raise ValueError(
                f"Unsupported wkwargs {sorted(unsupported)}: use `periodic`, `window_dtype` and "
                "`blackman_coeff`; a window never needs gradients in preprocessing."
            )

    def _device_window(self, audio, stft_cfg):
        """The `win_length` window on `audio`'s device: a host recipe moved over, or the backend's
        own build in `window_dtype` (else the backend's default for `audio`, see `AudioOps.window`)."""
        self._check_wkwargs(stft_cfg)
        host = self._host_window(stft_cfg)
        if host is not None:
            return self.ops.asarray(host, like=audio)
        name = _WINDOW_ALIASES.get(stft_cfg.window_fn, stft_cfg.window_fn)
        return self.ops.window(
            name,
            stft_cfg.win_length,
            periodic=stft_cfg.periodic,
            dtype=stft_cfg.window_dtype,
            like=audio,
            blackman_coeff=stft_cfg.blackman_coeff,
        )

    def _prepare_window_and_framing(self, window, stft_cfg, frames_on_win_length: bool):
        """Centre the window in an `n_fft` frame unless the framing is on `win_length`."""
        if frames_on_win_length:
            return window, stft_cfg.win_length
        if stft_cfg.win_length < stft_cfg.n_fft:
            left = (stft_cfg.n_fft - stft_cfg.win_length) // 2
            right = stft_cfg.n_fft - stft_cfg.win_length - left
            return self.ops.pad(window, left, right), stft_cfg.n_fft
        return window, stft_cfg.n_fft

    def _stft_window(self, audio, *, spectrogram_config):
        """The window on `audio`'s device in its working dtype, and the frame length. Built per
        call: a device window depends on the audio's device and dtype, and the op is cheap."""
        stft_cfg = spectrogram_config.stft_config
        return self._prepare_window_and_framing(
            self._device_window(audio, stft_cfg), stft_cfg, self._frames_on_win_length(spectrogram_config)
        )

    # ── Spectrogram hooks ────────────────────────────────────────────────

    def _frames_on_win_length(self, spectrogram_config):
        """Whether frames are `win_length` long with the window left-aligned in the FFT buffer, instead
        of `n_fft` long with the window centred."""
        return (
            (
                spectrogram_config.preemphasis is not None
                and spectrogram_config.preemphasis_mode in ("per_frame", "htk_per_frame")
            )
            or spectrogram_config.remove_dc_offset
            or bool(spectrogram_config.stft_config.extra_samples_per_frame)
            or spectrogram_config.stft_config.center == "left"  # a truthy string must not centre-pad
            # left-aligning the window inside the FFT buffer means framing on `win_length` and letting
            # `rfft(n=n_fft)` zero-pad on the right, instead of centring the window in an `n_fft` frame
            or spectrogram_config.stft_config.left_align_fft
        )

    def _frame_count(self, lengths, stft_cfg):
        """Frames the framing yields for `lengths` samples, excluding the extra centered frame."""
        lengths = lengths + 2 * stft_cfg.pad
        if stft_cfg.center == "left":
            count = (
                lengths + stft_cfg.win_length // 2 - (stft_cfg.win_length + stft_cfg.extra_samples_per_frame)
            ) // stft_cfg.hop_length + 1
            return max(0, count) if isinstance(count, int) else count.clip(min=0)
        if not stft_cfg.center:
            return (lengths - (stft_cfg.win_length + stft_cfg.extra_samples_per_frame)) // stft_cfg.hop_length + 1
        return lengths // stft_cfg.hop_length

    def _padded_frame_count(self, padded_length, spectrogram_config, **kwargs) -> int:
        """Width of the extracted features' frame axis, and therefore of the padding mask.

        Override when the framing produces a frame count the geometry above can't express
        (e.g. a model that right-pads to a whole number of hops).
        """
        stft_cfg = spectrogram_config.stft_config
        width = self._frame_count(padded_length, stft_cfg)
        if stft_cfg.center and stft_cfg.center != "left":
            width += 1  # a centered STFT emits the extra frame centered at t=0
        if spectrogram_config.drop_last_frame:
            width -= 1  # the log stage drops the trailing frame
        return int(width)

    def _valid_frame_counts(self, audio_lengths, spectrogram_config):
        """Per-utterance number of frames covered by real (non-padding) audio.

        Honours `count_frames_by_hop`; override only for framing a config can't describe.
        """
        stft_cfg = spectrogram_config.stft_config
        if spectrogram_config.count_frames_by_hop:
            return (audio_lengths + 2 * stft_cfg.pad + stft_cfg.hop_length - 1) // stft_cfg.hop_length
        return self._frame_count(audio_lengths, stft_cfg)

    # ── Spectrogram backend ──────────────────────────────────────────────

    def _frame_waveform(self, audio, frame_length, stft_cfg):
        """Centre- or left-pad as `center` configures and cut `(..., n_frames, frame_length)` frames, in the audio dtype."""
        if stft_cfg.center == "left":
            audio = self.ops.pad(audio, stft_cfg.win_length // 2, 0)  # semicausal (USM/Gemma): zeros prepended only
        elif stft_cfg.center:
            half = frame_length // 2
            mode = _PAD_MODES[stft_cfg.pad_mode]
            if mode == "reflect" and half > audio.shape[-1] - 1:
                raise ValueError(
                    f"Cannot reflect-pad {half} samples on each side of a {audio.shape[-1]}-sample clip; "
                    "reflect padding needs at least frame_length // 2 + 1 samples."
                )
            audio = self.ops.pad(audio, half, half, mode=mode)
        return self.ops.frame(audio, frame_length, stft_cfg.hop_length)

    def _process_frames(self, frames, *, spectrogram_config, **kwargs):
        """Hook: per-frame signal conditioning after frame extraction.

        Called after framing, before windowing and FFT. Applies DC-offset removal, per-frame
        (kaldi-style) preemphasis, and USM/HTK-style extended-frame preemphasis when
        ``stft_config.extra_samples_per_frame`` is set. Override for non-standard frame processing
        that doesn't fit these knobs.
        """
        if spectrogram_config.stft_config.extra_samples_per_frame:
            # USM-style extended frames: preemphasis consumes the extra trailing sample,
            # reducing the frame back to `win_length`.
            preemphasis = spectrogram_config.preemphasis
            if preemphasis is None or preemphasis <= 0.0:
                return frames[..., :-1]
            if spectrogram_config.preemphasis_mode == "htk_per_frame":
                # HTK flavor: first sample scaled by (1 - p) instead of replicate-padded
                first = frames[..., :1] * (1.0 - preemphasis)
                rest = frames[..., 1:-1] - preemphasis * frames[..., :-2]
                return self.ops.concat([first, rest], -1)
            if spectrogram_config.preemphasis_mode == "per_frame":
                return frames[..., 1:] - preemphasis * frames[..., :-1]
            return frames[..., :-1]  # "waveform" mode: already applied upstream
        if spectrogram_config.remove_dc_offset:
            frames = frames - self.ops.mean(frames, -1, keepdims=True)
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None and spectrogram_config.preemphasis_mode == "per_frame":
            # Replicate-pad first sample (x0 - p*x0, not x0*(1-p)): bit-exact with kaldi.
            first = frames[..., :1] - preemphasis * frames[..., :1]
            rest = frames[..., 1:] - preemphasis * frames[..., :-1]
            frames = self.ops.concat([first, rest], -1)
        return frames

    def _preemphasize_waveform(self, audio, preemphasis, audio_ranges=None):
        """Waveform-level preemphasis `x[n] - p * x[n-1]` (first sample unchanged), zeroing the samples
        past each clip's length in ``audio_ranges``. Used when ``spectrogram_config.preemphasis_mode ==
        "waveform"`` (ASR models: Parakeet/Cohere/Nemotron); assumes right-padded ranges (each starts at 0)."""
        out = self.ops.concat([audio[..., :1], audio[..., 1:] - preemphasis * audio[..., :-1]], -1)
        if audio_ranges is not None:
            lengths = self.ops.asarray(self._lengths_from_ranges(audio_ranges), like=audio)
            valid = self.ops.arange(out.shape[-1], like=audio)[None, :] < lengths[:, None]
            out = self.ops.masked_fill(out, ~valid, 0.0)
        return out

    def _stft_framed(self, frames, window, stft_cfg):
        """Window, FFT and normalise `frames`; returns the complex spectrum as ``(..., freq, frames)``.

        Promotion rule: frames and window are cast to the wider of their two dtypes (numpy's
        `result_type`) before the multiplication, so a float64 window lifts float32 frames to float64
        and a float32 window keeps them in float32. `rfft` zero-pads a `win_length` frame to `n_fft`
        on the right, so a left-aligned window needs no explicit pad. The result is a view
        (`swapaxes`), never a contiguous copy: a copy shifts the mel GEMM by an ulp against legacy.
        """
        common = np.result_type(self.ops.dtype_name(frames), self.ops.dtype_name(window)).name
        frames = self.ops.astype(frames, common) * self.ops.astype(window, common)
        if stft_cfg.fft_dtype == "float64":
            frames = self.ops.astype(frames, "float64")
        spec = self.ops.rfft(frames, stft_cfg.n_fft)
        if stft_cfg.fft_dtype == "complex64" and self.ops.dtype_name(spec) == "complex128":
            # the legacy numpy `spectrogram()` wrote its FFT into a complex64 buffer; the caller widens
            # again when the magnitudes are taken in float64
            spec = self.ops.astype(spec, "complex64")
        if stft_cfg.normalized in (True, "window"):
            # the norm is summed in the window's dtype; dividing by a Python float keeps `spec`'s dtype
            norm = self.ops.sqrt(self.ops.sum(self.ops.pow(window, 2.0), None))
            spec = spec / float(self.ops.to_numpy(norm))
        elif stft_cfg.normalized == "frame_length":
            spec = spec / math.sqrt(stft_cfg.n_fft)
        elif stft_cfg.normalized is not False:
            raise ValueError(f"Invalid normalized value: {stft_cfg.normalized!r}")
        return self.ops.swapaxes(spec, -1, -2)

    def _dither_waveform(self, audio, audio_ranges=None, *, dither):
        """Additive Gaussian dither, applied when the merged ``dither`` is nonzero, drawn from the
        library's global stream (`np.random.seed` / `torch.manual_seed` apply); deterministic models
        override (Cohere-ASR). ``audio_ranges`` is ``None`` on unbatched calls. The noise is drawn in
        the audio's dtype on every backend."""
        return audio + dither * self.ops.randn(tuple(audio.shape), like=audio)

    def _spectrum_magnitude(self, stft_out, spectrogram_config, **kwargs):
        """Magnitude (`power=1`) or power (`power=2`) spectrum of the complex STFT.

        With a `computation_dtype` the spectrum is first cast to the complex of `computation_dtype`
        (a complex64 FFT yields float64 magnitudes; a float64 window with `"float32"` narrows);
        `magnitude_mode="sqrt_sum_squares"` is the NeMo form, a few ulp away from `abs`. Without
        one the magnitudes keep the spectrum's dtype: the log stage casts to float32.
        """
        stft_cfg = spectrogram_config.stft_config
        if spectrogram_config.computation_dtype:
            stft_out = self.ops.astype(stft_out, _COMPLEX_OF[spectrogram_config.computation_dtype])
        if stft_cfg.magnitude_mode == "sqrt_sum_squares":
            # contiguous: the mel GEMM's accumulation order follows this operand's layout (measured:
            # 1 float32 ulp on parakeet, cohere_asr and nemotron_asr_streaming without it; the legacy
            # NeMo reduction materialised a contiguous tensor).
            real, imag = self.ops.real(stft_out), self.ops.imag(stft_out)
            magnitudes = self.ops.sqrt(self.ops.contiguous(real * real + imag * imag))
        else:
            magnitudes = self.ops.abs(stft_out)
        return self.ops.pow(magnitudes, stft_cfg.power) if stft_cfg.power != 1.0 else magnitudes

    def _project_to_mel(self, features, *, spectrogram_config, mel_filters=None, **kwargs):
        """`features` ``(..., freq, frames)`` through the bank: ``(..., mel, frames)``, or ``(..., frames, mel)``
        for `matmul_order="features_first"`. The two filters-first orders are two legacy GEMM orders (see
        `MelScaleConfig`); they differ in the last ulp. The host bank is converted per call and matched to the
        feature dtype (torch refuses mixed dtypes; numpy would promote); it is small."""
        bank = self.mel_filters if mel_filters is None else mel_filters
        bank = self.ops.astype(self.ops.asarray(bank, like=features), self.ops.dtype_name(features))
        order = spectrogram_config.mel_scale_config.matmul_order
        if order == "features_first":
            mel = self.ops.matmul(self.ops.swapaxes(features, -2, -1), bank)
        elif order == "filters_first_matmul":
            mel = self.ops.matmul(self.ops.swapaxes(bank, -2, -1), features)
        elif order == "filters_first":
            mel = self.ops.swapaxes(self.ops.matmul(self.ops.swapaxes(features, -2, -1), bank), -2, -1)
        else:
            raise ValueError(f"matmul_order must be one of {_MATMUL_ORDERS}, got {order!r}.")
        if spectrogram_config.mel_floor != 0.0:  # a zero floor on non-negative inputs is a no-op; skip the kernel
            mel = self.ops.clamp_min(mel, spectrogram_config.mel_floor)
        return mel

    def _log_compress(self, features, *, spectrogram_config, reference=1.0, min_value=1e-10, db_range=None, **kwargs):
        log_mel = spectrogram_config.log_mode
        if log_mel is None:
            result = self.ops.astype(features, "float32")
        else:
            if spectrogram_config.pre_log_offset is not None:
                result = features + spectrogram_config.pre_log_offset
            else:
                result = self.ops.clamp_min(features, spectrogram_config.mel_floor)

            if log_mel == "log":
                result = self.ops.astype(self.ops.log(result), "float32")
            elif log_mel == "log10":
                result = self.ops.astype(self.ops.log10(result), "float32")
            elif log_mel == "dB":
                power = spectrogram_config.stft_config.power
                if power == 2.0:
                    result = self._to_db(result, 10.0, reference, min_value, db_range)
                elif power == 1.0:
                    result = self._to_db(result, 20.0, reference, min_value, db_range)
                else:
                    raise ValueError(f"Cannot use log_mel option 'dB' with power {power}")
                result = self.ops.astype(result, "float32")
            else:
                raise ValueError(f"Unknown log_mel option: {log_mel}")

        if spectrogram_config.drop_last_frame:
            result = result[..., :-1]
        result = self._shape_log_features(result, spectrogram_config, **kwargs)
        if spectrogram_config.transpose_features:
            result = self.ops.swapaxes(result, -2, -1)
        return result

    def _to_db(self, spectrogram, multiplier, reference, min_value, db_range):
        """`multiplier * log10(spectrogram / reference)`: 10 for a power spectrogram, 20 for an amplitude one.

        Same arithmetic as `audio_utils.power_to_db` / `audio_utils.amplitude_to_db`, through the backend primitives,
        except that the `db_range` peak is taken per spectrogram (over the last two axes), not over the whole array.
        """
        if reference <= 0.0:
            raise ValueError("reference must be greater than zero")
        if min_value <= 0.0:
            raise ValueError("min_value must be greater than zero")
        reference = max(min_value, reference)
        spectrogram = self.ops.clamp_min(spectrogram, min_value)
        spectrogram = multiplier * (self.ops.log10(spectrogram) - math.log10(reference))
        if db_range is not None:
            if db_range <= 0.0:
                raise ValueError("db_range must be greater than zero")
            peak = self.ops.amax(spectrogram, (-2, -1), keepdims=True)  # per spectrogram, like floor_below_peak
            spectrogram = self.ops.maximum(spectrogram, peak - db_range)
        return spectrogram

    def _shape_log_features(self, result, spectrogram_config, **kwargs):
        """Hook: rescale the log-domain features. Runs after the log and before any transpose."""
        if spectrogram_config.subtract_mean:
            result = result - self.ops.mean(result, -1, keepdims=True)
        if spectrogram_config.floor_below_peak is not None:
            peak = self.ops.amax(result, (-2, -1), keepdims=True)  # per spectrogram
            result = self.ops.maximum(result, peak - spectrogram_config.floor_below_peak)
        if spectrogram_config.log_shift is not None:
            result = result + spectrogram_config.log_shift
        if spectrogram_config.log_scale is not None:
            result = result * spectrogram_config.log_scale
        return result

    # ── Host helpers ─────────────────────────────────────────────────────

    def _lengths_from_ranges(self, ranges):
        """Host-side integer array of `end - start` per `(start, end)` range."""
        return np.asarray([end - start for start, end in ranges], dtype=np.int64)

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        """Build the mel filter bank described by ``spectrogram_config.mel_scale_config`` as a host numpy
        array ``(num_frequency_bins, n_mels)``, float32 unless a computation dtype says otherwise.

        Derives the geometry (number of frequency bins, FFT size, frequency range) and the dtype,
        then builds with one of the module-level builders (the ``"torchaudio"`` ones through `self.ops`):

        - ``_kaldi_banks_librosa``: ``triangularize_in_mel_space`` under ``bank_rounding="librosa"``
        - ``_kaldi_banks_torchaudio``: ``triangularize_in_mel_space`` with no zeroed bands
        - ``_kaldi_banks_zero_bands_torchaudio``: ``triangularize_in_mel_space`` with ``bands_to_zero``
        - ``_standard_banks``: standard triangular filters, either rounding

        Dtype policy: ``mel_scale_config.computation_dtype`` falling back to the top-level
        ``spectrogram_config.computation_dtype`` (pipelines that run in float64 for legacy-FE parity
        need float64 filters as well). When only the mel-level dtype is set, the filters are built in
        that dtype and rounded to float32 afterwards, since the rest of the pipeline runs in float32.
        """
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        n_fft = stft_cfg.n_fft
        num_frequency_bins = 1 + n_fft // 2
        min_frequency = mel_cfg.f_min
        max_frequency = mel_cfg.f_max if mel_cfg.f_max is not None else self.sampling_rate / 2
        computation_dtype = mel_cfg.computation_dtype or spectrogram_config.computation_dtype

        rounding = mel_cfg.bank_rounding
        if rounding not in ("torchaudio", "librosa"):
            raise ValueError(f"bank_rounding must be 'torchaudio' or 'librosa', got {rounding!r}.")
        # validated here so a bad `matmul_order` fails at init, not on the first call
        if mel_cfg.matmul_order not in _MATMUL_ORDERS:
            raise ValueError(f"matmul_order must be one of {_MATMUL_ORDERS}, got {mel_cfg.matmul_order!r}.")
        args = (mel_cfg.n_mels, num_frequency_bins, min_frequency, max_frequency, self.sampling_rate, n_fft)
        if mel_cfg.triangularize_in_mel_space and rounding == "librosa":
            banks = _kaldi_banks_librosa(*args, mel_cfg, computation_dtype)
        elif mel_cfg.triangularize_in_mel_space and mel_cfg.bands_to_zero == 0:
            banks = _kaldi_banks_torchaudio(self.ops, *args, computation_dtype)
        elif mel_cfg.triangularize_in_mel_space:
            banks = _kaldi_banks_zero_bands_torchaudio(self.ops, *args, mel_cfg, computation_dtype)
        else:
            banks = _standard_banks(self.ops, *args, mel_cfg, computation_dtype)

        if not isinstance(banks, np.ndarray):  # built through the ops: the backend's bits, carried to the host
            banks = self.ops.to_numpy(banks)
        if mel_cfg.computation_dtype is not None and not spectrogram_config.computation_dtype:
            banks = banks.astype(np.float32)  # only the bank asked for precision; the pipeline runs in float32
        return np.ascontiguousarray(banks)
