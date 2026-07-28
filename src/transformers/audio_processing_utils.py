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

from dataclasses import fields, replace
from typing import Any, Unpack

import numpy as np

from .audio_processing_base import AudioProcessingMixin
from .audio_utils import (
    AudioInput,
    SpectrogramConfig,
    _array_namespace,
    _clamp_min,
    amplitude_to_db,
    make_list_of_audio,
    power_to_db,
)
from .feature_extraction_utils import BatchFeature
from .processing_utils import AudioKwargs
from .tokenization_utils_base import TruncationStrategy
from .utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class BaseAudioProcessor(AudioProcessingMixin):
    valid_kwargs = AudioKwargs

    force_mono: bool = True
    add_channel_dim: bool = False
    padding = True
    padding_side = "right"
    padding_value = 0.0
    return_padding_mask = True
    mask_level = None
    do_batch_spectrogram = True
    model_input_names = ["audio"]
    dither: float = 0.0

    def __init__(
        self,
        sampling_rate: int | None = None,
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
        # _set_attributes runs in the backend subclasses' __init__, not here, for remote-code BC.

    def _set_attributes(self, **kwargs):
        """Called from the backend subclasses' ``__init__`` (not the base, for remote-code BC)."""
        super()._set_attributes(**kwargs)
        if self.spectrogram_config is not None:
            if self.spectrogram_config.mel_scale_config is not None and not hasattr(self, "mel_filters"):
                self.mel_filters = self._mel_filter_bank(self.spectrogram_config)
        self._cached_stft_window = None

    def _standardize_kwargs(
        self,
        **kwargs,
    ) -> dict:
        if isinstance(kwargs.get("spectrogram_config"), dict):
            kwargs["spectrogram_config"] = SpectrogramConfig.from_dict(kwargs["spectrogram_config"])
        if kwargs.get("spectrogram_config") is not None and kwargs.get("do_extract_spectrogram") is None:
            kwargs["do_extract_spectrogram"] = True
        return kwargs

    def _validate_preprocess_kwargs(
        self,
        sampling_rate: int | None = None,
        max_length: int | None = None,
        truncation: bool | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        if truncation and max_length is None:
            raise ValueError("When setting `truncation=True`, make sure that `max_length` is defined.")

    def _serialize_value(self, key, value):
        if key == "spectrogram_config" and hasattr(value, "to_dict"):
            return value.to_dict()
        return value

    def __call__(self, audio: AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
        return self.preprocess(audio, *args, **kwargs)

    def preprocess(self, audio: AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
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
        audio = self._prepare_audio_like_inputs(audio=audio, sampling_rate=sampling_rate)
        return self._preprocess(audio, *args, **kwargs)

    def _prepare_audio_like_inputs(self, audio: AudioInput, *args, sampling_rate: int | None = None, **kwargs) -> list:
        audio = self._prepare_audio_structure(audio, sampling_rate=sampling_rate)
        audio = [self.process_audio(audio_el) for audio_el in audio]
        return audio

    def _prepare_audio_structure(self, audio: AudioInput, sampling_rate: int | None = None) -> list:
        is_url_input = isinstance(audio, str) or (
            isinstance(audio, (list, tuple)) and all(isinstance(el, str) for el in audio)
        )

        if is_url_input:
            audio = self.fetch_audio(audio)
        else:
            # `PreprocessingMixin.preprocess` setdefaults `sampling_rate` from `self.sampling_rate`,
            # so an omitted rate no-ops here; only a genuine caller mismatch raises.
            if sampling_rate is not None and sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this audio processor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `audio` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )

        audio = make_list_of_audio(audio)
        return audio

    def process_audio(self, *args, **kwargs):
        return self._process_audio(*args, **kwargs)

    def _preprocess(
        self,
        audio: list[np.ndarray] | list["torch.Tensor"],
        padding: bool | str | PaddingStrategy | None,
        max_length: int | None,
        truncation: bool | str | TruncationStrategy | None,
        pad_to_multiple_of: int | None,
        return_tensors: str | TensorType | None,
        spectrogram_config: SpectrogramConfig | None = None,
        do_extract_spectrogram: bool | None = True,
        do_batch_spectrogram: bool | None = True,
        **kwargs: Any,
    ) -> BatchFeature:
        # Path 1: per-waveform spectrogram extraction, padded at the feature level.
        if do_extract_spectrogram and not do_batch_spectrogram:
            features = self.extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
            feature_lengths = [f.shape[0] for f in features]
            features = self._postprocess_features(features, feature_lengths)
            features, feature_ranges = self._pad_features(
                features,
                padding,
                max_length,
                truncation,
                pad_to_multiple_of,
            )
            output = {"audio_features": self._stack_features(features)}
            if self.return_padding_mask:
                output["audio_features_mask"] = self._get_mask(feature_ranges, features[0].shape[0])
            output = self._postprocess_output(output, feature_ranges=feature_ranges, **kwargs)
            return BatchFeature(data=output, tensor_type=return_tensors)

        # Path 2: pad audio first, then optionally extract a spectrogram on the padded batch.
        audio, audio_ranges = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)
        padded_length = audio[0].shape[-1]
        batched = self._to_batch(audio)

        if do_extract_spectrogram:
            output = {
                "audio_features": self.extract_spectrogram(
                    batched,
                    spectrogram_config=spectrogram_config,
                    audio_ranges=audio_ranges,
                    **kwargs,
                )
            }
        else:
            output = {"audio_values": batched}

        if self.return_padding_mask:
            # Features live on the frame axis: map audio ranges → feature ranges via hop_length,
            # unless ``mask_level="audio"`` forces an audio-sample-level mask.
            if do_extract_spectrogram and self.mask_level != "audio":
                spec_cfg = spectrogram_config or self.spectrogram_config
                audio_lengths = np.array([end - start for start, end in audio_ranges])
                feature_lengths = self._get_features_lengths(audio_lengths, spec_cfg)
                mask_ranges = [(0, int(length)) for length in feature_lengths]
                mask_length = int(self._get_features_lengths(padded_length, spec_cfg, include_center_frame=True))
            else:
                mask_ranges = audio_ranges
                mask_length = padded_length
            mask_key = "audio_features_mask" if do_extract_spectrogram else "audio_values_mask"
            output[mask_key] = self._get_mask(mask_ranges, mask_length)

        output = self._postprocess_output(output, audio_ranges=audio_ranges, **kwargs)
        return BatchFeature(data=output, tensor_type=return_tensors)

    def _postprocess_features(self, features, feature_lengths):
        """Hook: per-utterance feature processing after extraction, before feature-level padding.
        Override for normalization that must happen on unpadded features
        """
        return features

    def _postprocess_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        """Hook: augment or modify the output dict after main processing.
        Override to add custom fields (e.g., audio_embed_sizes) or post-hoc normalization on the stacked/batched output.
        """
        return output

    def _get_padding_strategies(self, padding=False, max_length=None):
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

        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.padding_value is None):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy

    def pad(
        self,
        audio: list[np.ndarray] | list["torch.Tensor"],
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
    ) -> tuple[list, list[tuple[int, int]]]:
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)

        if truncation:
            trunc_length = max_length
            if pad_to_multiple_of is not None and (trunc_length % pad_to_multiple_of != 0):
                trunc_length = ((trunc_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
            audio = [self._truncate_single(audio_el, max_length=trunc_length) for audio_el in audio]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(audio_el.shape[-1] for audio_el in audio)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        actual_lengths = [audio_el.shape[-1] for audio_el in audio]

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            audio = [self._pad_single(audio_el, max_length=max_length) for audio_el in audio]

        audio_ranges = []
        for i, length in enumerate(actual_lengths):
            padded_length = audio[i].shape[-1]
            if self.padding_side == "left":
                audio_ranges.append((padded_length - length, padded_length))
            else:
                audio_ranges.append((0, length))

        return audio, audio_ranges

    def _truncate_single(self, audio_el, max_length: int):
        return audio_el[..., :max_length] if audio_el.shape[-1] > max_length else audio_el

    def _pad_features(self, features, padding, max_length, truncation, pad_to_multiple_of):
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)
        if truncation and max_length is not None:
            features = [f[:max_length] for f in features]
        actual_lengths = [f.shape[0] for f in features]
        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(actual_lengths)
            padding_strategy = PaddingStrategy.MAX_LENGTH
        if max_length is not None and pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        if padding_strategy == PaddingStrategy.MAX_LENGTH and max_length is not None:
            features = [f if f.shape[0] >= max_length else self._pad_feature_single(f, max_length) for f in features]
        return features, [(0, length) for length in actual_lengths]

    def _pad_feature_single(self, feature, max_length):
        """Right-pad one feature array/tensor along its first (time) axis with `padding_value`."""
        return self._pad_axis(feature, 0, max_length - feature.shape[0], axis=0, value=self.padding_value)

    def _process_audio(self, audio_el):
        audio_el = self._as_backend_array(audio_el)
        if audio_el.ndim > 1:
            if self.force_mono and audio_el.shape[0] > 1:
                audio_el = self._mean_axis0(audio_el)
            elif audio_el.shape[0] == 1:
                audio_el = self._squeeze_axis0(audio_el)
            else:
                raise ValueError("Audio has more than one channel but force_mono is False")
        return audio_el

    def _to_batch(self, audio):
        batch = self._stack(audio)
        if self.add_channel_dim:
            batch = self._insert_channel_dim(batch)
        return batch

    def _pad_single(self, audio, max_length: int) -> AudioInput:
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio
        pad = max_length - current_length
        if self.padding_side == "right":
            left, right = 0, pad
        elif self.padding_side == "left":
            left, right = pad, 0
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")
        return self._pad_axis(audio, left, right, axis=-1, value=self.padding_value)

    def _stack_features(self, features):
        return self._stack(features)

    def _get_mask(self, ranges, padded_length):
        mask = self._zeros_int32((len(ranges), padded_length))
        for i, (start, end) in enumerate(ranges):
            mask[i, start:end] = 1
        return mask

    def _masked_mean_var_normalize(self, features, feature_lengths, epsilon=1e-5):
        """NeMo/Cohere-style per-utterance mean/variance normalization over the first
        `feature_lengths` frames of `features` (batch, frames, feature_dim), zeroing padded
        frames. `feature_lengths` is a CPU sequence/array of float32-representable counts."""
        xp = _array_namespace(features)
        lengths = self._astype(self._as_backend_array(np.asarray(feature_lengths)), "float32")
        mask = (xp.arange(features.shape[1])[None, :] < lengths[:, None])[..., None]
        masked = features * mask
        mean = (masked.sum(axis=1) / lengths[:, None])[:, None, :]
        variance = (((masked - mean) ** 2) * mask).sum(axis=1) / (lengths - 1)[:, None]
        std = xp.sqrt(variance)[:, None, :]
        return (features - mean) / (std + epsilon) * mask

    # ── Spectrogram core ─────────────────────────────────────────────────

    def extract_spectrogram(self, audio, *, spectrogram_config: SpectrogramConfig | None = None, **kwargs):
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        config_field_names = {f.name for f in fields(SpectrogramConfig)}
        overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in config_field_names}
        if overrides:
            spectrogram_config = replace(spectrogram_config, **overrides)

        norm_kwargs = {k: v for k, v in kwargs.items() if k not in ("audio_ranges", "feature_ranges")}

        if isinstance(audio, list):
            features = [self._extract_spectrogram(a, spectrogram_config=spectrogram_config, **kwargs) for a in audio]
            if spectrogram_config.mel_scale_config is not None:
                features = [
                    self._apply_mel_scale(f, spectrogram_config=spectrogram_config, **kwargs) for f in features
                ]
            features = [
                self._normalize_magnitude(f, spectrogram_config=spectrogram_config, **norm_kwargs) for f in features
            ]
        else:
            features = self._extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
            if spectrogram_config.mel_scale_config is not None:
                features = self._apply_mel_scale(features, spectrogram_config=spectrogram_config, **kwargs)
            features = self._normalize_magnitude(features, spectrogram_config=spectrogram_config, **norm_kwargs)

        return features

    def _extract_spectrogram(self, audio, *, spectrogram_config, **kwargs):
        return self._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _stft(self, audio, *, spectrogram_config, **kwargs):
        stft_cfg = spectrogram_config.stft_config
        needs_manual_framing = self._needs_manual_framing(spectrogram_config)
        if stft_cfg.frame_extension:
            if stft_cfg.frame_extension != 1:
                raise ValueError(f"Only frame_extension=1 is supported, got {stft_cfg.frame_extension}.")
            if stft_cfg.center is True:
                raise ValueError("frame_extension requires center=False or center='left', not symmetric centering.")
            if spectrogram_config.remove_dc_offset:
                raise ValueError("remove_dc_offset is not supported with frame_extension.")
        elif spectrogram_config.preemphasis_mode == "htk_per_frame":
            raise ValueError("preemphasis_mode='htk_per_frame' requires frame_extension=1.")
        if stft_cfg.fft_dtype is not None:
            if stft_cfg.fft_dtype not in ("float64", "native"):
                raise ValueError(f"fft_dtype must be None, 'float64' or 'native', got {stft_cfg.fft_dtype!r}.")
            if not needs_manual_framing:
                raise ValueError(
                    "fft_dtype applies to the manual-framing path only; this configuration uses the native STFT."
                )
        n_fft = stft_cfg.n_fft
        win_length = stft_cfg.win_length or n_fft
        hop_length = stft_cfg.hop_length or win_length // 2

        if spectrogram_config.computation_dtype and stft_cfg.fft_dtype is None:
            # with `fft_dtype`, the cast happens at the FFT boundary instead
            dtype_str = spectrogram_config.computation_dtype
            if isinstance(audio, np.ndarray):
                audio = audio.astype(dtype_str)
            else:
                import torch

                audio = audio.to(getattr(torch, dtype_str))
        if self.dither > 0:
            audio = self._apply_dither(audio, kwargs.get("audio_ranges"))
        if spectrogram_config.waveform_scale is not None:
            audio = audio * spectrogram_config.waveform_scale
        if spectrogram_config.preemphasis is not None and spectrogram_config.preemphasis_mode == "waveform":
            audio = self._preemphasize_waveform(audio, spectrogram_config.preemphasis, kwargs.get("audio_ranges"))

        # Cache window on first call; reuse on subsequent calls with same config
        if self._cached_stft_window is not None and spectrogram_config is self.spectrogram_config:
            window, frame_length = self._cached_stft_window
        else:
            window = self._create_stft_window(win_length, stft_cfg, audio)
            window, frame_length = self._prepare_window_and_framing(window, win_length, n_fft, needs_manual_framing)
            if spectrogram_config is self.spectrogram_config:
                self._cached_stft_window = (window, frame_length)

        if needs_manual_framing:
            audio_dtype = audio.dtype
            frames = self._frame_audio(
                audio, window, frame_length + stft_cfg.frame_extension, hop_length, n_fft, stft_cfg
            )
            frames = self._apply_frame_processing(frames, spectrogram_config=spectrogram_config, **kwargs)
            stft_out = self._window_and_fft(frames, window, frame_length, n_fft, stft_cfg, audio_dtype=audio_dtype)
        else:
            stft_out = self._native_stft(audio, window, frame_length, hop_length, n_fft, stft_cfg)

        magnitudes = self._compute_magnitudes(stft_out, stft_cfg.power, spectrogram_config=spectrogram_config)
        return self._cast_stft_output(magnitudes, spectrogram_config)

    # ── Spectrogram hooks ────────────────────────────────────────────────

    def _needs_manual_framing(self, spectrogram_config):
        """Whether the STFT requires manual framing (unfold-based) instead of a native STFT."""
        return (
            (
                spectrogram_config.preemphasis is not None
                and spectrogram_config.preemphasis_mode in ("per_frame", "htk_per_frame")
            )
            or spectrogram_config.remove_dc_offset
            or bool(spectrogram_config.stft_config.frame_extension)
            or spectrogram_config.stft_config.center == "left"  # truthy string would center-pad natively
        )

    def _cast_stft_output(self, magnitudes, spectrogram_config):
        """Cast STFT output to the desired output dtype. Default: no-op."""
        return magnitudes

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        """
        Convert raw audio sample lengths to the number of feature frames after spectrogram extraction.

        For centered STFT returns `audio_lengths // hop_length` (plus 1 when
        `include_center_frame=True`); for non-centered STFT returns the exact frame count
        `(audio_lengths - win_length) // hop_length + 1`.

        Override this method in subclasses that use non-standard STFT framing (e.g.,
        unfold-based with extra samples, or model-specific frame counting).
        """
        stft_cfg = spectrogram_config.stft_config
        win_length = stft_cfg.win_length or stft_cfg.n_fft
        hop_length = stft_cfg.hop_length or win_length // 2
        if stft_cfg.center == "left":
            lengths = (audio_lengths + win_length // 2 - (win_length + stft_cfg.frame_extension)) // hop_length + 1
            return max(0, lengths) if isinstance(lengths, int) else lengths.clip(min=0)
        if not stft_cfg.center:
            return (audio_lengths - win_length) // hop_length + 1
        lengths = audio_lengths // hop_length
        if include_center_frame:
            lengths = lengths + 1
        return lengths

    # ── Spectrogram backend ──────────────────────────────────────────────

    def _create_stft_window(self, win_length, stft_cfg, audio):
        raise NotImplementedError

    def _prepare_window_and_framing(self, window, win_length, n_fft, needs_manual_framing):
        if needs_manual_framing and win_length < n_fft:
            return window, win_length
        if win_length < n_fft:
            left_pad = (n_fft - win_length) // 2
            right_pad = n_fft - win_length - left_pad
            window = self._pad_axis(window, left_pad, right_pad, axis=-1, value=0.0)
        return window, n_fft

    def _frame_audio(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        """Extract overlapping frames from the audio signal.

        Handles center padding and dtype promotion. Returns frames of shape
        (..., num_frames, frame_length). Implemented by backend subclasses.
        """
        raise NotImplementedError

    def _apply_frame_processing(self, frames, *, spectrogram_config, **kwargs):
        """Hook: per-frame signal conditioning after frame extraction.

        Called after framing, before windowing and FFT. Applies DC-offset removal, per-frame
        (kaldi-style) preemphasis, and USM/HTK-style extended-frame preemphasis when
        ``stft_config.frame_extension`` is set. Override for non-standard frame processing
        that doesn't fit these knobs, e.g. boundary-frame masking (Phi4-multimodal).
        """
        if spectrogram_config.stft_config.frame_extension:
            # USM-style extended frames: preemphasis consumes the extra trailing sample,
            # reducing the frame back to `win_length`.
            preemphasis = spectrogram_config.preemphasis
            if preemphasis is None or preemphasis <= 0.0:
                return frames[..., :-1]
            if spectrogram_config.preemphasis_mode == "htk_per_frame":
                # HTK flavor: first sample scaled by (1 - p) instead of replicate-padded
                first = frames[..., :1] * (1.0 - preemphasis)
                rest = frames[..., 1:-1] - preemphasis * frames[..., :-2]
                return self._concat_last([first, rest])
            if spectrogram_config.preemphasis_mode == "per_frame":
                return frames[..., 1:] - preemphasis * frames[..., :-1]
            return frames[..., :-1]  # "waveform" mode: already applied upstream
        if spectrogram_config.remove_dc_offset:
            frames = frames - self._mean_last(frames)
        preemphasis = spectrogram_config.preemphasis
        if preemphasis is not None and spectrogram_config.preemphasis_mode == "per_frame":
            # Replicate-pad first sample (x0 - p*x0, not x0*(1-p)): bit-exact with kaldi.
            first = frames[..., :1] - preemphasis * frames[..., :1]
            rest = frames[..., 1:] - preemphasis * frames[..., :-1]
            frames = self._concat_last([first, rest])
        return frames

    def _preemphasize_waveform(self, audio, preemphasis, audio_ranges=None):
        """Waveform-level preemphasis (first sample unchanged), zeroing padded samples via
        ``audio_ranges``. Used when ``spectrogram_config.preemphasis_mode == "waveform"``
        (ASR models: Parakeet/Cohere/Nemotron). Implemented by backend subclasses."""
        raise NotImplementedError

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg, audio_dtype=None):
        """Apply window, zero-pad, FFT, and normalize. Returns complex STFT of shape (..., freq, time).
        Implemented by backend subclasses."""
        raise NotImplementedError

    def _apply_dither(self, audio, audio_ranges=None):
        """Additive dither, applied when ``self.dither`` is nonzero. Backend defaults add
        unseeded Gaussian noise; deterministic implementations (Cohere-ASR) override.
        ``audio_ranges`` is ``None`` on unbatched calls. Implemented by backend subclasses."""
        raise NotImplementedError

    def _native_stft(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        """Native STFT (e.g. torch.stft). Returns complex output. Implemented by backend subclasses."""
        raise NotImplementedError

    def _compute_magnitudes(self, stft_out, power, spectrogram_config=None):
        """Convert complex STFT output to a real-valued magnitude spectrogram.
        Implemented by backend subclasses. Overridable for custom magnitude computation (e.g. Parakeet)."""
        raise NotImplementedError

    def _apply_mel_scale(self, *args, **kwargs):
        """Apply mel filterbank to spectrogram features."""
        raise NotImplementedError

    def _normalize_magnitude(
        self, features, *, spectrogram_config, reference=1.0, min_value=1e-10, db_range=None, **kwargs
    ):
        log_mel = spectrogram_config.log_mode
        if log_mel is None:
            return self._astype(features, "float32")

        if spectrogram_config.pre_log_offset is not None:
            result = features + spectrogram_config.pre_log_offset
        else:
            result = _clamp_min(features, spectrogram_config.mel_floor)

        if log_mel == "log":
            result = self._astype(_array_namespace(result).log(result), "float32")
        elif log_mel == "log10":
            result = self._astype(_array_namespace(result).log10(result), "float32")
        elif log_mel == "dB":
            power = spectrogram_config.stft_config.power
            if power == 2.0:
                result = power_to_db(result, reference, min_value, db_range)
            elif power == 1.0:
                result = amplitude_to_db(result, reference, min_value, db_range)
            else:
                raise ValueError(f"Cannot use log_mel option 'dB' with power {power}")
            result = self._astype(result, "float32")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")

        if spectrogram_config.skip_last_frame:
            result = result[..., :-1]
        return self._apply_post_log_normalization(result, spectrogram_config)

    def _apply_post_log_normalization(self, result, spectrogram_config):
        if spectrogram_config.clip_max_offset is not None:
            max_vals = self._amax_over_features(result)
            result = _array_namespace(result).maximum(result, max_vals - spectrogram_config.clip_max_offset)
        if spectrogram_config.post_log_shift is not None:
            result = result + spectrogram_config.post_log_shift
        if spectrogram_config.post_log_scale is not None:
            result = result * spectrogram_config.post_log_scale
        return result

    # ── Backend array-API primitives ─────────────────────────────────────

    def _astype(self, x, dtype_name):
        raise NotImplementedError

    def _amax_over_features(self, x):
        raise NotImplementedError

    def _zeros_int32(self, shape):
        raise NotImplementedError

    def _as_backend_array(self, x):
        raise NotImplementedError

    def _mean_axis0(self, x):
        raise NotImplementedError

    def _squeeze_axis0(self, x):
        raise NotImplementedError

    def _pad_axis(self, x, left, right, axis, value=0.0):
        raise NotImplementedError

    def _stack(self, seq):
        raise NotImplementedError

    def _insert_channel_dim(self, batch):
        raise NotImplementedError

    def _mean_last(self, x):
        raise NotImplementedError

    def _concat_last(self, parts):
        raise NotImplementedError

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        """Build the mel filter bank described by ``spectrogram_config.mel_scale_config``.

        Backend-agnostic dispatcher: derives the geometry (number of frequency bins, FFT
        size, frequency range) and the computation dtype, then delegates the numerical
        construction to one of three backend leaves:

        - ``_kaldi_exact_mel_banks``: ``triangularize_in_mel_space`` with no zeroed bands
        - ``_kaldi_mel_banks_with_zero_bands``: ``triangularize_in_mel_space`` with ``bands_to_zero``
        - ``_standard_mel_banks``: standard triangular (librosa/torchaudio-style) filters

        Dtype policy: the dtype name is resolved as ``mel_scale_config.computation_dtype``
        falling back to the top-level ``spectrogram_config.computation_dtype`` (pipelines
        that run in float64 for legacy-FE parity need float64 filters as well), and passed
        to the leaves as a string (or None); each backend resolves it natively. When only
        the mel-level dtype is set, the filters are built in that dtype and cast back to
        the backend's default float afterwards, since the rest of the pipeline runs in
        default precision.
        """
        stft_cfg = spectrogram_config.stft_config
        mel_cfg = spectrogram_config.mel_scale_config
        n_fft = stft_cfg.n_fft
        num_frequency_bins = 1 + n_fft // 2
        min_frequency = mel_cfg.f_min
        max_frequency = mel_cfg.f_max if mel_cfg.f_max is not None else self.sampling_rate / 2
        computation_dtype = mel_cfg.computation_dtype or spectrogram_config.computation_dtype

        if mel_cfg.triangularize_in_mel_space and mel_cfg.bands_to_zero == 0:
            mel_filters = self._kaldi_exact_mel_banks(
                mel_cfg.n_mels,
                num_frequency_bins,
                min_frequency,
                max_frequency,
                self.sampling_rate,
                n_fft,
                mel_cfg,
                computation_dtype,
            )
        elif mel_cfg.triangularize_in_mel_space:
            mel_filters = self._kaldi_mel_banks_with_zero_bands(
                mel_cfg.n_mels,
                num_frequency_bins,
                min_frequency,
                max_frequency,
                self.sampling_rate,
                n_fft,
                mel_cfg,
                computation_dtype,
            )
        else:
            mel_filters = self._standard_mel_banks(
                mel_cfg.n_mels,
                num_frequency_bins,
                min_frequency,
                max_frequency,
                self.sampling_rate,
                n_fft,
                mel_cfg,
                computation_dtype,
            )

        # Cast back when only the mel-level dtype requested higher precision.
        if mel_cfg.computation_dtype is not None and not spectrogram_config.computation_dtype:
            mel_filters = self._cast_mel_filters_to_default_float(mel_filters)
        return mel_filters

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
        """Mel filter bank triangularized in mel space, without zeroed bands.

        Each backend leaf owns its ecosystem's rounding pattern: the torch leaf matches
        ``torchaudio.compliance.kaldi.get_mel_banks`` arithmetic, the numpy leaf matches
        the legacy numpy feature extractors. Numerically equivalent, not bit-identical.
        Implemented by backend subclasses."""
        raise NotImplementedError

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
        """Mel filter bank triangularized in mel space with the lowest ``bands_to_zero``
        frequency bins zeroed out. Implemented by backend subclasses."""
        raise NotImplementedError

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
        """Standard (non-kaldi) triangular mel filter bank. Each backend leaf owns its
        ecosystem's rounding pattern (numpy: librosa/legacy numpy FEs; torch: torchaudio).
        Implemented by backend subclasses."""
        raise NotImplementedError

    def _cast_mel_filters_to_default_float(self, mel_filters):
        """Cast a built filter bank to the backend's default float dtype (torch:
        ``torch.get_default_dtype()``, numpy: float32). Implemented by backend subclasses."""
        raise NotImplementedError
