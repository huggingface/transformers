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
from typing import Unpack

import numpy as np
from huggingface_hub.dataclasses import validate_typed_dict

from .audio_processing_base import AudioProcessingMixin
from .audio_utils import AudioInput, SpectrogramConfig, make_list_of_audio
from .feature_extraction_utils import BatchFeature
from .tokenization_utils_base import PaddingStrategy, TruncationStrategy
from .processing_utils import AudioKwargs
from .utils import PaddingStrategy, TensorType, logging

from typing import TypedDict


logger = logging.get_logger(__name__)


class AudioKwargs(TypedDict, total=False):
    sampling_rate: int | None
    spectrogram_config: dict | SpectrogramConfig | None
    do_extract_spectrogram: bool | None
    do_resample: bool | None
    return_tensors: str | TensorType | None
    padding: bool | str | PaddingStrategy | None
    max_length: int | None
    truncation: bool | str | TruncationStrategy | None
    pad_to_multiple_of: int | None


class BaseAudioProcessor(AudioProcessingMixin):
    model_input_names = ["audio"]
    valid_kwargs = AudioKwargs
    unused_kwargs = None

    # global defaults
    sample_rate: int = None
    force_mono: bool = None
    add_channel_dim: bool = False

    # padding defaults
    padding = True
    padding_side = "right"
    padding_value = 0.0
    max_length = None
    truncation = None
    pad_to_multiple_of = None

    return_padding_mask = True
    mask_level = None  # None = auto (features for spectrogram, audio for raw), "audio" = always audio-level
    spectrogram_config = None
    do_extract_spectrogram = None

    def __init__(
        self,
        sample_rate: int | None = None,
        force_mono: bool | None = None,
        **kwargs,
    ):
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if self.sample_rate is None:
            raise ValueError(
                f"`sample_rate` must be set either as a class attribute on {self.__class__.__name__} "
                "or passed to __init__."
            )

        if force_mono is not None:
            self.force_mono = force_mono
        if self.force_mono is None:
            raise ValueError(
                f"`force_mono` must be set either as a class attribute on {self.__class__.__name__} "
                "or passed to __init__."
            )

        super().__init__(**kwargs)

        # Standardize init attributes (coerce dicts to config dataclasses)
        attributes = {key: getattr(self, key) for key in self._valid_kwargs_names}
        attributes = self._standardize_kwargs(**attributes)
        for key, value in attributes.items():
            setattr(self, key, value)

        # Derive mel_filters from spectrogram_config if mel_scale_config is set
        # TODO: maybe the mel spectrogram initialization should be lazy?
        if self.spectrogram_config is not None and self.spectrogram_config.mel_scale_config is not None:
            if not hasattr(self, "mel_filters"):
                self.mel_filters = self._mel_filter_bank(self.spectrogram_config)

    def __call__(self, audio: AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
        return self.preprocess(audio, *args, **kwargs)

    def preprocess(self, audio: AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
        """
        Preprocess an audio or a batch of audio.
        """
        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self.
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Standardize kwargs (coerce dicts to config dataclasses)
        kwargs = self._standardize_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        return self._preprocess_audio_like_inputs(audio, *args, **kwargs)

    def _preprocess_audio_like_inputs(
        self,
        audio: AudioInput,
        *args,
        sample_rate: int | None = None,
        **kwargs: Unpack[AudioKwargs],
    ) -> BatchFeature:
        audio = self._prepare_audio_like_inputs(audio=audio, sample_rate=sample_rate)
        return self._preprocess(audio, *args, **kwargs)

    def _to_batch(self, audio):
        """Stack a list of audio arrays/tensors into a batch. Implemented by backend subclasses."""
        raise NotImplementedError

    def _get_mask(self, audio_ranges, padded_length, do_extract_spectrogram, spectrogram_config):
        """Build attention mask dict from audio_ranges. Returns a dict of {key: mask} to merge into output.
        Implemented by backend subclasses."""
        raise NotImplementedError

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
        do_batch_spectrogram=None,
        **kwargs,
    ) -> BatchFeature:
        if do_batch_spectrogram is None:
            do_batch_spectrogram = getattr(self, "do_batch_spectrogram", True)
        if do_extract_spectrogram and not do_batch_spectrogram:
            # Per-waveform extraction path: extract → postprocess → pad features → mask
            features = self.extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
            feature_lengths = [f.shape[0] for f in features]
            features = self._postprocess_features(features, feature_lengths)
            features, feature_ranges = self._pad_features(
                features, padding, max_length, truncation, pad_to_multiple_of
            )
            output = {"audio_features": self._stack_features(features)}
            if self.return_padding_mask:
                padded_length = features[0].shape[0]
                output.update(self._get_feature_mask(feature_ranges, padded_length))
            output = self._postprocess_output(output, feature_ranges=feature_ranges, **kwargs)
        else:
            # Standard path: pad audio → optionally batch → extract/passthrough
            audio, audio_ranges = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)
            padded_length = audio[0].shape[-1]

            if do_extract_spectrogram:
                audio = self._to_batch(audio) if do_batch_spectrogram else audio
                feature = self.extract_spectrogram(audio, spectrogram_config=spectrogram_config, audio_ranges=audio_ranges, **kwargs)
                output = {"audio_features": feature}
            else:
                output = {"audio_values": self._to_batch(audio)}

            if self.return_padding_mask:
                output.update(self._get_mask(
                    audio_ranges, padded_length, do_extract_spectrogram=do_extract_spectrogram, spectrogram_config=spectrogram_config
                ))
            output = self._postprocess_output(output, audio_ranges=audio_ranges, **kwargs)

        return BatchFeature(data=output, tensor_type=return_tensors)

    def _postprocess_features(self, features, feature_lengths):
        """Hook: per-utterance feature processing after extraction, before feature-level padding.

        Override for normalization that must happen on unpadded features
        (e.g., SeamlessM4t mean/variance normalization).
        """
        return features

    def _postprocess_output(self, output, audio_ranges=None, feature_ranges=None, **kwargs):
        """Hook: augment or modify the output dict after main processing.

        Override to add custom fields (e.g., audio_embed_sizes) or
        post-hoc normalization on the stacked/batched output.
        """
        return output

    def _pad_features(self, features, padding, max_length, truncation, pad_to_multiple_of):
        """Pad a list of 2D feature arrays along the time axis (axis 0).
        Implemented by backend subclasses."""
        raise NotImplementedError

    def _stack_features(self, features):
        """Stack a list of feature arrays/tensors into a batch.
        Implemented by backend subclasses."""
        raise NotImplementedError

    def _get_feature_mask(self, feature_ranges, padded_length):
        """Build attention mask dict from feature_ranges.
        Implemented by backend subclasses."""
        raise NotImplementedError

    def _prepare_audio_like_inputs(self, audio: AudioInput, *args, sample_rate: int | None = None, **kwargs) -> list:
        """
        Prepare audio-like inputs for processing by structuring and then converting each
        audio item via `process_audio`.

        Analogous to `_prepare_image_like_inputs` in the image processing pipeline.
        """
        audio = self._prepare_audio_structure(audio, sample_rate=sample_rate)
        audio = [self.process_audio(audio_el) for audio_el in audio]
        return audio

    def _prepare_audio_structure(self, audio: AudioInput, sample_rate: int | None = None) -> list:
        """
        Prepare the audio structure for processing: fetch URL inputs, validate sample rate,
        and flatten into a list of audio arrays.

        Analogous to `_prepare_images_structure` in the image processing pipeline.
        """
        is_url_input = isinstance(audio, str) or (
            isinstance(audio, (list, tuple)) and all(isinstance(el, str) for el in audio)
        )

        if is_url_input:
            # URL inputs: load directly at the correct sample rate
            audio = self.fetch_audio(audio)
        else:
            # Array inputs: validate that the user-provided sample rate matches the model's
            if sample_rate is not None:
                if sample_rate != self.sample_rate:
                    raise ValueError(
                        f"The model corresponding to this audio processor: {self.__class__.__name__} was trained using a"
                        f" sample rate of {self.sample_rate}. Please make sure that the provided `audio` input"
                        f" was sampled with {self.sample_rate} and not {sample_rate}."
                    )
            else:
                logger.warning(
                    f"It is strongly recommended to pass the `sample_rate` argument to `{self.__class__.__name__}()`. "
                    "Failing to do so can result in silent errors that might be hard to debug."
                )

        audio = make_list_of_audio(audio)
        return audio

    def _process_audio(self, *args, **kwargs):
        """
        Process a single raw audio input into the backend's working format.

        Implemented by backend subclasses (e.g., `TorchAudioBackend`). Converts a raw input
        (NumPy array) to the backend's internal format (e.g., `torch.Tensor`), handles
        mono conversion if needed.
        """
        raise NotImplementedError

    def process_audio(self, *args, **kwargs):
        return self._process_audio(*args, **kwargs)

    def pad(
        self,
        audio: AudioInput, # TODO: this type makes it unclear to know the have an iterable
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
    ) -> tuple[list, list[tuple[int, int]]]:
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)

        if truncation:
            if max_length is None:
                # TODO: maybe this check should happen in the _validate_preprocess_kwargs method
                raise ValueError("When setting `truncation=True`, make sure that `max_length` is defined.")
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
        """Truncate a single audio element to max_length along the time axis."""
        if audio_el.shape[-1] > max_length:
            return audio_el[..., :max_length]
        return audio_el

    def _pad_single(self, audio, max_length: int) -> AudioInput:
        """
        Pad a single input (on left/right) up to predefined length or max length in the batch.

        Implemented by backend subclasses.
        """
        raise NotImplementedError

    def extract_spectrogram(self, audio, *, spectrogram_config: SpectrogramConfig | None = None, **kwargs):
        """
        Extract spectrogram features from audio.

        Both the numpy and torch backends implement this method in a batched/sequential manner.
        It is batched by default, but can be set to be sequential.
        This can extract just a spectrogram or a Mel spectrogram if a mel config is provided.

        Any extra kwargs whose names match ``SpectrogramConfig`` fields will
        override the corresponding value on the config for this call.

        Note: Models that bypass the base STFT pipeline entirely (e.g., GraniteSpeech
        using torchaudio.transforms.MelSpectrogram, or MusicgenMelody using chroma
        features) can set ``do_extract_spectrogram=True`` without providing a
        ``spectrogram_config``. They must override this method completely.
        """
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        config_field_names = {f.name for f in fields(SpectrogramConfig)}
        overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in config_field_names}
        if overrides:
            spectrogram_config = replace(spectrogram_config, **overrides)

        if isinstance(audio, list):
            features = [
                self._extract_spectrogram(a, spectrogram_config=spectrogram_config, **kwargs)
                for a in audio
            ]
            if spectrogram_config.mel_scale_config is not None:
                features = [
                    self._apply_mel_scale(f, spectrogram_config=spectrogram_config, **kwargs)
                    for f in features
                ]
            features = [
                self._normalize_magnitude(f, spectrogram_config=spectrogram_config, **kwargs)
                for f in features
            ]
        else:
            features = self._extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
            if spectrogram_config.mel_scale_config is not None:
                features = self._apply_mel_scale(features, spectrogram_config=spectrogram_config, **kwargs)
            features = self._normalize_magnitude(features, spectrogram_config=spectrogram_config, **kwargs)

        return features

    # ── Spectrogram extraction pipeline ──────────────────────────────────
    #
    # The full feature-extraction pipeline executed by `extract_spectrogram`:
    #
    #   1. _extract_spectrogram   (STFT → power/magnitude spectrogram)
    #      a. _stft                        – orchestrates steps b–g (overridable for fully custom STFTs)
    #      b.   _needs_manual_framing      – decide framing strategy (hook)
    #      c.   _create_stft_window        – create the STFT window (backend)
    #      d.   _prepare_window_and_framing– pad/reshape window, decide frame length (backend)
    #      e.   manual path (needs_manual_framing=True):
    #             _frame_audio             – center pad + frame extraction (backend)
    #             _apply_frame_processing  – per-frame conditioning (hook)
    #             _window_and_fft          – window + zero-pad + FFT + normalize → complex (backend)
    #           native path (needs_manual_framing=False):
    #             _native_stft             – native STFT returning complex output (backend)
    #      f.   _compute_magnitudes        – complex → real magnitudes (backend, shared by both paths)
    #      g.   _cast_stft_output          – cast output dtype (hook, no-op by default)
    #   2. _apply_mel_scale       (mel filterbank projection)
    #   3. _normalize_magnitude   (log / dB scaling, optional per-utterance norm)
    #
    # Backend subclasses (NumpyAudioBackend, TorchAudioBackend) implement the
    # full pipeline.  Model-specific processors can override individual hooks
    # (_apply_frame_processing) or the entire _stft when the base STFT path
    # is insufficient.
    #
    # ``audio_ranges`` is passed through as a kwarg from ``_preprocess`` so that
    # model-specific overrides (e.g., Parakeet waveform-level preemphasis,
    # Phi4 boundary masking) can access original audio lengths without stashing
    # state on ``self``.

    def _extract_spectrogram(self, audio, *, spectrogram_config, **kwargs):
        """Orchestrate the STFT pipeline.

        Runs the sub-steps listed above in order. Override this only when the
        pipeline ordering itself needs to change. Otherwise, override individual hooks.
        """
        return self._stft(audio, spectrogram_config=spectrogram_config, **kwargs)

    def _stft(self, audio, *, spectrogram_config, **kwargs):
        """Compute the STFT and return a power/magnitude spectrogram.

        Orchestrates the sub-steps listed in the pipeline documentation above.
        Backend subclasses implement the individual leaf methods; model-specific
        processors can override this entirely for a fully custom STFT
        (e.g., Gemma3n's unfold-based STFT with extra-sample framing).
        """
        stft_cfg = spectrogram_config.stft_config
        n_fft = stft_cfg.n_fft
        win_length = stft_cfg.win_length or n_fft
        hop_length = stft_cfg.hop_length or win_length // 2
        needs_manual_framing = self._needs_manual_framing(spectrogram_config)

        if spectrogram_config.computation_dtype:
            dtype_str = spectrogram_config.computation_dtype
            if isinstance(audio, np.ndarray):
                audio = audio.astype(dtype_str)
            else:
                import torch
                audio = audio.to(getattr(torch, dtype_str))
        if spectrogram_config.waveform_scale is not None:
            audio = audio * spectrogram_config.waveform_scale
        window = self._create_stft_window(win_length, stft_cfg, audio)
        window, frame_length = self._prepare_window_and_framing(window, win_length, n_fft, needs_manual_framing)

        if needs_manual_framing:
            frames = self._frame_audio(audio, window, frame_length, hop_length, n_fft, stft_cfg)
            frames = self._apply_frame_processing(frames, spectrogram_config=spectrogram_config, **kwargs)
            stft_out = self._window_and_fft(frames, window, frame_length, n_fft, stft_cfg)
        else:
            stft_out = self._native_stft(audio, window, frame_length, hop_length, n_fft, stft_cfg)

        magnitudes = self._compute_magnitudes(stft_out, stft_cfg.power)
        return self._cast_stft_output(magnitudes, spectrogram_config)

    def _create_stft_window(self, win_length, stft_cfg, audio):
        """Create the STFT window. Implemented by backend subclasses."""
        raise NotImplementedError

    def _prepare_window_and_framing(self, window, win_length, n_fft, needs_manual_framing):
        """Pad/reshape window and determine frame length. Implemented by backend subclasses."""
        raise NotImplementedError

    def _frame_audio(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        """Extract overlapping frames from the audio signal.

        Handles center padding and dtype promotion. Returns frames of shape
        (..., num_frames, frame_length). Implemented by backend subclasses.
        """
        raise NotImplementedError

    def _window_and_fft(self, frames, window, frame_length, n_fft, stft_cfg):
        """Apply window, zero-pad, FFT, and normalize. Returns complex STFT of shape (..., freq, time).
        Implemented by backend subclasses."""
        raise NotImplementedError

    def _native_stft(self, audio, window, frame_length, hop_length, n_fft, stft_cfg):
        """Native STFT (e.g. torch.stft). Returns complex output. Implemented by backend subclasses."""
        raise NotImplementedError

    def _compute_magnitudes(self, stft_out, power):
        """Convert complex STFT output to a real-valued magnitude spectrogram.
        Implemented by backend subclasses. Overridable for custom magnitude computation (e.g. Parakeet)."""
        raise NotImplementedError

    def _cast_stft_output(self, magnitudes, spectrogram_config):
        """Cast STFT output to the desired output dtype. Default: no-op."""
        return magnitudes

    def _needs_manual_framing(self, spectrogram_config):
        """Whether the STFT requires manual framing (unfold-based) instead of a native STFT.

        Manual framing is needed when per-frame processing must happen between
        frame extraction and windowing (e.g. per-frame preemphasis, DC offset removal,
        or left-aligned FFT padding).

        Override in model-specific processors that handle preemphasis at the
        waveform level (in ``_stft``) and don't need per-frame processing.
        """
        return (
            (spectrogram_config.preemphasis is not None)
            or spectrogram_config.remove_dc_offset
        )

    def _compute_magnitudes(self, stft_out, power):
        """Convert complex STFT output to a real-valued magnitude spectrogram.

        Only used in the non-manual-framing STFT path.  Override for
        non-standard magnitude computation (e.g. Parakeet's view_as_real path).
        """
        raise NotImplementedError

    def _apply_frame_processing(self, frames, *, spectrogram_config, **kwargs):
        """Hook: per-frame signal conditioning after frame extraction.

        Called after framing, before windowing and FFT. Default backend
        implementations apply dither, DC-offset removal, and standard
        preemphasis.

        Override for non-standard frame processing, e.g. HTK-style
        preemphasis (Gemma3n).
        """
        raise NotImplementedError

    def _apply_mel_scale(self, *args, **kwargs):
        """Apply mel filterbank to spectrogram features."""
        raise NotImplementedError

    def _normalize_magnitude(self, *args, **kwargs):
        """Apply magnitude normalization (log, log10, or dB scaling) to spectrogram features."""
        raise NotImplementedError

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        raise NotImplementedError

    def _get_features_lengths(self, audio_lengths, spectrogram_config, include_center_frame=False):
        """
        Convert raw audio sample lengths to the number of feature frames after spectrogram extraction.

        By default returns `audio_lengths // hop_length`, which gives the number of valid (non-padding)
        feature frames for centered STFT. When `include_center_frame=True` and the STFT uses centering,
        adds 1 to account for the extra frame produced by centered STFT.

        Override this method in subclasses that use non-standard STFT configurations (e.g., unfold-based
        or non-centered STFT).
        """
        hop_length = spectrogram_config.stft_config.hop_length
        lengths = audio_lengths // hop_length
        if include_center_frame and spectrogram_config.stft_config.center:
            lengths = lengths + 1
        return lengths

    def _get_padding_strategies(self, padding=False, max_length=None):
        """Find the correct padding strategy."""
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

    def _standardize_kwargs(
        self,
        **kwargs,
    ) -> dict:
        """Coerce dict configs to their dataclass form."""
        if isinstance(kwargs.get("spectrogram_config"), dict):
            kwargs["spectrogram_config"] = SpectrogramConfig.from_dict(
                kwargs["spectrogram_config"]
            )
        if kwargs.get("spectrogram_config") is not None and kwargs.get("do_extract_spectrogram") is None:
            kwargs["do_extract_spectrogram"] = True
        return kwargs

    def _validate_preprocess_kwargs(
        self,
        sample_rate: int | None = None,
        max_length: int | None = None,
        truncation: bool | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        """Validate the kwargs for the preprocess method."""
        if truncation and max_length is None:
            raise ValueError(
                "When setting `truncation=True`, make sure that `max_length` is defined."
            )

    def to_dict(self):
        output = super().to_dict()
        # Serialize config dataclasses to plain dicts for JSON persistence
        for key in ("spectrogram_config",):
            if key in output and hasattr(output[key], "to_dict"):
                output[key] = output[key].to_dict()

        # Filter out None values that are class defaults
        filtered_dict = {}
        for key, value in output.items():
            if value is None:
                class_default = getattr(type(self), key, "NOT_FOUND")
                # Keep None if user explicitly set it (class default is non-None)
                if class_default != "NOT_FOUND" and class_default is not None:
                    filtered_dict[key] = value
            else:
                filtered_dict[key] = value

        return filtered_dict
