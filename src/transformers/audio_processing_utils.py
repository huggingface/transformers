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

from huggingface_hub.dataclasses import validate_typed_dict

from .audio_processing_base import AudioProcessingMixin
from .audio_utils import AudioInput, SpectrogramConfig, make_list_of_audio, mel_filter_bank
from .feature_extraction_utils import BatchFeature
from .processing_utils import AudioKwargs
from .utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


class AudioProcessingKwargs(AudioKwargs, total=False):
    """Extended keyword arguments for the audio processing pipeline."""

    do_pad_values: bool | None
    do_values_normalize: bool | None
    spectrogram_config: dict | SpectrogramConfig | None
    do_extract_spectrogram: bool | None
    do_feature_normalize: bool | None
    do_pad_features: bool | None
    do_resample: bool | None


class BaseAudioProcessor(AudioProcessingMixin):
    model_input_names = ["audio"]
    valid_kwargs = AudioProcessingKwargs
    unused_kwargs = None
    feature_size = 1
    padding = True
    padding_side = "right"
    padding_value = 0.0
    max_length = None
    truncation = None
    return_attention_mask = True

    sample_rate: int = None
    force_mono: bool = None

    # Pipeline stage defaults
    do_pad_values = None
    do_values_normalize = None
    normalize_before_pad = True
    spectrogram_config = None
    do_extract_spectrogram = None
    do_feature_normalize = None
    feature_normalize_before_pad = True
    do_pad_features = None
    do_resample = False
    add_channel_dim = False
    pad_to_multiple_of = None
    transpose_features = False

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

    def __call__(self, audio: AudioInput, *args, **kwargs: Unpack[AudioProcessingKwargs]) -> BatchFeature:
        return self.preprocess(audio, *args, **kwargs)

    def preprocess(self, audio: AudioInput, *args, **kwargs: Unpack[AudioProcessingKwargs]) -> BatchFeature:
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
        **kwargs: Unpack[AudioProcessingKwargs],
    ) -> BatchFeature:
        audio = self._prepare_audio_like_inputs(audio=audio, sample_rate=sample_rate)
        return self._preprocess(audio, *args, **kwargs)

    def _preprocess(self, *args, **kwargs):
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
    ):
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

        if padding_strategy != PaddingStrategy.DO_NOT_PAD:
            audio = [self._pad_single(audio_el, max_length=max_length) for audio_el in audio]

        return audio

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

    def extract_spectrogram(self, audio, *, do_batch_spectrogram: bool = True, spectrogram_config: SpectrogramConfig | None = None, **kwargs):
        """
        Both the numpy and torch backends implement this method in a batched/ sequential manner.
        Is is batched by default, but can be set to be sequential.
        This can extract just a spectrogram or a Mel spectrogram if a mel config is provided.

        Any extra kwargs whose names match ``SpectrogramConfig`` fields will
        override the corresponding value on the config for this call.
        """
        if spectrogram_config is None:
            spectrogram_config = self.spectrogram_config

        config_field_names = {f.name for f in fields(SpectrogramConfig)}
        overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in config_field_names}
        if overrides:
            spectrogram_config = replace(spectrogram_config, **overrides)

        if do_batch_spectrogram:
            features = self._extract_spectrogram(audio, spectrogram_config=spectrogram_config, **kwargs)
            if spectrogram_config.mel_scale_config is not None:
                features = self._apply_mel_scale(features, spectrogram_config=spectrogram_config, **kwargs)
        else:
            features = [self._extract_spectrogram(audio_el, spectrogram_config=spectrogram_config, **kwargs) for audio_el in audio]
            if spectrogram_config.mel_scale_config is not None:
                features = [self._apply_mel_scale(feature_el, spectrogram_config=spectrogram_config, **kwargs) for feature_el in features]
        return features

    def _extract_spectrogram(self, *args, **kwargs):
        """
        Compute the (power) spectrogram via STFT.

        Implemented by backend subclasses (e.g., ``TorchAudioBackend``).
        """
        raise NotImplementedError

    def _apply_mel_scale(self, *args, **kwargs):
        """
        Apply mel filterbank to a spectrogram.

        Implemented by backend subclasses (e.g., ``TorchAudioBackend``).
        """
        raise NotImplementedError

    def _mel_filter_bank(self, spectrogram_config: SpectrogramConfig):
        raise NotImplementedError

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
