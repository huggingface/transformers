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

from typing import Unpack

from huggingface_hub.dataclasses import validate_typed_dict

from .audio_processing_base import AudioProcessingMixin
from .audio_utils import AudioInput, SpectrogramConfig, NormalizationConfig, make_list_of_audio, mel_filter_bank
from .feature_extraction_utils import BatchFeature
from .processing_utils import AudioKwargs
from .utils import TensorType, logging


logger = logging.get_logger(__name__)


class AudioProcessingKwargs(AudioKwargs, total=False):
    """Extended keyword arguments for the audio processing pipeline."""

    do_pad_values: bool | None
    do_values_normalize: bool | None
    normalization_config: dict | NormalizationConfig | None
    spectrogram_config: dict | SpectrogramConfig | None
    do_feature_normalize: bool | None
    feature_normalization_config: dict | NormalizationConfig | None
    do_pad_features: bool | None
    do_resample: bool | None


class BaseAudioProcessor(AudioProcessingMixin):
    model_input_names = ["audio"]
    valid_kwargs = AudioProcessingKwargs
    unused_kwargs = None
    padding = True
    padding_side = "right"
    padding_value = 0.0
    max_length = None
    truncation = None

    sample_rate: int = None
    force_mono: bool = None

    # Pipeline stage defaults
    do_pad_values = None
    do_values_normalize = None
    normalization_config = None
    spectrogram_config = None
    do_feature_normalize = None
    feature_normalization_config = None
    do_pad_features = None
    do_resample = False

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

        # Derive max_length and mel_filters from spectrogram_config
        if self.spectrogram_config is not None:
            sc = self.spectrogram_config 
            if not hasattr(self, "mel_filters"):
                self.mel_filters = mel_filter_bank(
                    num_frequency_bins=1 + sc.stft_config.n_fft // 2,
                    num_mel_filters=sc.mel_scale_config.n_mels,
                    min_frequency=sc.mel_scale_config.f_min,
                    max_frequency=sc.mel_scale_config.f_max if sc.mel_scale_config.f_max is not None else self.sample_rate / 2,
                    sampling_rate=self.sample_rate,
                    norm=sc.mel_scale_config.norm,
                    mel_scale=sc.mel_scale_config.mel_scale,
                )

    def __call__(self, audio: AudioInput, *args, **kwargs: Unpack[AudioProcessingKwargs]) -> BatchFeature:
        return self.preprocess(audio, *args, **kwargs)

    def process_audio(self, *args, **kwargs):
        """
        Process a single raw audio input into the backend's working format.

        Implemented by backend subclasses (e.g., `TorchAudioBackend`). Converts a raw input
        (NumPy array) to the backend's internal format (e.g., `torch.Tensor`), handles
        mono conversion if needed.
        """
        raise NotImplementedError

    def _preprocess(self, *args, **kwargs):
        """
        Perform the actual batch audio preprocessing pipeline (stages 3-7).

        Implemented by backend subclasses (e.g., `TorchAudioBackend`). Receives a list of
        already-prepared audio tensors and applies the configured preprocessing operations.
        Returns a `BatchFeature` with the processed audio values.
        """
        raise NotImplementedError

    def pad(self, *args, **kwargs):
        """
        Pad a single audio tensor to a target length.

        Implemented by backend subclasses (e.g., `TorchAudioBackend`).
        """
        raise NotImplementedError

    def pad_values(self, *args, **kwargs):
        """
        Pad raw audio values to a target length (pipeline stage 3).

        Implemented by backend subclasses.
        """
        raise NotImplementedError

    def values_normalize(self, *args, **kwargs):
        """
        Normalize raw audio values (pipeline stage 4).

        Implemented by backend subclasses.
        """
        raise NotImplementedError

    def extract_spectrogram(self, *args, **kwargs):
        """
        Extract spectrogram from audio (pipeline stage 5).

        Implemented by model-specific processor subclasses.
        """
        raise NotImplementedError

    def feature_normalize(self, *args, **kwargs):
        """
        Normalize extracted features (pipeline stage 6).

        Implemented by backend subclasses.
        """
        raise NotImplementedError

    def pad_features(self, *args, **kwargs):
        """
        Pad extracted features to a target length (pipeline stage 7).

        Implemented by backend subclasses.
        """
        raise NotImplementedError

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

    def _standardize_kwargs(
        self,
        **kwargs,
    ) -> dict:
        """Coerce dict configs to their dataclass form."""
        if isinstance(kwargs.get("normalization_config"), dict):
            kwargs["normalization_config"] = NormalizationConfig.from_dict(kwargs["normalization_config"])
        if isinstance(kwargs.get("spectrogram_config"), dict):
            kwargs["spectrogram_config"] = SpectrogramConfig.from_dict(
                kwargs["spectrogram_config"]
            )
        if isinstance(kwargs.get("feature_normalization_config"), dict):
            kwargs["feature_normalization_config"] = NormalizationConfig.from_dict(
                kwargs["feature_normalization_config"]
            )
        return kwargs

    def _validate_preprocess_kwargs(
        self,
        sample_rate: int | None = None,
        max_length: int | None = None,
        truncation: bool | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        do_values_normalize: bool | None = None,
        normalization_config: NormalizationConfig | None = None,
        do_feature_normalize: bool | None = None,
        feature_normalization_config: NormalizationConfig | None = None,
        **kwargs,
    ):
        """Validate the kwargs for the preprocess method."""
        if do_values_normalize and normalization_config is None:
            raise ValueError(
                "`do_values_normalize=True` requires `normalization_config` to be set."
            )
        if do_feature_normalize and feature_normalization_config is None:
            raise ValueError(
                "`do_feature_normalize=True` requires `feature_normalization_config` to be set."
            )
        if truncation and max_length is None:
            raise ValueError(
                "When setting `truncation=True`, make sure that `max_length` is defined."
            )

    def _preprocess_audio_like_inputs(
        self,
        audio: AudioInput,
        *args,
        sample_rate: int | None = None,
        **kwargs: Unpack[AudioProcessingKwargs],
    ) -> BatchFeature:
        audio = self._prepare_audio_like_inputs(audio=audio, sample_rate=sample_rate)
        return self._preprocess(audio, *args, **kwargs)

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

    def _prepare_audio_like_inputs(self, audio: AudioInput, *args, sample_rate: int | None = None, **kwargs) -> list:
        """
        Prepare audio-like inputs for processing by structuring and then converting each
        audio item via `process_audio`.

        Analogous to `_prepare_image_like_inputs` in the image processing pipeline.
        """
        audio = self._prepare_audio_structure(audio, sample_rate=sample_rate)
        audio = [self.process_audio(audio_el) for audio_el in audio]
        return audio

    def to_dict(self):
        output = super().to_dict()
        # Serialize config dataclasses to plain dicts for JSON persistence
        for key in ("normalization_config", "spectrogram_config", "feature_normalization_config"):
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
