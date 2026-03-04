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

from functools import lru_cache
from typing import Unpack

from huggingface_hub.dataclasses import validate_typed_dict

from .audio_processing_base import AudioProcessingMixin
from .audio_utils import AudioInput, make_list_of_audio
from .feature_extraction_utils import BatchFeature
from .processing_utils import AudioKwargs
from .utils import TensorType, logging


logger = logging.get_logger(__name__)


class BaseAudioProcessor(AudioProcessingMixin):
    model_input_names = ["audio"]
    valid_kwargs = AudioKwargs
    unused_kwargs = None
    padding = True
    padding_side = "right"
    padding_value = 0.0

    def __init__(
        self,
        sample_rate: int,
        force_mono: bool,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.force_mono = force_mono

        super().__init__(**kwargs)

    def __call__(self, audio: AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
        return self.preprocess(audio, *args, **kwargs)

    def process_audio(self, *args, **kwargs):
        """
        Process a single raw audio input into the backend's working format.

        Implemented by backend subclasses (e.g., `TorchBackend`). Converts a raw input
        (NumPy array) to the backend's internal format (e.g., `torch.Tensor`), handles
        mono conversion if needed.
        """
        raise NotImplementedError

    def _preprocess(self, *args, **kwargs):
        """
        Perform the actual batch audio preprocessing (truncation, padding, stacking).

        Implemented by backend subclasses (e.g., `TorchBackend`). Receives a list of
        already-prepared audio tensors and applies the configured preprocessing operations.
        Returns a `BatchFeature` with the processed audio values.
        """
        raise NotImplementedError

    def pad(self, *args, **kwargs):
        """
        Pad a single audio tensor to a target length.

        Implemented by backend subclasses (e.g., `TorchBackend`).
        """
        raise NotImplementedError

    def preprocess(self, audio: AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
        # args are not validated, but their order in the `preprocess` and `_preprocess` signatures must be the same

        # Perform type validation on received kwargs
        validate_typed_dict(self.valid_kwargs, kwargs)

        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self._valid_kwargs_names:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        return self._preprocess_audio_like_inputs(audio, *args, **kwargs)

    def _further_process_kwargs(
        self,
        **kwargs,
    ) -> dict:
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
        """
        Validate the kwargs for the preprocess method.
        """
        validate_preprocess_arguments(
            sample_rate=sample_rate,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    def _preprocess_audio_like_inputs(
        self,
        audio: AudioInput,
        *args,
        sample_rate: int | None = None,
        **kwargs: Unpack[AudioKwargs],
    ) -> BatchFeature:
        audio = self._prepare_audio_like_inputs(audio=audio, sample_rate=sample_rate)
        return self._preprocess(audio, *args, **kwargs)

    def _prepare_audio_structure(self, audio: AudioInput, sample_rate: int | None = None) -> list:
        """
        Prepare the audio structure for processing: handle URL inputs, validate sample rate,
        and flatten into a list of audio arrays.

        Analogous to `_prepare_images_structure` in the image processing pipeline.
        """
        if not (isinstance(audio, str) or (isinstance(audio, (list, tuple)) and all(isinstance(el, str) for el in audio))):
            # NOTE: we want to force the user to either:
            # 1. pass the sample rate when provided audio is array-type, to avoid silent errors that might be hard to debug
            # 2. pass url-type audio inputs, that we can load in the correct sample rate directly
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
        elif isinstance(audio, str):
            audio = [audio]

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
        return super().to_dict()


@lru_cache(maxsize=10)
def validate_preprocess_arguments(
    sample_rate: int | None = None,
    max_length: int | None = None,
    truncation: bool | None = None,
    pad_to_multiple_of: int | None = None,
    return_tensors: str | TensorType | None = None,
):
    """
    Checks validity of typically used arguments in a `BaseAudioProcessor` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    """
    pass
