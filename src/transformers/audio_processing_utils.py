# coding=utf-8
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
from typing import Optional, Union, Unpack

import numpy as np
from huggingface_hub.dataclasses import validate_typed_dict

from .audio_processing_base import AudioProcessingMixin
from .audio_utils import AudioInput, make_list_of_audio
from .feature_extraction_utils import BatchFeature
from .image_utils import validate_kwargs
from .processing_utils import AudioKwargs
from .utils import TensorType, logging
from .utils.import_utils import is_torch_available, requires


if is_torch_available():
    import torch
    import torch.nn.functional as F


logger = logging.get_logger(__name__)


@requires(backends=("torch",))
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

        kwargs = self.filter_out_unused_kwargs(kwargs)
        self._init_kwargs_from_valid_kwargs(kwargs)

    def __call__(self, audio: AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
        return self.preprocess(audio, *args, **kwargs)

    def preprocess(self, audio: AudioInput, *args, **kwargs: Unpack[AudioKwargs]) -> BatchFeature:
        # args are not validated, but their order in the `preprocess` and `_preprocess` signatures must be the same
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_kwargs_names)

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
        sample_rate: Optional[int] = None,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
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
        sample_rate: Optional[int] = None,
        **kwargs: Unpack[AudioKwargs],
    ) -> BatchFeature:
        audio = self._prepare_audio_like_inputs(audio=audio, sample_rate=sample_rate)
        return self._preprocess(audio, *args, **kwargs)

    def _prepare_audio_like_inputs(self, audio: AudioInput, sample_rate: Optional[int] = None) -> list["torch.Tensor"]:
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

        if self.force_mono:
            # TODO: audio proc, to change
            audio = [a.mean(axis=1) if a.ndim > 1 else a for a in audio]

        audio = [torch.from_numpy(audio_el) if isinstance(audio_el, np.ndarray) else audio_el for audio_el in audio]

        return audio

    def _preprocess(
        self,
        audio: list["torch.Tensor"],
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        **kwargs,
    ) -> BatchFeature:
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        is_batched = len(audio) > 1

        if truncation and max_length is None:
            raise ValueError("When setting ``truncation=True``, make sure that ``max_length`` is defined.")

        if is_batched and not truncation and max_length is not None and max(audio_el.shape[-1] for audio_el in audio) > max_length:
            logger.warning(
                f"Truncation is set to False but `max_length` is set to {max_length} with the longest audio being "
                f"{max(audio_el.shape[-1] for audio_el in audio)}. We will set truncation to True."
            )
            truncation = True

        if truncation:
            audio = [audio_el[..., :max_length] for audio_el in audio]

        if max_length is None:
            max_length = max(audio_el.shape[-1] for audio_el in audio)

        if padding:
            audio = [self.pad(audio_el, max_length) for audio_el in audio]

        audio = torch.stack(audio, dim=0) if return_tensors else audio
        return BatchFeature(data={"audio": audio}, tensor_type=return_tensors)

    def pad(self, audio: "torch.Tensor", max_length: int) -> "torch.Tensor":
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio

        if self.padding_value is None:
            raise ValueError(
                "Asking to pad but the audio processor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `audio_processor.padding_value = 0.0`."
            )

        if self.padding_side == "right":
            pad_args = (0, max_length - current_length)
        elif self.padding_side == "left":
            pad_args = (max_length - current_length, 0)
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")

        return F.pad(audio, pad_args, "constant", self.padding_value)

    def to_dict(self):
        return super().to_dict()


@lru_cache(maxsize=10)
def validate_preprocess_arguments(
    sample_rate: Optional[int] = None,
    max_length: Optional[int] = None,
    truncation: Optional[bool] = None,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
):
    """
    Checks validity of typically used arguments in a `BaseAudioProcessor` `preprocess` method.
    Raises `ValueError` if arguments incompatibility is caught.
    """
    pass
