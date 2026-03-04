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


import numpy as np

from .audio_processing_utils import BaseAudioProcessor
from .feature_extraction_utils import BatchFeature
from .utils import logging
from .utils.import_utils import requires


logger = logging.get_logger(__name__)


class NumpyBackend(BaseAudioProcessor):
    """NumPy backend for portable CPU-only audio processing."""

    @property
    def backend(self) -> str:
        return "numpy"

    def process_audio(self, audio_el):
        """
        Process a single raw audio input into a np.ndarray.

        Handles mono conversion (averaging channels) and ensures numpy format.
        """
        if not isinstance(audio_el, np.ndarray):
            audio_el = np.asarray(audio_el)

        if self.force_mono:
            audio_el = audio_el.mean(axis=1) if audio_el.ndim > 1 else audio_el

        return audio_el

    def pad(self, audio: np.ndarray, max_length: int) -> np.ndarray:
        """Pad a single audio array to a target length using np.pad."""
        current_length = audio.shape[-1]
        if current_length >= max_length:
            return audio

        if self.padding_value is None:
            raise ValueError(
                "Asking to pad but the audio processor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `audio_processor.padding_value = 0.0`."
            )

        pad_length = max_length - current_length
        if self.padding_side == "right":
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, pad_length)]
        elif self.padding_side == "left":
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(pad_length, 0)]
        else:
            raise ValueError(f"Invalid padding side: {self.padding_side}")

        return np.pad(audio, pad_width, mode="constant", constant_values=self.padding_value)

    def _preprocess(
        self,
        audio: list[np.ndarray],
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        **kwargs,
    ) -> BatchFeature:
        """Preprocess using NumPy backend: truncation, padding, stacking."""
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

        audio = np.stack(audio, axis=0) if return_tensors else audio
        return BatchFeature(data={"audio": audio}, tensor_type=return_tensors)


@requires(backends=("torch",))
class TorchBackend(BaseAudioProcessor):
    """Torch backend for audio processing."""

    @property
    def backend(self) -> str:
        return "torch"

    def process_audio(self, audio_el):
        """
        Process a single raw audio input into a torch.Tensor.

        Handles mono conversion (averaging channels) and numpy-to-torch conversion.
        """
        import torch

        if self.force_mono:
            audio_el = audio_el.mean(axis=1) if audio_el.ndim > 1 else audio_el

        if isinstance(audio_el, np.ndarray):
            audio_el = torch.from_numpy(audio_el)

        return audio_el

    def pad(self, audio: "torch.Tensor", max_length: int) -> "torch.Tensor":
        """Pad a single audio tensor to a target length using torch.nn.functional.pad."""
        import torch.nn.functional as F

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
        """Preprocess using Torch backend: truncation, padding, stacking."""
        import torch

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
