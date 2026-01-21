# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union
import torch

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, logging


logger = logging.get_logger(__name__)


class VibeVoiceAcousticTokenizerFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a VibeVoiceAcousticTokenizer feature extractor.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The number of channels.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used for padding.
        normalize_audio (`bool`, *optional*, defaults to `True`):
            Whether to normalize audio to a target dB FS.
        target_dB_FS (`float`, *optional*, defaults to -25):
            Target dB FS for normalization.
        eps (`float`, *optional*, defaults to 1e-06):
            A small value to avoid division by zero when normalizing.

    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=24000,
        padding_value=0.0,
        normalize_audio=True,
        target_dB_FS=-25,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.normalize_audio = normalize_audio
        self.target_dB_FS = target_dB_FS
        self.eps = eps

    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: Optional[int] = None,
        padding: Optional[Union[bool, str, PaddingStrategy]] = True,
        pad_to_multiple_of: Optional[int] = None,
        max_length: Optional[int] = None,
        return_attention_mask: Optional[bool] = True,
        return_tensors: Optional[Union[str, type]] = None,
    ) -> BatchFeature:
        """
        Main method to prepare audio for the VibeVoice model.

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`:
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a torch tensor,
                a list of numpy arrays or a list of torch tensors.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.
            
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided audio input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        # Ensure batch
        audio = make_list_of_audio(audio)

        # Ensure torch tensors and mono
        for idx, example in enumerate(audio):
            example = torch.tensor(example, dtype=torch.float32)
            if example.ndim != 1:
                raise ValueError(f"Audio should be mono, got shape: {example.shape}")
            audio[idx] = example

        if self.normalize_audio:
            for idx, example in enumerate(audio):
                rms = torch.sqrt(torch.mean(example**2))
                example *= 10 ** (self.target_dB_FS / 20) / (rms + self.eps)
                max_val = torch.max(torch.abs(example))
                if max_val > 1.0:
                    example = example / (max_val + self.eps)
                audio[idx] = example

        output_values = BatchFeature({"input_values": audio})
        if padding:
            output_values = self.pad(
                output_values,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                padding=padding,
                max_length=max_length,
                return_tensors=return_tensors,
            )
            if return_attention_mask:
                output_values["padding_mask"] = output_values.pop("attention_mask")

        # add channel dimension if missing
        if output_values["input_values"].ndim == 2:
            output_values["input_values"] = output_values["input_values"][:, None, :]

        return output_values


__all__ = ["VibeVoiceAcousticTokenizerFeatureExtractor"]
