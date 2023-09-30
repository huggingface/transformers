# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for ImageBind."""


import warnings
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio.compliance.kaldi as ta_kaldi

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
from .image_processing_imagebind import ImageBindImageProcessor


logger = logging.get_logger(__name__)


def valid_batched_clipped_audio(raw_speech):
    """
    Determines whether raw mono-channel audio input (or any other 1D data) is batched and clipped. The following
    conditions will be recognized as valid audio:

    - unbatched: `List[float]`, `np.ndarray` (`ndim=1`)
    - batched: `List[List[float]]`, `List[np.ndarray]` (`ndim=1`), `np.ndarray` (`ndim=2`)
    - batched and clipped: `List[List[List[float]]]`, `List[List[np.ndarray]]` (`ndim=1`), List[np.ndarray] (`ndim=2`), np.ndarray (`ndim=3`)
    """
    valid_audio = False
    if isinstance(raw_speech, np.ndarray) and (1 <= len(raw_speech.shape) <= 3):
        # unbatched, batched, or batched and clipped np.ndarray
        valid_audio = True
    elif isinstance(raw_speech, (list, tuple)):
        if isinstance(raw_speech[0], np.ndarray) and (1 <= len(raw_speech[0].shape) <= 2):
            # batched or batched and clipped List[np.ndarray]
            valid_audio = True
        elif isinstance(raw_speech[0], float):
            # unbatched List[float]
            valid_audio = True
        elif isinstance(raw_speech[0], (list, tuple)):
            if isinstance(raw_speech[0][0], np.ndarray) and (len(raw_speech[0][0].shape == 1)):
                # batched and clipped List[List[np.ndarray]]
                valid_audio = True
            elif isinstance(raw_speech, (float, list, tuple)):
                # batched List[List[float]], batched and clipped List[List[List[float]]]
                valid_audio = True
    return valid_audio


def batch_and_clip_ndarray(array, data_dim=1, dtype=np.float32):
    """
    Turns a possibly nested list of np.ndarrays into a batched and clipped output of type `List[List[np.ndarray]]`.
    """
    if isinstance(array, (list, tuple)) and isinstance(array[0], (list, tuple)) and isinstance(array[0][0], np.ndarray):
        if array[0][0].ndim == data_dim:
            return [[base_array.astype(dtype=dtype) for base_array in clip] for clip in array]
        else:
            raise ValueError(
                f"`For List[List[np.ndarray]]` inputs the internal `np.ndarray`s are expected to have dimension"
                f" {data_dim} but got dimension {array[0][0].ndim}"
            )
    elif isinstance(array, (list, tuple) and isinstance(array[0], np.ndarray)):
        if array[0].ndim == data_dim + 1:
            return [[np.asarray(base_array, dtype=dtype) for base_array in clip] for clip in array]
        elif array[0].ndim == data_dim:
            return [[base_array.astype(dtype=dtype) for base_array in array]]
        else:
            raise ValueError(
                f"For `List[np.ndarray]` inputs the internal `np.ndarray`s are expected to have dimension"
                f" {data_dim} or {data_dim + 1} but got dimension {array[0].ndim}"
            )
    elif isinstance(array, np.ndarray):
        if array.ndim == data_dim + 2:
            return [[np.asarray(raw_input, dtype=dtype) for raw_input in clip] for clip in array]
        elif array.ndim == data_dim + 1:
            return [[np.asarray(raw_input, dtype=dtype) for raw_input in array]]
        elif array.ndim == data_dim:
            return [[array.astype(dtype=dtype)]]
        else:
            raise ValueError(
                f"`np.ndarray` inputs are expected to have dimension in"
                f" `[{data_dim}, {data_dim + 1}, {data_dim + 2}]` but instead got {array.ndim}"
            )
    else:
        raise ValueError(f"Could not make batched and clipped audio from {array}")


class ImageBindFeatureExtractor(ImageBindImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class ImageBindFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use ImageBindImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)


# NOTE: ImageBind follow Audio Spectrogram Transformer for audio processing
# Based on ASTFeatureExtractor
class ImageBindAudioFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Audio Spectrogram Transformer (AST) feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using TorchAudio, pads/truncates them to a fixed
    length and normalizes them using a mean and standard deviation.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of Mel-frequency bins.
        max_length (`int`, *optional*, defaults to 204):
            Maximum length to which to pad/truncate the extracted features.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the log-Mel features using `mean` and `std`.
        mean (`float`, *optional*, defaults to -4.268):
            The mean value used to normalize the log-Mel features. Uses the AudioSet mean by default.
        std (`float`, *optional*, defaults to 9.138):
            The standard deviation value used to normalize the log-Mel features. Uses the AudioSet standard deviation
            by default.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        num_mel_bins=128,
        max_length=204,
        padding_value=0.0,
        do_normalize=True,
        mean=-4.268,
        std=9.138,
        return_attention_mask=False,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        self.return_attention_mask = return_attention_mask

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        # waveform = waveform * (2**15)  # Kaldi compliance: 16-bit signed integers
        # Mean center the waveform
        waveform -= waveform.mean()
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        fbank = ta_kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.sampling_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_shift=10,
        )

        n_frames = fbank.shape[0]
        difference = max_length - n_frames

        # pad or truncate, depending on difference
        if difference > 0:
            pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))
            fbank = pad_module(fbank)
        elif difference < 0:
            fbank = fbank[0:max_length, :]

        fbank = fbank.numpy()

        return fbank

    def normalize(self, input_values: np.ndarray) -> np.ndarray:
        return (input_values - (self.mean)) / (self.std * 2)

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]], List[List[List[float]]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`, `List[List[List[float]]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of numpy
                arrays or a (possibly nested) list of float values. The supported input types are as follows:

                - unbatched: `List[float]`, `np.ndarray` (`ndim=1`)
                - batched: `List[List[float]]`, `List[np.ndarray]` (`ndim=1`), `np.ndarray` (`ndim=2`)
                - batched with clips: `List[List[List[float]]]`, `List[List[np.ndarray]]` (`ndim=1`), `List[np.ndarray]` (`ndim=2`), np.ndarray (`ndim=3`)
                
                The input will always be interpreted as mono channel audio, not stereo, i.e. a single float per timestep.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )
        
        if not valid_batched_clipped_audio(raw_speech):
            raise ValueError(
                f"Only unbatched, batched, and batched and clipped mono-channel audio is supported for input to {self}"
            )
        
        # Handle the cases where there are no np.ndarrays in raw_speech
        if isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], float):
            raw_speech = [[np.asarray(raw_speech, dtype=np.float32)]]
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], (list, tuple)):
            if isinstance(raw_speech[0][0], float):
                # List[List[float]]
                raw_speech = [[np.asarray(audio, dtype=np.float32) for audio in raw_speech]]
            elif isinstance(raw_speech[0][0], (list, tuple)):
                # List[List[List[float]]]
                raw_speech = [[np.asarray(audio, dtype=np.float32) for audio in clip] for clip in raw_speech]

        # always return batched and clipped audio of type [List[List[np.ndarray]]]
        raw_speech = batch_and_clip_ndarray(raw_speech, data_dim=1, dtype=np.float32)

        # extract fbank features and pad/truncate to max_length
        features = [[self._extract_fbank_features(waveform, max_length=self.max_length) for waveform in clip] for clip in raw_speech]

        # convert into BatchFeature
        padded_inputs = BatchFeature({"input_features": features})

        # make sure spectrograms are in array format
        input_values = padded_inputs.get("input_features")
        if isinstance(input_values[0][0], list):
            padded_inputs["input_features"] = [[np.asarray(feature, dtype=np.float32) for feature in clip] for clip in input_values]

        # normalization
        if self.do_normalize:
            padded_inputs["input_features"] = [
                [self.normalize(feature) for feature in clip] for clip in padded_inputs["input_features"]
            ]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


class ImageBindImuFeatureExtractor(SequenceFeatureExtractor):
    """
    Feature extractor for ImageBind IMU data.
    """
    pass
