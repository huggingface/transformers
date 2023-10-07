# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""
Feature extractor class for Music Audio Efficient Spectrogram Transformer.
"""


from typing import List, Optional, Union

import numpy as np
import torch

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class MAESTFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Music Audio Efficient Spectrogram Transformer (MAEST) feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw audio, pads/truncates them to a fixed length and normalizes
    them using a mean and standard deviation.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 96):
            Number of Mel-frequency bins.
        max_length (`int`, *optional*, defaults to 1876):
            Maximum length to which to pad/truncate the extracted features. Set to -1 to deactivate the functionallity.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the log-Mel features using `mean` and `std`.
        mean (`float`, *optional*, defaults to 2.06755686098554):
            The mean value used to normalize the log-Mel features. Uses the Discogs20 mean by default.
        std (`float`, *optional*, defaults to 1.268292820667291):
            The standard deviation value used to normalize the log-Mel features. Uses the Discogs20 standard deviation
            by default.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
        n_fft (`int`, *optional*, defaults to 512):
            Length of the FFT window.
        hop_length (`int`, *optional*, defaults to 256):
            Number of samples between successive frames.
        log_compression (`str`, *optional*, defaults to `logC`):
            Type of log compression to apply to the mel-spectrogram. Can be one of [`None`, `log`, `logC`].
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        num_mel_bins=96,
        max_length=1876,
        padding_value=0.0,
        do_normalize=True,
        mean=2.06755686098554,
        std=1.268292820667291,
        return_attention_mask=False,
        n_fft=512,
        hop_length=256,
        log_compression="logC",
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.log_compression = log_compression
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        self.return_attention_mask = return_attention_mask

        self.window = window_function(
            window_length=self.n_fft,
            name="hann",
        ).tolist()

        self.mel_fb = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=self.num_mel_bins,
            min_frequency=0,
            max_frequency=self.sampling_rate / 2,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        ).tolist()

    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
        max_length: int,
    ) -> np.ndarray:
        """
        Get mel-spectrogram features using audio_utils.
        """

        melspec = spectrogram(
            waveform,
            window=np.array(self.window),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2,
            mel_filters=np.array(self.mel_fb),
            min_value=1e-30,
            mel_floor=1e-30,
            pad_mode="constant",
        )

        if not self.log_compression:
            pass
        elif self.log_compression == "log":
            melspec = np.log(melspec + np.finfo(float).eps)
        elif self.log_compression == "logC":
            melspec = np.log10(1 + melspec * 10000)
        else:
            raise ValueError(
                f"`log_compression` can only be one of [None, 'log', 'logC'], but got: {self.log_compression}"
            )

        melspec = torch.Tensor(melspec)
        n_frames = melspec.shape[0]

        if max_length > 0:
            difference = max_length - n_frames

            # pad or truncate, depending on difference
            if difference > 0:
                pad_module = torch.nn.ZeroPad2d((0, 0, 0, difference))
                melspec = pad_module(melspec)
            elif difference < 0:
                melspec = melspec[0:max_length, :]

        return melspec.numpy()

    def normalize(self, input_values: np.ndarray) -> np.ndarray:
        return (input_values - (self.mean)) / (self.std * 2)

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
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

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # extract fbank features and pad/truncate to max_length
        features = [self._extract_fbank_features(waveform, max_length=self.max_length) for waveform in raw_speech]

        # convert into BatchFeature
        padded_inputs = BatchFeature({"input_values": features})

        # make sure list is in array format
        input_values = padded_inputs.get("input_values")
        if isinstance(input_values[0], list):
            padded_inputs["input_values"] = [np.asarray(feature, dtype=np.float32) for feature in input_values]

        # normalization
        if self.do_normalize:
            padded_inputs["input_values"] = [self.normalize(feature) for feature in input_values]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs
