# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""Feature extraction class for Trillsson_efficient."""
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class Trillsson_efficientFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Trillsson_efficient feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in Hertz per second (Hz).
        padding_value (`float`, defaults to 0.0):
            The value that is used to fill the padding values / vectors.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to zero-mean unit-variance normalize the input.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~Trillsson_efficientFeatureExtractor.__call__`] should return `attention_mask`.
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel bins.
        log_floor (`float`, *optional*, defaults to 1e-12):
            Floor on the log-mel spectrogram.
        log_additive_offset (`float`, *optional*, defaults to 0.001):
            Offset on the log-mel spectrogram.
        window_length_secs (`float`, *optional*, defaults to 0.025):
            Window length in seconds.
        hop_length_secs (`float`, *optional*, defaults to 0.010):
            Hop length in seconds.
        f_max (`float`, *optional*, defaults to 7500.0):
            Maximum frequency for the Mel filterbank.
        f_min (`float`, *optional*, defaults to 125.0):
            Minimum frequency for the Mel filterbank.
        fft_length (`int`, *optional*):
            FFT length. If `None`, it will be computed from the window length. Defaults to `None`.
    """

    model_input_names = ["input_values"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
        num_mel_bins=80,
        log_floor=1e-12,
        log_additive_offset=0.001,
        window_length_secs=0.025,
        hop_length_secs=0.010,
        f_max=7500.0,
        f_min=125.0,
        fft_length=None,
        **kwargs
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.num_mel_bins = num_mel_bins
        self.log_floor = log_floor
        self.log_additive_offset = log_additive_offset
        self.window_length_secs = window_length_secs
        self.hop_length_secs = hop_length_secs
        self.f_max = f_max
        self.f_min = f_min
        self.fft_length = fft_length

    @classmethod
    def pad_symmetric(cls, raw_speech: np.ndarray, padding: int):
        """Symmetric pad a 2D Tensor"""
        if raw_speech.ndim != 2:
            raise ValueError(f"Raw speech must be rank 2 got {raw_speech.shape}")
        raw_speech_len = raw_speech.shape[1]
        if padding > raw_speech_len:
            pad_raw_speech = cls.repeat_n_times_with_extra(raw_speech, padding, raw_speech_len)
        else:
            pad_raw_speech = np.pad(raw_speech, [(0, 0), (0, padding)], mode="symmetric")
        return pad_raw_speech

    @classmethod
    def pad_waveform(cls, raw_speech: np.ndarray, padding: int, mode: str, padding_value: float):
        """Pad waveform"""
        if raw_speech.ndim != 2:
            raise ValueError(f"Raw speech must be rank 2 got {raw_speech.shape}")
        if mode == "symmetric":
            return cls.pad_symmetric(raw_speech, padding)
        else:
            pad_raw_speech = np.pad(raw_speech, [(0, 0), (0, padding)], mode=mode, constant_values=padding_value)
            return pad_raw_speech

    @staticmethod
    def repeat_n_times_with_extra(raw_speech: np.ndarray, padding: int, raw_speech_len: int):
        """Pad symmetric longer than the original raw speech"""
        if raw_speech.ndim != 2:
            raise ValueError(f"Raw speech must be rank 2 got {raw_speech.shape}")
        num_copies = np.floor_divide(padding, raw_speech_len)
        r = np.fliplr(raw_speech)
        f = np.concatenate((r, raw_speech), axis=1)
        copies = np.tile(f, [1, np.floor_divide(num_copies, 2)])
        if num_copies % 2 != 0:
            copies = np.concatenate((copies, r), axis=1)
        pre_pad = np.concatenate([raw_speech, copies], axis=1)
        extra = padding % raw_speech_len
        pad_raw_speech = np.pad(pre_pad, [(0, 0), (0, extra)], mode="symmetric")
        return pad_raw_speech

    @staticmethod
    def stabilized_log(data: np.ndarray, floor: float, additive_offset: float):
        """Stabilized MelSpectrogram"""
        return np.log(np.maximum(data, floor) + additive_offset)

    @staticmethod
    def rescale(input_array: np.ndarray, rate: float = 1.0 / 128.0, offset: int = -1):
        """Rescale the input array"""
        return input_array * rate + offset

    def log_mel_spectrogram(
        self,
        raw_speech: np.ndarray,
        audio_sample_rate: int,
    ):
        """Convert waveform to LogMelSpectrogram"""
        window_length_samples = int(round(audio_sample_rate * self.window_length_secs))
        hop_length_samples = int(round(audio_sample_rate * self.hop_length_secs))
        if self.fft_length:
            fft_length = self.fft_length
        else:
            fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        spectrogram = tf.abs(
            tf.signal.stft(
                tf.cast(raw_speech, tf.dtypes.float64),
                frame_length=window_length_samples,
                frame_step=hop_length_samples,
                fft_length=fft_length,
                window_fn=tf.signal.hann_window,
            )
        )
        to_mel = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=fft_length // 2 + 1,
            sample_rate=audio_sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
            dtype=tf.dtypes.float64,
        )

        mel = spectrogram @ to_mel
        mel = np.transpose(mel, (0, 2, 1)).astype(np.float32)
        log_mel = self.stabilized_log(mel, self.log_floor, self.log_additive_offset)
        return log_mel

    def compute_feature(
        self,
        input_values: List[np.ndarray],
        sampling_rate: int,
        required_length: int,
        pad_mode: str,
        frame_width: int,
        frame_hop: int,
        padding: bool = False,
    ) -> List[np.ndarray]:
        """Compute the feature"""
        extracted_input_values = []
        if padding:
            max_length = max(raw_speech.shape[0] for raw_speech in input_values)
            # set required length to max length if required length is less than max length else use required length
            required_length = max(required_length, max_length)

        for raw_speech in input_values:
            if raw_speech.ndim != 1:
                raise ValueError(f"Raw speech must be rank 1 got {raw_speech.shape}")
            raw_speech = np.expand_dims(raw_speech, axis=0)

            # pad to max length
            if required_length:
                n = raw_speech.shape[1]
                delta = required_length - n
                if delta > 0:
                    raw_speech = self.pad_waveform(raw_speech, delta, mode=pad_mode, padding_value=self.padding_value)

            mel_speech = self.log_mel_spectrogram(raw_speech, sampling_rate)
            mel_speech = tf.signal.frame(mel_speech, frame_length=frame_width, frame_step=frame_hop, axis=2).numpy()
            mel_speech = np.squeeze(mel_speech, axis=0)
            mel_speech = np.transpose(mel_speech, (1, 2, 0))
            # rescale
            if self.do_normalize:
                mel_speech = self.rescale(mel_speech)
            extracted_input_values.append(mel_speech)

        return extracted_input_values

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        frame_hop: Optional[int] = 7,
        required_length: Optional[int] = 32000,
        num_mel_bins: Optional[int] = 80,
        frame_width: Optional[int] = 195,
        pad_mode: Optional[str] = "symmetric",
        return_tensors: Optional[Union[str, TensorType]] = None,
        padding: Optional[bool] = False,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to compute and prepare for the model one or several sequence(s). sequences.

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            sampling_rate (:obj:`int`, `optional`):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            frame_hop (:obj:`int`, `optional`, defaults to 195):
                The number of samples between successive frames.
            required_length (:obj:`int`, `optional`, defaults to 32000):
                The minimum length of the sequence after padding.
            num_mel_bins (:obj:`int`, `optional`, defaults to 80):
                The number of mel bins.
            frame_width (:obj:`int`, `optional`, defaults to 195):
                The number of samples in each frame.
            pad_mode (:obj:`str`, `optional`, defaults to 'symmetric'):
                The padding mode to use. Can be one of 'constant', 'symmetric', 'reflect', 'wrap'.
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                - :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                - :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            padding (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to pad the input values to the max length of the batch.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
            logger.warning(
                "It is strongly recommended to pass the ``sampling_rate`` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # convert input values to correct format
        if not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(array, dtype=np.float32) for array in raw_speech]
        elif (
            not isinstance(raw_speech, np.ndarray)
            and isinstance(raw_speech[0], np.ndarray)
            and raw_speech[0].dtype is np.dtype(np.float64)
        ):
            raw_speech = [array.astype(np.float32) for array in raw_speech]
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # compute features
        input_values = self.compute_feature(
            raw_speech, sampling_rate, required_length, pad_mode, frame_width, frame_hop, padding
        )
        encoded_inputs = BatchFeature({"input_values": input_values})

        if return_tensors is not None:
            encoded_inputs = encoded_inputs.convert_to_tensors(return_tensors)
        return encoded_inputs
