# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for Htdemucs."""

from typing import List, Optional, Union

import numpy as np

from ... import is_torch_available
from ...audio_utils import spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class HtdemucsFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs an Htdemucs feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Instantiating a feature extractor with the defaults will yield a similar configuration to that of the
    [facebook/Htdemucs_24khz](https://huggingface.co/facebook/Htdemucs_24khz) architecture.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features. Use 1 for mono, 2 for stereo.
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio waveform should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        segment (`float`, *optional*, defaults to 7.8):
            Duration of the chunks of audio to ideally evaluate the model on.
    """

    model_input_names = ["input_values", "padding_mask"]

    def __init__(
        self,
        feature_size: int = 2,
        sampling_rate: int = 44100,
        padding_value: float = 0.0,
        segment: float = 7.8,
        overlap: float = 0.25,
        n_fft: int = 4096,
        hop_length: int = 1024,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.segment = segment
        self.overlap = overlap
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio using a numpy-based implementation.
        """
        *other, length = waveform.shape
        waveform = waveform.reshape(-1, length)

        stft = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=None,
            log_mel="log10",
        )
        _, freqs, frame = stft.shape

        stft = stft.view(*other, freqs, frame)
        B, C, Fr, T = stft.shape
        stft = np.view_as_real(stft).permute(0, 1, 4, 2, 3)

        stft = stft.reshape(B, C * 2, Fr, T)
        return stft

    def _torch_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio using the PyTorch STFT implementation.
        """
        waveform = torch.from_numpy(waveform).type(torch.float32)

        *other, length = waveform.shape
        waveform = waveform.reshape(-1, length)

        window = torch.hann_window(self.n_fft)
        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=window,
            win_length=self.n_fft,
            normalized=True,
            return_complex=True,
        )

        _, freqs, frame = stft.shape

        stft = stft.view(*other, freqs, frame)
        B, C, Fr, T = stft.shape
        stft = torch.view_as_real(stft).permute(0, 1, 4, 2, 3)

        stft = stft.reshape(B, C * 2, Fr, T)
        return stft.numpy()

    def __call__(
        self,
        raw_audio: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Optional[Union[bool, str, PaddingStrategy]] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_audio (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. The numpy array must be of shape
                `(num_samples,)` for mono audio (`feature_size = 1`), or `(2, num_samples)` for stereo audio
                (`feature_size = 2`).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, *optional*, defaults to `False`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
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
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        if padding and truncation:
            raise ValueError("Both padding and truncation were set. Make sure you only set one.")
        elif padding is None:
            # by default let's pad the inputs
            padding = True

        is_batched = bool(
            isinstance(raw_audio, (list, tuple)) and (isinstance(raw_audio[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_audio = [np.asarray(audio, dtype=np.float32).T for audio in raw_audio]
        elif not is_batched and not isinstance(raw_audio, np.ndarray):
            raw_audio = np.asarray(raw_audio, dtype=np.float32)
        elif isinstance(raw_audio, np.ndarray) and raw_audio.dtype is np.dtype(np.float64):
            raw_audio = raw_audio.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_audio = [np.asarray(raw_audio).T]

        # verify inputs are valid
        for idx, example in enumerate(raw_audio):
            if example.ndim > 2:
                raise ValueError(f"Expected input shape (channels, length) but got shape {example.shape}")
            if self.feature_size == 1 and example.ndim != 1:
                raise ValueError(f"Expected mono audio but example has {example.shape[-1]} channels")
            if self.feature_size == 2 and example.shape[-1] != 2:
                raise ValueError(f"Expected stereo audio but example has {example.shape[-1]} channels")

        seq_length = max(array.shape[0] for array in raw_audio)
        max_shift = int(0.5 * self.sampling_rate)
        offset = np.random.randint(0, max_shift)
        offset = 12623
        valid_length = int(self.segment * self.sampling_rate)

        # TODO(SG): how to handle max length
        max_length = seq_length + 2 * max_shift if max_length is None else max_length

        encoded_inputs = BatchFeature({"input_values": raw_audio})
        padded_inputs = self.pad(
            encoded_inputs,
            max_length=max_length,
            truncation=truncation,
            padding="max_length",
            return_attention_mask=True,
        )
        input_values = padded_inputs["input_values"].transpose(0, 2, 1)

        segment_length = int(self.sampling_rate * self.segment)
        stride = int((1 - self.overlap) * segment_length)
        offset_length = seq_length + max_shift - offset
        offsets = range(0, offset_length, stride)
        extract_fbank_features = (
            self._torch_extract_fbank_features if is_torch_available() else self._np_extract_fbank_features
        )

        for arr_offset in offsets:
            length = min(offset_length - arr_offset, segment_length)
            delta = valid_length - length

            arr_offset = arr_offset + offset
            start = arr_offset - delta // 2
            end = start + valid_length

            correct_start = max(0, start)
            correct_end = min(max_length, end)

            pad_left = correct_start - start
            pad_right = end - correct_end

            out = np.pad(input_values[..., correct_start:correct_end], (pad_left, pad_right))
            input_features = [extract_fbank_features(waveform) for waveform in out]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs
