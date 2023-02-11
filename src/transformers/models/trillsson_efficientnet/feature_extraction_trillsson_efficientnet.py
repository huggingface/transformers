# coding=utf-8
# Copyright 2022 The Google Authors and The HuggingFace Inc. team. All rights reserved.
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
import math
from typing import List, Optional, Union

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class TrillssonEfficientNetFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a TrillssonEfficientNet feature extractor.

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
        mel_high_frequency_q (`float`, *optional*, defaults to 1127.0):
            Constant used in the mel scale.
        mel_break_frequency_hertz (`float`, *optional*, defaults to 700.0):
            Constant used in the mel scale.
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
        mel_high_frequency_q=1127.0,
        mel_break_frequency_hertz=700.0,
        fft_length=None,
        **kwargs,
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
        self.mel_high_frequency_q = mel_high_frequency_q
        self.mel_break_frequency_hertz = mel_break_frequency_hertz
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
    def hann_window(window_length, periodic=True, dtype=np.float32):
        """Generate a [Hann window][hann]."""

        if window_length == 1:
            return np.ones([1], dtype=dtype)

        even = 1 - np.mod(window_length, 2)
        n = window_length + periodic * even - 1
        count = np.arange(window_length, dtype=dtype)
        cos_arg_np = 2 * np.pi * count / n

        return (0.5 - 0.5 * np.cos(cos_arg_np)).astype(dtype)

    @staticmethod
    def _infer_frame_shape(signal, frame_length, frame_step, pad_end, axis=-1):
        """Infers the shape of the return value of `frame`."""

        if signal.ndim is None:
            return None

        if axis is None:
            return [None] * (signal.ndim + 1)

        signal_shape = list(signal.shape)
        num_frames = None
        frame_axis = signal_shape[axis]
        outer_dimensions = signal_shape[:axis]
        inner_dimensions = signal_shape[axis:][1:]

        if signal_shape and frame_axis is not None:
            if frame_step is not None and pad_end:
                # Double negative is so that we round up.
                num_frames = max(0, -(-frame_axis // frame_step))
            elif frame_step is not None and frame_length is not None:
                assert not pad_end
                num_frames = max(0, (frame_axis - frame_length + frame_step) // frame_step)

        return tuple(outer_dimensions + [num_frames, frame_length] + inner_dimensions)

    @staticmethod
    def _pad_for_rfft(input_tensor, fft_length, is_reverse=False):
        """Pads `input_tensor` to `fft_length` on its inner-most `fft_rank` dims."""

        fft_length = np.array([fft_length], dtype=np.int32)
        fft_shape = [dim if dim != -1 else None for dim in fft_length]

        # Edge case: skip empty tensors.
        if input_tensor.ndim is not None and any(dim == 0 for dim in input_tensor.shape):
            return input_tensor

        # Slice the last FFT-rank dimensions from input_tensor's shape.
        input_fft_shape = list(input_tensor.shape[-1:])

        if input_tensor.ndim is not None and input_fft_shape is not None:
            # In reverse, we only pad the inner-most dimension to fft_length / 2 + 1.
            if is_reverse and fft_shape[-1] is not None:
                fft_shape[-1] = fft_shape[-1] // 2 + 1

            paddings = [[0, max(fft_dim - input_dim, 0)] for fft_dim, input_dim in zip(fft_shape, input_fft_shape)]

            if any(pad > 0 for _, pad in paddings):
                outer_paddings = [[0, 0]] * max((input_tensor.ndim - 1), 0)
                input_tensor = np.pad(input_tensor, outer_paddings + paddings)

            return input_tensor

    def convert_frequency(self, frequencies, to_mel_scale=True):
        """Converts frequencies to the mel scale (if `to_mel_scale`) or from the mel scale to Hertz (if not
        `to_mel_scale`)."""
        if to_mel_scale:
            return self.mel_high_frequency_q * np.log(1.0 + (frequencies / self.mel_break_frequency_hertz))
        else:
            return self.mel_break_frequency_hertz * (np.exp(frequencies / self.mel_high_frequency_q) - 1.0)

    def frame(self, signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
        """Expands `signal`'s `axis` dimension into frames of `frame_length`"""

        if not signal.ndim >= 1:
            raise ValueError("Expect input waveform is not None")

        result_shape = self._infer_frame_shape(signal, frame_length, frame_step, pad_end, axis)
        signal_shape = signal.shape

        # Axis can be negative. Convert it to positive.
        axis = range(len(signal_shape))[axis]
        outer_dimensions, length_samples, inner_dimensions = np.split(
            signal_shape, indices_or_sections=[axis, axis + 1]
        )
        length_samples = length_samples.item()

        num_outer_dimensions = outer_dimensions.size
        num_inner_dimensions = inner_dimensions.size

        # If padding is requested, pad the input signal tensor with pad_value.
        if pad_end:
            assert pad_value is not None, "Expect pad value is not None if do pad end"

            # Calculate number of frames, using double negatives to round up.
            num_frames = -(-length_samples // frame_step)

            # Pad the signal by up to frame_length samples based on how many samples
            # are remaining starting from last_frame_position.
            pad_samples = np.maximum(0, frame_length + frame_step * (num_frames - 1) - length_samples)

            # Pad the inner dimension of signal by pad_samples.
            paddings = np.concatenate(
                [
                    np.zeros([num_outer_dimensions, 2], dtype=pad_samples.dtype),
                    np.array([[0, pad_samples]]),
                    np.zeros([num_inner_dimensions, 2], dtype=pad_samples.dtype),
                ],
                0,
            )

            signal = np.pad(signal, paddings, constant_values=pad_value)

            signal_shape = signal.shape
            length_samples = signal_shape[axis]

        else:
            num_frames = np.maximum(0, 1 + (length_samples - frame_length) // frame_step).item()

        # Returns the greatest common divisor via Euclid's algorithm
        subframe_length = math.gcd(frame_length, frame_step)

        subframes_per_frame = frame_length // subframe_length
        subframes_per_hop = frame_step // subframe_length
        num_subframes = length_samples // subframe_length

        subframe_shape = np.concatenate([outer_dimensions, [num_subframes, subframe_length], inner_dimensions], 0)

        subframes = np.reshape(signal, subframe_shape)

        frame_selector = np.reshape(np.arange(num_frames) * subframes_per_hop, [num_frames, 1])

        subframe_selector = np.reshape(np.arange(subframes_per_frame), [1, subframes_per_frame])

        selector = frame_selector + subframe_selector

        frames = np.reshape(
            np.take(subframes, selector, axis=axis),
            np.concatenate([outer_dimensions, [num_frames, frame_length], inner_dimensions], 0),
        )

        # Check the result shape
        if result_shape != frames.shape:
            raise ValueError(f"Expect results shape {result_shape}, but got {frames.shape}")

        return frames

    def linear_to_mel_weight_matrix(
        self,
        num_mel_bins=20,
        num_spectrogram_bins=129,
        sample_rate=8000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=3800.0,
        dtype=np.float32,
    ):
        """Returns a matrix to warp linear scale spectrograms to the [mel scale][mel].
        https://en.wikipedia.org/wiki/Mel_scale
        """

        # HTK excludes the spectrogram DC bin.
        bands_to_zero = 1
        nyquist_hertz = sample_rate / 2.0
        linear_frequencies = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins, dtype=dtype)[bands_to_zero:]
        spectrogram_bins_mel = np.expand_dims(self.convert_frequency(linear_frequencies), 1)

        # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
        # center of each band is the lower and upper edge of the adjacent bands.
        # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
        # num_mel_bins + 2 pieces.
        band_edges_mel = self.frame(
            np.linspace(
                self.convert_frequency(lower_edge_hertz), self.convert_frequency(upper_edge_hertz), num_mel_bins + 2
            ),
            frame_length=3,
            frame_step=1,
        )

        # Split the triples up and reshape them into [1, num_mel_bins] tensors.
        lower_edge_mel, center_mel, upper_edge_mel = tuple(
            np.reshape(t, [1, num_mel_bins]) for t in np.split(band_edges_mel, 3, axis=1)
        )

        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the mel domain, not Hertz.
        lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)
        upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)

        # Intersect the line segments with each other and zero.
        mel_weights_matrix = np.maximum(0.0, np.minimum(lower_slopes, upper_slopes))

        # Re-add the zeroed lower bins we sliced out above.
        return np.pad(mel_weights_matrix, [[bands_to_zero, 0], [0, 0]])

    def stft(self, signals, frame_length, frame_step, fft_length=None, pad_end=False):
        """Computes the [Short-time Fourier Transform][stft] of `signals`."""

        if fft_length is None:
            fft_length = int(2 ** np.ceil(np.log(frame_length) / np.log(2.0)))

        framed_signals = self.frame(signals, frame_length, frame_step, pad_end=pad_end)

        # window the framed signals.
        window = self.hann_window(frame_length, dtype=framed_signals.dtype)
        framed_signals *= window

        # produces the (fft_length/2 + 1) unique components of the FFT of the real
        # windowed signals in framed_signals.
        framed_signals = self._pad_for_rfft(framed_signals, fft_length)
        return np.fft.rfft(framed_signals)

    def log_mel_spectrogram(self, raw_speech: np.ndarray, audio_sample_rate: int):
        """Convert waveform to LogMelSpectrogram"""
        window_length_samples = int(round(audio_sample_rate * self.window_length_secs))
        hop_length_samples = int(round(audio_sample_rate * self.hop_length_secs))
        if self.fft_length:
            fft_length = self.fft_length
        else:
            fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        spectrogram = np.abs(
            self.stft(
                raw_speech,
                frame_length=window_length_samples,
                frame_step=hop_length_samples,
                fft_length=fft_length,
            )
        )
        to_mel = self.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=fft_length // 2 + 1,
            sample_rate=audio_sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
            dtype=np.float32,
        )
        mel = spectrogram @ to_mel
        mel = np.transpose(mel, (0, 2, 1)).astype(np.float32)

        # stabilized mel spectrogram
        log_mel = np.log(np.maximum(mel, self.log_floor) + self.log_additive_offset)
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
            mel_speech = self.frame(mel_speech, frame_length=frame_width, frame_step=frame_hop, axis=2)
            mel_speech = np.squeeze(mel_speech, axis=0)
            mel_speech = np.transpose(mel_speech, (1, 2, 0))
            # rescale
            if self.do_normalize:
                mel_speech = mel_speech * (1.0 / 128.0) - 1.0
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
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to compute and prepare for the model one or several sequence(s). sequences.

        Args:
            raw_speech (*np.ndarray*, *List[float]*, *List[np.ndarray]*, *List[List[float]]*):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the *raw_speech* input was sampled. It is strongly recommended to pass
                *sampling_rate* at the forward call to prevent silent errors.
            frame_hop (`int`, *optional*, defaults to 195):
                The number of samples between successive frames.
            required_length (`int`, *optional*, defaults to 32000):
                The minimum length of the sequence after padding.
            num_mel_bins (`int`, *optional*, defaults to 80):
                The number of mel bins.
            frame_width (`int`, *optional*, defaults to 195):
                The number of samples in each frame.
            pad_mode (`str`, *optional*, defaults to 'symmetric'):
                The padding mode to use. Can be one of 'constant', 'symmetric', 'reflect', 'wrap'.
            return_tensors (`str` or [`~file_utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            padding (`bool`, *optional*, defaults to `False`):
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
