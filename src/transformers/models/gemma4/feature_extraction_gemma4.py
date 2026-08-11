# Copyright 2026 Google LLC
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

import math
import warnings
from collections.abc import Sequence

import numpy as np

from ...audio_utils import mel_filter_bank, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging


logger = logging.get_logger(__name__)


def _unfold(array: np.ndarray, dimension: int, size: int, step: int) -> np.ndarray:
    """A basic NumPy equivalent of PyTorch's unfold for 2D arrays along the last dim."""
    if array.ndim != 2:
        raise ValueError("This unfold implementation currently supports 2D arrays (batch, time).")
    if dimension != -1 and dimension != array.ndim - 1:
        raise ValueError("This unfold implementation only supports unfolding the last dimension.")

    batch_size, original_length = array.shape
    num_frames = (original_length - size) // step + 1

    if num_frames <= 0:
        return np.zeros((batch_size, 0, size), dtype=array.dtype)

    output_shape = (batch_size, num_frames, size)
    output_strides = (array.strides[0], array.strides[1] * step, array.strides[1])

    return np.lib.stride_tricks.as_strided(array, shape=output_shape, strides=output_strides)


class Gemma4AudioFeatureExtractor(SequenceFeatureExtractor):
    """An audio feature extractor Universal Speech Models https://huggingface.co/papers/2303.01037.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the attention mask for the generated MEL spectrograms.
        frame_length_ms (`float`, *optional*, defaults to 20.0):
            The length of a frame in milliseconds.
        hop_length_ms (`float`, *optional*, defaults to 10.0):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        min_frequency (`float`, *optional*, defaults to 0.0):
            The minimum frequency (in Hz) for the Mel filterbank.
        max_frequency (`float`, *optional*, defaults to 8000.0):
            The maximum frequency (in Hz) for the Mel filterbank.
        preemphasis (`float`, *optional*, defaults to 0.0):
            The preemphasis coefficient.
        preemphasis_htk_flavor (`bool`, *optional*, defaults to `True`):
            Whether to use HTK-style preemphasis.
        fft_overdrive (`bool`, *optional*, defaults to `False`):
            Whether to use FFT overdrive.
        dither (`float`, *optional*, defaults to 0.0):
            Adds dithering. In other words, adds a small Gaussian noise to each frame.
            E.g. use 0.0001 to add dithering with a normal distribution centered
            around 0.0 with standard deviation 0.0001 (assuming [-1,+1] range of raw_speech).
            The value 0.0 means no dithering.
            Dithering has similar effect as `spectrogram(mel_floor=...)`. It reduces
            the high log_mel_fbank values for signals with hard-zero sections,
            when VAD cutoff is present in the signal.
        input_scale_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor applied to the input waveform.
        mel_floor (`float`, *optional*, defaults to 0.001):
            Minimum value for Mel spectrograms to avoid log(0).
        per_bin_mean (`Optional[Sequence[float]]`, *optional*):
            Mean values for per-bin normalization.
        per_bin_stddev (`Optional[Sequence[float]]`, *optional*):
            Standard deviation values for per-bin normalization.
    """

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        frame_length_ms: float = 20.0,
        hop_length_ms: float = 10.0,
        min_frequency: float = 0.0,
        max_frequency: float = 8000.0,
        preemphasis: float = 0.0,
        preemphasis_htk_flavor: bool = True,
        fft_overdrive: bool = False,
        dither: float = 0.0,
        input_scale_factor: float = 1.0,
        mel_floor: float = 1e-3,
        per_bin_mean: Sequence[float] | None = None,
        per_bin_stddev: Sequence[float] | None = None,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.preemphasis = preemphasis
        self.preemphasis_htk_flavor = preemphasis_htk_flavor
        self.fft_overdrive = fft_overdrive
        self.dither = dither
        self.input_scale_factor = input_scale_factor
        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))
        self.mel_floor = np.array(mel_floor, dtype=np.float64)

        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        if self.fft_overdrive:
            fft_length *= 2
        self.fft_length = fft_length

        # Use periodic Hann window, matching sl.STFT default (signal.hann_window)
        # For even frame_length: window[n] = 0.5 - 0.5 * cos(2*pi*n / frame_length)
        self.window = window_function(self.frame_length).astype(np.float32)

        # Use HuggingFace's mel_filter_bank for compatibility.
        # Suppress the expected warning about all-zero upper mel filters;
        # with fft_length=512 (257 bins) and 128 mel filters the uppermost
        # triangular filter falls between frequency bins, which is harmless.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mel_filters = mel_filter_bank(
                num_frequency_bins=self.fft_length // 2 + 1,
                num_mel_filters=feature_size,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                sampling_rate=self.sampling_rate,
                norm=None,
                mel_scale="htk",
            )

        if per_bin_mean is not None:
            self.per_bin_mean = np.array(per_bin_mean).reshape(1, 1, feature_size)
        else:
            self.per_bin_mean = None

        if per_bin_stddev is not None:
            self.per_bin_stddev = np.array(per_bin_stddev).reshape(1, 1, feature_size)
        else:
            self.per_bin_stddev = None

    def _extract_spectrogram(self, waveform: np.ndarray, attention_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """"""
        if waveform.ndim == 1:  # If single waveform, add batch dimension
            waveform = np.expand_dims(waveform, axis=0)

        if self.dither > 0.0:
            waveform = waveform + self.dither * np.random.randn(*waveform.shape).astype(waveform.dtype)

        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor

        # Semicausal time padding: prepend frame_length // 2 zeros so that the
        # first STFT frame is centered at t=0, matching sl.STFT(time_padding='semicausal').
        pad_left = self.frame_length // 2
        waveform = np.pad(waveform, ((0, 0), (pad_left, 0)), mode="constant")
        attention_mask = np.pad(attention_mask, (pad_left, 0), mode="constant", constant_values=0)

        frame_size_for_unfold = self.frame_length + 1

        # NumPy equivalent of unfold for [B, NumFrames, frame_size_for_unfold]
        frames_to_process = _unfold(waveform, dimension=-1, size=frame_size_for_unfold, step=self.hop_length)

        if self.preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first_in_frame = frames_to_process[..., :1] * (1.0 - self.preemphasis)
                rest_in_frame = frames_to_process[..., 1:-1] - self.preemphasis * frames_to_process[..., :-2]
                frames = np.concatenate([first_in_frame, rest_in_frame], axis=-1)
            else:
                frames = frames_to_process[..., 1:] - self.preemphasis * frames_to_process[..., :-1]
        else:
            frames = frames_to_process[..., :-1]

        # Apply window, then RFFT. np.fft.rfft with n=fft_length implicitly
        # right-pads frames to fft_length.
        frames = frames * self.window  # Broadcasting window
        stft = np.fft.rfft(frames, n=self.fft_length, axis=-1)

        magnitude_spec = np.abs(stft)

        mel_spec = np.matmul(magnitude_spec, self.mel_filters)
        log_mel_spec = np.log(mel_spec + self.mel_floor)

        if self.per_bin_mean is not None:
            log_mel_spec = log_mel_spec - self.per_bin_mean  # Broadcasting

        if self.per_bin_stddev is not None:
            log_mel_spec = log_mel_spec / self.per_bin_stddev  # Broadcasting

        mel_spectrogram = log_mel_spec.squeeze(0)
        num_mel_frames = mel_spectrogram.shape[0]

        # Build a frame-aware mask: a mel frame is valid only when every sample
        # in its analysis window [i*hop, i*hop + frame_size - 1] is real audio.
        # We check this by looking at the last sample of each frame's window.
        frame_end_indices = np.arange(num_mel_frames) * self.hop_length + frame_size_for_unfold - 1
        mask = attention_mask[frame_end_indices].astype(bool)
        return mel_spectrogram, mask

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        padding: bool | str | PaddingStrategy = "longest",
        max_length: int | None = 480_000,
        truncation: bool = True,
        pad_to_multiple_of: int | None = 128,
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = True,
        **kwargs,
    ) -> BatchFeature:
        """Creates a batch of MEL spectrograms from the provided raw speech.

        This implementation uses a different algorithm for windowing and preemphasis compared to the built-in
        `transformers.audio_utils.spectrogram()` function that _will_ result in different outputs. Consider this
        carefully when selecting an audio feature extractor, especially with pre-trained models.

        Args:
            raw_speech:
                The audio for which MEL spectrograms are created.
            padding (`Union[bool, str, PaddingStrategy]`, *optional*, defaults to `"longest"`):
                The padding strategy to use for batches of audio with different lengths.
            max_length (`int`, *optional*, defaults to 480000):
                If provided, defines the maximum length of the audio to allow. Audio longer than this will be
                truncated if `truncation=True`.
            truncation (`bool`, *optional*, defaults to `True`):
                Whether or not to truncate audio above `max_length`.
            pad_to_multiple_of (`int`, *optional*, defaults to 128):
                When padding, pad to a multiple of this value. The default value is defined for optimal TPU support.
            return_tensors (`Union[str, TensorType]`, *optional*, defaults to `None`):
                The type of tensors to return (e.g., NumPy, or Torch).
            return_attention_mask (`bool`, *optional*, defaults to `True`):
                Whether to return the attention mask for the generated MEL spectrograms.
        """

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        is_batched_sequence = isinstance(raw_speech, Sequence) and isinstance(raw_speech[0], (np.ndarray, Sequence))
        is_batched = is_batched_numpy or is_batched_sequence

        if is_batched:
            raw_speech = [np.asarray([rs]).T for rs in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        if not is_batched:  # always return a batch
            raw_speech = [np.asarray([raw_speech])]

        batched_speech = self.pad(
            BatchFeature({"input_features": raw_speech}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        prepared_speech = []
        prepared_speech_mask = []
        for speech, mask in zip(batched_speech.input_features, batched_speech.attention_mask):
            speech, mask = self._extract_spectrogram(speech.T, mask)
            prepared_speech.append(speech.astype(np.float32))
            prepared_speech_mask.append(mask)

        prepared_speech = [speech * mask[..., None] for speech, mask in zip(prepared_speech, prepared_speech_mask)]

        return BatchFeature(
            {"input_features": prepared_speech, "input_features_mask": prepared_speech_mask},
            tensor_type=return_tensors,
        )


__all__ = ["Gemma4AudioFeatureExtractor"]
