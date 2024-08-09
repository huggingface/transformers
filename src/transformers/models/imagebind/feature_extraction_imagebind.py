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
"""Feature extractor class for ImageBind."""

from fractions import Fraction
from typing import List, Optional, Tuple, Union

import numpy as np

from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_speech_available, is_torch_available, logging


if is_speech_available():
    import torchaudio.compliance.kaldi as ta_kaldi

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


def valid_batched_clipped_audio(raw_speech):
    """
    Determines whether raw mono-channel audio input (or any other 1D data) is batched and clipped. The following
    conditions will be recognized as valid audio:

    - unbatched: `List[float]`, `np.ndarray` (`ndim=1`)
    - batched: `List[List[float]]`, `List[np.ndarray]` (`ndim=1`), `np.ndarray` (`ndim=2`)
    - batched and clipped: `List[List[List[float]]]`, `List[List[np.ndarray]]` (`ndim=1`), List[np.ndarray] (`ndim=2`), np.ndarray (`ndim=3`)
    """
    if isinstance(raw_speech, np.ndarray):
        return 1 <= raw_speech.ndim <= 3
    if isinstance(raw_speech, (list, tuple)):
        first_elem = raw_speech[0]
        if isinstance(first_elem, float):
            return True
        if isinstance(first_elem, np.ndarray):
            return 1 <= first_elem.ndim <= 2
        if isinstance(first_elem, (list, tuple)):
            second_elem = first_elem[0]
            if isinstance(second_elem, (float, np.ndarray)):
                return True
            if isinstance(second_elem, (list, tuple)):
                return isinstance(second_elem[0], float)

    return False


def convert_raw_speech_to_numpy_array(raw_speech):
    """If not already in numpy array format, convert raw_speech to a numpy array."""
    if isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], float):
        raw_speech = [[np.asarray(raw_speech, dtype=np.float32)]]
    elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], (list, tuple)):
        if isinstance(raw_speech[0][0], float):
            # List[List[float]]
            raw_speech = [[np.asarray(audio, dtype=np.float32)] for audio in raw_speech]
        elif isinstance(raw_speech[0][0], (list, tuple)):
            # List[List[List[float]]]
            raw_speech = [[np.asarray(audio, dtype=np.float32) for audio in clip] for clip in raw_speech]

    return raw_speech


def batch_and_clip_ndarray(array, data_dim=1, dtype=np.float32):
    """
    Turns a possibly nested list of np.ndarrays into a batched and clipped output of type `List[List[np.ndarray]]`.
    """
    if (
        isinstance(array, (list, tuple))
        and isinstance(array[0], (list, tuple))
        and isinstance(array[0][0], np.ndarray)
    ):
        if array[0][0].ndim == data_dim:
            return [[base_array.astype(dtype=dtype) for base_array in clips] for clips in array]
        else:
            raise ValueError(
                f"`For List[List[np.ndarray]]` inputs the internal `np.ndarray`s are expected to have dimension"
                f" {data_dim} but got dimension {array[0][0].ndim}"
            )
    elif isinstance(array, (list, tuple)) and isinstance(array[0], np.ndarray):
        if array[0].ndim == data_dim + 1:
            return [[np.asarray(base_array, dtype=dtype) for base_array in clips] for clips in array]
        elif array[0].ndim == data_dim:
            return [[base_array.astype(dtype=dtype)] for base_array in array]
        else:
            raise ValueError(
                f"For `List[np.ndarray]` inputs the internal `np.ndarray`s are expected to have dimension"
                f" {data_dim} or {data_dim + 1} but got dimension {array[0].ndim}"
            )
    elif isinstance(array, np.ndarray):
        array = array.astype(dtype=dtype)
        if array.ndim == data_dim + 2:
            return [list(clips) for clips in array]
        elif array.ndim == data_dim + 1:
            return [[clip] for clip in array]
        elif array.ndim == data_dim:
            return [[array]]
        else:
            raise ValueError(
                f"`np.ndarray` inputs are expected to have dimension in"
                f" `[{data_dim}, {data_dim + 1}, {data_dim + 2}]` but instead got {array.ndim}"
            )
    else:
        raise ValueError(f"Could not make batched and clipped audio from {array}")


# Adapted from https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/clip_sampling.py#L346
def uniform_chunk_sampling(
    total_duration: float, chunk_duration: float, num_chunks: int
) -> List[Tuple[Fraction, Fraction]]:
    """
    Uniformly sample `num_chunks` chunks of duration `chunk_duration` from an audio/video of total duration `total_duration`.

    Args:
        total_duration (`float`): s
            Total duration of the audio/video.
        chunk_duration (`float`):
            Duration of each chunk.
        num_chunks (`int`):
            Number of chunks to sample.
    """
    chunk_duration_fraction = Fraction(chunk_duration)
    max_possible_clip_start = Fraction(max(total_duration - chunk_duration, 0))
    uniform_clip = Fraction(max_possible_clip_start / max(num_chunks - 1, 1))

    result = []
    for clip_index in range(num_chunks):
        clip_start_sec = uniform_clip * clip_index
        clip_end_sec = clip_start_sec + chunk_duration_fraction
        result.append((clip_start_sec, clip_end_sec))

    return result


class ImageBindFeatureExtractor(SequenceFeatureExtractor):
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
        padding_value (`float`, *optional*, defaults to 0.0):
            The value to pad with when applying the padding strategy defined by the `padding` argument to
            [ImageBindAudioFeatureExtractor.__call__`].
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the log-Mel features using `mean` and `std`.
        mean (`float`, *optional*, defaults to -4.268):
            The mean value used to normalize the log-Mel features. Uses the AudioSet mean by default.
        std (`float`, *optional*, defaults to 9.138):
            The standard deviation value used to normalize the log-Mel features. Uses the AudioSet standard deviation
            by default.
        do_chunk (`bool`, *optional*, defaults to `True`):
            Whether or not to sample multiple chunks from the input audio. If `False`, the entire audio will be used.
        chunk_duration (`float`, *optional*, defaults to 2.0):
            The duration of each chunk in seconds.
        num_chunks (`int`, *optional*, defaults to 3):
            The number of chunks to sample from the input audio.
    """

    model_input_names = ["input_features"]

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
        do_chunk=True,
        chunk_duration=2.0,
        num_chunks=3,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
        self.do_chunk = do_chunk
        self.chunk_duration = chunk_duration
        self.num_chunks = num_chunks

        if not is_speech_available():
            mel_filters = mel_filter_bank(
                num_frequency_bins=256,
                num_mel_filters=self.num_mel_bins,
                min_frequency=20,
                max_frequency=sampling_rate // 2,
                sampling_rate=sampling_rate,
                norm=None,
                mel_scale="kaldi",
                triangularize_in_mel_space=True,
            )

            self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
            self.window = window_function(400, "hann", periodic=False)

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

        if is_speech_available():
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform,
                sample_frequency=self.sampling_rate,
                window_type="hanning",
                num_mel_bins=self.num_mel_bins,
            )
        else:
            waveform = np.squeeze(waveform)
            fbank = spectrogram(
                waveform,
                self.window,
                frame_length=400,
                hop_length=160,
                fft_length=512,
                power=2.0,
                center=False,
                preemphasis=0.97,
                mel_filters=self.mel_filters,
                log_mel="log",
                mel_floor=1.192092955078125e-07,
                remove_dc_offset=True,
            ).T

            fbank = torch.from_numpy(fbank)

        # Convert to [mel_bins, num_frames] shape
        fbank = fbank.transpose(0, 1)
        # pad to max_length
        n_frames = fbank.size(1)
        difference = max_length - n_frames

        if abs(difference) / n_frames > 0.2:
            logger.warning_once(
                f"Large padding or truncation for {tuple(waveform.shape)} waveform with {n_frames} frames and {max_length} max_length."
            )

        # pad or truncate
        if difference > 0:
            fbank = torch.nn.functional.pad(fbank, (0, difference), mode="constant", value=0)
        elif difference < 0:
            fbank = fbank[:, 0:max_length]
        # Add 1 channel so that dimension of fbank is [1, num_mel_bins, num_frames]
        fbank = fbank.unsqueeze(0)
        fbank = fbank.numpy()

        return fbank

    def normalize(self, input_values: np.ndarray, mean: float, std: float) -> np.ndarray:
        return (input_values - (mean)) / (std)

    def chunk(self, raw_speech: np.ndarray, chunk_duration: float, num_chunks: int) -> List[np.ndarray]:
        audio_duration = raw_speech.shape[0] / self.sampling_rate
        if chunk_duration > audio_duration:
            logger.warning_once(
                "Chunk duration is greater than audio duration. Chunks will be repeated, consider adjusting either `chunk_duration` or `num_chunks`"
                "to avoid unnecessary memory/compute usage."
            )
        all_clips_timepoints = uniform_chunk_sampling(audio_duration, chunk_duration, num_chunks)

        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = raw_speech[
                int(clip_timepoints[0] * self.sampling_rate) : int(clip_timepoints[1] * self.sampling_rate)
            ]
            all_clips.append(waveform_clip)

        return all_clips

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]], List[List[List[float]]]],
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        do_chunk: Optional[bool] = None,
        chunk_duration: Optional[float] = None,
        num_chunks: Optional[int] = None,
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
            do_normalize (`bool`, *optional*, defaults `self.do_normalize`):
                Whether or not to normalize the log-Mel features.
            mean (`float`, *optional*, defaults `self.mean`):
                The mean value used to normalize the log-Mel features.
            std (`float`, *optional*, defaults `self.std`):
                The standard deviation value used to normalize the log-Mel features.
            do_chunk (`bool`, *optional*, defaults `self.do_chunk`):
                Whether or not to sample multiple chunks from the input audio. If `False`, the entire audio will be used.
            chunk_duration (`float`, *optional*, defaults `self.chunk_duration`):
                The duration of each chunk in seconds.
            num_chunks (`int`, *optional*, defaults `self.num_chunks`):
                The number of chunks to sample from the input audio. If audio duration is less than `chunk_duration` * `num_chunks`,
                chunks will overlap to cover the entire audio. If `chunk_duration` is greater than audio duration, the
                chunks will be repeated until `num_chunks` is reached.
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

        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        mean = mean if mean is not None else self.mean
        std = std if std is not None else self.std
        do_chunk = do_chunk if do_chunk is not None else self.do_chunk
        chunk_duration = chunk_duration if chunk_duration is not None else self.chunk_duration
        num_chunks = num_chunks if num_chunks is not None else self.num_chunks

        raw_speech = convert_raw_speech_to_numpy_array(raw_speech)
        raw_speech = batch_and_clip_ndarray(raw_speech, data_dim=1, dtype=np.float32)

        if do_chunk and len(raw_speech[0]) == 1:
            raw_speech = [self.chunk(audio[0], chunk_duration, num_chunks) for audio in raw_speech]

        features = [
            [self._extract_fbank_features(waveform, max_length=self.max_length) for waveform in clip]
            for clip in raw_speech
        ]

        features = np.asarray(features)
        padded_inputs = BatchFeature({"input_features": features})

        if do_normalize:
            padded_inputs["input_features"] = [
                [self.normalize(feature, mean, std) for feature in clip] for clip in padded_inputs["input_features"]
            ]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs
