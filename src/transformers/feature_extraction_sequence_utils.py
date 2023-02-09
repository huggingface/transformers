# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
 Sequence feature extraction class for common feature extractors to preprocess sequences.
"""
import math
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.fft import fft

from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .utils import PaddingStrategy, TensorType, is_tf_tensor, is_torch_tensor, logging, to_numpy


logger = logging.get_logger(__name__)


class SequenceFeatureExtractor(FeatureExtractionMixin):
    """
    This is a general feature extraction class for speech recognition.

    Args:
        feature_size (`int`):
            The feature dimension of the extracted features.
        sampling_rate (`int`):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`):
            The value that is used to fill the padding values / vectors.
    """

    def __init__(self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

        self.padding_side = kwargs.pop("padding_side", "right")
        self.return_attention_mask = kwargs.pop("return_attention_mask", True)

        super().__init__(**kwargs)

    def pad(
        self,
        processed_features: Union[
            BatchFeature,
            List[BatchFeature],
            Dict[str, BatchFeature],
            Dict[str, List[BatchFeature]],
            List[Dict[str, BatchFeature]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
        max sequence length in the batch.

        Padding side (left/right) padding values are defined at the feature extractor level (with `self.padding_side`,
        `self.padding_value`)

        <Tip>

        If the `processed_features` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            processed_features ([`BatchFeature`], list of [`BatchFeature`], `Dict[str, List[float]]`, `Dict[str, List[List[float]]` or `List[Dict[str, List[float]]]`):
                Processed inputs. Can represent one input ([`BatchFeature`] or `Dict[str, List[float]]`) or a batch of
                input values / vectors (list of [`BatchFeature`], *Dict[str, List[List[float]]]* or *List[Dict[str,
                List[float]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], (dict, BatchFeature)):
            processed_features = {
                key: [example[key] for example in processed_features] for key in processed_features[0].keys()
            }

        # The model's main input name, usually `input_values`, has be passed for padding
        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                "You should supply an instance of `transformers.BatchFeature` or list of `transformers.BatchFeature`"
                f" to this method that includes {self.model_input_names[0]}, but you provided"
                f" {list(processed_features.keys())}"
            )

        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = (
            return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        )

        if not required_input:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        # If we have PyTorch/TF tensors or lists as inputs, we cast them as Numpy arrays
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]

        if return_tensors is None:
            if is_tf_tensor(first_element):
                return_tensors = "tf"
            elif is_torch_tensor(first_element):
                return_tensors = "pt"
            elif isinstance(first_element, (int, float, list, tuple, np.ndarray)):
                return_tensors = "np"
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

        for key, value in processed_features.items():
            if isinstance(value[0], (int, float)):
                processed_features[key] = to_numpy(value)
            else:
                processed_features[key] = [to_numpy(v) for v in value]

        # Convert padding_strategy in PaddingStrategy
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)

        required_input = processed_features[self.model_input_names[0]]

        batch_size = len(required_input)
        if not all(len(v) == batch_size for v in processed_features.values()):
            raise ValueError("Some items in the output dictionary have a different batch size than others.")

        truncated_inputs = []
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in processed_features.items())
            # truncation
            inputs_slice = self._truncate(
                inputs,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                truncation=truncation,
            )
            truncated_inputs.append(inputs_slice)

        if padding_strategy == PaddingStrategy.LONGEST:
            # make sure that `max_length` cannot be longer than the longest truncated length
            max_length = max(len(input_slice[self.model_input_names[0]]) for input_slice in truncated_inputs)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            # padding
            outputs = self._pad(
                truncated_inputs[i],
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                if value.dtype is np.dtype(np.float64):
                    value = value.astype(np.float32)
                batch_outputs[key].append(value)

        return BatchFeature(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        processed_features: Union[Dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            processed_features:
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length: maximum length of the returned list and optionally padding length (see below)
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The feature_extractor padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        required_input = processed_features[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) < max_length

        if return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = np.ones(len(required_input), dtype=np.int32)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (0, difference)
                    )
                padding_shape = ((0, difference), (0, 0)) if self.feature_size > 1 else (0, difference)
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input, padding_shape, "constant", constant_values=self.padding_value
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (difference, 0)
                    )
                padding_shape = ((difference, 0), (0, 0)) if self.feature_size > 1 else (difference, 0)
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input, padding_shape, "constant", constant_values=self.padding_value
                )
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return processed_features

    def _truncate(
        self,
        processed_features: Union[Dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        truncation: Optional[bool] = None,
    ):
        """
        Truncate inputs to predefined length or max length in the batch

        Args:
            processed_features:
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length:
                maximum length of the returned list and optionally padding length (see below)
            pad_to_multiple_of (optional) :
                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to
                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs
                which benefit from having sequence lengths be a multiple of 128.
            truncation (optional):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
        """
        if not truncation:
            return processed_features
        elif truncation and max_length is None:
            raise ValueError("When setting ``truncation=True``, make sure that ``max_length`` is defined.")

        required_input = processed_features[self.model_input_names[0]]

        # find `max_length` that fits `pad_to_multiple_of`
        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_truncated = len(required_input) > max_length

        if needs_to_be_truncated:
            processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]
            if "attention_mask" in processed_features:
                processed_features["attention_mask"] = processed_features["attention_mask"][:max_length]

        return processed_features

    def _get_padding_strategies(self, padding=False, max_length=None):
        """
        Find the correct padding strategy
        """

        # Get padding strategy
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined"
                )

        # Test if we have a padding value
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.padding_value is None):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy

    @staticmethod
    def hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
        """Convert Hz to Mels.

        Args:
            freqs (`float`):
                Frequencies in Hz
            mel_scale (`str`, *optional*, defaults to `"htk"`):
                Scale to use, `htk` or `slaney`.

        Returns:
            mels (`float`):
                Frequency in Mels
        """

        if mel_scale not in ["slaney", "htk"]:
            raise ValueError('mel_scale should be one of "htk" or "slaney".')

        if mel_scale == "htk":
            return 2595.0 * math.log10(1.0 + (freq / 700.0))

        # Fill in the linear part
        frequency_min = 0.0
        f_sp = 200.0 / 3

        mels = (freq - frequency_min) / f_sp

        # Fill in the log-scale part
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - frequency_min) / f_sp
        logstep = math.log(6.4) / 27.0

        if freq >= min_log_hz:
            mels = min_log_mel + math.log(freq / min_log_hz) / logstep

        return mels

    @staticmethod
    def mel_to_hz(mels: np.array, mel_scale: str = "htk") -> np.array:
        """Convert mel bin numbers to frequencies.

        Args:
            mels (`np.array`):
                Mel frequencies
            mel_scale (`str`, *optional*, `"htk"`):
                Scale to use: `htk` or `slaney`.

        Returns:
            freqs (`np.array`):
                Mels converted in Hz
        """

        if mel_scale not in ["slaney", "htk"]:
            raise ValueError('mel_scale should be one of "htk" or "slaney".')

        if mel_scale == "htk":
            return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

        # Fill in the linear scale
        frequency_min = 0.0
        f_sp = 200.0 / 3
        freqs = frequency_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - frequency_min) / f_sp
        logstep = math.log(6.4) / 27.0

        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        return freqs

    @staticmethod
    def create_triangular_filterbank(
        all_freqs: np.array,
        f_pts: np.array,
    ) -> np.array:
        """Create a triangular filter bank.


        Args:
            all_freqs (`np.array`):
                STFT freq points of size (`n_freqs`).
            f_pts (`np.array`):
                Filter mid points of size (`n_filter`).

        Returns:
            fb (np.array):
                The filter bank of size (`n_freqs`, `n_filter`).
        """
        # Adapted from Librosa
        # calculate the difference between each filter mid point and each stft freq point in hertz
        f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
        slopes = np.expand_dims(f_pts, 0) - np.expand_dims(all_freqs, 1)  # (n_freqs, n_filter + 2)
        # create overlapping triangles
        zero = np.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
        up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
        fb = np.maximum(zero, np.minimum(down_slopes, up_slopes))

        return fb

    def get_mel_filter_banks(
        self,
        n_freqs: int,
        frequency_min: float,
        frequency_max: float,
        n_mels: int,
        sample_rate: int,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
    ) -> np.array:
        """
        Create a frequency bin conversion matrix used to obtain the Mel Spectrogram. This is called a *mel filter
        bank*, and various implementation exist, which differ in the number of filters, the shape of the filters, the
        way the filters are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The
        goal of these features is to approximate the non-linear human perception of the variation in pitch with respect
        to the frequency. This code is heavily inspired from the *torchaudio* implementation, see
        [here](https://pytorch.org/audio/stable/transforms.html) for more details.


        Note:
            Different banks of MEL filters were introduced in the litterature. The following variation are supported:
                - MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHz and a speech bandwidth of `[0, 4600]` Hz
                - MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a speech bandwidth `[0, 8000]` Hz (sampling rate ≥ 16 kHz).
                - MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate of 16 kHz, and speech bandwidth [133, 6854] Hz. This version also includes an area normalization.
                - HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes sampling rate of 12.5 kHz and speech bandwidth [0, 6250] Hz
            The default parameters of `torchaudio`'s mel filterbanks implement the `"htk"` filers while `torchlibrosa` uses the `"slaney"` implementation.

        Args:
            n_freqs (`int`):
                Number of frequencies to highlight/apply.
            frequency_min (`float`):
                Minimum frequency of interest(Hz).
            frequency_max (`float`):
                Maximum frequency of interest(Hz).
            n_mels (`int`):
                Number of mel filterbanks.
            sample_rate (`int`):
                Sample rate of the audio waveform
            norm (`str`, *optional*):
                If "slaney", divide the triangular mel weights by the width of the mel band (area normalization).
            mel_scale (`str`, *optional*, `"htk"`):
                Scale to use: `htk` or `slaney`. (Default: `htk`)

        Returns:
            Tensor: Triangular filter banks (fb matrix) of size (`n_freqs`, `n_mels`) meaning number of frequencies to
            highlight/apply to x the number of filterbanks. Each column is a filterbank so that assuming there is a
            matrix A of size (..., `n_freqs`), the applied result would be `A * melscale_fbanks(A.size(-1), ...)`.

        """

        if norm is not None and norm != "slaney":
            raise ValueError('norm must be one of None or "slaney"')

        # freq bins
        all_freqs = np.linspace(0, sample_rate // 2, n_freqs)

        # calculate mel freq bins
        m_min = self.hz_to_mel(frequency_min, mel_scale=mel_scale)
        m_max = self.hz_to_mel(frequency_max, mel_scale=mel_scale)

        m_pts = np.linspace(m_min, m_max, n_mels + 2)
        f_pts = self.mel_to_hz(m_pts, mel_scale=mel_scale)

        # create filterbank
        filterbank = self.create_triangular_filterbank(all_freqs, f_pts)

        if norm is not None and norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
            filterbank *= np.expand_dims(enorm, 0)

        if (filterbank.max(axis=0) == 0.0).any():
            warnings.warn(
                "At least one mel filterbank has all zero values. "
                f"The value for `n_mels` ({n_mels}) may be set too high. "
                f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
            )

        return filterbank

    def _stft(self, frames, window):
        """
        Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same
        results as `torch.stft`. #TODO @Arthur batching this could allow more usage, good first issue.

        Args:
            frames (`np.array` of dimension `(num_frames, self.n_fft)`):
                A framed audio signal obtained using `self._fram_wav`.
            window (`np.array` of dimension `(self.n_freqs, self.n_mels)`:
                A array reprensenting the function that will be used to reduces the amplitude of the discontinuities at
                the boundaries of each frame when computing the FFT. Each frame will be multiplied by the window. For
                more information on this phenomena, called *Spectral leakage*, refer to [this
                tutorial]https://download.ni.com/evaluation/pxi/Understanding%20FFTs%20and%20Windowing.pdf
        """
        frame_size = frames.shape[1]
        fft_size = self.n_fft

        if fft_size is None:
            fft_size = frame_size

        if fft_size < frame_size:
            raise ValueError("FFT size must greater or equal the frame size")
        # number of FFT bins to store
        num_fft_bins = (fft_size >> 1) + 1

        data = np.empty((len(frames), num_fft_bins), dtype=np.complex64)
        fft_signal = np.zeros(fft_size)

        for f, frame in enumerate(frames):
            if window is not None:
                np.multiply(frame, window, out=fft_signal[:frame_size])
            else:
                fft_signal[:frame_size] = frame
            data[f] = fft(fft_signal, axis=0)[:num_fft_bins]
        return data.T

    def _power_to_db(self, mel_spectrogram, a_min=1e-10, ref=1.0):
        """
        Convert a mel spectrogram from power to db scale, this function is the numpy implementation of
        librosa.power_to_lb.
        
        Note: 
            The motivation behind applying the log function on the mel spectrgram is that humans do not hear loudness on a linear scale.
            Generally to double the percieved volume of a sound we need to put 8 times as much energy into it. This means that large variations 
            in energy may not sound all that different if the sound is loud to begin with. This compression operation makes the mel features match
            more closely what humans actually hear.
        """
        log_spec = 10 * np.log10(np.clip(mel_spectrogram, a_min=a_min, a_max=None))
        log_spec -= 10.0 * np.log10(np.maximum(a_min, ref))
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = np.clip(log_spec, min=np.maximum(log_spec) - self.top_db, max=np.inf)
        return log_spec

    def _fram_wave(self, waveform: np.array, center: bool = True):
        """
        In order to compute the short time fourier transform, the waveform needs to be split in overlapping windowed
        segments called `frames`.

        The window length (self.window_length) defines how much of the signal is contained in each frame, while the hop
        length defines the step between the beginning of each new frame.

        #TODO @Arthur **This method does not support batching yet as we are mainly focus on inference. If you want this
        to be added feel free to open an issue and ping @arthurzucker on Github**

        Args:
            waveform (`np.array`) of shape (sample_length,):
                The raw waveform which will be split into smaller chunks.
            center (`bool`, defaults to `True`):
                Whether or not to center each frame around the middle of the frame. Centering is done by reflecting the
                waveform on the left and on the right.

        Return:
            framed_waveform (`np.array` of shape (waveform.shape // self.hop_length , self.n_fft)):
                The framed waveforms that can be fed to `np.fft`.
        """
        frames = []
        for i in range(0, waveform.shape[0] + 1, self.hop_length):
            half_window = (self.n_fft - 1) // 2 + 1
            if center:
                start = i - half_window if i > half_window else 0
                end = i + half_window if i < waveform.shape[0] - half_window else waveform.shape[0]
                frame = waveform[start:end]
                if start == 0:
                    padd_width = (-i + half_window, 0)
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

                elif end == waveform.shape[0]:
                    padd_width = (0, (i - waveform.shape[0] + half_window))
                    frame = np.pad(frame, pad_width=padd_width, mode="reflect")

            else:
                frame = waveform[i : i + self.n_fft]
                frame_width = frame.shape[0]
                if frame_width < waveform.shape[0]:
                    frame = np.lib.pad(
                        frame, pad_width=(0, self.n_fft - frame_width), mode="constant", constant_values=0
                    )
            frames.append(frame)

        frames = np.stack(frames, 0)
        return frames
