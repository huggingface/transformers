# coding=utf-8
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
"""Feature extractor class for CLAP."""


import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torchvision

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class CLAPFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a CLAP feature extractor.

    This feature extractor inherits from [`CLAPFeatureExtractor`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time
    Fourier Transform` which should match pytorch's `torch.stft` equivalent.

    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted MEL spectrograms. This corresponds to the number of frequency bins (intervals) that are computer, for each fourrier step.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz). This only serves to warn users if the audio fed to the feature extractor does not have the same sampling rate.
        hop_length (`int`, defaults to 160):
            Length of the overlaping windows for the STFT used to obtain the Mel Spectrogram. The audio will be split in smaller `frames` with a step of `hop_length` between each frame.
        chunk_length_s (`int`, defaults to 10):
            The maximum input lenght of the model in seconds. This is used to pad the audio.
        n_fft (`int`, defaults to 400):
            Size of the Fourier transform. TODO will properly explain this
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, False):
            Whether or not the model should return the attention masks coresponding to the input.
        frequency_min (`float`, *optional*, 0):
            The lowest frequency of interest. The STFT TODO (not sure) will not be computed for values below this.
        frequency_max (`float`, *optional*, 14_000):
            The highest frequency of interest. The STFT TODO (not sure) will not be computed for values above this.
        top_db (`float`, *optional*):
            The highest decibel value used to convert the mel spectrogram to the log scale. For more details see the `SequenceFeatureExtractor._power_to_db` function
        truncation (`str`, *optional*, `"fusions"`):
            Truncation pattern for long audio inputs. Two patterns are available:
                - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and  a downsampled version of the entire mel spectrogram. These 4 spectrogram will have a dimension of `n_fft, feature_size`. TODO check this
            If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a copy of the original mel obtained from the padded audio. 
                - `rand_trunc` will select a random crop of the mel spectrogram.
        padding (`str`, *optional*, `"repeatpad"`):
            Padding pattern for shorter audio inputs. Three patterns were originaly implemented: 
                - `repeatpad`:
                - `repeat`:
                - `pad`:
    """

    model_input_names = ["input_features", "is_longer"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=48_000,
        hop_length=480,
        chunk_length_s=10,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        frequency_min: float = 0,
        frequency_max: float = 14_000,
        top_db: int = None,
        truncation: str = "fusion",
        padding: str = "repeatpad",
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.top_db = top_db
        self.truncation = truncation
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length_s
        self.nb_max_samples = chunk_length_s * sampling_rate
        self.nb_max_frames = self.nb_max_samples // hop_length
        self.sampling_rate = sampling_rate
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.mel_filters = self.get_mel_filter_banks(
            n_freqs=int(1 + n_fft // 2),
            n_mels=feature_size,
            frequency_min=frequency_min,
            frequency_max=frequency_max,
            sample_rate=sampling_rate,
            norm=None,
            mel_scale="htk",
        )
        self.mel_filters_slaney = self.get_mel_filter_banks(
            n_freqs=int(1 + n_fft // 2),
            n_mels=feature_size,
            frequency_min=frequency_min,
            frequency_max=frequency_max,
            sample_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, excpet for the mel filter banks, which do not need to be saved or printed as they are too long.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "mel_filters_slaney" in output:
            del output["mel_filters_slaney"]
        return output

    def _np_extract_fbank_features(self, waveform: np.array, mel_filters: Optional[np.array] = None) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of the provided audio using the `hanning` window. Two different banks of filters were used:
            - self.
        """
        window = np.hanning(self.n_fft + 1)[:-1]

        frames = self._fram_wave(waveform)
        stft = self._stft(frames, window=window)

        # if the imaginary parts are taken : (real, imag) = stftl; real ** 2 + imag ** 2
        magnitudes = np.abs(stft) ** 2
        mel_spec = np.matmul(mel_filters.T, magnitudes)
        log_mel_spec = self._power_to_db(mel_spec)
        return log_mel_spec.T

    @staticmethod
    # Copied from transformers.models.wav2vec2.feature_extraction_wav2vec2.Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray], attention_mask: List[np.ndarray], padding_value: float = 0.0
    ) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(vector[:length].var() + 1e-7)
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    def _random_mel_fusion(self, mel, total_frames, chunk_frames):
        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
        if len(ranges[1]) == 0:
            # if the audio is too short, we just use the first chunk
            ranges[1] = [0]
        if len(ranges[2]) == 0:
            # if the audio is too short, we just use the first chunk
            ranges[2] = [0]
        # randomly choose index for each part
        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])
        # select mel
        mel_chunk_front = mel[idx_front : idx_front + chunk_frames, :]
        mel_chunk_middle = mel[idx_middle : idx_middle + chunk_frames, :]
        mel_chunk_back = mel[idx_back : idx_back + chunk_frames, :]

        # shrink the mel TODO add this as a numpy function, also no hard codes `64`
        mel_shrink = np.resize(mel, [chunk_frames, self.feature_size])  # current flags are probalby wrong
        import torch

        mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, self.feature_size])(torch.tensor(mel[None]))[0]
        # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

        # stack
        mel_fusion = np.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back, mel_shrink], axis=0)
        return mel_fusion

    def _get_input_mel(self, waveform: np.array, max_length, truncation, padding) -> np.array:
        """
        Possible cases :
            - wave > max_length
                - rand_trun
                - fusion
            - wave < max_length
                - repeat
                - fusion

                TODO the max length should be 10x the sampling rate of the provided audio.

        """
        if waveform.shape[0] > max_length:
            if truncation == "rand_trunc":
                longer = True
                # random crop to max_length (for compatibility) -> this should be handled by self.pad
                overflow = len(waveform) - max_length
                idx = np.random.randint(0, overflow + 1)
                waveform = waveform[idx : idx + max_length]
                input_mel = self._np_extract_fbank_features(waveform, self.mel_filters_slaney)[None,:]
            elif truncation == "fusion":
                mel = self._np_extract_fbank_features(waveform, self.mel_filters)
                chunk_frames = max_length // self.hop_length + 1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is larger than max_length but smaller than max_length+hop_length.
                    # In this case, we just use the whole audio.
                    input_mel = np.stack([mel, mel, mel, mel], axis=0)
                    longer = False
                else:
                    input_mel = self._random_mel_fusion(mel, total_frames, chunk_frames)
                    longer = True
            else:
                raise NotImplementedError(f"data_truncating {truncation} not implemented")

        else:
            longer = False
            # only use repeat as a new possible value for padding. you repeat the audio before applying the usual max_length padding
            if waveform.shape[0] < max_length:
                if padding == "repeat":
                    n_repeat = int(max_length / len(waveform))
                    waveform = np.stack(np.tile(waveform, n_repeat + 1))[:max_length]
                if padding == "repeatpad":
                    n_repeat = int(max_length / len(waveform))
                    waveform = np.stack(np.tile(waveform, n_repeat))
                waveform = np.pad(waveform, (0, max_length - waveform.shape[0]), mode="constant", constant_values=0)

            if truncation == "fusion":
                input_mel = self._np_extract_fbank_features(waveform, self.mel_filters)
                input_mel = np.stack([input_mel, input_mel, input_mel, input_mel], axis=0)
            else:
                input_mel = self._np_extract_fbank_features(waveform, self.mel_filters_slaney)[None, :]

        return input_mel, longer

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: str = "fusion",
        padding: Optional[str] = "repeatpad",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            truncation (`str`, *optional*):
                Truncation pattern for long audio inputs. Two patterns are available:
                    - `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and  a downsampled version of the entire mel spectrogram. These 4 spectrogram will have a dimension of `n_fft, feature_size`. TODO check this
                If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a copy of the original mel obtained from the padded audio. 
                    - `rand_trunc` will select a random crop of the mel spectrogram.
            padding (`str`, *optional*):
                Padding pattern for shorter audio inputs. Three patterns were originaly implemented: 
                    - `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
                    - `repeat`: the audio is repeated and then cut to fit the `max_length`
                    - `pad`: the audio is padded.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.np.array` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
        """
        truncation = truncation if truncation is not None else self.truncation
        padding = padding if padding else self.padding
        
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float64) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float64)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float64)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray(raw_speech)]

        # convert to mel spectrogram, truncate and pad if needed.
        padded_inputs = [
            self._get_input_mel(
                waveform,
                max_length if max_length else self.nb_max_samples,
                truncation,
                padding
            )
            for waveform in raw_speech
        ]

        input_mel = []
        is_longer = []
        for mel, longer in padded_inputs:
            input_mel.append(mel)
            is_longer.append(longer)

        if truncation == "fusion" and sum(is_longer) == 0:
            # if no audio is longer than 10s, then randomly select one audio to be longer
            rand_idx = np.random.randint(0, len(input_mel))
            is_longer[rand_idx] = True

        if isinstance(input_mel[0], List):
            input_mel = [np.asarray(feature, dtype=np.float64) for feature in input_mel]

        input_features = {"input_features": input_mel, "is_longer": is_longer}
        input_features = BatchFeature(input_features)

        if return_tensors is not None:
            input_features = input_features.convert_to_tensors(return_tensors)

        return input_features
