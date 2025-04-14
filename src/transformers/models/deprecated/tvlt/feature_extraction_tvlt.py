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
"""Feature extractor class for TVLT."""

from math import ceil
from typing import List, Optional, Union

import numpy as np

from ....audio_utils import mel_filter_bank, spectrogram, window_function
from ....feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor
from ....utils import TensorType, logging


logger = logging.get_logger(__name__)


class TvltFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a TVLT audio feature extractor. This feature extractor can be used to prepare audios for the model.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        spectrogram_length (`Dict[str, int]` *optional*, defaults to 2048):
            The time length of each audio spectrogram.
        num_channels (`int` *optional*, defaults to 1):
            Number of audio channels.
        patch_size (`List[int]` *optional*, defaults to `[16, 16]`):
            The patch size of audio patch embedding.
        feature_size (`int`, *optional*, defaults to 128):
            The frequency length of audio spectrogram.
        sampling_rate (`int`, *optional*, defaults to 44100):
            The sampling rate at which the audio files should be digitalized expressed in Hertz (Hz).
        hop_length_to_sampling_rate (`int`, *optional*, defaults to 86):
            Hop length is length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
            For example, with sampling rate 44100, the hop length is 512, with 44100 / 512 = 86
        n_fft (`int`, *optional*, defaults to 2048):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """

    model_input_names = ["audio_values", "audio_mask"]

    def __init__(
        self,
        spectrogram_length=2048,
        num_channels=1,
        patch_size=[16, 16],
        feature_size=128,
        sampling_rate=44100,
        hop_length_to_sampling_rate=86,
        n_fft=2048,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )

        self.spectrogram_length = spectrogram_length
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.freq_len = feature_size // self.patch_size[1]
        self.n_fft = n_fft
        self.hop_length = sampling_rate // hop_length_to_sampling_rate
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=22050.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        ).T

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
        log_spec = spectrogram(
            waveform,
            window_function(self.n_fft, "hann"),
            frame_length=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
            mel_filters=self.mel_filters.T,
            log_mel="dB",
            db_range=80.0,
        )
        log_spec = log_spec[:, :-1]
        log_spec = log_spec - 20.0
        log_spec = np.clip(log_spec / 40.0, -2.0, 0.0) + 1.0
        return log_spec

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = True,
        sampling_rate: Optional[int] = None,
        resample: bool = False,
        mask_audio: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare one or several audio(s) for the model.

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            return_attention_mask (`bool`, *optional*, default to `True`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default. [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For TvltTransformer models, `attention_mask` should alwys be passed for batched inference, to avoid
                subtle bugs.

                </Tip>

            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline. Current model supports sampling rate 16000 and 44100.
            resample (`bool`, *optional*, defaults to `False`):
                If the sampling rate is not matched, resample the input audio to match.
            mask_audio (`bool`, *optional*, defaults to `False`):
                Whether or not to mask input audio for MAE task.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **audio_values** -- Audio values to be fed to a model, of shape (batch_size, num_channels, height,
              width).

            - **audio_mask** -- Audio masks to be fed to a model, of shape (batch_size, num_audio_patches).
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    "This feature extractor is set to support sampling rate"
                    f" of {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled"
                    f" with {self.sampling_rate} and not {sampling_rate}."
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
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)
        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        # Convert audio signals to log mel spectrograms, truncate by time axis
        audio_features = [
            self._np_extract_fbank_features(waveform.squeeze()).T[: self.spectrogram_length] for waveform in raw_speech
        ]
        if isinstance(audio_features[0], List):
            audio_features = [np.asarray(feature, dtype=np.float32) for feature in audio_features]

        # Create audio attention mask
        max_patch_len = max(
            [ceil(feature.shape[0] / self.patch_size[0]) * self.freq_len for feature in audio_features]
        )  # The maximum number of audio patches in a batch
        if return_attention_mask:
            audio_mask = [
                (ceil(feature.shape[0] / self.patch_size[0]) * self.freq_len) * [1]
                + (max_patch_len - ceil(feature.shape[0] / self.patch_size[0]) * self.freq_len) * [0]
                for feature in audio_features
            ]
            audio_mask = np.array(audio_mask).astype(np.float32)

        # convert into correct format for padding
        max_time_len = max_patch_len // self.freq_len * self.patch_size[0]  # The maximum audio size in a batch
        padded_audio_features = np.ones([len(audio_features), 1, max_time_len, self.feature_size]).astype(np.float32)
        padded_audio_features = padded_audio_features * self.padding_value
        for i in range(len(audio_features)):
            feature = audio_features[i]
            padded_audio_features[i, :, : feature.shape[0], :] = feature

        # return as BatchFeature
        if return_attention_mask:
            data = {"audio_values": padded_audio_features, "audio_mask": audio_mask}
        else:
            data = {"audio_values": padded_audio_features}

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)
        return encoded_inputs


__all__ = ["TvltFeatureExtractor"]
