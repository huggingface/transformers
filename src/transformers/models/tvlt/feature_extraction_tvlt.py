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
from numpy.fft import fft

from ...feature_extraction_sequence_utils import BatchFeature, SequenceFeatureExtractor
from ...utils import TensorType, logging


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
        feature_size (`int`, defaults to 128):
            The frequency length of audio spectrogram.
        sampling_rate (`int`, defaults to 44100):
            The sampling rate at which the audio files should be digitalized expressed in Hertz (Hz).
        hop_length_to_sampling_rate (`int`, defaults to 86):
            Hop length is length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
            For example, with sampling rate 44100, the hop length is 512, with 44100 / 512 = 86
        n_fft (`int`, defaults to 2048):
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
        self.mel_filters = self.get_mel_filters(sampling_rate, n_fft, n_mels=feature_size)

    # Copied from transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor.get_mel_filters with 45.245640471924965->59.99247463746737
    def get_mel_filters(self, sr, n_fft, n_mels=128, dtype=np.float32):
        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 59.99247463746737

        mels = np.linspace(min_mel, max_mel, n_mels + 2)

        mels = np.asanyarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

        return weights

    # Copied from transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor.fram_wave
    def fram_wave(self, waveform, center=True):
        """
        Transform a raw waveform into a list of smaller waveforms. The window length defines how much of the signal is
        contain in each frame (smalle waveform), while the hope length defines the step between the beginning of each
        new frame.

        Centering is done by reflecting the waveform which is first centered around `frame_idx * hop_length`.
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
        return np.stack(frames, 0)

    # Copied from transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor.stft
    def stft(self, frames, window):
        """
        Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal. Should give the same
        results as `torch.stft`.
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

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of the provided audio, gives similar results whisper's original torch
        implementation with 1e-5 tolerance.
        """
        window = np.hanning(self.n_fft + 1)[:-1]

        frames = self.fram_wave(waveform)
        stft = self.stft(frames, window=window)
        magnitudes = np.abs(stft[:, :-1]) ** 2

        filters = self.mel_filters
        mel_spec = filters @ magnitudes

        log_spec = 10.0 * np.log10(np.maximum(1e-10, mel_spec))
        log_spec -= 10.0 * np.log10(np.maximum(1e-10, 1.0))
        log_spec = np.maximum(log_spec, log_spec.max() - 80.0)
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
                values, a list of numpy arrays or a list of list of float values.
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

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
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
