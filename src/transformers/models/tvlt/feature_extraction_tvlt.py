# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

from typing import List, Optional, Union

import numpy as np
from numpy.fft import fft

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor, BatchFeature
from ...image_utils import is_torch_tensor
from ...utils import TensorType, is_vision_available, is_speech_available, logging


logger = logging.get_logger(__name__)

if is_speech_available():
    import torchaudio

    
class TvltFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a TVLT audio feature extractor. This feature extractor can be used to prepare audios for the model.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:

    """

    model_input_names = ["audio_values", "audio_masks"]

    def __init__(
        self,
        audio_size=1024,
        num_channels=1,
        audio_patch_size=[16, 16],
        feature_size=128,
        sampling_rate=44100,
        hop_length=512,
        chunk_length=30,
        n_fft=2048,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        self.audio_size = audio_size
        self.num_channels = num_channels
        self.audio_patch_size = audio_patch_size
        self.freq_len = feature_size // self.audio_patch_size[1]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.mel_filters = self.get_mel_filters(sampling_rate, n_fft, n_mels=feature_size)

    def normalize_pixel(self, audio, mean, std):
        # normalize
        if not isinstance(mean, np.ndarray):
            mean = np.array(mean).astype(video.dtype)
        if not isinstance(std, np.ndarray):
            std = np.array(std).astype(video.dtype)

        return (audio - mean[None, None, None]) / std[None, None, None]

    # Copied from transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor.get_mel_filters
    def get_mel_filters(self, sr, n_fft, n_mels=128, dtype=np.float32):
        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

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

    # Copied from transformers.models.whisper.feature_extraction_whisper.WhisperFeatureExtractor._np_extract_fbank_features
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

        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.T

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several video(s) or image(s) and audio(s).

        <Tip warning={true}>

        NumPy arrays are converted to PIL images when resizing, so the most efficient is to pass PIL images.

        </Tip>

        Args:

            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            truncation (`bool`, *optional*, default to `True`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*, defaults to None):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.
                [What are attention masks?](../glossary#attention-mask)
                <Tip>
                For WhisperTransoformer models, `attention_mask` should alwys be passed for batched inference, to avoid
                subtle bugs.
                </Tip>
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
                pipeline.
            padding_value (`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **audio_values** -- Audio values to be fed to a model, of shape (batch_size, num_channels, height, width).

            - **audio_masks** -- Audio masks to be fed to a model, of shape (batch_size, num_audio_channels).

        Main method to featurize and prepare for the model one or several sequence(s).
        Args:

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

        audio_features = [
            self._np_extract_fbank_features(waveform.squeeze())[: self.audio_size] for waveform in raw_speech
        ]
        if isinstance(audio_features[0], List):
            audio_features = [np.asarray(feature, dtype=np.float32) for feature in audio_features]
        max_patch_len = max([feature.shape[0] // self.audio_patch_size[0] * self.freq_len for feature in audio_features])
        # Pad to multiple of audio patch size
        max_patch_len = (max_patch_len // 16 + 1) * 16 if max_patch_len % 16 > 0 else max_patch_len
        max_time_len = max_patch_len // self.freq_len * self.audio_patch_size[0]
        audio_masks = [
            (feature.shape[0] // self.audio_patch_size[0] * self.freq_len) * [1]
            + (max_patch_len - feature.shape[0] // self.audio_patch_size[0] * self.freq_len) * [0]
            for feature in audio_features
        ]
        audio_masks = np.array(audio_masks).astype(np.float32)

        # convert into correct format for padding
        padded_audio_features = np.zeros([len(audio_features), 1, max_time_len, 128]).astype(np.float32)
        for i in range(len(audio_features)):
            feature = audio_features[i]
            padded_audio_features[i, :, : feature.shape[0], :] = feature

        # return as BatchFeature
        data = {"audio_values": padded_audio_features, "audio_masks": audio_masks}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
