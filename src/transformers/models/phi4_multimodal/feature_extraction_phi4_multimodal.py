# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
Processor class for Phi4Multimodal
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.image_processing_utils import BatchFeature
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


AudioInput = Tuple[Union[np.ndarray, torch.Tensor], int]


def speechlib_mel(sample_rate, n_fft, n_mels, fmin=None, fmax=None):
    """Create a Mel filter-bank the same as SpeechLib FbankFC.

    Args:
        sample_rate (int): Sample rate in Hz. number > 0 [scalar]
        n_fft (int): FFT size. int > 0 [scalar]
        n_mel (int): Mel filter size. int > 0 [scalar]
        fmin (float): lowest frequency (in Hz). If None use 0.0.
            float >= 0 [scalar]
        fmax: highest frequency (in Hz). If None use sample_rate / 2.
            float >= 0 [scalar]

    Returns
        out (numpy.ndarray): Mel transform matrix
            [shape=(n_mels, 1 + n_fft/2)]
    """

    bank_width = int(n_fft // 2 + 1)
    if fmax is None:
        fmax = sample_rate / 2
    if fmin is None:
        fmin = 0
    assert fmin >= 0, "fmin cannot be negtive"
    assert fmin < fmax <= sample_rate / 2, "fmax must be between (fmin, samplerate / 2]"

    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    def bin2mel(fft_bin):
        return 1127.0 * np.log(1.0 + fft_bin * sample_rate / (n_fft * 700.0))

    def f2bin(f):
        return int((f * n_fft / sample_rate) + 0.5)

    # Spec 1: FFT bin range [f2bin(fmin) + 1, f2bin(fmax) - 1]
    klo = f2bin(fmin) + 1
    khi = f2bin(fmax)

    khi = max(khi, klo)

    # Spec 2: SpeechLib uses trianges in Mel space
    mlo = mel(fmin)
    mhi = mel(fmax)
    m_centers = np.linspace(mlo, mhi, n_mels + 2)
    ms = (mhi - mlo) / (n_mels + 1)

    matrix = np.zeros((n_mels, bank_width), dtype=np.float32)
    for m in range(0, n_mels):
        left = m_centers[m]
        center = m_centers[m + 1]
        right = m_centers[m + 2]
        for fft_bin in range(klo, khi):
            mbin = bin2mel(fft_bin)
            if left < mbin < right:
                matrix[m, fft_bin] = 1.0 - abs(center - mbin) / ms

    return matrix


class Phi4MultimodalFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_audio_embeds", "audio_embed_sizes", "audio_attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        audio_compression_rate: int = 8,
        audio_downsample_rate: int = 1,
        audio_feat_stride: int = 1,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        self.audio_compression_rate = audio_compression_rate
        self.audio_downsample_rate = audio_downsample_rate
        self.audio_feat_stride = audio_feat_stride

        self._eightk_method = "fillzero"
        self._mel = speechlib_mel(16000, 512, 80, fmin=None, fmax=7690).T

        self._hamming400 = np.hamming(400)  # for 16k audio
        self._hamming200 = np.hamming(200)  # for 8k audio

    def duration_to_frames(self, duration):
        """duration in s, estimated frames"""
        frame_rate = 10

        num_frames = duration * 1000 // frame_rate
        return num_frames

    def __call__(
        self,
        audios: List[AudioInput],
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        # Ref: https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py#L161
        returned_input_audio_embeds = []
        returned_audio_embed_sizes = []
        audio_frames_list = []

        for audio_data, sample_rate in audios:
            audio_embeds = self._extract_features(audio_data, sample_rate)
            audio_frames = len(audio_embeds) * self.audio_feat_stride
            audio_embed_size = self._compute_audio_embed_size(audio_frames)

            returned_input_audio_embeds.append(torch.tensor(audio_embeds))
            returned_audio_embed_sizes.append(torch.tensor(audio_embed_size).long())
            audio_frames_list.append(audio_frames)

        returned_input_audio_embeds = pad_sequence(returned_input_audio_embeds, batch_first=True)
        returned_audio_embed_sizes = torch.stack(returned_audio_embed_sizes, dim=0)
        audio_frames = torch.tensor(audio_frames_list)
        returned_audio_attention_mask = (
            torch.arange(0, audio_frames.max()).unsqueeze(0) < audio_frames.unsqueeze(1) if len(audios) > 1 else None
        )

        data = {
            "input_audio_embeds": returned_input_audio_embeds,
            "audio_embed_sizes": returned_audio_embed_sizes,
        }
        if returned_audio_attention_mask is not None:
            data["audio_attention_mask"] = returned_audio_attention_mask

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _extract_spectrogram(self, wav, fs):
        """Extract spectrogram features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        if wav.ndim > 1:
            wav = np.squeeze(wav)

        # by default, we extract the mean if stereo
        if len(wav.shape) == 2:
            wav = wav.mean(1)

        # Resample to 16000 or 8000 if needed
        if fs > 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 16000)
            fs = 16000
        elif 8000 < fs < 16000:
            wav = scipy.signal.resample_poly(wav, 1, fs // 8000)
            fs = 8000
        elif fs < 8000:
            raise RuntimeError(f"Unsupported sample rate {fs}")

        if fs == 8000:
            if self._eightk_method == "resample":
                # Input audio is 8 kHz. Convert to 16 kHz before feature
                # extraction
                wav = scipy.signal.resample_poly(wav, 2, 1)
                fs = 16000
            # Do nothing here for fillzero method
        elif fs != 16000:
            # Input audio is not a supported sample rate.
            raise RuntimeError(f"Input data using an unsupported sample rate: {fs}")

        preemphasis = 0.97

        if fs == 8000:
            n_fft = 256
            win_length = 200
            hop_length = 80
            fft_window = self._hamming200
        elif fs == 16000:
            n_fft = 512
            win_length = 400
            hop_length = 160
            fft_window = self._hamming400

        # Spec 1: SpeechLib cut remaining sample insufficient for a hop
        n_batch = (wav.shape[0] - win_length) // hop_length + 1
        # Here we don't use stride_tricks since the input array may not satisfy
        # memory layout requirement and we need writeable output
        # Here we only use list of views before copy to desination
        # so it is more efficient than broadcasting
        y_frames = np.array(
            [wav[_stride : _stride + win_length] for _stride in range(0, hop_length * n_batch, hop_length)],
            dtype=np.float32,
        )

        # Spec 2: SpeechLib applies preemphasis within each batch
        y_frames_prev = np.roll(y_frames, 1, axis=1)
        y_frames_prev[:, 0] = y_frames_prev[:, 1]
        y_frames = (y_frames - preemphasis * y_frames_prev) * 32768

        S = np.fft.rfft(fft_window * y_frames, n=n_fft, axis=1).astype(np.complex64)

        if fs == 8000:
            # Need to pad the output to look like 16 kHz data but with zeros in
            # the 4 to 8 kHz bins.
            frames, bins = S.shape
            padarray = np.zeros((frames, bins))
            S = np.concatenate((S[:, 0:-1], padarray), axis=1)  # Nyquist bin gets set to zero

        spec = np.abs(S).astype(np.float32)
        return spec

    def _extract_features(self, wav, fs):
        """Extract log filterbank features from waveform.
        Args:
            wav (1D array): waveform of the input
            fs (int): sampling rate of the waveform, 16000 or 8000.
                If fs=8000, the waveform will be resampled to 16000Hz.
        Output:
            log_fbank (2D array): a TxD matrix of log Mel filterbank features.
                D=80, and T is the number of frames.
        """
        spec = self._extract_spectrogram(wav, fs)
        spec_power = spec**2

        fbank_power = np.clip(spec_power.dot(self._mel), 1.0, None)
        log_fbank = np.log(fbank_power).astype(np.float32)

        return log_fbank

    def _compute_audio_embed_size(self, audio_frames):
        integer = audio_frames // self.audio_compression_rate
        remainder = audio_frames % self.audio_compression_rate

        result = integer if remainder == 0 else integer + 1

        integer = result // self.audio_downsample_rate
        remainder = result % self.audio_downsample_rate
        result = integer if remainder == 0 else integer + 1  # qformer compression

        return result


__all__ = ["Phi4MultimodalFeatureExtractor"]
