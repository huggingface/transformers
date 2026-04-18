# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for Qwen3TTS."""

import numpy as np

from ... import is_torch_available
from ...audio_utils import mel_filter_bank
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class Qwen3TTSFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Qwen3TTS feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech for use in Qwen3TTS speaker embedding extraction.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The feature dimension of the extracted features (number of mel filters).
        sampling_rate (`int`, *optional*, defaults to 24000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        n_fft (`int`, *optional*, defaults to 1024):
            Size of the Fourier transform.
        hop_length (`int`, *optional*, defaults to 256):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        win_length (`int`, *optional*, defaults to 1024):
            Length of the window function applied to each frame.
        fmin (`float`, *optional*, defaults to 0.0):
            Minimum frequency for the mel filter bank.
        fmax (`float`, *optional*, defaults to 12000.0):
            Maximum frequency for the mel filter bank.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the attention mask.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=24000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        fmin=0.0,
        fmax=12000.0,
        padding_value=0.0,
        return_attention_mask=False,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + n_fft // 2,
            num_mel_filters=feature_size,
            min_frequency=fmin,
            max_frequency=fmax,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _extract_mel_features(self, waveform: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Compute the mel spectrogram of the provided audio using dynamic range compression.

        Uses `log(clamp(x, min=1e-5))` compression matching the original Qwen3TTS implementation.
        """
        if not is_torch_available():
            raise ImportError("PyTorch is required for Qwen3TTS feature extraction.")

        waveform = torch.from_numpy(waveform).to(device, torch.float32)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        hann_window = torch.hann_window(self.win_length, device=device)

        # Reflect padding to match original implementation
        padding = (self.n_fft - self.hop_length) // 2
        waveform = torch.nn.functional.pad(waveform, (padding, padding), mode="reflect")

        spec = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

        mel_filters = torch.from_numpy(self.mel_filters).to(device, torch.float32)
        mel_spec = torch.matmul(mel_filters.T, spec.squeeze(0)).unsqueeze(0)

        # Dynamic range compression: log(clamp(x, min=1e-5))
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        if device != "cpu":
            mel_spec = mel_spec.detach().cpu()
        return mel_spec.numpy()

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        return_tensors: str | TensorType | None = None,
        return_attention_mask: bool | None = None,
        sampling_rate: int | None = None,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            device (`str`, *optional*, defaults to `"cpu"`):
                The device to use for computation.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input was"
                f" sampled with {self.sampling_rate} and not {sampling_rate}."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, list, tuple)))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        if not is_batched:
            raw_speech = [raw_speech]

        batched_speech = {"input_features": []}
        for speech in raw_speech:
            mel_features = self._extract_mel_features(speech, device=device)
            # Transpose to (time, n_mels) for speaker encoder compatibility
            batched_speech["input_features"].append(mel_features.squeeze(0).T)

        padded_inputs = BatchFeature(batched_speech)

        if return_attention_mask is None:
            return_attention_mask = self.return_attention_mask

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


__all__ = ["Qwen3TTSFeatureExtractor"]
