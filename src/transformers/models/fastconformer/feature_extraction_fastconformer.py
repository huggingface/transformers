# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Feature extractor class for FastConformer
"""

import math
import warnings
from typing import List, Optional, Union

import numpy as np

from ... import is_torch_available
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


if is_torch_available():
    import torch
    import librosa

logger = logging.get_logger(__name__)

# Constants for preprocessing (matching NeMo implementation)
LOG_ZERO_GUARD_VALUE = 2 ** -24
EPSILON = 1e-5


class FastConformerFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a FastConformer feature extractor that matches NeMo's preprocessing exactly.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using the same implementation as NeMo's
    AudioToMelSpectrogramPreprocessor for perfect numerical equivalence.

    Args:
        feature_size (`int`, *optional*, defaults to 128):
            The feature dimension of the extracted features (number of mel bins).
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
        win_length (`int`, *optional*, defaults to 400):
            Window length for STFT.
        n_fft (`int`, *optional*, defaults to 512):
            Size of the Fourier transform.
        window_size (`float`, *optional*, defaults to 0.025):
            Window size in seconds.
        window_stride (`float`, *optional*, defaults to 0.01):
            Window stride in seconds.
        window (`str`, *optional*, defaults to "hann"):
            Window function type.
        normalize (`str`, *optional*, defaults to "per_feature"):
            Normalization type: "per_feature" or "all_features".
        preemph (`float`, *optional*, defaults to 0.97):
            Pre-emphasis factor.
        mag_power (`float`, *optional*, defaults to 2.0):
            Magnitude power for spectrogram.
        mel_norm (`str`, *optional*, defaults to "slaney"):
            Mel filterbank normalization.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to True):
            Whether to return attention mask for proper sequence length handling.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        win_length=400,
        n_fft=512,
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        normalize="per_feature",
        preemph=0.97,
        mag_power=2.0,
        mel_norm="slaney",
        padding_value=0.0,
        return_attention_mask=True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        
        # Core parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize = normalize
        self.preemph = preemph
        self.mag_power = mag_power
        self.mel_norm = mel_norm
        
        # Validation
        if self.win_length > self.n_fft:
            warnings.warn(f"win_length ({self.win_length}) > n_fft ({self.n_fft}), truncating window")
        if self.normalize not in ["per_feature", "all_features"]:
            raise ValueError("normalize must be 'per_feature' or 'all_features'")
        
        # Mel filterbank will be created on-demand
        self._filterbanks = None

    def get_filterbanks(self, device, dtype):
        """Get mel filterbanks, creating them if needed."""
        if self._filterbanks is None:
            if not is_torch_available():
                raise ImportError("PyTorch is required for FastConformer feature extraction")
            
            self._filterbanks = torch.tensor(
                librosa.filters.mel(
                    sr=self.sampling_rate,
                    n_fft=self.n_fft,
                    n_mels=self.feature_size,
                    fmin=0,
                    fmax=self.sampling_rate / 2,
                    norm=self.mel_norm,
                ),
                dtype=torch.float32,
            )
        
        return self._filterbanks.to(device=device, dtype=dtype)

    def get_window(self, win_length: int, window_type: str, device, dtype) -> torch.Tensor:
        """Get window function based on type."""
        window_fns = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
        }
        
        if window_type not in window_fns:
            raise ValueError(f"Unsupported window type: {window_type}. Supported: {list(window_fns.keys())}")
        
        return window_fns[window_type](win_length, periodic=False, device=device, dtype=dtype)

    def get_seq_len(self, audio_len: int, n_fft: int, hop_length: int) -> int:
        """Calculate sequence length after STFT with center=True padding."""
        # For center=True, padding is n_fft // 2 on each side
        pad_amount = n_fft // 2 * 2  # Total padding
        seq_len = (audio_len + pad_amount - n_fft) // hop_length + 1
        return max(1, int(seq_len))  # Ensure at least 1 frame

    def preemphasis_batch(self, x: torch.Tensor, preemph: float) -> torch.Tensor:
        """Apply preemphasis filter to batch of audio signals."""
        # x: (B, T)
        x0 = x[:, :1]
        x_rest = x[:, 1:] - preemph * x[:, :-1]
        return torch.cat([x0, x_rest], dim=1)

    def get_log_zero_guard_value(self, value, dtype: torch.dtype) -> float:
        """Get log zero guard value from config."""
        if isinstance(value, str):
            if value == "tiny":
                return torch.finfo(dtype).tiny
            elif value == "eps":
                return torch.finfo(dtype).eps
            else:
                raise ValueError(f"Unsupported log_zero_guard_value string: {value}")
        return float(value)

    def normalize_mel_features(
        self, 
        mel: torch.Tensor, 
        mask: torch.Tensor, 
        normalize_type: str,
        eps: float = EPSILON
    ) -> torch.Tensor:
        """Apply normalization to mel features with proper masking."""
        mel_masked = mel * mask
        
        if normalize_type == "per_feature":
            # Per-feature normalization: (B, T, F) -> normalize across T for each F
            lengths = mask.sum(dim=1)  # (B, F)
            mean = mel_masked.sum(dim=1) / lengths.clamp(min=1)  # (B, F)
            mean = mean.unsqueeze(1)  # (B, 1, F)
            
            # Calculate std with bias correction
            variance = ((mel_masked - mean) ** 2 * mask).sum(dim=1) / (lengths - 1).clamp(min=1)
            std = torch.sqrt(variance).unsqueeze(1)  # (B, 1, F)
            
        elif normalize_type == "all_features":
            # Global normalization: normalize across all valid T and F
            total_valid = mask.sum(dim=(1, 2)).clamp(min=1)  # (B,)
            mean = mel_masked.sum(dim=(1, 2)) / total_valid  # (B,)
            mean = mean.view(-1, 1, 1)
            
            variance = ((mel_masked - mean) ** 2 * mask).sum(dim=(1, 2)) / (total_valid - 1).clamp(min=1)
            std = torch.sqrt(variance).view(-1, 1, 1)
        else:
            raise ValueError(f"Unsupported normalize_type: {normalize_type}")
        
        normalized_mel = (mel - mean) / (std + eps)
        return normalized_mel

    def get_logmel(self, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to log-mel spectrograms for inference (matching NeMo exactly).
        
        Args:
            x: Audio tensor of shape (B, T)
            seq_len: Sequence lengths of shape (B,)
            
        Returns:
            Log-mel features of shape (B, F, T_frames)
        """
        original_dtype = x.dtype
        device = x.device
        B = x.shape[0]

        # Input validation
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input (B, T), got {x.dim()}D")
        if seq_len.shape[0] != B:
            raise ValueError(f"seq_len batch size {seq_len.shape[0]} != audio batch size {B}")

        # Apply preemphasis
        if self.preemph is not None and self.preemph > 0:
            x = self.preemphasis_batch(x, self.preemph)

        # Get window
        window = self.get_window(self.win_length, self.window, device=device, dtype=x.dtype)

        # STFT computation
        stft_out = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )  # (B, F, T_frames)
        
        # Convert to magnitude and apply power
        abs_stft = torch.abs(stft_out)
        if self.mag_power != 1.0:
            abs_stft = abs_stft.pow(self.mag_power)

        # Apply mel filterbank
        filterbanks = self.get_filterbanks(device=device, dtype=abs_stft.dtype)
        mel = torch.matmul(filterbanks, abs_stft)  # (B, n_mels, T_frames)
        mel = mel.permute(0, 2, 1)  # (B, T_frames, n_mels)

        # Apply log with zero guard
        log_guard = self.get_log_zero_guard_value(LOG_ZERO_GUARD_VALUE, mel.dtype)
        mel = torch.log(mel + log_guard)

        # Create mask for valid frames
        max_frames = mel.shape[1]
        valid_mask = torch.arange(max_frames, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        mask = valid_mask.unsqueeze(-1)  # (B, T, 1)

        # Apply normalization with masking
        normalized_mel = self.normalize_mel_features(mel, mask, self.normalize)
        
        # Mask invalid frames
        normalized_mel = normalized_mel.masked_fill(~mask, self.padding_value)
        
        # Return in NeMo format: (B, F, T)
        normalized_mel = normalized_mel.permute(0, 2, 1)
        
        return normalized_mel.to(original_dtype)

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: bool = False,
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be processed. Each sequence can be a numpy array,
                a list of float values, a list of numpy arrays or a list of list of float values.
            truncation (`bool`, *optional*, defaults to `False`):
                Whether to truncate the input sequences.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.
            padding (`bool`, *optional*, defaults to `False`):
                Whether to pad the input sequences.
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the call for clarity.
            device (`str`, *optional*, defaults to `"cpu"`):
                The device to use for computation.
        """
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        # Handle input format
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech_list = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech_list = [np.asarray(raw_speech, dtype=np.float32)]
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech_list = [raw_speech.astype(np.float32)]
        else:
            raw_speech_list = [raw_speech] if not is_batched else raw_speech

        # Convert to tensors and get lengths
        audio_lengths = torch.tensor([len(audio) for audio in raw_speech_list], dtype=torch.long)
        
        # Pad to the longest sequence
        max_len = max(len(audio) for audio in raw_speech_list)
        padded_audios = []
        for audio in raw_speech_list:
            pad_len = max_len - len(audio)
            if pad_len > 0:
                audio = np.pad(audio, (0, pad_len), constant_values=self.padding_value)
            padded_audios.append(audio)
        
        # Stack into batch tensor
        batch_audio = torch.stack([torch.from_numpy(audio).float() for audio in padded_audios], dim=0)
        
        # Compute sequence lengths for mel features
        seq_lens = torch.tensor([
            self.get_seq_len(length.item(), self.n_fft, self.hop_length) 
            for length in audio_lengths
        ], dtype=torch.long)
        
        # Extract mel features using NeMo-matching implementation
        if device != "cpu" and torch.cuda.is_available():
            batch_audio = batch_audio.to(device)
            seq_lens = seq_lens.to(device)
        
        logmel_features = self.get_logmel(batch_audio, seq_lens)  # (B, F, T)
        
        # Move back to CPU if needed
        if device != "cpu":
            logmel_features = logmel_features.cpu()
            seq_lens = seq_lens.cpu()
        
        # Transpose to HuggingFace format: (B, T, F)
        input_features = logmel_features.transpose(1, 2)
        
        # Create attention mask based on sequence lengths
        max_frames = input_features.shape[1]
        attention_mask = torch.arange(max_frames).unsqueeze(0) < seq_lens.unsqueeze(1)
        
        # Prepare output
        encoded_inputs = BatchFeature({
            "input_features": input_features,
            "attention_mask": attention_mask,
            "input_lengths": seq_lens,  # Add sequence lengths for encoder
        })

        # Convert to requested tensor format
        if return_tensors is not None:
            encoded_inputs = encoded_inputs.convert_to_tensors(return_tensors)
        
        # Handle attention mask return
        if return_attention_mask is False:
            encoded_inputs.pop("attention_mask", None)
        elif return_attention_mask is None and not self.return_attention_mask:
            encoded_inputs.pop("attention_mask", None)

        return encoded_inputs 