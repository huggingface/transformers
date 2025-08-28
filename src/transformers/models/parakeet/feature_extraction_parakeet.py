# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

import warnings
from collections.abc import Sequence
from typing import Any, Optional

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...tokenization_utils_base import AudioInput
from ...utils import is_librosa_available, is_torch_available, logging, requires_backends


if is_torch_available():
    import torch

EPSILON = 1e-10
LOG_ZERO_GUARD_VALUE = 2**-24

logger = logging.get_logger(__name__)


class ParakeetFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Parakeet feature extractor that matches NeMo's preprocessing exactly.

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
        window (`str`, *optional*, defaults to `"hann"`):
            Window function type.
        normalize (`str`, *optional*, defaults to `"per_feature"`):
            Normalization type: "per_feature" or "all_features".
        preemphasis (`float`, *optional*, defaults to 0.97):
            Pre-emphasis factor.
        mag_power (`float`, *optional*, defaults to 2.0):
            Magnitude power for spectrogram.
        mel_scale (`str`, *optional*, defaults to `"slaney"`):
            Mel filterbank normalization.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether or not [`~ParakeetFeatureExtractor.__call__`] should return `attention_mask`.
    """

    model_input_names = ["input_features", "attention_mask"]

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
        preemphasis=0.97,
        mag_power=2.0,
        mel_scale="slaney",
        padding_value=0.0,
        return_attention_mask=True,
        **kwargs,
    ):
        requires_backends(self, ["librosa"])

        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.return_attention_mask = return_attention_mask

        # Core parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.normalize = normalize
        self.preemphasis = preemphasis
        self.mag_power = mag_power
        self.mel_scale = mel_scale

        # Validation
        if self.win_length > self.n_fft:
            warnings.warn(f"win_length ({self.win_length}) > n_fft ({self.n_fft}), truncating window")
        if self.normalize not in ["per_feature", "all_features"]:
            raise ValueError("normalize must be 'per_feature' or 'all_features'")

        # Mel filterbank will be created on-demand
        self._filterbanks = None
        self._window = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """Load feature extractor and attempt to load preprocessing weights from model."""
        # Load feature extractor normally
        instance = super().from_pretrained(model_name_or_path, **kwargs)

        # Try to load preprocessing weights from model files
        try:
            from pathlib import Path

            model_path = Path(model_name_or_path)

            # Try safetensors first, then pytorch_model.bin
            model_files = ["model.safetensors", "pytorch_model.bin"]

            for file_name in model_files:
                weight_file = model_path / file_name
                if weight_file.exists():
                    if file_name.endswith(".safetensors"):
                        try:
                            from safetensors.torch import load_file

                            weights = load_file(weight_file)
                        except ImportError:
                            continue
                    else:
                        weights = torch.load(weight_file, map_location="cpu")

                    # Load preprocessing weights if they exist
                    if "feature_extractor.window" in weights:
                        instance._window = weights["feature_extractor.window"]
                        logger.info("Loaded pretrained window function")

                    if "feature_extractor.filterbanks" in weights:
                        instance._filterbanks = weights["feature_extractor.filterbanks"]
                        logger.info("Loaded pretrained mel filterbanks")

                    break

        except Exception as e:
            logger.debug(f"Could not load preprocessing weights: {e}")
            # Fallback to computed weights - this is fine
            pass

        return instance

    def get_filterbanks(self, device, dtype):
        """Get mel filterbanks, creating them if needed."""
        if self._filterbanks is not None:
            return self._filterbanks.to(device=device, dtype=dtype)

        if not is_torch_available():
            raise ImportError("PyTorch is required for Parakeet feature extraction")

        if not is_librosa_available():
            raise ImportError(
                "librosa is required for FastConformer feature extraction. "
                "Please install it with `pip install librosa`."
            )

        import librosa

        self._filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=self.sampling_rate,
                n_fft=self.n_fft,
                n_mels=self.feature_size,
                fmin=0,
                fmax=self.sampling_rate / 2,
                norm=self.mel_scale,
            ),
            dtype=torch.float32,
        )

        return self._filterbanks.to(device=device, dtype=dtype)

    def get_window(self, win_length: int, window_type: str, device, dtype) -> "torch.Tensor":
        """Get window function based on type."""
        # Use pretrained window if available and correct length
        if self._window is not None and self._window.numel() == win_length:
            return self._window.to(device=device, dtype=dtype)

        window_fns = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
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

    def preemphasis_batch(self, x: "torch.Tensor", preemph: float) -> "torch.Tensor":
        """Apply preemphasis filter to batch of audio signals."""
        # x: (B, T)
        x0 = x[:, :1]
        x_rest = x[:, 1:] - preemph * x[:, :-1]
        return torch.cat([x0, x_rest], dim=1)

    def get_log_zero_guard_value(self, value, dtype: "torch.dtype") -> float:
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
        self, mel: "torch.Tensor", mask: "torch.Tensor", normalize_type: str, eps: float = EPSILON
    ) -> "torch.Tensor":
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

    def get_logmel(self, x: "torch.Tensor", seq_len: "torch.Tensor") -> "torch.Tensor":
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
        if self.preemphasis is not None and self.preemphasis > 0:
            x = self.preemphasis_batch(x, self.preemphasis)

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

    def _get_audios_and_audio_lengths(self, audios: AudioInput) -> tuple["torch.Tensor", Sequence[int]]:
        """
        Coerces audio inputs to torch tensors and extracts audio lengths prior to stacking.

        Args:
            audios (`AudioInput`):
                Audio sequence, numpy array, or torch tensor.
        """
        requires_backends(self, ["torch"])
        import numpy as np

        # Handle different input types
        if isinstance(audios, np.ndarray):
            # Single numpy array
            audios = torch.from_numpy(audios)
        elif isinstance(audios, (list, tuple)) and len(audios) > 0:
            if isinstance(audios[0], np.ndarray):
                # List of numpy arrays
                audios = [torch.from_numpy(arr) for arr in audios]
            elif not isinstance(audios[0], torch.Tensor):
                # Convert list of numbers to tensor
                audios = [torch.tensor(arr, dtype=torch.float32) for arr in audios]

        if isinstance(audios, torch.Tensor):
            if audios.ndim == 1:
                audios = audios.unsqueeze(0)
            if not torch.is_floating_point(audios):
                audios = audios.float()

            if audios.shape[0] > 1:
                logger.warning("Audio samples are already collated; assuming they all have the same length")
            lengths = [audios.shape[-1]] * audios.shape[0]
            return audios, lengths

        elif isinstance(audios, Sequence) and isinstance(audios[0], torch.Tensor):
            if not torch.is_floating_point(audios[0]):
                audios = [audio.float() for audio in audios]
            lengths = [audio.shape[-1] if audio.dim() > 0 else audio.numel() for audio in audios]
            # Squeeze and handle different tensor shapes
            squeezed_audios = []
            for audio in audios:
                if audio.dim() > 1:
                    squeezed_audios.append(audio.squeeze())
                else:
                    squeezed_audios.append(audio)
            audios = torch.nn.utils.rnn.pad_sequence(squeezed_audios, batch_first=True, padding_value=0.0)
            return audios, lengths

        raise TypeError(
            f"Invalid audio provided. Audio should be torch tensors, numpy arrays, or sequences thereof. Got {type(audios)}"
        )

    def __call__(
        self,
        audios: AudioInput,
        sampling_rate: Optional[int] = 16000,
        device: Optional[str] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several audio sequence(s).

        Args:
            audios (`AudioInput`):
                Audio sequence(s) as torch tensor(s), numpy array(s), or list of arrays. Can be:
                - A single torch tensor of shape (T,) or (B, T)
                - A list of torch tensors/numpy arrays of varying lengths
                - A single numpy array
                Audio lengths are automatically inferred and padding is handled internally.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the audio input was sampled. It is strongly recommended to pass
                `sampling_rate` at the call for clarity.
            device (`str`, *optional*):
                The device to use for computation and output tensors. If None, uses the same device as the input tensor.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.
        """
        if not is_torch_available():
            raise ImportError("PyTorch is required for Parakeet feature extraction")

        if not is_librosa_available():
            raise ImportError(
                "librosa is required for Parakeet feature extraction. Please install it with `pip install librosa`."
            )

        if return_attention_mask is None:
            return_attention_mask = self.return_attention_mask

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided audio input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        # Get audio tensors and lengths using helper method
        batch_audio, audio_lengths = self._get_audios_and_audio_lengths(audios)
        batch_audio = batch_audio.float()

        # Convert audio_lengths to tensor
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        # Determine target device: use explicit device parameter, otherwise use input tensor's device
        target_device = device if device is not None else batch_audio.device

        # Move tensors to target device
        batch_audio = batch_audio.to(target_device)
        audio_lengths = audio_lengths.to(target_device)

        # Compute sequence lengths for mel features
        seq_lens = torch.tensor(
            [self.get_seq_len(length.item(), self.n_fft, self.hop_length) for length in audio_lengths],
            dtype=torch.long,
            device=target_device,
        )

        # Extract mel features using NeMo-matching implementation
        logmel_features = self.get_logmel(batch_audio, seq_lens)  # (B, F, T)

        # Transpose to HuggingFace format: (B, T, F)
        input_features = logmel_features.transpose(1, 2)

        # Create attention mask based on sequence lengths
        max_frames = input_features.shape[1]
        attention_mask = torch.arange(max_frames, device=target_device).unsqueeze(0) < seq_lens.unsqueeze(1)

        # Prepare output - always return PyTorch tensors
        encoded_inputs = BatchFeature({"input_features": input_features})
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask

        return encoded_inputs

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = super().to_dict()
        # Remove cached tensors that shouldn't be serialized
        output.pop("_filterbanks", None)
        output.pop("_window", None)
        return output


__all__ = ["ParakeetFeatureExtractor"]
