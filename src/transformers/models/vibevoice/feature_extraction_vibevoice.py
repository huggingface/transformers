# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for VibeVoice."""

import os
from typing import List, Optional, Union, Dict, Any

import numpy as np
import torch

from ...feature_extraction_utils import FeatureExtractionMixin
from ...utils import logging

logger = logging.get_logger(__name__)


def normalize_audio(audio: np.ndarray, target_dB_FS: float = -25, eps: float = 1e-6) -> np.ndarray:
    """
    Normalize audio to target dB FS level and avoid clipping.
    
    Args:
        audio (np.ndarray): Input audio signal
        target_dB_FS (float): Target dB FS level for the audio. Default: -25
        eps (float): Small value to avoid division by zero. Default: 1e-6
        
    Returns:
        np.ndarray: Normalized audio signal
    """
    # Adjust to target dB FS
    rms = np.sqrt(np.mean(audio**2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    audio = audio * scalar
    
    # Avoid clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / (max_val + eps)
    
    return audio


class VibeVoiceFeatureExtractor(FeatureExtractionMixin):
    """
    Feature extractor for VibeVoice acoustic tokenizer models.

    This feature extractor handles audio preprocessing for VibeVoice models, including:
    - Audio format conversion (stereo to mono)
    - Optional audio normalization
    - Streaming support for infinite-length audio
    
    Args:
        sampling_rate (int, optional): Expected sampling rate. Defaults to 24000.
        normalize_audio (bool, optional): Whether to normalize audio. Defaults to True.
        target_dB_FS (float, optional): Target dB FS for normalization. Defaults to -25.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.
    """
    model_input_names = ["input_features"]
    
    def __init__(
        self,
        sampling_rate: int = 24000,
        normalize_audio: bool = True,
        target_dB_FS: float = -25,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.sampling_rate = sampling_rate
        self.normalize_audio = normalize_audio
        self.target_dB_FS = target_dB_FS
        self.eps = eps
        
        # Save config
        self.feature_extractor_dict = {
            "sampling_rate": sampling_rate,
            "normalize_audio": normalize_audio,
            "target_dB_FS": target_dB_FS,
            "eps": eps,
        }
    
    def _ensure_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono if needed.
        
        Args:
            audio (np.ndarray): Input audio array
            
        Returns:
            np.ndarray: Mono audio array
        """
        if len(audio.shape) == 1:
            return audio
        elif len(audio.shape) == 2:
            if audio.shape[0] == 2:  # (2, time)
                return np.mean(audio, axis=0)
            elif audio.shape[1] == 2:  # (time, 2)
                return np.mean(audio, axis=1)
            else:
                # If one dimension is 1, squeeze it
                if audio.shape[0] == 1:
                    return audio.squeeze(0)
                elif audio.shape[1] == 1:
                    return audio.squeeze(1)
                else:
                    raise ValueError(f"Unexpected audio shape: {audio.shape}")
        else:
            raise ValueError(f"Audio should be 1D or 2D, got shape: {audio.shape}")
    
    def _process_single_audio(self, audio: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Process a single audio array.
        
        Args:
            audio: Single audio input
            
        Returns:
            np.ndarray: Processed audio
        """
        # Convert to numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)
        
        # Ensure mono
        audio = self._ensure_mono(audio)
        
        # Normalize if requested
        if self.normalize_audio:
            audio = normalize_audio(audio, self.target_dB_FS, self.eps)
        
        return audio
    
    def __call__(
        self,
        audio: Union[str, np.ndarray, List[float], List[np.ndarray], List[List[float]], List[str]] = None,
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        """
        Process audio for VibeVoice models.
        
        Args:
            audio: Audio input(s) to process. Can be:
                - str: Path to audio file
                - np.ndarray: Audio array
                - List[float]: Audio as list of floats
                - List[np.ndarray]: Batch of audio arrays
                - List[str]: Batch of audio file paths
            sampling_rate (int, optional): Sampling rate of the input audio
            return_tensors (str, optional): Return format ('pt' for PyTorch, 'np' for NumPy)
            
        Returns:
            dict: Processed audio inputs with keys:
                - input_features: Audio tensor(s) ready for the model
        """
        if audio is None:
            raise ValueError("Audio input is required")
        
        # Validate sampling rate
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            logger.warning(
                f"Input sampling rate ({sampling_rate}) differs from expected "
                f"sampling rate ({self.sampling_rate}). Please resample your audio."
            )

        import pudb; pudb.set_trace()
        
        # Handle different input types
        if isinstance(audio, str):
            # Single audio file path
            audio = self._load_audio_from_path(audio)
            is_batched = False
        elif isinstance(audio, list):
            if len(audio) == 0:
                raise ValueError("Empty audio list provided")
            
            # Check if it's a list of file paths
            if all(isinstance(item, str) for item in audio):
                # Batch of audio file paths
                audio = [self._load_audio_from_path(path) for path in audio]
                is_batched = True
            else:
                # Check if it's batched audio arrays
                is_batched = isinstance(audio[0], (np.ndarray, list))
        else:
            # Single audio array or list
            is_batched = False
        
        # Process audio
        if is_batched:
            processed_audio = [self._process_single_audio(a) for a in audio]
        else:
            processed_audio = [self._process_single_audio(audio)]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            if len(processed_audio) == 1:
                # Create a proper batch dimension (B, T)
                input_features = torch.from_numpy(processed_audio[0]).unsqueeze(0).unsqueeze(1)
            else:
                # For batched input with different lengths, create a batch properly
                input_features = torch.stack([torch.from_numpy(a) for a in processed_audio]).unsqueeze(1)
        elif return_tensors == "np":
            if len(processed_audio) == 1:
                input_features = processed_audio[0][np.newaxis, np.newaxis, :]
            else:
                input_features = np.stack(processed_audio)[:, np.newaxis, :]
        else:
            input_features = processed_audio[0] if len(processed_audio) == 1 else processed_audio
        
        outputs = {
            "audio": input_features,  # Use "audio" instead of "input_features"
        }
        
        return outputs

    def _load_audio_from_path(self, audio_path: str) -> np.ndarray:
        """
        Load audio from file path.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.ndarray: Loaded audio array
        """
        # Get file extension to determine loading method
        file_ext = os.path.splitext(audio_path)[1].lower()
        
        if file_ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            # Audio file - use librosa
            import librosa
            audio_array, sr = librosa.load(
                audio_path, 
                sr=self.sampling_rate, 
                mono=True
            )
            return audio_array
        elif file_ext == '.pt':
            # PyTorch tensor file
            audio_tensor = torch.load(audio_path, map_location='cpu').squeeze()
            if isinstance(audio_tensor, torch.Tensor):
                audio_array = audio_tensor.numpy()
            else:
                audio_array = np.array(audio_tensor)
            return audio_array.astype(np.float32)
        elif file_ext == '.npy':
            # NumPy file
            audio_array = np.load(audio_path)
            return audio_array.astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: .wav, .mp3, .flac, .m4a, .ogg, .pt, .npy, .npz"
            )
    
    def preprocess_audio(
        self, 
        audio_path_or_array: Union[str, np.ndarray],
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Convenience method to preprocess audio from file path or array.
        This method is kept for backward compatibility but __call__ is recommended.
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
            normalize: Whether to normalize (overrides default setting)
            
        Returns:
            np.ndarray: Preprocessed audio array
        """
        if isinstance(audio_path_or_array, str):
            audio_array = self._load_audio_from_path(audio_path_or_array)
        else:
            audio_array = np.array(audio_path_or_array, dtype=np.float32)
        
        # Override normalization setting if specified
        original_normalize = self.normalize_audio
        if normalize is not None:
            self.normalize_audio = normalize
        
        try:
            processed = self._process_single_audio(audio_array)
        finally:
            # Restore original setting
            self.normalize_audio = original_normalize
        
        return processed
    
    # Override to_dict method for configuration saving
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the object to a dict containing all attributes needed for serialization.
        """
        return self.feature_extractor_dict

    def save_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
        output_path: str = "output.wav",
        sampling_rate: Optional[int] = None,
        normalize: bool = False,
        batch_prefix: str = "audio_",
    ):
        """
        Save audio data to WAV file(s).
        
        Args:
            audio: Audio data to save. Can be:
                - torch.Tensor: PyTorch tensor with shape (B, C, T) or (B, T) or (T)
                - np.ndarray: NumPy array with shape (B, C, T) or (B, T) or (T)
                - List of tensors or arrays
            output_path: Path where to save the audio. If saving multiple files,
                this is treated as a directory and individual files will be saved inside.
            sampling_rate: Sampling rate for the saved audio. Defaults to the processor's rate.
            normalize: Whether to normalize audio before saving.
            batch_prefix: Prefix for batch files when saving multiple audios.
                
        Returns:
            List[str]: Paths to the saved audio files.
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "soundfile is required to save audio files. "
                "Install it with: pip install soundfile"
            )
        
        # Ensure audio is in the right format
        if isinstance(audio, torch.Tensor):
            # Convert PyTorch tensor to numpy
            audio_np = audio.float().detach().cpu().numpy()
        elif isinstance(audio, np.ndarray):
            audio_np = audio
        elif isinstance(audio, list):
            # Handle list of tensors or arrays
            if all(isinstance(a, torch.Tensor) for a in audio):
                audio_np = [a.float().detach().cpu().numpy() for a in audio]
            else:
                audio_np = audio
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        saved_paths = []
        
        # Handle based on shape or type
        if isinstance(audio_np, list):
            # Multiple separate audios to save
            output_dir = output_path
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each audio
            for i, audio_item in enumerate(audio_np):
                audio_item = self._prepare_audio_for_save(audio_item, normalize)
                file_path = os.path.join(output_dir, f"{batch_prefix}{i}.wav")
                sf.write(file_path, audio_item, sampling_rate)
                saved_paths.append(file_path)
                
        else:
            # Handle different dimensions
            if len(audio_np.shape) >= 3:  # (B, C, T) or similar
                # Get batch size
                batch_size = audio_np.shape[0]
                
                if batch_size > 1:
                    # Multiple audios in a batch
                    output_dir = output_path
                    
                    # Ensure output directory exists
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save each audio in the batch
                    for i in range(batch_size):
                        # Extract single audio and remove channel dim if present
                        single_audio = audio_np[i]
                        if len(single_audio.shape) > 1:
                            if single_audio.shape[0] == 1:  # (1, T)
                                single_audio = single_audio.squeeze(0)
                        
                        single_audio = self._prepare_audio_for_save(single_audio, normalize)
                        file_path = os.path.join(output_dir, f"{batch_prefix}{i}.wav")
                        sf.write(file_path, single_audio, sampling_rate)
                        saved_paths.append(file_path)
                else:
                    # Single audio with batch and channel dims
                    audio_item = audio_np.squeeze()  # Remove batch and channel dimensions
                    audio_item = self._prepare_audio_for_save(audio_item, normalize)
                    sf.write(output_path, audio_item, sampling_rate)
                    saved_paths.append(output_path)
            else:
                # Single audio without batch dimension
                audio_item = self._prepare_audio_for_save(audio_np, normalize)
                sf.write(output_path, audio_item, sampling_rate)
                saved_paths.append(output_path)
        
        return saved_paths

    def _prepare_audio_for_save(self, audio: np.ndarray, normalize: bool) -> np.ndarray:
        """
        Prepare audio for saving by ensuring it's the right shape and optionally normalizing.
        
        Args:
            audio: Audio data as numpy array
            normalize: Whether to normalize audio
            
        Returns:
            np.ndarray: Processed audio ready for saving
        """
        # Ensure right dimensionality
        if len(audio.shape) > 1 and audio.shape[0] == 1:  # (1, T)
            audio = audio.squeeze(0)
        
        # Normalize if requested
        if normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
        
        return audio


__all__ = ["VibeVoiceFeatureExtractor"]