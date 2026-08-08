# coding=utf-8
# Copyright 2026 Microsoft Research and The HuggingFace Inc. team.
# Licensed under the MIT License.

"""Feature extractor class for BEATs."""

import numpy as np
from typing import List, Optional, Union

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import logging

logger = logging.get_logger(__name__)


class BEATsFeatureExtractor(SequenceFeatureExtractor):
    """
    Feature extractor for BEATs model.

    Converts raw audio waveforms to Mel Filterbank features (fbank),
    which are the input format expected by BEATsModel.

    Args:
        sampling_rate (`int`, *optional*, defaults to 16000):
            Target sampling rate. BEATs was trained on 16kHz audio.
        num_mel_bins (`int`, *optional*, defaults to 128):
            Number of Mel filterbank bins.
        frame_length (`int`, *optional*, defaults to 25):
            Frame length in milliseconds.
        frame_shift (`int`, *optional*, defaults to 10):
            Frame shift in milliseconds.
        fbank_mean (`float`, *optional*, defaults to 15.41663):
            Mean for fbank normalization.
        fbank_std (`float`, *optional*, defaults to 6.55582):
            Std for fbank normalization.
        padding_value (`float`, *optional*, defaults to 0.0):
            Value used for padding.
        return_tensors (`str`, *optional*):
            Return type. Can be `"pt"` for PyTorch tensors.

    Example:
```python
        from transformers import BEATsFeatureExtractor
        import torch

        feature_extractor = BEATsFeatureExtractor()
        audio = torch.randn(16000)  # 1 second of audio at 16kHz
        inputs = feature_extractor(audio, return_tensors="pt")
        print(inputs["input_values"].shape)  # (1, time, 128)
```
    """

    model_input_names = ["input_values"]

    def __init__(
        self,
        sampling_rate: int = 16000,
        num_mel_bins: int = 128,
        frame_length: int = 25,
        frame_shift: int = 10,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=num_mel_bins,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.fbank_mean = fbank_mean
        self.fbank_std = fbank_std

    def _extract_fbank(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract Mel Filterbank features from a waveform.
        Uses torchaudio.compliance.kaldi for compatibility with BEATs.
        """
        try:
            import torch
            import torchaudio.compliance.kaldi as ta_kaldi
        except ImportError:
            raise ImportError("torchaudio is required for BEATsFeatureExtractor.")

        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        waveform_tensor = waveform_tensor * (2 ** 15)

        fbank = ta_kaldi.fbank(
            waveform_tensor,
            num_mel_bins=self.num_mel_bins,
            sample_frequency=self.sampling_rate,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
        )
        fbank = fbank.numpy()
        fbank = (fbank - self.fbank_mean) / (2 * self.fbank_std)
        return fbank

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[np.ndarray], "torch.Tensor", List["torch.Tensor"]],
        sampling_rate: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Extract fbank features from raw audio.

        Args:
            raw_speech: Audio waveform(s). Shape: `(num_samples,)` or list of such arrays.
            sampling_rate: Must match `self.sampling_rate` (16000).
            return_tensors: If `"pt"`, return PyTorch tensors.

        Returns:
            BatchFeature with `input_values` key containing fbank features.
            Shape: `(batch, time, num_mel_bins)`.
        """
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"BEATsFeatureExtractor expects {self.sampling_rate}Hz audio, got {sampling_rate}Hz."
            )

        # Handle single or batch input
        if isinstance(raw_speech, (list, tuple)):
            waveforms = raw_speech
        else:
            waveforms = [raw_speech]

        # Convert to numpy
        processed = []
        for waveform in waveforms:
            if hasattr(waveform, "numpy"):
                waveform = waveform.numpy()
            waveform = np.array(waveform, dtype=np.float32)
            fbank = self._extract_fbank(waveform)
            processed.append(fbank)

        encoded = BatchFeature({"input_values": processed})

        if return_tensors is not None:
            encoded = encoded.convert_to_tensors(return_tensors)

        return encoded


__all__ = ["BEATsFeatureExtractor"]