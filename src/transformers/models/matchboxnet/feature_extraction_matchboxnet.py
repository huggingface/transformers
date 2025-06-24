import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from transformers import FeatureExtractionMixin

class MatchboxNetFeatureExtractor(FeatureExtractionMixin):
    """
    Feature extractor for MatchboxNet audio classification.

    Args:
        target_sr (int): Target sampling rate, e.g., 16000.
        n_mfcc (int): Number of MFCC coefficients, e.g., 64.
        n_fft (int): FFT window size in samples (default: 0.025 * target_sr).
        hop_length (int): Hop length in samples (default: 0.010 * target_sr).
        fixed_length (int): Number of time frames to pad/truncate to (e.g., 128).
    """

    model_input_names = ["input_ids"]

    def __init__(
        self,
        target_sr: int = 16000,
        n_mfcc: int = 64,
        n_fft: int = None,
        hop_length: int = None,
        fixed_length: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        # Set FFT window length and hop length if not provided
        self.n_fft = n_fft if n_fft is not None else int(0.025 * target_sr)
        self.hop_length = hop_length if hop_length is not None else int(0.010 * target_sr)
        self.fixed_length = fixed_length
        # Required by the pipeline to know sampling rate
        self.sampling_rate = target_sr

    def _get_mfcc_transform(self) -> torchaudio.transforms.MFCC:
        """
        Create a torchaudio MFCC transform configured to our parameters.
        """
        return torchaudio.transforms.MFCC(
            sample_rate=self.target_sr,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mfcc,
                "win_length": self.n_fft,
                "window_fn": torch.hann_window,
            },
        )

    def __call__(
        self,
        audio_input,
        sampling_rate: int = None,
        return_tensors: str = None,
        **kwargs,
    ) -> dict:
        """
        Process raw audio into MFCC features.

        Args:
            audio_input: Path to file, numpy array, or torch Tensor (1D or 2D).
            sampling_rate: Original sampling rate if array or tensor provided.
            return_tensors: Set to 'pt' to return torch.Tensor instead of numpy.

        Returns:
            Dict with key 'input_ids' and value of shape (n_mfcc, fixed_length).
        """
        # Load or convert input to numpy array and determine sr
        if isinstance(audio_input, str):
            waveform, sr = torchaudio.load(audio_input)
            # If multichannel, average to mono
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            array = waveform.squeeze(0).numpy()
        elif isinstance(audio_input, np.ndarray):
            array = audio_input
            sr = sampling_rate or self.target_sr
        elif isinstance(audio_input, torch.Tensor):
            array = audio_input.squeeze().numpy()
            sr = sampling_rate or self.target_sr
        else:
            raise ValueError(f"Unsupported input type: {type(audio_input)}")

        # Resample if sampling rate differs
        if sr != self.target_sr:
            waveform = torch.from_numpy(array).unsqueeze(0)
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.target_sr,
            )
            array = waveform.squeeze(0).numpy()
            sr = self.target_sr

        # Compute MFCC features
        waveform = torch.from_numpy(array).float().unsqueeze(0)
        mfcc = self._get_mfcc_transform()(waveform).squeeze(0)  # Shape: (n_mfcc, T)
        T_actual = mfcc.shape[1]

        # Pad or truncate to fixed_length frames
        if T_actual < self.fixed_length:
            total_pad = self.fixed_length - T_actual
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            mfcc = F.pad(mfcc, (pad_left, pad_right), mode="constant", value=0.0)
        else:
            start = (T_actual - self.fixed_length) // 2
            mfcc = mfcc[:, start : start + self.fixed_length]

        features = {"input_ids": mfcc.numpy()}

        # Convert to torch.Tensor if requested by pipeline
        if return_tensors == "pt":
            features = {k: torch.tensor(v) for k, v in features.items()}

        return features
