import types
from transformers import FeatureExtractionMixin
from transformers.tokenization_utils_base import BatchEncoding

from transformers.utils.import_utils import is_torch_available

if is_torch_available():
    
    import torch
    import torchaudio
    import numpy as np
    import torch.nn.functional as F
   
else:
    
    raise ImportError("MatchboxNet requires PyTorch.")
  

class MatchboxNetFeatureExtractor(FeatureExtractionMixin):
    model_input_names = ["input_ids"]

    def __init__(self,
                 target_sr: int = 16000,
                 n_mfcc: int = 64,
                 n_fft: int = None,
                 hop_length: int = None,
                 fixed_length: int = 128,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft or int(0.025 * target_sr)
        self.hop_length = hop_length or int(0.010 * target_sr)
        self.fixed_length = fixed_length
        self.sampling_rate = target_sr

    def _get_mfcc_transform(self):
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

    def __call__(self,
                 audio_input,
                 sampling_rate: int = None,
                 return_tensors: str = None,
                 **kwargs) -> BatchEncoding:

        # 1) Charger le signal en numpy
        if isinstance(audio_input, str):
            waveform, sr = torchaudio.load(audio_input)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            array = waveform.squeeze(0).numpy()
        elif isinstance(audio_input, np.ndarray):
            array = audio_input
            sr = sampling_rate if sampling_rate is not None else self.target_sr
        elif isinstance(audio_input, torch.Tensor):
            # Tensor 1D ou 2D (canal x time)
            if audio_input.ndim == 1:
                array = audio_input.numpy()
            else:
                array = audio_input.squeeze(0).numpy()
            sr = sampling_rate if sampling_rate is not None else self.target_sr
        else:
            raise ValueError(f"Unsupported input type: {type(audio_input)}")

        # 2) Resample si nécessaire
        if sr != self.target_sr:
            wav = torch.from_numpy(array).unsqueeze(0)
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.target_sr
            )
            array = wav.squeeze(0).numpy()

        # 3) Calcul des MFCC
        wav  = torch.from_numpy(array).float().unsqueeze(0)
        mfcc = self._get_mfcc_transform()(wav).squeeze(0)
        T    = mfcc.shape[1]

        # 4) Pad / truncate
        if T < self.fixed_length:
            pad   = self.fixed_length - T
            left  = pad // 2
            right = pad - left
            mfcc = F.pad(mfcc, (left, right), value=0.0)
        else:
            start = (T - self.fixed_length) // 2
            mfcc  = mfcc[:, start:start + self.fixed_length]

        # 5) On prépare le dict numpy
        features = {"input_ids": mfcc.numpy()}

        # 6) Si on veut des Tensors PyTorch…
        if return_tensors == "pt":
            # ajoute la batch‐dim
            tensor_feats = {
                k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
                for k, v in features.items()
            }
            batch = BatchEncoding(tensor_feats)

            # monkey‐patch .to(dtype=…) attendu par le pipeline
            def _to(self, *args, device=None, dtype=None, **kwargs):
                new = {}
                for k, v in self.items():
                    if dtype is not None and hasattr(v, "to"):
                        new[k] = v.to(dtype=dtype)
                    else:
                        new[k] = v
                return BatchEncoding(new)

            batch.to = types.MethodType(_to, batch)
            return batch

        # 7) Sinon on renvoie un BatchEncoding classique (avec numpy)
        return BatchEncoding(features)
