# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

import numpy as np

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, is_torch_available, is_torchaudio_available, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_torchaudio_available():
    import torchaudio


@requires(backends=("torch", "torchaudio"))
class GraniteSpeech5FeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Granite Speech 5.0 feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of mel filter banks.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        n_fft (`int`, *optional*, defaults to 512):
            Size of the Fourier transform.
        win_length (`int`, *optional*, defaults to 400):
            Window length in samples.
        hop_length (`int`, *optional*, defaults to 160):
            Length of the overlapping windows for the STFT used to obtain the mel spectrogram, in samples.
        delta_win_length (`int`, *optional*, defaults to 3):
            Window length used to compute the delta features.
        logmel_floor_db (`float`, *optional*, defaults to 8.0):
            The log-mel features are floored at this many dB below the per-sample maximum.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio.
    """

    model_input_names = ["input_features", "attention_mask"]

    # fixed for this architecture, and mirrored by [`GraniteSpeech5Subsampling`], which sizes the encoder's
    # input projection from them: every mel frame is concatenated with its delta, and pairs of consecutive
    # frames are stacked
    delta_expansion = 2
    frame_stacking = 2

    def __init__(
        self,
        num_mel_bins: int = 80,
        sampling_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        delta_win_length: int = 3,
        logmel_floor_db: float = 8.0,
        padding_value: float = 0.0,
        **kwargs,
    ):
        kwargs.pop("feature_size", None)
        super().__init__(
            feature_size=num_mel_bins * self.delta_expansion * self.frame_stacking,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.num_mel_bins = num_mel_bins
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.delta_win_length = delta_win_length
        self.logmel_floor_db = logmel_floor_db
        self.mel_filters = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=num_mel_bins,
        )

    def get_num_encoder_frames(self, num_raw_samples: int) -> int:
        """Number of stacked encoder frames produced for a raw waveform of `num_raw_samples` samples."""
        mel_frames = num_raw_samples // self.hop_length
        return -(-mel_frames // self.frame_stacking)

    def __call__(
        self,
        raw_speech: "np.ndarray | list[float] | list[np.ndarray] | list[list[float]]",
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = None,
        padding: str | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        return_attention_mask: bool = True,
        device: str | None = "cpu",
        **kwargs,
    ) -> BatchFeature:
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

        # Convert to torch tensor
        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)) and isinstance(raw_speech[0], np.ndarray):
            raw_speech = [torch.tensor(speech) for speech in raw_speech]

        is_batched_torch = isinstance(raw_speech, torch.Tensor) and len(raw_speech.shape) > 1
        if is_batched_torch and len(raw_speech.shape) > 2:
            logger.warning(
                f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                "We will take the mean of the channels to convert to mono."
            )
            raw_speech = raw_speech.mean(-1)

        is_batched_sequence = isinstance(raw_speech, (list, tuple))
        if is_batched_sequence:
            for speech in raw_speech:
                if len(speech.shape) > 1:
                    logger.warning(
                        f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                        "We will take the mean of the channels to convert to mono."
                    )
                    speech = speech.mean(-1)

        if is_batched_torch or is_batched_sequence:
            raw_speech = [speech[:, None].to(torch.float32) for speech in raw_speech]
        else:
            raw_speech = [raw_speech[:, None].to(torch.float32)]

        audio_lengths = [len(speech) for speech in raw_speech]
        batched_speech = BatchFeature({"input_features": raw_speech, "audio_lengths": audio_lengths})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        raw_audio = padded_inputs.input_features.squeeze(-1)

        input_features = self._extract_features(raw_audio, device=device)

        data = {"input_features": input_features}
        if return_attention_mask:
            encoder_frame_counts = torch.tensor(
                [self.get_num_encoder_frames(length) for length in padded_inputs.audio_lengths.tolist()],
                device=input_features.device,
            )
            max_enc_frames = input_features.shape[1]
            attention_mask = (
                torch.arange(max_enc_frames, device=input_features.device)[None, :] < encoder_frame_counts[:, None]
            )
            data["attention_mask"] = attention_mask.long()

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _extract_features(self, audio: "torch.Tensor", device: str | None = "cpu") -> "torch.Tensor":
        """Compute the stacked log-mel(+delta) features consumed by the conformer encoder."""
        if device is not None:
            mel_filters = self.mel_filters.to(device)
            audio = audio.to(device)
        else:
            mel_filters = self.mel_filters

        batch_size = audio.shape[0]
        with torch.no_grad():
            # right-pad the waveform so the trailing partial frame-stacking group is filled rather than dropped
            mel_frames = audio.shape[1] // self.hop_length
            num_frames = self.frame_stacking * -(-mel_frames // self.frame_stacking)
            num_samples_needed = (num_frames - 1) * self.hop_length + 1
            if audio.shape[1] < num_samples_needed:
                audio = torch.nn.functional.pad(audio, (0, num_samples_needed - audio.shape[1]))
            mel = mel_filters(audio.float())[..., :num_frames]
            logmel = mel.clamp_min_(1e-10).log10_()
            mx = logmel.amax(dim=(-2, -1), keepdim=True)
            logmel = torch.maximum(logmel, mx - self.logmel_floor_db).div_(4).add_(1)
            deltas = torchaudio.functional.compute_deltas(logmel, win_length=self.delta_win_length)
            logmel = torch.cat((logmel, deltas), dim=-2)
            logmel = logmel.transpose(-1, -2)
            input_features = logmel.reshape(batch_size, -1, self.frame_stacking * logmel.shape[-1])
        return input_features


__all__ = ["GraniteSpeech5FeatureExtractor"]
