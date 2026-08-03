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
class GraniteSpeechNarFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        frame_stacking: int = 2,
        padding_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.frame_stacking = frame_stacking
        self.mel_filters = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=feature_size,
        )

    def get_num_encoder_frames(self, num_raw_samples: int) -> int:
        """Number of stacked encoder frames produced for a raw waveform of `num_raw_samples` samples."""
        mel_frames = num_raw_samples // self.hop_length + 1
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

        input_features = self._extract_mel_spectrograms(raw_audio, device=device)

        encoder_frame_counts = torch.tensor(
            [self.get_num_encoder_frames(length) for length in padded_inputs.audio_lengths.tolist()],
            device=input_features.device,
        )
        max_enc_frames = input_features.shape[1]
        input_features_mask = (
            torch.arange(max_enc_frames, device=input_features.device)[None, :] < (encoder_frame_counts[:, None])
        )

        return BatchFeature(
            data={
                "input_features": input_features,
                "input_features_mask": input_features_mask,
            },
            tensor_type=return_tensors,
        )

    def _extract_mel_spectrograms(self, audio: "torch.Tensor", device: str | None = "cpu") -> "torch.Tensor":
        """Compute the stacked log-mel features consumed by the conformer encoder."""
        if device is not None:
            mel_filters = self.mel_filters.to(device)
            audio = audio.to(device)
        else:
            mel_filters = self.mel_filters

        batch_size = audio.shape[0]
        with torch.no_grad():
            mel = mel_filters(audio.float())
            remainder = mel.shape[-1] % self.frame_stacking
            if remainder != 0:
                mel = torch.nn.functional.pad(mel, (0, self.frame_stacking - remainder))
            logmel = mel.transpose(-1, -2).clamp_min_(1e-10).log10_()
            mx = logmel.amax(dim=(-2, -1), keepdim=True)
            logmel = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
            input_features = logmel.reshape(batch_size, -1, self.frame_stacking * self.feature_size)
        return input_features


__all__ = ["GraniteSpeechNarFeatureExtractor"]
