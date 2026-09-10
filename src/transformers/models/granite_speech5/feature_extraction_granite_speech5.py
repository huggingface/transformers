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
from ...utils import PaddingStrategy, TensorType, is_torch_available, is_torchaudio_available, logging
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

    # hardcoded in modeling for this architecture (see [`GraniteSpeech5Subsampling`]):
    # every mel frame is concatenated with its delta, and pairs of consecutive frames are stacked
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

        clips = self._as_mono_clips(raw_speech)
        raw_audio, audio_lengths = self._pad_clips(
            clips,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        input_features = self._extract_features(raw_audio, device=device)

        data = {"input_features": input_features}
        if return_attention_mask:
            mel_frames = torch.tensor(audio_lengths, device=input_features.device) // self.hop_length
            encoder_frame_counts = -(-mel_frames // self.frame_stacking)
            max_enc_frames = input_features.shape[1]
            attention_mask = (
                torch.arange(max_enc_frames, device=input_features.device)[None, :] < encoder_frame_counts[:, None]
            )
            data["attention_mask"] = attention_mask.long()

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _as_mono_clips(self, raw_speech) -> list["torch.Tensor"]:
        """Normalise any accepted input into a list of 1-D float32 waveforms."""
        if isinstance(raw_speech, np.ndarray) or (isinstance(raw_speech, torch.Tensor) and raw_speech.ndim > 1):
            # a single array is one clip unless it is batched, i.e. (batch_size, num_samples[, num_channels])
            clips = list(raw_speech) if raw_speech.ndim > 1 else [raw_speech]
        elif isinstance(raw_speech, (list, tuple)):
            clips = list(raw_speech)
        else:
            clips = [raw_speech]

        mono_clips = []
        for clip in clips:
            if not isinstance(clip, torch.Tensor):
                clip = torch.from_numpy(np.require(clip, requirements=["W"]))
            if clip.ndim > 1:
                logger.warning(
                    f"Only mono-channel audio is supported for input to {self.__class__.__name__}. "
                    "We will take the mean of the channels to convert to mono."
                )
                clip = clip.mean(-1)
            mono_clips.append(clip.to(torch.float32))
        return mono_clips

    def _pad_clips(
        self,
        clips: list["torch.Tensor"],
        padding: str | bool | None,
        max_length: int | None,
        truncation: bool,
        pad_to_multiple_of: int | None,
    ) -> tuple["torch.Tensor", list[int]]:
        """
        [`~feature_extraction_sequence_utils.SequenceFeatureExtractor.pad()`] targets extracted features,
        so its per-clip padding and allocations are cheap for mel frames but costly for raw waveforms.
        A preallocated buffer with one copy per clip is ~8× faster for 30 s batches.

        TODO: @eustlb, this should be covered by #44394
        """
        lengths = [clip.shape[0] for clip in clips]
        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)

        if truncation and max_length is not None:
            lengths = [min(length, max_length) for length in lengths]

        if padding_strategy == PaddingStrategy.MAX_LENGTH:
            target_length = max_length
        else:
            target_length = max(lengths)
            if padding_strategy == PaddingStrategy.DO_NOT_PAD and min(lengths) != target_length:
                raise ValueError(
                    "Cannot build a batch of unequal-length waveforms with `padding=False`; pass "
                    "`padding='longest'`/`padding='max_length'` or feed one clip at a time."
                )
        if pad_to_multiple_of is not None:
            target_length = -(-target_length // pad_to_multiple_of) * pad_to_multiple_of
        if max(lengths) > target_length:
            raise ValueError(
                f"Got a waveform of {max(lengths)} samples, which does not fit the batch length of "
                f"{target_length} implied by `padding`/`max_length`; pass `truncation=True` or a larger "
                "`max_length`."
            )

        buffer = np.full((len(clips), target_length), self.padding_value, dtype=np.float32)
        for index, (clip, length) in enumerate(zip(clips, lengths)):
            buffer[index, :length] = clip[:length].detach().cpu().numpy()
        raw_audio = torch.from_numpy(buffer)
        return raw_audio, lengths

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
