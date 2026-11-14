# Copyright 2026 Zyphra and The HuggingFace Inc. team. All rights reserved.
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
#
# Audio preparation and lane schedule ported from ZAC-1 (https://github.com/Zyphra/ZAC-1, Apache-2.0, Copyright Zyphra):
# `zac/encoder.py::prepare_audio`, `zac/batcher.py::IndependentBatcher` and `zac/semantic.py`.
import copy
import math
from typing import Any

import numpy as np

from ...audio_utils import AudioInput, make_list_of_audio, mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
from ...utils.import_utils import is_torch_available, is_torchaudio_available, requires


if is_torch_available():
    import torch
    import torch.nn.functional as F

if is_torchaudio_available():
    import torchaudio


logger = logging.get_logger(__name__)


@requires(backends=("torch", "torchaudio"))
class ZacFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a ZAC-1 feature extractor. It prepares the audio the way ZAC-1 does (mono, peak normalized only above 1,
    cut to whole code frames within `max_seconds`), pads it to a multiple of `hop_length` for the acoustic encoder and
    computes, for each encoder lane, the Wav2Vec2-BERT log-mel features of the lane's audio for the semantic teacher.

    Audio longer than `lane_frames` code frames is encoded in several lanes; every lane after the first starts
    `lane_context_frames` frames early to re-read them as left context.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features (mono audio).
        sampling_rate (`int`, *optional*, defaults to 44100):
            The sampling rate at which the audio should be digitalized, expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value used to pad the audio.
        hop_length (`int`, *optional*, defaults to 2940):
            Number of audio samples per code frame.
        lane_frames (`int`, *optional*, defaults to 300):
            Number of code frames per encoder lane.
        lane_context_frames (`int`, *optional*, defaults to 150):
            Number of left-context frames of every lane after the first.
        max_seconds (`float`, *optional*, defaults to 600.0):
            Longer audio is cut to the whole code frames within this duration.
        min_seconds (`float`, *optional*, defaults to 0.5):
            Shorter audio is rejected.
    """

    model_input_names = ["input_values", "padding_mask", "input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=44100,
        padding_value=0.0,
        hop_length=2940,
        lane_frames=300,
        lane_context_frames=150,
        max_seconds=600.0,
        min_seconds=0.5,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.hop_length = hop_length
        self.lane_frames = lane_frames
        self.lane_context_frames = lane_context_frames
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds
        # The preprocessing of the Wav2Vec2-BERT 2.0 teacher (`SeamlessM4TFeatureExtractor`): Kaldi-compatible log-mel
        # filterbanks at 16 kHz, normalized per mel bin, two frames stacked per feature
        self.semantic_sampling_rate = 16000
        self.semantic_num_mel_bins = 80
        self.semantic_stride = 2
        self.audio_padder = SequenceFeatureExtractor(feature_size=1, sampling_rate=sampling_rate, padding_value=0.0)
        self.audio_padder.model_input_names = ["audio", "padding_mask"]
        self._semantic_resampler = None
        self._semantic_mel_filters = None

    def _semantic_features(self, waveform: np.ndarray) -> np.ndarray:
        """Wav2Vec2-BERT features `[num_frames // 2, 160]` of one 16 kHz clip."""
        if self._semantic_mel_filters is None:
            self._semantic_mel_filters = mel_filter_bank(
                num_frequency_bins=257,
                num_mel_filters=self.semantic_num_mel_bins,
                min_frequency=20,
                max_frequency=self.semantic_sampling_rate // 2,
                sampling_rate=self.semantic_sampling_rate,
                norm=None,
                mel_scale="kaldi",
                triangularize_in_mel_space=True,
            )
        features = spectrogram(
            waveform * (2**15),  # Kaldi compliance: 16-bit signed integers
            window_function(400, "povey", periodic=False),
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            preemphasis=0.97,
            mel_filters=self._semantic_mel_filters,
            log_mel="log",
            mel_floor=1.192092955078125e-07,
            remove_dc_offset=True,
        ).T
        features = (features - features.mean(0, keepdims=True)) / np.sqrt(
            features.var(0, ddof=1, keepdims=True) + 1e-7
        )
        num_frames = features.shape[0] - features.shape[0] % self.semantic_stride
        return features[:num_frames].reshape(num_frames // self.semantic_stride, -1).astype(np.float32)

    def _resample_to_semantic(self, audio: "torch.Tensor") -> "torch.Tensor":
        # `torchaudio.transforms.Resample` caches a kernel built in float64; `torchaudio.functional.resample` builds it
        # in the input dtype, which differs slightly from the kernel the teacher was trained with.
        if self._semantic_resampler is None:
            self._semantic_resampler = torchaudio.transforms.Resample(self.sampling_rate, self.semantic_sampling_rate)
        return self._semantic_resampler(audio)

    def _prepare(self, audio) -> "torch.Tensor":
        """Mono float32 audio cut to whole code frames within `max_seconds` (`zac/encoder.py::prepare_audio`)."""
        audio = torch.as_tensor(np.asarray(audio, dtype=np.float32))
        if audio.ndim == 1:
            audio = audio[None]
        if audio.ndim != 2 or not audio.shape[-1] or not bool(torch.isfinite(audio).all()):
            raise ValueError(
                "Expected a finite, non-empty waveform of shape `(num_samples,)` or `(channels, num_samples)`."
            )
        audio = audio.mean(0)
        if audio.shape[-1] < self.sampling_rate * self.min_seconds:
            raise ValueError(f"Audio shorter than {self.min_seconds} s is not supported.")
        peak = float(audio.abs().amax())
        if peak > 1.0:
            audio = audio / peak
        maximum = math.floor(self.max_seconds * self.sampling_rate / self.hop_length) * self.hop_length
        return audio[:maximum].contiguous()

    def _lanes(self, num_frames: int) -> list[tuple[int, int, int]]:
        """`(left, context, count)` of each lane: the lane reads frames `[left, left + context + count)` and outputs the
        last `count` of them."""
        lanes, start = [], 0
        while start < num_frames:
            context = 0 if start == 0 else self.lane_context_frames
            count = min(num_frames - start, self.lane_frames - context)
            lanes.append((start - context, context, count))
            start += count
        return lanes

    def __call__(
        self,
        audio: AudioInput,
        sampling_rate: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
            audio (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                One clip of shape `(num_samples,)` or `(channels, num_samples)`, or a list of clips (channels are
                averaged).
            sampling_rate (`int`, *optional*):
                The sampling rate of `audio`; it must be `self.sampling_rate`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to return (`"pt"` or `"np"`).
        """
        if sampling_rate is None:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )
        elif sampling_rate != self.sampling_rate:
            raise ValueError(
                f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a "
                f"sampling rate of {self.sampling_rate}. Please make sure that the provided `audio` input was sampled "
                f"with {self.sampling_rate} and not {sampling_rate}."
            )

        clips = [self._prepare(clip) for clip in make_list_of_audio(audio)]
        padded = self.audio_padder.pad(
            BatchFeature({"audio": clips}),
            padding=True,
            pad_to_multiple_of=self.hop_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = padded["audio"][:, None, :]
        padding_mask = padded["attention_mask"]

        lane_samples = self.lane_frames * self.hop_length
        features, masks = [], []
        for clip in clips:
            clip_features, clip_masks = [], []
            for left, context, count in self._lanes(math.ceil(clip.shape[-1] / self.hop_length)):
                lane = clip[left * self.hop_length : (left + context + count) * self.hop_length]
                lane = F.pad(lane, (0, lane_samples - lane.shape[-1]))
                features_ = torch.from_numpy(self._semantic_features(self._resample_to_semantic(lane).numpy()))
                clip_features.append(features_)
                clip_masks.append(torch.ones(features_.shape[0], dtype=torch.long))
            features.append(torch.stack(clip_features))
            masks.append(torch.stack(clip_masks))

        num_lanes = max(len(clip_features) for clip_features in features)
        input_features = torch.zeros((len(clips), num_lanes, *features[0].shape[1:]), dtype=torch.float32)
        input_features_mask = torch.zeros((len(clips), num_lanes, features[0].shape[1]), dtype=torch.long)
        for i, (clip_features, clip_masks) in enumerate(zip(features, masks)):
            input_features[i, : len(clip_features)] = clip_features
            input_features_mask[i, : len(clip_masks)] = clip_masks

        return BatchFeature(
            {
                "input_values": input_values,
                "padding_mask": padding_mask,
                "input_features": input_features,
                "input_features_mask": input_features_mask,
            },
            tensor_type=return_tensors,
        )

    def to_dict(self) -> dict[str, Any]:
        # the semantic preprocessing is fixed by the teacher: only the init arguments are saved
        output = copy.deepcopy(
            {k: v for k, v in self.__dict__.items() if not k.startswith(("_semantic", "semantic_", "audio_padder"))}
        )
        output["feature_extractor_type"] = self.__class__.__name__
        return output


__all__ = ["ZacFeatureExtractor"]
