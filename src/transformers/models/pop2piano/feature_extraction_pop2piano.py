# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
""" Feature extractor class for Pop2Piano"""

import warnings
from typing import List, Optional, Union

import essentia
import essentia.standard
import librosa
import numpy as np
import scipy
import torch
from torch.nn.utils.rnn import pad_sequence

from ...audio_utils import fram_wave, get_mel_filter_banks, stft
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class Pop2PianoFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Pop2Piano feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
    This class extracts rhythm and does preprocesses before being passed through the transformer model.
        n_bars (`int`, *optional*, defaults to 2):
            Determines `n_steps` in method `preprocess_mel`.
        sampling_rate (`int`, *optional*, defaults to 22050):
            Sample rate of audio signal.
        use_mel (`bool`, *optional*, defaults to `True`):
            Whether to preprocess for `LogMelSpectrogram` or not. For the current implementation this must be `True`.
        padding_value (`int`, *optional*, defaults to 0):
            Padding value used to pad the audio. Should correspond to silences.
        n_fft (`int`, *optional*, defaults to 4096):
            Size of Fast Fourier Transform, creates n_fft // 2 + 1 bins.
        hop_length (`int`, *optional*, defaults to 1024):
            Length of hop between Short-Time Fourier Transform windows.
        f_min (`float`, *optional*, defaults to 10.0):
            Minimum frequency.
        n_mels (`int`, *optional*, defaults to 512):
            Number of mel filterbanks.
    """
    model_input_names = ["input_features"]

    def __init__(
        self,
        n_bars: int = 2,
        sampling_rate: int = 22050,
        use_mel: int = True,
        padding_value: int = 0,
        n_fft: int = 4096,
        hop_length: int = 1024,
        f_min: float = 10.0,
        n_mels: int = 512,
        feature_size=None,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.n_bars = n_bars
        self.sampling_rate = sampling_rate
        self.use_mel = use_mel
        self.padding_value = padding_value
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.n_mels = n_mels

    def log_mel_spectogram(self, sequence):
        """Generates MelSpectrogram then applies log base e."""

        mel_fb = get_mel_filter_banks(
            nb_frequency_bins=(self.n_fft // 2) + 1,
            nb_mel_filters=self.n_mels,
            frequency_min=self.f_min,
            frequency_max=float(self.sampling_rate // 2),
            sample_rate=self.sampling_rate,
            norm=None,
            mel_scale="htk",
        ).astype(np.float32)

        spectrogram = []
        for seq in sequence:
            window = np.hanning(self.n_fft + 1)[:-1]
            framed_audio = fram_wave(seq, self.hop_length, self.n_fft)
            spec = stft(framed_audio, window, fft_window_size=self.n_fft)
            spec = np.abs(spec) ** 2.0
            spectrogram.append(spec)

        spec_shape = spec.shape
        spectrogram = torch.Tensor(spectrogram).view(-1, *spec_shape)
        log_melspec = (
            torch.matmul(spectrogram.transpose(-1, -2), torch.from_numpy(mel_fb))
            .transpose(-1, -2)
            .clamp(min=1e-6)
            .log()
        )

        return log_melspec

    def extract_rhythm(self, raw_audio):
        """
        This algorithm(`RhythmExtractor2013`) extracts the beat positions and estimates their confidence as well as
        tempo in bpm for an audio signal. For more information please visit
        https://essentia.upf.edu/reference/std_RhythmExtractor2013.html .
        """
        essentia_tracker = essentia.standard.RhythmExtractor2013(method="multifeature")
        bpm, beat_times, confidence, estimates, essentia_beat_intervals = essentia_tracker(raw_audio)

        return bpm, beat_times, confidence, estimates, essentia_beat_intervals

    def interpolate_beat_times(self, beat_times, steps_per_beat, extend=False):
        beat_times_function = scipy.interpolate.interp1d(
            np.arange(beat_times.size),
            beat_times,
            bounds_error=False,
            fill_value="extrapolate",
        )
        if extend:
            beat_steps_8th = beat_times_function(np.linspace(0, beat_times.size, beat_times.size * steps_per_beat + 1))
        else:
            beat_steps_8th = beat_times_function(
                np.linspace(0, beat_times.size - 1, beat_times.size * steps_per_beat - 1)
            )
        return beat_steps_8th

    def extrapolate_beat_times(self, beat_times, n_extend=1):
        beat_times_function = scipy.interpolate.interp1d(
            np.arange(beat_times.size),
            beat_times,
            bounds_error=False,
            fill_value="extrapolate",
        )
        ext_beats = beat_times_function(np.linspace(0, beat_times.size + n_extend - 1, beat_times.size + n_extend))

        return ext_beats

    def preprocess_mel(
        self,
        audio,
        beatstep,
        n_bars,
        padding_value,
    ):
        """Preprocessing for `LogMelSpectrogram`"""

        n_steps = n_bars * 4
        n_target_step = len(beatstep)
        ext_beatstep = self.extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)

        def split_audio(audio):
            """
            Split audio corresponding beat intervals. Each audio's lengths are different. Because each corresponding
            beat interval times are different.
            """

            batch = []

            for i in range(0, n_target_step, n_steps):
                start_idx = i
                end_idx = min(i + n_steps, n_target_step)

                start_sample = int(ext_beatstep[start_idx] * self.sampling_rate)
                end_sample = int(ext_beatstep[end_idx] * self.sampling_rate)
                feature = audio[start_sample:end_sample]
                batch.append(feature)
            return batch

        batch = split_audio(audio)
        batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)

        return batch, ext_beatstep

    def single_preprocess(
        self,
        beatstep,
        feature_tokens=None,
        audio=None,
        n_bars=None,
    ):
        """Preprocessing method for a single sequence."""

        if feature_tokens is None and audio is None:
            raise ValueError("Both `feature_tokens` and `audio` can't be None at the same time!")
        if feature_tokens is not None:
            if len(feature_tokens.shape) != 1:
                raise ValueError(f"Expected `feature_tokens` shape to be (n, ) but found {feature_tokens.shape}")
        if audio is not None:
            if len(audio.shape) != 1:
                raise ValueError(f"Expected `audio` shape to be (n, ) but found {feature_tokens.shape}")
        n_bars = self.n_bars if n_bars is None else n_bars

        if beatstep[0] > 0.01:
            warnings.warn(f"Inference Warning : beatstep[0] is not 0 ({beatstep[0]}). all beatstep will be shifted.")
            beatstep = beatstep - beatstep[0]

        if self.use_mel:
            batch, ext_beatstep = self.preprocess_mel(
                audio,
                beatstep,
                n_bars=n_bars,
                padding_value=self.padding_value,
            )
        else:
            raise NotImplementedError("use_mel must be True")

        return batch, ext_beatstep

    def __call__(
        self,
        raw_audio: Union[np.ndarray, List[float], List[np.ndarray]],
        audio_sr: int,
        steps_per_beat: int = 2,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        **kwargs,
    ) -> BatchFeature:
        """
        Args:
        Main method to featurize and prepare for the model one sequence. Please note that `Pop2PianoFeatureExtractor`
        only accepts one raw_audio at a time.
            raw_audio (`np.ndarray`, `List`):
                Denotes the raw_audio.
            audio_sr (`int`):
                Denotes the Sampling Rate of `raw_audio`.
            steps_per_beat (`int`, *optional*, defaults to 2):
                Denotes Steps per beat.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to 'pt'):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        warnings.warn("Please make sure to have the audio sampling_rate as 44100, to get the optimal performence!")
        warnings.warn(
            "Pop2PianoFeatureExtractor only takes one raw_audio at a time, if you want to extract features from more than a single audio then you might need to call it multiple times."
        )

        # If it's [np.ndarray]
        if isinstance(raw_audio, list) and isinstance(raw_audio[0], np.ndarray):
            raw_audio = raw_audio[0]

        bpm, beat_times, confidence, estimates, essentia_beat_intervals = self.extract_rhythm(raw_audio=raw_audio)
        beatsteps = self.interpolate_beat_times(beat_times, steps_per_beat, extend=True)

        if self.sampling_rate != audio_sr and self.sampling_rate is not None:
            # Change audio_sr to self.sampling_rate
            raw_audio = librosa.core.resample(
                raw_audio, orig_sr=audio_sr, target_sr=self.sampling_rate, res_type="kaiser_best"
            )

        audio_sr = self.sampling_rate
        start_sample = int(beatsteps[0] * audio_sr)
        end_sample = int(beatsteps[-1] * audio_sr)
        _audio = torch.from_numpy(raw_audio)[start_sample:end_sample]
        fzs = None

        batch, ext_beatstep = self.single_preprocess(
            feature_tokens=fzs,
            audio=_audio,
            beatstep=beatsteps - beatsteps[0],
            n_bars=self.n_bars,
        )

        # Apply LogMelSpectogram
        batch = self.log_mel_spectogram(batch).transpose(-1, -2)

        batch = batch.cpu().numpy()
        output = BatchFeature(
            {
                "input_features": batch,
                "beatsteps": beatsteps,
                "ext_beatstep": ext_beatstep,
            }
        )

        if return_tensors is not None:
            output = output.convert_to_tensors(return_tensors)

        return output
