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
"""
Feature extractor class for Pop2Piano
"""
import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch.cuda

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging

import os
import scipy
import librosa
import essentia
import warnings
import essentia.standard
from torch.nn.utils.rnn import pad_sequence
from .configuration_pop2piano import Pop2PianoConfig

logger = logging.get_logger(__name__)

ESSENTIA_SAMPLERATE = 44100

class Pop2PianoFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Pop2Piano feature extractor.
    This feature extractor inherits from [`Pop2PianoFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.
    This class extracts mel-filter bank features from raw speech
    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, defaults to 160):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """

    model_input_names = ["input_features"]

    def __init__(self,
                 config: Pop2PianoConfig,
        ):
        self.config = config

    def extract_rhythm(self, raw_audio):
        """
        This algorithm(`RhythmExtractor2013`) extracts the beat positions and estimates their confidence as well as tempo in bpm for
        an audio signal
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
            beat_steps_8th = beat_times_function(
                np.linspace(0, beat_times.size, beat_times.size * steps_per_beat + 1)
            )
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
        ext_beats = beat_times_function(
            np.linspace(0, beat_times.size + n_extend - 1, beat_times.size + n_extend)
        )

        return ext_beats

    def preprocess_mel(
            self, audio, beatstep, n_bars, padding_value,
    ):
        n_steps = n_bars * 4
        n_target_step = len(beatstep)
        sample_rate = self.config.dataset.get("sample_rate", None)
        ext_beatstep = self.extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)

        def split_audio(audio):
            # Split audio corresponding beat intervals.
            # Each audio's lengths are different.
            # Because each corresponding beat interval times are different.
            batch = []

            for i in range(0, n_target_step, n_steps):
                start_idx = i
                end_idx = min(i + n_steps, n_target_step)

                start_sample = int(ext_beatstep[start_idx] * sample_rate)
                end_sample = int(ext_beatstep[end_idx] * sample_rate)
                feature = audio[start_sample:end_sample]
                batch.append(feature)
            return batch

        batch = split_audio(audio)
        batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)

        ext_beatstep = torch.from_numpy(ext_beatstep)

        return batch, ext_beatstep

    def single_preprocess(
            self,
            beatstep,
            feature_tokens=None,
            audio=None,
            n_bars=None,
    ):
        """
        generate a long audio sequence
        feature_tokens or audio : shape (time, )
        beatstep : shape (time, )
        - input_ids가 해당하는 beatstep 값들
        (offset 빠짐, 즉 beatstep[0] == 0)
        - beatstep[-1] : input_ids가 끝나는 지점의 시간값
        (즉 beatstep[-1] == len(y)//sr)
        """

        if feature_tokens is None and audio is None:
            raise ValueError("Both `feature_tokens` and `audio` can't be None at the same time!")
        if feature_tokens is not None:
            if len(feature_tokens.shape) != 1:
                raise ValueError(f"Expected `feature_tokens` shape to be (n, ) but found {feature_tokens.shape}")
        if audio is not None:
            if len(audio.shape) != 1:
                raise ValueError(f"Expected `audio` shape to be (n, ) but found {feature_tokens.shape}")
        n_bars = self.config.dataset.get("n_bars", None) if n_bars is None else n_bars

        if beatstep[0] > 0.01:
            warnings.warn(f"Inference Warning : beatstep[0] is not 0 ({beatstep[0]}). all beatstep will be shifted.")
            beatstep = beatstep - beatstep[0]

        if self.config.dataset.get("use_mel", None):
            input_ids = None
            batch, ext_beatstep = self.preprocess_mel(
                                                        audio,
                                                        beatstep,
                                                        n_bars=n_bars,
                                                        padding_value=self.config.pad_token_id,
                                                        )
        else:
            raise NotImplementedError("use_mel must be True")

        return input_ids, batch, ext_beatstep

    def __call__(self,
                 audio_path: str = None,
                 raw_audio:Union[np.ndarray, torch.tensor, list] = None,
                 audio_sr:int =None,
                 beatsteps=None,
                 steps_per_beat=2,
                 stereo_amp=0.5,
                 n_bars=2,
                 click_amp=0.2,
                 add_click=False,
                 ):

        # If only raw_audio is present then audio_sr also must be present
        if raw_audio is not None and audio_sr is None and audio_path is None:
            raise ValueError("`raw_audio` found but `audio_sr` not found")
            return

        if raw_audio is None and audio_path is None:
            raise ValueError("Either `raw_audio` or `audio_path` is needed!")
            return
        elif raw_audio is not None and audio_path is not  None:
            warnings.warn("Found both `raw_audio` and `audio_path` to be present, so using `audio_path`")
        elif raw_audio is None and audio_path is not None:
            raw_audio, sr = librosa.load(audio_path, sr=ESSENTIA_SAMPLERATE)
        else:
            sr = audio_sr

        if beatsteps is None:
            bpm, beat_times, confidence, estimates, essentia_beat_intervals = self.extract_rhythm(raw_audio=raw_audio)
            beat_times = np.array(beat_times)
            beatsteps = self.interpolate_beat_times(beat_times, steps_per_beat, extend=True)

        if self.config.dataset.get("use_mel", None) is not None:
            if self.config.dataset.get("sample_rate", None) != ESSENTIA_SAMPLERATE and self.config.dataset.get("sample_rate", None) is not None:
                # Change raw_audio_sr to config.dataset.sample_rate
                raw_audio = librosa.core.resample(
                    raw_audio, orig_sr=sr, target_sr=self.config.dataset.get("sample_rate"), res_type='kaiser_best'
                )
                sr = self.config.dataset.get("sample_rate")
                start_sample = int(beatsteps[0] * sr)
                end_sample = int(beatsteps[-1] * sr)
                _audio = torch.from_numpy(raw_audio)[start_sample:end_sample]
                fzs = None
        else:
            raise NotImplementedError("use_mel must be True")

        input_ids, batch, ext_beatstep = self.single_preprocess(
            feature_tokens=fzs,
            audio=_audio,
            beatstep=beatsteps - beatsteps[0],
            n_bars=n_bars,
        )

        return {"input_ids" : torch.from_numpy(input_ids) if input_ids is not None else None,
                "batch" : batch,
                "ext_beatstep" : ext_beatstep,
                }
