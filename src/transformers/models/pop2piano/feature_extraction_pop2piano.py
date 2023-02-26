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


import librosa
import essentia
import warnings
import scipy.interpolate as interp
from torch.nn.utils.rnn import pad_sequence
from .configuration_pop2piano import Pop2PianoConfig, Pop2PianoProcessorConfig

logger = logging.get_logger(__name__)

class LogMelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=4096,
            hop_length=1024,
            f_min=10.0,
            n_mels=512,
        )

    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                X = self.melspectrogram(x)
                X = X.clamp(min=1e-6).log()

        return X

class ConcatEmbeddingToMel(nn.Module):
    def __init__(self, embedding_offset, n_vocab, n_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_dim)
        self.embedding_offset = embedding_offset

    def forward(self, feature, index_value):
        """
        index_value : (batch, )
        feature : (batch, time, feature_dim)
        """
        index_shifted = index_value - self.embedding_offset

        # (batch, 1, feature_dim)
        composer_embedding = self.embedding(index_shifted).unsqueeze(1)
        # print(composer_embedding.shape, feature.shape)
        # (batch, 1 + time, feature_dim)
        inputs_embeds = torch.cat([composer_embedding, feature], dim=1)
        return inputs_embeds

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

    ESSENTIA_SAMPLERATE = 44100
    model_input_names = ["input_features"]

    def __init__(self,
                 config: Pop2PianoProcessorConfig,
                 model_config: Pop2PianoConfig
        ):
        self.config = config
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.config.dataset.use_mel:
            self.spectrogram = LogMelSpectrogram()
            if self.config.dataset.mel_is_conditioned:
                n_dim = 512
                composer_n_vocab = len(self.config.composer_to_feature_token)
                embedding_offset = min(self.config.composer_to_feature_token.values())
                self.mel_conditioner = ConcatEmbeddingToMel(
                    embedding_offset=embedding_offset,
                    n_vocab=composer_n_vocab,
                    n_dim=n_dim,
                )
                # TODO
                self.mel_conditioner.load_state_dict(...)
                # TODO
        else:
            self.spectrogram = None

    def extract_rhythm(self, raw_audio):
        """
        This algorithm(`RhythmExtractor2013`) extracts the beat positions and estimates their confidence as well as tempo in bpm for
        an audio signal
        """
        essentia_tracker = essentia.standard.RhythmExtractor2013(method="multifeature")
        (
            bpm,
            beat_times,
            confidence,
            estimates,
            essentia_beat_intervals,
        ) = essentia_tracker(raw_audio)
        return bpm, beat_times, confidence, estimates,

    def interpolate_beat_times(self, beat_times, steps_per_beat, extend=False):
        beat_times_function = interp.interp1d(
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

    def extrapolate_beat_times(beat_times, n_extend=1):
        beat_times_function = interp.interp1d(
            np.arange(beat_times.size),
            beat_times,
            bounds_error=False,
            fill_value="extrapolate",
        )
        ext_beats = beat_times_function(
            np.linspace(0, beat_times.size + n_extend - 1, beat_times.size + n_extend)
        )

        return ext_beats

    def prepare_inference_mel(
            self, audio, beatstep, n_bars, padding_value, composer_value=None
    ):
        n_steps = n_bars * 4
        n_target_step = len(beatstep)
        sample_rate = self.config.dataset.sample_rate
        ext_beatstep = extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)

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

        inputs_embeds = self.spectrogram(batch).transpose(-1, -2)
        if self.config.dataset.mel_is_conditioned:
            composer_value = torch.tensor(composer_value).to(self.device)
            composer_value = composer_value.repeat(inputs_embeds.shape[0])
            inputs_embeds = self.mel_conditioner(inputs_embeds, composer_value)
        return inputs_embeds, ext_beatstep

    def single_preprocess(
            self,
            beatstep,
            feature_tokens=None,
            audio=None,
            max_length=256,
            max_batch_size=64,
            n_bars=None,
            composer_value=None,
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
        n_bars = self.config.dataset.n_bars if n_bars is None else n_bars

        if beatstep[0] > 0.01:
            warnings.warn(f"Inference Warning : beatstep[0] is not 0 ({beatstep[0]}). all beatstep will be shifted.")
            beatstep = beatstep - beatstep[0]

        input_ids = None
        inputs_embeds, ext_beatstep = self.prepare_inference_mel(
            audio,
            beatstep,
            n_bars=n_bars,
            padding_value=self.model_config.pad_token_id,
            composer_value=composer_value,
        )


        #
        # batch_size = inputs_embeds.shape[0]
        # # Considering GPU capacity, some sequence would not be generated at once.
        # relative_tokens = list()
        # for i in range(0, batch_size, max_batch_size):
        #     start = i
        #     end = min(batch_size, i + max_batch_size)
        #
        #     if input_ids is None:
        #         _input_ids = None
        #         _inputs_embeds = inputs_embeds[start:end]
        #     else:
        #         _input_ids = input_ids[start:end]
        #         _inputs_embeds = None
        #
        #     _relative_tokens = self.transformer.generate(
        #         input_ids=_input_ids,
        #         inputs_embeds=_inputs_embeds,
        #         max_length=max_length,
        #     )
        #     _relative_tokens = _relative_tokens.cpu().numpy()
        #     relative_tokens.append(_relative_tokens)
        #
        # max_length = max([rt.shape[-1] for rt in relative_tokens])
        # for i in range(len(relative_tokens)):
        #     relative_tokens[i] = np.pad(
        #         relative_tokens[i],
        #         [(0, 0), (0, max_length - relative_tokens[i].shape[-1])],
        #         constant_values=PAD,
        #     )
        # relative_tokens = np.concatenate(relative_tokens)
        #
        # pm, notes = self.tokenizer.relative_batch_tokens_to_midi(
        #     relative_tokens,
        #     beatstep=ext_beatstep,
        #     bars_per_batch=n_bars,
        #     cutoff_time_idx=(n_bars + 1) * 4,
        # )
        #
        # return relative_tokens, notes,

        return input_ids, inputs_embeds, ext_beatstep


    def __call__(self,
                 raw_audio:Union[np.ndarray, torch.tensor, list] = None,
                 audio_sr:int =None,
                 audio_path:str = None,
                 composer=None,
                 model="generated",
                 steps_per_beat=2,
                 stereo_amp=0.5,
                 n_bars=2,
                 # ignore_duplicate=True,
                 # show_plot=False,
                 save_midi=False,
                 save_mix=False,
                 midi_path=None,
                 mix_path=None,
                 click_amp=0.2,
                 add_click=False,
                 max_batch_size=None,
                 beatsteps=None,
                 mix_sample_rate=None,

                 ):

        # If only raw_audio is present then audio_sr also must be present
        if raw_audio is not None and audio_sr is None and audio_path is None:
            raise ValueError("`raw_audio found but` `audio_sr` not found")
            return

        if raw_audio is None and audio_path is None:
            raise ValueError("Either `raw_audio` or `audio_path` is needed!")
            return
        elif raw_audio is not None and audio_path is not  None:
            warnings.warn("Found both `raw_audio` and `audio_path` to be present, so using `audio_path`")
        elif raw_audio is None and audio_path is not None:
            extension = os.path.splitext(audio_path)[1]
            mix_path = (
                audio_path.replace(extension, f".{model}.{composer}.wav")
                if mix_path is None
                else mix_path
            )
            midi_path = (
                audio_path.replace(extension, f".{model}.{composer}.mid")
                if midi_path is None
                else midi_path
            )
            raw_audio, sr = librosa.load(audio_path, sr=ESSENTIA_SAMPLERATE)
        else:
            sr = audio_sr

        max_batch_size = 64 // n_bars if max_batch_size is None else max_batch_size

        # select composer randomly if not already given
        composer_to_feature_token = self.config.composer_to_feature_token
        if composer is None:
            composer = np.random.sample(list(composer_to_feature_token.keys()), 1)[0]
        composer_value = composer_to_feature_token[composer]
        mix_sample_rate = (
            config.dataset.sample_rate if mix_sample_rate is None else mix_sample_rate
        )

        bpm, beat_times, confidence, estimates, essentia_beat_intervals = self.extract_rhythm(raw_audio=raw_audio)
        beat_times = np.array(beat_times)
        beatsteps = self.interpolate_beat_times(beat_times, steps_per_beat, extend=True)

        # Change raw_audio_sr to config.dataset.sample_rate
        raw_audio = librosa.core.resample(
            raw_audio, orig_sr=sr, target_sr=config.dataset.sample_rate
        )
        sr = config.dataset.sample_rate
        start_sample = int(beatsteps[0] * sr)
        end_sample = int(beatsteps[-1] * sr)
        _audio = torch.from_numpy(raw_audio)[start_sample:end_sample].to(self.device)
        fzs = None




        relative_tokens, notes, pm = self.single_preprocess(
            feature_tokens=fzs,
            audio=_audio,
            beatstep=beatsteps - beatsteps[0],
            max_length=config.dataset.target_length * max(1, (n_bars // config.dataset.n_bars)),
            max_batch_size=max_batch_size,
            n_bars=n_bars,
            composer_value=composer_value,
        )






