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
""" Feature extractor class for Pop2Piano """

import copy
from typing import Any, Dict, List, Optional, Union

import os
import torch
import scipy
import librosa
import essentia
import warnings
import note_seq
import pretty_midi
import numpy as np
import soundfile as sf
import essentia.standard
import IPython.display as ipd
from IPython.display import display
from torch.nn.utils.rnn import pad_sequence
from .configuration_pop2piano import Pop2PianoConfig

from ...utils import TensorType, logging
from ...feature_extraction_utils import BatchFeature
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor

logger = logging.get_logger(__name__)

ESSENTIA_SAMPLERATE: int = 44100

TOKEN_SPECIAL: int = 0
TOKEN_NOTE: int = 1
TOKEN_VELOCITY: int = 2
TOKEN_TIME: int = 3

DEFAULT_VELOCITY: int = 77

TIE: int = 2
EOS: int = 1
PAD: int = 0

class Pop2PianoFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a Pop2Piano feature extractor.
    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.
    This class loads audio, extracts rhythm and does preprocesses before being passed through `LogMelSpectrogram`.
    This class also contains postprocessing methods to convert model outputs to midi audio and stereo-mix.
    Args:
        n_bars (`int`, *optional*, defaults to 2):
            Determines `n_steps` in method `preprocess_mel`.
        sample_rate (`int`, *optional*, defaults to 22050):
            Sample rate of audio signal.
        use_mel (`bool`, *optional*, defaults to `True`):
            Whether to preprocess for `LogMelSpectrogram` or not.
            For the current implementation this must be `True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding value used to pad the audio. Should correspond to silences.
        vocab_size_special (`int`, *optional*, 4):
            No of special values.
        vocab_size_note (`int`, *optional*, defaults to 128):
             Note values indicate a pitch event for one of the MIDI pitches. But only the 88 pitches corresponding to piano keys are actually used.
             This represents the number of Note Values.
        vocab_size_velocity (`int`, *optional*, defaults to 2):
            No of Velocity tokens.
        vocab_size_time (`int`, *optional*, defaults to 100):
            Beat Shift [100 values] Indicates the relative time shift within the segment quantized into 8th-note beats(half-beats).
            This represents the number of Beat Shifts.
    """
    model_input_names = ["input_features"]

    def __init__(self,
                 n_bars:int = 2,
                 sample_rate:int = 22050,
                 use_mel:int = True,
                 pad_token_id:int = 0,
                 vocab_size_special:int = 4,
                 vocab_size_note:int = 128,
                 vocab_size_velocity:int = 2,
                 vocab_size_time:int = 100,
                 **kwargs
        ):
        self.n_bars = n_bars
        self.sample_rate = sample_rate
        self.use_mel = use_mel
        self.pad_token_id = pad_token_id
        self.vocab_size_special = vocab_size_special
        self.vocab_size_note = vocab_size_note
        self.vocab_size_velocity = vocab_size_velocity
        self.vocab_size_time = vocab_size_time

    def extract_rhythm(self, raw_audio):
        """
        This algorithm(`RhythmExtractor2013`) extracts the beat positions and estimates their confidence as well as tempo in bpm for an audio signal.
        For more information please visit https://essentia.upf.edu/reference/std_RhythmExtractor2013.html .
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
        """ Preprocessing for `LogMelSpectrogram` """

        n_steps = n_bars * 4
        n_target_step = len(beatstep)
        ext_beatstep = self.extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)

        def split_audio(audio):
            """
            Split audio corresponding beat intervals.
            Each audio's lengths are different.
            Because each corresponding beat interval times are different.
            """

            batch = []

            for i in range(0, n_target_step, n_steps):
                start_idx = i
                end_idx = min(i + n_steps, n_target_step)

                start_sample = int(ext_beatstep[start_idx] * self.sample_rate)
                end_sample = int(ext_beatstep[end_idx] * self.sample_rate)
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
        """ preprocessing method for a single sequence. """
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
            input_ids = None
            batch, ext_beatstep = self.preprocess_mel(audio,
                                                      beatstep,
                                                      n_bars=n_bars,
                                                      padding_value=self.pad_token_id,
                                                      )
        else:
            raise NotImplementedError("use_mel must be True")

        return input_ids, batch, ext_beatstep

    def __call__(self,
                 audio_path:str = None,
                 raw_audio:Union[np.ndarray, torch.tensor, list] = None,
                 audio_sr:int =None,
                 beatsteps=None,
                 steps_per_beat=2,
                 # n_bars=2,
                 add_click=False,
                 click_amp=0.2,
                 ):
        """
        Main method to featurize and prepare for the model one or several sequence(s).
        Args:
            audio_path (`string`):
                Audio path from where the audio sequence will be loaded.
                Can be `None if `raw_audio` and `audio_sr` is given.
            raw_audio (`np.ndarray`, `torch.tensor`, `List`):
                Denotes the raw_audio. Can be `None` if `audio_path` is present.
                If `audio_path` is `None` then `raw_audio` and `audio_sr` must not be `None`.
            audio_sr (`int`):
                Denotes the Sample Rate of `raw_audio`.
                If `audio_path` is `None` then `audio_sr` and `raw_audio` must not be `None`.
            beatsteps (`np.ndarray`):
                beatsteps is generated by extracting rhythm from raw_audio and then using interpolation.
                If beatsteps is given then it reduces the computation.
            steps_per_beat (`int`, *optional*, defaults to 2):
                Denotes Steps per beat.
            add_click (`bool`, *optional*, defaults to `False`):
                Constructs a `"click track"`.
            click_amp (`float`, *optional*, defaults to 0.2):
                Amplitude for `"click track"`.
        """

        # If only raw_audio is present then audio_sr also must be present
        if raw_audio is not None and audio_sr is None and audio_path is None:
            raise ValueError("`raw_audio` found but `audio_sr` not found")
            return

        if raw_audio is None and audio_path is None:
            raise ValueError("Either `raw_audio` or `audio_path` is needed!")
            return
        elif raw_audio is not None and audio_path is not  None:
            warnings.warn("Found both `raw_audio` and `audio_path` to be present, so using `raw_audio`")
        elif raw_audio is None and audio_path is not None:
            raw_audio, sr = librosa.load(audio_path, sr=ESSENTIA_SAMPLERATE)
        else:
            sr = audio_sr

        if beatsteps is None:
            bpm, beat_times, confidence, estimates, essentia_beat_intervals = self.extract_rhythm(raw_audio=raw_audio)
            beat_times = np.array(beat_times)
            beatsteps = self.interpolate_beat_times(beat_times, steps_per_beat, extend=True)

        if self.use_mel is not None:
            if self.sample_rate != ESSENTIA_SAMPLERATE and self.sample_rate is not None:
                # Change `raw_audio_sr` to `self.sample_rate`
                raw_audio = librosa.core.resample(
                    raw_audio, orig_sr=sr, target_sr=self.sample_rate, res_type='kaiser_best'
                )
                sr = self.sample_rate
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
            n_bars=self.n_bars,
        )

        return {"input_ids" : input_ids,
                "batch" : batch,
                "beatsteps" : beatsteps,
                "ext_beatstep" : ext_beatstep,
                "raw_audio" : raw_audio,
                "sr" : sr,
                }

    def detokenize(self, token, time_idx_offset):
        if token >= (self.vocab_size_special + self.vocab_size_note + self.vocab_size_velocity):
            type, value =  TOKEN_TIME, ((token - (self.vocab_size_special + self.vocab_size_note + self.vocab_size_velocity)) + time_idx_offset)
        elif token >= (self.vocab_size_special + self.vocab_size_note):
            type, value =  TOKEN_VELOCITY, (token - (self.vocab_size_special + self.vocab_size_note))
            value = int(value)
        elif token >= self.vocab_size_special:
            type, value =  TOKEN_NOTE, (token - self.vocab_size_special)
            value = int(value)
        else:
            type, value = TOKEN_SPECIAL, token
            value = int(value)

        return [type, value]

    def relative_batch_tokens_to_midi(
            self,
            tokens,
            beatstep,
            beat_offset_idx=None,
            bars_per_batch=None,
            cutoff_time_idx=None,
    ):
        beat_offset_idx = 0 if beat_offset_idx is None else beat_offset_idx
        notes = None
        bars_per_batch = 2 if bars_per_batch is None else bars_per_batch

        N = len(tokens)
        for n in range(N):
            _tokens = tokens[n]
            _start_idx = beat_offset_idx + n * bars_per_batch * 4
            _cutoff_time_idx = cutoff_time_idx + _start_idx
            _notes = self.relative_tokens_to_notes(
                _tokens,
                start_idx=_start_idx,
                cutoff_time_idx=_cutoff_time_idx,
            )

            if len(_notes) == 0:
                pass
            elif notes is None:
                notes = _notes
            else:
                notes = np.concatenate((notes, _notes), axis=0)

        if notes is None:
            notes = []
        midi = self.notes_to_midi(notes, beatstep, offset_sec=beatstep[beat_offset_idx])
        return midi, notes

    def relative_tokens_to_notes(self, tokens, start_idx, cutoff_time_idx=None):
        # TODO remove legacy
        # decoding If the first token is an arranger
        if tokens[0] >= (self.vocab_size_special + self.vocab_size_note + self.vocab_size_velocity + self.vocab_size_time):
            tokens = tokens[1:]

        words = [self.detokenize(token, time_idx_offset=0) for token in tokens]

        if hasattr(start_idx, "item"):
            """ if numpy or torch tensor """
            start_idx = start_idx.item()

        current_idx = start_idx
        current_velocity = 0
        note_onsets_ready = [None for i in range(self.vocab_size_note + 1)]
        notes = []
        for type, number in words:
            if type == TOKEN_SPECIAL:
                if number == EOS:
                    break
            elif type == TOKEN_TIME:
                current_idx += number
                if cutoff_time_idx is not None:
                    current_idx = min(current_idx, cutoff_time_idx)

            elif type == TOKEN_VELOCITY:
                current_velocity = number
            elif type == TOKEN_NOTE:
                pitch = number
                if current_velocity == 0:
                    # note_offset
                    if note_onsets_ready[pitch] is None:
                        # offset without onset
                        pass
                    else:
                        onset_idx = note_onsets_ready[pitch]
                        if onset_idx >= current_idx:
                            # No time shift after previous note_on
                            pass
                        else:
                            offset_idx = current_idx
                            notes.append(
                                [onset_idx, offset_idx, pitch, DEFAULT_VELOCITY]
                            )
                            note_onsets_ready[pitch] = None
                else:
                    # note_on
                    if note_onsets_ready[pitch] is None:
                        note_onsets_ready[pitch] = current_idx
                    else:
                        # note-on already exists
                        onset_idx = note_onsets_ready[pitch]
                        if onset_idx >= current_idx:
                            # No time shift after previous note_on
                            pass
                        else:
                            offset_idx = current_idx
                            notes.append(
                                [onset_idx, offset_idx, pitch, DEFAULT_VELOCITY]
                            )
                            note_onsets_ready[pitch] = current_idx
            else:
                raise ValueError

        for pitch, note_on in enumerate(note_onsets_ready):
            # force offset if no offset for each pitch
            if note_on is not None:
                if cutoff_time_idx is None:
                    cutoff = note_on + 1
                else:
                    cutoff = max(cutoff_time_idx, note_on + 1)

                offset_idx = max(current_idx, cutoff)
                notes.append([note_on, offset_idx, pitch, DEFAULT_VELOCITY])

        if len(notes) == 0:
            return []
        else:
            notes = np.array(notes)
            note_order = notes[:, 0] * 128 + notes[:, 1]
            notes = notes[note_order.argsort()]
            return notes

    def notes_to_midi(self, notes, beatstep, offset_sec=None):
        new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
        new_inst = pretty_midi.Instrument(program=0)
        new_notes = []
        if offset_sec is None:
            offset_sec = 0.0

        for onset_idx, offset_idx, pitch, velocity in notes:
            new_note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=beatstep[onset_idx] - offset_sec,
                end=beatstep[offset_idx] - offset_sec,
            )
            new_notes.append(new_note)
        new_inst.notes = new_notes
        new_pm.instruments.append(new_inst)
        new_pm.remove_invalid_notes()
        return new_pm

    def get_stereo(self, pop_y, midi_y, pop_scale=0.99):
        """ Generates stereo audio using `"pop audio(`pop_y`)"` and `"generated midi audio(`midi_y`)"` """
        if len(pop_y) > len(midi_y):
            midi_y = np.pad(midi_y, (0, len(pop_y) - len(midi_y)))
        elif len(pop_y) < len(midi_y):
            pop_y = np.pad(pop_y, (0, -len(pop_y) + len(midi_y)))
        stereo = np.stack((midi_y, pop_y * pop_scale))
        return stereo

    def postprocess(self,
                    relative_tokens,
                    beatsteps,
                    ext_beatstep,
                    raw_audio,
                    sr,
                    mix_sample_rate=None,
                    save_path=None,
                    output_name=None,
                    save_midi = False,
                    save_mix = False,
                    click_amp = 0.2,
                    stereo_amp=0.5,
                    add_click = False,
                    show_plot=True,
        ):
        """ Postprocess step. It also saves the `"generated midi audio"`, `"stereo-mix"` """
        if (save_midi is not False or save_mix is not False) and save_path is None:
            raise ValueError("If you want to save any mix or midi file then you must define save_path.")

        if save_path is not None and (save_midi is False and save_mix is False):
            raise ValueError("You are setting save_path but not saving anything, use save_midi=True to "
                             "save the midi file and use save_mix to save the mix file or do both!")

        if output_name is None:
            import time
            output_name = int(time.time())
            warnings.warn("output_name is set to None, so using the name output_name")

        mix_sample_rate = self.sample_rate if mix_sample_rate is None else mix_sample_rate

        if save_path is not None:
            if os.path.isdir(save_path):
                midi_path = os.path.join(save_path, f"midi_output_{output_name}.mid")
                mix_path =  os.path.join(save_path, f"mix_output_{output_name}.wav")
            else:
                raise ValueError(f"Is {save_path} a directory?")
        pm, notes = self.relative_batch_tokens_to_midi(relative_tokens,
                                                       beatstep=ext_beatstep,
                                                       bars_per_batch=self.n_bars,
                                                       cutoff_time_idx=(self.n_bars + 1) * 4,
                                                       )
        for n in pm.instruments[0].notes:
            n.start += beatsteps[0]
            n.end += beatsteps[0]

        if save_midi:
            pm.write(midi_path)
            print(f"midi file saved at {midi_path}!")

        if show_plot or save_mix:
            if mix_sample_rate != sr:
                raw_audio = librosa.core.resample(raw_audio, orig_sr=sr, target_sr=mix_sample_rate)
                sr = mix_sample_rate
            if add_click:
                clicks = (
                        librosa.clicks(times=beatsteps, sr=sr, length=len(raw_audio)) * click_amp
                )
                raw_audio = raw_audio + clicks
            pm_raw_audio = pm.fluidsynth(sr)
            stereo = self.get_stereo(raw_audio, pm_raw_audio, pop_scale=stereo_amp)

            sf.write(
                file=mix_path,
                data=stereo.T,
                samplerate=sr,
                format="wav",
            )
            print(f"stereo-mix file saved at {mix_path}!")

        if show_plot:
            display("Stereo MIX", ipd.Audio(stereo, rate=sr))
            display("Rendered MIDI", ipd.Audio(pm_raw_audio, rate=sr))
            display("Original Song", ipd.Audio(raw_audio, rate=sr))
            display(note_seq.plot_sequence(note_seq.midi_to_note_sequence(pm)))

        return pm
