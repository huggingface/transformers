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

import os
import warnings
from typing import List, Optional, Union

import essentia
import essentia.standard
import librosa
import numpy as np
import pretty_midi
import scipy
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)

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

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
    This class loads audio, extracts rhythm and does preprocesses before being passed through `LogMelSpectrogram`. This:
    class also contains postprocessing methods to convert model outputs to midi audio and stereo-mix.
        n_bars (`int`, *optional*, defaults to 2):
            Determines `n_steps` in method `preprocess_mel`.
        sampling_rate (`int`, *optional*, defaults to 22050):
            Sample rate of audio signal.
        use_mel (`bool`, *optional*, defaults to `True`):
            Whether to preprocess for `LogMelSpectrogram` or not. For the current implementation this must be `True`.
        padding_value (`int`, *optional*, defaults to 0):
            Padding value used to pad the audio. Should correspond to silences.
        vocab_size_special (`int`, *optional*, defaults to 4):
            Number of special values.
        vocab_size_note (`int`, *optional*, defaults to 128):
            This represents the number of Note Values. Note values indicate a pitch event for one of the MIDI pitches.
            But only the 88 pitches corresponding to piano keys are actually used.
        vocab_size_velocity (`int`, *optional*, defaults to 2):
            Number of Velocity tokens.
        vocab_size_time (`int`, *optional*, defaults to 100):
            This represents the number of Beat Shifts. Beat Shift [100 values] Indicates the relative time shift within
            the segment quantized into 8th-note beats(half-beats).
    """
    model_input_names = ["input_features"]

    def __init__(
        self,
        n_bars: int = 2,
        sampling_rate: int = 22050,
        use_mel: int = True,
        padding_value: int = 0,
        vocab_size_special: int = 4,
        vocab_size_note: int = 128,
        vocab_size_velocity: int = 2,
        vocab_size_time: int = 100,
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
        self.vocab_size_special = vocab_size_special
        self.vocab_size_note = vocab_size_note
        self.vocab_size_velocity = vocab_size_velocity
        self.vocab_size_time = vocab_size_time

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
        warnings.warn(
            "Pop2PianoFeatureExtractor only takes one raw_audio at a time, if you want to extract features from more than a single audio then you might need to call it multiple times."
        )

        # If it's [np.ndarray]
        if isinstance(raw_audio, list) and isinstance(raw_audio[0], np.ndarray):
            raw_audio = raw_audio[0]

        bpm, beat_times, confidence, estimates, essentia_beat_intervals = self.extract_rhythm(raw_audio=raw_audio)
        beatsteps = self.interpolate_beat_times(beat_times, steps_per_beat, extend=True)

        if self.sampling_rate != audio_sr and self.sampling_rate is not None:
            # Change `raw_audio_sr` to `self.sampling_rate`
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

    def decode(self, token, time_idx_offset):
        """Decodes the tokens generated by the transformer"""

        if token >= (self.vocab_size_special + self.vocab_size_note + self.vocab_size_velocity):
            type, value = TOKEN_TIME, (
                (token - (self.vocab_size_special + self.vocab_size_note + self.vocab_size_velocity)) + time_idx_offset
            )
        elif token >= (self.vocab_size_special + self.vocab_size_note):
            type, value = TOKEN_VELOCITY, (token - (self.vocab_size_special + self.vocab_size_note))
            value = int(value)
        elif token >= self.vocab_size_special:
            type, value = TOKEN_NOTE, (token - self.vocab_size_special)
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
        """Converts tokens to midi"""

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
        # decoding If the first token is an arranger
        if tokens[0] >= (
            self.vocab_size_special + self.vocab_size_note + self.vocab_size_velocity + self.vocab_size_time
        ):
            tokens = tokens[1:]

        words = [self.decode(token, time_idx_offset=0) for token in tokens]

        if hasattr(start_idx, "item"):
            """if numpy or torch tensor"""
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
                            notes.append([onset_idx, offset_idx, pitch, DEFAULT_VELOCITY])
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
                            notes.append([onset_idx, offset_idx, pitch, DEFAULT_VELOCITY])
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
        """Converts notes to midi"""

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
        """Generates stereo audio using `pop audio(`pop_y`)` and `generated midi audio(`midi_y`)`"""

        if len(pop_y) > len(midi_y):
            midi_y = np.pad(midi_y, (0, len(pop_y) - len(midi_y)))
        elif len(pop_y) < len(midi_y):
            pop_y = np.pad(pop_y, (0, -len(pop_y) + len(midi_y)))
        stereo = np.stack((midi_y, pop_y * pop_scale))
        return stereo

    def _to_np(self, tensor):
        """Converts pytorch tensor to np.ndarray."""
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        else:
            raise ValueError("dtype not understood! Please use wither torch.Tensor or np.ndarray")

    def postprocess(
        self,
        relative_tokens: Union[np.ndarray, torch.Tensor],
        beatsteps: Union[np.ndarray, torch.Tensor],
        ext_beatstep: Union[np.ndarray, torch.Tensor],
        raw_audio: Union[np.ndarray, List[float], List[np.ndarray]],
        sampling_rate: int,
        mix_sampling_rate=None,
        save_path: str = None,
        audio_file_name: str = None,
        save_midi: bool = False,
        save_mix: bool = False,
        click_amp: float = 0.2,
        stereo_amp: float = 0.5,
        add_click: bool = False,
    ):
        r"""
        Args:
        Postprocess step. It also saves the `"generated midi audio"`, `"stereo-mix"`
            relative_tokens ([`~utils.TensorType`]):
                Output of `Pop2PianoConditionalGeneration` model.
            beatsteps ([`~utils.TensorType`]):
                beatsteps returned by `Pop2PianoFeatureExtractor.__call__`
            ext_beatstep ([`~utils.TensorType`]):
                ext_beatstep returned by `Pop2PianoFeatureExtractor.__call__`
            raw_audio (`np.ndarray`, `List`):
                Denotes the raw_audio.
            sampling_rate (`int`):
                Denotes the Sampling Rate of `raw_audio`.
            mix_sampling_rate (`int`, *optional*):
                Denotes the Sampling Rate for `stereo-mix`.
            audio_file_name (`str`, *optional*):
                Name of the file to be saved.
            save_path (`str`, *optional*):
                Path where the `stereo-mix` and `midi-audio` is to be saved.
            save_midi (`bool`, *optional*):
                Whether to save `midi-audio` or not.
            save_mix (`bool`, *optional*):
                Whether to save `stereo-mix` or not.
            add_click (`bool`, *optional*, defaults to `False`):
                Constructs a `"click track"`.
            click_amp (`float`, *optional*, defaults to 0.2):
                Amplitude for `"click track"`.

        Returns:
            `pretty_midi.pretty_midi.PrettyMIDI` : returns pretty_midi object.
        """

        relative_tokens = self._to_np(relative_tokens)
        beatsteps = self._to_np(beatsteps)
        ext_beatstep = self._to_np(ext_beatstep)

        if (save_midi or save_mix) and save_path is None:
            raise ValueError("If you want to save any mix or midi file then you must define save_path.")

        if save_path and (not save_midi and not save_mix):
            raise ValueError(
                "You are setting save_path but not saving anything, use save_midi=True to "
                "save the midi file and use save_mix to save the mix file or do both!"
            )

        mix_sampling_rate = sampling_rate if mix_sampling_rate is None else mix_sampling_rate

        if save_path is not None:
            if os.path.isdir(save_path):
                midi_path = os.path.join(save_path, f"midi_output_{audio_file_name}.mid")
                mix_path = os.path.join(save_path, f"mix_output_{audio_file_name}.wav")
            else:
                raise ValueError(f"Is {save_path} a directory?")

        pm, notes = self.relative_batch_tokens_to_midi(
            tokens=relative_tokens,
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

        if save_mix:
            if mix_sampling_rate != sampling_rate:
                raw_audio = librosa.core.resample(raw_audio, orig_sr=sampling_rate, target_sr=mix_sampling_rate)
                sampling_rate = mix_sampling_rate
            if add_click:
                clicks = librosa.clicks(times=beatsteps, sr=sampling_rate, length=len(raw_audio)) * click_amp
                raw_audio = raw_audio + clicks
            pm_raw_audio = pm.fluidsynth(sampling_rate)
            stereo = self.get_stereo(raw_audio, pm_raw_audio, pop_scale=stereo_amp)

            sf.write(
                file=mix_path,
                data=stereo.T,
                samplerate=sampling_rate,
                format="wav",
            )
            print(f"stereo-mix file saved at {mix_path}!")

        return pm
