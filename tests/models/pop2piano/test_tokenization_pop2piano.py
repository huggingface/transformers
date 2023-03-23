# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import os
import tempfile
import unittest

from datasets import load_dataset

from transformers.testing_utils import (
    is_essentia_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_soundfile_availble,
    is_torch_available,
    is_torchaudio_available,
    require_essentia,
    require_librosa,
    require_pretty_midi,
    require_scipy,
    require_soundfile,
    require_torch,
    slow,
)


if is_torch_available():
    import torch

    from transformers import Pop2PianoForConditionalGeneration

requirements = (
    is_torch_available()
    and is_torchaudio_available()
    and is_essentia_available()
    and is_scipy_available()
    and is_librosa_available()
    and is_soundfile_availble()
    and is_pretty_midi_available()
)
if requirements:
    import pretty_midi

    from transformers import Pop2PianoFeatureExtractor, Pop2PianoTokenizer


@require_torch
@require_librosa
@require_soundfile
@require_pretty_midi
class Pop2PianoTokenizerTest(unittest.TestCase):
    def test_call(self):
        tokenizer = Pop2PianoTokenizer.from_pretrained("susnato/pop2piano_dev")
        input = {
            "relative_tokens": torch.ones([120, 96]),
            "beatsteps": torch.ones(
                [
                    955,
                ]
            ),
            "ext_beatstep": torch.ones(
                [
                    958,
                ]
            ),
            "raw_audio": torch.zeros(
                [
                    141301,
                ]
            ),
            "sampling_rate": 44100,
            "save_midi": False,
            "save_mix": False,
        }

        output = tokenizer(**input)
        self.assertTrue(isinstance(output, pretty_midi.pretty_midi.PrettyMIDI))

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples], [x["sampling_rate"] for x in speech_samples]

    @slow
    @require_scipy
    @require_essentia
    @require_librosa
    def test_midi_saving(self):
        tokenizer = Pop2PianoTokenizer.from_pretrained("susnato/pop2piano_dev")
        feaure_extractor = Pop2PianoFeatureExtractor.from_pretrained("susnato/pop2piano_dev")
        model = Pop2PianoForConditionalGeneration.from_pretrained("susnato/pop2piano_dev")

        input_speech, sampling_rate = self._load_datasamples(1)
        fe_outputs = feaure_extractor(input_speech, audio_sr=sampling_rate[0], return_tensors="pt")
        model_outputs = model.generate(fe_outputs)
        filename = "tmp-file"

        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer(
                relative_tokens=model_outputs,
                beatsteps=fe_outputs["beatsteps"],
                ext_beatstep=fe_outputs["ext_beatstep"],
                raw_audio=input_speech,
                sampling_rate=sampling_rate,
                mix_sampling_rate=sampling_rate,
                save_path=tmpdirname,
                audio_file_name=filename,
                save_midi=True,
            )

            # check if files are saved there or not
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, f"midi_output_{filename}.mid")))
