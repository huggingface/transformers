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

import shutil
import tempfile
import unittest

import numpy as np
import pytest
from datasets import load_dataset

from transformers.testing_utils import (
    require_essentia,
    require_librosa,
    require_pretty_midi,
    require_scipy,
    require_torch,
)
from transformers.tokenization_utils import BatchEncoding
from transformers.utils.import_utils import (
    is_essentia_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_torch_available,
)


requirements_available = (
    is_torch_available()
    and is_essentia_available()
    and is_scipy_available()
    and is_librosa_available()
    and is_pretty_midi_available()
)

if requirements_available:
    import pretty_midi

    from transformers import (
        Pop2PianoFeatureExtractor,
        Pop2PianoForConditionalGeneration,
        Pop2PianoProcessor,
        Pop2PianoTokenizer,
    )


@require_scipy
@require_torch
@require_librosa
@require_essentia
@require_pretty_midi
class Pop2PianoProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
        tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")
        processor = Pop2PianoProcessor(feature_extractor, tokenizer)

        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return Pop2PianoTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return Pop2PianoFeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_additional_features(self):
        processor = Pop2PianoProcessor(
            tokenizer=self.get_tokenizer(),
            feature_extractor=self.get_feature_extractor(),
        )
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(
            unk_token="-1",
            eos_token="1",
            pad_token="0",
            bos_token="2",
        )
        feature_extractor_add_kwargs = self.get_feature_extractor()

        processor = Pop2PianoProcessor.from_pretrained(
            self.tmpdirname,
            unk_token="-1",
            eos_token="1",
            pad_token="0",
            bos_token="2",
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, Pop2PianoTokenizer)

        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.feature_extractor, Pop2PianoFeatureExtractor)

    def get_inputs(self):
        """get inputs for both feature extractor and tokenizer"""
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        speech_samples = ds.sort("id").select([0])["audio"]
        input_speech = [x["array"] for x in speech_samples][0]
        sampling_rate = [x["sampling_rate"] for x in speech_samples][0]

        feature_extractor_outputs = self.get_feature_extractor()(
            audio=input_speech, sampling_rate=sampling_rate, return_tensors="pt"
        )
        model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
        token_ids = model.generate(input_features=feature_extractor_outputs["input_features"], composer="composer1")
        dummy_notes = [
            [
                pretty_midi.Note(start=0.441179, end=2.159456, pitch=70, velocity=77),
                pretty_midi.Note(start=0.673379, end=0.905578, pitch=73, velocity=77),
                pretty_midi.Note(start=0.905578, end=2.159456, pitch=73, velocity=77),
                pretty_midi.Note(start=1.114558, end=2.159456, pitch=78, velocity=77),
                pretty_midi.Note(start=1.323537, end=1.532517, pitch=80, velocity=77),
            ],
            [
                pretty_midi.Note(start=0.441179, end=2.159456, pitch=70, velocity=77),
            ],
        ]

        return input_speech, sampling_rate, token_ids, dummy_notes

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Pop2PianoProcessor(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        input_speech, sampling_rate, _, _ = self.get_inputs()

        feature_extractor_outputs = feature_extractor(
            audio=input_speech, sampling_rate=sampling_rate, return_tensors="np"
        )
        processor_outputs = processor(audio=input_speech, sampling_rate=sampling_rate, return_tensors="np")

        for key in feature_extractor_outputs.keys():
            self.assertTrue(np.allclose(feature_extractor_outputs[key], processor_outputs[key], atol=1e-4))

    def test_processor_batch_decode(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Pop2PianoProcessor(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        audio, sampling_rate, token_ids, _ = self.get_inputs()
        feature_extractor_output = feature_extractor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")

        encoded_processor = processor.batch_decode(
            token_ids=token_ids,
            feature_extractor_output=feature_extractor_output,
            return_midi=True,
        )

        encoded_tokenizer = tokenizer.batch_decode(
            token_ids=token_ids,
            feature_extractor_output=feature_extractor_output,
            return_midi=True,
        )
        # check start timings
        encoded_processor_start_timings = [token.start for token in encoded_processor["notes"]]
        encoded_tokenizer_start_timings = [token.start for token in encoded_tokenizer["notes"]]
        self.assertListEqual(encoded_processor_start_timings, encoded_tokenizer_start_timings)

        # check end timings
        encoded_processor_end_timings = [token.end for token in encoded_processor["notes"]]
        encoded_tokenizer_end_timings = [token.end for token in encoded_tokenizer["notes"]]
        self.assertListEqual(encoded_processor_end_timings, encoded_tokenizer_end_timings)

        # check pitch
        encoded_processor_pitch = [token.pitch for token in encoded_processor["notes"]]
        encoded_tokenizer_pitch = [token.pitch for token in encoded_tokenizer["notes"]]
        self.assertListEqual(encoded_processor_pitch, encoded_tokenizer_pitch)

        # check velocity
        encoded_processor_velocity = [token.velocity for token in encoded_processor["notes"]]
        encoded_tokenizer_velocity = [token.velocity for token in encoded_tokenizer["notes"]]
        self.assertListEqual(encoded_processor_velocity, encoded_tokenizer_velocity)

    def test_tokenizer_call(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Pop2PianoProcessor(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        _, _, _, notes = self.get_inputs()

        encoded_processor = processor(
            notes=notes,
        )

        self.assertTrue(isinstance(encoded_processor, BatchEncoding))

    def test_processor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Pop2PianoProcessor(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        audio, sampling_rate, _, notes = self.get_inputs()

        inputs = processor(
            audio=audio,
            sampling_rate=sampling_rate,
            notes=notes,
        )

        self.assertListEqual(
            list(inputs.keys()),
            ["input_features", "beatsteps", "extrapolated_beatstep", "token_ids"],
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_model_input_names(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()

        processor = Pop2PianoProcessor(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
        )

        audio, sampling_rate, _, notes = self.get_inputs()
        feature_extractor(audio, sampling_rate, return_tensors="pt")

        inputs = processor(
            audio=audio,
            sampling_rate=sampling_rate,
            notes=notes,
        )
        self.assertListEqual(
            list(inputs.keys()),
            ["input_features", "beatsteps", "extrapolated_beatstep", "token_ids"],
        )
