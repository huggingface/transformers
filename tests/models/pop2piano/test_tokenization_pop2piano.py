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

import unittest

import numpy as np
from datasets import load_dataset

from transformers.feature_extraction_utils import BatchFeature
from transformers.testing_utils import (
    is_pretty_midi_available,
    is_torch_available,
    require_pretty_midi,
    require_torch,
    slow,
)


if is_torch_available():
    import torch

    from transformers.models.pop2piano.modeling_pop2piano import (
        Pop2PianoForConditionalGeneration,
        Pop2PianoGreedySearchEncoderDecoderOutput,
    )

requirements = is_torch_available() and is_pretty_midi_available()
if requirements:
    import pretty_midi

    from transformers import Pop2PianoFeatureExtractor, Pop2PianoTokenizer


@require_torch
@require_pretty_midi
class Pop2PianoTokenizerTest(unittest.TestCase):
    def test_call(self):
        tokenizer = Pop2PianoTokenizer.from_pretrained("susnato/pop2piano_dev")
        model_output = Pop2PianoGreedySearchEncoderDecoderOutput(sequences=[torch.ones([120, 96])])
        input_features = BatchFeature({"beatsteps": torch.ones([1, 955]), "ext_beatstep": torch.ones([1, 1000])})

        output = tokenizer(relative_tokens=model_output.sequences, input_features=input_features)
        self.assertTrue(isinstance(output[0], pretty_midi.pretty_midi.PrettyMIDI))

    def test_call_batched(self):
        tokenizer = Pop2PianoTokenizer.from_pretrained("susnato/pop2piano_dev")
        model_output = Pop2PianoGreedySearchEncoderDecoderOutput(
            sequences=[torch.ones([120, 96]), torch.zeros([100, 50])]
        )
        input_features = BatchFeature(
            {
                "beatsteps": torch.ones([2, 955]),
                "ext_beatstep": torch.ones([2, 1000]),
                "attention_mask_beatsteps": torch.ones([2, 955]),
                "attention_mask_ext_beatstep": torch.ones([2, 1000]),
            }
        )

        output = tokenizer(relative_tokens=model_output.sequences, input_features=input_features)
        # check length
        self.assertTrue(len(output) == 2)
        # check object type
        self.assertTrue(isinstance(output[0], pretty_midi.pretty_midi.PrettyMIDI))
        self.assertTrue(isinstance(output[1], pretty_midi.pretty_midi.PrettyMIDI))

    # This is the test for a real music from K-Pop genre.
    @slow
    def test_real_music(self):
        model = Pop2PianoForConditionalGeneration.from_pretrained("susnato/pop2piano_dev").to("cuda")
        model.eval()
        feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("susnato/pop2piano_dev")
        tokenizer = Pop2PianoTokenizer.from_pretrained("susnato/pop2piano_dev")
        ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

        output_fe = feature_extractor(ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"]).to(
            "cuda"
        )
        output_model = model.generate(output_fe, composer="composer1")
        output_tokenizer = tokenizer(
            relative_tokens=output_model.sequences,
            input_features=output_fe,
        )[0]

        # Checking if no of notes are same
        self.assertEqual(len(output_tokenizer.instruments[0].notes), 59)

        predicted_timings = []
        for i in output_tokenizer.instruments[0].notes:
            predicted_timings.append(i.start)

        # Checking note start timings(first 6)
        EXPECTED_START_TIMINGS = [
            0.4876190423965454,
            0.7314285635948181,
            0.9752380847930908,
            1.4396371841430664,
            1.6718367338180542,
            1.904036283493042,
        ]

        np.allclose(EXPECTED_START_TIMINGS, predicted_timings[:6])

        # Checking note end timings(last 6)
        EXPECTED_END_TIMINGS = [
            12.341403007507324,
            12.567797183990479,
            12.567797183990479,
            12.567797183990479,
            12.794191360473633,
            12.794191360473633,
        ]

        np.allclose(EXPECTED_END_TIMINGS, predicted_timings[-6:])
