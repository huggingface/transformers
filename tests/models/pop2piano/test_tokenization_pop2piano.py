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

from ...generation.test_utils import GreedySearchEncoderDecoderOutput


if is_torch_available():
    import torch

    from transformers.models.pop2piano.modeling_pop2piano import Pop2PianoForConditionalGeneration

requirements = is_torch_available() and is_pretty_midi_available()
if requirements:
    import pretty_midi

    from transformers import Pop2PianoFeatureExtractor, Pop2PianoTokenizer


@require_torch
@require_pretty_midi
class Pop2PianoTokenizerTest(unittest.TestCase):
    def test_call(self):
        tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")
        model_output = GreedySearchEncoderDecoderOutput(sequences=torch.ones([120, 96]))
        input_features = BatchFeature(
            {"beatsteps": torch.ones([1, 955]), "extrapolated_beatstep": torch.ones([1, 1000])}
        )

        output = tokenizer(token_ids=model_output.sequences, feature_extractor_output=input_features)[
            "pretty_midi_objects"
        ][0]
        self.assertTrue(isinstance(output, pretty_midi.pretty_midi.PrettyMIDI))

    def test_call_batched(self):
        tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")
        model_output = GreedySearchEncoderDecoderOutput(
            sequences=torch.concatenate(
                [
                    torch.randint(size=[120, 96], low=0, high=70, dtype=torch.long),
                    torch.zeros(size=[1, 96], dtype=torch.long),
                    torch.randint(size=[50, 96], low=0, high=40, dtype=torch.long),
                    torch.zeros(size=[1, 96], dtype=torch.long),
                ],
                axis=0,
            )
        )
        input_features = BatchFeature(
            {
                "beatsteps": torch.ones([2, 955]),
                "extrapolated_beatstep": torch.ones([2, 1000]),
                "attention_mask": torch.concatenate(
                    [
                        torch.ones([120, 96], dtype=torch.long),
                        torch.zeros([1, 96], dtype=torch.long),
                        torch.ones([50, 96], dtype=torch.long),
                        torch.zeros([1, 96], dtype=torch.long),
                    ],
                    axis=0,
                ),
                "attention_mask_beatsteps": torch.ones([2, 955]),
                "attention_mask_extrapolated_beatstep": torch.ones([2, 1000]),
            }
        )

        output = tokenizer(token_ids=model_output.sequences, feature_extractor_output=input_features)[
            "pretty_midi_objects"
        ]
        # check length
        self.assertTrue(len(output) == 2)
        # check object type
        self.assertTrue(isinstance(output[0], pretty_midi.pretty_midi.PrettyMIDI))
        self.assertTrue(isinstance(output[1], pretty_midi.pretty_midi.PrettyMIDI))

    # This is the test for a real music from K-Pop genre.
    @slow
    def test_real_music(self):
        model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")
        model.eval()
        feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
        tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")
        ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

        output_fe = feature_extractor(ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"])
        output_model = model.generate(input_features=output_fe["input_features"], composer="composer1")
        output_tokenizer = tokenizer(
            token_ids=output_model,
            feature_extractor_output=output_fe,
        )[
            "pretty_midi_objects"
        ][0]

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
