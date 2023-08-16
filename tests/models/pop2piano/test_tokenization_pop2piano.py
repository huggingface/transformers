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
"""
Please note that Pop2PianoTokenizer is too far from our usual tokenizers and thus cannot use the TokenizerTesterMixin class.
"""

import unittest

from transformers.feature_extraction_utils import BatchFeature
from transformers.testing_utils import (
    is_pretty_midi_available,
    is_torch_available,
    require_pretty_midi,
    require_torch,
)
from transformers.tokenization_utils import BatchEncoding


if is_torch_available():
    import torch


requirements_available = is_torch_available() and is_pretty_midi_available()
if requirements_available:
    import pretty_midi

    from transformers import Pop2PianoTokenizer


## TODO : changing checkpoints from `susnato/pop2piano_dev` to `sweetcocoa/pop2piano` after the PR is approved


@require_torch
@require_pretty_midi
class Pop2PianoTokenizerTest(unittest.TestCase):
    def test_call(self):
        tokenizer = Pop2PianoTokenizer.from_pretrained("susnato/pop2piano_dev")
        notes = [
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

        output = tokenizer(
            notes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=10,
            return_attention_mask=True,
        )
        self.assertTrue(isinstance(output, BatchEncoding))

    def test_batch_decode(self):
        tokenizer = Pop2PianoTokenizer.from_pretrained("susnato/pop2piano_dev")
        model_output = torch.concatenate(
            [
                torch.randint(size=[120, 96], low=0, high=70, dtype=torch.long),
                torch.zeros(size=[1, 96], dtype=torch.long),
                torch.randint(size=[50, 96], low=0, high=40, dtype=torch.long),
                torch.zeros(size=[1, 96], dtype=torch.long),
            ],
            axis=0,
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

        output = tokenizer.batch_decode(token_ids=model_output, feature_extractor_output=input_features)[
            "pretty_midi_objects"
        ]
        # check length
        self.assertTrue(len(output) == 2)
        # check object type
        self.assertTrue(isinstance(output[0], pretty_midi.pretty_midi.PrettyMIDI))
        self.assertTrue(isinstance(output[1], pretty_midi.pretty_midi.PrettyMIDI))
