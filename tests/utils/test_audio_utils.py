# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import pytest

from transformers.audio_utils import hertz_to_mel, mel_to_hertz


class AudioUtilsFunctionTester(unittest.TestCase):
    def test_hertz_to_mel(self):
        self.assertEqual(hertz_to_mel(0.0), 0.0)
        self.assertAlmostEqual(hertz_to_mel(100), 150.48910241)

        inputs = np.array([100, 200])
        expected = np.array([150.48910241, 283.22989816])
        self.assertTrue(np.allclose(hertz_to_mel(inputs), expected))

        self.assertEqual(hertz_to_mel(0.0, "slaney"), 0.0)
        self.assertEqual(hertz_to_mel(100, "slaney"), 1.5)

        inputs = np.array([60, 100, 200, 1000, 1001, 2000])
        expected = np.array([0.9, 1.5, 3.0, 15.0, 15.01453781, 25.08188016])
        self.assertTrue(np.allclose(hertz_to_mel(inputs, "slaney"), expected))

        with pytest.raises(ValueError):
            hertz_to_mel(100, mel_scale=None)

    def test_mel_to_hertz(self):
        self.assertEqual(mel_to_hertz(0.0), 0.0)
        self.assertAlmostEqual(mel_to_hertz(150.48910241), 100)

        inputs = np.array([150.48910241, 283.22989816])
        expected = np.array([100, 200])
        self.assertTrue(np.allclose(mel_to_hertz(inputs), expected))

        self.assertEqual(mel_to_hertz(0.0, "slaney"), 0.0)
        self.assertEqual(mel_to_hertz(1.5, "slaney"), 100)

        inputs = np.array([0.9, 1.5, 3.0, 15.0, 15.01453781, 25.08188016])
        expected = np.array([60, 100, 200, 1000, 1001, 2000])
        self.assertTrue(np.allclose(mel_to_hertz(inputs, "slaney"), expected))

        with pytest.raises(ValueError):
            mel_to_hertz(100, mel_scale=None)
