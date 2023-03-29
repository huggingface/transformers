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

from transformers.audio_utils import amplitude_to_db, hertz_to_mel, mel_to_hertz, power_to_db


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

    def test_mel_filter_bank(self):
        pass

    def test_power_to_db(self):
        spectrogram = np.zeros((2, 3))
        spectrogram[0, 0] = 2.0
        spectrogram[0, 1] = 0.5
        spectrogram[0, 2] = 0.707
        spectrogram[1, 1] = 1.0

        output = power_to_db(spectrogram, reference=1.0)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-100.0, 0.0, -100.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, reference=2.0)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-103.01029996, -3.01029996, -103.01029996]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, min_value=1e-6)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-60.0, 0.0, -60.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, db_range=80)
        expected = np.array([[3.01029996, -3.01029996, -1.50580586], [-76.98970004, 0.0, -76.98970004]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, reference=2.0, db_range=80)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-80.0, -3.01029996, -80.0]])
        self.assertTrue(np.allclose(output, expected))

        output = power_to_db(spectrogram, reference=2.0, min_value=1e-6, db_range=80)
        expected = np.array([[0.0, -6.02059991, -4.51610582], [-63.01029996, -3.01029996, -63.01029996]])
        self.assertTrue(np.allclose(output, expected))

        with pytest.raises(ValueError):
            power_to_db(spectrogram, reference=0.0)
        with pytest.raises(ValueError):
            power_to_db(spectrogram, min_value=0.0)
        with pytest.raises(ValueError):
            power_to_db(spectrogram, db_range=-80)

    def test_amplitude_to_db(self):
        spectrogram = np.zeros((2, 3))
        spectrogram[0, 0] = 2.0
        spectrogram[0, 1] = 0.5
        spectrogram[0, 2] = 0.707
        spectrogram[1, 1] = 1.0

        output = amplitude_to_db(spectrogram, reference=1.0)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-100.0, 0.0, -100.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, reference=2.0)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-106.02059991, -6.02059991, -106.02059991]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, min_value=1e-3)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-60.0, 0.0, -60.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, db_range=80)
        expected = np.array([[6.02059991, -6.02059991, -3.01161172], [-73.97940009, 0.0, -73.97940009]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, reference=2.0, db_range=80)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-80.0, -6.02059991, -80.0]])
        self.assertTrue(np.allclose(output, expected))

        output = amplitude_to_db(spectrogram, reference=2.0, min_value=1e-3, db_range=80)
        expected = np.array([[0.0, -12.04119983, -9.03221164], [-66.02059991, -6.02059991, -66.02059991]])
        self.assertTrue(np.allclose(output, expected))

        with pytest.raises(ValueError):
            amplitude_to_db(spectrogram, reference=0.0)
        with pytest.raises(ValueError):
            amplitude_to_db(spectrogram, min_value=0.0)
        with pytest.raises(ValueError):
            amplitude_to_db(spectrogram, db_range=-80)
