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

from transformers.audio_utils import amplitude_to_db, hertz_to_mel, mel_filter_bank, mel_to_hertz, power_to_db


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

    def test_mel_filter_bank_shape(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=13,
            min_frequency=100,
            max_frequency=4000,
            sampling_rate=16000,
            norm=None,
            mel_scale="htk",
        )
        self.assertEqual(mel_filters.shape, (513, 13))

        mel_filters = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=13,
            min_frequency=100,
            max_frequency=4000,
            sampling_rate=16000,
            norm="slaney",
            mel_scale="slaney",
        )
        self.assertEqual(mel_filters.shape, (513, 13))

    def test_mel_filter_bank_htk(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=16,
            num_mel_filters=4,
            min_frequency=0,
            max_frequency=2000,
            sampling_rate=4000,
            norm=None,
            mel_scale="htk",
        )
        # fmt: off
        expected = np.array([
            [0.0       , 0.0       , 0.0       , 0.0       ],
            [0.61454786, 0.0       , 0.0       , 0.0       ],
            [0.82511046, 0.17488954, 0.0       , 0.0       ],
            [0.35597035, 0.64402965, 0.0       , 0.0       ],
            [0.0       , 0.91360726, 0.08639274, 0.0       ],
            [0.0       , 0.55547007, 0.44452993, 0.0       ],
            [0.0       , 0.19733289, 0.80266711, 0.0       ],
            [0.0       , 0.0       , 0.87724349, 0.12275651],
            [0.0       , 0.0       , 0.6038449 , 0.3961551 ],
            [0.0       , 0.0       , 0.33044631, 0.66955369],
            [0.0       , 0.0       , 0.05704771, 0.94295229],
            [0.0       , 0.0       , 0.0       , 0.83483975],
            [0.0       , 0.0       , 0.0       , 0.62612982],
            [0.0       , 0.0       , 0.0       , 0.41741988],
            [0.0       , 0.0       , 0.0       , 0.20870994],
            [0.0       , 0.0       , 0.0       , 0.0       ]
        ])
        # fmt: on
        self.assertTrue(np.allclose(mel_filters, expected))

    def test_mel_filter_bank_slaney(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=16,
            num_mel_filters=4,
            min_frequency=0,
            max_frequency=2000,
            sampling_rate=4000,
            norm=None,
            mel_scale="slaney",
        )
        # fmt: off
        expected = np.array([
            [0.0       , 0.0       , 0.0       , 0.0       ],
            [0.39869419, 0.0       , 0.0       , 0.0       ],
            [0.79738839, 0.0       , 0.0       , 0.0       ],
            [0.80391742, 0.19608258, 0.0       , 0.0       ],
            [0.40522322, 0.59477678, 0.0       , 0.0       ],
            [0.00652903, 0.99347097, 0.0       , 0.0       ],
            [0.0       , 0.60796161, 0.39203839, 0.0       ],
            [0.0       , 0.20939631, 0.79060369, 0.0       ],
            [0.0       , 0.0       , 0.84685344, 0.15314656],
            [0.0       , 0.0       , 0.52418477, 0.47581523],
            [0.0       , 0.0       , 0.2015161 , 0.7984839 ],
            [0.0       , 0.0       , 0.0       , 0.9141874 ],
            [0.0       , 0.0       , 0.0       , 0.68564055],
            [0.0       , 0.0       , 0.0       , 0.4570937 ],
            [0.0       , 0.0       , 0.0       , 0.22854685],
            [0.0       , 0.0       , 0.0       , 0.0       ]
        ])
        # fmt: on
        self.assertTrue(np.allclose(mel_filters, expected))

    def test_mel_filter_bank_slaney_norm(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=16,
            num_mel_filters=4,
            min_frequency=0,
            max_frequency=2000,
            sampling_rate=4000,
            norm="slaney",
            mel_scale="slaney",
        )
        # fmt: off
        expected = np.array([
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [1.19217795e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.38435591e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [2.40387905e-03, 5.86232616e-04, 0.00000000e+00, 0.00000000e+00],
            [1.21170110e-03, 1.77821783e-03, 0.00000000e+00, 0.00000000e+00],
            [1.95231437e-05, 2.97020305e-03, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 1.81763684e-03, 1.04857612e-03, 0.00000000e+00],
            [0.00000000e+00, 6.26036972e-04, 2.11460963e-03, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 2.26505954e-03, 3.07332945e-04],
            [0.00000000e+00, 0.00000000e+00, 1.40202503e-03, 9.54861093e-04],
            [0.00000000e+00, 0.00000000e+00, 5.38990521e-04, 1.60238924e-03],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.83458185e-03],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.37593638e-03],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 9.17290923e-04],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 4.58645462e-04],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
        ])
        # fmt: on
        self.assertTrue(np.allclose(mel_filters, expected))

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
