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

from transformers.audio_utils import (
    amplitude_to_db,
    hertz_to_mel,
    mel_filter_bank,
    mel_to_hertz,
    power_to_db,
    spectrogram,
    window_function,
)


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

        inputs = np.array([60, 100, 200, 1000, 1001, 2000])
        expected = np.array([92.6824, 150.4899, 283.2313, 999.9907, 1000.6534, 1521.3674])
        self.assertTrue(np.allclose(hertz_to_mel(inputs, "kaldi"), expected))

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

        inputs = np.array([92.6824, 150.4899, 283.2313, 999.9907, 1000.6534, 1521.3674])
        expected = np.array([60, 100, 200, 1000, 1001, 2000])
        self.assertTrue(np.allclose(mel_to_hertz(inputs, "kaldi"), expected))

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

        mel_filters = mel_filter_bank(
            num_frequency_bins=513,
            num_mel_filters=13,
            min_frequency=100,
            max_frequency=4000,
            sampling_rate=16000,
            norm="slaney",
            mel_scale="slaney",
            triangularize_in_mel_space=True,
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

    def test_mel_filter_bank_kaldi(self):
        mel_filters = mel_filter_bank(
            num_frequency_bins=16,
            num_mel_filters=4,
            min_frequency=0,
            max_frequency=2000,
            sampling_rate=4000,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )
        # fmt: off
        expected = np.array(
        [[0.0000, 0.0000, 0.0000, 0.0000],
        [0.6086, 0.0000, 0.0000, 0.0000],
        [0.8689, 0.1311, 0.0000, 0.0000],
        [0.4110, 0.5890, 0.0000, 0.0000],
        [0.0036, 0.9964, 0.0000, 0.0000],
        [0.0000, 0.6366, 0.3634, 0.0000],
        [0.0000, 0.3027, 0.6973, 0.0000],
        [0.0000, 0.0000, 0.9964, 0.0036],
        [0.0000, 0.0000, 0.7135, 0.2865],
        [0.0000, 0.0000, 0.4507, 0.5493],
        [0.0000, 0.0000, 0.2053, 0.7947],
        [0.0000, 0.0000, 0.0000, 0.9752],
        [0.0000, 0.0000, 0.0000, 0.7585],
        [0.0000, 0.0000, 0.0000, 0.5539],
        [0.0000, 0.0000, 0.0000, 0.3599],
        [0.0000, 0.0000, 0.0000, 0.1756]]
        )
        # fmt: on
        self.assertTrue(np.allclose(mel_filters, expected, atol=5e-5))

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

    def test_window_function(self):
        window = window_function(16, "hann")
        self.assertEqual(len(window), 16)

        # fmt: off
        expected = np.array([
            0.0, 0.03806023, 0.14644661, 0.30865828, 0.5, 0.69134172, 0.85355339, 0.96193977,
            1.0, 0.96193977, 0.85355339, 0.69134172, 0.5, 0.30865828, 0.14644661, 0.03806023,
        ])
        # fmt: on
        self.assertTrue(np.allclose(window, expected))

    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    def test_spectrogram_impulse(self):
        waveform = np.zeros(40)
        waveform[9] = 1.0  # impulse shifted in time

        spec = spectrogram(
            waveform,
            window_function(12, "hann", frame_length=16),
            frame_length=16,
            hop_length=4,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (9, 11))

        expected = np.array([[0.0, 0.0669873, 0.9330127, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(spec, expected))

    def test_spectrogram_integration_test(self):
        waveform = self._load_datasamples(1)[0]

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (257, 732))

        # fmt: off
        expected = np.array([
            0.02464888, 0.04648664, 0.05872392, 0.02311783, 0.0327175 ,
            0.02433643, 0.01198814, 0.02055709, 0.01559287, 0.01394357,
            0.01299037, 0.01728045, 0.0254554 , 0.02486533, 0.02011792,
            0.01755333, 0.02100457, 0.02337024, 0.01436963, 0.01464558,
            0.0211017 , 0.0193489 , 0.01272165, 0.01858462, 0.03722598,
            0.0456542 , 0.03281558, 0.00620586, 0.02226466, 0.03618042,
            0.03508182, 0.02271432, 0.01051649, 0.01225771, 0.02315293,
            0.02331886, 0.01417785, 0.0106844 , 0.01791214, 0.017177  ,
            0.02125114, 0.05028201, 0.06830665, 0.05216664, 0.01963666,
            0.06941418, 0.11513043, 0.12257859, 0.10948435, 0.08568069,
            0.05509328, 0.05047818, 0.047112  , 0.05060737, 0.02982424,
            0.02803827, 0.02933729, 0.01760491, 0.00587815, 0.02117637,
            0.0293578 , 0.03452379, 0.02194803, 0.01676056,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 400], expected))

        spec = spectrogram(
            waveform,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            fft_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (257, 732))
        self.assertTrue(np.allclose(spec[:64, 400], expected))

        mel_filters = mel_filter_bank(
            num_frequency_bins=256,
            num_mel_filters=400,
            min_frequency=20,
            max_frequency=8000,
            sampling_rate=16000,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )

        mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))

        spec = spectrogram(
            waveform,
            window_function(400, "povey", periodic=False),
            frame_length=400,
            hop_length=160,
            fft_length=512,
            power=2.0,
            center=False,
            pad_mode="reflect",
            onesided=True,
            preemphasis=0.97,
            mel_filters=mel_filters,
            log_mel="log",
            mel_floor=1.1920928955078125e-07,
            remove_dc_offset=True,
        )
        self.assertEqual(spec.shape, (400, 584))

        # fmt: off
        expected = np.array([-15.94238515,  -8.20712299,  -8.22704352, -15.94238515,
       -15.94238515, -15.94238515, -15.94238515, -15.94238515,
        -6.52463769,  -7.73677889, -15.94238515, -15.94238515,
       -15.94238515, -15.94238515,  -4.18650018,  -3.37195286,
       -15.94238515, -15.94238515, -15.94238515, -15.94238515,
        -4.70190154,  -2.4217066 , -15.94238515, -15.94238515,
       -15.94238515, -15.94238515,  -5.62755239,  -3.53385194,
       -15.94238515, -15.94238515, -15.94238515, -15.94238515,
        -9.43303023,  -8.77480925, -15.94238515, -15.94238515,
       -15.94238515, -15.94238515,  -4.2951092 ,  -5.51585994,
       -15.94238515, -15.94238515, -15.94238515,  -4.40151721,
        -3.95228878, -15.94238515, -15.94238515, -15.94238515,
        -6.10365415,  -4.59494697, -15.94238515, -15.94238515,
       -15.94238515,  -8.10727767,  -6.2585298 , -15.94238515,
       -15.94238515, -15.94238515,  -5.60161702,  -4.47217004,
       -15.94238515, -15.94238515, -15.94238515,  -5.91641988]
        )
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 400], expected, atol=1e-5))

    def test_spectrogram_center_padding(self):
        waveform = self._load_datasamples(1)[0]

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=True,
            pad_mode="reflect",
        )
        self.assertEqual(spec.shape, (257, 732))

        # fmt: off
        expected = np.array([
            0.1287945 , 0.12792738, 0.08311573, 0.03155122, 0.02470202,
            0.00727857, 0.00910694, 0.00686163, 0.01238981, 0.01473668,
            0.00336144, 0.00370314, 0.00600871, 0.01120164, 0.01942998,
            0.03132008, 0.0232842 , 0.01124642, 0.02754783, 0.02423725,
            0.00147893, 0.00038027, 0.00112299, 0.00596233, 0.00571529,
            0.02084235, 0.0231855 , 0.00810006, 0.01837943, 0.00651339,
            0.00093931, 0.00067426, 0.01058399, 0.01270507, 0.00151734,
            0.00331913, 0.00302416, 0.01081792, 0.00754549, 0.00148963,
            0.00111943, 0.00152573, 0.00608017, 0.01749986, 0.01205949,
            0.0143082 , 0.01910573, 0.00413786, 0.03916619, 0.09873404,
            0.08302026, 0.02673891, 0.00401255, 0.01397392, 0.00751862,
            0.01024884, 0.01544606, 0.00638907, 0.00623633, 0.0085103 ,
            0.00217659, 0.00276204, 0.00260835, 0.00299299,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 0], expected))

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=True,
            pad_mode="constant",
        )
        self.assertEqual(spec.shape, (257, 732))

        # fmt: off
        expected = np.array([
            0.06558744, 0.06889656, 0.06263352, 0.04264418, 0.03404115,
            0.03244197, 0.02279134, 0.01646339, 0.01452216, 0.00826055,
            0.00062093, 0.0031821 , 0.00419456, 0.00689327, 0.01106367,
            0.01712119, 0.01721762, 0.00977533, 0.01606626, 0.02275621,
            0.01727687, 0.00992739, 0.01217688, 0.01049927, 0.01022947,
            0.01302475, 0.01166873, 0.01081812, 0.01057327, 0.00767912,
            0.00429567, 0.00089625, 0.00654583, 0.00912084, 0.00700984,
            0.00225026, 0.00290545, 0.00667712, 0.00730663, 0.00410813,
            0.00073102, 0.00219296, 0.00527618, 0.00996585, 0.01123781,
            0.00872816, 0.01165121, 0.02047945, 0.03681747, 0.0514379 ,
            0.05137928, 0.03960042, 0.02821562, 0.01813349, 0.01201322,
            0.01260964, 0.00900654, 0.00207905, 0.00456714, 0.00850599,
            0.00788239, 0.00664407, 0.00824227, 0.00628301,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 0], expected))

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=128,
            center=False,
        )
        self.assertEqual(spec.shape, (257, 728))

        # fmt: off
        expected = np.array([
            0.00250445, 0.02161521, 0.06232229, 0.04339567, 0.00937727,
            0.01080616, 0.00248685, 0.0095264 , 0.00727476, 0.0079152 ,
            0.00839946, 0.00254932, 0.00716622, 0.005559  , 0.00272623,
            0.00581774, 0.01896395, 0.01829788, 0.01020514, 0.01632692,
            0.00870888, 0.02065827, 0.0136022 , 0.0132382 , 0.011827  ,
            0.00194505, 0.0189979 , 0.026874  , 0.02194014, 0.01923883,
            0.01621437, 0.00661967, 0.00289517, 0.00470257, 0.00957801,
            0.00191455, 0.00431664, 0.00544359, 0.01126213, 0.00785778,
            0.00423469, 0.01322504, 0.02226548, 0.02318576, 0.03428908,
            0.03648811, 0.0202938 , 0.011902  , 0.03226198, 0.06347476,
            0.01306318, 0.05308729, 0.05474771, 0.03127991, 0.00998512,
            0.01449977, 0.01272741, 0.00868176, 0.00850386, 0.00313876,
            0.00811857, 0.00538216, 0.00685749, 0.00535275,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:64, 0], expected))

    def test_spectrogram_shapes(self):
        waveform = self._load_datasamples(1)[0]

        spec = spectrogram(
            waveform,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (201, 732))

        spec = spectrogram(
            waveform,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            power=1.0,
            center=False,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (201, 729))

        spec = spectrogram(
            waveform,
            window_function(400, "hann"),
            frame_length=400,
            hop_length=128,
            fft_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.assertEqual(spec.shape, (257, 732))

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=64,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec.shape, (512, 1464))

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=64,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec.shape, (512, 1464))

        spec = spectrogram(
            waveform,
            window_function(512, "hann"),
            frame_length=512,
            hop_length=512,
            power=1.0,
            center=True,
            pad_mode="reflect",
            onesided=False,
        )
        self.assertEqual(spec.shape, (512, 183))

    def test_mel_spectrogram(self):
        waveform = self._load_datasamples(1)[0]

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

        spec = spectrogram(
            waveform,
            window_function(800, "hann", frame_length=1024),
            frame_length=1024,
            hop_length=128,
            power=2.0,
        )
        self.assertEqual(spec.shape, (513, 732))

        spec = spectrogram(
            waveform,
            window_function(800, "hann", frame_length=1024),
            frame_length=1024,
            hop_length=128,
            power=2.0,
            mel_filters=mel_filters,
        )
        self.assertEqual(spec.shape, (13, 732))

        # fmt: off
        expected = np.array([
            1.08027889e+02, 1.48080673e+01, 7.70758213e+00, 9.57676639e-01,
            8.81639061e-02, 5.26073833e-02, 1.52736155e-02, 9.95350117e-03,
            7.95364356e-03, 1.01148004e-02, 4.29241020e-03, 9.90708797e-03,
            9.44153646e-04
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[:, 300], expected))

    def test_spectrogram_power(self):
        waveform = self._load_datasamples(1)[0]

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=None,
        )
        self.assertEqual(spec.shape, (257, 732))
        self.assertEqual(spec.dtype, np.complex64)

        # fmt: off
        expected = np.array([
             0.01452305+0.01820039j, -0.01737362-0.01641946j,
             0.0121028 +0.01565081j, -0.02794554-0.03021514j,
             0.04719803+0.04086519j, -0.04391563-0.02779365j,
             0.05682834+0.01571325j, -0.08604821-0.02023657j,
             0.07497991+0.0186641j , -0.06366091-0.00922475j,
             0.11003416+0.0114788j , -0.13677941-0.01523552j,
             0.10934535-0.00117226j, -0.11635598+0.02551187j,
             0.14708674-0.03469823j, -0.1328196 +0.06034218j,
             0.12667368-0.13973421j, -0.14764774+0.18912019j,
             0.10235471-0.12181523j, -0.00773012+0.04730498j,
            -0.01487191-0.07312611j, -0.02739162+0.09619419j,
             0.02895459-0.05398273j,  0.01198589+0.05276592j,
            -0.02117299-0.10123465j,  0.00666388+0.09526499j,
            -0.01672773-0.05649684j,  0.02723125+0.05939891j,
            -0.01879361-0.062954j  ,  0.03686557+0.04568823j,
            -0.07394181-0.07949649j,  0.06238583+0.13905765j,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[64:96, 321], expected))

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=1.0,
        )
        self.assertEqual(spec.shape, (257, 732))
        self.assertEqual(spec.dtype, np.float64)

        # fmt: off
        expected = np.array([
            0.02328461, 0.02390484, 0.01978448, 0.04115711, 0.0624309 ,
            0.05197181, 0.05896072, 0.08839577, 0.07726794, 0.06432579,
            0.11063128, 0.13762532, 0.10935163, 0.11911998, 0.15112405,
            0.14588428, 0.18860507, 0.23992978, 0.15910825, 0.04793241,
            0.07462307, 0.10001811, 0.06125769, 0.05411011, 0.10342509,
            0.09549777, 0.05892122, 0.06534349, 0.06569936, 0.05870678,
            0.10856833, 0.1524107 , 0.11463385, 0.05766969, 0.12385171,
            0.14472842, 0.11978184, 0.10353675, 0.07244056, 0.03461861,
            0.02624896, 0.02227475, 0.01238363, 0.00885281, 0.0110049 ,
            0.00807005, 0.01033663, 0.01703181, 0.01445856, 0.00585615,
            0.0132431 , 0.02754132, 0.01524478, 0.0204908 , 0.07453328,
            0.10716327, 0.07195779, 0.08816078, 0.18340898, 0.16449876,
            0.12322842, 0.1621659 , 0.12334293, 0.06033659,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[64:128, 321], expected))

        spec = spectrogram(
            waveform,
            window_function(400, "hann", frame_length=512),
            frame_length=512,
            hop_length=128,
            power=2.0,
        )
        self.assertEqual(spec.shape, (257, 732))
        self.assertEqual(spec.dtype, np.float64)

        # fmt: off
        expected = np.array([
            5.42173162e-04, 5.71441371e-04, 3.91425507e-04, 1.69390778e-03,
            3.89761780e-03, 2.70106923e-03, 3.47636663e-03, 7.81381316e-03,
            5.97033510e-03, 4.13780799e-03, 1.22392802e-02, 1.89407300e-02,
            1.19577805e-02, 1.41895693e-02, 2.28384770e-02, 2.12822221e-02,
            3.55718732e-02, 5.75663000e-02, 2.53154356e-02, 2.29751552e-03,
            5.56860259e-03, 1.00036217e-02, 3.75250424e-03, 2.92790355e-03,
            1.06967501e-02, 9.11982451e-03, 3.47171025e-03, 4.26977174e-03,
            4.31640586e-03, 3.44648538e-03, 1.17870830e-02, 2.32290216e-02,
            1.31409196e-02, 3.32579296e-03, 1.53392460e-02, 2.09463164e-02,
            1.43476883e-02, 1.07198600e-02, 5.24763530e-03, 1.19844836e-03,
            6.89007982e-04, 4.96164430e-04, 1.53354369e-04, 7.83722571e-05,
            1.21107812e-04, 6.51257360e-05, 1.06845939e-04, 2.90082477e-04,
            2.09049831e-04, 3.42945241e-05, 1.75379610e-04, 7.58524227e-04,
            2.32403356e-04, 4.19872697e-04, 5.55520924e-03, 1.14839673e-02,
            5.17792348e-03, 7.77232368e-03, 3.36388536e-02, 2.70598419e-02,
            1.51852425e-02, 2.62977779e-02, 1.52134784e-02, 3.64050455e-03,
        ])
        # fmt: on
        self.assertTrue(np.allclose(spec[64:128, 321], expected))

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
