# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Vocos model."""

import tempfile
import unittest

from datasets import Audio, load_dataset

from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import VocosModel, VocosWithEncodecModel


from transformers import VocosConfig, VocosFeatureExtractor, VocosWithEncodecConfig


class VocosModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.input_channels = 8
        self.hidden_dim = 16
        self.intermediate_dim = 32
        self.num_layers = 2
        self.kernel_size = 3
        self.padding = 1
        self.layer_scale_init_value = 0.1
        self.use_adaptive_norm = False
        self.num_bandwidths = 1
        self.layer_norm_eps = 1e-6
        self.n_fft = 16
        self.hop_length = 8
        self.spec_padding = "center"
        self.seq_length = 10

    def get_config(self):
        return VocosConfig(
            input_channels=self.input_channels,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            padding=self.padding,
            layer_scale_init_value=self.layer_scale_init_value,
            use_adaptive_norm=self.use_adaptive_norm,
            num_bandwidths=self.num_bandwidths,
            layer_norm_eps=self.layer_norm_eps,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            spec_padding=self.spec_padding,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_values = floats_tensor([self.batch_size, self.input_channels, self.seq_length])
        return config, input_values

    def create_and_check_model(self, config, features):
        model = VocosModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            audio = model(features.to(torch_device))
        if config.spec_padding == "center":
            # the expected output using PyTorch's ISTFT
            expected_len = (self.seq_length - 1) * config.hop_length
        else:
            # when padding is same "same" padding, the expected output using the custom ISTFT implementation
            pad = (config.n_fft - config.hop_length) // 2
            expected_len = (self.seq_length - 1) * config.hop_length + config.n_fft - 2 * pad
        self.parent.assertEqual(audio.shape, (self.batch_size, expected_len))


@require_torch
class VocosModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = VocosModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=VocosConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_save_load_strict(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = VocosModel(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            _, info = VocosModel.from_pretrained(tmpdirname, output_loading_info=True)
        self.assertEqual(info["missing_keys"], [])


class VocosWithEncodecModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.audio_length = 512
        self.seq_length = 12

    def get_config(self):
        return VocosWithEncodecConfig()

    def prepare_config_and_inputs(self):
        config = self.get_config()
        model = VocosWithEncodecModel(config).to(torch_device).eval()
        codebook_size = model.encodec_model.quantizer.layers[0].codebook.embed.shape[0]
        codes = ids_tensor([model.num_quantizers, self.batch_size, self.seq_length], codebook_size).to(torch_device)
        audio = floats_tensor([self.batch_size, self.audio_length]).to(torch_device)
        bandwidth_id = torch.tensor(0, dtype=torch.long, device=torch_device)
        return config, codes, audio, bandwidth_id

    def create_and_check_model(self, config, codes, audio, bandwidth_id):
        model = VocosWithEncodecModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            audio_from_codes = model(codes=codes, bandwidth_id=bandwidth_id)
        if config.spec_padding == "center":
            expected_len_codes = (self.seq_length - 1) * config.hop_length + config.n_fft
        elif config.spec_padding == "same":
            expected_len_codes = self.seq_length * config.hop_length
        self.parent.assertEqual(audio_from_codes.shape, (self.batch_size, expected_len_codes))
        with torch.no_grad():
            audio_from_audio, out_codes = model(audio=audio, bandwidth_id=bandwidth_id, return_codes=True)
        actual_seq_length = out_codes.shape[-1]
        if config.spec_padding == "center":
            expected_len = (actual_seq_length - 1) * config.hop_length + config.n_fft
        elif config.spec_padding == "same":
            expected_len = actual_seq_length * config.hop_length
        self.parent.assertEqual(audio_from_audio.shape, (self.batch_size, expected_len))


@require_torch
class VocosWithEncodecModelTest(unittest.TestCase):
    def setUp(self):
        self.tester = VocosWithEncodecModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=VocosWithEncodecConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, codes, audio, bandwidth_id = self.tester.prepare_config_and_inputs()
        self.tester.create_and_check_model(config, codes, audio, bandwidth_id)

    def test_save_load_strict(self):
        config, codes, audio, bandwidth_id = self.tester.prepare_config_and_inputs()
        model = VocosWithEncodecModel(config=config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            _, info = VocosWithEncodecModel.from_pretrained(tmpdir, output_loading_info=True)
        self.assertListEqual(info["missing_keys"], [])


@require_torch
class VocosModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.feature_extractor = VocosFeatureExtractor.from_pretrained("Manel/Vocos", resume_download=True)
        self.model = VocosModel.from_pretrained("Manel/Vocos", resume_download=True).to(torch_device).eval()
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=self.feature_extractor.sampling_rate))
        self.speech = ds[0]["audio"]["array"]

    def test_inference(self):
        EXPECTED = torch.tensor(
            [
                0.0001700431457720697,
                0.00010000158363254741,
                -5.997690459480509e-05,
                -8.697436715010554e-05,
                3.8385427615139633e-05,
                0.0001993452024180442,
                0.00026118403184227645,
                0.00024136024876497686,
                0.0002001010434469208,
                0.000260183762293309,
                0.000239697823417373,
                1.3868119822291192e-05,
                -6.546344957314432e-05,
                2.3145852537709288e-05,
                0.0001909736020024866,
                0.00043056777212768793,
                0.00040265079587697983,
                -7.634644862264395e-05,
                -0.0007267086184583604,
                -0.0012220395728945732,
            ],
            dtype=torch.float32,
        )

        inputs = self.feature_extractor(
            self.speech, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            audio = self.model(inputs.input_features)

        expected_shape = torch.Size([1, 140544])
        self.assertEqual(audio.shape, expected_shape)

        torch.testing.assert_close(audio[0][: EXPECTED.shape[0]], EXPECTED, rtol=1e-4, atol=1e-4)


@require_torch
class VocosWithEncodecModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model = (
            VocosWithEncodecModel.from_pretrained("Manel/Vocos-Encodec", resume_download=True).to(torch_device).eval()
        )
        self.config = VocosWithEncodecConfig.from_pretrained("Manel/Vocos-Encodec", resume_download=True)
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        self.speech = ds[0]["audio"]["array"]

    def test_inference_audio_and_codes(self):
        EXPECTED_AUDIO = torch.tensor(
            [
                -0.00015610073751304299,
                0.0006738820229656994,
                0.0014662687899544835,
                0.0019666007719933987,
                0.0018747239373624325,
                0.0016342204762622714,
                0.0013575436314567924,
                0.0010286348406225443,
                0.00036631093826144934,
                -7.642315176781267e-05,
                -0.0005207710200920701,
                -0.0007273774244822562,
                -0.0006747262086719275,
                -6.980449688853696e-05,
                0.0008167537162080407,
                0.0008955168887041509,
                0.0011381119256839156,
                0.0012689086142927408,
                0.0016888295067474246,
                0.001389320706948638,
            ],
            dtype=torch.float32,
        )

        EXPECTED_AUDIO_FROM_CODES = torch.tensor(
            [
                -0.00015610073751304299,
                0.0006738820229656994,
                0.0014662687899544835,
                0.0019666007719933987,
                0.0018747239373624325,
                0.0016342204762622714,
                0.0013575436314567924,
                0.0010286348406225443,
                0.00036631093826144934,
                -7.642315176781267e-05,
                -0.0005207710200920701,
                -0.0007273774244822562,
                -0.0006747262086719275,
                -6.980449688853696e-05,
                0.0008167537162080407,
                0.0008955168887041509,
                0.0011381119256839156,
                0.0012689086142927408,
                0.0016888295067474246,
                0.001389320706948638,
            ]
        )

        audio_tensor = torch.tensor(self.speech, dtype=torch.float32, device=torch_device).unsqueeze(0)
        bandwidth_id = torch.tensor([0], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            output_from_audio = self.model(audio=audio_tensor, bandwidth_id=bandwidth_id)

        expected_shape = torch.Size((1, 140800))
        self.assertEqual(output_from_audio.shape, expected_shape)
        torch.testing.assert_close(
            output_from_audio[0, : EXPECTED_AUDIO.shape[0]], EXPECTED_AUDIO, rtol=1e-4, atol=1e-4
        )

        with torch.no_grad():
            codes = self.model.encodec_model.quantizer.encode(
                self.model.encodec_model.encoder(audio_tensor.unsqueeze(1)),
                bandwidth=self.config.encodec_config.target_bandwidths[0],
            )
            output_from_codes = self.model(codes=codes, bandwidth_id=bandwidth_id)

        self.assertEqual(output_from_codes.shape, output_from_audio.shape)
        self.assertEqual(output_from_codes.shape, expected_shape)
        torch.testing.assert_close(
            output_from_codes[0, : EXPECTED_AUDIO_FROM_CODES.shape[0]], EXPECTED_AUDIO_FROM_CODES, rtol=1e-4, atol=1e-4
        )
