# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

import inspect
import json
import tempfile
import unittest

from transformers.testing_utils import require_torch, require_torch_gpu, slow, torch_device
from transformers.utils import is_datasets_available, is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor


if is_datasets_available():
    from datasets import Audio, load_dataset


if is_torch_available():
    import torch

    from transformers import VocosFeatureExtractor, VocosModel


from transformers import VocosConfig


class VocosModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 2
        self.n_mels = 100
        self.hidden_size = 16
        self.intermediate_size = 32
        self.num_layers = 2
        self.kernel_size = 3
        self.padding = 1
        self.layer_scale_init_value = 0.1
        self.use_adaptive_norm = False
        self.num_bandwidths = 1
        self.layer_norm_eps = 1e-6
        self.n_fft = 16
        self.hop_length = 8
        self.istft_padding = "center"
        self.seq_length = 10
        self.hidden_act = "gelu"

    def get_config(self):
        return VocosConfig(
            n_mels=self.n_mels,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_layers=self.num_layers,
            kernel_size=self.kernel_size,
            padding=self.padding,
            layer_scale_init_value=self.layer_scale_init_value,
            use_adaptive_norm=self.use_adaptive_norm,
            num_bandwidths=self.num_bandwidths,
            layer_norm_eps=self.layer_norm_eps,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            istft_padding=self.istft_padding,
            hidden_act=self.hidden_act,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_values = floats_tensor([self.batch_size, self.n_mels, self.seq_length])
        return config, input_values

    def prepare_config_and_inputs_for_common(self):
        config, features = self.prepare_config_and_inputs()
        return config, {"input_features": features}

    def create_and_check_model(self, config, features):
        model = VocosModel(config=config).to(torch_device).eval()
        with torch.no_grad():
            output = model(features.to(torch_device))

        if config.istft_padding == "center":
            # when padding is `center`,  output is computed using PyTorch's ISTFT
            expected_len = (self.seq_length - 1) * config.hop_length

        elif config.istft_padding == "same":
            # when padding is `same`, output is computed using custom ISTFT implementation in `custom_istft`
            pad = (config.n_fft - config.hop_length) // 2
            expected_len = (self.seq_length - 1) * config.hop_length + config.n_fft - 2 * pad

        self.parent.assertEqual(output.audio.shape, (self.batch_size, expected_len))


@require_torch
class VocosModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VocosModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False
    test_resize_embeddings = False
    test_attention_outputs = False
    has_attentions = False
    test_missing_keys = False
    test_can_init_all_missing_weights = False

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

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = VocosModel(config)
        signature = inspect.signature(model.forward)
        arg_names = list(signature.parameters.keys())
        self.assertListEqual(arg_names, ["input_features", "attention_mask", "kwargs"])

    @unittest.skip(
        reason="The VocosModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_save_load(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = [
                    "embed.weight",
                    "dwconv.weight",
                    "pwconv1.weight",
                    "pwconv2.weight",
                    "out.weight",
                ]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

    @unittest.skip(
        reason="The VocosModel is not transformers based, thus it does not have the usual `hidden_states` logic"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="The VocosModel does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="VocosModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VocosModel does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="VocosModel does not output any loss term in the forward pass")
    def test_training(self):
        pass

    @unittest.skip(reason="VocosModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="VocosModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="VocosModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="VocosModel has no attention layers")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip("VocosModel cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip(reason="The VocosModel does not have `hidden_states`")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    def test_save_load_strict(self):
        config, _ = self.model_tester.prepare_config_and_inputs()
        model = VocosModel(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            _, info = VocosModel.from_pretrained(tmpdirname, output_loading_info=True)
        self.assertEqual(info["missing_keys"], set())


@require_torch
class VocosModelIntegrationTest(unittest.TestCase):
    """
    See code for reproducing expected outputs: https://gist.github.com/Manalelaidouni/f09cbaac2c56199f0e5e05c498fdb6b0
    Outputs should be computed on GPU because the mel spectrogram outputs differ on CPU and GPU.
    """

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=24000))
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    def setUp(self):
        with open("tests/fixtures/vocos/vocos_mel_integration.json", "r") as f:
            self.mel_expected = json.load(f)[0]
        with open("tests/fixtures/vocos/vocos_mel_batch_integration.json", "r") as f:
            self.mel_batch_expected = json.load(f)

    @slow
    @require_torch_gpu
    def test_inference(self):
        hf_repo_id = "Manel/vocos-mel-24khz"
        feature_extractor = VocosFeatureExtractor.from_pretrained(hf_repo_id)
        model = VocosModel.from_pretrained(hf_repo_id).to(torch_device).eval()

        audio_np = self._load_datasamples(1)[0]
        audio = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0).to(torch_device)

        EXPECTED_AUDIO = torch.tensor(self.mel_expected["reconstructed_audio"], dtype=torch.float32).to(torch_device)

        inputs = feature_extractor(audio, return_tensors="pt", device=torch_device)
        with torch.no_grad():
            audio_output = model(**inputs).audio

        torch.testing.assert_close(
            audio_output.squeeze(0)[: EXPECTED_AUDIO.shape[0]],
            EXPECTED_AUDIO,
            rtol=1e-5,
            atol=1e-5,
        )

    @slow
    @require_torch_gpu
    def test_inference_batch(self):
        repo_id = "Manel/vocos-mel-24khz"
        feature_extractor = VocosFeatureExtractor.from_pretrained(repo_id)
        model = VocosModel.from_pretrained(repo_id).to(torch_device).eval()

        audios = self._load_datasamples(3)

        inputs = feature_extractor(audio=audios, return_tensors="pt", device=torch_device)
        hf_batch_output = model(**inputs).audio

        for i, saved in enumerate(self.mel_batch_expected["reconstructed_audio"]):
            expected = torch.tensor(saved, dtype=torch.float32, device=torch_device)
            torch.testing.assert_close(
                hf_batch_output[i, : expected.shape[0]],
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
