# # # # # # # # # # # # # # # # # # # # # # #
#  WIP - Currently has old draft FS2 tests  #
# # # # # # # # # # # # # # # # # # # # # # #


# # coding=utf-8
# # Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
""" Testing suite for the PyTorch FastSpeech2Conformer model. """

import unittest

from transformers import FastSpeech2ConformerConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, _config_zero_init


if is_torch_available():
    import torch

    from transformers import FastSpeech2ConformerModel


class FastSpeech2ConformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_hidden_layers=4,
        hidden_size=24,
        seq_length=7,
        is_training=False,
        vocab_size=20,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return config, input_ids

    def get_config(self):
        return FastSpeech2ConformerConfig(
            hidden_size=self.hidden_size, encoder_layers=self.num_hidden_layers, decoder_layers=self.num_hidden_layers
        )

    def create_and_check_model(self, config, input_ids, *args):
        model = FastSpeech2ConformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, return_dict=True)

        # total of 8 keys in result
        self.parent.assertEqual(len(result), 5)
        # check batch sizes match
        for value in result.values():
            self.parent.assertEqual(value.size(0), self.batch_size)
        # check duration, pitch, and energy have the appopriate shapes
        # duration: (batch_size, max_text_length), pitch and energy: (batch_size, max_text_length, 1)
        self.parent.assertEqual(result["duration_outputs"].shape + (1,), result["pitch_outputs"].shape)
        self.parent.assertEqual(result["pitch_outputs"].shape, result["energy_outputs"].shape)
        # check predicted mel-spectrogram has correct dimension
        self.parent.assertEqual(result["spectrogram"].size(2), model.config.num_mel_bins)

    def prepare_config_and_inputs_for_common(self):
        config, input_ids = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@require_torch
class FastSpeech2ConformerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (FastSpeech2ConformerModel,) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False
    test_torchscript = False
    has_attentions = False

    def setUp(self):
        self.model_tester = FastSpeech2ConformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FastSpeech2ConformerConfig)
        # TODO confirm the below line should be deleted
        # self.config_tester.create_and_test_config_common_properties = lambda: None

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # skip as this model does not output attentions
    def test_attention_outputs(self):
        pass

    # skip as this model does not output attentions
    def test_retain_grad_hidden_states_attentions(sef):
        pass

    def test_hidden_states_output(self):
        # TODO
        pass

    @unittest.skip(reason="FastSpeech2Conformer does not accept inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="FastSpeech2Conformer has no input embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="FastSpeech2Conformer does not have forward chunking implemented")
    def test_feed_forward_chunking(self):
        pass

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "norm" in name:
                        if "bias" in name:
                            self.assertEqual(
                                param.data.mean().item(),
                                0.0,
                                msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                            )
                        if "weight" in name:
                            self.assertEqual(
                                param.data.mean().item(),
                                1.0,
                                msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                            )
                    elif "conv" in name:
                        # For Conv1D layers initialized with Kaiming Normal, the mean should be close to 0
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif "embed" in name:
                        # For Embedding layers initialized with standard normal, the mean should be close to 0
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # TODO confirm this
    @unittest.skip(reason="FastSpeech2Conformer does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = FastSpeech2ConformerModel.from_pretrained("connor-henderson/fastspeech2_conformer")
        self.assertIsNotNone(model)


@require_torch
@slow
class FastSpeech2ConformerModelIntegrationTest(unittest.TestCase):
    # @unittest.skipIf(torch_device != "cpu", "cannot make deterministic on GPU")
    # @unittest.skip(reason="Not pushed to hub yet")
    def test_inference_integration(self):
        model = FastSpeech2ConformerModel.from_pretrained("connor-henderson/fastspeech2_conformer")
        model.to(torch_device)
        # tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("jaketae/FastSpeech2Conformer-ljspeech")
        # text = ["Test that this generates speech"]
        # inputs_old = tokenizer(text, return_tensors="pt", padding=True).to(torch_device)
        inputs = torch.tensor([4, 15, 6, 4, 9, 18, 4, 9, 12, 6, 40, 15, 3, 21, 47, 4, 6, 6, 17, 27, 39])
        outputs_dict = model(inputs, return_dict=True)
        spectrogram = outputs_dict["spectrogram"]

        # mel-spectrogram is too large (1, 205, 80),
        # so only check top-left 100 elements
        expected_mel_spectrogram = torch.tensor(
            [
                [-1.2426, -1.7286, -1.6754, -1.7451, -1.6402, -1.5219, -1.4480, -1.3345, -1.4031, -1.4497],
                [-0.7858, -1.4966, -1.3602, -1.4876, -1.2949, -1.0723, -1.0021, -0.7553, -0.6521, -0.6929],
                [-0.7298, -1.3908, -1.0369, -1.2656, -1.0342, -0.7883, -0.7420, -0.5249, -0.3734, -0.3977],
                [-0.4784, -1.3508, -1.1558, -1.4678, -1.2820, -1.0252, -1.0868, -0.9006, -0.8947, -0.8448],
                [-0.3963, -1.2895, -1.2813, -1.6147, -1.4658, -1.2560, -1.4134, -1.2650, -1.3255, -1.1715],
                [-1.4914, -1.3097, -0.3821, -0.3898, -0.5748, -0.9040, -1.0755, -1.0575, -1.2205, -1.0572],
                [0.0197, -0.0582, 0.9147, 1.1512, 1.1651, 0.6628, -0.1010, -0.3085, -0.2285, 0.2650],
                [1.1780, 0.1803, 0.7251, 1.5728, 1.6678, 0.4542, -0.1572, -0.1787, 0.0744, 0.8168],
                [-0.2078, -0.3211, 1.1096, 1.5085, 1.4632, 0.6299, -0.0515, 0.0589, 0.8609, 1.4429],
                [0.7831, -0.2663, 1.0352, 1.4489, 0.9088, 0.0247, -0.3995, 0.0078, 1.2446, 1.6998],
            ],
            device=torch_device,
        )

        self.assertTrue(torch.allclose(spectrogram[0, :10, :10], expected_mel_spectrogram, atol=1e-4))
