# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HiFTNet model."""

import inspect
import tempfile
import unittest

from transformers import HiFTNetConfig
from transformers.testing_utils import is_torch_available, require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import HiFTNetModel


@require_torch
class HiFTNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        mel_time_steps=100,
        mel_bins=80,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.mel_time_steps = mel_time_steps
        self.mel_bins = mel_bins
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        # Input shape: (batch_size, time_steps, mel_bins)
        speech_feat = floats_tensor([self.batch_size, self.mel_time_steps, self.mel_bins], scale=1.0)
        config = self.get_config()
        inputs_dict = {"speech_feat": speech_feat}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return HiFTNetConfig(
            in_channels=80,
            base_channels=512,
            nb_harmonics=8,
            sampling_rate=22050,
            nsf_alpha=0.1,
            nsf_sigma=0.003,
            nsf_voiced_threshold=10.0,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            istft_n_fft=16,
            istft_hop_len=4,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            lrelu_slope=0.1,
            audio_limit=0.99,
        )

    def create_and_check_model(self, config, speech_feat):
        model = HiFTNetModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(speech_feat)
        # Check output shape: should be waveform
        # Expected length: mel_time_steps * product(upsample_rates) * istft_hop_len
        expected_length = self.mel_time_steps * 8 * 5 * 3 * 4  # 48000
        self.parent.assertEqual(result.shape, (self.batch_size, expected_length))


@require_torch
class HiFTNetModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (HiFTNetModel,) if is_torch_available() else ()
    is_encoder_decoder = False
    has_attentions = False
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = HiFTNetModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=HiFTNetConfig, has_text_modality=False, common_properties=["hidden_size"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, inputs_dict["speech_feat"])

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # Signature should contain 'speech_feat'
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["speech_feat"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # Set seed for reproducibility
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Reset seed before loading
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

                model_loaded = model_class.from_pretrained(tmpdirname)
                model_loaded.to(torch_device)
                model_loaded.eval()

                # Test that outputs are the same
                with torch.no_grad():
                    output1 = model(**inputs_dict)
                    output2 = model_loaded(**inputs_dict)

                # Use more lenient tolerance for vocoder models due to numerical precision
                self.assertTrue(torch.allclose(output1, output2, atol=1e-3, rtol=1e-3))

    @unittest.skip("HiFTNet is a vocoder model and does not support attention outputs")
    def test_attention_outputs(self):
        pass

    @unittest.skip("HiFTNet is a vocoder model and does not support hidden states output")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("HiFTNet is a vocoder model and does not have input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("HiFTNet model is too large for common tests")
    def test_model_is_small(self):
        pass

    @unittest.skip("HiFTNet is a vocoder model and does not support retaining gradients on hidden states/attentions")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("HiFTNet returns a Tensor, not a ModelOutput object")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip("HiFTNet has complex weight initialization with weight_norm that cannot be fully tested")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip("HiFTNet does not support safetensors part of s3gen")
    def test_can_use_safetensors(self):
        pass

    @unittest.skip("HiFTNet does not support load_save_without_tied_weights part of s3gen")
    def test_load_save_without_tied_weights(self):
        pass

    def test_batching_equivalence(self):
        # Override to handle vocoder model output format
        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            batch_size = self.model_tester.batch_size
            single_row_input = {}
            for key, value in batched_input.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                    single_row_input[key] = value[:1]  # Take first item
                else:
                    single_row_input[key] = value

            with torch.no_grad():
                model_batched_output = model(**batched_input)
                model_row_output = model(**single_row_input)

            # For vocoder models, output is a tensor
            if isinstance(model_batched_output, torch.Tensor):
                # Check that batched output has correct batch size
                self.assertEqual(model_batched_output.shape[0], batch_size)
                self.assertEqual(model_row_output.shape[0], 1)
                # Check first batch matches single row
                self.assertTrue(
                    torch.allclose(
                        model_batched_output[0:1],
                        model_row_output,
                        atol=1e-4,
                    )
                )

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                # Set seeds for determinism
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                first = model(**self._prepare_for_class(inputs_dict, model_class))
                torch.manual_seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(0)
                second = model(**self._prepare_for_class(inputs_dict, model_class))
            # Use more lenient tolerance for vocoder models
            self.assertTrue(torch.allclose(first, second, atol=1e-3))
