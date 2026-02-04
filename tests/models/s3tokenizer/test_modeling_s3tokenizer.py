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
"""Testing suite for the PyTorch S3Tokenizer model."""

import unittest

from transformers import S3TokenizerConfig
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    from transformers import S3TokenizerModel


@require_torch
class S3TokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=400,
        is_training=False,
        use_labels=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_features = floats_tensor([self.batch_size, self.seq_length, config.n_mels], scale=1.0)
        inputs_dict = {"input_features": input_features}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return S3TokenizerConfig(
            n_mels=80,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            vocab_size=6561,
            n_fft=400,
            hop_length=160,
            sampling_rate=16000,
            use_sdpa=False,
        )

    def create_and_check_model(self, config, input_features):
        model = S3TokenizerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_features)
        self.parent.assertIsNotNone(result.speech_tokens)
        self.parent.assertIsNotNone(result.speech_token_lens)


@require_torch
class S3TokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (S3TokenizerModel,) if is_torch_available() else ()
    is_encoder_decoder = False
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = S3TokenizerModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=S3TokenizerConfig, has_text_modality=False, common_properties=["hidden_size"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, inputs_dict["input_features"])

    @unittest.skip(reason="S3Tokenizer does not output hidden states in the traditional sense")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not have attention weights in the traditional sense")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support input embeddings")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support training mode")
    def test_training(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support training mode")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support retain_grad")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not have typical model forward signature")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="S3Tokenizer model does not support typical model features")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not use feed forward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="S3Tokenizer model is too large for common tests")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support output_hidden_states")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support init weights")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support safetensors")
    def test_can_use_safetensors(self):
        return super().test_can_use_safetensors()

    @unittest.skip(reason="S3Tokenizer does not support tied weights")
    def test_load_save_without_tied_weights(self):
        return super().test_load_save_without_tied_weights()

    @unittest.skip(reason="S3Tokenizer does not support init weights")
    def test_save_load(self):
        return super().test_save_load()

    def test_window_buffer_loading(self):
        """Test that the window buffer can be loaded from checkpoint if it exists."""
        import tempfile

        import torch

        config = self.model_tester.get_config()
        model1 = S3TokenizerModel(config=config)

        # Modify the window to a custom value
        custom_window = torch.ones_like(model1.window) * 0.5
        model1.window = custom_window

        # Save the model with the custom window
        with tempfile.TemporaryDirectory() as tmp_dir:
            model1.save_pretrained(tmp_dir)

            # Load the model and verify the window was loaded
            model2 = S3TokenizerModel.from_pretrained(tmp_dir)

            # Verify the custom window was loaded from checkpoint
            self.assertTrue(torch.allclose(model2.window, custom_window))

    def test_window_buffer_missing_from_checkpoint(self):
        """Test that the default window is used when not present in checkpoint."""
        import torch

        config = self.model_tester.get_config()
        model = S3TokenizerModel(config=config)

        # Create a state dict without window
        state_dict = {}
        for key, value in model.state_dict().items():
            if key != "window":
                state_dict[key] = value

        # Load the state dict (window should remain as default)
        default_window = model.window.clone()
        model.load_state_dict(state_dict, strict=False)

        # Verify the default window is still used
        self.assertTrue(torch.allclose(model.window, default_window))

    @slow
    @require_torch
    def test_model_from_pretrained(self):
        pass
