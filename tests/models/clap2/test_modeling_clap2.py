# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch CLAP2 model."""

import unittest

import numpy as np

from transformers import Clap2AudioConfig, Clap2Config, Clap2TextConfig
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import Clap2AudioModel, Clap2Model, Clap2Processor, Clap2TextModel


class Clap2AudioModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        num_mel_bins=16,
        patch_size=4,
        num_time_patches=5,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_mel_bins = num_mel_bins
        self.patch_size = patch_size
        self.num_freq_patches = num_mel_bins // patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.is_training = is_training
        self.seq_length = self.num_freq_patches * num_time_patches
        self.encoder_seq_length = self.seq_length + 1  # the register token

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.seq_length, self.patch_size**2])
        # the last clip is two time patches shorter
        input_features_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.long, device=torch_device)
        input_features_mask[-1, -2 * self.num_freq_patches :] = 0
        return self.get_config(), input_features, input_features_mask

    def get_config(self):
        return Clap2AudioConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_mel_bins=self.num_mel_bins,
            patch_size=self.patch_size,
        )

    def create_and_check_model(self, config, input_features, input_features_mask):
        model = Clap2AudioModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_features, input_features_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_features, input_features_mask = self.prepare_config_and_inputs()
        return config, {"input_features": input_features, "input_features_mask": input_features_mask}


class Clap2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        # right padding: the last text is two tokens shorter
        attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.long, device=torch_device)
        attention_mask[-1, -2:] = 0
        return self.get_config(), input_ids, attention_mask

    def get_config(self):
        return Clap2TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=512,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
        )

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = Clap2TextModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


class Clap2ModelTester:
    def __init__(self, parent, is_training=True):
        self.parent = parent
        self.text_model_tester = Clap2TextModelTester(parent)
        self.audio_model_tester = Clap2AudioModelTester(parent)
        self.batch_size = self.text_model_tester.batch_size
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        _, input_features, input_features_mask = self.audio_model_tester.prepare_config_and_inputs()
        return self.get_config(), input_ids, attention_mask, input_features, input_features_mask

    def get_config(self):
        return Clap2Config(
            text_config=self.text_model_tester.get_config().to_dict(),
            audio_config=self.audio_model_tester.get_config().to_dict(),
            projection_dim=24,
            # unscaled logits, so that the fp16 eager/sdpa comparison is on the scale of the cosine similarities
            logit_scale_init_value=0.0,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, input_features, input_features_mask):
        model = Clap2Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, input_features, attention_mask, input_features_mask, return_loss=True)
        batch_size = (self.audio_model_tester.batch_size, self.text_model_tester.batch_size)
        self.parent.assertEqual(result.logits_per_audio.shape, batch_size)
        self.parent.assertEqual(result.logits_per_text.shape, batch_size[::-1])
        self.parent.assertEqual(result.audio_embeds.shape, (batch_size[0], 24))
        self.parent.assertTrue(torch.isfinite(result.loss))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, input_features, input_features_mask = self.prepare_config_and_inputs()
        return config, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
            "return_loss": False,
        }


@require_torch
class Clap2AudioModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Clap2AudioModel,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = Clap2AudioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Clap2AudioConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    @unittest.skip(reason="SDPA can't dispatch on flash with not None `attention_mask`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Clap2AudioModel has no input embeddings layer")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Clap2AudioModel takes patches, not inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Clap2AudioModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="The audio RoPE takes the patch positions of a padded batch, not arbitrary coordinates")
    def test_vision_axial_rope(self):
        pass

    def test_padding_patches_are_ignored(self):
        config, input_features, input_features_mask = self.model_tester.prepare_config_and_inputs()
        model = Clap2AudioModel(config).to(torch_device).eval()
        num_valid = int(input_features_mask[-1].sum())
        with torch.no_grad():
            batched = model(input_features, input_features_mask).pooler_output[-1]
            alone = model(input_features[-1:, :num_valid]).pooler_output[0]
        torch.testing.assert_close(batched, alone, rtol=1e-5, atol=1e-5)


@require_torch
class Clap2TextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Clap2TextModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Clap2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Clap2TextConfig, hidden_size=48)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    @unittest.skip(reason="SDPA can't dispatch on flash with not None `attention_mask`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Clap2TextModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="Clap2TextModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass


@require_torch
class Clap2ModelTest(ModelTesterMixin, unittest.TestCase):
    additional_model_inputs = ["input_features", "input_features_mask"]
    all_model_classes = (Clap2Model,) if is_torch_available() else ()
    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Clap2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Clap2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    @unittest.skip(reason="SDPA can't dispatch on flash with not None `attention_mask`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Flash attention equivalence is tested in individual model tests")
    def flash_attn_inference_equivalence(self, attn_implementation, padding_side, atol=4e-2, rtol=4e-2):
        pass

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="CLAP2 takes input_ids and audio patches, not inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="The audio RoPE takes the patch positions of a padded batch, not arbitrary coordinates")
    def test_vision_axial_rope(self):
        pass


@require_torch
class Clap2ModelIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    @slow
    def test_inference(self):
        model_id = "laion/clap2-xxs-audioxl-s1B"
        model = Clap2Model.from_pretrained(model_id).to(torch_device)
        processor = Clap2Processor.from_pretrained(model_id)
        time = np.arange(3 * 32000) / 32000
        audio = (0.5 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        texts = ["a pure sine tone", "a dog barking", "people applauding"]
        inputs = processor(text=texts, audio=audio, sampling_rate=32000, padding="max_length", return_tensors="pt").to(
            torch_device
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # cosine similarities of OpenCLIP's `hf-hub:laion/clap2-xxs-audioxl-s1B` on the same inputs
        expected = torch.tensor([[0.16627082, 0.15809676, 0.09023137]], device=torch_device)
        torch.testing.assert_close(outputs.audio_embeds @ outputs.text_embeds.T, expected, rtol=1e-4, atol=1e-4)
