# Copyright 2026 OpenMOSS and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch MOSS-TTS Delay model."""

import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForTextToWaveform,
        MossTTSDelayConfig,
        MossTTSDelayModel,
    )


class MossTTSDelayModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        n_vq=2,
        audio_vocab_size=16,
        text_vocab_size=99,
        hidden_size=32,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_vq = n_vq
        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size
        self.hidden_size = hidden_size

    def get_config(self):
        return MossTTSDelayConfig(
            language_config={
                "vocab_size": self.text_vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "max_position_embeddings": 64,
                "use_cache": True,
                "tie_word_embeddings": False,
            },
            n_vq=self.n_vq,
            audio_vocab_size=self.audio_vocab_size,
            audio_pad_code=self.audio_vocab_size,
            pad_token_id=0,
            im_start_token_id=1,
            im_end_token_id=2,
            audio_start_token_id=3,
            audio_end_token_id=4,
            audio_user_slot_token_id=5,
            audio_assistant_gen_slot_token_id=6,
            audio_assistant_delay_slot_token_id=7,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        text_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(self.batch_size, self.seq_length, 1),
            device=torch_device,
        )
        audio_ids = torch.randint(
            low=0,
            high=config.audio_vocab_size,
            size=(self.batch_size, self.seq_length, config.n_vq),
            device=torch_device,
        )
        input_ids = torch.cat([text_ids, audio_ids], dim=-1)
        attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.bool, device=torch_device)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}

    def create_labels(self, input_ids):
        labels = input_ids.clone()
        labels[:, 0, :] = -100
        return labels


@require_torch
class MossTTSDelayModelTest(unittest.TestCase):
    all_model_classes = (MossTTSDelayModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = MossTTSDelayModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MossTTSDelayConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()
        config = self.model_tester.get_config()
        self.assertEqual(config.model_type, "moss_tts_delay")
        self.assertEqual(config.language_config.model_type, "qwen3")
        self.assertEqual(config.hidden_size, config.language_config.hidden_size)
        self.assertEqual(config.vocab_size, config.language_config.vocab_size)

    def test_model_from_config(self):
        config = self.model_tester.get_config()
        model = MossTTSDelayModel(config).to(torch_device).eval()
        self.assertIsInstance(model, MossTTSDelayModel)
        self.assertEqual(len(model.emb_ext), config.n_vq)
        self.assertEqual(len(model.lm_heads), config.n_vq + 1)

    def test_forward(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = MossTTSDelayModel(config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(len(outputs.logits), config.n_vq + 1)
        self.assertEqual(
            outputs.logits[0].shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.vocab_size),
        )
        for logits in outputs.logits[1:]:
            self.assertEqual(
                logits.shape,
                (self.model_tester.batch_size, self.model_tester.seq_length, config.audio_vocab_size + 1),
            )

    def test_forward_with_labels(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = MossTTSDelayModel(config).to(torch_device).eval()
        labels = self.model_tester.create_labels(input_ids)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.all_sum_losses.shape, (self.model_tester.batch_size, config.n_vq + 1))
        self.assertEqual(outputs.all_token_nums.shape, (self.model_tester.batch_size, config.n_vq + 1))
        self.assertEqual(outputs.channel_losses.shape, (config.n_vq + 1,))

    def test_forward_with_channelwise_loss_weight(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = MossTTSDelayModel(config).to(torch_device).eval()
        labels = self.model_tester.create_labels(input_ids)
        weights = [1.0] + [0.5] * config.n_vq

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            channelwise_loss_weight=weights,
        )

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.sample_losses.shape, (self.model_tester.batch_size,))

    def test_forward_rejects_wrong_input_rank(self):
        config = self.model_tester.get_config()
        model = MossTTSDelayModel(config).to(torch_device).eval()
        input_ids = torch.zeros(
            (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
        )

        with self.assertRaisesRegex(ValueError, "shape should be exactly"):
            model(input_ids=input_ids)

    def test_get_input_embeddings_standard_and_multichannel(self):
        config, input_ids, _ = self.model_tester.prepare_config_and_inputs()
        model = MossTTSDelayModel(config).to(torch_device).eval()

        self.assertIs(model.get_input_embeddings(), model.language_model.get_input_embeddings())
        inputs_embeds = model.get_input_embeddings(input_ids)
        self.assertEqual(
            inputs_embeds.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size),
        )

    def test_auto_classes(self):
        config = self.model_tester.get_config()

        self.assertIsInstance(AutoConfig.for_model("moss_tts_delay"), MossTTSDelayConfig)
        self.assertIsInstance(AutoModel.from_config(config), MossTTSDelayModel)
        self.assertIsInstance(AutoModelForTextToWaveform.from_config(config), MossTTSDelayModel)

    def test_auto_classes_from_pretrained(self):
        config = self.model_tester.get_config()
        model = MossTTSDelayModel(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded_config = AutoConfig.from_pretrained(tmp_dir)
            loaded_model = AutoModel.from_pretrained(tmp_dir)
            loaded_tta_model = AutoModelForTextToWaveform.from_pretrained(tmp_dir)

        self.assertIsInstance(loaded_config, MossTTSDelayConfig)
        self.assertIsInstance(loaded_model, MossTTSDelayModel)
        self.assertIsInstance(loaded_tta_model, MossTTSDelayModel)
