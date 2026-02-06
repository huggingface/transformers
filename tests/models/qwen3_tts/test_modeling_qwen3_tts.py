# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

if is_torch_available():
    import torch
    from transformers import Qwen3TTSTalkerConfig, Qwen3TTSTalkerForConditionalGeneration


@require_torch
class Qwen3TTSTalkerModelTester:
    """Tester for Qwen3TTSTalkerModel"""

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=16,
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_code_groups=4,
        intermediate_size=512,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_code_groups = num_code_groups
        self.intermediate_size = intermediate_size

    def get_config(self):
        return Qwen3TTSTalkerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            use_sliding_window=False,
            num_code_groups=self.num_code_groups,
            code_predictor_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "max_position_embeddings": 2048,
                "num_code_groups": self.num_code_groups,
                "use_sliding_window": False,
                "pad_token_id": None,
            },
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # Codec input tokens
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length), device=torch_device)
        # Attention mask (no padding in this test)
        attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        # Input embeddings for prefill (batch, seq, hidden)
        inputs_embeds = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size, device=torch_device, dtype=torch.float32
        )
        return config, input_ids, attention_mask, inputs_embeds

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, inputs_embeds = self.prepare_config_and_inputs()
        inputs_dict = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_qwen3tts_forward(self, config, input_ids, attention_mask, inputs_embeds):
        batch_size, seq_length = inputs_embeds.shape[:2]

        # Test forward pass
        model = Qwen3TTSTalkerForConditionalGeneration(config).to(torch_device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # Check output shapes
        self.parent.assertIsNotNone(outputs.logits)
        self.parent.assertEqual(outputs.logits.shape, (batch_size, seq_length, config.vocab_size))
        self.parent.assertIsNone(outputs.loss)
        self.parent.assertIsNotNone(outputs.past_hidden)
        self.parent.assertEqual(outputs.past_hidden.shape, (batch_size, 1, config.hidden_size))

        # Test with labels (training mode)
        model.train()
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=torch_device)
        outputs_with_loss = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

        self.parent.assertIsNotNone(outputs_with_loss.loss)
        self.parent.assertEqual(outputs_with_loss.loss.dim(), 0)  # Scalar loss

        # Test output_attentions
        model.eval()
        with torch.no_grad():
            outputs_with_attn = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=True
            )
        self.parent.assertIsNotNone(outputs_with_attn.attentions)

        # Test output_hidden_states
        with torch.no_grad():
            outputs_with_hidden = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        self.parent.assertIsNotNone(outputs_with_hidden.hidden_states)


@require_torch
class Qwen3TTSTalkerModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Qwen3TTSTalkerModelTester(self)

    def test_config(self):
        config = self.model_tester.get_config()
        self.assertIsNotNone(config)

    def test_prepare_config_and_inputs(self):
        config, input_ids, attention_mask, inputs_embeds = self.model_tester.prepare_config_and_inputs()
        self.assertEqual(input_ids.shape, (self.model_tester.batch_size, self.model_tester.seq_length))
        self.assertEqual(attention_mask.shape, (self.model_tester.batch_size, self.model_tester.seq_length))
        self.assertEqual(
            inputs_embeds.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size),
        )

    def test_forward(self):
        config, input_ids, attention_mask, inputs_embeds = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_qwen3tts_forward(config, input_ids, attention_mask, inputs_embeds)
