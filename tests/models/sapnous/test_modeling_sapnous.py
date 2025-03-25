# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. Team and AIRAS Inc. Team. All rights reserved.
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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .modeling_sapnous import SapnousT1ForCausalLM
from .configuration_sapnous import SapnousT1Config

class TestSapnousModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = SapnousT1Config(
            vocab_size=32000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        cls.model = SapnousT1ForCausalLM(cls.config)

    def test_model_forward(self):
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        outputs = self.model(input_ids)
        
        self.assertIsNotNone(outputs)
        self.assertTrue(hasattr(outputs, 'logits'))
        self.assertEqual(outputs.logits.shape, (1, 10, self.config.vocab_size))

    def test_weight_tying(self):
        self.model.tie_weights()
        self.assertTrue(torch.equal(self.model.lm_head.weight, self.model.model.embeddings.weight))

    def test_auto_model_registration(self):
        model = AutoModelForCausalLM.from_config(self.config)
        self.assertIsInstance(model, SapnousT1ForCausalLM)

    def test_vision_embeddings(self):
        # Test vision input processing
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, 10))
        
        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values)
        self.assertIsNotNone(outputs)
        self.assertTrue(hasattr(outputs, 'logits'))
        
        # Vision input should increase sequence length
        expected_seq_length = 10 + (224 // 16) ** 2 + 1  # text_len + num_patches + cls_token
        self.assertEqual(outputs.logits.shape, (batch_size, expected_seq_length, self.config.vocab_size))
    
    def test_attention_mask(self):
        # Test attention mask handling
        batch_size = 2
        seq_length = 15
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        attention_mask[:, -5:] = 0  # Mask out last 5 tokens
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertIsNotNone(outputs)
        self.assertEqual(outputs.logits.shape, (batch_size, seq_length, self.config.vocab_size))
    
    def test_generation_with_vision(self):
        # Test text generation with vision input
        pixel_values = torch.randn(1, 3, 224, 224)
        input_ids = torch.randint(0, self.config.vocab_size, (1, 5))
        
        outputs = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_length=20,
            num_beams=1
        )
        
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.dim(), 2)
        self.assertTrue(outputs.size(1) <= 20)

if __name__ == '__main__':
    unittest.main()