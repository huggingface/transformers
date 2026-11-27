# Copyright 2026 The Jina-AI and HuggingFace Inc. teams. All rights reserved.
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
"""Testing suite for the PyTorch Jina Embeddings V5 model."""

import unittest

from parameterized import parameterized

from transformers import AutoModel, AutoTokenizer, JinaEmbeddingsV5Config, is_torch_available
from transformers.testing_utils import cleanup, require_peft, require_torch, slow, torch_device

from ...model_tester_config import ModelTesterConfigMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import JinaEmbeddingsV5Model


class JinaEmbeddingsV5ModelTester(ModelTesterConfigMixin):
    if is_torch_available():
        base_model_class = JinaEmbeddingsV5Model

    defaults = {}

    small_config_overrides = {
        "batch_size": 13,
        "seq_length": 7,
        "is_training": True,
        "use_input_mask": True,
        "vocab_size": 99,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 37,
        "hidden_act": "silu",
        "max_position_embeddings": 512,
        "initializer_range": 0.02,
        "pad_token_id": 0,
        "is_decoder": True,
    }

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = None
        if self.use_input_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length]).to(torch_device)
        config = self.get_config()
        return config, input_ids, attention_mask

    def get_config(self):
        return JinaEmbeddingsV5Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            is_decoder=self.is_decoder,
        )

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = JinaEmbeddingsV5Model(config=config)
        model.to(torch_device)
        model.eval()
        result_with_mask = model(input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(
            result_with_mask.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size)
        )
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class JinaEmbeddingsV5ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (JinaEmbeddingsV5Model,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = JinaEmbeddingsV5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=JinaEmbeddingsV5Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_encoder_and_decoder_variants(self):
        encoder_config = JinaEmbeddingsV5Config(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            is_decoder=False,
        )
        self.assertFalse(encoder_config.is_causal)
        self.assertEqual(encoder_config.head_dim, 8)

        encoder_model = JinaEmbeddingsV5Model(encoder_config).to(torch_device).eval()
        self.assertFalse(encoder_model.layers[0].self_attn.is_causal)
        self.assertIsNone(encoder_model.layers[0].self_attn.q_norm)
        self.assertIsNone(encoder_model.layers[0].self_attn.k_norm)

        input_ids_a = torch.tensor([[1, 2, 3, 4]], device=torch_device)
        input_ids_b = torch.tensor([[1, 2, 3, 9]], device=torch_device)
        with torch.no_grad():
            out_a = encoder_model(input_ids=input_ids_a).last_hidden_state
            out_b = encoder_model(input_ids=input_ids_b).last_hidden_state
        self.assertFalse(torch.allclose(out_a[:, 0], out_b[:, 0], atol=1e-5))

        decoder_config = JinaEmbeddingsV5Config(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            is_decoder=True,
        )
        self.assertTrue(decoder_config.is_causal)
        decoder_model = JinaEmbeddingsV5Model(decoder_config).to(torch_device).eval()
        self.assertTrue(decoder_model.layers[0].self_attn.is_causal)
        self.assertIsNotNone(decoder_model.layers[0].self_attn.q_norm)
        self.assertIsNotNone(decoder_model.layers[0].self_attn.k_norm)

        with torch.no_grad():
            dec_a = decoder_model(input_ids=input_ids_a).last_hidden_state
            dec_b = decoder_model(input_ids=input_ids_b).last_hidden_state
        torch.testing.assert_close(dec_a[:, 0], dec_b[:, 0], atol=1e-5, rtol=1e-5)

    @parameterized.expand([(False, True), (True, False)])
    def test_is_causal_override_matches_eager_and_sdpa(self, is_decoder: bool, is_causal: bool):
        config = JinaEmbeddingsV5Config(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            is_decoder=is_decoder,
            is_causal=is_causal,
        )
        eager_model = JinaEmbeddingsV5Model(config).to(torch_device).eval()
        eager_model.set_attn_implementation("eager")
        sdpa_model = JinaEmbeddingsV5Model(config).to(torch_device).eval()
        sdpa_model.load_state_dict(eager_model.state_dict())
        sdpa_model.set_attn_implementation("sdpa")

        input_ids = torch.tensor([[1, 2, 3, 4]], device=torch_device)
        with torch.no_grad():
            eager_out = eager_model(input_ids=input_ids).last_hidden_state
            sdpa_out = sdpa_model(input_ids=input_ids).last_hidden_state
        torch.testing.assert_close(eager_out, sdpa_out, atol=1e-5, rtol=1e-5)


@require_torch
class JinaEmbeddingsV5EncoderModelTest(JinaEmbeddingsV5ModelTest):
    def setUp(self):
        self.model_tester = JinaEmbeddingsV5ModelTester(self, is_decoder=False)
        self.config_tester = ConfigTester(self, config_class=JinaEmbeddingsV5Config, hidden_size=37, is_decoder=False)


@require_torch
class JinaEmbeddingsV5IntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @parameterized.expand(
        [
            ("nano", "jinaai/jina-embeddings-v5-text-nano"),
            ("small", "jinaai/jina-embeddings-v5-text-small"),
        ]
    )
    @require_peft
    @slow
    def test_retrieval(self, _name: str, model_id: str):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(torch_device).eval()
        model.load_adapter(model_id, adapter_name="retrieval", adapter_kwargs={"subfolder": "adapters/retrieval"})
        model.set_adapter("retrieval")

        texts = [
            "Query: What is the capital of France?",
            "Document: Paris is the capital of France.",
            "Document: Bananas are yellow fruits.",
        ]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            attention_mask = inputs["attention_mask"]
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                pooled = hidden[:, -1]
            else:
                seq_lens = attention_mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lens]
            embeds = F.normalize(pooled, p=2, dim=-1)

        sim_pos = (embeds[0] @ embeds[1]).item()
        sim_neg = (embeds[0] @ embeds[2]).item()
        self.assertGreater(sim_pos, sim_neg + 0.2)
