# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Kolibri1_1 model."""

import unittest

from parameterized import parameterized

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator_memory, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin


if is_torch_available():
    import torch

    from transformers import Kolibri1_1ForCausalLM, Kolibri1_1Model


class Kolibri1_1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Kolibri1_1Model

    def __init__(self, parent):
        super().__init__(parent=parent)
        # Covers all four sparse roles: an indexer and a reindexer, each producing its K/V or reading another layer's
        self.num_hidden_layers = 5
        self.layer_types = ["sliding_attention"] + ["indexed_attention"] * 4
        self.index_layers = [1, 3]
        self.kv_source_layers = [0, 1, 1, 1, 4]
        # Small budgets, so that keys are selected within the tester's sequence length
        self.index_topk = 2
        self.index_pool_size = 4
        self.index_compression = 2
        self.index_num_heads = 1
        self.index_num_local_tokens = 2
        self.index_num_sink_tokens = 1
        # A dense MLP layer in front, like the checkpoints of Kolibri 1.1
        self.mlp_only_layers = [0]
        self.mlp_only_intermediate_size = 40
        # Two query heads per key head, so that the indexers can score keys with more than one head
        self.num_key_value_heads = 1


@require_torch
class Kolibri1_1ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Kolibri1_1ModelTester
    # The indexer groups queries by their first visible key (`Kolibri1_1Indexer.query_groups`): data-dependent control flow
    test_torch_exportable = False

    @unittest.skip(reason="Sparse layers always overlay their key selection as an attention mask, which flash rejects")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Float8 quantization + TP numerical noise exceeds match threshold")
    def test_tp_generation_quantized(self):
        pass

    @parameterized.expand([("dynamic",), ("static",)])
    def test_cached_decode_matches_full_forward(self, cache_implementation):
        # A decoding step selects keys for one query from the cache, the full forward for all queries at once
        config = self.model_tester.get_config()
        model = Kolibri1_1ForCausalLM(config).to(torch_device).eval()
        input_ids = torch.randint(3, config.vocab_size, (2, 12), device=torch_device)
        outputs = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            min_new_tokens=6,
            max_new_tokens=6,
            do_sample=False,
            cache_implementation=cache_implementation,
            output_logits=True,
            return_dict_in_generate=True,
        )
        with torch.no_grad():
            full_logits = model(outputs.sequences).logits
        # Not bit-identical: both score the same keys, but in matrix products of different shapes
        decode_logits = torch.stack(outputs.logits, dim=1)
        torch.testing.assert_close(decode_logits, full_logits[:, input_ids.shape[1] - 1 : -1], rtol=1e-5, atol=1e-5)


@slow
@require_torch_accelerator_memory(memory=90)
class Kolibri1_1IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    TEST_MODEL_ID = "Aleph-Alpha/Kolibri-1.1"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Dropped again by the mixin's `tearDownClass`.
        cls.model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = Kolibri1_1ForCausalLM.from_pretrained(cls.TEST_MODEL_ID, device_map="auto")
        return cls.model

    def generate_reply(self, content, max_new_tokens):
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = self.get_model()
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def test_model_logits(self):
        input_ids = [452, 22090, 493, 1678, 2459]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        # fmt: off
        EXPECTED_MEAN = torch.tensor([[-2.4225, -4.4516, -4.4104, -5.9657, -5.6187]])
        EXPECTED_SLICE = torch.tensor(
            [-6.1875, -8.1875, -9.8125, -9.4375, -8.2500, -7.5625, -9.0625, -8.1250, -6.3125, -1.2109]
        )
        # fmt: on
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    def test_model_generation(self):
        EXPECTED_TEXT = (
            "Ein Mixture-of-Experts-Modell ist ein neuronales Netzwerk, bei dem für jede Eingabe nur ein kleiner Teil "
            "der Parameter (die „Experten“) aktiviert wird, während ein Router entscheidet, welche Experten für die"
        )
        text = self.generate_reply("Erkläre in einem Satz, was ein Mixture-of-Experts-Modell ist.", 40)
        self.assertEqual(EXPECTED_TEXT, text)

    def test_model_recall_beyond_index_pool(self):
        # About 18k tokens, more than an indexer's pool of 16384 keys: the code is far beyond the sliding window and
        # the local tokens, so the sparse layers must keep its keys both in the pool and among the selected keys
        filler = "Der Himmel ist blau. Das Gras ist grün. Die Sonne scheint. " * 1300
        prompt = f"Merke dir: Der Geheimcode lautet 4721.\n\n{filler}\nWie lautet der Geheimcode? Antworte nur mit der Zahl."
        text = self.generate_reply(prompt, 10)
        self.assertEqual("4721", text)
