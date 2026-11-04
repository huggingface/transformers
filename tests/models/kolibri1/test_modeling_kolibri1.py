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
"""Testing suite for the PyTorch Kolibri1 model."""

import unittest

from huggingface_hub.errors import StrictDataclassClassValidationError

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator_memory,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin


if is_torch_available():
    import torch

    from transformers import Kolibri1Config, Kolibri1ForCausalLM, Kolibri1Model


class Kolibri1ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Kolibri1Model

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.layer_types = ["sliding_attention", "full_attention"]


@require_torch
class Kolibri1ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Kolibri1ModelTester

    def test_sliding_window_required(self):
        with self.assertRaises(StrictDataclassClassValidationError):
            Kolibri1Config(sliding_window=None)

    def test_no_load_balancing_loss(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Kolibri1ForCausalLM(config).to(torch_device).eval()
        input_ids = input_dict["input_ids"]

        with torch.no_grad():
            plain = model(input_ids, labels=input_ids)
            with_router_logits = model(input_ids, labels=input_ids, output_router_logits=True)

        self.assertIsNone(with_router_logits.aux_loss)
        self.assertEqual(len(with_router_logits.router_logits), config.num_hidden_layers)
        torch.testing.assert_close(with_router_logits.loss, plain.loss)

    @require_torch_gpu
    def test_router_with_activations_in_another_dtype(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Kolibri1ForCausalLM(config).to(torch_device, torch.bfloat16).eval()
        inputs_embeds = model.get_input_embeddings()(input_dict["input_ids"]).float()

        with torch.no_grad(), torch.autocast(torch_device, dtype=torch.bfloat16):
            logits = model(inputs_embeds=inputs_embeds).logits

        self.assertFalse(logits.isnan().any())


@slow
@require_torch_accelerator_memory(memory=180)
class Kolibri1IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    TEST_MODEL_ID = "Aleph-Alpha/Kolibri-1-BF16"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Dropped again by the mixin's `tearDownClass`.
        cls.model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = Kolibri1ForCausalLM.from_pretrained(
                cls.TEST_MODEL_ID, device_map="auto", experts_implementation="eager"
            )
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
        EXPECTED_MEAN = Expectations(
            {
                ("cuda", None): torch.tensor([[-3.6648, -4.0749, -3.8838, -4.3631, -3.9150]]),
                ("rocm", (9, 4)): torch.tensor([[-3.6640, -4.0664, -3.6125, -4.4985, -4.4048]]),
            }
        ).get_expectation()
        EXPECTED_SLICE = Expectations(
            {
                ("cuda", None): torch.tensor(
                    [-2.0000, -7.0938, -8.3750, -9.5000, -8.6875, -8.6875, -8.9375, -7.9062, -4.7188, 1.8594]
                ),
                ("rocm", (9, 4)): torch.tensor(
                    [-2.0000, -7.0625, -8.3125, -9.5000, -8.6875, -8.6250, -8.8750, -7.8750, -4.6875, 1.8516]
                ),
            }
        ).get_expectation()
        # fmt: on
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, atol=1e-2, rtol=1e-2)

    def test_model_generation(self):
        EXPECTED_TEXT = (
            "Ein Mixture-of-Experts-Modell ist ein neuronales Netz, das aus mehreren spezialisierten Experten "
            "besteht, von denen bei jeder Vorhersage nur ein kleiner Teil aktiviert wird, um Effizienz und "
            "Skalierbarkeit zu erhöhen."
        )
        text = self.generate_reply("Erkläre in einem Satz, was ein Mixture-of-Experts-Modell ist.", 40)
        self.assertEqual(EXPECTED_TEXT, text)

    def test_model_recall_beyond_sliding_window(self):
        filler = "Der Himmel ist blau. Das Gras ist grün. Die Sonne scheint. " * 150
        prompt = f"Merke dir: Der Geheimcode lautet 4721.\n\n{filler}\nWie lautet der Geheimcode? Antworte nur mit der Zahl."
        text = self.generate_reply(prompt, 10)
        self.assertEqual("4721", text)
