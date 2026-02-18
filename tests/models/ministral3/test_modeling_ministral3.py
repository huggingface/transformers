# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Ministral3 model."""

import gc
import unittest

import pytest

from transformers import AutoTokenizer, Mistral3ForConditionalGeneration, is_torch_available
from transformers.testing_utils import (
    Expectations,
    backend_empty_cache,
    cleanup,
    require_deterministic_for_xpu,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        Ministral3Model,
    )


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class Ministral3ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Ministral3Model


@require_torch
class Ministral3ModelTest(CausalLMModelTest, unittest.TestCase):
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = Ministral3ModelTester

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Ministral3 flash attention does not support right padding")


@require_torch
class Ministral3IntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_3b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = Mistral3ForConditionalGeneration.from_pretrained(
            "mistralai/Ministral-3-3B-Instruct-2512", device_map="auto"
        )
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        # fmt: off
        EXPECTED_MEANS = Expectations(
            {
                ("cuda", None): torch.tensor([[-1.1503, -1.9935, -0.4457, -1.0717, -1.9182, -1.1431, -0.9697, -1.7098]]),
                ("xpu", None): torch.tensor([[-0.9800, -2.4773, -0.2386, -1.0664, -1.8994, -1.3792, -1.0531, -1.8832]]),
            }
        )
        # fmt: on
        EXPECTED_MEAN = EXPECTED_MEANS.get_expectation()
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_deterministic_for_xpu
    def test_model_3b_generation(self):
        # fmt: off
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", None): "My favourite condiment is 100% pure olive oil. It's a staple in my kitchen and I use it in",
                ("xpu", None): "My favourite condiment is iced tea. I love the way it makes me feel. It’s like a little bubble bath for",
            }
        )
        # fmt: on
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            "mistralai/Ministral-3-3B-Instruct-2512", device_map="auto"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(text, EXPECTED_TEXT)

        del model
        backend_empty_cache(torch_device)
        gc.collect()
