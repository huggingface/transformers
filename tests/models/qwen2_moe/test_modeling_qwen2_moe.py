# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen2MoE model."""

import unittest

import pytest

from transformers import AutoTokenizer, Qwen2MoeConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    run_first,
    run_test_using_subprocess,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        Qwen2MoeForCausalLM,
        Qwen2MoeForQuestionAnswering,
        Qwen2MoeForSequenceClassification,
        Qwen2MoeForTokenClassification,
        Qwen2MoeModel,
    )


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class Qwen2MoeModelTester(CausalLMModelTester):
    config_class = Qwen2MoeConfig
    if is_torch_available():
        base_model_class = Qwen2MoeModel
        causal_lm_class = Qwen2MoeForCausalLM
        sequence_class = Qwen2MoeForSequenceClassification
        token_class = Qwen2MoeForTokenClassification
        question_answering_class = Qwen2MoeForQuestionAnswering


@require_torch
class Qwen2MoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Qwen2MoeModel,
            Qwen2MoeForCausalLM,
            Qwen2MoeForSequenceClassification,
            Qwen2MoeForTokenClassification,
            Qwen2MoeForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Qwen2MoeModel,
            "text-classification": Qwen2MoeForSequenceClassification,
            "token-classification": Qwen2MoeForTokenClassification,
            "text-generation": Qwen2MoeForCausalLM,
            "question-answering": Qwen2MoeForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    test_all_params_have_gradient = False
    model_tester_class = Qwen2MoeModelTester

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
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Qwen2Moe flash attention does not support right padding")

    # Ignore copy
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_experts = 8
        config.expert_interval = 2
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = Qwen2MoeForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_experts))
        torch.testing.assert_close(result.aux_loss.cpu(), torch.tensor(2, dtype=torch.float32), rtol=1e-2, atol=1e-2)

        # First, we make sure that adding padding tokens doesn't change the loss
        # loss(input_ids, attention_mask=None) == loss(input_ids + padding, attention_mask=attention_mask_with_padding)
        pad_length = 1000
        # Add padding tokens (assume that pad_token_id=1) to input_ids
        padding_block = torch.ones(input_ids.shape[0], pad_length, dtype=torch.int32).to(torch_device)
        padded_input_ids = torch.cat((padding_block, input_ids), dim=1)  # this is to simulate padding to the left
        padded_attention_mask = padded_input_ids.ne(1).to(torch_device)

        padded_result = model(padded_input_ids, attention_mask=padded_attention_mask)
        torch.testing.assert_close(result.aux_loss.cpu(), padded_result.aux_loss.cpu(), rtol=1e-4, atol=1e-4)

        # We make sure that the loss of including padding tokens != the loss without padding tokens
        # if attention_mask=None --> we don't exclude padding tokens
        include_padding_result = model(padded_input_ids, attention_mask=None)

        # This is to mimic torch.testing.assert_not_close
        self.assertNotAlmostEqual(include_padding_result.aux_loss.item(), result.aux_loss.item())


@require_torch
class Qwen2MoeIntegrationTest(unittest.TestCase):
    model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = Qwen2MoeForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-MoE-A2.7B", device_map="auto", dtype=torch.float16
            )
        return cls.model

    @classmethod
    def tearDownClass(cls):
        if cls.model is not None:
            del cls.model
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_a2_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-4.2106, -3.6411, -4.9111, -4.2840, -4.9950, -3.4438, -3.5262, -4.1624]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:10]
        EXPECTED_SLICE = torch.tensor([2.3008, -0.6777, -0.1287, -1.4043, -1.7393, -1.7627, -2.0547, -2.4414, -3.0332, -2.1406])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

    @slow
    def test_model_a2_7b_generation(self):
        EXPECTED_TEXT_COMPLETION = """To be or not to be, that is the question. This is the question that has been asked by many people over the"""
        prompt = "To be or not to"
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", use_fast=False)
        model = self.get_model()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    # run this test as the first test within this class and run with a separate process
    # (to avoid potential CPU memory issue caused by `device_map="auto"`.)
    @run_first
    @run_test_using_subprocess
    @slow
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_a2_7b_long_prompt_flash_attn(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = Qwen2MoeForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-MoE-A2.7B",
            device_map="auto",
            dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

    @slow
    def test_model_a2_7b_long_prompt_sdpa(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = assistant_model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        cleanup(torch_device, gc_collect=True)

        EXPECTED_TEXT_COMPLETION = """To be or not to be, that is the question. This is the question that has been asked by many people over the"""
        prompt = "To be or not to"
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", use_fast=False)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_speculative_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer the sl"
        )
        prompt = "To be or not to"
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", use_fast=False)
        model = Qwen2MoeForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", device_map="auto", dtype=torch.float16)
        assistant_model = model
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        set_seed(0)
        generated_ids = model.generate(
            input_ids, max_new_tokens=20, do_sample=True, temperature=0.3, assistant_model=assistant_model
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
