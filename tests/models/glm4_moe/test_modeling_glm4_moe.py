# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GLM-4-MoE model."""

import unittest

import pytest

from transformers import AutoTokenizer, Glm4MoeConfig, is_torch_available, set_seed
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_torch_large_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        Glm4MoeForCausalLM,
        Glm4MoeForQuestionAnswering,
        Glm4MoeForSequenceClassification,
        Glm4MoeForTokenClassification,
        Glm4MoeModel,
    )
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class Glm4MoeModelTester(CausalLMModelTester):
    config_class = Glm4MoeConfig
    if is_torch_available():
        base_model_class = Glm4MoeModel
        causal_lm_class = Glm4MoeForCausalLM
        sequence_class = Glm4MoeForSequenceClassification
        token_class = Glm4MoeForTokenClassification
        question_answering_class = Glm4MoeForQuestionAnswering


@require_torch
class Glm4MoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Glm4MoeModel,
            Glm4MoeForCausalLM,
            Glm4MoeForSequenceClassification,
            Glm4MoeForTokenClassification,
            Glm4MoeForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Glm4MoeModel,
            "text-classification": Glm4MoeForSequenceClassification,
            "token-classification": Glm4MoeForTokenClassification,
            "text-generation": Glm4MoeForCausalLM,
            "question-answering": Glm4MoeForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    test_all_params_have_gradient = False
    model_tester_class = Glm4MoeModelTester

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
        self.skipTest(reason="Glm4Moe flash attention does not support right padding")

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
        model = Glm4MoeForCausalLM(config)
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


# Run on runners with larger accelerators (for example A10 instead of T4) with a lot of CPU RAM (e.g. g5-12xlarge)
@require_torch_multi_accelerator
@require_torch_large_accelerator
@require_torch
class Glm4MoeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = Glm4MoeForCausalLM.from_pretrained(
                "/model/GLM-4-MoE-100B-A10B", device_map="auto", load_in_4bit=True
            )

        return cls.model

    @require_bitsandbytes
    @slow
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_speculative_generation(self):
        EXPECTED_TEXT_COMPLETION = (
            "To be or not to be: the role of the liver in the pathogenesis of obesity and type 2 diabetes.\nThe"
        )
        prompt = "To be or not to"
        tokenizer = AutoTokenizer.from_pretrained("/model/GLM-4-MoE-100B-A10B", use_fast=False)
        model = self.get_model()
        assistant_model = model
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        set_seed(0)
        generated_ids = model.generate(
            input_ids, max_new_tokens=20, do_sample=True, temperature=0.3, assistant_model=assistant_model
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
