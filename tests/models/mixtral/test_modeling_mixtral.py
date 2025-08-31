# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Mixtral model."""

import unittest

import pytest

from transformers import MixtralConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        MixtralForCausalLM,
        MixtralForQuestionAnswering,
        MixtralForSequenceClassification,
        MixtralForTokenClassification,
        MixtralModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MixtralModelTester(CausalLMModelTester):
    config_class = MixtralConfig
    if is_torch_available():
        base_model_class = MixtralModel
        causal_lm_class = MixtralForCausalLM
        sequence_class = MixtralForSequenceClassification
        token_class = MixtralForTokenClassification
        question_answering_class = MixtralForQuestionAnswering


@require_torch
class MistralModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            MixtralModel,
            MixtralForCausalLM,
            MixtralForSequenceClassification,
            MixtralForTokenClassification,
            MixtralForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": MixtralModel,
            "text-classification": MixtralForSequenceClassification,
            "token-classification": MixtralForTokenClassification,
            "text-generation": MixtralForCausalLM,
            "question-answering": MixtralForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    model_tester_class = MixtralModelTester

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
        self.skipTest(reason="Mistral flash attention does not support right padding")

    # Ignore copy
    def test_load_balancing_loss(self):
        r"""
        Let's make sure we can actually compute the loss and do a backward on it.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.num_local_experts = 8
        config.output_router_logits = True
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = MixtralForCausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(result.router_logits[0].shape, (91, config.num_local_experts))
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
class MixtralIntegrationTest(unittest.TestCase):
    @slow
    @require_torch_accelerator
    def test_small_model_logits(self):
        model_id = "hf-internal-testing/Mixtral-tiny"
        dummy_input = torch.LongTensor([[0, 1, 0], [0, 1, 0]]).to(torch_device)

        model = MixtralForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(torch_device)
        # TODO: might need to tweak it in case the logits do not match on our daily runners
        # these logits have been obtained with the original megablocks implementation.
        # ("cuda", 8) for A100/A10, and ("cuda", 7) for T4
        # considering differences in hardware processing and potential deviations in output.
        # fmt: off
        EXPECTED_LOGITS = Expectations(
            {
                ("cuda", 7): torch.Tensor([[0.1640, 0.1621, 0.6093], [-0.8906, -0.1640, -0.6093], [0.1562, 0.1250, 0.7226]]).to(torch_device),
                ("cuda", 8): torch.Tensor([[0.1631, 0.1621, 0.6094], [-0.8906, -0.1621, -0.6094], [0.1572, 0.1270, 0.7227]]).to(torch_device),
                ("rocm", 9): torch.Tensor([[0.1641, 0.1621, 0.6094], [-0.8906, -0.1631, -0.6094], [0.1572, 0.1260, 0.7227]]).to(torch_device),
            }
        )
        # fmt: on
        expected_logit = EXPECTED_LOGITS.get_expectation()

        with torch.no_grad():
            logits = model(dummy_input).logits

        logits = logits.float()

        torch.testing.assert_close(logits[0, :3, :3], expected_logit, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits[1, :3, :3], expected_logit, atol=1e-3, rtol=1e-3)

    @slow
    @require_torch_accelerator
    def test_small_model_logits_batched(self):
        model_id = "hf-internal-testing/Mixtral-tiny"
        dummy_input = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 2, 3], [1, 1, 2, 3, 4, 5, 6, 7, 8]]).to(torch_device)
        attention_mask = dummy_input.ne(0).to(torch.long)

        model = MixtralForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(torch_device)

        # TODO: might need to tweak it in case the logits do not match on our daily runners
        #
        # ("cuda", 8) for A100/A10, and ("cuda", 7) for T4.
        #
        # considering differences in hardware processing and potential deviations in generated text.

        EXPECTED_LOGITS_LEFT_UNPADDED = Expectations(
            {
                ("xpu", 3): [[0.2236, 0.5195, -0.3828], [0.8203, -0.2295, 0.6055], [0.2676, -0.7070, 0.2461]],
                ("cuda", 7): [[0.2236, 0.5195, -0.3828], [0.8203, -0.2275, 0.6054], [0.2656, -0.7070, 0.2460]],
                ("cuda", 8): [[0.2217, 0.5195, -0.3828], [0.8203, -0.2295, 0.6055], [0.2676, -0.7109, 0.2461]],
                ("rocm", 9): [[0.2236, 0.5195, -0.3828], [0.8203, -0.2285, 0.6055], [0.2637, -0.7109, 0.2451]],
            }
        )
        expected_left_unpadded = torch.tensor(EXPECTED_LOGITS_LEFT_UNPADDED.get_expectation(), device=torch_device)

        EXPECTED_LOGITS_RIGHT_UNPADDED = Expectations(
            {
                ("xpu", 3): [[0.2178, 0.1270, -0.1641], [-0.3496, 0.2988, -1.0312], [0.0693, 0.7930, 0.8008]],
                ("cuda", 7): [[0.2167, 0.1269, -0.1640], [-0.3496, 0.2988, -1.0312], [0.0688, 0.7929, 0.8007]],
                ("cuda", 8): [[0.2178, 0.1260, -0.1621], [-0.3496, 0.2988, -1.0312], [0.0693, 0.7930, 0.8008]],
                ("rocm", 9): [[0.2197, 0.1250, -0.1611], [-0.3516, 0.3008, -1.0312], [0.0684, 0.7930, 0.8008]],
            }
        )
        expected_right_unpadded = torch.tensor(EXPECTED_LOGITS_RIGHT_UNPADDED.get_expectation(), device=torch_device)

        with torch.no_grad():
            logits = model(dummy_input, attention_mask=attention_mask).logits
        logits = logits.float()

        torch.testing.assert_close(
            logits[0, -3:, -3:],
            expected_left_unpadded,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.testing.assert_close(
            logits[1, -3:, -3:],
            expected_right_unpadded,
            atol=1e-3,
            rtol=1e-3,
        )
