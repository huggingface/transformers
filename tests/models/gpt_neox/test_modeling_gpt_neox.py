# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GPTNeoX model."""

import unittest

from transformers import AutoTokenizer, GPTNeoXConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        GPTNeoXForCausalLM,
        GPTNeoXForQuestionAnswering,
        GPTNeoXForSequenceClassification,
        GPTNeoXForTokenClassification,
        GPTNeoXModel,
    )


class GPTNeoXModelTester(CausalLMModelTester):
    config_class = GPTNeoXConfig
    if is_torch_available():
        base_model_class = GPTNeoXModel
        causal_lm_class = GPTNeoXForCausalLM
        question_answering_class = GPTNeoXForQuestionAnswering
        sequence_classification_class = GPTNeoXForSequenceClassification
        token_classification_class = GPTNeoXForTokenClassification


@require_torch
class GPTNeoXModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            GPTNeoXModel,
            GPTNeoXForCausalLM,
            GPTNeoXForQuestionAnswering,
            GPTNeoXForSequenceClassification,
            GPTNeoXForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GPTNeoXModel,
            "question-answering": GPTNeoXForQuestionAnswering,
            "text-classification": GPTNeoXForSequenceClassification,
            "text-generation": GPTNeoXForCausalLM,
            "token-classification": GPTNeoXForTokenClassification,
            "zero-shot": GPTNeoXForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = GPTNeoXModelTester

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass


@require_torch
class GPTNeoXLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_gptneox(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
        for checkpointing in [True, False]:
            model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped")

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)
            # The hub repo. is updated on 2023-04-04, resulting in poor outputs.
            # See: https://github.com/huggingface/transformers/pull/24193
            expected_output = "My favorite food is a good old-fashioned, old-fashioned, old-fashioned.\n\nI'm not sure"

            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)

    @slow
    def test_lm_generate_flex_attn_gptneox(self):
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
        for checkpointing in [True, False]:
            model = GPTNeoXForCausalLM.from_pretrained(
                "EleutherAI/pythia-410m-deduped", attn_implementation="flex_attention"
            )
            self.assertTrue(model.config._attn_implementation == "flex_attention")

            if checkpointing:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
            model.to(torch_device)

            inputs = tokenizer("My favorite food is", return_tensors="pt").to(torch_device)
            # The hub repo. is updated on 2023-04-04, resulting in poor outputs.
            # See: https://github.com/huggingface/transformers/pull/24193
            expected_output = "My favorite food is a good old-fashioned, old-fashioned, old-fashioned.\n\nI'm not sure"

            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=20)
            output_str = tokenizer.batch_decode(output_ids)[0]

            self.assertEqual(output_str, expected_output)

    def pythia_integration_test(self):
        model_name_or_path = "EleutherAI/pythia-70m"
        model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to(torch_device)
        EXPECTED_LOGITS = torch.tensor([1069.0000,  228.7500, 1072.0000, 1072.0000, 1069.0000, 1068.0000, 1068.0000, 1071.0000, 1071.0000, 1071.0000, 1073.0000, 1070.0000, 1071.0000, 1075.0000, 1073.0000, 1075.0000, 1074.0000, 1069.0000, 1072.0000, 1071.0000, 1071.0000, 1071.0000, 1070.0000, 1069.0000, 1069.0000, 1069.0000, 1070.0000, 1075.0000, 1073.0000, 1074.0000])  # fmt: skip
        input_ids = [29, 93, 303, 64, 5478, 49651, 10394, 187, 34, 12939, 875]
        # alternative: tokenizer('<|im_start|>system\nA chat between')
        input_ids = torch.as_tensor(input_ids)[None].to(torch_device)
        outputs = model(input_ids)["logits"][:, -1][0, :30]
        torch.testing.assert_close(EXPECTED_LOGITS, outputs, rtol=1e-5, atol=1e-5)
