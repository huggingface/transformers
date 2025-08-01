# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SmolLM3 model."""

import gc
import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import AutoTokenizer, SmolLM3Config, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    backend_empty_cache,
    is_flaky,
    require_bitsandbytes,
    require_flash_attn,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)
from transformers.utils.import_utils import is_torch_greater_or_equal


if is_torch_available():
    import torch

    from transformers import (
        SmolLM3ForCausalLM,
        SmolLM3ForQuestionAnswering,
        SmolLM3ForSequenceClassification,
        SmolLM3ForTokenClassification,
        SmolLM3Model,
    )


from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
)


class SmolLM3ModelTester(CausalLMModelTester):
    config_class = SmolLM3Config
    if is_torch_available():
        base_model_class = SmolLM3Model
        causal_lm_class = SmolLM3ForCausalLM
        sequence_class = SmolLM3ForSequenceClassification
        token_class = SmolLM3ForTokenClassification
        question_answering_class = SmolLM3ForQuestionAnswering


@require_torch
class SmolLM3ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            SmolLM3Model,
            SmolLM3ForCausalLM,
            SmolLM3ForSequenceClassification,
            SmolLM3ForTokenClassification,
            SmolLM3ForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = SmolLM3ModelTester
    pipeline_model_mapping = (
        {
            "feature-extraction": SmolLM3Model,
            "text-classification": SmolLM3ForSequenceClassification,
            "token-classification": SmolLM3ForTokenClassification,
            "text-generation": SmolLM3ForCausalLM,
            "question-answering": SmolLM3ForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @require_torch_sdpa
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, *args):
        # flaky test_eager_matches_sdpa_inference_24_fp32_pad_left_output_attentions
        return getattr(ModelTesterMixin, self._testMethodName)(self)


@require_torch
class SmolLM3IntegrationTest(unittest.TestCase):
    model_id = "HuggingFaceTB/SmolLM3-3B"

    @slow
    def test_model_3b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = SmolLM3ForCausalLM.from_pretrained(self.model_id, device_map="auto")
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[9.3306, 8.1721, 6.4764, 7.6011, 11.1218, 7.5343, 7.1195, 8.0956]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor(
            [15.7759, 17.6274, 16.3404, 14.5543, 13.1366, 14.2475, 15.8710, 15.6753, 12.3856, 13.0386, 14.0792, 12.7253,
             13.9634, 12.1271, 12.4320, 16.0329, 17.3975, 17.1396, 17.8666, 17.0103, 17.2962, 16.8777, 16.7144, 16.3023,
             16.6084, 12.4649, 12.0723, 14.1148, 14.8239, 15.2733])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-4, atol=1e-4)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_model_3b_generation(self):
        EXPECTED_TEXT_COMPLETION = """Gravity is the force that pulls objects toward the center of the Earth. It is a force that is always present, even"""
        prompt = "Gravity is the force"
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = SmolLM3ForCausalLM.from_pretrained(self.model_id, device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @require_bitsandbytes
    @slow
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_3b_long_prompt(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = SmolLM3ForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            load_in_4bit=True,
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

        del assistant_model
        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, pad_token="<|finetune_right_pad_id|>", padding_side="right"
        )
        EXPECTED_TEXT_COMPLETION = "Gravity is the force that pulls objects toward the center of the Earth. It is a force that is always present, and"
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = SmolLM3ForCausalLM.from_pretrained(
            self.model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompt = ["Gravity is the force"]
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        strict = is_torch_greater_or_equal("2.7.0")  # Due to https://github.com/pytorch/pytorch/issues/150994
        exported_program = convert_and_export_with_cache(model, strict=strict)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
