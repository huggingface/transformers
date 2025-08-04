# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch GptOss model."""

import unittest
import json
import os


import pytest
from parameterized import parameterized

from tests.tensor_parallel.test_tensor_parallel import TensorParallelTestBase
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GptOssConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        GptOssForCausalLM,
        GptOssModel,
    )


class GptOssModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = GptOssConfig
        base_model_class = GptOssModel
        causal_lm_class = GptOssForCausalLM

    pipeline_model_mapping = (
        {
            "feature-extraction": GptOssModel,
            "text-generation": GptOssForCausalLM,
        }
        if is_torch_available()
        else {}
    )


@require_torch
class GptOssModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (GptOssModel, GptOssForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": GptOssModel,
            "text-generation": GptOssForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = GptOssModelTester

    def setUp(self):
        self.model_tester = GptOssModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GptOssConfig, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("GptOss's forcefully disables sdpa due to Sink")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @unittest.skip("GptOss's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("GptOss has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("GptOss has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("GptOss has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("GptOss has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("GptOss has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesn't work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("GptOss has HybridCache which auto-compiles. Compile and FA2 don't work together.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip("GptOss eager/FA2 attention outputs are expected to be different")
    def test_flash_attn_2_equivalence(self):
        pass


RESULTS_PATH = os.path.join(
    os.path.dirname(__file__).split("transformers")[0], "tests/fixtures/gpt_oss/integration_tests.json"
)


@slow
@require_torch_accelerator
class GptOssIntegrationTest(unittest.TestCase):
    input_text = [
        "Roses are red, violets",
        "How are you? Tell me the name of the president of",
    ]

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @staticmethod
    def load_and_forward(model_id, attn_implementation, input_text, **pretrained_kwargs):
        if not isinstance(attn_implementation, list):
            attn_implementation = [attn_implementation]

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **pretrained_kwargs).to(
            torch_device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        outputs = []
        for attn in attn_implementation:
            model.set_attn_implementation(attn)
            inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(torch_device)
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            output_text = tokenizer.batch_decode(output, skip_special_tokens=False)
            outputs.append(output_text)
        return outputs

    @require_read_token
    @parameterized.expand(
        [
            # (quantized, model, kernels, attn_impl, mode)
            (False, "120b", False, "eager", "eval"),
            (False, "120b", False, "eager", "train"),
            (False, "120b", False, "ft-hf-o-c/vllm-flash-attn3", "eval"),
            (False, "120b", False, "ft-hf-o-c/vllm-flash-attn3", "train"),
            (False, "120b", True, "eager", "eval"),
            (False, "120b", True, "eager", "train"),
            (False, "120b", True, "ft-hf-o-c/vllm-flash-attn3", "eval"),
            (False, "120b", True, "ft-hf-o-c/vllm-flash-attn3", "train"),
            (True, "120b", False, "eager", "eval"),
            (True, "120b", False, "eager", "train"),
            (True, "120b", False, "ft-hf-o-c/vllm-flash-attn3", "eval"),
            (True, "120b", False, "ft-hf-o-c/vllm-flash-attn3", "train"),
            (True, "120b", True, "eager", "eval"),
            (True, "120b", True, "eager", "train"),
            (True, "120b", True, "ft-hf-o-c/vllm-flash-attn3", "eval"),
            (True, "120b", True, "ft-hf-o-c/vllm-flash-attn3", "train"),
            (False, "20b", False, "eager", "eval"),
            (False, "20b", False, "eager", "train"),
            (False, "20b", False, "ft-hf-o-c/vllm-flash-attn3", "eval"),
            (False, "20b", False, "ft-hf-o-c/vllm-flash-attn3", "train"),
            (False, "20b", True, "eager", "eval"),
            (False, "20b", True, "eager", "train"),
            (False, "20b", True, "ft-hf-o-c/vllm-flash-attn3", "eval"),
            (False, "20b", True, "ft-hf-o-c/vllm-flash-attn3", "train"),
            (True, "20b", False, "eager", "eval"),
            (True, "20b", False, "eager", "train"),
            (True, "20b", False, "ft-hf-o-c/vllm-flash-attn3", "eval"),
            (True, "20b", False, "ft-hf-o-c/vllm-flash-attn3", "train"),
            (True, "20b", True, "eager", "eval"),
            (True, "20b", True, "eager", "train"),
            (True, "20b", True, "ft-hf-o-c/vllm-flash-attn3", "eval"),
            (True, "20b", True, "ft-hf-o-c/vllm-flash-attn3", "train"),
        ]
    )
    def test_model_outputs(self, quantized, model, kernels, attn_impl, mode):
        model_id = f"/fsx/vb/new-oai/gpt-oss-{model}-trfs"
        output_text = self.load_and_forward(
            model_id,
            attn_impl,
            self.input_text,
            use_kernels=kernels,
        )

        # Flatten outputs if needed (since we loop over attn_impl)
        if isinstance(output_text[0], list):
            output_text = output_text[0]

        result_entry = {
            "quantized": quantized,
            "model": model,
            "kernels": kernels,
            "attn_impl": attn_impl,
            "mode": mode,
            "outputs": output_text,
        }

        # Append to result.json for comparison
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                results = json.load(f)
        else:
            results = []

        results.append(result_entry)

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        # Optionally, assert that at least output shape is correct
        self.assertIsInstance(output_text, list)
        self.assertTrue(all(isinstance(x, str) for x in output_text))


@slow
@require_torch_multi_accelerator
class GptOssTPTest(TensorParallelTestBase):
    def test_model_training(self):
        self.run_tensor_parallel_test(
            model_id="/fsx/vb/new-oai/gpt-oss-20b-trfs",
            mode="training",
            expected_output="you with something?",
        )
        self.run_tensor_parallel_test(
            model_id="/fsx/vb/new-oai/gpt-oss-120b-trfs",
            mode="training",
            expected_output="you with something?",
        )

    def test_model_generate(self):
        self.run_tensor_parallel_test(
            model_id="/fsx/vb/new-oai/gpt-oss-20b-trfs",
            mode="generate",
            expected_output="with something",
        )
        self.run_tensor_parallel_test(
            model_id="/fsx/vb/new-oai/20b-converted-quantized",
            mode="generate",
            expected_output="with something",
        )
