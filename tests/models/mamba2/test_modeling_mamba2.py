# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import unittest

from transformers import AutoTokenizer, Mamba2Config, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        Mamba2ForCausalLM,
        Mamba2Model,
    )
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer


class Mamba2ConfigTester(ConfigTester):
    def _create_config(self, hidden_size: int, num_heads: int, expand: int, head_dim: int):
        _input_dict = self.inputs_dict.copy()
        _input_dict["hidden_size"] = hidden_size
        _input_dict["num_heads"] = num_heads
        _input_dict["expand"] = expand
        _input_dict["head_dim"] = head_dim
        return self.config_class(**_input_dict)

    def test_hidden_size_compatibility(self):
        self._create_config(hidden_size=2, num_heads=2, expand=2, head_dim=2)
        self._create_config(hidden_size=4, num_heads=4, expand=2, head_dim=2)
        self._create_config(hidden_size=2, num_heads=4, expand=4, head_dim=2)
        with self.parent.assertRaises(ValueError):
            self._create_config(hidden_size=2, num_heads=4, expand=2, head_dim=4)
        with self.parent.assertRaises(ValueError):
            self._create_config(hidden_size=4, num_heads=2, expand=4, head_dim=2)

    def run_common_tests(self):
        self.test_hidden_size_compatibility()
        return super().run_common_tests()


class Mamba2ModelTester(CausalLMModelTester):
    config_class = Mamba2Config
    if is_torch_available():
        base_model_class = Mamba2Model
        causal_lm_class = Mamba2ForCausalLM


@require_torch
class Mamba2ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (Mamba2Model, Mamba2ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Mamba2Model, "text-generation": Mamba2ForCausalLM} if is_torch_available() else {}
    )
    model_tester_class = Mamba2ModelTester

    @unittest.skip(reason="Mamba 2 weights are not tied")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="A large mamba2 would be necessary (and costly) for that")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@require_torch
@slow
@require_read_token
class Mamba2IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_id = "mistralai/Mamba-Codestral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, from_slow=True, legacy=False)
        self.prompt = ("[INST]Write a hello world program in C++.",)

    @require_read_token
    @slow
    @require_torch
    def test_simple_generate(self):
        """
        Simple generate test to avoid regressions.
        Note: state-spaces (cuda) implementation and pure torch implementation
        have irreconciliable differences as of now, which will cause this test to fail
        in an environment with state-spaces installed.
        """
        tokenizer = self.tokenizer
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        model.to(torch_device)
        input_ids = tokenizer("[INST]Write a hello world program in C++.[/INST]", return_tensors="pt")["input_ids"].to(
            torch_device
        )

        out = model.generate(input_ids, do_sample=False, use_cache=True, max_new_tokens=30)
        output_sentence = tokenizer.decode(out[0])
        ground_truth_sentences = Expectations(
            {
                ("xpu", 3): """<s>[INST]Write a hello world program in C++.[/INST] Sure, here is a simple "Hello, World!" program written in C++:\n\n```cpp\n#include <iostream>\n""",
                ("cuda", 7): """<s>[INST]Write a hello world program in C++.[/INST] Sure, here is a simple "Hello, World!" program in C++:\n\n```cpp\n#include <iostream>\n\n""",
            }
        )  # fmt: skip
        ground_truth_sentence = ground_truth_sentences.get_expectation()
        self.assertEqual(output_sentence, ground_truth_sentence)

    @require_read_token
    @slow
    @require_torch_accelerator
    def test_batched_equivalence_with_cache(self):
        """
        Verifies that batched generation matches individual generation.
        Important because of the specific caching mechanism + statefulness of mamba model.
        Depending on precision and devices, differences can be observed from generation to generation.
        """
        tokenizer = self.tokenizer
        prompt = [
            "[INST]Write C#.[/INST]",
            "[INST]Write a hello world in C++.[/INST]",
            "[INST] Write a simple Fibonacci number computation function in Rust that does memoization, with comments, in safe Rust.[/INST]",
        ]

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # batched generation
        tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest").to(torch_device)
        batched_gen = model.generate(**tokenized_prompts, max_new_tokens=30, use_cache=True)
        batched_output = tokenizer.batch_decode(batched_gen, skip_special_tokens=True)

        # individual generation

        for index_gen, individual_prompt in enumerate(prompt):
            inputs = tokenizer(individual_prompt, return_tensors="pt", padding="longest").to(torch_device)
            individual_gen = model.generate(**inputs, max_new_tokens=30, use_cache=True)
            individual_output = tokenizer.batch_decode(individual_gen, skip_special_tokens=True)[0]
            self.assertEqual(individual_output[:100], batched_output[index_gen][:100])

    @require_read_token
    @slow
    @require_torch_accelerator
    def test_batched_equivalence_without_cache(self):
        """
        Verifies that batched generation matches individual generation without cache.
        Important because of the specific caching mechanism + statefulness of mamba model.
        Depending on precision and devices, differences can be observed from generation to generation.
        """
        tokenizer = self.tokenizer
        prompt = [
            "[INST]Write C#.[/INST]",
            "[INST]Write a hello world in C++.[/INST]",
            "[INST] Write a simple Fibonacci number computation function in Rust that does memoization, with comments, in safe Rust.[/INST]",
        ]

        model = Mamba2ForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16).to(torch_device)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # batched generation
        tokenized_prompts = tokenizer(prompt, return_tensors="pt", padding="longest").to(torch_device)
        batched_gen = model.generate(**tokenized_prompts, max_new_tokens=30, use_cache=True)
        batched_output = tokenizer.batch_decode(batched_gen, skip_special_tokens=True)

        # individual generation

        for index_gen, individual_prompt in enumerate(prompt):
            inputs = tokenizer(individual_prompt, return_tensors="pt", padding="longest").to(torch_device)
            individual_gen = model.generate(**inputs, max_new_tokens=30, use_cache=True)
            individual_output = tokenizer.batch_decode(individual_gen, skip_special_tokens=True)[0]
            self.assertEqual(individual_output[:100], batched_output[index_gen][:100])

    @slow
    @require_torch_accelerator
    def test_mamba2_mixer_train_vs_eval_equivalence(self):
        # Based on https://github.com/sustcsonglin/flash-linear-attention/issues/63
        # Credit to zhixuan-lin

        B, T, D = 4, 512, 768
        dtype = torch.bfloat16
        config = Mamba2Config(num_heads=24, head_dim=64, hidden_size=768, expand=2, n_groups=1)

        torch.manual_seed(42)
        with torch.autocast(device_type=torch_device, dtype=dtype):
            with torch.no_grad():
                mixer = Mamba2Mixer(config, layer_idx=0).to(torch_device)
                hidden_states = torch.rand(size=(B, T, D), dtype=dtype, device=torch_device)

                mixer.train()
                out_train = mixer(hidden_states)

                mixer.eval()
                out_eval = mixer(hidden_states)

                torch.testing.assert_close(out_train, out_eval, rtol=1e-3, atol=1e-3)
