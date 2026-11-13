# Copyright 2026 Ricursive AI and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Ripple model."""

import re
import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    require_torch_accelerator_memory,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import RippleForCausalLM, RippleModel


class RippleModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = RippleModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        # One layer of each type: sliding (RoPE + window) then full (global, no positional encoding)
        self.layer_types = ["sliding_attention", "full_attention"]
        # The common generation tests check cache shapes against the full sequence, so the window stays larger than
        # the test sequences
        self.sliding_window = 32
        # The released checkpoints use sqrt(hidden_size)
        self.embed_scale = self.hidden_size**0.5
        # Like the released checkpoints: grouped-query attention (KV heads divisible by the tensor-parallel size of
        # the tests), and a head dim that is not hidden_size / heads
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.head_dim = 2 * self.hidden_size // self.num_attention_heads


@require_torch
class RippleModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = RippleModelTester


@slow
@require_torch_accelerator
@require_torch_accelerator_memory(memory=140)
class RippleIntegrationTest(unittest.TestCase):
    model_id = "ricursiveAI/Ripple-65B-Preview"
    # Greedy completions, identical to the hub remote code (`skip_special_tokens` keeps the thinking block)
    completions = {
        "What is the capital of France?": (
            "<think>\nWe need answer simple. Need comply. Capital of France is Paris. Final concise.\n</think>\n"
            "The capital of France is **Paris**."
        ),
        "Write a haiku about the sea.": (
            "Waves fold the gray light, / salt wind turns the gulls to stone, / tide keeps its own time."
        ),
        "Explain in one sentence why the sky is blue.": (
            "Sunlight scatters in the atmosphere, and shorter blue wavelengths scatter more than longer red ones, so "
            "scattered blue light reaches your eyes from all directions."
        ),
    }
    # The default is the B200 (cuda 10.0) record; it can be replaced by another device's, keeping the B200 entry
    expected_completions = Expectations({(None, None): completions, ("cuda", (10, 0)): completions})

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.tokenizer.padding_side = "left"
        cls.model = RippleForCausalLM.from_pretrained(cls.model_id, dtype="auto", device_map="auto")

    @classmethod
    def tearDownClass(cls):
        cls.model = None
        cleanup(torch_device, gc_collect=True)

    def generate(self, prompts, max_new_tokens=100):
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = self.tokenizer.apply_chat_template(
            conversations, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def test_greedy_chat(self):
        for prompt, completion in self.expected_completions.get_expectation().items():
            self.assertEqual(self.generate([prompt]), [completion])

    def test_greedy_chat_batched(self):
        expected = dict(self.expected_completions.get_expectation())
        # bf16 near-tie (one ulp) between "stone" and "ghosts" in the padded row, the same flip as the remote code
        expected["Write a haiku about the sea."] = expected["Write a haiku about the sea."].replace("stone", "ghosts")
        self.assertEqual(self.generate(list(expected)), list(expected.values()))

    def test_recall_beyond_sliding_window(self):
        """The fact sits more than `sliding_window` (1024) tokens before the question: only the full-attention layers
        can carry it, through the window of the sliding layers and the generated thinking block."""
        filler = "The sky is blue. The grass is green. The sun is shining. " * 110  # ~1500 tokens
        prompt = f"Remember this: the secret code is 4721.\n\n{filler}\nWhat is the secret code? Answer with the number only."
        text = self.generate([prompt], max_new_tokens=200)[0]
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        self.assertEqual(answer, "4721")
