# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers import DecisionConfig, VonConfig, is_torch_available
from transformers.decision_utils import DecisionPrompt, Question
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device

from ...test_decision_head_common import DecisionHeadTesterMixin, check_decision, decision_prompt


if is_torch_available():
    import torch

    from transformers import ModernBertConfig, ModernBertModel, VonDecisionHead


@require_torch
class VonDecisionHeadTest(DecisionHeadTesterMixin, unittest.TestCase):
    all_model_classes = (VonDecisionHead,) if is_torch_available() else ()
    config = VonConfig(hidden_size=16, noul_prior={"a": -0.5, "b": 0.3})

    def get_prompts(self):
        # A `[MASK]` before every mark and a last mark closing the block; a zero-shot `noul` has a second block
        noul = Question.from_request("n", {"type": "noul", "instructions": "True?"}, DecisionConfig())
        return [decision_prompt([4, 6, 8, 10]), DecisionPrompt(list(range(16)), [4, 6, 8, 11, 13, 15], [noul])]

    def test_options_are_independent(self):
        # Options only attend to the prompt and to themselves, at positions that restart after the prompt (also for the
        # sliding window): reversing them reverses their hidden states, with every attention implementation
        config = ModernBertConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            local_attention=8,
            pad_token_id=0,
        )
        backbone = ModernBertModel(config).eval()
        head = VonDecisionHead(self.config).eval()
        for prompt in self.get_prompts():
            input_ids, marks, reads = [], [], []
            for start, block_reads, end in head.blocks(prompt):
                input_ids += prompt.input_ids[start : block_reads[0]]
                for begin, stop in reversed(list(zip(block_reads, [*block_reads[1:], end - 1]))):
                    marks.append(len(input_ids) + 1)
                    input_ids += prompt.input_ids[begin:stop]
                input_ids.append(prompt.input_ids[end - 1])
                marks.append(len(input_ids))
                reads += block_reads
            reversed_prompt = DecisionPrompt(input_ids, marks, prompt.questions)
            reversed_reads = [read for _, block, _ in head.blocks(reversed_prompt) for read in block[::-1]]
            states = {}
            for implementation in ("sdpa", "eager"):
                backbone.set_attn_implementation(implementation)
                with torch.no_grad():
                    states[implementation] = backbone(**head.backbone_inputs(prompt, backbone))[0][0, reads]
                    reversed_states = backbone(**head.backbone_inputs(reversed_prompt, backbone))[0][0]
                torch.testing.assert_close(reversed_states[reversed_reads], states[implementation])
            torch.testing.assert_close(states["eager"], states["sdpa"])


@slow
@require_torch_accelerator
class VonIntegrationTest(unittest.TestCase):
    def test_von(self):
        expected = {
            "intent": {"refund": 0.9158, "other": 0.0842},
            "urgent": 0.8436,
            "tone": {"0": 0.7911, "1": 0.2089},
        }
        check_decision(self, "wfzyx/von", "ArthurZ/von", expected, dtype=torch.float32, device=torch_device)
