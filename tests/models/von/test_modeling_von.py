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

import tempfile
import unittest

from transformers import DecisionConfig, VonConfig, is_torch_available
from transformers.decision_utils import DecisionPrompt, Question
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import AutoModel, ModernBertConfig, ModernBertModel, VonDecisionHead


@require_torch
class VonDecisionHeadTest(unittest.TestCase):
    all_model_classes = (VonDecisionHead,) if is_torch_available() else ()

    def setUp(self):
        config = DecisionConfig()
        choice = {"type": "choice", "instructions": "Pick one", "criteria": {"a": None, "b": None, "c": None}}
        noul = {"type": "noul", "instructions": "True?"}
        # A `[MASK]` before every mark and a last mark closing the block, a zero-shot `noul` has a second block
        self.prompts = [
            DecisionPrompt(list(range(12)), [4, 6, 8, 10], [Question.from_request("q", choice, config)]),
            DecisionPrompt(list(range(16)), [4, 6, 8, 11, 13, 15], [Question.from_request("n", noul, config)]),
        ]
        self.head = VonDecisionHead(VonConfig(hidden_size=16, noul_prior={"a": -0.5, "b": 0.3})).eval()

    def test_scores_and_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.head.save_pretrained(tmp)
            loaded = AutoModel.from_pretrained(tmp).eval()
        for prompt in self.prompts:
            hidden_states = torch.randn(len(prompt.input_ids), 16)
            with torch.no_grad():
                (scores,), (loaded_scores,) = self.head(hidden_states, prompt), loaded(hidden_states, prompt)
            self.assertEqual(len(scores), len(prompt.questions[0].options))
            torch.testing.assert_close(scores, loaded_scores)

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
        for prompt in self.prompts:
            input_ids, marks, reads = [], [], []
            for start, block_reads, end in self.head.blocks(prompt):
                input_ids += prompt.input_ids[start : block_reads[0]]
                for begin, stop in reversed(list(zip(block_reads, [*block_reads[1:], end - 1]))):
                    marks.append(len(input_ids) + 1)
                    input_ids += prompt.input_ids[begin:stop]
                input_ids.append(prompt.input_ids[end - 1])
                marks.append(len(input_ids))
                reads += block_reads
            reversed_prompt = DecisionPrompt(input_ids, marks, prompt.questions)
            reversed_reads = [read for _, block, _ in self.head.blocks(reversed_prompt) for read in block[::-1]]
            states = {}
            for implementation in ("sdpa", "eager"):
                backbone.set_attn_implementation(implementation)
                with torch.no_grad():
                    states[implementation] = backbone(**self.head.backbone_inputs(prompt, backbone))[0][0, reads]
                    reversed_states = backbone(**self.head.backbone_inputs(reversed_prompt, backbone))[0][0]
                torch.testing.assert_close(reversed_states[reversed_reads], states[implementation])
            torch.testing.assert_close(states["eager"], states["sdpa"])
