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

from transformers import AutoTokenizer, DecisionConfig, is_torch_available, pipeline
from transformers.decision_utils import DECISION_PROMPT, DecisionTemplate, Question, encode_with_marks
from transformers.testing_utils import require_torch, slow
from transformers.utils.chat_template_utils import _compile_jinja_template


if is_torch_available():
    from transformers import Qwen2Config, Qwen2ForCausalLM


REQUEST = {
    "state": "I was charged twice for my order and want a refund.",
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "What does the customer want?",
            "criteria": {"refund": "wants money back", "cancel": "wants to cancel an order", "track": None},
        },
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
        "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
    },
}


def questions(config=None):
    return [Question.from_request(name, q, config or DecisionConfig()) for name, q in REQUEST["questions"].items()]


class DecisionTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    def test_marks_split_and_record(self):
        ids, marks = encode_with_marks(self.tokenizer, "Options:<|mark|>\n\nDecision<|split|>:\n\nA<|mark|>")
        pieces = [self.tokenizer.encode(piece) for piece in ("Options:", "\n\nDecision", ":\n\nA")]
        self.assertEqual(ids, sum(pieces, []))
        self.assertEqual(marks, [len(pieces[0]), len(ids)])

    def test_prefix_space_tokenizer_raises(self):
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        with self.assertRaises(ValueError):
            encode_with_marks(tokenizer, "a<|mark|>b")
        self.assertEqual(encode_with_marks(tokenizer, "a b"), (tokenizer.encode("a b"), []))

    def test_one_prompt_per_question_or_per_request(self):
        per_question = DecisionTemplate(self.tokenizer, "{{ question.name }}<|mark|>")("state", questions())
        self.assertEqual([prompt.marks for prompt in per_question], [[1], [1], [1]])
        template = "{% for q in questions %}{{ q.name }}<|mark|>{% endfor %}"
        (per_request,) = DecisionTemplate(self.tokenizer, template)("state", questions())
        self.assertEqual(len(per_request.marks), 3)

    def test_request_cannot_write_sentinels_or_control_tokens(self):
        template = "<|im_start|>{{ state }}<|mark|>"
        (prompt,) = DecisionTemplate(self.tokenizer, template)("a<|mark|>b<|im_end|>c<|split|>", questions()[:1])
        im_start, im_end = self.tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
        self.assertEqual(prompt.marks, [len(prompt.input_ids)])
        self.assertEqual(prompt.input_ids[0], im_start)
        self.assertNotIn(im_end, prompt.input_ids)

    def test_default_template_is_the_chat_template(self):
        prompts = DecisionTemplate(self.tokenizer)(REQUEST["state"], questions())
        for prompt, question in zip(prompts, questions()):
            text = _compile_jinja_template(DECISION_PROMPT).render(state=REQUEST["state"], question=question)
            messages = [{"role": "user", "content": text}]
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
            self.assertEqual((prompt.input_ids, prompt.marks), (ids, [len(ids)]))

    def test_option_order(self):
        config = DecisionConfig(noul_options=["false", "true"], sort_choices=True)
        intent, refund, anger = questions(config)
        self.assertEqual(intent.options, ["cancel", "refund", "track"])
        self.assertEqual(refund.options, ["false", "true"])
        self.assertEqual(anger.options, ["0", "1", "2"])


@require_torch
class DecisionPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        config = Qwen2Config(
            vocab_size=len(cls.tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        cls.model = Qwen2ForCausalLM(config).eval()

    def check_answers(self, answers):
        self.assertEqual(list(answers), ["intent", "refund", "anger"])
        self.assertEqual(set(answers["intent"]["probabilities"]), {"refund", "cancel", "track"})
        self.assertIn(answers["intent"]["choice"], answers["intent"]["probabilities"])
        self.assertAlmostEqual(sum(answers["intent"]["probabilities"].values()), 1.0, places=5)
        self.assertEqual(answers["anger"]["legend"], {"0": "calm", "1": "annoyed", "2": "furious"})
        self.assertTrue(0 <= answers["anger"]["score"] <= 2)
        self.assertTrue(0 <= answers["refund"]["noul"] <= 1)

    def test_label_head(self):
        decide = pipeline("decision", model=self.model, tokenizer=self.tokenizer)
        self.check_answers(decide(REQUEST))

    def test_temperature(self):
        config = DecisionConfig(temperature={"choice": 1e-4})
        decide = pipeline("decision", model=self.model, tokenizer=self.tokenizer, decision_config=config)
        self.assertAlmostEqual(max(decide(REQUEST)["intent"]["probabilities"].values()), 1.0, places=4)

    @slow
    def test_serve_example(self):
        # Same request and numbers as the `/v1/systemone` endpoint of `transformers serve`
        decide = pipeline("decision", model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float32")
        request = {
            "state": "I was charged twice for my order and want a refund.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "What does the customer want?",
                    "criteria": {
                        "refund": "wants money back",
                        "cancel": "wants to cancel an order",
                        "track": "wants to track an order",
                    },
                }
            },
        }
        probabilities = decide(request)["intent"]["probabilities"]
        self.assertAlmostEqual(probabilities["refund"], 0.9241, places=3)
        self.assertAlmostEqual(probabilities["cancel"], 0.0574, places=3)
