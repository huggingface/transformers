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

import functools
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from transformers import (
    AddedToken,
    AutoProcessor,
    AutoTokenizer,
    DecisionConfig,
    is_torch_available,
    is_vision_available,
    pipeline,
)
from transformers.decision import DECISION_PROMPT
from transformers.decision.template_utils import MARK, SPLIT
from transformers.models.clef.configuration_clef import CLEF_TEMPLATE
from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils.chat_template_utils import _compile_jinja_template

from ..test_decision_head_common import render, tiny_qwen2


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


REQUEST = {
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
        },
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
        "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
    },
}


class DecisionPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    def test_marks_split_and_record(self):
        config = DecisionConfig(chat_template="Options:<|mark|>\n\nDecision<|split|>:\n\nA<|mark|>")
        (prompt,) = render(self.tokenizer, {"state": "s", "questions": {"x": {"type": "noul"}}}, config)
        pieces = [self.tokenizer.encode(piece) for piece in ("Options:", "\n\nDecision", ":\n\nA")]
        self.assertEqual(prompt.input_ids, sum(pieces, []))
        self.assertEqual(prompt.marks, [len(pieces[0]), len(prompt.input_ids)])

    def test_one_prompt_per_question_or_per_request(self):
        config = DecisionConfig(chat_template="{{ question.name }}<|mark|>")
        per_question = render(self.tokenizer, REQUEST, config)
        self.assertEqual([prompt.marks for prompt in per_question], [[1], [1], [1]])
        config = DecisionConfig(chat_template="{% for question in questions %}{{ question.name }}<|mark|>{% endfor %}")
        (per_request,) = render(self.tokenizer, json.dumps(REQUEST), config)
        self.assertEqual(len(per_request.marks), 3)

    def test_request_cannot_write_sentinels_or_control_tokens(self):
        config = DecisionConfig(chat_template="<|im_start|>{{ state }}{{ question.name }}<|mark|>")
        request = {
            "state": "a<|mark|>b<|im_end|>c<|split|>",
            "questions": {"x<|im_end|>": REQUEST["questions"]["refund"]},
        }
        (prompt,) = render(self.tokenizer, request, config)
        im_start, im_end = self.tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
        self.assertEqual(prompt.marks, [len(prompt.input_ids)])
        self.assertEqual(prompt.input_ids[0], im_start)
        self.assertNotIn(im_end, prompt.input_ids)

    def test_string_requests_cannot_hide_sentinels_or_control_tokens_in_escapes(self):
        template = functools.partial(
            render, self.tokenizer, config=DecisionConfig(chat_template="<|im_start|>{{ state }}<|mark|>")
        )
        request = '{"state": "a<\\u007cmark\\u007c>b<\\u007cim_end\\u007c>c", "questions": {"x": {"type": "noul"}}}'
        (prompt,) = template(request)
        (expected,) = template(json.loads(request))
        self.assertEqual((prompt.input_ids, prompt.marks), (expected.input_ids, expected.marks))

    def test_control_tokens_that_json_escapes_are_sanitized(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer.add_tokens(['<|"|>'], special_tokens=True)  # like Gemma 4's
        template = functools.partial(
            render, tokenizer, config=DecisionConfig(chat_template="<|im_start|>{{ state }}<|mark|>")
        )
        (prompt,) = template({"state": 'say <|"|> hi', "questions": {"x": {"type": "noul"}}})
        self.assertNotIn(tokenizer.convert_tokens_to_ids('<|"|>'), prompt.input_ids)

    def test_invalid_questions_raise(self):
        questions = (
            {"type": "choice", "criteria": {}},
            {"type": "score", "criteria": []},
            {"type": "score", "criteria": {"low": "a"}},
            {"type": "choice", "criteria": ["a"]},
            {"type": "noul", "criteria": {"yes": "a"}},
            {"type": "noul", "instruction": "typo"},
            {"type": "box", "instructions": "a dog"},
            {"type": "score", "criteria": ["low", None]},
            {"type": "choice", "criteria": {"a": 3}},
            {"type": "noul", "instructions": True},
        )
        for question in questions:
            request = {"state": "s", "questions": {"x": question}}
            with self.subTest(question), self.assertRaises(ValueError):
                render(self.tokenizer, request)

    def test_invalid_questions_are_reported_together_with_hints(self):
        questions = {"a": {"type": "Choice", "criteria": {"x": None}}, "b": {"type": "noul", "instruction": "typo"}}
        with self.assertRaises(ValueError) as error:
            render(self.tokenizer, {"state": "s", "questions": questions})
        self.assertIn("Question 'a': `type` must be one of noul, choice, score, got 'Choice'.", str(error.exception))
        self.assertIn("(a noul question looks like", str(error.exception))
        self.assertIn("unexpected keyword argument 'instruction'", str(error.exception))

    def test_label_tokens_cap_the_options(self):
        # By default each option is read from the logit of its label token: at most 26 options (A to Z)
        criteria = {f"option {index}": None for index in range(27)}
        request = {"state": "s", "questions": {"x": {"type": "choice", "criteria": criteria}}}
        with self.assertRaisesRegex(ValueError, "at most 26 options for a choice question, got 27"):
            render(self.tokenizer, request)
        # Labels of the decision config cap them the same way
        config = DecisionConfig(labels={"choice": ["a", "b"]}, chat_template="{{ question.name }}<|mark|>")
        request = {"state": "s", "questions": {"x": {"type": "choice", "criteria": {"x": None, "y": None, "z": None}}}}
        with self.assertRaisesRegex(ValueError, "at most 2 options"):
            render(self.tokenizer, request, config)

    def test_labels_none_lifts_the_cap(self):
        # Heads that read spans declare no labels: any number of options, and the template gets no labels
        config = DecisionConfig(
            labels={"noul": None, "choice": None, "score": None},
            chat_template="{% for q in questions %}{{ q.name }}{{ q.labels|length }}<|mark|>{% endfor %}",
        )
        questions = {
            "pick": {"type": "choice", "criteria": {f"option {index}": None for index in range(40)}},
            "rate": {"type": "score", "criteria": [f"level {index}" for index in range(30)]},
            "yes": {"type": "noul"},
        }
        (prompt,) = render(self.tokenizer, {"state": "s", "questions": questions}, config)
        self.assertEqual([len(question.options) for question in prompt.questions], [40, 30, 2])
        self.assertEqual([question.labels for question in prompt.questions], [[], [], []])
        # Only the types it lists: the others keep their default labels and cap
        config = DecisionConfig(labels={"choice": None}, chat_template="{{ question.name }}<|mark|>")
        questions = {"rate": {"type": "score", "criteria": [f"level {index}" for index in range(27)]}}
        with self.assertRaisesRegex(ValueError, "at most 26 options for a score question"):
            render(self.tokenizer, {"state": "s", "questions": questions}, config)

    def test_labels_none_survive_save_and_reload(self):
        config = DecisionConfig(labels={"noul": None, "choice": None, "score": ["low", "high"]})
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            saved = json.loads(Path(directory, "decision.json").read_text(encoding="utf-8"))
            reloaded = DecisionConfig.from_pretrained(directory)
        self.assertEqual(saved["labels"], {"noul": None, "choice": None, "score": ["low", "high"]})
        self.assertEqual(reloaded.labels, config.labels)

    def test_each_request_carries_its_questions(self):
        for request in (REQUEST["state"], '{"a": 1}', {"questions": REQUEST["questions"]}):
            with self.assertRaises(ValueError):
                render(self.tokenizer, request)

    @require_vision
    def test_assets_are_images_where_their_placeholders_are(self):
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        template = functools.partial(render, processor)
        red, blue = Image.new("RGB", (64, 48), "red"), Image.new("RGB", (512, 900), "blue")
        request = {
            "state": "The customer sent img-1 and asks about it.",
            "questions": {"q": {"type": "choice", "criteria": {"a": "like img-2", "b": None}}},
            "assets": {"img-1": red, "img-2": blue},
        }
        (prompt,) = template(request)
        # The same chat, with the images as content blocks
        text = _compile_jinja_template(DECISION_PROMPT).render(state=request["state"], question=prompt.questions[0])
        before, middle, after = re.split("img-1|img-2", text)
        content = [{"type": "text", "text": before}, {"type": "image", "image": red}]
        content += [
            {"type": "text", "text": middle},
            {"type": "image", "image": blue},
            {"type": "text", "text": after},
        ]
        expected = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertEqual(prompt.input_ids, expected["input_ids"][0].tolist())
        for key, value in prompt.inputs.items():
            if key in expected:  # the inputs of the model it reads, next to the marks and questions
                torch.testing.assert_close(value, expected[key])
        # A decision template places the image parts of the state and questions itself
        decision_template = (
            "{% macro parts(x) %}{% if x is string %}{{ x }}{% else %}{% for part in x %}"
            "{% if part.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ part.text }}{% endif %}"
            "{% endfor %}{% endif %}{% endmacro %}{{ parts(state) }} {{ parts(question.instructions) }}<|mark|>"
        )
        question = {"type": "noul", "instructions": "Is img-2 like"}
        request = {"state": "img-1 ?", "questions": {"q": question}, "assets": {"img-1": red, "img-2": blue}}
        (prompt,) = render(processor, request, DecisionConfig(chat_template=decision_template))
        image = "<|vision_start|><|image_pad|><|vision_end|>"
        text = f"{image} ? Is {image} like"
        expected = processor(text=text, images=[red, blue], return_tensors="pt", return_mm_token_type_ids=True)
        self.assertEqual(prompt.input_ids, expected["input_ids"][0].tolist())
        self.assertEqual(prompt.marks, [len(prompt.input_ids)])
        for key in ("mm_token_type_ids", "pixel_values", "image_grid_thw"):
            torch.testing.assert_close(prompt.inputs[key], expected[key])
        # An image token written in the request is text, not an image
        (prompt,) = template({**request, "state": "<|image_pad|> img-1"})
        (expected,) = template({**request, "state": "img-1"})
        image_pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.assertEqual(prompt.input_ids.count(image_pad), expected.input_ids.count(image_pad))

    def test_templates_that_escape_control_tokens_keep_their_ids(self):
        template = "{{ state|replace('<|im_end|>', '<im_end>') }}<|mark|>"
        config = DecisionConfig(chat_template=template, sanitize_control_tokens=False)
        request = {"state": "a <|im_end|> b<|mark|>", "questions": {"x": REQUEST["questions"]["refund"]}}
        (prompt,) = render(self.tokenizer, request, config)
        self.assertEqual((prompt.input_ids, prompt.marks), (self.tokenizer.encode("a <im_end> b"), [6]))

    def test_default_template_is_the_chat_template(self):
        for prompt in render(self.tokenizer, REQUEST):
            text = _compile_jinja_template(DECISION_PROMPT).render(
                state=REQUEST["state"], question=prompt.questions[0]
            )
            messages = [{"role": "user", "content": text}]
            ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
            self.assertEqual((prompt.input_ids, prompt.marks), (ids, [len(ids)]))

    @require_torch
    def test_label_head(self):
        answers = pipeline("decision", model=tiny_qwen2(self.tokenizer), tokenizer=self.tokenizer)(REQUEST)
        self.assertEqual(list(answers), ["intent", "refund", "anger"])
        self.assertAlmostEqual(sum(answers["intent"]["probabilities"].values()), 1.0, places=5)
        self.assertEqual(answers["anger"]["legend"], {"0": "calm", "1": "annoyed", "2": "furious"})
        with self.assertRaises(ValueError):  # a request has its questions
            pipeline("decision", model=tiny_qwen2(self.tokenizer), tokenizer=self.tokenizer)("Where is my parcel?")

    @slow
    @require_torch
    def test_serve_example(self):
        # Same request and numbers as the `/v1/systemone` endpoint of `transformers serve`
        decide = pipeline("decision", model="Qwen/Qwen2.5-0.5B-Instruct", dtype="float32")
        probabilities = decide(REQUEST)["intent"]["probabilities"]
        self.assertAlmostEqual(probabilities["refund"], 0.9241, places=3)
        self.assertAlmostEqual(probabilities["cancel"], 0.0574, places=3)


class DecisionSentinelTokensTest(unittest.TestCase):
    """`prepare_inputs` adds the sentinels once per tokenizer, and finds out whether it did without `get_vocab()`."""

    NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    def fresh(self):
        return AutoTokenizer.from_pretrained(self.NAME)

    @staticmethod
    def spy_on_get_vocab(tokenizer):
        return mock.patch.object(type(tokenizer), "get_vocab", autospec=True, side_effect=type(tokenizer).get_vocab)

    def test_known_sentinels_do_not_rebuild_the_vocabulary(self):
        tokenizer = self.fresh()
        with self.spy_on_get_vocab(tokenizer) as spy:
            render(tokenizer, REQUEST)  # the first call adds the sentinels: it may read the vocabulary
            spy.reset_mock()
            render(tokenizer, REQUEST)
            render(tokenizer, REQUEST)
            self.assertEqual(spy.call_count, 0)

    def test_fresh_tokenizer_gains_the_two_sentinels_once(self):
        tokenizer = self.fresh()
        size = len(tokenizer)
        self.assertNotIn(MARK, tokenizer.get_vocab())
        render(tokenizer, REQUEST)
        self.assertEqual(len(tokenizer), size + 2)
        render(tokenizer, REQUEST)
        self.assertEqual(len(tokenizer), size + 2)
        self.assertEqual(tokenizer.convert_tokens_to_ids([MARK, SPLIT]), [size, size + 1])
        added = {token.content: token for token in tokenizer.added_tokens_decoder.values()}
        for sentinel in (MARK, SPLIT):
            self.assertTrue(added[sentinel].special)
            self.assertFalse(added[sentinel].normalized)

    def test_tokenizer_with_the_sentinels_is_left_alone(self):
        tokenizer = self.fresh()
        tokenizer.add_tokens([MARK, SPLIT], special_tokens=True)
        size, ids = len(tokenizer), tokenizer.convert_tokens_to_ids([MARK, SPLIT])
        before = dict(tokenizer.added_tokens_decoder)
        render(tokenizer, REQUEST)
        self.assertEqual(len(tokenizer), size)
        self.assertEqual(tokenizer.convert_tokens_to_ids([MARK, SPLIT]), ids)
        self.assertEqual(tokenizer.added_tokens_decoder, before)

    def test_same_prompts_as_the_vocabulary_check(self):
        def reference(request, config):
            tokenizer = self.fresh()
            if MARK not in tokenizer.get_vocab():  # the check that `prepare_inputs` used to make
                tokenizer.add_tokens([AddedToken(token, special=True, normalized=False) for token in (MARK, SPLIT)])
            return [(p.input_ids, p.marks) for p in render(tokenizer, request, config)]

        requests = [
            {"state": "s", "questions": {"x": {"type": "noul"}}},
            {"state": "I want a refund", "questions": {"intent": REQUEST["questions"]["intent"]}},
            {"state": "I want a refund", "questions": {"anger": REQUEST["questions"]["anger"]}},
            REQUEST,
        ]
        for name, config in (("default", DecisionConfig()), ("clef", DecisionConfig(chat_template=CLEF_TEMPLATE))):
            for request in requests:
                with self.subTest(template=name, questions=list(request["questions"])):
                    expected = reference(request, config)
                    tokenizer = self.fresh()
                    for _ in range(2):  # sentinels added, then known
                        self.assertEqual(
                            [(p.input_ids, p.marks) for p in render(tokenizer, request, config)], expected
                        )

    def test_tokenizer_without_unknown_token(self):
        tokenizer = self.fresh()
        self.assertIsNone(tokenizer.unk_token_id)
        size = len(tokenizer)
        render(tokenizer, REQUEST)
        render(tokenizer, REQUEST)
        self.assertEqual(len(tokenizer), size + 2)

    def test_tokenizer_with_an_unknown_token_and_sentinels_in_its_base_vocabulary(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        from transformers import PreTrainedTokenizerFast

        def build(words):
            backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")

        def prepare(tokenizer):
            config = DecisionConfig(chat_template="{{ state }}<|mark|>")
            return config.prepare_inputs({"state": "state", "questions": {"x": {"type": "noul"}}}, tokenizer)

        words = ["[UNK]", "state", "x", "yes", "no"]
        # an unknown token: an unknown sentinel answers with its id, which is not "known"
        tokenizer = build(words)
        self.assertEqual(tokenizer.convert_tokens_to_ids(MARK), tokenizer.unk_token_id)
        prepare(tokenizer)
        prepare(tokenizer)
        self.assertEqual(len(tokenizer), len(words) + 2)
        # sentinels that are ordinary tokens of the base vocabulary are used as they are
        tokenizer = build([*words, MARK, SPLIT])
        ids = tokenizer.convert_tokens_to_ids([MARK, SPLIT])
        prepare(tokenizer)
        self.assertEqual((len(tokenizer), tokenizer.convert_tokens_to_ids([MARK, SPLIT])), (len(words) + 2, ids))

    def test_processor_exposing_a_tokenizer(self):
        tokenizer = self.fresh()
        size = len(tokenizer)
        # `getattr(processor, "tokenizer", processor)`: a text-only processor delegates to its tokenizer
        processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=tokenizer.apply_chat_template)
        with self.spy_on_get_vocab(tokenizer) as spy:
            first = render(processor, REQUEST)
            spy.reset_mock()
            second = render(processor, REQUEST)
            self.assertEqual(spy.call_count, 0)
        self.assertEqual(len(tokenizer), size + 2)
        self.assertEqual([p.input_ids for p in first], [p.input_ids for p in second])

    def test_tokenizer_whose_unknown_tokens_get_a_fallback_id_without_an_unknown_token(self):
        # Like a slow sentencepiece tokenizer built with `unk_token=None`: `piece_to_id` answers the model's own unknown
        # id (0) for unknown pieces, while `unk_token_id` is None. A toy slow tokenizer, no sentencepiece model needed
        from transformers.tokenization_python import PythonBackend

        class FallbackIdTokenizer(PythonBackend):
            def __init__(self):
                self.vocab = {"<piece0>": 0, "state": 1, "x": 2}
                super().__init__(unk_token=None)

            @property
            def vocab_size(self):
                return len(self.vocab)

            def get_vocab(self):
                return {**self.vocab, **self.added_tokens_encoder}

            def _tokenize(self, text):
                return text.split()

            def _convert_token_to_id(self, token):
                return self.vocab.get(token, 0)  # not `unk_token_id`

            def _convert_id_to_token(self, index):
                return {i: t for t, i in self.vocab.items()}.get(index)

        tokenizer = FallbackIdTokenizer()
        self.assertIsNone(tokenizer.unk_token_id)
        self.assertEqual(tokenizer.convert_tokens_to_ids(MARK), 0)  # the fallback, for a token it does not know
        size = len(tokenizer)
        config = DecisionConfig(chat_template="{{ state }}<|mark|>")
        request = {"state": "state", "questions": {"x": {"type": "noul"}}}
        with self.spy_on_get_vocab(tokenizer) as spy:
            config.prepare_inputs(request, tokenizer)
            spy.reset_mock()
            config.prepare_inputs(request, tokenizer)
            self.assertEqual(spy.call_count, 0)
        self.assertEqual(len(tokenizer), size + 2)  # added once
        mark, split = tokenizer.convert_tokens_to_ids([MARK, SPLIT])
        self.assertEqual(len({mark, split, 0}), 3)
        self.assertEqual(tokenizer.convert_ids_to_tokens([mark, split]), [MARK, SPLIT])

    def test_tokenizer_with_only_the_split_sentinel(self):
        # The old check only looked at `<|mark|>`: with just `<|split|>` present it adds both. `<|split|>` is already
        # there, so that is a no-op for it, and `<|mark|>` is new: two tokens are added and the ids resolve
        tokenizer = self.fresh()
        tokenizer.add_tokens([SPLIT], special_tokens=True)
        split_id, size = tokenizer.convert_tokens_to_ids(SPLIT), len(tokenizer)
        reference = self.fresh()
        reference.add_tokens([SPLIT], special_tokens=True)
        if MARK not in reference.get_vocab():  # the old check
            reference.add_tokens([AddedToken(token, special=True, normalized=False) for token in (MARK, SPLIT)])
        render(tokenizer, REQUEST)
        self.assertEqual(len(tokenizer), len(reference))
        self.assertEqual(len(tokenizer), size + 1)
        self.assertEqual(tokenizer.convert_tokens_to_ids(SPLIT), split_id)
        self.assertEqual(
            tokenizer.convert_tokens_to_ids([MARK, SPLIT]), reference.convert_tokens_to_ids([MARK, SPLIT])
        )

    def test_unknown_token_that_is_the_mark_is_known_like_the_vocabulary_check_says(self):
        from tokenizers import Tokenizer, models, pre_tokenizers

        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel({MARK: 0, "state": 1, "x": 2}, unk_token=MARK))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=MARK)
        self.assertIn(MARK, tokenizer.get_vocab())  # old check: known, nothing added
        tokenizer.get_vocab()
        size = len(tokenizer)
        DecisionConfig(chat_template="{{ state }}<|mark|>").prepare_inputs(
            {"state": "state", "questions": {"x": {"type": "noul"}}}, tokenizer
        )
        self.assertEqual(len(tokenizer), size)
