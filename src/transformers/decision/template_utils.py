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
"""
Decision models answer typed questions (`choice`, `score`, `noul`) about a `state` with a probability per option, from one
forward pass. Their chat template writes `<|mark|>` where the head reads and `<|split|>` where the prompt was tokenized
in pieces: the text between two sentinels is tokenized on its own.
"""

import difflib
import functools
import json
import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Literal

from huggingface_hub.dataclasses import strict
from huggingface_hub.errors import StrictDataclassError

from ..tokenization_utils_base import BatchEncoding
from ..utils.chat_template_utils import (
    _compile_jinja_template,
    _get_control_token_patterns,
    _get_template_variables,
    encode_marked_chat,
    mark_control_tokens,
)
from .configuration_utils import DecisionConfig


MARK, SPLIT = "<|mark|>", "<|split|>"
_SENTINEL_RE = re.compile(f"({re.escape(MARK)}|{re.escape(SPLIT)})")


# Text, a JSON object or an array, as in the System One API
Content = str | dict | list


# Request schema of each question type: building one from the request's dict validates it. Docstrings are the examples
# shown with errors.
@strict
@dataclass
class NoulQuestion:
    """{"type": "noul", "instructions": "Is this message spam?", "criteria": {"true": "Unsolicited ads", "false": "A real conversation"}}"""

    type: Literal["noul"] = "noul"
    instructions: Content | None = None
    criteria: dict[Literal["true", "false"], Content | None] | None = None


@strict
@dataclass
class ChoiceQuestion:
    """{"type": "choice", "instructions": "What is the tone?", "criteria": {"calm": "A polite message", "angry": null}}"""

    criteria: dict[str, Content | None]
    type: Literal["choice"] = "choice"
    instructions: Content | None = None

    def validate_options(self):
        if not self.criteria:
            raise ValueError("A `choice` question needs `criteria` with its options.")


@strict
@dataclass
class ScoreQuestion:
    """{"type": "score", "instructions": "How urgent is it?", "criteria": ["Can wait", "This week", "Today"]}"""

    criteria: list[Content]
    type: Literal["score"] = "score"
    instructions: Content | None = None

    def validate_levels(self):
        if not self.criteria:
            raise ValueError("A `score` question needs `criteria` with its levels.")


QUESTION_SCHEMAS = {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}
# The question types, in the order of their `question_type_ids`
QUESTION_TYPES = tuple(QUESTION_SCHEMAS)

# The prompt of models without a decision template, wrapped in their own chat template
DECISION_PROMPT = """
{%- macro text(x) %}{{ "" if x is none else x if x is string else x|tojson }}{% endmacro -%}
{%- set ns = namespace(lines=[text(question.instructions)]) -%}
{%- for option in question.options -%}
    {%- set label, answer = question.labels[loop.index0], text(question.descriptions[loop.index0]) -%}
    {%- if question.type == "noul" -%}
        {%- set answer = answer or (option if question.labels != ["Yes", "No"] else "") -%}
        {%- set ns.lines = ns.lines + ([label ~ ": " ~ answer] if answer else []) -%}
    {%- else -%}
        {%- set answer = (option ~ ": " ~ answer if answer else option) if question.type == "choice" else answer -%}
        {%- set ns.lines = ns.lines + [label ~ ". " ~ answer] -%}
    {%- endif -%}
{%- endfor -%}
{{- text(state) ~ "\\n\\n" ~ ns.lines|select|join("\\n") ~ "\\n" -}}
{%- if question.type == "noul" -%}
    {{- "Answer " ~ question.labels[0] ~ " or " ~ question.labels[1] ~ " only." -}}
{%- else -%}
    {%- set kind = "letter" if question.labels == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: question.labels|length]|list else "label" -%}
    {{- "Answer with the " ~ kind ~ " of the " ~ ("option" if question.type == "choice" else "level") -}}
    {{- " that fits best (" ~ question.labels[0] ~ " to " ~ question.labels[-1] ~ ") only." -}}
{%- endif -%}
"""


@dataclass
class Question:
    """One System One question, with its options (choice keys, score levels or true/false) in template order."""

    name: str
    type: Literal["noul", "choice", "score"]
    instructions: Any
    options: list[str]
    labels: list[str]
    descriptions: list[Any]
    legend: list[Any] | None = None

    @classmethod
    def from_request(cls, name: str, question: dict[str, Any], config: DecisionConfig) -> "Question":
        kind = question.get("type") if isinstance(question, dict) else None
        if kind not in QUESTION_SCHEMAS:
            close = difflib.get_close_matches(str(kind).lower(), QUESTION_TYPES, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else f" For example: {NoulQuestion.__doc__}"
            got = f"type {kind!r}" if isinstance(question, dict) else repr(question)
            raise ValueError(
                f"Question {name!r}: expected a `type` among {', '.join(QUESTION_TYPES)}, got {got}.{hint}"
            )
        schema = QUESTION_SCHEMAS[kind]
        try:
            question = asdict(schema(**question))
        except (TypeError, StrictDataclassError) as error:
            if field := re.match(r"Validation error for field '(\w+)'", str(error)):
                reason = f"`{field[1]}` can't be {question[field[1]]!r}"
            else:  # an unknown field (`TypeError`) or a class validator
                reason = str(error.__cause__ or error).split("() ", 1)[-1]
                reason = reason.replace("got an unexpected keyword argument", "unknown field")
            reason += "" if reason.endswith((".", "?")) else "."
            raise ValueError(f"Question {name!r}: {reason} A {kind} question looks like {schema.__doc__}") from error
        criteria = question["criteria"]
        if kind == "noul":
            criteria = {option: (criteria or {}).get(option) for option in config.noul_options}
        elif kind == "choice":
            criteria = dict(sorted(criteria.items()) if config.sort_choices else criteria.items())
        else:
            criteria = {str(index): level for index, level in enumerate(criteria)}
        labels = config.labels.get(kind) or (["Yes", "No"] if kind == "noul" else list(string.ascii_uppercase))
        if len(labels) < len(criteria):
            raise ValueError(
                f"Question {name!r}: this model reads at most {len(labels)} options for a {kind} question, got "
                f"{len(criteria)} (`labels.{kind}` in its decision config)."
            )
        options, descriptions = list(criteria), list(criteria.values())
        legend = descriptions if kind == "score" else None
        return cls(name, kind, question.get("instructions"), options, labels[: len(options)], descriptions, legend)

    def answer(self, probabilities: list[float]) -> dict[str, Any]:
        """The System One answer, from the probabilities of the options."""
        by_option, n_options = dict(zip(self.options, probabilities)), len(probabilities)
        if self.type == "noul":
            return {"type": "noul", "noul": by_option["true"]}

        best = max(by_option, key=by_option.get)
        if self.type == "choice":
            confidence = 1.0 if n_options == 1 else max(0.0, (by_option[best] - 1 / n_options) / (1 - 1 / n_options))
            return {"type": "choice", "choice": best, "confidence": confidence, "probabilities": by_option}

        distance = sum(probability * abs(level - int(best)) for level, probability in enumerate(probabilities))
        uniform_distance = sum(abs(level - (n_options - 1) / 2) for level in range(n_options)) / n_options
        confidence = 1.0 if n_options == 1 else max(0.0, 1 - distance / uniform_distance)
        score = sum(level * probability for level, probability in enumerate(probabilities))
        answer = {"type": "score", "score": score, "legend": dict(zip(self.options, self.legend))}
        return {
            **answer,
            "confidence": confidence,
            "probabilities": by_option,
        }


def label_token_ids(tokenizer, question: Question) -> list[int]:
    """The ids of the labels of `question`, which must be distinct single known tokens."""
    ids = tokenizer(question.labels, add_special_tokens=False)["input_ids"]
    label_ids = [token_ids[0] for token_ids in ids if len(token_ids) == 1 and token_ids[0] != tokenizer.unk_token_id]
    if len(set(label_ids)) != len(ids):
        raise ValueError(f"The labels of question {question.name!r} are not distinct single tokens for this model.")
    return label_ids


def encode_with_marks(tokenizer, text: str, markers_re: re.Pattern | None = None) -> tuple[list[int], list[int]]:
    """Tokenize the text between sentinels on its own; each `<|mark|>` becomes the number of tokens before it."""
    ids, marks = [], []
    for piece in _SENTINEL_RE.split(text):
        if piece == MARK:
            marks.append(len(ids))
        elif piece and piece != SPLIT:
            ids += (
                encode_marked_chat(tokenizer, piece, markers_re)
                if markers_re
                else tokenizer.encode(piece, add_special_tokens=False)
            )
    return ids, marks


@dataclass
class DecisionPrompt:
    input_ids: list[int]
    marks: list[int]
    questions: list[Question]

    def model_inputs(self) -> BatchEncoding:
        """
        The inputs of the heads, as tensors: `input_ids`, `marks` (the number of tokens before each `<|mark|>`), and the
        `num_options` and `question_type_ids` (index in `QUESTION_TYPES`) of each question.
        """
        inputs = {
            "input_ids": self.input_ids,
            "marks": self.marks,
            "num_options": [len(question.options) for question in self.questions],
            "question_type_ids": [QUESTION_TYPES.index(question.type) for question in self.questions],
        }
        return BatchEncoding(inputs, tensor_type="pt")


def parse_questions(questions: dict[str, Any], config: DecisionConfig) -> list[Question]:
    """
    Validates the `questions` of a System One request and puts them in template order, raising one `ValueError` that
    lists every invalid question, each with an example of a valid one.
    """
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"`questions` maps names to questions, with at least one question; got {questions!r}.")
    parsed, errors = [], []
    for name, question in questions.items():
        try:
            parsed.append(Question.from_request(name, question, config))
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError("\n".join(errors))
    return parsed


class DecisionTemplate:
    """
    Renders a System One request into [`DecisionPrompt`]s: one per question, or one for all when the template uses
    `questions`. Without a decision template, the generic `DECISION_PROMPT` goes in the tokenizer's chat template.
    """

    def __init__(self, tokenizer, config: DecisionConfig | None = None, questions: dict[str, Any] | None = None):
        self.tokenizer, self.config = tokenizer, config or DecisionConfig()
        template = self.config.chat_template
        if template is None:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": "\0"}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prefix, suffix = chat.split("\0")
            template = f"{{% raw %}}{prefix}{{% endraw %}}{DECISION_PROMPT}{{% raw %}}{suffix}{{% endraw %}}{MARK}"
        self.template = _compile_jinja_template(template)
        self.one_prompt = "questions" in _get_template_variables(template)
        control_tokens = tokenizer._get_control_tokens(template) if self.config.sanitize_control_tokens else ()
        self.markers_re = _get_control_token_patterns(control_tokens)[1]
        # Requests are marked as JSON text, where a token like `<|"|>` reads `<|\"|>`
        json_tokens = tuple(json.dumps(token, ensure_ascii=False)[1:-1] for token in control_tokens)
        self.control_tokens_re = _get_control_token_patterns(json_tokens)[0]
        # Each distinct set of questions is validated once: keyed by its JSON, a repeated set is a cache hit
        self._parse_questions = functools.lru_cache(maxsize=256)(
            lambda questions: parse_questions(json.loads(questions), self.config)
        )
        # The questions asked when a call gets a bare state: the ones given here, then those of the last request
        self.questions = None if questions is None else self.parse_questions(self.sanitize(questions))

    def sanitize(self, value: Any) -> Any:
        """One pass through JSON drops the sentinels and marks the control tokens (keys included) as text."""
        text = json.dumps(value, ensure_ascii=False)
        return json.loads(mark_control_tokens(_SENTINEL_RE.sub("", text), self.control_tokens_re))

    def parse_questions(self, questions: dict[str, Any]) -> list[Question]:
        """[`parse_questions`], cached: the returned list is shared between calls and must not be modified."""
        return self._parse_questions(json.dumps(questions, ensure_ascii=False))

    def __call__(self, request: dict[str, Any] | str | Any) -> list[DecisionPrompt]:
        """
        Renders `request`: a `{"state": ..., "questions": ...}` dict or its JSON, whose questions become the current
        ones, or a bare state (any other value), asked the current questions.
        """
        if isinstance(request, str):  # parsed before sanitizing, so that `\u` escapes can't hide sentinels or tokens
            try:
                parsed = json.loads(request)
            except json.JSONDecodeError:
                parsed = None
            request = parsed if isinstance(parsed, dict) and "questions" in parsed else request
        request = self.sanitize(request)
        if isinstance(request, dict) and "questions" in request:
            if "state" not in request:
                raise ValueError(
                    'A request is {"state": ..., "questions": {name: question, ...}}, `state` is missing.'
                )
            self.questions, state = self.parse_questions(request["questions"]), request["state"]
        elif self.questions is None:
            raise ValueError(
                'No questions to ask yet: send {"state": ..., "questions": {...}} once, or set `questions` up front.'
            )
        else:
            state = request
        questions = self.questions
        prompts = []
        for group in [questions] if self.one_prompt else [[question] for question in questions]:
            text = self.template.render(state=state, question=group[0], questions=group)
            prompts.append(DecisionPrompt(*encode_with_marks(self.tokenizer, text, self.markers_re), group))
        return prompts


__all__ = [
    "DECISION_PROMPT",
    "MARK",
    "QUESTION_TYPES",
    "SPLIT",
    "DecisionPrompt",
    "DecisionTemplate",
    "Question",
    "encode_with_marks",
    "label_token_ids",
    "parse_questions",
]
