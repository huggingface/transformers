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


# Request schema of each question type: building one from the request's dict validates it
@strict
@dataclass
class NoulQuestion:
    type: Literal["noul"] = "noul"
    instructions: Any = None
    criteria: dict[Literal["true", "false"], Any] | None = None


@strict
@dataclass
class ChoiceQuestion:
    criteria: dict[str, Any]
    type: Literal["choice"] = "choice"
    instructions: Any = None

    def validate_options(self):
        if not self.criteria:
            raise ValueError("A `choice` question needs `criteria` with its options.")


@strict
@dataclass
class ScoreQuestion:
    criteria: list[Any]
    type: Literal["score"] = "score"
    instructions: Any = None

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
        if (kind := question.get("type")) not in QUESTION_SCHEMAS:
            raise ValueError(f"Question {name!r} has type {kind!r}, expected one of {', '.join(QUESTION_TYPES)}.")
        try:
            question = asdict(QUESTION_SCHEMAS[kind](**question))
        except (TypeError, StrictDataclassError) as error:
            raise ValueError(f"Invalid question {name!r}: {' '.join(str(error).split())}") from error
        criteria = question["criteria"]
        if kind == "noul":
            criteria = {option: (criteria or {}).get(option) for option in config.noul_options}
        elif kind == "choice":
            criteria = dict(sorted(criteria.items()) if config.sort_choices else criteria.items())
        else:
            criteria = {str(index): level for index, level in enumerate(criteria)}
        labels = config.labels.get(kind) or (["Yes", "No"] if kind == "noul" else list(string.ascii_uppercase))
        if len(labels) < len(criteria):
            raise ValueError(f"labels.{kind} needs at least {len(criteria)} labels for question {name!r}.")
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
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in question.labels]
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


class DecisionTemplate:
    """
    Renders a System One request into [`DecisionPrompt`]s: one per question, or one for all when the template uses
    `questions`. Without a decision template, the generic `DECISION_PROMPT` goes in the tokenizer's chat template.
    """

    def __init__(self, tokenizer, config: DecisionConfig | None = None):
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

    def __call__(self, request: dict[str, Any] | str) -> list[DecisionPrompt]:
        # One pass through JSON drops the request's sentinels and marks its control tokens (keys included) as text.
        # String requests are parsed first, so that `\u` escapes can't hide them.
        text = json.dumps(json.loads(request) if isinstance(request, str) else request, ensure_ascii=False)
        request = json.loads(mark_control_tokens(_SENTINEL_RE.sub("", text), self.control_tokens_re))
        questions = [
            Question.from_request(name, question, self.config) for name, question in request["questions"].items()
        ]
        prompts = []
        for group in [questions] if self.one_prompt else [[question] for question in questions]:
            text = self.template.render(state=request["state"], question=group[0], questions=group)
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
]
