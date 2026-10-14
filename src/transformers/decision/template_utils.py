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
from dataclasses import dataclass
from typing import Any

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
    type: str
    instructions: Any
    options: list[str]
    labels: list[str]
    descriptions: list[Any]
    legend: list[Any] | None = None

    @classmethod
    def from_request(cls, name: str, question: dict[str, Any], config: DecisionConfig) -> "Question":
        kind, criteria = question["type"], question.get("criteria")
        if kind in ("choice", "score") and not criteria:
            raise ValueError(f"Question {name!r} needs `criteria` with its options.")
        if kind == "noul":
            criteria = {option: (criteria or {}).get(option) for option in config.noul_options}
        elif kind == "choice":
            criteria = dict(sorted(criteria.items()) if config.sort_choices else criteria.items())
        elif kind == "score":
            criteria = {str(index): level for index, level in enumerate(criteria)}
        else:
            raise ValueError(f"Question {name!r} has type {kind!r}, expected choice, score or noul.")
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
            return {"type": "choice", "choice": best, "probabilities": by_option, "confidence": confidence}
        distance = sum(probability * abs(level - int(best)) for level, probability in enumerate(probabilities))
        uniform_distance = sum(abs(level - (n_options - 1) / 2) for level in range(n_options)) / n_options
        confidence = 1.0 if n_options == 1 else max(0.0, 1 - distance / uniform_distance)
        score = sum(level * probability for level, probability in enumerate(probabilities))
        answer = {"type": "score", "score": score, "legend": dict(zip(self.options, self.legend))}
        return {**answer, "probabilities": by_option, "confidence": confidence}


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
    "SPLIT",
    "DecisionPrompt",
    "DecisionTemplate",
    "Question",
    "encode_with_marks",
    "label_token_ids",
]
