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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub.dataclasses import strict

from .utils import cached_file
from .utils.chat_template_utils import (
    _compile_jinja_template,
    _get_control_token_patterns,
    _get_template_variables,
    encode_marked_chat,
    mark_control_tokens,
)


DECISION_CONFIG_NAME, DECISION_TEMPLATE_NAME = "decision_config.json", "decision_template.jinja"
MARK, SPLIT = "<|mark|>", "<|split|>"
_SENTINEL_RE = re.compile(f"({re.escape(MARK)}|{re.escape(SPLIT)})")

# The prompt of models without a decision template, wrapped in their own chat template
DECISION_PROMPT = """
{%- macro text(x) %}{{ "" if x is none else x if x is string else x|tojson }}{% endmacro -%}
{%- set q = question -%}
{%- set ns = namespace(lines=[text(q.instructions)]) -%}
{%- for option in q.options -%}
    {%- set label, answer = q.labels[loop.index0], text(q.descriptions[loop.index0]) -%}
    {%- if q.type == "noul" -%}
        {%- set answer = answer or (option if q.labels != ["Yes", "No"] else "") -%}
        {%- set ns.lines = ns.lines + ([label ~ ": " ~ answer] if answer else []) -%}
    {%- else -%}
        {%- set answer = (option ~ ": " ~ answer if answer else option) if q.type == "choice" else answer -%}
        {%- set ns.lines = ns.lines + [label ~ ". " ~ answer] -%}
    {%- endif -%}
{%- endfor -%}
{{- text(state) ~ "\\n\\n" ~ ns.lines|select|join("\\n") ~ "\\n" -}}
{%- if q.type == "noul" -%}
    {{- "Answer " ~ q.labels[0] ~ " or " ~ q.labels[1] ~ " only." -}}
{%- else -%}
    {%- set kind = "letter" if q.labels == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: q.labels|length]|list else "label" -%}
    {{- "Answer with the " ~ kind ~ " of the " ~ ("option" if q.type == "choice" else "level") -}}
    {{- " that fits best (" ~ q.labels[0] ~ " to " ~ q.labels[-1] ~ ") only." -}}
{%- endif -%}
"""


@strict
@dataclass
class DecisionConfig:
    """
    Settings of a decision model, saved as `decision_config.json` (and its chat template as `decision_template.jinja`).

    Args:
        labels (`dict[str, list[str]]`, *optional*):
            Single-token labels of the options per question type, for heads that read them from the logits. Defaults to
            `Yes`/`No` for `noul` and `A`, `B`, `C`, ... otherwise.
        temperature (`dict[str, float]`, *optional*):
            Temperature per question type, applied to the option scores before the softmax. Defaults to 1.
        calibration (`dict[str, float]`, *optional*):
            Input-conditioned temperature (`bias`, `entropy`, `log_tokens`, `n_options`, `lo`, `hi`), see
            `CalibratedTemperatureProcessor`.
        noul_band (`list[float]`, *optional*):
            `[edge, slope]` of the band that keeps the probability of `noul` answers away from 0.5, see `NoulBandProcessor`.
        chat_template (`str`, *optional*):
            Jinja template rendering `state` with `question` (one prompt per question) or `questions` (one prompt for
            all of them). Defaults to a generic prompt in the tokenizer's chat template.
        noul_options (`list[str]`, *optional*, defaults to `["true", "false"]`):
            Order in which the template receives the two options of a `noul` question.
        sort_choices (`bool`, *optional*, defaults to `False`):
            Whether the template receives the options of a `choice` question sorted by key.
    """

    labels: dict[str, list[str]] = field(default_factory=dict)
    temperature: dict[str, float | int] = field(default_factory=dict)
    calibration: dict[str, float | int] | None = None
    noul_band: list[float] | None = None
    chat_template: str | None = None
    noul_options: list[str] = field(default_factory=lambda: ["true", "false"])
    sort_choices: bool = False

    def validate_noul_options(self):
        if sorted(self.noul_options) != ["false", "true"]:
            raise ValueError(f"noul_options must order 'true' and 'false', got {self.noul_options!r}.")

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "DecisionConfig":
        return cls() if path is None else cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs) -> "DecisionConfig":
        config = cls.from_file(cached_file(pretrained_model_name_or_path, DECISION_CONFIG_NAME, **kwargs))
        kwargs["_raise_exceptions_for_missing_entries"] = False
        if (template := cached_file(pretrained_model_name_or_path, DECISION_TEMPLATE_NAME, **kwargs)) is not None:
            config.chat_template = Path(template).read_text(encoding="utf-8")
        return config

    def save_pretrained(self, save_directory: str | Path):
        path, config = Path(save_directory), asdict(self)
        path.mkdir(parents=True, exist_ok=True)
        if (template := config.pop("chat_template")) is not None:
            (path / DECISION_TEMPLATE_NAME).write_text(template, encoding="utf-8")
        (path / DECISION_CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


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
        if kind == "noul":
            items = [(option, (criteria or {}).get(option)) for option in config.noul_options]
        elif kind == "choice":
            items = sorted(criteria.items()) if config.sort_choices else list(criteria.items())
        elif kind == "score":
            items = [(str(i), level) for i, level in enumerate(criteria)]
        else:
            raise ValueError(f"Question {name!r} has type {kind!r}, expected choice, score or noul.")
        labels = config.labels.get(kind) or (["Yes", "No"] if kind == "noul" else list(string.ascii_uppercase))
        if len(labels) < len(items):
            raise ValueError(f"labels.{kind} needs at least {len(items)} labels for question {name!r}.")
        options, descriptions = [key for key, _ in items], [description for _, description in items]
        legend = descriptions if kind == "score" else None
        return cls(name, kind, question.get("instructions"), options, labels[: len(items)], descriptions, legend)

    def answer(self, probabilities: list[float]) -> dict[str, Any]:
        """The System One answer, from the probabilities of the options."""
        p, n = dict(zip(self.options, probabilities)), len(probabilities)
        if self.type == "noul":
            return {"type": "noul", "noul": p["true"]}
        best = max(p, key=p.get)
        if self.type == "choice":
            confidence = 1.0 if n == 1 else max(0.0, (p[best] - 1 / n) / (1 - 1 / n))
            return {"type": "choice", "choice": best, "probabilities": p, "confidence": confidence}
        distance = sum(q * abs(i - int(best)) for i, q in enumerate(probabilities))
        uniform_distance = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
        confidence = 1.0 if n == 1 else max(0.0, 1 - distance / uniform_distance)
        score = sum(i * q for i, q in enumerate(probabilities))
        legend = dict(zip(self.options, self.legend))
        return {"type": "score", "score": score, "legend": legend, "probabilities": p, "confidence": confidence}


def label_token_ids(tokenizer, question: Question) -> list[int]:
    """The ids of the labels of `question`, which must be distinct single known tokens."""
    ids = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in question.labels]
    if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != len(ids) or [tokenizer.unk_token_id] in ids:
        raise ValueError(f"The labels of question {question.name!r} are not distinct single tokens for this model.")
    return [i[0] for i in ids]


def encode_with_marks(tokenizer, text: str, markers_re: re.Pattern | None = None) -> tuple[list[int], list[int]]:
    """Tokenize the text between sentinels on its own; each `<|mark|>` becomes the number of tokens before it."""
    ids, marks = [], []
    for piece in _SENTINEL_RE.split(text):
        if piece == MARK:
            marks.append(len(ids))
        elif piece and piece != SPLIT:
            ids += encode_marked_chat(tokenizer, piece, markers_re) if markers_re else tokenizer.encode(piece)
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

    def __init__(self, tokenizer, config: DecisionConfig | None = None, **chat_template_kwargs):
        self.tokenizer, self.config, self.chat_template_kwargs = (
            tokenizer,
            config or DecisionConfig(),
            chat_template_kwargs,
        )
        template = self.config.chat_template
        if template is None:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": "\0"}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prefix, suffix = chat.split("\0")
            template = f"{{% raw %}}{prefix}{{% endraw %}}{DECISION_PROMPT}{{% raw %}}{suffix}{{% endraw %}}{MARK}"
        self.template, self.one_prompt = (
            _compile_jinja_template(template),
            "questions" in _get_template_variables(template),
        )
        self.control_tokens_re, self.markers_re = _get_control_token_patterns(tokenizer._get_control_tokens(template))

    def __call__(self, request: dict[str, Any] | str) -> list[DecisionPrompt]:
        # One pass through JSON drops the request's sentinels and marks its control tokens (keys included) as text
        text = request if isinstance(request, str) else json.dumps(request, ensure_ascii=False)
        request = json.loads(mark_control_tokens(_SENTINEL_RE.sub("", text), self.control_tokens_re))
        questions = [Question.from_request(name, q, self.config) for name, q in request["questions"].items()]
        prompts = []
        for group in [questions] if self.one_prompt else [[question] for question in questions]:
            text = self.template.render(
                state=request["state"], question=group[0], questions=group, **self.chat_template_kwargs
            )
            prompts.append(DecisionPrompt(*encode_with_marks(self.tokenizer, text, self.markers_re), group))
        return prompts
