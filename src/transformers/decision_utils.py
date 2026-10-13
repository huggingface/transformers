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
Decision models answer typed questions (`choice`, `score`, `noul`) about a `state` with a probability for every allowed
answer, read from one forward pass instead of generated (the System One API of Jev-like models).

A decision model is a chat template, a backbone loaded with the Auto classes and a small head (`decision_head.safetensors`).
The template writes two zero-width sentinels: `<|mark|>` records how many tokens come before it (the positions the head
reads) and `<|split|>` only splits. The text between two sentinels is tokenized on its own, which reproduces models that
were trained on prompts tokenized piece by piece.
"""

import json
import math
import re
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .utils import cached_file
from .utils.chat_template_utils import (
    _compile_jinja_template,
    _get_control_token_patterns,
    _get_template_variables,
    encode_marked_chat,
    mark_control_tokens,
)


DECISION_CONFIG_NAME = "decision_config.json"
DECISION_HEAD_NAME = "decision_head.safetensors"
QUESTION_TYPES = ("choice", "score", "noul")
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


@dataclass
class DecisionConfig:
    """
    Settings of a decision model, saved as `decision_config.json`.

    Args:
        labels (`dict[str, list[str]]`, *optional*):
            Single-token labels of the options per question type, for heads that read them from the logits. Defaults to
            `Yes`/`No` for `noul` and `A`, `B`, `C`, ... otherwise.
        temperature (`dict[str, float]`, *optional*):
            Temperature per question type, applied to the option scores before the softmax. Defaults to 1.
        chat_template (`str`, *optional*):
            Jinja template rendering `state` with `question` (one prompt per question) or `questions` (one prompt for
            all of them). Defaults to a generic prompt in the tokenizer's chat template.
        head (`dict`, *optional*):
            `{"type": ..., **kwargs}` of the [`DecisionHead`] saved in `decision_head.safetensors`. Defaults to reading
            the logits of the `labels` at each `<|mark|>`.
        noul_options (`list[str]`, *optional*, defaults to `["true", "false"]`):
            Order in which the template receives the two options of a `noul` question.
        sort_choices (`bool`, *optional*, defaults to `False`):
            Whether the template receives the options of a `choice` question sorted by key.
        adapter (`str`, *optional*):
            Subfolder of the PEFT adapter to load onto the backbone, for decision LoRAs saved next to the original
            checkpoint.
    """

    labels: dict[str, list[str]] = field(default_factory=dict)
    temperature: dict[str, float] = field(default_factory=dict)
    chat_template: str | None = None
    head: dict[str, Any] | None = None
    noul_options: list[str] = field(default_factory=lambda: ["true", "false"])
    sort_choices: bool = False
    adapter: str | None = None

    def __post_init__(self):
        if not set(self.labels) | set(self.temperature) <= set(QUESTION_TYPES):
            raise ValueError(f"labels and temperature are keyed by question type, one of {QUESTION_TYPES}.")
        for kind, values in self.labels.items():
            if len(values) < (2 if kind == "noul" else 1) or len(set(values)) != len(values) or "" in values:
                raise ValueError(f"labels.{kind} must contain distinct non-empty labels (at least two for noul).")
        for kind, value in self.temperature.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"temperature.{kind} must be a finite number > 0, got {value!r}.")
        if self.head is not None and "type" not in self.head:
            raise ValueError("head must have a `type`.")
        if sorted(self.noul_options) != ["false", "true"]:
            raise ValueError(f"noul_options must order 'true' and 'false', got {self.noul_options!r}.")

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "DecisionConfig":
        """Load a `decision_config.json` file, or return the defaults when no file is given."""
        return cls() if path is None else cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs) -> "DecisionConfig":
        """Load the `decision_config.json` of a local directory or a Hub repo (`kwargs` go to `cached_file`)."""
        return cls.from_file(cached_file(pretrained_model_name_or_path, DECISION_CONFIG_NAME, **kwargs))

    def save_pretrained(self, save_directory: str | Path):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / DECISION_CONFIG_NAME).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


@dataclass
class Question:
    """One System One question, with its options in the order the template renders them."""

    name: str
    type: Literal["choice", "score", "noul"]
    instructions: str | dict[str, Any] | list[Any] | None
    # Response keys: choice keys, score indices, or true/false
    options: list[str]
    labels: list[str]
    descriptions: list[str | dict[str, Any] | list[Any] | None]
    legend: list[str | dict[str, Any] | list[Any]] | None = None

    @classmethod
    def from_request(cls, name: str, question: dict[str, Any], config: DecisionConfig) -> "Question":
        kind, criteria, legend = question["type"], question.get("criteria"), None
        if kind == "noul":
            options = list(config.noul_options)
            descriptions = [(criteria or {}).get(option) for option in options]
        elif kind == "choice":
            items = sorted(criteria.items()) if config.sort_choices else list(criteria.items())
            options, descriptions = [key for key, _ in items], [description for _, description in items]
        elif kind == "score":
            options, descriptions, legend = [str(i) for i in range(len(criteria))], list(criteria), list(criteria)
        else:
            raise ValueError(f"Question {name!r} has type {kind!r}, expected one of {QUESTION_TYPES}.")
        labels = config.labels.get(kind) or (["Yes", "No"] if kind == "noul" else list(string.ascii_uppercase))
        if len(labels) < len(options):
            raise ValueError(f"labels.{kind} needs at least {len(options)} labels for question {name!r}.")
        return cls(name, kind, question.get("instructions"), options, labels[: len(options)], descriptions, legend)

    def answer(self, probabilities: list[float]) -> dict[str, Any]:
        """The System One answer, from the probabilities of the options."""
        option_probabilities = dict(zip(self.options, probabilities))
        if self.type == "noul":
            return {"type": "noul", "noul": option_probabilities["true"]}
        n, best = len(probabilities), max(option_probabilities, key=option_probabilities.get)
        if self.type == "choice":
            confidence = 1.0 if n == 1 else max(0.0, (option_probabilities[best] - 1 / n) / (1 - 1 / n))
            return {"type": "choice", "choice": best, "probabilities": option_probabilities, "confidence": confidence}
        distance = sum(p * abs(i - int(best)) for i, p in enumerate(probabilities))
        uniform_distance = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
        return {
            "type": "score",
            "score": sum(int(level) * p for level, p in option_probabilities.items()),
            "legend": dict(zip(self.options, self.legend)),
            "probabilities": option_probabilities,
            "confidence": 1.0 if n == 1 else max(0.0, 1 - distance / uniform_distance),
        }


def label_token_ids(tokenizer, question: Question) -> list[int]:
    """Tokenize each label of `question` on its own, requiring distinct single-token labels."""
    label_tokens = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in question.labels]
    if any(len(ids) != 1 for ids in label_tokens):
        raise ValueError(f"The labels of question {question.name!r} are not single tokens for this model.")
    label_ids = [ids[0] for ids in label_tokens]
    if tokenizer.unk_token_id in label_ids:
        raise ValueError(f"The labels of question {question.name!r} contain an unknown token for this model.")
    if len(set(label_ids)) != len(label_ids):
        raise ValueError(f"Two labels of question {question.name!r} share a token for this model.")
    return label_ids


def encode_with_marks(tokenizer, text: str, markers_re: re.Pattern | None = None) -> tuple[list[int], list[int]]:
    """
    Tokenize the text between sentinels on its own, drop `<|split|>`, and replace each `<|mark|>` by the number of
    tokens before it. `markers_re` encodes the control tokens marked by `mark_control_tokens` as text.

    Returns:
        `tuple[list[int], list[int]]`: The input ids and the marks.
    """
    pieces = _SENTINEL_RE.split(text)
    if len(pieces) > 1 and _prepends_to_pieces(tokenizer):
        # ponytail: such tokenizers add a prefix space to every piece, until tokenizers can encode from an offset
        raise ValueError(f"`{MARK}` and `{SPLIT}` need a tokenizer that does not prepend a prefix space to its input.")
    ids, marks = [], []
    for piece in pieces:
        if piece == MARK:
            marks.append(len(ids))
        elif piece and piece != SPLIT:
            ids += encode_marked_chat(tokenizer, piece, markers_re) if markers_re else tokenizer.encode(piece)
    return ids, marks


def _prepends_to_pieces(tokenizer) -> bool:
    """Whether the normalizer or pre-tokenizer prepends a prefix (Metaspace, Prepend, ByteLevel with a prefix space)."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return True
    stack = [json.loads(backend.to_str()).get(key) for key in ("normalizer", "pre_tokenizer")]
    while stack:
        part = stack.pop()
        if isinstance(part, list):
            stack += part
        elif isinstance(part, dict):
            if (
                part.get("type") == "Prepend"
                or part.get("add_prefix_space")
                or part.get("prepend_scheme") in ("first", "always")
            ):
                return True
            stack += [part.get("normalizers"), part.get("pretokenizers")]
    return False


@dataclass
class DecisionPrompt:
    input_ids: list[int]
    marks: list[int]
    questions: list[Question]


class DecisionTemplate:
    """
    Renders a System One request into [`DecisionPrompt`]s: one per question, or a single one when the template uses
    `questions`. Without a decision template, the generic `DECISION_PROMPT` goes through the tokenizer's chat template.
    """

    def __init__(self, tokenizer, chat_template: str | None = None, **chat_template_kwargs):
        if chat_template is None:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": "\0"}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prefix, suffix = chat.split("\0")
            chat_template = (
                f"{{% raw %}}{prefix}{{% endraw %}}{DECISION_PROMPT}{{% raw %}}{suffix}{{% endraw %}}{MARK}"
            )
        self.tokenizer = tokenizer
        self.template = _compile_jinja_template(chat_template)
        self.chat_template_kwargs = chat_template_kwargs
        self.one_prompt = "questions" in _get_template_variables(chat_template)
        self.control_tokens_re, self.markers_re = _get_control_token_patterns(
            tokenizer._get_control_tokens(chat_template)
        )

    def __call__(self, state: Any, questions: list[Question]) -> list[DecisionPrompt]:
        # The request's sentinels are dropped (a `tojson(ensure_ascii=true)` would unwrap marked ones) and its control
        # tokens are encoded as text
        state = mark_control_tokens(_drop_sentinels(state), self.control_tokens_re)
        questions = [
            Question(**mark_control_tokens(_drop_sentinels(asdict(q)), self.control_tokens_re)) for q in questions
        ]
        prompts = []
        for group in [questions] if self.one_prompt else [[question] for question in questions]:
            text = self.template.render(state=state, question=group[0], questions=group, **self.chat_template_kwargs)
            prompts.append(DecisionPrompt(*encode_with_marks(self.tokenizer, text, self.markers_re), group))
        return prompts


def _drop_sentinels(obj: Any) -> Any:
    if isinstance(obj, str):
        return _SENTINEL_RE.sub("", obj)
    if isinstance(obj, dict):
        return {key: _drop_sentinels(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_drop_sentinels(value) for value in obj]
    return obj
