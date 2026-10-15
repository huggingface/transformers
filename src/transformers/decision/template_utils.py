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
in pieces: the text between two sentinels is tokenized on its own. Images are `<|image:digest|>` sentinels, written in
place of the placeholders of a request's `assets` and expanded by the processor.
"""

import difflib
import functools
import hashlib
import json
import os
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from huggingface_hub.dataclasses import strict
from huggingface_hub.errors import StrictDataclassError

from ..image_utils import load_image
from ..tokenization_utils_base import BatchEncoding
from ..utils.chat_template_utils import (
    _compile_jinja_template,
    _get_control_token_patterns,
    _get_template_variables,
    encode_marked_chat,
    mark_control_tokens,
)
from .configuration_utils import DecisionConfig


MARK, SPLIT, IMAGE = "<|mark|>", "<|split|>", "<|image:{}|>"
_IMAGE_RE = re.compile(r"<\|image:([0-9a-f]{16})\|>")
_SENTINEL_RE = re.compile(f"({re.escape(MARK)}|{re.escape(SPLIT)}|{_IMAGE_RE.pattern.replace('(', '(?:')})")


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
    # The backbone's other inputs: per-token ones like `token_type_ids`, and the processor's tensors like `pixel_values`
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)

    def backbone_inputs(self) -> BatchEncoding:
        """The inputs of the backbone, as a batch of one: `input_ids` and the `backbone_kwargs`."""
        inputs = {"input_ids": [self.input_ids]}
        for key, value in self.backbone_kwargs.items():
            inputs[key] = [value] if isinstance(value, list) else value
        return BatchEncoding(inputs, tensor_type="pt")

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
    `questions`. Without a decision template, the generic `DECISION_PROMPT` goes in the chat template of the processor.

    Args:
        processor (`PreTrainedTokenizerBase` or `ProcessorMixin`):
            The tokenizer, or the processor of models that see images.
        config ([`DecisionConfig`], *optional*):
            The settings of the decision model.
        questions (`dict`, *optional*):
            The questions asked when a call gets a bare state, until a request brings others.
    """

    def __init__(self, processor, config: DecisionConfig | None = None, questions: dict[str, Any] | None = None):
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.processor = processor if getattr(processor, "image_processor", None) is not None else None
        self.config = config or DecisionConfig()
        template = self.config.chat_template
        if template is None:
            prefix, suffix = self._render_chat("\0").split("\0")
            template = f"{{% raw %}}{prefix}{{% endraw %}}{DECISION_PROMPT}{{% raw %}}{suffix}{{% endraw %}}{MARK}"
        self.template = _compile_jinja_template(template)
        self.one_prompt = "questions" in _get_template_variables(template)
        control_tokens = self.tokenizer._get_control_tokens(template) if self.config.sanitize_control_tokens else ()
        self.markers_re = _get_control_token_patterns(control_tokens)[1]
        # Requests are marked as JSON text, where a token like `<|"|>` reads `<|\"|>`
        json_tokens = tuple(json.dumps(token, ensure_ascii=False)[1:-1] for token in control_tokens)
        self.control_tokens_re = _get_control_token_patterns(json_tokens)[0]
        # What the chat template writes for an image, which the processor expands into the image's tokens
        self.image_placeholder = None
        if self.processor is not None:
            with_image, without = self._render_chat("\0", images=1), self._render_chat("\0")
            start = len(os.path.commonprefix([with_image, without]))
            end = len(os.path.commonprefix([with_image[start:][::-1], without[start:][::-1]]))
            self.image_placeholder = with_image[start : len(with_image) - end]
        # Each distinct set of questions is validated once: keyed by its JSON, a repeated set is a cache hit
        self._parse_questions = functools.lru_cache(maxsize=256)(
            lambda questions: parse_questions(json.loads(questions), self.config)
        )
        # The questions asked when a call gets a bare state: the ones given here, then those of the last request
        self.questions = None if questions is None else self.parse_questions(self.sanitize(questions))
        self.question_assets = {}

    def _render_chat(self, text: str, images: int = 0) -> str:
        if self.processor is None:
            content = text
        else:
            content = [{"type": "image"}] * images + [{"type": "text", "text": text}]
        renderer, kwargs = self.processor or self.tokenizer, {"enable_thinking": False}
        if self.processor is not None and "enable_thinking" not in _get_template_variables(renderer.chat_template):
            kwargs = {}  # processors pass the variables their template doesn't use to the processor, with a warning
        return renderer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, **kwargs
        )

    def sanitize(self, value: Any) -> Any:
        """One pass through JSON drops the sentinels and marks the control tokens (keys included) as text."""
        text = json.dumps(value, ensure_ascii=False)
        return json.loads(mark_control_tokens(_SENTINEL_RE.sub("", text), self.control_tokens_re))

    def parse_questions(self, questions: dict[str, Any]) -> list[Question]:
        """[`parse_questions`], cached: the returned list is shared between calls and must not be modified."""
        return self._parse_questions(json.dumps(questions, ensure_ascii=False))

    def encode_images(self, images: list) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
        """
        The tokens of each image as the processor expands them (`input_ids` and per-token inputs like `token_type_ids`),
        and the processor's tensors for all of them, from one call so that it batches (and pads) them like for a chat.
        """
        if self.processor is None:
            raise ValueError("This model doesn't see images: its repo has no processor with an image processor.")
        separator = self.tokenizer.eos_token
        text = separator.join([self.image_placeholder] * len(images))
        outputs = self.processor(text=text, images=images, return_tensors="pt", add_special_tokens=False)
        shape = outputs["input_ids"].shape
        per_token = {
            key: value[0].tolist()
            for key, value in outputs.items()
            if key != "attention_mask" and getattr(value, "shape", None) == shape
        }
        tensors = {key: value for key, value in outputs.items() if key not in per_token and key != "attention_mask"}
        tokens, start, separator_id = [], 0, self.tokenizer.convert_tokens_to_ids(separator)
        for end in [*(i for i, token in enumerate(per_token["input_ids"]) if token == separator_id), None]:
            tokens.append({key: values[start:end] for key, values in per_token.items()})
            start = None if end is None else end + 1
        return tokens, tensors

    def __call__(
        self, request: dict[str, Any] | str | Any, assets: dict[str, Any] | None = None
    ) -> list[DecisionPrompt]:
        """
        Renders `request`: a `{"state": ..., "questions": ..., "assets": ...}` dict or its JSON, whose questions become
        the current ones, or a bare state (any other value), asked the current questions.

        `assets` map placeholders to images, as in the System One API: each placeholder in a string of the state, the
        instructions or the criteria is replaced with its image. An image is an asset (`{"type": "image", "format":
        "png", "data_base64": ...}`), or anything [`~image_utils.load_image`] takes, like a PIL image, a path or a URL.
        """
        if isinstance(request, str):  # parsed before sanitizing, so that `\u` escapes can't hide sentinels or tokens
            try:
                parsed = json.loads(request)
            except json.JSONDecodeError:
                parsed = None
            request = parsed if isinstance(parsed, dict) and "questions" in parsed else request
        if isinstance(request, dict) and "questions" in request:
            if "state" not in request:
                raise ValueError(
                    'A request is {"state": ..., "questions": {name: question, ...}}, `state` is missing.'
                )
            assets = {**request.get("assets", {}), **(assets or {})}
            request = _with_sentinels(
                self.sanitize({"state": request["state"], "questions": request["questions"]}), assets
            )
            self.questions, state = self.parse_questions(request["questions"]), request["state"]
            # The images of the questions stay with them, for the bare states asked next
            questions_text = json.dumps(request["questions"])
            self.question_assets = {name: asset for name, asset in assets.items() if _sentinel(name) in questions_text}
        elif self.questions is None:
            raise ValueError(
                'No questions to ask yet: send {"state": ..., "questions": {...}} once, or set `questions` up front.'
            )
        else:
            assets = {**self.question_assets, **(assets or {})}
            state = _with_sentinels(self.sanitize(request), assets)
        by_digest, images = {_sentinel(name): asset for name, asset in assets.items()}, {}
        prompts = []
        for group in [self.questions] if self.one_prompt else [[question] for question in self.questions]:
            text = self.template.render(state=state, question=group[0], questions=group)
            used = []
            for digest in _IMAGE_RE.findall(text):
                if digest not in images:
                    images[digest] = _load_asset(by_digest[IMAGE.format(digest)])
                used.append(images[digest])
            expansions, tensors = self.encode_images(used) if used else ([], {})
            # Each image as the text the processor expands it into, so that the prompt tokenizes like a chat
            texts = iter(self.tokenizer.batch_decode([expansion["input_ids"] for expansion in expansions]))
            input_ids, marks = encode_with_marks(
                self.tokenizer, _IMAGE_RE.sub(lambda _: next(texts), text), self.markers_re
            )
            prompts.append(
                DecisionPrompt(input_ids, marks, group, {**_image_inputs(input_ids, expansions), **tensors})
            )
        return prompts


def _image_inputs(input_ids: list[int], expansions: list[dict[str, list[int]]]) -> dict[str, list[int]]:
    """
    The per-token inputs of the images (like `token_type_ids`), at their tokens in `input_ids` and 0 elsewhere. Each
    image is found by its tokens where these inputs are set, which are special tokens that tokenize the same anywhere.
    """
    inputs, cursor = {}, 0
    for expansion in expansions:
        ids, streams = expansion["input_ids"], {key: value for key, value in expansion.items() if key != "input_ids"}
        set_positions = [i for i in range(len(ids)) if any(values[i] for values in streams.values())]
        if not set_positions:
            continue
        low, high = set_positions[0], set_positions[-1] + 1
        start = next(i for i in range(cursor, len(input_ids)) if input_ids[i : i + high - low] == ids[low:high])
        for key, values in streams.items():
            inputs.setdefault(key, [0] * len(input_ids))[start : start + high - low] = values[low:high]
        cursor = start + high - low
    return inputs


def _sentinel(placeholder: str) -> str:
    return IMAGE.format(hashlib.sha256(placeholder.encode()).hexdigest()[:16])


def _with_sentinels(value: Any, assets: dict[str, Any]) -> Any:
    """Replaces the placeholders of `assets` in the strings of `value` with their image sentinels, longest first."""
    if isinstance(value, str):
        for placeholder in sorted(assets, key=len, reverse=True):
            value = value.replace(placeholder, _sentinel(placeholder))
        return value
    if isinstance(value, dict):
        return {key: _with_sentinels(item, assets) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_sentinels(item, assets) for item in value]
    return value


def _load_asset(asset: Any):
    if isinstance(asset, dict):  # a System One asset
        if asset.get("type") != "image" or "data_base64" not in asset:
            raise ValueError(f"Assets are images with `data_base64`, got {sorted(asset)}.")
        asset = f"data:image/{asset.get('format', 'png')};base64,{asset['data_base64']}"
    return load_image(asset)


__all__ = [
    "DECISION_PROMPT",
    "MARK",
    "QUESTION_TYPES",
    "SPLIT",
    "DecisionPrompt",
    "IMAGE",
    "DecisionTemplate",
    "Question",
    "encode_with_marks",
    "label_token_ids",
    "parse_questions",
]
