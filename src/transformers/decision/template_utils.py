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
TLDR, decison models just format the questions so that a single forward pass generates an answer "where" you need it.
Depending on the method, they add a `[MASK]` token and read the prediction from there, or ...

In any case, the `DecisionPrompt`, the standard way to pass inputs to such models, needs to be transformed into token ids.
There is a lot of string manipulation involed ans as such we leverage chat-templates.
Now WHY do we need the rest of the code?


"""

import copy
import functools
import json
import re
import string
from dataclasses import asdict, dataclass, field, fields
from difflib import get_close_matches
from typing import Any, Literal

from huggingface_hub.dataclasses import strict
from huggingface_hub.errors import StrictDataclassError

from ..image_utils import load_image
from ..tokenization_utils_base import AddedToken, BatchEncoding
from ..utils.chat_template_utils import _compile_jinja_template, _get_template_variables
from .configuration_utils import DecisionConfig


MARK, SPLIT = "<|mark|>", "<|split|>"


# Text, a JSON object or an array, as in the System One API
Content = str | dict | list


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


# Models that don't ship a decision_template.jinja will default to this one
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
    questions_schemes = {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}
    questions_types = tuple(questions_schemes)

    @classmethod
    def from_request(cls, name: str, question: dict[str, Any], config: DecisionConfig) -> "Question":
        kind = question.get("type") if isinstance(question, dict) else None
        if kind not in cls.questions_schemes:
            close = get_close_matches(str(kind).lower(), cls.questions_types, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else f" For example: {NoulQuestion.__doc__}"
            got = f"type {kind!r}" if isinstance(question, dict) else repr(question)
            raise ValueError(
                f"Question {name!r}: expected a `type` among {', '.join(cls.questions_types)}, got {got}.{hint}"
            )

        schema = cls.questions_schemes[kind]
        try:
            question = asdict(schema(**question))
        except (TypeError, StrictDataclassError) as error:
            if field := re.match(r"Validation error for field '(\w+)'", str(error)):
                reason = f"`{field[1]}` can't be {question[field[1]]!r}"
            else:  # an unknown field (`TypeError`) or a class validator
                reason = str(error.__cause__ or error).split("() ", 1)[-1].split(". Did you mean", 1)[0]
                reason = reason.replace("got an unexpected keyword argument", "unknown field")
                unknown = re.match(r"unknown field '(\w+)'", reason)
                if unknown and (close := get_close_matches(unknown[1], [f.name for f in fields(schema)], n=1)):
                    reason += f". Did you mean {close[0]!r}?"
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
        `num_options` and `question_type_ids` (index in `Question.questions_types`) of each question.
        """
        inputs = {
            "input_ids": self.input_ids,
            "marks": self.marks,
            "num_options": [len(question.options) for question in self.questions],
            "question_type_ids": [type(question).questions_types.index(question.type) for question in self.questions],
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
    Renders a System One request into [`DecisionPrompt`]s with `apply_chat_template`: one per question, or one for all
    when the decision template uses `questions`. A decision template gets the request as `state` and `question` (or
    `questions`), with the images of its `assets` as `{"type": "image", "image": ...}` parts of `state`, and writes
    `<|mark|>` where a head reads and `<|split|>` where the prompt is tokenized in pieces. Without one, the generic
    `DECISION_PROMPT` is the user message of the model's chat template, read after the generation prompt.

    Args:
        processor (`PreTrainedTokenizerBase` or `ProcessorMixin`):
            The tokenizer, or the processor of models that see images.
        config ([`DecisionConfig`], *optional*):
            The settings of the decision model.
        questions (`dict`, *optional*):
            The questions asked when a call gets a bare state, until a request brings others.
    """

    def __init__(self, processor, config: DecisionConfig | None = None, questions: dict[str, Any] | None = None):
        # A copy with the sentinels as special tokens: they split the text they're in, and requests can't write them
        processor = copy.deepcopy(processor)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.tokenizer.add_tokens([AddedToken(token, special=True, normalized=False) for token in (MARK, SPLIT)])
        self.mark_id, self.split_id = self.tokenizer.convert_tokens_to_ids([MARK, SPLIT])
        self.processor = processor if getattr(processor, "image_processor", None) is not None else None
        self.config = config or DecisionConfig()
        self.decision_template = self.config.chat_template
        self.prompt = _compile_jinja_template(self.decision_template or DECISION_PROMPT)
        self.variables = _get_template_variables(self.decision_template or DECISION_PROMPT)
        self.one_prompt = "questions" in self.variables
        self.chat_kwargs = {"enable_thinking": False}
        if self.processor is not None and "enable_thinking" not in _get_template_variables(processor.chat_template):
            self.chat_kwargs = {}  # processors pass the variables their template doesn't use to the processor
        # Each distinct set of questions is validated once: keyed by its JSON, a repeated set is a cache hit
        self._parse_questions = functools.lru_cache(maxsize=256)(
            lambda questions: parse_questions(json.loads(questions), self.config)
        )
        # The questions asked when a call gets a bare state: the ones given here, then those of the last request
        self.questions = None if questions is None else self.parse_questions(questions)
        self.question_assets = {}

    def parse_questions(self, questions: dict[str, Any]) -> list[Question]:
        """[`parse_questions`], cached: the returned list is shared between calls and must not be modified."""
        return self._parse_questions(json.dumps(questions, ensure_ascii=False))

    def __call__(
        self, request: dict[str, Any] | str | Any, assets: dict[str, Any] | None = None
    ) -> list[DecisionPrompt]:
        """
        Answers a decision `request`: a `{"state": ..., "questions": ..., "assets": ...}` as a dict or a JSON.
        The question is cached, this way you can pass only a `str` (a state) and it will re-use the question.

        `assets` map placeholders to images, as in the System One API: each placeholder in a string of the state, the
        instructions or the criteria is replaced with its image. An image is an asset (`{"type": "image", "format":
        "png", "data_base64": ...}`), or anything [`~image_utils.load_image`] takes, like a PIL image, a path or a URL.
        """
        if isinstance(request, str):  # a request as JSON, parsed so that `\u` escapes can't hide control tokens
            try:
                parsed = json.loads(request)
            except json.JSONDecodeError:
                parsed = None
            request = parsed if isinstance(parsed, dict) and "questions" in parsed else request
        if not self.config.sanitize_control_tokens:  # templates that escape control tokens still get no sentinels
            request = _drop_sentinels(request)
        if isinstance(request, dict) and "questions" in request:
            if "state" not in request:
                raise ValueError(
                    'A request is {"state": ..., "questions": {name: question, ...}}, `state` is missing.'
                )
            assets = {**request.get("assets", {}), **(assets or {})}
            self.questions, state = self.parse_questions(request["questions"]), request["state"]
            questions_text = json.dumps(request["questions"], ensure_ascii=False)
            self.question_assets = {name: asset for name, asset in assets.items() if name in questions_text}
        elif self.questions is None:
            raise ValueError(
                'No questions to ask yet: send {"state": ..., "questions": {...}} once, or set `questions` up front.'
            )
        else:
            assets, state = {**self.question_assets, **(assets or {})}, request
        if assets and self.processor is None:
            raise ValueError("Images need a model whose processor handles them.")
        load = functools.cache(lambda name: _load_asset(assets[name]))
        sanitize = self.config.sanitize_control_tokens
        prompts = []
        for group in [self.questions] if self.one_prompt else [[question] for question in self.questions]:
            if self.decision_template is None:
                text = self.prompt.render(state=state, question=group[0])
                content = _split_at_placeholders(text, assets, load) if self.processor is not None else text
                kwargs = {"sanitize_control_tokens": sanitize}
            else:
                variables = {"state": state, "question": asdict(group[0]), "questions": [asdict(q) for q in group]}
                variables = {name: value for name, value in variables.items() if name in self.variables}
                variables = _image_parts(variables, assets, load) if assets else variables
                # The processor takes the images in the order the template renders them
                found = []
                numbered = _number_images(variables, found)
                order = re.findall("\0(\\d+)\0", self.prompt.render(**numbered))
                images = [found[int(index)] for index in order]
                content, kwargs = "", {"chat_template": self.decision_template, **variables}
                kwargs["sanitize_control_tokens"] = list(variables) if sanitize else False
                if images:
                    kwargs["processor_kwargs"] = {"images": images}
            inputs = (self.processor or self.tokenizer).apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **self.chat_kwargs,
                **kwargs,
            )
            prompts.append(self._prompt(inputs, group))
        return prompts

    def _prompt(self, inputs: dict[str, Any], questions: list[Question]) -> DecisionPrompt:
        """Drops the sentinels from the inputs: the marks are the number of tokens before each `<|mark|>`."""
        ids = inputs["input_ids"][0]
        token_to_keep = (ids != self.mark_id) & (ids != self.split_id)
        marks = token_to_keep.cumsum(0)[ids == self.mark_id].tolist()
        per_token = {
            key: value[:, token_to_keep]
            for key, value in inputs.items()
            if getattr(value, "shape", ())[-1:] == ids.shape
        }
        tensors = {key: value for key, value in inputs.items() if key not in per_token}
        input_ids = per_token.pop("input_ids")[0].tolist()
        per_token.pop("attention_mask", None)
        # The default prompt is read after the generation prompt
        return DecisionPrompt(input_ids, marks or [len(input_ids)], questions, {**per_token, **tensors})


def _drop_sentinels(request: Any) -> Any:
    if isinstance(request, dict) and "assets" in request:  # images aren't JSON, and have no sentinels
        return {**_drop_sentinels({k: v for k, v in request.items() if k != "assets"}), "assets": request["assets"]}
    return json.loads(json.dumps(request, ensure_ascii=False).replace(MARK, "").replace(SPLIT, ""))


def _image_parts(value: Any, assets: dict[str, Any], load) -> Any:
    """Strings with asset placeholders become lists of text and image parts, in `value` and what it contains."""
    if isinstance(value, str):
        parts = _split_at_placeholders(value, assets, load)
        return parts if any(part["type"] == "image" for part in parts) else value
    if isinstance(value, dict):
        return {key: _image_parts(item, assets, load) for key, item in value.items()}
    if isinstance(value, list):
        return [_image_parts(item, assets, load) for item in value]
    return value


def _number_images(value: Any, images: list) -> Any:
    """`value` with each image part as a text part holding its index in `images`, where the image is appended."""
    if isinstance(value, dict) and value.get("type") == "image":
        images.append(value["image"])
        return {"type": "text", "text": f"\0{len(images) - 1}\0"}
    if isinstance(value, dict):
        return {key: _number_images(item, images) for key, item in value.items()}
    if isinstance(value, list):
        return [_number_images(item, images) for item in value]
    return value


def _split_at_placeholders(text: str, assets: dict[str, Any], load) -> list[dict[str, Any]]:
    """The content blocks of `text`: its text, with an image block where a placeholder of `assets` is."""
    if not assets:
        return [{"type": "text", "text": text}]
    pattern = re.compile("|".join(re.escape(name) for name in sorted(assets, key=len, reverse=True)))
    content, last = [], 0
    for match in pattern.finditer(text):
        content += [{"type": "text", "text": text[last : match.start()]}, {"type": "image", "image": load(match[0])}]
        last = match.end()
    content.append({"type": "text", "text": text[last:]})
    return [block for block in content if block.get("text") != ""]


def _load_asset(asset: Any):
    if isinstance(asset, dict):  # a System One asset
        if asset.get("type") != "image" or "data_base64" not in asset:
            raise ValueError(f"Assets are images with `data_base64`, got {sorted(asset)}.")
        asset = f"data:image/{asset.get('format', 'png')};base64,{asset['data_base64']}"
    return load_image(asset)


__all__ = [
    "DECISION_PROMPT",
    "MARK",
    "SPLIT",
    "DecisionPrompt",
    "DecisionTemplate",
    "Question",
    "label_token_ids",
    "parse_questions",
]
