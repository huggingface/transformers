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
TLDR: decision models answer a question in a single forward pass, by reading the answer at a known position of the
prompt. A request is `{"state": ..., "questions": {name: question}}`, rendered with a chat template into a prompt per
question (or one for all of them, when the template uses `questions`).

A model with a `decision_template.jinja` places two sentinels, e.g. `ArthurZ/clef-flash` (system prompt cut):

    <|im_start|>user
    STATE:
    <|split|>{"message":"WIN a FREE iPhone, click here!"}<|split|>

    SCHEMA FIELDS:
    <|split|>
    FIELD 1
    ID: tone
    TYPE: choice
    INSTRUCTION: <|mark|>What is the tone?<|mark|>
    ALLOWED OPTIONS:
    OPTION 1: <|mark|>{"option_id":"calm"}<|mark|>
    OPTION 2: <|mark|>{"description":"Pressures the reader","option_id":"pushy"}<|mark|>
    END FIELD
    <|split|>
    <|im_end|>
    <|im_start|>assistant

- `<|mark|>` (▮ below): where the head reads, which depends on the head:

      jev    ... [decision]:▮                       reads 1 token: the one before each mark (one per question)
      kev    ... opt A▮ opt B▮ opt C▮ state▮         reads 1 token before each mark, one mark after each option
      von    ... [MASK]▮ [MASK]▮ ... ▮              reads 1 token per option; the last mark ends the block
      clef   INSTRUCTION: ▮What is the tone?▮        reads every token BETWEEN a pair of marks, pooled
             OPTION 1:    ▮{"option_id":"calm"}▮     (hidden states + output embeddings of the span)
             OPTION 2:    ▮{"option_id":"pushy"}▮    → spans sit mid-sequence, not at the end

- `<|split|>`: the text between two sentinels is tokenized on its own, like the original code did, so the state can't
  merge into a token with `STATE:\n` around it.

Both are dropped from the input ids: a [`DecisionPrompt`] keeps the ids and `marks`, the number of tokens before each
`<|mark|>` (here `[70, 75, 85, 92, 98, 112]`).

Any other model gets `DECISION_PROMPT` as the user message of its own chat template, and the logits of the option
labels (`A`, `B`, ... or `Yes`/`No`) are read after the generation prompt, e.g. for `Qwen/Qwen2.5-0.5B-Instruct`:

    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {"message": "WIN a FREE iPhone, click here!"}

    What is the tone?
    A. calm
    B. pushy: Pressures the reader
    Answer with the letter of the option that fits best (A to B) only.<|im_end|>
    <|im_start|>assistant

The rest of this file handles the different input validation, escaping user text from the mark/split tokens,
and replacing images with the corresponding assets.

We support arbitrary kind of inputs, users should be able to expand it to his needs.
"""

import copy
import functools
import json
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from huggingface_hub.dataclasses import strict
from huggingface_hub.errors import StrictDataclassError

from ..image_utils import load_image
from ..tokenization_utils_base import AddedToken, BatchEncoding
from ..utils.chat_template_utils import _compile_jinja_template, _get_template_variables
from .configuration_utils import DecisionConfig


MARK, SPLIT = "<|mark|>", "<|split|>"

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


# Models that don't ship a decision_template.jinja default to this one, the prompt of the `v1/systemone` endpoint of
# `transformers serve`: any chat model answers with the next-token logits of the option labels
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
    questions_schemes = {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}
    questions_types = tuple(questions_schemes)

    @classmethod
    def from_request(cls, name: str, question: dict[str, Any], config: DecisionConfig) -> "Question":
        kind = question.get("type") if isinstance(question, dict) else None
        if kind not in cls.questions_schemes:
            got = kind if isinstance(question, dict) else question
            raise ValueError(
                f"Question {name!r}: `type` must be one of {', '.join(cls.questions_types)}, got {got!r}."
            )

        schema = cls.questions_schemes[kind]
        try:
            question = asdict(schema(**question))
        except (TypeError, StrictDataclassError) as error:
            raise ValueError(f"Question {name!r}: {error} (a {kind} question looks like {schema.__doc__})") from error

        criteria = question["criteria"]
        if kind == "noul":
            criteria = {option: (criteria or {}).get(option) for option in config.noul_options}
        elif kind == "choice":
            criteria = dict(criteria)
        else:
            criteria = {str(index): level for index, level in enumerate(criteria)}

        labels = config.labels.get(kind) or (["Yes", "No"] if kind == "noul" else list(string.ascii_uppercase))
        if len(labels) < len(criteria):
            raise ValueError(
                f"Question {name!r}: this model reads at most {len(labels)} options for a {kind} question, got "
                f"{len(criteria)} (`labels.{kind}` in its decision config)."
            )
        options, descriptions = list(criteria), list(criteria.values())
        return cls(name, kind, question.get("instructions"), options, labels[: len(options)], descriptions)

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
        return {
            "type": "score",
            "score": score,
            "legend": dict(zip(self.options, self.descriptions)),
            "confidence": confidence,
            "probabilities": by_option,
        }


def label_token_ids(tokenizer, labels: tuple[str, ...]) -> list[int]:
    """
    The ids of the `labels` of a question, which must be distinct single known tokens. Only for models without a decision
    head (e.g. `ArthurZ/d3-nano`): the score of each option is the logit of its label token, at the last token before
    the mark. Heads (clef, jev, ...) read hidden states instead.
    """
    ids = tokenizer(list(labels), add_special_tokens=False)["input_ids"]
    label_ids = [token_ids[0] for token_ids in ids if len(token_ids) == 1 and token_ids[0] != tokenizer.unk_token_id]
    if len(set(label_ids)) != len(ids):
        raise ValueError(f"The labels {list(labels)} are not distinct single tokens for this model.")
    return label_ids


@dataclass
class DecisionPrompt:
    """
    A rendered prompt, without its sentinels.

    Args:
        input_ids (`list[int]`):
            The token ids, without `<|mark|>` and `<|split|>`.
        marks (`list[int]`):
            The number of tokens before each `<|mark|>`: the lm_head reads at `marks[i] - 1` (the token right before the
            mark), or the span between two marks. Without marks in the template, `[len(input_ids)]`, the last token.
        questions (`list[Question]`):
            The questions the prompt asks, one or all of them.
        backbone_kwargs (`dict`, *optional*):
            The backbone's other inputs: per-token ones like `token_type_ids`, and the processor's tensors like
            `pixel_values`.
    """

    input_ids: list[int]
    marks: list[int]
    questions: list[Question]
    backbone_kwargs: dict[str, Any] = field(default_factory=dict)

    def backbone_inputs(self) -> BatchEncoding:
        """The inputs of the backbone, as a batch of one: `input_ids` and the `backbone_kwargs`."""
        kwargs = {key: [value] if isinstance(value, list) else value for key, value in self.backbone_kwargs.items()}
        return BatchEncoding({"input_ids": [self.input_ids], **kwargs}, tensor_type="pt")

    def model_inputs(self) -> BatchEncoding:
        """
        The inputs of the heads, as tensors: `input_ids`, `marks` (the number of tokens before each `<|mark|>`), and the
        `num_options` and `question_type_ids` (index in `Question.questions_types`) of each question.
        """
        inputs = {
            "input_ids": self.input_ids,
            "marks": self.marks,
            "num_options": [len(question.options) for question in self.questions],
            "question_type_ids": [Question.questions_types.index(question.type) for question in self.questions],
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


def _split_at_placeholders(text: str, assets: dict[str, Any], load) -> list[dict[str, Any]]:
    """
    The content blocks of `text`: its text, with an image block where a placeholder of `assets` is. This goes from the
    System One way of passing an image (a placeholder in the text, the image in `assets`) to the chat template way (an
    image block in the content), which the processor turns into image tokens, e.g. for Qwen2.5-VL:

        "Is <img-1> a cat?", {"img-1": ...}
        -> [{"type": "text", "text": "Is "}, {"type": "image", "image": ...}, {"type": "text", "text": " a cat?"}]
        -> "Is <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|> a cat?"  (one `<|image_pad|>` per patch)
    """
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
        asset = asset["data_base64"]
    return load_image(asset)


def _insert_images(template_inputs: dict[str, Any], assets: dict[str, Any], load) -> tuple[dict[str, Any], list]:
    """
    Puts the images of a request into the inputs of a decision template (`state`, `question`), where an image is a
    placeholder anywhere in a string, e.g. deep in the state:

        template_inputs = {"state": {"photo": "img-1"}, "question": {"instructions": "Is it like img-2?"}}
        assets = {"img-1": <red>, "img-2": <blue>}

    Returns:
        - the inputs with each such string as content blocks, so the template prints the model's image tokens there:
          {"state": {"photo": [{"type": "image", "image": <red>}]},
           "question": {"instructions": [{"type": "text", "text": "Is it like "}, {"type": "image", "image": <blue>}, ...]}}
        - the images, `[<red>, <blue>]`, for the processor: it only collects images from `messages`, not from template
          inputs. They are in dict order, the order templates print them in (state, instructions, options).
    """
    images = []

    def insert(value):  # visits the values in dict order
        if isinstance(value, dict):
            return {key: insert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [insert(item) for item in value]
        if isinstance(value, str):
            parts = _split_at_placeholders(value, assets, load)
            if any(part["type"] == "image" for part in parts):
                images.extend(part["image"] for part in parts if part["type"] == "image")
                return parts
        return value

    return insert(template_inputs), images


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
        template = self.decision_template or DECISION_PROMPT
        self.prompt, self.template_input_names = _compile_jinja_template(template), _get_template_variables(template)
        self.one_prompt = "questions" in self.template_input_names
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
            # images aren't JSON, and have no sentinels
            request_assets = request.get("assets") if isinstance(request, dict) else None
            request = {k: v for k, v in request.items() if k != "assets"} if request_assets is not None else request
            request = json.loads(json.dumps(request, ensure_ascii=False).replace(MARK, "").replace(SPLIT, ""))
            if request_assets is not None:
                request["assets"] = request_assets

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
                template_inputs = {
                    name: value
                    for name, value in (
                        ("state", state),
                        ("question", asdict(group[0])),
                        ("questions", [asdict(q) for q in group]),
                    )
                    if name in self.template_input_names
                }
                images = []
                if assets:
                    template_inputs, images = _insert_images(template_inputs, assets, load)
                content, kwargs = "", {"chat_template": self.decision_template, **template_inputs}
                kwargs["sanitize_control_tokens"] = sanitize
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
        """
        Drops the sentinels from the inputs: the marks are the number of tokens before each `<|mark|>`.
        Creates the `DecisionPrompt` object
        """
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


__all__ = [
    "DECISION_PROMPT",
    "DecisionPrompt",
    "DecisionTemplate",
    "Question",
    "label_token_ids",
    "parse_questions",
]
