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

A model with a `decision_template.jinja` places two sentinel tokens:

- `<|mark|>`: where the head reads, which depends on the head:

      jev    ... [decision]:<|mark|>
             reads 1 token: the one before each mark (one per question)
      kev    ... opt A<|mark|> opt B<|mark|> opt C<|mark|> state<|mark|>
             reads 1 token before each mark, one mark after each option
      von    ... [MASK]<|mark|> [MASK]<|mark|> ... <|mark|>
             reads 1 token per option; the last mark ends the block
      clef   INSTRUCTION: <|mark|>What is the tone?<|mark|>
             OPTION 1: <|mark|>{"option_id":"calm"}<|mark|>
             OPTION 2: <|mark|>{"option_id":"pushy"}<|mark|>
             reads every token between a pair of marks, pooled (hidden states + output embeddings of the span):
             the spans sit mid-sequence, not at the end

- `<|split|>`: is added to enforce splitting the encoding and is only there for "safety", ensuring user text never parses any control / special tokens.

Both are dropped from the input ids: a [`DecisionPrompt`] keeps the token ids and the number of tokens before each
`<|mark|>`.

We otherwise can turn any model into a decision model with a `DECISION_PROMPT`, where the logits of the option labels (`A`, `B`, ... or `Yes`/`No`) are read after the generation prompt, e.g. for `Qwen/Qwen2.5-0.5B-Instruct`:

```
    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {"message": "WIN a FREE iPhone, click here!"}

    What is the tone?
    A. calm
    B. pushy: Pressures the reader
    Answer with the letter of the option that fits best (A to B) only.<|im_end|>
    <|im_start|>assistant
```

The rest of this file handles the different input validation, escaping user text from the mark/split tokens,
and replacing images with the corresponding assets.

We support arbitrary kind of inputs, users should be able to expand it to his needs.
"""

import functools
import json
import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Literal

from huggingface_hub.dataclasses import strict
from huggingface_hub.errors import StrictDataclassError

from ..image_utils import load_image
from ..tokenization_utils_base import AddedToken
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
    This goes from the System One way of passing an image (a placeholder in the text, the image in `assets`) to the chat template way (an
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


def chat_template_kwargs(config: DecisionConfig, request: dict[str, Any] | str, processor) -> list[dict[str, Any]]:
    """
    The keyword arguments of `processor.apply_chat_template` for each prompt of a System One `request` (`{"state": ...,
    "questions": ..., "assets": ...}`, or its JSON): one per question, or one for all of them when the decision template
    reads `questions`. Each also has the parsed questions of its prompt (`decision_questions`), which [`~DecisionMixin.decide`] and
    `decode_decision` read. Adds the sentinels to the tokenizer of `processor`, once, and keeps their ids in `config`.

    `assets` map placeholders to images, as in the System One API: each placeholder in a string of the state, the
    instructions or the criteria is replaced with its image. An image is an asset (`{"type": "image", "format": "png",
    "data_base64": ...}`), or anything [`~image_utils.load_image`] takes, like a PIL image, a path or a URL.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    if MARK not in tokenizer.get_vocab():  # special: they split the text they're in, and requests can't write them
        tokenizer.add_tokens([AddedToken(token, special=True, normalized=False) for token in (MARK, SPLIT)])
    config._sentinel_ids = tuple(tokenizer.convert_tokens_to_ids([MARK, SPLIT]))

    if isinstance(request, str):  # parsed, so that `\u` escapes can't hide control tokens
        request = json.loads(request)
    if not isinstance(request, dict) or "questions" not in request or "state" not in request:
        raise ValueError('A request is {"state": ..., "questions": {name: question, ...}}, as a dict or its JSON.')
    if not config.sanitize_control_tokens:  # templates that escape control tokens still get no sentinels
        assets = request.get("assets")  # images aren't JSON, and have no sentinels
        request = {key: value for key, value in request.items() if key != "assets"}
        request = json.loads(json.dumps(request, ensure_ascii=False).replace(MARK, "").replace(SPLIT, ""))
        request["assets"] = assets
    questions, state, assets = parse_questions(request["questions"], config), request["state"], request.get("assets")
    has_images = getattr(processor, "image_processor", None) is not None
    if assets and not has_images:
        raise ValueError("Images need a model whose processor handles them.")

    template = config.chat_template or DECISION_PROMPT
    template_input_names = _get_template_variables(template)
    chat_kwargs = {"enable_thinking": False}
    if has_images and "enable_thinking" not in _get_template_variables(processor.chat_template):
        chat_kwargs = {}  # processors pass the variables their template doesn't use to the processor
    load = functools.cache(lambda name: _load_asset(assets[name]))
    prompts = []
    one_prompt = "questions" in template_input_names
    for group in [questions] if one_prompt else [[question] for question in questions]:
        kwargs = {"add_generation_prompt": True, "sanitize_control_tokens": config.sanitize_control_tokens}
        if config.chat_template is None:  # the generic prompt, as the user message of the model's chat template
            text = _compile_jinja_template(DECISION_PROMPT).render(state=state, question=group[0])
            content = _split_at_placeholders(text, assets, load) if has_images else text
        else:
            template_inputs = {"state": state, "question": asdict(group[0]), "questions": [asdict(q) for q in group]}
            template_inputs = {name: value for name, value in template_inputs.items() if name in template_input_names}
            if assets:
                template_inputs, images = _insert_images(template_inputs, assets, load)
                kwargs["processor_kwargs"] = {"images": images} if images else {}
            content = ""
            kwargs |= {"chat_template": config.chat_template, **template_inputs}
        conversation = [{"role": "user", "content": content}]
        prompts.append({"conversation": conversation, **chat_kwargs, **kwargs, "decision_questions": group})
    return prompts


def decode_decision(scores: list, questions: list[Question]) -> dict[str, dict[str, Any]]:
    """The System One answers to `questions`, from the option scores [`~DecisionMixin.decide`] returns for them."""
    return {question.name: question.answer(scores.softmax(-1).tolist()) for question, scores in zip(questions, scores)}


__all__ = [
    "DECISION_PROMPT",
    "Question",
    "chat_template_kwargs",
    "decode_decision",
    "label_token_ids",
    "parse_questions",
]
