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
Handler for the /v1/systemone endpoint (System One typed decisions, in Jev's request and answer shapes).

Answers typed questions (`choice`, `score`, `noul`) about a `state` with a probability for every allowed answer. No
text is generated: each allowed answer gets a single-token label, and its probability is read from the model's logits
in one forward pass.
"""

import json
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal

from ...utils import logging
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .utils import BaseHandler, GenerateManager, Modality


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin


logger = logging.get_logger(__name__)


# Request types follow https://api.typesafe.ai/openapi.json.
class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    # One letter label per option, at most 26.
    criteria: dict[str, str | dict[str, Any] | list[Any] | None] = Field(min_length=1, max_length=26)


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: list[str | dict[str, Any] | list[Any]] = Field(min_length=1, max_length=26)


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: str | dict[str, Any] | list[Any] | None = None
    false: str | dict[str, Any] | list[Any] | None = None


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"]
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: NoulCriteria | None = None


class SystemOneRequest(BaseModel):
    model: str
    state: str | dict[str, Any] | list[Any]
    images: list[Annotated[str, Field(pattern=r"^data:image/[^;]+;base64,")]] = Field(
        default_factory=list, max_length=4
    )
    questions: dict[str, Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]] = (
        Field(min_length=1, max_length=64)
    )


class TransformersSystemOneRequestParams(SystemOneRequest):
    model_config = ConfigDict(extra="forbid")

    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Question:
    name: str
    type: str
    # The instructions and one line per labelled answer
    text: str
    # The line asking for the label, when the question is asked on its own
    closing: str
    # Answer keys in label order: the `criteria` keys for `choice`, the level indices for `score`, `true`/`false` for `noul`
    options: list[str]
    labels: list[str]
    descriptions: list[str | dict[str, Any] | list[Any] | None]
    legend: list[str | dict[str, Any] | list[Any]] | None = None


def render_content(content: str | dict[str, Any] | list[Any] | None) -> str:
    """Render validated System One content as text, preserving JSON values in objects and arrays."""
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def parse_question(name: str, question: ChoiceQuestion | ScoreQuestion | NoulQuestion) -> Question:
    """Render a validated question as instructions followed by one line per labelled answer."""
    legend = None

    if isinstance(question, NoulQuestion):
        options = ["true", "false"]
        labels = ["Yes", "No"]
        criteria = question.criteria
        descriptions = [criteria.true, criteria.false] if criteria is not None else [None, None]
        lines = []
        for label, criterion in zip(labels, descriptions):
            description = render_content(criterion)
            if description:
                lines.append(f"{label}: {description}")
        closing = f"Answer {labels[0]} or {labels[1]} only."
    else:
        labels = list(string.ascii_uppercase[: len(question.criteria)])
        if isinstance(question, ChoiceQuestion):
            options = list(question.criteria)
            descriptions = list(question.criteria.values())
            answers = []
            for option, criterion in zip(options, descriptions):
                description = render_content(criterion)
                answers.append(f"{option}: {description}" if description else option)
            answer_kind = "option"
        else:
            options = [str(i) for i in range(len(question.criteria))]
            legend = question.criteria
            descriptions = list(question.criteria)
            answers = [render_content(level) for level in descriptions]
            answer_kind = "level"

        lines = [f"{label}. {answer}" for label, answer in zip(labels, answers)]
        closing = f"Answer with the letter of the {answer_kind} that fits best ({labels[0]} to {labels[-1]}) only."

    instructions = render_content(question.instructions)
    return Question(
        name=name,
        type=question.type,
        text="\n".join(line for line in [instructions, *lines] if line),
        closing=closing,
        options=options,
        labels=labels,
        descriptions=descriptions,
        legend=legend,
    )


def build_answers(questions: list[Question], probabilities: list[list[float]]) -> dict[str, dict]:
    """Build System One answers keyed by question name from each question's label probabilities."""
    answers = {}
    for question, question_probabilities in zip(questions, probabilities):
        option_probabilities = dict(zip(question.options, question_probabilities))
        best = max(option_probabilities, key=option_probabilities.get)
        if question.type == "choice":
            answer = {
                "type": "choice",
                "choice": best,
                "probabilities": option_probabilities,
                "confidence": option_probabilities[best],
            }
        elif question.type == "score":
            answer = {
                "type": "score",
                "score": sum(int(level) * p for level, p in option_probabilities.items()),
                "legend": dict(zip(question.options, question.legend)),
                "probabilities": option_probabilities,
                "confidence": option_probabilities[best],
            }
        else:
            answer = {"type": "noul", "noul": option_probabilities["true"]}
        answers[question.name] = answer
    return answers


class SystemOneHandler(BaseHandler):
    """Handler for the `/v1/systemone` endpoint.

    Autoregressive models get one prompt per question and answer with their next token, all questions in one batched
    forward pass. Returns one JSON response; streaming is not supported.
    """

    def _validate_request(self, body: dict) -> TransformersSystemOneRequestParams:
        """Validate the entire request before resolving or loading a model."""
        if "model" not in body and self.model_manager.force_model is not None:
            body = {**body, "model": self.model_manager.force_model}
        try:
            return TransformersSystemOneRequestParams.model_validate(body, strict=True)
        except ValidationError as error:
            detail = [{**item, "loc": ("body", *item["loc"])} for item in error.errors(include_url=False)]
            raise HTTPException(status_code=422, detail=detail) from error

    async def handle_request(self, body: dict, request_id: str) -> "JSONResponse":
        """Validate the request, load the model, and answer every question.

        Args:
            body (`dict`): The raw JSON request body (System One format).
            request_id (`str`): Unique request identifier (from header or auto-generated).

        Returns:
            `JSONResponse`: `model`, `answers` keyed by question name, and `usage`.
        """
        request = self._validate_request(body)
        model_id, model, processor = self._resolve_model(body)

        logger.warning(f"[Request received] Model: {model_id}, questions: {len(request.questions)}")

        gen_manager: GenerateManager = self.generation_state.get_manager(model_id, use_cb=False)  # type: ignore[assignment]

        inputs, questions, labels, input_tokens = self._prepare_inputs(model, processor, request)
        probabilities = await gen_manager.async_submit(self._compute_probabilities, model, inputs, labels)
        answers = build_answers(questions, probabilities)

        return JSONResponse(
            {
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        )

    def get_processor_inputs_from_request(
        self, request: SystemOneRequest, modality: Modality
    ) -> tuple[list[list[dict]], list[Question]]:
        """Build one processor-compatible conversation per question and retain its answer metadata."""
        if request.images and modality not in (Modality.VLM, Modality.MULTIMODAL):
            raise HTTPException(status_code=400, detail="The selected model does not support image input.")
        state = render_content(request.state)
        questions = [parse_question(name, question) for name, question in request.questions.items()]
        processor_inputs = []
        for question in questions:
            text = f"{state}\n\n{question.text}\n{question.closing}"
            if modality == Modality.LLM:
                content = text
            else:
                content = [{"type": "image", "url": image} for image in request.images]
                content.append({"type": "text", "text": text})
            processor_inputs.append([{"role": "user", "content": content}])
        return processor_inputs, questions

    def _prepare_inputs(
        self,
        model: "PreTrainedModel",
        processor: "ProcessorMixin | PreTrainedTokenizerFast",
        request: TransformersSystemOneRequestParams,
    ) -> tuple[dict, list[Question], list[list[int]], int]:
        """Render the model's decision template and batch text and images for one forward pass."""
        tokenizer = getattr(processor, "tokenizer", processor)
        modality = self.model_manager.get_model_modality(model, processor=processor)
        processor_inputs, questions = self.get_processor_inputs_from_request(request, modality)

        prompts = []
        for messages, question in zip(processor_inputs, questions):
            template_kwargs = self._get_chat_template_kwargs(processor.chat_template, tokenizer, request, question)
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **template_kwargs
            )
            prompts.append(prompt)

        processor_kwargs = {"images": [request.images for _ in questions]} if request.images else {}
        inputs = processor(
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_special_tokens=False,
            **processor_kwargs,
        )
        if modality == Modality.LLM:
            inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp(min=0)

        labels = [self._tokenize_labels(tokenizer, question) for question in questions]
        input_tokens = int(inputs["attention_mask"].sum().item())
        return inputs, questions, labels, input_tokens

    def _get_chat_template_kwargs(
        self,
        templates: str | dict[str, str] | None,
        tokenizer: "PreTrainedTokenizerFast",
        request: TransformersSystemOneRequestParams,
        question: Question,
    ) -> dict:
        """Merge template options and add raw question fields for a named System One template."""
        # The answer is read right after the prompt, so reasoning is always off
        chat_template_kwargs = {
            **self.chat_template_kwargs,
            **request.chat_template_kwargs,
            "enable_thinking": False,
        }

        # Special case: a named `systemone` template consumes raw question fields.
        # Ordinary chat templates use the already-formatted messages.
        template = None
        for candidate in (templates, tokenizer.chat_template):
            if isinstance(candidate, dict) and "systemone" in candidate:
                template = candidate["systemone"]
                break

        if template is None:
            return chat_template_kwargs

        options = [
            {"key": key, "label": label, "description": description}
            for key, label, description in zip(question.options, question.labels, question.descriptions)
        ]
        return {
            **chat_template_kwargs,
            "chat_template": template,
            "id": question.name,
            "type": question.type,
            "state": request.state,
            "instructions": request.questions[question.name].instructions,
            "options": options,
            "images": request.images,
        }

    @staticmethod
    def _compute_probabilities(model: "PreTrainedModel", inputs: dict, labels: list[list[int]]) -> list[list[float]]:
        """Run the model and normalize each question's logits over its allowed answer tokens."""
        import torch

        inputs = {
            name: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
        probabilities = []
        for question_logits, label_ids in zip(logits, labels):
            answer_logits = question_logits[label_ids].float()
            answer_probabilities = answer_logits.softmax(dim=-1)
            probabilities.append(answer_probabilities.tolist())
        return probabilities

    @staticmethod
    def _tokenize_labels(tokenizer: "PreTrainedTokenizerFast", question: Question) -> list[int]:
        """Tokenize each label independently and require distinct single-token answers."""
        label_tokens = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in question.labels]
        if any(len(ids) != 1 for ids in label_tokens):
            raise HTTPException(
                status_code=400,
                detail=f"The labels of question {question.name!r} are not single tokens for this model.",
            )
        label_ids = [ids[0] for ids in label_tokens]
        if len(set(label_ids)) != len(label_ids):
            raise HTTPException(
                status_code=400, detail=f"Two labels of question {question.name!r} share a token for this model."
            )
        return label_ids
