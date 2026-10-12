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
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from ...utils import logging
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .utils import BaseHandler, GenerateManager, Modality


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin

    from .model_manager import ModelManager
    from .utils import GenerationState


logger = logging.get_logger(__name__)


class DecisionConfig(BaseModel):
    """Explicit server settings for decision prompts and next-token scoring."""

    model_config = ConfigDict(extra="forbid")

    labels: dict[Literal["choice", "score", "noul"], list[Annotated[str, Field(min_length=1)]]] = Field(
        default_factory=dict
    )
    temperature: dict[Literal["choice", "score", "noul"], Annotated[float, Field(gt=0, allow_inf_nan=False)]] = Field(
        default_factory=dict
    )
    chat_template: Annotated[str, Field(min_length=1)] | None = None

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "DecisionConfig":
        """Load explicit JSON settings, or return defaults when no file is supplied."""
        return cls() if path is None else cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels):
        for kind, values in labels.items():
            if len(values) < (2 if kind == "noul" else 1) or len(set(values)) != len(values):
                raise ValueError(f"labels.{kind} must contain distinct labels (at least two for noul).")
        return labels


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
    type: Literal["choice", "score", "noul"]
    instructions: str | dict[str, Any] | list[Any] | None
    # Response keys in label order: choice keys, score indices, or true/false.
    options: list[str]
    labels: list[str]
    descriptions: list[str | dict[str, Any] | list[Any] | None]
    legend: list[str | dict[str, Any] | list[Any]] | None = None


def render_content(content: str | dict[str, Any] | list[Any] | None) -> str:
    """Render validated System One content as text, preserving JSON values in objects and arrays."""
    if content is None:
        return ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def parse_question(
    name: str, question: ChoiceQuestion | ScoreQuestion | NoulQuestion, labels: list[str] | None = None
) -> Question:
    """Extract question metadata for prompt rendering and answer scoring, preserving raw JSON values.

    Labels default to Yes/No for booleans and A/B/C/... otherwise.
    """
    legend = None
    n_options = 2 if isinstance(question, NoulQuestion) else len(question.criteria)
    if labels is None:
        labels = ["Yes", "No"] if isinstance(question, NoulQuestion) else list(string.ascii_uppercase[:n_options])
    else:
        labels = labels[:n_options]
    if len(labels) != n_options:
        raise HTTPException(
            status_code=400,
            detail=f"labels.{question.type} needs at least {n_options} labels for question {name!r}.",
        )

    if isinstance(question, NoulQuestion):
        options = ["true", "false"]
        criteria = question.criteria
        descriptions = [criteria.true, criteria.false] if criteria is not None else [None, None]
    elif isinstance(question, ChoiceQuestion):
        options = list(question.criteria)
        descriptions = list(question.criteria.values())
    else:
        options = [str(i) for i in range(n_options)]
        legend = question.criteria
        descriptions = list(question.criteria)

    return Question(
        name=name,
        type=question.type,
        instructions=question.instructions,
        options=options,
        labels=labels,
        descriptions=descriptions,
        legend=legend,
    )


def render_decision_prompt(state: str | dict[str, Any] | list[Any], question: Question) -> str:
    """Render the fallback decision prompt without chat role markers.

    A custom chat template can use the raw state and question fields to replace this wording.
    """
    labels = question.labels
    answers = [render_content(description) for description in question.descriptions]
    if question.type == "noul":
        if labels != ["Yes", "No"]:
            answers = [answer or option for answer, option in zip(answers, question.options)]
        lines = [f"{label}: {answer}" for label, answer in zip(labels, answers) if answer]
        closing = f"Answer {labels[0]} or {labels[1]} only."
    else:
        if question.type == "choice":
            answers = [
                f"{option}: {answer}" if answer else option for option, answer in zip(question.options, answers)
            ]
        answer_kind = "option" if question.type == "choice" else "level"
        lines = [f"{label}. {answer}" for label, answer in zip(labels, answers)]
        label_kind = "letter" if labels == list(string.ascii_uppercase[: len(labels)]) else "label"
        closing = (
            f"Answer with the {label_kind} of the {answer_kind} that fits best ({labels[0]} to {labels[-1]}) only."
        )

    text = "\n".join(line for line in [render_content(question.instructions), *lines] if line)
    return f"{render_content(state)}\n\n{text}\n{closing}"


def build_answers(questions: list[Question], probabilities: list[list[float]]) -> dict[str, dict]:
    """Build System One answers keyed by question name from each question's label probabilities."""
    answers = {}
    for question, question_probabilities in zip(questions, probabilities):
        option_probabilities = dict(zip(question.options, question_probabilities))
        best = max(option_probabilities, key=option_probabilities.get)
        if question.type != "noul":
            n = len(question_probabilities)
            if n == 1:
                confidence = 1.0
            elif question.type == "choice":
                confidence = max(0.0, (option_probabilities[best] - 1 / n) / (1 - 1 / n))
            else:
                distance = sum(p * abs(i - int(best)) for i, p in enumerate(question_probabilities))
                uniform_distance = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
                confidence = max(0.0, 1 - distance / uniform_distance)
        if question.type == "choice":
            answer = {
                "type": "choice",
                "choice": best,
                "probabilities": option_probabilities,
                "confidence": confidence,
            }
        elif question.type == "score":
            answer = {
                "type": "score",
                "score": sum(int(level) * p for level, p in option_probabilities.items()),
                "legend": dict(zip(question.options, question.legend)),
                "probabilities": option_probabilities,
                "confidence": confidence,
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

    def __init__(
        self,
        model_manager: "ModelManager",
        generation_state: "GenerationState",
        chat_template_kwargs: dict | None = None,
        decision_config: DecisionConfig | str | None = None,
    ):
        super().__init__(model_manager, generation_state, chat_template_kwargs)
        self.decision_config = (
            decision_config
            if isinstance(decision_config, DecisionConfig)
            else DecisionConfig.from_file(decision_config)
        )

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
        temperatures = [self.decision_config.temperature.get(question.type, 1.0) for question in questions]
        probabilities = await gen_manager.async_submit(
            self._compute_probabilities, model, inputs, labels, temperatures
        )
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
        questions = []
        processor_inputs = []
        for name, request_question in request.questions.items():
            question = parse_question(name, request_question, self.decision_config.labels.get(request_question.type))
            questions.append(question)
            text = render_decision_prompt(request.state, question)
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
        """Apply the model's chat template and batch text and images for one forward pass."""
        tokenizer = getattr(processor, "tokenizer", processor)
        modality = self.model_manager.get_model_modality(model, processor=processor)
        processor_inputs, questions = self.get_processor_inputs_from_request(request, modality)

        chat_template_kwargs = {
            **self.chat_template_kwargs,
            **request.chat_template_kwargs,
            "enable_thinking": False,
        }
        prompts = []
        for messages, question in zip(processor_inputs, questions):
            if self.decision_config.chat_template is not None:
                chat_template_kwargs.update(
                    chat_template=self.decision_config.chat_template,
                    state=request.state,
                    question=question,
                    images=request.images,
                )
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **chat_template_kwargs
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

    @staticmethod
    def _compute_probabilities(
        model: "PreTrainedModel", inputs: dict, labels: list[list[int]], temperatures: list[float] | None = None
    ) -> list[list[float]]:
        """Run the model and normalize each question's logits over its allowed answer tokens."""
        import torch

        inputs = {
            name: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1]
        probabilities = []
        for i, (question_logits, label_ids) in enumerate(zip(logits, labels)):
            answer_logits = question_logits[label_ids].float()
            if temperatures is not None:
                answer_logits = answer_logits / temperatures[i]
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
        if tokenizer.unk_token_id in label_ids:
            raise HTTPException(
                status_code=400,
                detail=f"The labels of question {question.name!r} contain an unknown token for this model.",
            )
        if len(set(label_ids)) != len(label_ids):
            raise HTTPException(
                status_code=400, detail=f"Two labels of question {question.name!r} share a token for this model."
            )
        return label_ids
