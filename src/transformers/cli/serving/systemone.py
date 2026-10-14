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

from typing import TYPE_CHECKING, Annotated, Any, Literal

from ...decision import DECISION_PROMPT, MARK, SPLIT, DecisionConfig, Question, label_token_ids
from ...utils import logging
from ...utils.chat_template_utils import _compile_jinja_template
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .utils import BaseHandler, GenerateManager, Modality


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin

    from .model_manager import ModelManager
    from .utils import GenerationState


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
        template = self.decision_config.chat_template or ""
        if MARK in template or SPLIT in template or self.decision_config.calibration or self.decision_config.noul_band:
            raise ValueError(
                "`calibration`, `noul_band` and templates with `<|mark|>` or `<|split|>` (decision heads) are not "
                "supported by `transformers serve` yet, use `pipeline('decision')`."
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
        answers = {
            question.name: question.answer(question_probabilities)
            for question, question_probabilities in zip(questions, probabilities)
        }

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
            try:
                question = Question.from_request(name, request_question.model_dump(), self.decision_config)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            questions.append(question)
            text = _compile_jinja_template(DECISION_PROMPT).render(state=request.state, question=question)
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

        try:
            labels = [label_token_ids(tokenizer, question) for question in questions]
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
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
