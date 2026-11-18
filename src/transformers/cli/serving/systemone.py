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

Answers typed questions (`choice`, `score`, `noul`) about a `state`, and the images of its `assets`, with a probability
for every allowed answer, by running the served model through [`DecisionPipeline`]. No text is generated.
"""

from typing import TYPE_CHECKING, Any

from ...decision import DecisionConfig, parse_questions
from ...utils import logging
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .utils import BaseHandler, Modality


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast

    from .model_manager import EngineClient, ModelManager
    from .utils import GenerationState


logger = logging.get_logger(__name__)


class SystemOneRequest(BaseModel):
    """A System One request (https://api.typesafe.ai/openapi.json), answered by [`DecisionPipeline`]."""

    model_config = ConfigDict(extra="forbid")

    model: str
    state: str | dict[str, Any] | list[Any]
    # Validated by `parse_questions`, like the pipeline's
    questions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=64)
    # Images keyed by the placeholder they replace in the state, instructions or criteria; checked by the template
    assets: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=4)


class SystemOneHandler(BaseHandler):
    """Handler for the `/v1/systemone` endpoint.

    Requests are answered by a [`DecisionPipeline`] around the served model, so the prompts, images, heads and answers
    are the pipeline's. Returns one JSON response; streaming is not supported.
    """

    def __init__(
        self,
        model_manager: "ModelManager",
        generation_state: "GenerationState",
        chat_template_kwargs: dict | None = None,
        decision_config: DecisionConfig | str | None = None,
    ):
        super().__init__(model_manager, generation_state, chat_template_kwargs)
        # Without one, each model's pipeline reads the decision config of its repo
        self.decision_config = (
            DecisionConfig.from_file(decision_config) if isinstance(decision_config, str) else decision_config
        )
        self.pipelines: dict[str, DecisionPipeline] = {}

    def _validate_request(self, body: dict) -> SystemOneRequest:
        """Validate the entire request, questions included, before resolving or loading a model."""
        if "model" not in body and self.model_manager.force_model is not None:
            body = {**body, "model": self.model_manager.force_model}
        try:
            request = SystemOneRequest.model_validate(body, strict=True)
        except ValidationError as error:
            detail = [{**item, "loc": ("body", *item["loc"])} for item in error.errors(include_url=False)]
            raise HTTPException(status_code=422, detail=detail) from error
        try:
            parse_questions(request.questions, self.decision_config or DecisionConfig())
        except ValueError as error:
            detail = [{"loc": ("body", "questions"), "msg": str(error), "type": "value_error"}]
            raise HTTPException(status_code=422, detail=detail) from error
        return request

    def _pipeline(
        self, model_id: str, model: "PreTrainedModel", processor: "ProcessorMixin | PreTrainedTokenizerFast"
    ) -> "DecisionPipeline":
        """The decision pipeline of the model, built again when the model was reloaded."""
        from transformers import ProcessorMixin
        from transformers.pipelines import DecisionPipeline

        pipe = self.pipelines.get(model_id)
        if pipe is None or pipe.model is not model:
            pipe = self.pipelines[model_id] = DecisionPipeline(
                model=model,
                tokenizer=getattr(processor, "tokenizer", processor),
                processor=processor if isinstance(processor, ProcessorMixin) else None,
                decision_config=self.decision_config,
            )
        return pipe

    @staticmethod
    def _answer(pipe: "DecisionPipeline", request: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """The pipeline's steps, in the inference thread: the answers and the number of input tokens."""
        model_inputs = pipe.preprocess(request)
        answers = pipe.postprocess(pipe.forward(model_inputs))
        return answers, sum(inputs["input_ids"].shape[-1] for inputs, _ in model_inputs["prompts"])

    async def handle_request(self, body: dict, request_id: str) -> "JSONResponse":
        """Validate the request, load the model, and answer every question.

        Args:
            body (`dict`): The raw JSON request body (System One format).
            request_id (`str`): Unique request identifier (from header or auto-generated).

        Returns:
            `JSONResponse`: `model`, `answers` keyed by question name, and `usage`.
        """
        request = self._validate_request(body)
        engine = self._get_engine(body)

        logger.warning(f"[Request received] Model: {engine.model_id}, questions: {len(request.questions)}")

        inputs, questions, labels, input_tokens = self._prepare_inputs(engine, request)
        temperatures = [self.decision_config.temperature.get(question.type, 1.0) for question in questions]
        probabilities = await engine.submit(self._compute_probabilities, dict(inputs), labels, temperatures)
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
        self, engine: "EngineClient", request: TransformersSystemOneRequestParams
    ) -> tuple[dict, list[Question], list[list[int]], int]:
        """Apply the model's chat template and batch text and images for one forward pass."""
        processor, modality = engine.processor, engine.modality
        tokenizer = getattr(processor, "tokenizer", processor)
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
