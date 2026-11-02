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

from .utils import BaseHandler, GenerateManager


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin
    from transformers.pipelines import DecisionPipeline

    from .model_manager import ModelManager
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

    def _model_class(self, model_id: str) -> "type[PreTrainedModel] | None":
        """The decision model of a checkpoint shipping its head next to its weights, as in `pipeline("decision")`.

        `None` (the class its config names) for any other model.
        """
        from transformers.decision.modeling_utils import maybe_get_decision_model_config
        from transformers.models.auto.modeling_auto import MODEL_FOR_DECISION_MAPPING

        repo, revision = model_id.split("@", 1)
        config = maybe_get_decision_model_config(repo, revision=revision)
        return None if config is None else MODEL_FOR_DECISION_MAPPING[type(config)]

    def _validate_request(self, body: dict) -> SystemOneRequest:
        """Validate the request's fields before resolving or loading a model (questions: `_validate_questions`)."""
        if "model" not in body and self.model_manager.force_model is not None:
            body = {**body, "model": self.model_manager.force_model}
        try:
            return SystemOneRequest.model_validate(body, strict=True)
        except ValidationError as error:
            detail = [{**item, "loc": ("body", *item["loc"])} for item in error.errors(include_url=False)]
            raise HTTPException(status_code=422, detail=detail) from error

    @staticmethod
    def _validate_questions(questions: dict[str, Any], decision_config: DecisionConfig) -> None:
        """Validate the questions with the model's decision config, whose labels bound the number of options."""
        try:
            parse_questions(questions, decision_config)
        except ValueError as error:
            detail = [{"loc": ("body", "questions"), "msg": str(error), "type": "value_error"}]
            raise HTTPException(status_code=422, detail=detail) from error

    def _pipeline(
        self, model_id: str, model: "PreTrainedModel", processor: "ProcessorMixin | PreTrainedTokenizerFast"
    ) -> "DecisionPipeline":
        """The decision pipeline of the model, built again when the model was reloaded."""
        from transformers import ProcessorMixin
        from transformers.pipelines import DecisionPipeline

        decision_pipeline = self.pipelines.get(model_id)
        if decision_pipeline is None or decision_pipeline.model is not model:
            decision_pipeline = self.pipelines[model_id] = DecisionPipeline(
                model=model,
                tokenizer=getattr(processor, "tokenizer", processor),
                processor=processor if isinstance(processor, ProcessorMixin) else None,
                decision_config=self.decision_config,
            )
        return decision_pipeline

    @staticmethod
    def _answer(decision_pipeline: "DecisionPipeline", request: SystemOneRequest) -> tuple[dict[str, Any], int]:
        """The pipeline's steps, in the inference thread: the answers and the number of input tokens."""
        model_inputs = decision_pipeline.preprocess(request.model_dump())
        answers = decision_pipeline.postprocess(decision_pipeline.forward(model_inputs))
        input_tokens = sum(prompt["input_ids"].numel() for prompt, _ in model_inputs["prompts"])
        return answers, input_tokens

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
        decision_pipeline = self._pipeline(model_id, model, processor)
        self._validate_questions(request.questions, decision_pipeline.decision_config)
        try:
            answers, input_tokens = await gen_manager.async_submit(self._answer, decision_pipeline, request)
        except ValueError as error:  # a request this model can't answer
            raise HTTPException(status_code=400, detail=str(error)) from error
        return JSONResponse(
            {
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        )
