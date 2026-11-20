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
from ...utils.import_utils import is_serve_available


if is_serve_available():
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .utils import BaseHandler, answer_questions, request_logger


if TYPE_CHECKING:
    from .model_manager import ModelManager
    from .utils import GenerationState


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

    Requests are answered by a [`DecisionPipeline`] around the served model, in the model's process (see
    `answer_questions`), so the prompts, images, heads and answers are the pipeline's. Returns one JSON response;
    streaming is not supported.
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
        request_logger.info("Request %s: model %s, %d questions", request_id, engine.model_id, len(request.questions))

        try:
            answers, input_tokens = await engine.submit(
                answer_questions, request.model_dump(include={"state", "questions", "assets"}), self.decision_config
            )
        except ValueError as error:  # e.g. labels that aren't single tokens, or images for a text model
            raise HTTPException(status_code=400, detail=str(error)) from error
        return JSONResponse(
            {
                "model": body["model"],
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        )
