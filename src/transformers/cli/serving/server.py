# Copyright 2025 The HuggingFace Team. All rights reserved.
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
FastAPI app factory.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ...utils import logging
from .chat_completion import ChatCompletionHandler
from .model_manager import ModelManager
from .response import ResponseHandler
from .utils import X_REQUEST_ID


logger = logging.get_logger(__name__)


def build_server(
    model_manager: ModelManager,
    chat_handler: ChatCompletionHandler,
    response_handler: ResponseHandler,
    enable_cors: bool = False,
) -> FastAPI:
    """Build and return a configured FastAPI application.

    Args:
        model_manager: Handles model loading, caching, and cleanup.
        chat_handler: Handles `/v1/chat/completions` requests.
        response_handler: Handles `/v1/responses` requests.
        enable_cors: If `True`, adds permissive CORS middleware (allow all origins).

    Returns:
        A FastAPI app ready to be passed to uvicorn.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        model_manager.shutdown()

    app = FastAPI(lifespan=lifespan)

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.warning_once("CORS allow origin is set to `*`. Not recommended for production.")

    # ---- Middleware ----

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Get or set the request ID in the header."""
        request_id = request.headers.get(X_REQUEST_ID) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[X_REQUEST_ID] = request_id
        return response

    # ---- Routes ----

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: dict):
        return chat_handler.handle_request(body, request.state.request_id)

    @app.post("/v1/responses")
    async def responses(request: Request, body: dict):
        return response_handler.handle_request(body, request.state.request_id)

    @app.get("/v1/models")
    @app.options("/v1/models")
    def list_models():
        return JSONResponse({"object": "list", "data": model_manager.get_gen_models()})

    @app.get("/health")
    def health():
        return JSONResponse({"status": "ok"})

    return app
