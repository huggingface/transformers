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
CLI entry point for `transformers serve`.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Annotated

import typer

from transformers.utils import logging
from transformers.utils.import_utils import (
    is_fastapi_available,
    is_openai_available,
    is_pydantic_available,
    is_uvicorn_available,
)

from .serving.utils import set_torch_seed


serve_dependencies_available = (
    is_pydantic_available() and is_fastapi_available() and is_uvicorn_available() and is_openai_available()
)

logger = logging.get_logger(__name__)


class Serve:
    def __init__(
        self,
        # TODO: maybe rename it to model ?
        force_model: Annotated[str | None, typer.Argument(help="Model to preload and use for all requests.")] = None,
        # Model options
        device: Annotated[str, typer.Option(help="Device for inference; defaults to 'auto'.")] = "auto",
        dtype: Annotated[str | None, typer.Option(help="Override model dtype. 'auto' derives from weights.")] = "auto",
        attn_implementation: Annotated[
            str | None, typer.Option(help="Attention implementation (e.g. flash_attention_2).")
        ] = None,
        quantization: Annotated[
            str | None, typer.Option(help="Quantization method: 'bnb-4bit' or 'bnb-8bit'.")
        ] = None,
        trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code when loading.")] = False,
        # TODO: auto-detect processor from GGUF base_model metadata so this flag isn't needed
        processor: Annotated[
            str | None, typer.Option(help="Processor/tokenizer model ID. Needed for GGUF models.")
        ] = None,
        model_timeout: Annotated[
            int, typer.Option(help="Seconds before idle model is unloaded. Ignored when model is set.")
        ] = 300,
        # Server options
        host: Annotated[str, typer.Option(help="Server listen address.")] = "localhost",
        port: Annotated[int, typer.Option(help="Server listen port.")] = 8000,
        enable_cors: Annotated[bool, typer.Option(help="Enable permissive CORS.")] = False,
        log_level: Annotated[str, typer.Option(help="Logging level (e.g. 'info', 'warning').")] = "info",
        default_seed: Annotated[int | None, typer.Option(help="Default torch seed.")] = None,
        compile: Annotated[
            bool,
            typer.Option(
                help="Enable static cache + torch.compile for faster decode (~2.6x). First request triggers compilation (~30s)."
            ),
        ] = False,
        non_blocking: Annotated[
            bool, typer.Option(hidden=True, help="Run server in a background thread. Used by tests.")
        ] = False,
    ) -> None:
        if not serve_dependencies_available:
            raise ImportError("Missing dependencies for serving. Install with `pip install transformers[serving]`")

        import uvicorn

        from .serving.chat_completion import ChatCompletionHandler
        from .serving.model_manager import ModelManager
        from .serving.response import ResponseHandler
        from .serving.server import build_server
        from .serving.utils import InferenceThread

        # Seed
        if default_seed is not None:
            set_torch_seed(default_seed)

        # Logging
        transformers_logger = logging.get_logger("transformers")
        transformers_logger.setLevel(logging.log_levels[log_level.lower()])

        # Preloaded models should never be auto-unloaded
        if force_model:
            model_timeout = -1

        model_manager = ModelManager(
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            quantization=quantization,
            model_timeout=model_timeout,
            force_model=force_model,
            processor_id=processor,
        )

        # Single persistent thread for all generate() calls — required for
        # torch.compile + CUDA graphs which use thread-local storage.
        inference_thread = InferenceThread()

        chat_handler = ChatCompletionHandler(
            model_manager=model_manager,
            force_model=force_model,
            force_processor=processor,
            inference_thread=inference_thread,
            compile=compile,
        )

        response_handler = ResponseHandler(
            model_manager=model_manager,
            force_model=force_model,
            force_processor=processor,
            inference_thread=inference_thread,
            compile=compile,
        )

        app = build_server(model_manager, chat_handler, response_handler=response_handler, enable_cors=enable_cors)

        config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
        self.server = uvicorn.Server(config)

        if non_blocking:
            self.start_server()
        else:
            self.server.run()

    def start_server(self):
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.server.serve())

        self._thread = threading.Thread(target=_run, name="uvicorn-thread", daemon=False)
        self._thread.start()

    def kill_server(self):
        if not self._thread or not self._thread.is_alive():
            return
        self.server.should_exit = True
        self._thread.join(timeout=2)


Serve.__doc__ = """
Run a FastAPI server to serve models on-demand with an OpenAI compatible API.
Models will be loaded and unloaded automatically based on usage and a timeout.

\b
Endpoints:
    POST /v1/chat/completions — Chat completions (streaming + non-streaming).
    GET  /v1/models           — Lists available models.
    GET  /health              — Health check.

Requires FastAPI and Uvicorn: pip install transformers[serving]
"""
