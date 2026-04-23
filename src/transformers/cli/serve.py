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

import asyncio
import threading
from typing import Annotated

import typer

from transformers.utils import logging
from transformers.utils.import_utils import is_serve_available

from .serving.utils import set_torch_seed


logger = logging.get_logger(__name__)


class Serve:
    def __init__(
        self,
        force_model: Annotated[str | None, typer.Argument(help="Model to preload and use for all requests.")] = None,
        # Model options
        continuous_batching: Annotated[
            bool,
            typer.Option(help="Enable continuous batching with paged attention. Configure with --cb-* flags."),
        ] = False,
        cb_block_size: Annotated[
            int | None, typer.Option(help="KV cache block size in tokens for continuous batching.")
        ] = None,
        cb_num_blocks: Annotated[
            int | None, typer.Option(help="Number of KV cache blocks for continuous batching.")
        ] = None,
        cb_max_batch_tokens: Annotated[
            int | None, typer.Option(help="Maximum tokens per batch for continuous batching.")
        ] = None,
        cb_max_memory_percent: Annotated[
            float | None, typer.Option(help="Max GPU memory fraction for KV cache (0.0-1.0).")
        ] = None,
        cb_use_cuda_graph: Annotated[
            bool | None, typer.Option(help="Enable CUDA graphs for continuous batching.")
        ] = None,
        attn_implementation: Annotated[
            str | None, typer.Option(help="Attention implementation (e.g. flash_attention_2).")
        ] = None,
        compile: Annotated[bool, typer.Option(help="Enable torch.compile for faster inference.")] = False,
        quantization: Annotated[
            str | None, typer.Option(help="Quantization method: 'bnb-4bit' or 'bnb-8bit'.")
        ] = None,
        device: Annotated[str, typer.Option(help="Device for inference (e.g. 'auto', 'cuda:0', 'cpu').")] = "auto",
        dtype: Annotated[str | None, typer.Option(help="Override model dtype. 'auto' derives from weights.")] = "auto",
        trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code when loading.")] = False,
        model_timeout: Annotated[
            int, typer.Option(help="Seconds before idle model is unloaded. Ignored when force_model is set.")
        ] = 300,
        # Server options
        host: Annotated[str, typer.Option(help="Server listen address.")] = "localhost",
        port: Annotated[int, typer.Option(help="Server listen port.")] = 8000,
        enable_cors: Annotated[bool, typer.Option(help="Enable permissive CORS.")] = False,
        log_level: Annotated[str, typer.Option(help="Logging level (e.g. 'info', 'warning').")] = "warning",
        default_seed: Annotated[int | None, typer.Option(help="Default torch seed.")] = None,
        non_blocking: Annotated[
            bool, typer.Option(hidden=True, help="Run server in a background thread. Used by tests.")
        ] = False,
    ) -> None:
        if not is_serve_available():
            raise ImportError("Missing dependencies for serving. Install with `pip install transformers[serving]`")

        import uvicorn

        from .serving.chat_completion import ChatCompletionHandler
        from .serving.completion import CompletionHandler
        from .serving.model_manager import ModelManager
        from .serving.response import ResponseHandler
        from .serving.server import build_server
        from .serving.transcription import TranscriptionHandler
        from .serving.utils import GenerationState

        # Seed
        if default_seed is not None:
            set_torch_seed(default_seed)

        # Logging
        transformers_logger = logging.get_logger("transformers")
        transformers_logger.setLevel(logging.log_levels[log_level.lower()])

        self._model_manager = ModelManager(
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            quantization=quantization,
            model_timeout=model_timeout,
            force_model=force_model,
        )
        from transformers import ContinuousBatchingConfig

        cb_kwargs = {
            k: v
            for k, v in {
                "block_size": cb_block_size,
                "num_blocks": cb_num_blocks,
                "max_batch_tokens": cb_max_batch_tokens,
                "max_memory_percent": cb_max_memory_percent,
                "use_cuda_graph": cb_use_cuda_graph,
            }.items()
            if v is not None
        }
        cb_config = ContinuousBatchingConfig(**cb_kwargs) if cb_kwargs else None
        self._generation_state = GenerationState(
            continuous_batching=continuous_batching,
            compile=compile,
            cb_config=cb_config,
        )

        self._chat_handler = ChatCompletionHandler(
            model_manager=self._model_manager,
            generation_state=self._generation_state,
        )

        self._completion_handler = CompletionHandler(
            model_manager=self._model_manager,
            generation_state=self._generation_state,
        )

        self._response_handler = ResponseHandler(
            model_manager=self._model_manager,
            generation_state=self._generation_state,
        )

        self._transcription_handler = TranscriptionHandler(self._model_manager, self._generation_state)

        app = build_server(
            self._model_manager,
            self._chat_handler,
            completion_handler=self._completion_handler,
            response_handler=self._response_handler,
            transcription_handler=self._transcription_handler,
            enable_cors=enable_cors,
        )

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
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

    def reset_loaded_models(self):
        """Clear all loaded models from memory."""
        self._model_manager.shutdown()

    def kill_server(self):
        self._generation_state.shutdown()
        self._model_manager.shutdown()
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
    POST /v1/completions      — Legacy text completions from a prompt.
    GET  /v1/models           — Lists available models.
    GET  /health              — Health check.

Requires FastAPI and Uvicorn: pip install transformers[serving]
"""
