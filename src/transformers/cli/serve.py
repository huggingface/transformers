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
import enum
import json
import threading
from typing import Annotated

import typer

from transformers.utils import logging
from transformers.utils.import_utils import is_serve_available

from .serving.utils import set_torch_seed


logger = logging.get_logger(__name__)


class ReasoningMode(str, enum.Enum):
    ON = "on"
    OFF = "off"
    AUTO = "auto"


class Serve:
    def __init__(
        self,
        force_model: Annotated[
            str | None, typer.Argument(help="Load this model first and use it for every request")
        ] = None,
        # Model options
        continuous_batching: Annotated[
            bool,
            typer.Option(help="Enable continuous batching with paged attention; configure it with --cb-* options"),
        ] = False,
        attn_implementation: Annotated[
            str | None, typer.Option(help="Set the attention implementation, for example, `flash_attention_2`")
        ] = None,
        compile: Annotated[bool, typer.Option(help="Compile with `torch.compile` for faster inference")] = False,
        quantization: Annotated[
            str | None, typer.Option(help="Set the quantization method: `bnb-4bit` or `bnb-8bit`")
        ] = None,
        reasoning: Annotated[
            ReasoningMode,
            typer.Option(
                help=(
                    "Set the reasoning mode: `auto` uses the chat template default. This only applies to models "
                    "whose chat template supports reasoning."
                )
            ),
        ] = ReasoningMode.AUTO.value,  # type: ignore[invalid-parameter-default]
        chat_template_kwargs: Annotated[
            str | None,
            typer.Option(
                help=(
                    "Default JSON arguments for `apply_chat_template`; per-request `chat_template_kwargs` override these"
                )
            ),
        ] = None,
        device: Annotated[
            str, typer.Option(help="Device for inference, for example, `auto`, `cuda:0`, or `cpu`")
        ] = "auto",
        dtype: Annotated[
            str | None, typer.Option(help="Override model dtype; `auto` derives it from the weights")
        ] = "auto",
        trust_remote_code: Annotated[bool, typer.Option(help="Allow custom model code to run locally")] = False,
        model_timeout: Annotated[
            int, typer.Option(help="Unload idle models after this many seconds; ignored with a fixed model")
        ] = 300,
        # Continuous batching tuning
        cb_block_size: Annotated[
            int | None, typer.Option(help="KV-cache block size in tokens for continuous batching")
        ] = None,
        cb_num_blocks: Annotated[
            int | None, typer.Option(help="Number of KV-cache blocks for continuous batching")
        ] = None,
        cb_max_batch_tokens: Annotated[
            int | None, typer.Option(help="Maximum tokens per batch for continuous batching")
        ] = None,
        cb_max_memory_percent: Annotated[
            float | None, typer.Option(help="Maximum GPU-memory fraction for KV cache, from `0.0` to `1.0`")
        ] = None,
        cb_use_cuda_graph: Annotated[
            bool | None, typer.Option(help="Enable CUDA graphs for continuous batching")
        ] = None,
        # Server options
        host: Annotated[str, typer.Option(help="Host address to listen on")] = "localhost",
        port: Annotated[int, typer.Option(help="Port to listen on")] = 8000,
        enable_cors: Annotated[bool, typer.Option(help="Allow cross-origin requests from any origin")] = False,
        log_level: Annotated[str, typer.Option(help="Logging level, for example, `info` or `warning`")] = "warning",
        default_seed: Annotated[int | None, typer.Option(help="Default PyTorch seed")] = None,
        non_blocking: Annotated[
            bool, typer.Option(hidden=True, help="Run server in a background thread. Used by tests.")
        ] = False,
    ) -> None:
        if not is_serve_available():
            raise ImportError(
                "Serving dependencies are missing. Install them with `pip install transformers[serving]`."
            )

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

        if chat_template_kwargs:
            chat_template_kwargs = json.loads(chat_template_kwargs)
            if not isinstance(chat_template_kwargs, dict):
                raise typer.BadParameter("Expected a JSON object for `--chat-template-kwargs`")
        else:
            chat_template_kwargs = {}

        if reasoning == ReasoningMode.ON:
            chat_template_kwargs["enable_thinking"] = True
        elif reasoning == ReasoningMode.OFF:
            chat_template_kwargs["enable_thinking"] = False

        self._chat_handler = ChatCompletionHandler(
            model_manager=self._model_manager,
            generation_state=self._generation_state,
            chat_template_kwargs=chat_template_kwargs,
        )

        self._completion_handler = CompletionHandler(
            model_manager=self._model_manager,
            generation_state=self._generation_state,
        )

        self._response_handler = ResponseHandler(
            model_manager=self._model_manager,
            generation_state=self._generation_state,
            chat_template_kwargs=chat_template_kwargs,
        )

        self._transcription_handler = TranscriptionHandler(self._model_manager, self._generation_state)

        app = build_server(
            self._model_manager,
            self._chat_handler,
            completion_handler=self._completion_handler,
            response_handler=self._response_handler,
            transcription_handler=self._transcription_handler,
            generation_state=self._generation_state,
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
Run a FastAPI server with an OpenAI-compatible API; models load on demand and unload after a timeout

\b
Endpoints:
    POST /v1/chat/completions — Chat completions, streaming or non-streaming
    POST /v1/completions      — Legacy text completions from a prompt
    GET  /v1/models           — List available models
    GET  /health              — Check server health

Install FastAPI and Uvicorn with `pip install transformers[serving]`
"""
