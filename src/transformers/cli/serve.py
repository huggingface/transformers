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

import enum
import gc
import json
from typing import TYPE_CHECKING, Annotated

import typer

from transformers.utils import logging
from transformers.utils.import_utils import is_serve_available


if TYPE_CHECKING:
    import socket


class ReasoningMode(str, enum.Enum):
    ON = "on"
    OFF = "off"
    AUTO = "auto"


class Serve:
    def __init__(
        self,
        force_model: Annotated[
            str | None,
            typer.Argument(
                help=(
                    "Model to preload and use for all requests. GGUF weights are named `<repo>:<file>.gguf`, "
                    "and their config and tokenizer come from the file itself."
                )
            ),
        ] = None,
        # Model options
        continuous_batching: Annotated[
            bool,
            typer.Option(help="Enable continuous batching with paged attention. Configure with --cb-* flags."),
        ] = False,
        attn_implementation: Annotated[
            str | None, typer.Option(help="Attention implementation (e.g. flash_attention_2).")
        ] = None,
        compile: Annotated[bool, typer.Option(help="Enable torch.compile for faster inference.")] = False,
        quantization: Annotated[
            str | None, typer.Option(help="Quantization method: 'bnb-4bit' or 'bnb-8bit'.")
        ] = None,
        reasoning: Annotated[
            ReasoningMode,
            typer.Option(
                help=(
                    "Reasoning mode. 'auto' uses the chat template default. Only applies to models that "
                    "support reasoning via their chat template (e.g. Qwen3, Gemma 4) — for other models "
                    "this flag has no effect."
                )
            ),
        ] = ReasoningMode.AUTO.value,  # type: ignore[invalid-parameter-default]
        chat_template_kwargs: Annotated[
            str | None,
            typer.Option(
                help=(
                    "Default JSON kwargs forwarded to apply_chat_template "
                    "(e.g. '{\"enable_thinking\": true}'); per-request chat_template_kwargs override these."
                )
            ),
        ] = None,
        decision_config: Annotated[
            str | None,
            typer.Option(help="Path to a JSON file with decision labels, temperatures, and a chat template."),
        ] = None,
        device: Annotated[str, typer.Option(help="Device for inference (e.g. 'auto', 'cuda:0', 'cpu').")] = "auto",
        dtype: Annotated[str | None, typer.Option(help="Override model dtype. 'auto' derives from weights.")] = "auto",
        trust_remote_code: Annotated[bool, typer.Option(help="Trust remote code when loading.")] = False,
        model_timeout: Annotated[
            int, typer.Option(help="Seconds before idle model is unloaded. Ignored when force_model is set.")
        ] = 300,
        max_model_len: Annotated[
            int | None,
            typer.Option(
                help="The longest a request can be, prompt and answer, in tokens. Defaults to the model's context length."
            ),
        ] = None,
        # Continuous batching tuning
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
        # Server options
        host: Annotated[str, typer.Option(help="Server listen address.")] = "localhost",
        port: Annotated[int, typer.Option(help="Server listen port.")] = 8000,
        enable_cors: Annotated[bool, typer.Option(help="Enable permissive CORS.")] = False,
        log_level: Annotated[
            str, typer.Option(help="Logging level of serve's messages (e.g. 'info', 'warning')")
        ] = "info",
        enable_log_requests: Annotated[
            bool,
            typer.Option(
                help="Log each request at info (model, prompt length, generation settings), its body at debug."
            ),
        ] = False,
        default_seed: Annotated[int | None, typer.Option(help="Default torch seed.")] = None,
        api_server_count: Annotated[
            int,
            typer.Option(
                help=(
                    "API server processes handling HTTP, tokenizing and decoding, each on a core of its own, for "
                    "when one isn't enough. Needs a model on the command line, which they share, loaded once."
                )
            ),
        ] = 1,
    ) -> None:
        if not is_serve_available():
            raise ImportError("Missing dependencies for serving. Install with `pip install transformers[serving]`")

        from transformers import ContinuousBatchingConfig

        from .serving.model_manager import ModelLoader
        from .serving.utils import GenerationState, set_serve_log_level

        self.log_level = logging.log_levels[log_level.lower()]
        self.enable_log_requests = enable_log_requests
        set_serve_log_level(self.log_level, enable_log_requests)

        cb_options = {
            "block_size": cb_block_size,
            "num_blocks": cb_num_blocks,
            "max_batch_tokens": cb_max_batch_tokens,
            "max_memory_percent": cb_max_memory_percent,
            "use_cuda_graph": cb_use_cuda_graph,
        }
        cb_set = any(value is not None for value in cb_options.values())
        self.generation_state = GenerationState(
            continuous_batching=continuous_batching,
            compile=compile,
            cb_config=ContinuousBatchingConfig(**cb_options) if cb_set else None,
            seed=default_seed,
            max_model_len=max_model_len,
        )

        self.loader = ModelLoader(
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            quantization=quantization,
            continuous_batching=continuous_batching,
        )

        self.chat_template_kwargs = self.parse_chat_template_kwargs(chat_template_kwargs, reasoning)
        self.force_model, self.model_timeout, self.decision_config = force_model, model_timeout, decision_config
        self.host, self.port, self.enable_cors = host, port, enable_cors

        if api_server_count > 1:
            self.run_api_servers(api_server_count)
        else:
            self.run()

    def run(self, socket: "socket.socket | None" = None, engine_connection: tuple | None = None) -> None:
        """
        Serves the API: on `socket` if given, else on the command line's host and port. In an API server (see
        `run_api_servers`), `engine_connection` connects it to the process of the model of the command line: the
        requests queue, shared by the API servers, this one's results queue, and its id.
        """
        import uvicorn

        from .serving.chat_completion import ChatCompletionHandler
        from .serving.completion import CompletionHandler
        from .serving.model_manager import ModelManager
        from .serving.response import ResponseHandler
        from .serving.server import build_server
        from .serving.systemone import SystemOneHandler
        from .serving.transcription import TranscriptionHandler
        from .serving.utils import exit_on_parent_exit, freeze_garbage_collector, set_serve_log_level

        set_serve_log_level(self.log_level, self.enable_log_requests)
        exit_on_parent_exit()

        model_manager = ModelManager(
            self.loader,
            self.generation_state,
            model_timeout=self.model_timeout,
            force_model=self.force_model,
            engine_connection=engine_connection,
        )
        chat_handler = ChatCompletionHandler(
            model_manager=model_manager,
            generation_state=self.generation_state,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        completion_handler = CompletionHandler(
            model_manager=model_manager,
            generation_state=self.generation_state,
        )

        response_handler = ResponseHandler(
            model_manager=model_manager,
            generation_state=self.generation_state,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        transcription_handler = TranscriptionHandler(model_manager)

        systemone_handler = SystemOneHandler(
            model_manager=model_manager,
            generation_state=self.generation_state,
            chat_template_kwargs=self.chat_template_kwargs,
            decision_config=self.decision_config,
        )

        app = build_server(
            model_manager,
            chat_handler,
            completion_handler=completion_handler,
            response_handler=response_handler,
            transcription_handler=transcription_handler,
            systemone_handler=systemone_handler,
            enable_cors=self.enable_cors,
        )

        async def stop_if_an_engine_died() -> None:
            if model_manager.is_dead:
                server.should_exit = True

        # uvicorn calls `callback_notify` on the event loop every `timeout_notify` seconds (it checks once a second)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
            callback_notify=stop_if_an_engine_died,
            timeout_notify=5,
        )
        server = uvicorn.Server(config)
        freeze_garbage_collector()
        server.run(sockets=None if socket is None else [socket])
        gc.unfreeze()  # the server has stopped: if the process goes on (tests), what it held can be freed again
        if model_manager.is_dead:
            raise RuntimeError("An engine died (see the error above): stopping the server")

    def run_api_servers(self, count: int) -> None:
        """
        Serves the API with `count` API servers, for when one process can't keep up. Each handles HTTP, tokenizes
        prompts and decodes answers in a process of its own (`run`), on a core of its own, and they share the process
        of the model of the command line, loaded once. Runs until it's stopped (Ctrl+C or SIGTERM), or until one of
        these processes stops: it then stops the others, and raises, for whatever runs `transformers serve` to restart
        it.
        """
        import multiprocessing
        import signal
        import sys
        import time
        from multiprocessing.connection import wait

        import uvicorn

        from .serving.model_manager import EngineClient

        if self.force_model is None:
            raise ValueError(
                "Several API servers need a model on the command line: "
                "`transformers serve MODEL --api-server-count N`."
            )
        # Ctrl+C and SIGTERM exit quietly, through the `finally` below, which stops the processes
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            signal.signal(stop_signal, lambda *_: sys.exit(0))

        socket = uvicorn.Config(app=None, host=self.host, port=self.port).bind_socket()
        model_process, requests, results = EngineClient.start_process(
            self.force_model, self.loader, self.generation_state, api_servers=count
        )
        spawn = multiprocessing.get_context("spawn")
        servers = [
            spawn.Process(
                target=self.run, args=(socket, (requests, results[index], index)), name=f"api-server-{index}"
            )
            for index in range(count)
        ]
        processes = [model_process, *servers]
        for server in servers:
            server.start()
        try:
            ready = wait([process.sentinel for process in processes])
            stopped = [process for process in processes if process.sentinel in ready]
            for process in stopped:
                process.join()  # it's exiting: for its exit code
            names = ", ".join(f"{process.name} (exit code {process.exitcode})" for process in stopped)
            raise RuntimeError(f"Process {names} stopped: stopping the server")
        finally:
            for process in processes:
                process.terminate()
            deadline = time.monotonic() + 5
            for process in processes:
                process.join(max(0.0, deadline - time.monotonic()))
            # An API server waits for its requests in progress, which can't finish with the model's process stopped
            for process in processes:
                if process.is_alive():
                    process.kill()
                    process.join()
            socket.close()

    @staticmethod
    def parse_chat_template_kwargs(chat_template_kwargs: str | None, reasoning: ReasoningMode) -> dict:
        """
        The kwargs `apply_chat_template` gets for every request (a request's own `chat_template_kwargs` override them):
        `--chat-template-kwargs`, a JSON object, with `enable_thinking` set by `--reasoning`, unless it's 'auto'.
        """
        kwargs = json.loads(chat_template_kwargs) if chat_template_kwargs else {}
        if not isinstance(kwargs, dict):
            raise typer.BadParameter("--chat-template-kwargs must be a JSON object")
        if reasoning != ReasoningMode.AUTO:
            kwargs["enable_thinking"] = reasoning == ReasoningMode.ON
        return kwargs


Serve.__doc__ = """
Run a FastAPI server to serve models on-demand with an OpenAI compatible API.
Models will be loaded and unloaded automatically based on usage and a timeout.

\b
Endpoints:
    POST /v1/chat/completions — Chat completions (streaming + non-streaming).
    POST /v1/completions      — Legacy text completions from a prompt.
    POST /v1/systemone        — System One typed decisions (choice, score, yes/no) with probabilities.
    GET  /v1/models           — Lists available models.
    GET  /health              — Health check.

Requires FastAPI and Uvicorn: pip install transformers[serving]
"""
