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
The models the server runs (`ModelManager`), each in a process of its own that its `EngineClient` talks to, and how
they load (`ModelLoader`).
"""

import asyncio
import itertools
import json
import multiprocessing
import queue
import threading
from collections.abc import Callable
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any

from huggingface_hub import CacheNotFound

import transformers
from transformers import BitsAndBytesConfig

from ...utils import logging
from .utils import (
    CBGenerateRunner,
    Detokenizer,
    EngineDeadError,
    GenerateRunner,
    GenerationState,
    Modality,
    ModelNotServedError,
    _StreamError,
    api_server_of,
    canonical_model_id,
    decision_model_class,
    exit_with_parent,
    freeze_garbage_collector,
    get_gen_models,
    make_progress_tqdm_class,
    make_request_id,
    model_owner,
    parse_model_id,
    prompt_length,
    set_torch_seed,
)


if TYPE_CHECKING:
    from transformers import (
        GenerationConfig,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        ProcessorMixin,
    )
    from transformers.utils.chat_parsing import ResponseParser


logger = logging.get_logger(__name__)

# Metal flash attention, the default for continuous batching on MPS where `kernels` can fetch it.
METAL_FLASH_ATTN = "kernels-community/metal-flash-sdpa"


class EngineClient:
    """
    A model, as handlers use it: its config and processor here, its weights in a process of its own (the engine).

    Args:
        model_id (`str`): The model, as ``'repo@revision'``, or ``'repo@revision:file.gguf'`` for a GGUF file.
        loader (`ModelLoader`): How the model loads.
        generation_state (`GenerationState`, *optional*): How the model generates.
        timeout (`float`, *optional*, defaults to -1): Seconds without requests before the process stops. -1: never.
        connection (`tuple`, *optional*): With several API servers, this one's connection to their shared process:
            the requests queue, its results queue, and its number.
    """

    def __init__(
        self,
        model_id: str,
        loader: "ModelLoader",
        generation_state: "GenerationState | None" = None,
        timeout: float = -1,
        connection: tuple | None = None,
    ):
        self.requests: Any = None
        self.results: Any = None
        self.api_server = 0
        self.managed_process: multiprocessing.process.BaseProcess | None = None
        self.is_dead = False
        self.fatal_error: str | None = None
        # The requests in progress, by id: the future of their result, and the streamer of those that stream
        self.pending_requests: dict[int, tuple[asyncio.Future, Detokenizer | None]] = {}
        self._request_counter = itertools.count()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._shutdown_requested = threading.Event()
        self._idle_shutdown: asyncio.TimerHandle | None = None

        self.model_id, self.loader, self.timeout = model_id, loader, timeout
        self.generation_state = generation_state or GenerationState()
        self.model_config, self.generation_config = loader.load_config(model_id)
        self.processor = loader.load_processor(model_id)

        self.externally_managed = connection is not None
        if self.externally_managed:
            self.requests, self.results, self.api_server = connection
            self._start_reading_results()

    def start(self, on_progress: Callable[[dict], None] | None = None) -> None:
        """Starts the model's process if it isn't running, and waits for the model to load."""
        with self._lock:
            if self.running:
                return
            if self.is_dead:
                raise EngineDeadError(self.fatal_error)
            process, requests, results = self.start_process(
                self.model_id, self.loader, self.generation_state, on_progress=on_progress
            )
            self.requests, self.results = requests, results[0]
            self.managed_process = process
            self._start_reading_results()

    async def generate_non_streaming(self, inputs: dict, generation_kwargs: dict) -> tuple[str, int, list[int]]:
        """Generates for `inputs`. Returns the text, the prompt's length, and the generated tokens."""
        generated_ids = await self._send_request("generate", args=(dict(inputs), generation_kwargs))
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        return text, prompt_length(inputs), generated_ids

    def generate_streaming(
        self, inputs: dict, generation_kwargs: dict, response_parser: "ResponseParser | None" = None
    ) -> tuple[asyncio.Queue, Detokenizer]:
        """Generates for `inputs`, streaming the answer. Returns its output queue and its streamer."""
        return self._send_streaming_request("generate", (dict(inputs), generation_kwargs), response_parser)

    async def submit(self, function: Callable, *args) -> Any:
        """Runs `function(model, processor, *args)` in the model's process, and returns its result."""
        return await self._send_request("submit", args=(function, args))

    def submit_streaming(self, function: Callable, *args) -> tuple[asyncio.Queue, Detokenizer]:
        """Like `submit`, streaming the tokens `function` generates. Returns its output queue and its streamer."""
        return self._send_streaming_request("submit", (function, args))

    def shutdown(self, reason: str = "The model's process stopped.") -> None:
        """Stops the model's process, if this client manages it. Its requests in progress fail with `reason`."""
        process = self.managed_process
        if process is None:  # not started, stopped already, or managed externally
            return

        # The client: it has no running engine any more, and its requests in progress fail
        self.managed_process = None
        self._shutdown_requested.set()
        self.requests.cancel_join_thread()
        error = EngineDeadError(reason) if self.is_dead else RuntimeError(reason)

        def fail_all_requests() -> None:
            for future, _ in self.pending_requests.values():
                if not future.done():
                    future.set_result(error)

        if self._loop is not None and not self._loop.is_closed():  # on the event loop, where their futures belong
            self._loop.call_soon_threadsafe(fail_all_requests)

        # Shutdown the process
        self._lock.acquire()
        process.terminate()

        def release_once_exited() -> None:
            process.join(5)
            if process.is_alive():  # it ignored SIGTERM
                process.kill()
                process.join()
            self._lock.release()

        threading.Thread(target=release_once_exited, daemon=True).start()

    async def _send_request(self, method: str, args: tuple, streamer: Detokenizer | None = None) -> Any:
        """Sends a request to the engine, to call `method(*args)` on its runner, and returns its result. With a
        `streamer`, the request streams: its tokens go to it as they come. Raises the error that failed it."""
        if not self.running:
            await asyncio.to_thread(self.start)

        self._loop = asyncio.get_running_loop()
        request_id = make_request_id(self.api_server, next(self._request_counter))
        future = self._loop.create_future()
        self.pending_requests[request_id] = (future, streamer)
        self._cancel_idle_shutdown()
        self.requests.put((method, request_id, *args, streamer is not None))
        try:
            # An error comes as a result too: set as the future's exception, it would be logged as never retrieved
            # when the request is cancelled meanwhile
            result = await future
        finally:
            self.pending_requests.pop(request_id, None)
            if future.cancelled():
                self.requests.put(("cancel", request_id))
            if not self.pending_requests:
                self._schedule_idle_shutdown()
        if isinstance(result, Exception):
            raise result
        return result

    def _send_streaming_request(
        self, method: str, args: tuple, response_parser: "ResponseParser | None" = None
    ) -> tuple[asyncio.Queue, Detokenizer]:
        """Sends a request that streams: creates its streamer, which receives its tokens (see `_start_reading_results`),
        and the task that waits for its end. Returns the streamer's output queue and the streamer."""
        output_queue = asyncio.Queue()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)._tokenizer  # type: ignore[union-attr]
        streamer = Detokenizer(tokenizer, output_queue, response_parser=response_parser)

        async def wait_for_the_end() -> None:
            try:
                await self._send_request(method, args=args, streamer=streamer)
                streamer.end()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                output_queue.put_nowait(_StreamError(str(e)))

        streamer.task = asyncio.create_task(wait_for_the_end())
        return output_queue, streamer

    def _mark_dead(self, reason: str) -> None:
        """Records that the engine can't generate any more: the server reports it, and stops."""
        self.is_dead = True
        self.fatal_error = reason
        logger.error(f"The engine of {self.model_id} died: {reason}")

    def _start_reading_results(self) -> None:
        """Starts the thread that reads the engine's results. It also notices if the model's process crashed."""
        process, shutdown_requested = self.managed_process, threading.Event()
        self._shutdown_requested = shutdown_requested

        def read_results(results_queue) -> None:
            def deliver_results(ids: list[int], tokens: list[int], finished: list[tuple[int, Any]]) -> None:
                """Delivers a message of the engine to the requests waiting in `_send_request`: each streamed token
                (`tokens`, with the id of its request in `ids`) to its request's streamer, and each result of
                `finished` to its request's future, which ends the request."""
                for request_id, token in zip(ids, tokens):
                    future, streamer = self.pending_requests.get(request_id, (None, None))
                    if future is None or future.done():
                        continue
                    try:
                        streamer.put_token(token)
                    except Exception as error:  # the request fails, not the others whose tokens came with its own
                        del self.pending_requests[request_id]
                        future.set_result(error)
                        self.requests.put(("cancel", request_id))
                for request_id, result in finished:
                    if isinstance(result, EngineDeadError) and not self.is_dead:
                        self._mark_dead(str(result))
                    future, _ = self.pending_requests.get(request_id, (None, None))
                    if future is not None and not future.done():
                        future.set_result(result)

            while True:
                try:
                    results = results_queue.get(timeout=1)
                except queue.Empty:
                    if shutdown_requested.is_set():
                        return
                    if process is not None and not process.is_alive():
                        self._mark_dead(f"The model's process crashed (exit code {process.exitcode}).")
                        self.shutdown(self.fatal_error)
                        return
                    continue
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(deliver_results, *results)

        threading.Thread(target=read_results, args=(self.results,), name="model-results", daemon=True).start()

    def _schedule_idle_shutdown(self) -> None:
        """Stops the model's process if no request comes for `timeout` seconds."""
        if self.timeout <= 0 or self.managed_process is None or self._loop is None:
            return

        def shutdown_if_idle() -> None:
            if not self.pending_requests and self.running:
                logger.info(f"Stopping a model's process after {self.timeout}s without requests")
                self.shutdown()

        self._idle_shutdown = self._loop.call_later(self.timeout, shutdown_if_idle)

    def _cancel_idle_shutdown(self) -> None:
        if self._idle_shutdown is not None:
            self._idle_shutdown.cancel()
            self._idle_shutdown = None

    @cached_property
    def modality(self) -> Modality:
        """The model's modality (LLM, VLM or multimodal)."""
        modality = ModelLoader.get_model_modality(self.model_config, self.processor)
        if modality is None:
            raise ValueError(f"Unknown modality for: {self.model_config.architectures}")
        return modality

    @cached_property
    def use_cb(self) -> bool:
        """Whether text generation uses continuous batching."""
        return self.generation_state.use_continuous_batching(self.model_config, self.processor)

    @property
    def running(self) -> bool:
        """Whether the model's process is running."""
        return self.externally_managed or self.managed_process is not None

    @staticmethod
    def start_process(
        model: str,
        loader: "ModelLoader",
        generation_state: "GenerationState",
        api_servers: int = 1,
        on_progress: Callable[[dict], None] | None = None,
    ) -> tuple[multiprocessing.process.BaseProcess, Any, list]:
        """Starts a model's process, and waits for the model to load. Returns the process, its requests queue, and its
        results queues (one per API server)."""
        load = partial(loader.load_model_and_processor, canonical_model_id(model))
        context = multiprocessing.get_context("spawn")
        requests, results = context.Queue(), [context.Queue() for _ in range(api_servers)]
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(
            target=EngineClient.run_engine,
            args=(load, generation_state, requests, results, writer, on_progress is not None, logging.get_verbosity()),
            name="model",
            daemon=True,
        )
        process.start()
        writer.close()
        try:
            message = reader.recv()
            while isinstance(message, dict):
                if on_progress is not None:
                    on_progress(message)
                message = reader.recv()
        except EOFError:
            raise RuntimeError("The model's process stopped while loading the model.") from None
        if message is not None:
            process.join(60)
            raise message
        return process, requests, results

    @staticmethod
    def run_engine(
        load: Callable,
        generation_state: "GenerationState",
        requests,
        results: list,
        writer,
        report_progress: bool,
        log_level: int,
    ) -> None:
        """What the model's process runs: loads the model, then hands each request to a runner."""
        logging.set_verbosity(log_level)  # the server's: a new process starts with the default
        exit_with_parent()
        if generation_state.seed is not None:
            set_torch_seed(generation_state.seed)

        def send_results(ids: list[int], tokens: list[int], finished: list[tuple[int, Any]]) -> None:
            if len(results) == 1:  # one API server: it's all for it
                results[0].put((ids, tokens, finished))
                return
            for api_server, results_queue in enumerate(results):
                streamed = [pair for pair in zip(ids, tokens) if api_server_of(pair[0]) == api_server]
                ended = [pair for pair in finished if api_server_of(pair[0]) == api_server]
                if streamed or ended:
                    results_queue.put(
                        ([request_id for request_id, _ in streamed], [token for _, token in streamed], ended)
                    )

        try:
            model, processor = load(on_progress=writer.send if report_progress else None)
            if generation_state.use_continuous_batching(model.config, processor):
                runner = CBGenerateRunner(model, processor, send_results, generation_state.cb_config)
            else:
                runner = GenerateRunner(model, processor, send_results)
        except Exception as error:
            logger.exception("Loading the model failed")
            writer.send(RuntimeError(f"{type(error).__name__}: {error}"))
            return
        writer.send(None)  # loaded
        writer.close()
        freeze_garbage_collector()

        while True:
            method, request_id, *args = requests.get()
            getattr(runner, method)(request_id, *args)


class ModelManager:
    """
    The models the server runs. Each runs in a process of its own, which handlers send their requests to through the
    model's `EngineClient` (see `get_engine`). A model's process starts with its first request (the model of the
    command line's, right away), and stops after `model_timeout` seconds without requests.

    Args:
        loader (`ModelLoader`, *optional*): How the models load.
        generation_state (`GenerationState`, *optional*): How the models generate.
        model_timeout (`int`, *optional*, defaults to 300): Seconds without requests before a model's process stops,
            freeing its memory. -1: never.
        force_model (`str`, *optional*): The only model served, if set: loaded right away.
        engine_connection (`tuple`, *optional*): In an API server, its connection to the process of the model of the
            command line, which `Serve.run_api_servers` started (see `EngineClient`).
    """

    def __init__(
        self,
        loader: "ModelLoader | None" = None,
        generation_state: GenerationState | None = None,
        model_timeout: int = 300,
        force_model: str | None = None,
        engine_connection: tuple | None = None,
    ):
        self.loader = loader or ModelLoader()
        self.generation_state = generation_state or GenerationState()
        self.force_model = force_model
        self.model_timeout = model_timeout if force_model is None else -1
        self.engines: dict[str, EngineClient] = {}

        # for /load_model
        self._loading_subscribers: dict[str, list[asyncio.Queue[str | None]]] = {}
        self._loading_tasks: dict[str, asyncio.Task] = {}

        if force_model is not None:
            self.preload_engine(force_model, engine_connection)

    def preload_engine(self, force_model: str, engine_connection: tuple | None) -> None:
        """
        Preloads the engine of the model of the command line: its process starts now, or in an API server, it connects
        to the one `Serve.run_api_servers` started (`engine_connection`). It never stops for being idle.
        """
        model_id = canonical_model_id(force_model)
        engine = EngineClient(model_id, self.loader, self.generation_state, connection=engine_connection)
        self.engines[model_id] = engine
        if engine_connection is None:
            engine.start()

    def get_engine(self, model: str | None) -> EngineClient:
        """
        A model, as handlers use it. Created on first use, reading its config and processor; its process starts with
        its first request. With a `force_model`, it's the only model served, and the one a request without a model
        gets.

        Args:
            model (`str`): The model, as requests name it: ``'repo'`` or ``'repo@revision'``, with ``':file.gguf'``
                for a GGUF file (see `canonical_model_id`).
        """
        if self.force_model is not None:
            if model is not None and model != self.force_model:
                raise ModelNotServedError(f"Server is pinned to '{self.force_model}'; requested '{model}'.")
            model = self.force_model
        if model is None:
            raise ModelNotServedError("The request names no model, and the server wasn't started with one.")
        model_id = canonical_model_id(model)
        if model_id not in self.engines:
            self.engines[model_id] = EngineClient(
                model_id, self.loader, self.generation_state, timeout=self.model_timeout
            )
        return self.engines[model_id]

    async def load_model_streaming(self, model: str):
        """Load a model and stream progress as SSE events.

        Handles three cases:
        1. Model already cached -> single ``ready`` event
        2. Load already in progress -> join existing subscriber stream
        3. First request -> start loading, broadcast to all subscribers

        Args:
            model (`str`): The model, as requests name it (see `get_engine`).

        Yields:
            `str`: SSE ``data: ...`` lines with progress updates.
        """
        try:
            engine = self.get_engine(model)
        except Exception as error:  # its config can't be read
            logger.error(f"Failed to load {model}: {error}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'model': model, 'message': str(error)})}\n\n"
            return
        model_id = engine.model_id
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Case 1: already loaded, its process running
        if engine.running:
            yield f"data: {json.dumps({'status': 'ready', 'model': model_id, 'cached': True})}\n\n"
            return

        # Case 2: load in progress -- join existing subscribers
        if model_id in self._loading_tasks:
            self._loading_subscribers[model_id].append(queue)
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            return

        # Case 3: first request -- start the load
        self._loading_subscribers[model_id] = [queue]
        loop = asyncio.get_running_loop()

        def enqueue(payload: dict):
            msg = f"data: {json.dumps(payload)}\n\n"

            def broadcast():
                for q in self._loading_subscribers.get(model_id, []):
                    q.put_nowait(msg)

            loop.call_soon_threadsafe(broadcast)

        async def run_load():
            try:
                await asyncio.to_thread(engine.start, on_progress=enqueue)
            except Exception as e:
                logger.error(f"Failed to load {model_id}: {e}", exc_info=True)
                enqueue({"status": "error", "model": model_id, "message": str(e)})
            finally:

                def _send_sentinel():
                    for q in self._loading_subscribers.pop(model_id, []):
                        q.put_nowait(None)
                    self._loading_tasks.pop(model_id, None)

                loop.call_soon_threadsafe(_send_sentinel)

        self._loading_tasks[model_id] = asyncio.create_task(run_load())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    def shutdown(self) -> None:
        """Stops the models' processes, freeing their memory."""
        for engine in self.engines.values():
            engine.shutdown()

    @property
    def is_dead(self) -> bool:
        """Whether the engine of one of the models died (see `EngineClient.is_dead`): the server then stops."""
        return any(engine.is_dead for engine in self.engines.values())

    def list_models(self, cache_dir: str | None = None) -> list[dict]:
        """List the models to show in `/v1/models`: the generative models of the HuggingFace cache, then the
        loaded models that are not in it, such as a model loaded from a local path. A server pinned to one
        model (`force_model`) lists that model alone.

        Args:
            cache_dir (`str`, *optional*): Path to the HuggingFace cache directory.
                Defaults to the standard cache location.

        Returns:
            `list[dict]`: OpenAI-compatible model list entries with ``id``, ``object``, etc., and a ``status``
            telling whether the model is loaded.
        """
        # The models whose process is running
        loaded = [model_id for model_id, engine in self.engines.items() if engine.running]

        if self.force_model is not None:
            # A pinned server only answers for its model, under the name it was given: list it alone.
            repo, _, _ = parse_model_id(self.force_model)
            return [
                {
                    "owned_by": model_owner(repo),
                    "id": self.force_model,
                    "object": "model",
                    "created": 0,
                    "status": {"value": "loaded" if canonical_model_id(self.force_model) in loaded else "unloaded"},
                }
            ]

        try:
            models = list(get_gen_models(cache_dir))
        except CacheNotFound:
            # Nothing was downloaded yet, as in a fresh container.
            models = []

        loaded_models = {}
        for name in loaded:
            model_id, revision, gguf_file = parse_model_id(name)
            owned_by = model_owner(model_id)
            # Named as the cache names them: `main` is left out, a GGUF file comes after a colon.
            if revision not in (None, "main"):
                model_id = f"{model_id}@{revision}"
            if gguf_file is not None:
                model_id = f"{model_id}:{gguf_file}"
            # `created` is the date of a checkpoint, which is only known for the ones of the cache.
            loaded_models[model_id] = {"owned_by": owned_by, "id": model_id, "object": "model", "created": 0}

        listed = {model["id"] for model in models}
        models += [model for model_id, model in loaded_models.items() if model_id not in listed]

        # `status` is not an OpenAI field: it tells which of the listed models are in memory.
        for index, model in enumerate(models):
            status = "loaded" if model["id"] in loaded_models else "unloaded"
            models[index] = {**model, "status": {"value": status}}
        return models


class ModelLoader:
    """
    How models load, with the server's settings. Each model's process loads its model with a copy of the server's
    loader (`load_model_and_processor`); the API server only loads each model's configs (`load_config`) and processor
    (`load_processor`, see `EngineClient`).

    Args:
        device: Device to place models on (e.g. "auto", "cuda", "cpu").
        dtype: Torch dtype override. "auto" derives from model weights.
        trust_remote_code: Whether to trust remote code when loading models.
        attn_implementation: Attention implementation override (e.g. "flash_attention_2").
        quantization: Quantization method ("bnb-4bit" or "bnb-8bit").
        continuous_batching: Whether models generate with continuous batching, to default to a Metal flash kernel on
            MPS.
    """

    def __init__(
        self,
        device: str = "auto",
        dtype: str | None = "auto",
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        quantization: str | None = None,
        continuous_batching: bool = False,
    ):
        # Convert numeric device strings (e.g. "0") to int so device_map works correctly
        self.device = int(device) if device.isdigit() else device
        self.dtype = self._resolve_dtype(dtype)
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = self._resolve_attn_implementation(
            attn_implementation, self.device, continuous_batching
        )
        self.quantization = quantization
        self._validate_args()

    def load_model_and_processor(
        self, model_id: str, on_progress: Callable | None = None
    ) -> "tuple[PreTrainedModel, ProcessorMixin | PreTrainedTokenizerFast]":
        """Loads the model `model_id` (see `parse_model_id`) and its processor, its progress events going to
        `on_progress`."""
        logger.warning(f"Loading {model_id}")
        if on_progress is not None:
            on_progress({"status": "loading", "model": model_id, "stage": "processor"})
        processor = self.load_processor(model_id)
        tqdm_class = None if on_progress is None else make_progress_tqdm_class(on_progress, model_id)
        # The "Loading weights" bar too, which isn't a download bar: through the logging's tqdm hook
        previous_hook = None
        if tqdm_class is not None:
            previous_hook = logging.set_tqdm_hook(lambda factory, args, kwargs: tqdm_class(*args, **kwargs))
        try:
            model = self._load_model(model_id, tqdm_class=tqdm_class, progress_callback=on_progress)
        finally:
            if tqdm_class is not None:
                logging.set_tqdm_hook(previous_hook)
        if on_progress is not None:
            on_progress({"status": "ready", "model": model_id, "cached": False})
        return model, processor

    def load_config(self, model_id: str) -> "tuple[PretrainedConfig, GenerationConfig]":
        """The config and generation config of the model `model_id`, which the API server reads of a model running in
        its process."""
        from transformers import AutoConfig, GenerationConfig

        repo, revision, gguf_file = parse_model_id(model_id)
        config = AutoConfig.from_pretrained(
            repo, gguf_file=gguf_file, revision=revision, trust_remote_code=self.trust_remote_code
        )
        try:  # like `from_pretrained`: the repo's generation config if it has one, else the model config's
            generation_config = GenerationConfig.from_pretrained(repo, revision=revision)
        except OSError:
            generation_config = GenerationConfig.from_model_config(config)
        return config, generation_config

    def load_processor(self, model_id: str) -> "ProcessorMixin | PreTrainedTokenizerFast":
        """The processor (or tokenizer) of the model `model_id`."""
        from transformers import AutoProcessor

        repo, revision, gguf_file = parse_model_id(model_id)
        kwargs = {"revision": revision, "trust_remote_code": self.trust_remote_code}
        if gguf_file is not None:  # a GGUF repo ships no tokenizer files: the vocabulary is in the file's metadata
            kwargs["gguf_file"] = gguf_file
        return AutoProcessor.from_pretrained(repo, **kwargs)

    @staticmethod
    def get_model_modality(
        config: "PretrainedConfig", processor: "ProcessorMixin | PreTrainedTokenizerFast | None" = None
    ) -> Modality | None:
        """Detect whether a model is an LLM or VLM from the architecture it loads with (see `_load_model`).

        Args:
            config (`PretrainedConfig`): The model's config.
            processor (`ProcessorMixin | PreTrainedTokenizerFast`, *optional*):
                If a plain tokenizer (not a multi-modal processor), short-circuits to LLM.

        Returns:
            `Modality`: The detected modality (``Modality.LLM``, ``Modality.VLM``, or ``Modality.MULTIMODAL``), or
            `None` for a model that doesn't generate text from a chat (e.g. a speech recognition model).
        """
        from transformers import PreTrainedTokenizerBase

        if processor is not None and isinstance(processor, PreTrainedTokenizerBase):
            return Modality.LLM

        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
            MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES,
        )

        if config.model_type in MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES:  # it loads with `AutoModelForMultimodalLM`
            return Modality.MULTIMODAL
        architecture = config.architectures[0] if config.architectures else None
        if architecture in MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES.values():
            return Modality.MULTIMODAL
        elif architecture in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values():
            return Modality.VLM
        elif architecture in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            return Modality.LLM
        return None

    def _load_model(
        self, model_id: str, tqdm_class: type | None = None, progress_callback: Callable | None = None
    ) -> "PreTrainedModel":
        """Load a model.

        Args:
            model_id (`str`): The model (see `parse_model_id`).
            tqdm_class (*optional*): tqdm subclass for progress bars during ``from_pretrained``.
            progress_callback (`Callable`, *optional*): Called with progress dicts during loading.

        Returns:
            `PreTrainedModel`: The loaded model.
        """
        from transformers import AutoConfig

        repo, revision, gguf_file = parse_model_id(model_id)

        model_kwargs = {
            "revision": revision,
            "attn_implementation": self.attn_implementation,
            "dtype": self.dtype,
            "device_map": self.device,
            "trust_remote_code": self.trust_remote_code,
            "tqdm_class": tqdm_class,
        }
        quantization_config = self.get_quantization_config()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        if gguf_file is not None:
            model_kwargs["gguf_file"] = gguf_file

        if progress_callback is not None:
            progress_callback({"status": "loading", "model": model_id, "stage": "config"})
        config = AutoConfig.from_pretrained(repo, **model_kwargs)

        from transformers.models.auto.modeling_auto import MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES

        decision_model = decision_model_class(model_id)
        if decision_model is not None:
            return decision_model.from_pretrained(repo, **model_kwargs)

        if config.model_type in MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES:
            from transformers import AutoModelForMultimodalLM

            return AutoModelForMultimodalLM.from_pretrained(repo, **model_kwargs)

        if not config.architectures:  # a GGUF file's config names none: by its model type, as a causal LM
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)
        architecture = getattr(transformers, config.architectures[0])
        return architecture.from_pretrained(repo, **model_kwargs)

    def get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Return a BitsAndBytesConfig based on the `quantization` setting, or None."""
        if self.quantization == "bnb-4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "bnb-8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    @staticmethod
    def _resolve_dtype(dtype: str | None):
        import torch

        if dtype in ("auto", None):
            return dtype
        resolved = getattr(torch, dtype, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(
                f"Unsupported dtype: '{dtype}'. Must be 'auto' or a valid torch dtype (e.g. 'float16', 'bfloat16')."
            )
        return resolved

    @staticmethod
    def _resolve_attn_implementation(
        attn_implementation: str | None, device: str | int, continuous_batching: bool
    ) -> str | None:
        """Default to the Metal flash kernel for continuous batching on `mps`, when `kernels` is installed."""
        if attn_implementation is not None or not continuous_batching:
            return attn_implementation

        import torch

        from ...utils.import_utils import is_kernels_available

        is_mps_device = (
            isinstance(device, str)
            and device.startswith("mps")
            or (device == "auto" and torch.backends.mps.is_available() and not torch.cuda.is_available())
        )
        if is_mps_device and is_kernels_available():
            logger.warning_once(
                f"Continuous batching on MPS with `kernels` installed: defaulting attention to `{METAL_FLASH_ATTN}`. "
                "Pass `--attn-implementation sdpa` to opt out."
            )
            return METAL_FLASH_ATTN
        return attn_implementation

    def _validate_args(self):
        if self.quantization is not None and self.quantization not in ("bnb-4bit", "bnb-8bit"):
            raise ValueError(
                f"Unsupported quantization method: '{self.quantization}'. Must be 'bnb-4bit' or 'bnb-8bit'."
            )
        from ...integrations.hub_kernels import is_kernel

        VALID_ATTN_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"}
        if (
            self.attn_implementation is not None
            and not is_kernel(self.attn_implementation)
            and self.attn_implementation not in VALID_ATTN_IMPLEMENTATIONS
        ):
            raise ValueError(
                f"Unsupported attention implementation: '{self.attn_implementation}'. "
                f"Must be one of {VALID_ATTN_IMPLEMENTATIONS} or a hub kernel (e.g. 'kernels-community/flash-attn2')."
            )
