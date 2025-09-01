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
import asyncio
import base64
import copy
import datetime
import enum
import functools
import gc
import io
import json
import re
import tempfile
import threading
import time
from argparse import ArgumentParser, Namespace
from collections.abc import AsyncGenerator, Generator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from threading import Thread
from typing import Optional, Union

from huggingface_hub import model_info
from huggingface_hub.constants import HF_HUB_OFFLINE
from tokenizers.decoders import DecodeStream

import transformers
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)
from transformers.utils.import_utils import (
    is_fastapi_available,
    is_librosa_available,
    is_openai_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vision_available,
)

from .. import (
    AutoConfig,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
    ProcessorMixin,
    TextIteratorStreamer,
)
from ..utils import is_torch_available, logging
from . import BaseTransformersCLICommand


if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        GenerationConfig,
        PreTrainedModel,
    )

    from ..generation.continuous_batching import ContinuousBatchingManager, RequestStatus

if is_librosa_available():
    import librosa

if is_vision_available():
    from PIL import Image

serve_dependencies_available = (
    is_pydantic_available() and is_fastapi_available() and is_uvicorn_available() and is_openai_available()
)
if serve_dependencies_available:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from openai.types.audio.transcription import Transcription
    from openai.types.audio.transcription_create_params import TranscriptionCreateParamsBase
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.chat.chat_completion_chunk import (
        ChatCompletionChunk,
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseContentPartAddedEvent,
        ResponseContentPartDoneEvent,
        ResponseCreatedEvent,
        ResponseError,
        ResponseErrorEvent,
        ResponseFailedEvent,
        ResponseInProgressEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )
    from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
    from pydantic import BaseModel, TypeAdapter, ValidationError

    # Expand OpenAI's request input types with an optional `generation_config` field
    class TransformersResponseCreateParamsStreaming(ResponseCreateParamsStreaming, total=False):
        """
        OpenAI's ResponseCreateParamsStreaming with an additional field for the generation config (as a json string).
        """

        generation_config: str

    class TransformersCompletionCreateParamsStreaming(CompletionCreateParamsStreaming, total=False):
        """
        OpenAI's CompletionCreateParamsStreaming with an additional field for the generation config (as a json string).
        """

        generation_config: str

    class TransformersTranscriptionCreateParams(TranscriptionCreateParamsBase, total=False):
        """
        OpenAI's TranscriptionCreateParamsBase with an additional field for the generation config (as a json string).
        """

        file: bytes  # Overwritten -- pydantic isn't happy with `typing.IO[bytes]`, present in the original type
        generation_config: str
        stream: Optional[bool] = False

    # Contrarily to OpenAI's output types, input types are `TypedDict`, which don't have built-in validation.
    response_validator = TypeAdapter(TransformersResponseCreateParamsStreaming)
    completion_validator = TypeAdapter(TransformersCompletionCreateParamsStreaming)
    transcription_validator = TypeAdapter(TransformersTranscriptionCreateParams)

    # Define request fields that are not yet used in `transformers serve`. Receiving these fields will raise an
    # HTTPException.
    UNUSED_RESPONSE_FIELDS = {
        "background",
        "include",
        "max_tool_calls",
        "previous_response_id",
        "prompt",
        "reasoning",
        "service_tier",
        "store",
        "text",
        "tool_choice",
        "top_logprobs",
        "truncation",
        "user",
    }

    UNUSED_CHAT_COMPLETION_FIELDS = {
        "audio",
        "function_call",
        "functions",
        "logprobs",
        "max_completion_tokens",
        "metadata",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "reasoning_effort",
        "response_format",
        "service_tier",
        "stop",
        "store",
        "stream_options",
        "tool_choice",
        "top_logprobs",
        "user",
        "web_search_options",
    }
    UNUSED_TRANSCRIPTION_FIELDS = {
        "chunking_strategy",
        "include",
        "language",
        "prompt",
        "response_format",
        "timestamp_granularities",
    }


logger = logging.get_logger(__name__)

# Possible tokens that indicate the start/end of a tool call
# TODO (joao, matt): streamline tool token detection logic
_TOOL_CALL_TOKENS = {
    "qwen": {
        "start": "<tool_call>",
        "end": "</tool_call>",
    },
}
_MODELS_WITH_TOOL_SUPPORT = list(_TOOL_CALL_TOKENS.keys())


class Modality(enum.Enum):
    LLM = "LLM"
    VLM = "VLM"
    STT = "STT"
    TTS = "TTS"


def serve_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.

    Returns: ServeCommand
    """
    return ServeCommand(args)


def create_generation_config_from_req(
    req: dict,
    model_generation_config: "GenerationConfig",
    **kwargs,
) -> "GenerationConfig":
    """
    Creates a generation config from the parameters of the request. If a generation config is passed in the request,
    it will be used as a baseline for parameterization. Otherwise, we will use the model's default generation config.
    Other parameters in the request will be applied on top of the baseline.

    Args:
        req (`dict`):
            The request which may optionally contain generation parameters.
        model_generation_config (`GenerationConfig`):
            The model's default generation config.
        kwargs (`dict`):
            Additional parameters to set in the generation config.

    Returns:
        The prepared `GenerationConfig` object.
    """
    # If there is a generation config in the request, it is a json string serialization from a `GenerationConfig`
    # object. For simplicity, flags set here take precedence over all other flags.
    if req.get("generation_config") is not None:
        generation_config = GenerationConfig(**json.loads(req["generation_config"]))
    else:
        generation_config = copy.deepcopy(model_generation_config)

    non_standard_kwargs = generation_config.update(**kwargs)
    # Set extra kwargs that are not in the `GenerationConfig` class (e.g. continuous batching flags)
    for k, v in non_standard_kwargs.items():
        if v is not None:
            setattr(generation_config, k, v)

    # Response-specific parameters
    if req.get("max_output_tokens") is not None:
        generation_config.max_new_tokens = int(req["max_output_tokens"])

    # Completion-specific parameters
    if req.get("max_tokens") is not None:
        generation_config.max_new_tokens = int(req["max_tokens"])
    if req.get("frequency_penalty") is not None:
        generation_config.repetition_penalty = float(req["frequency_penalty"])
    if req.get("logit_bias") is not None:
        generation_config.sequence_bias = req["logit_bias"]
    if req.get("stop") is not None:
        generation_config.stop_strings = req["stop"]
    if req.get("temperature") is not None:
        generation_config.temperature = float(req["temperature"])
        if float(req["temperature"]) == 0.0:
            generation_config.do_sample = False
    if req.get("top_p") is not None:
        generation_config.top_p = float(req["top_p"])
    if req.get("seed") is not None:
        torch.manual_seed(req["seed"])

    return generation_config


class ToolState:
    """Lightweight class to keep track of the tool call state."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the tool call state (assumes we're outside a tool call)."""
        self.inside_tool_call = False
        self.has_tool_name_defined = False
        self.arg_nesting_level = 0
        self.buffer = ""


class TimedModel:
    """
    A class that holds a PreTrainedModel instance and its associated processor.
    Automatically deletes the instances after a specified timeout.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        timeout_seconds: int,
        processor: Optional[Union["ProcessorMixin", "PreTrainedTokenizerFast"]] = None,
    ):
        self.model = model
        self._name_or_path = str(model.name_or_path)
        self.processor = processor
        self.timeout_seconds = timeout_seconds
        self._timer = threading.Timer(self.timeout_seconds, self.timeout_reached)
        self._timer.start()

    def reset_timer(self):
        """Reset the timer for the deletion of the instances."""
        self._timer.cancel()
        self._timer = threading.Timer(self.timeout_seconds, self.timeout_reached)
        self._timer.start()

    def delete_model(self):
        """Delete the wrapped model and processor and clean up resources."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # XXX: in case we manually delete the model, like on server shutdown
            self._timer.cancel()

    def timeout_reached(self):
        self.delete_model()
        logger.info(f"{self._name_or_path} was removed from memory after {self.timeout_seconds} seconds of inactivity")

    def is_deleted(self):
        """Check if the instances have been deleted."""
        return not hasattr(self, "model") or self.model is None


@dataclass
class ServeArguments:
    r"""
    Arguments for the serve CLI.

    See the metadata arg for each argument's description -- the metadata will be printed with
    `transformers serve --help`
    """

    continuous_batching: bool = field(
        default=False,
        metadata={"help": "Whether to use continuous batching for chat completions."},
    )
    device: str = field(
        default="auto",
        metadata={
            "help": "Device to use for inference; will default to `auto` and"
            "place the model on an accelerator if available."
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "`torch_dtype` is deprecated! Please use `dtype` argument instead.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype. If `'auto'` is passed, "
            "the dtype will be automatically derived from the model's weights.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Whether to trust remote code when loading a model."}
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit precision for the base model - works only with LoRA."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit precision for the base model - works only with LoRA."},
    )
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]})
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "Whether to use nested quantization."})

    # Serving settings
    host: str = field(default="localhost", metadata={"help": "Interface the server will listen to."})
    port: int = field(default=8000, metadata={"help": "Port the server will listen to."})
    model_timeout: int = field(
        default=300,
        metadata={"help": "Time in seconds after which a model will be removed from memory."},
    )

    # Other settings
    log_level: str = field(
        default="info", metadata={"help": "Logging level as a string. Example: 'info' or 'warning'."}
    )
    default_seed: Optional[int] = field(
        default=None, metadata={"help": "The default seed for torch, should be an integer."}
    )
    enable_cors: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable CORS. Some apps that make requests from external domains (e.g. Cursor) require "
                "CORS to be enabled."
            ),
        },
    )

    # TODO
    # Testing
    # As of 2025-07-11, testing on https://github.com/openai/openai-responses-starter-app/, validation on the
    # Response input is failing. The app works well without validation. Enable at some point in the future.
    input_validation: bool = field(
        default=False,
        metadata={
            "help": ("Whether to turn on strict input validation."),
        },
    )
    force_model: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name of the model to be forced on all requests. This is useful for testing Apps that don't allow "
                "changing models in the request."
            ),
        },
    )

    def __post_init__(self):
        """Only used for BC `torch_dtype` argument."""
        # In this case only the BC torch_dtype was given
        if self.torch_dtype is not None and self.dtype == "auto":
            self.dtype = self.torch_dtype


class ServeCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        dataclass_types = (ServeArguments,)
        serve_parser = parser.add_parser("serve", dataclass_types=dataclass_types)
        serve_parser.set_defaults(func=serve_command_factory)

    def __init__(self, args: ServeArguments):
        if not serve_dependencies_available:
            raise ImportError(
                "Missing dependencies for the serving CLI. Please install with `pip install transformers[serving]`"
            )

        # Store and process input arguments
        self.args = args
        self.use_continuous_batching = self.args.continuous_batching
        self.enable_cors = self.args.enable_cors

        if self.args.default_seed is not None:
            torch.manual_seed(self.args.default_seed)

        # Set up logging
        transformers_logger = logging.get_logger("transformers")
        transformers_logger.setLevel(logging.log_levels[self.args.log_level.lower()])

        cb_logger = logging.get_logger("transformers.generation.continuous_batching")
        cb_logger.setLevel(logging.log_levels[self.args.log_level.lower()])

        # Internal state:
        # 1. Tracks models in memory, to prevent reloading the model unnecessarily
        self.loaded_models: dict[str, TimedModel] = {}
        self.running_continuous_batching_manager: Optional[ContinuousBatchingManager] = None

        # 2. preserves information about the last call and last KV cache, to determine whether we can reuse the KV
        # cache and avoid re-running prefil
        self.last_messages = None
        self.last_kv_cache = None
        self.last_model = None

    def _validate_request(
        self,
        request: dict,
        schema: "_TypedDictMeta",  # noqa: F821
        validator: "TypeAdapter",
        unused_fields: set,
    ):
        """
        Validates the request against the schema, and checks for unexpected keys.

        Args:
            request (`dict`):
                The request to validate.
            schema (`_TypedDictMeta`):
                The schema of the request to validate. It is a `TypedDict` definition.
            validator (`TypeAdapter`):
                The validator to use to validate the request. Built from `schema`.
            unused_fields (`set`):
                Fields accepted by `schema`, but not used in `transformers serve`.

        Raises:
            HTTPException: If the request is invalid or contains unexpected or unused fields.
        """
        logger.debug(f"Validating request: {request}")

        # Validate unexpected keys -- Pydantic doesn't validate extra keys in the request.
        input_keys = set(request.keys())
        possible_keys = schema.__mutable_keys__
        unexpected_keys = input_keys - possible_keys
        if unexpected_keys:
            logger.error(f"Unexpected keys in the request: {unexpected_keys}")
            raise HTTPException(status_code=422, detail=f"Unexpected keys in the request: {unexpected_keys}")

        if self.args.input_validation:
            # Validate expected keys
            try:
                validator.validate_python(request)
            except ValidationError as e:
                logger.error(f"Validation error: {e.errors()}")
                raise HTTPException(status_code=422, detail=e.errors())

            # Validate unused fields
            unused_fields_in_request = input_keys & unused_fields
            if unused_fields_in_request:
                logger.error(f"Unused fields in the request: {unused_fields_in_request}")
                raise HTTPException(
                    status_code=422, detail=f"Unused fields in the request: {unused_fields_in_request}"
                )

    def validate_response_request(self, request: dict):
        self._validate_request(
            request=request,
            schema=TransformersResponseCreateParamsStreaming,
            validator=response_validator,
            unused_fields=UNUSED_RESPONSE_FIELDS,
        )

    def validate_chat_completion_request(self, request: dict):
        self._validate_request(
            request=request,
            schema=TransformersCompletionCreateParamsStreaming,
            validator=completion_validator,
            unused_fields=UNUSED_CHAT_COMPLETION_FIELDS,
        )

    def validate_transcription_request(self, request: dict):
        self._validate_request(
            request=request,
            schema=TransformersTranscriptionCreateParams,
            validator=transcription_validator,
            unused_fields=UNUSED_TRANSCRIPTION_FIELDS,
        )

    def build_chat_completion_chunk(
        self,
        request_id: Optional[str] = "",
        content: Optional[int] = None,
        model: Optional[str] = None,
        role: Optional[str] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[list["ChoiceDeltaToolCall"]] = None,
        decode_stream: Optional[DecodeStream] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ) -> str:
        """
        Builds a chunk of a streaming OpenAI Chat Completion response.

        IMPORTANT: The serialized chunk won't contain empty fields (fields with `None`). Some downstream apps,
        like Cursor, assume that when the field exists, it has data.

        Args:
            request_id (`str`):
                The request ID.
            content (`str`, *optional*):
                Content of the response from the model.
            model (`str`, *optional*):
                The model that generated the content.
            role (`str`, *optional*):
                The role of the next content, until a new role is defined.
            finish_reason (`str`, *optional*):
                The reason the generation by the model has finished.
            tool_calls (`list[ChoiceDeltaToolCall]`, *optional*):
                Data about the tool calls, when they are triggered.

        Returns:
            `str`: The built chunk, a string containing a JSON string with the payload.
        """
        if decode_stream is not None and content is not None and tokenizer is not None:
            content = decode_stream.step(tokenizer._tokenizer, content)
        chunk = ChatCompletionChunk(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        content=content,
                        role=role,
                        tool_calls=tool_calls,
                    ),
                    index=0,
                    finish_reason=finish_reason,
                )
            ],
            system_fingerprint="",
            object="chat.completion.chunk",
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def build_response_event(self, response: "BaseModel") -> str:
        """
        Builds a event of a streaming OpenAI Response response.

        IMPORTANT: The serialized chunk won't contain empty fields (fields with `None`). Some downstream apps,
        like Cursor, assume that when the field exists, it has data.

        Args:
            response (`BaseModel`):
                The response to build an event from. One of the multiple OpenAI Response output types

        Returns:
            `str`: The built chunk, a string containing a JSON string with the payload.
        """
        return f"data: {response.model_dump_json(exclude_none=True)}\n\n"

    def run(self):
        """
        Setup and run the FastAPI server for transformers serve.

        Models will be loaded and unloaded automatically based on usage and a timeout.

        The server will expose the following endpoints:
        - POST /v1/chat/completions: Generates chat completions.
        - POST /v1/responses: Generates responses.
        - POST /v1/audio/transcriptions: Generates transcriptions from audio.
        - GET /v1/models: Lists available models for 3rd party tools.

        Requires FastAPI and Uvicorn to be installed.
        """

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            for model in self.loaded_models.values():
                model.delete_model()
            if self.running_continuous_batching_manager is not None:
                self.running_continuous_batching_manager.stop(block=True, timeout=5)

        app = FastAPI(lifespan=lifespan)

        # Some apps that make requests from external domains (e.g. Cursor) require CORS to be enabled. However, for
        # security purposes, it's disabled by default
        if self.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            logger.warning_once(
                "CORS allow origin is set to `*`. This is not recommended for production environments."
            )
        else:
            logger.warning_once(
                "Some apps may require CORS. Consider launching the server with `--enable-cors` if you see errors."
            )

        @app.post("/v1/chat/completions")
        def chat_completion(request: dict):
            self.validate_chat_completion_request(request=request)

            if self.use_continuous_batching:
                output = self.continuous_batching_chat_completion(request)
            else:
                output = self.generate_chat_completion(request)
            return StreamingResponse(output, media_type="text/event-stream")

        @app.post("/v1/responses")
        def responses(request: dict):
            self.validate_response_request(request=request)

            output = self.generate_response(request)
            return StreamingResponse(output, media_type="text/event-stream")

        from fastapi import Request

        @app.post("/v1/audio/transcriptions")
        async def audio_transcriptions(request: Request):
            # Parses the multipart/form-data request into the request format used by other endpoints
            async with request.form() as form:
                parsed_request = TransformersTranscriptionCreateParams(
                    file=await form["file"].read(),
                    model=form["model"],
                    # TODO: add other fields
                )
                logger.debug(
                    f"Received file: {form['file'].filename}; MIME type: {form['file'].content_type}; "
                    f"size: {form['file'].size / 1024:.2f} KiB"
                )
            self.validate_transcription_request(request=parsed_request)

            output = self.generate_transcription(parsed_request)
            return StreamingResponse(output, media_type="text/event-stream")

        @app.options("/v1/models")
        @app.get("/v1/models")
        def get_all_models():
            return JSONResponse({"object": "list", "data": self.get_gen_models()})

        @app.get("/health")
        def healthcheck():
            return JSONResponse({"status": "ok"})

        uvicorn.run(app, host=self.args.host, port=self.args.port, log_level=self.args.log_level)

    @functools.cache
    def get_gen_models(self) -> list[dict[str, any]]:
        """
        This is by no means a limit to which models may be instantiated with `transformers serve`: any chat-based
        model working with generate can work.

        This is a limited list of models to ensure we have a discoverable /v1/models endpoint for third-party
        integrations.
        """
        models = [
            "Menlo/Jan-nano",
            "Menlo/Jan-nano-128k",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "HuggingFaceTB/SmolVLM-Instruct",
            "ibm-granite/granite-vision-3.2-2b",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ]

        if HF_HUB_OFFLINE:
            return [
                {
                    "id": model,
                    "object": "model",
                    "created": datetime.datetime.now().timestamp(),
                    "owned_by": model.split("/")[0],
                }
                for model in models
            ]
        else:
            model_infos = [model_info(model) for model in models]
            return [
                {
                    "id": model.id,
                    "object": "model",
                    "created": model.created_at.timestamp(),
                    "owned_by": model.author,
                }
                for model in model_infos
            ]

    def continuous_batching_chat_completion(self, req: dict) -> AsyncGenerator[str, None]:
        """
        Generates an OpenAI Chat Completion using continuous batching.

        Args:
            req (`dict`): The request to generate an OpenAI Chat Completion for.

        Returns:
            `Generator[str, None, None]`: A generator that yields the OpenAI Chat Completion chunks.
        """

        model_id_and_revision = self.process_model_name(req["model"])
        must_discard_cache = model_id_and_revision != self.last_model
        self.last_model = model_id_and_revision
        if must_discard_cache:
            # When switching models, terminate a continuous batching manager if it is running.
            if self.running_continuous_batching_manager is not None:
                self.running_continuous_batching_manager.stop(block=True, timeout=2)
                self.running_continuous_batching_manager = None
        model, processor = self.load_model_and_processor(model_id_and_revision)

        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

        generation_config = create_generation_config_from_req(
            req,
            model_generation_config=model.generation_config,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
            do_sample=False,
            scheduler="fifo",
        )

        if self.running_continuous_batching_manager is None:
            self.running_continuous_batching_manager = model.init_continuous_batching(
                generation_config=generation_config, streaming=True
            )

            # TODO (Joao, Lysandre): the logits processors should be fixed in continuous batching
            # and correctly applied in non-cb
            self.running_continuous_batching_manager.logit_processor = LogitsProcessorList()
            self.running_continuous_batching_manager.start()

        # TODO (Joao, Lysandre): this should also work with tool support
        inputs = processor.apply_chat_template(req["messages"], return_tensors="pt", add_generation_prompt=True).to(
            model.device
        )

        def stream_chat_completion(request_id, decode_stream):
            try:
                # Emit the assistant role to start the stream. Other chunks won't have a role, as it is implicit
                # they come from the assistant.
                yield self.build_chat_completion_chunk(request_id, role="assistant", model=model_id_and_revision)

                for result in self.running_continuous_batching_manager.request_id_iter(request_id):
                    if result.status == RequestStatus.FINISHED:
                        yield self.build_chat_completion_chunk(
                            request_id,
                            finish_reason="stop",
                            model=model_id_and_revision,
                        )
                        break
                    else:
                        yield self.build_chat_completion_chunk(
                            request_id=request_id,
                            content=result.generated_tokens[-1],
                            model=model_id_and_revision,
                            decode_stream=decode_stream,
                            tokenizer=tokenizer,
                        )

            except Exception as e:
                logger.error(str(e))
                self.running_continuous_batching_manager.cancel_request(request_id)
                yield f'data: {{"error": "{str(e)}"}}'

        async def cancellation_wrapper(_inputs):
            request_id = None
            try:
                decode_stream = DecodeStream(_inputs.tolist(), False)
                request_id = self.running_continuous_batching_manager.add_request(
                    _inputs, request_id=req.get("request_id"), max_new_tokens=generation_config.max_new_tokens
                )
                for chunk in stream_chat_completion(request_id, decode_stream):
                    yield chunk
                    await asyncio.sleep(0)  # Yield control to the event loop to check for cancellations
            except asyncio.CancelledError:
                if request_id is not None:
                    logger.warning(f"Request {request_id} was cancelled. Cleaning up.")
                    self.running_continuous_batching_manager.cancel_request(request_id)

        return cancellation_wrapper(inputs[0])

    @staticmethod
    def get_model_modality(model: "PreTrainedModel") -> Modality:
        model_classname = model.__class__.__name__
        if model_classname in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values():
            modality = Modality.VLM
        elif model_classname in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            modality = Modality.LLM
        else:
            raise ValueError(f"Unknown modality: {model_classname}")

        return modality

    @staticmethod
    def get_processor_inputs_from_inbound_messages(messages, modality: Modality):
        processor_inputs = []

        for message in messages:
            parsed_message = {"role": message["role"], "content": []}

            if modality == Modality.LLM:
                # Input: `content` is a string or a list of dictionaries with a "text" key.
                # Output: `content` is a string.
                if isinstance(message["content"], str):
                    parsed_content = message["content"]
                elif isinstance(message["content"], list):
                    parsed_content = []
                    for content in message["content"]:
                        if content["type"] == "text":
                            parsed_content.append(content["text"])
                    parsed_content = " ".join(parsed_content)
                parsed_message["content"] = parsed_content

            elif modality == Modality.VLM:
                # Input: `content` is a string or a list of dictionaries with a "type" key (possible types: "text",
                # "image_url").
                # Output: `content` is a list of dictionaries with a "type" key
                if isinstance(message["content"], str):
                    parsed_message["content"].append({"type": "text", "text": message["content"]})
                else:
                    for content in message["content"]:
                        if content["type"] == "text":
                            parsed_message["content"].append(content)
                        elif content["type"] == "image_url":
                            if "base64" in content["image_url"]["url"]:
                                image_data = re.sub("^data:image/.+;base64,", "", content["image_url"]["url"])
                                image = Image.open(BytesIO(base64.b64decode(image_data)))

                                file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                                url = file.name

                                image.save(file.name)
                            else:
                                url = content["image_url"]["url"]

                            parsed_message["content"].append({"type": "image", "url": url})
            processor_inputs.append(parsed_message)
        return processor_inputs

    def generate_chat_completion(self, req: dict) -> Generator[str, None, None]:
        """
        Generates an OpenAI Chat Completion using `generate`.

        Args:
            req (`dict`): The request to generate an OpenAI Chat Completion for.

        Returns:
            `Generator[str, None, None]`: A generator that yields the OpenAI Chat Completion chunks.
        """
        if self.args.force_model is not None:
            req["model"] = self.args.force_model

        messages: Iterable[ChatCompletionMessageParam] = req["messages"]

        # HACK for tiny-agents: it sends a request after the assistant message (???). Let's assume we can't have a
        # request whose last message is from the assistant.
        if messages[-1]["role"] == "assistant":
            return

        model_id_and_revision = self.process_model_name(req["model"])
        must_discard_cache = model_id_and_revision != self.last_model

        self.last_model = model_id_and_revision
        model, processor = self.load_model_and_processor(model_id_and_revision)

        modality = self.get_model_modality(model)
        processor_inputs = self.get_processor_inputs_from_inbound_messages(messages, modality)

        # ====== TOOL PREPROCESSING LOGIC ======
        tool_model_family = None
        for supported_model_families in _MODELS_WITH_TOOL_SUPPORT:
            if supported_model_families in model.config.architectures[0].lower():
                tool_model_family = supported_model_families
                break
        # TODO: trigger 2 constrained generations after the tool call start token is emitted:
        # 1. force generation to pick from the tool names
        # 2. force generation to pick from that tool's arguments
        # ====== END OF TOOL PREPROCESSING LOGIC ======

        inputs = processor.apply_chat_template(
            processor_inputs,
            add_generation_prompt=True,
            tools=req.get("tools"),
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        inputs = inputs.to(model.device)
        request_id = req.get("request_id", "req_0")

        # Temporary hack for GPTOSS 1: don't filter special tokens
        skip_special_tokens = True
        if "gptoss" in model.config.architectures[0].lower():
            skip_special_tokens = False

        generation_streamer = TextIteratorStreamer(
            processor,
            skip_special_tokens=skip_special_tokens,
            skip_prompt=True,
        )
        generation_config = create_generation_config_from_req(req, model_generation_config=model.generation_config)

        last_kv_cache = None
        if self.is_continuation(req) and not must_discard_cache:
            last_kv_cache = self.last_kv_cache

        generation_kwargs = {
            **inputs,
            "streamer": generation_streamer,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "past_key_values": last_kv_cache,
        }

        def stream_chat_completion(streamer, _request_id):
            # Temporary hack for GPTOS 2: filter out the CoT tokens. Full solution here implies defining new output
            # classes and piping the reasoning trace into a new field
            filter_cot = False
            cot_trace_end = None
            if "gptoss" in model.config.architectures[0].lower():
                filter_cot = True
                cot_trace_end = "<|channel|>final<|message|>"

            # Thin wrapper to save the KV cache after generation
            def generate_with_cache(**kwargs):
                generate_output = model.generate(**kwargs)
                self.last_kv_cache = generate_output.past_key_values

            thread = Thread(target=generate_with_cache, kwargs=generation_kwargs)
            results = ""

            try:
                thread.start()
                tool_state = ToolState()

                # Emit the assistant role to start the stream. Other chunks won't have a role, as it is implicit
                # they come from the assistant.
                yield self.build_chat_completion_chunk(request_id, role="assistant", model=model_id_and_revision)

                for result in streamer:
                    # Temporary hack for GPTOS 3: don't emit the final "<|return|>"
                    if "gptoss" in model.config.architectures[0].lower():
                        if result.endswith("<|return|>"):
                            result = result[: -len("<|return|>")]
                    results += result

                    # (related to temporary hack 2)
                    if filter_cot:
                        if cot_trace_end in results:  # end of reasoning trace observed -> stop filtering
                            filter_cot = False
                            continue
                        else:
                            continue

                    # ====== TOOL CALL LOGIC ======
                    if tool_model_family is not None:
                        # Start of a tool call: reset state variables, set `inside_tool_call`
                        if result.strip() == _TOOL_CALL_TOKENS[tool_model_family]["start"]:
                            tool_state.inside_tool_call = True
                            continue

                        # End of tool call: reset `inside_tool_call`, emit a `finish_reason`
                        if result.strip() == _TOOL_CALL_TOKENS[tool_model_family]["end"]:
                            tool_state.reset()
                            yield self.build_chat_completion_chunk(
                                request_id=_request_id,
                                role=None,
                                finish_reason="tool_calls",
                                model=model_id_and_revision,
                            )

                            continue
                        # Inside a tool call
                        if tool_state.inside_tool_call:
                            tool_state.buffer += result

                            # First step: extract the tool name (may need several tokens, and we can't emit a delta
                            # until we have the full name)
                            if not tool_state.has_tool_name_defined:
                                tool_name = re.search(r"\"name\": \"(.*?)\"", tool_state.buffer)
                                if tool_name is None:
                                    continue
                                else:
                                    tool_name = tool_name.group(1)
                                tool_state.has_tool_name_defined = True
                                tool = ChoiceDeltaToolCall(
                                    function=ChoiceDeltaToolCallFunction(name=tool_name),
                                    index=0,
                                    type="function",
                                    id=_request_id + "_tool_call",  # Only the first tool call delta has an id
                                )

                            # Second step: extract tool arguments. The tool arguments can be seen as a json string
                            # within the tool json string. We emit a delta for the arguments.
                            else:
                                # Empty text: skip
                                if result == "":
                                    continue
                                # Until we see the `"arguments": {` in the buffer, we skip
                                # TODO: other models will likely need more elaborate processing here
                                if '"arguments": {' not in tool_state.buffer:
                                    continue

                                # Handle nesting. We want to exclude the last } from the emitted arguments (it's
                                # closing the outermost nesting level, outside the arguments block)
                                tool_state.arg_nesting_level += result.count("{")
                                tool_state.arg_nesting_level -= result.count("}")
                                if tool_state.arg_nesting_level < 0:
                                    result = "".join(result.split("}")[:-2]) + "}"  # e.g. "4}}\n" -> "4}"

                                tool = ChoiceDeltaToolCall(
                                    function=ChoiceDeltaToolCallFunction(arguments=result),
                                    index=0,
                                    type="function",
                                )

                            yield self.build_chat_completion_chunk(
                                request_id=_request_id, role=None, tool_calls=[tool], model=model_id_and_revision
                            )
                            continue
                    # ====== END OF TOOL CALL LOGIC ======

                    # All non-tool related tokens are emitted as assistant messages. Empty text is skipped.
                    if result != "":
                        yield self.build_chat_completion_chunk(
                            _request_id, content=result, model=model_id_and_revision
                        )
                yield self.build_chat_completion_chunk(_request_id, finish_reason="stop", model=model_id_and_revision)

                thread.join()
            except Exception as e:
                logger.error(str(e))
                yield f'data: {{"error": "{str(e)}"}}'

            finally:
                thread.join()

        return stream_chat_completion(generation_streamer, request_id)

    def generate_response(self, req: dict) -> Generator[str, None, None]:
        """
        Generates an OpenAI Response using `generate`.

        Args:
            req (`dict`): The request to generate an OpenAI Response for.

        Returns:
            `Generator[str, None, None]`: A generator that yields the OpenAI Response events.
        """
        # TODO -- Implement non-streaming mode
        model_id_and_revision = self.process_model_name(req["model"])
        must_discard_cache = model_id_and_revision != self.last_model
        self.last_model = model_id_and_revision
        model, processor = self.load_model_and_processor(model_id_and_revision)

        if isinstance(req["input"], str):
            inputs = [{"role": "system", "content": req["instructions"]}] if "instructions" in req else []
            inputs.append({"role": "user", "content": req["input"]})
        elif isinstance(req["input"], list):
            if "instructions" in req:
                if req["input"][0]["role"] != "system":
                    inputs = [{"role": "system", "content": req["instructions"]}, *req["input"]]
                else:
                    inputs = req["input"]
                    inputs[0]["content"] = req["instructions"]
            else:
                inputs = req["input"]
        elif isinstance(req["input"], dict):
            inputs = [{"role": "system", "content": req["instructions"]}] if "instructions" in req else []
            inputs.append(req["input"])
        else:
            raise ValueError("inputs should be a list, dict, or str")

        inputs = processor.apply_chat_template(inputs, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        request_id = req.get("previous_response_id", "req_0")

        # Temporary hack for GPTOSS 1: don't filter special tokens
        skip_special_tokens = True
        if "gptoss" in model.config.architectures[0].lower():
            skip_special_tokens = False

        generation_streamer = TextIteratorStreamer(
            processor,
            skip_special_tokens=skip_special_tokens,
            skip_prompt=True,
        )
        generation_config = create_generation_config_from_req(req, model_generation_config=model.generation_config)

        last_kv_cache = None
        if self.is_continuation(req) and not must_discard_cache:
            last_kv_cache = self.last_kv_cache

        generation_kwargs = {
            "inputs": inputs,
            "attention_mask": torch.ones_like(inputs),
            "streamer": generation_streamer,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "past_key_values": last_kv_cache,
        }

        def stream_response(streamer, _request_id):
            # Temporary hack for GPTOS 2: filter out the CoT tokens. Full solution here implies defining new output
            # classes and piping the reasoning trace into a new field
            filter_cot = False
            cot_trace_end = None
            if "gptoss" in model.config.architectures[0].lower():
                filter_cot = True
                cot_trace_end = "<|channel|>final<|message|>"

            # Thin wrapper to save the KV cache after generation
            def generate_with_cache(**kwargs):
                generate_output = model.generate(**kwargs)
                self.last_kv_cache = generate_output.past_key_values

            thread = Thread(target=generate_with_cache, kwargs=generation_kwargs)
            sequence_number = 0
            output_index = 0
            content_index = 0

            try:
                thread.start()
                created_at = time.time()  # the spec expects a unix timestamp in seconds

                # We start by acknowledging the request (the request has `status="queued"`), and then by moving it to
                # in progress (`status="in_progress"`)
                response_created = ResponseCreatedEvent(
                    type="response.created",
                    sequence_number=sequence_number,
                    response=Response(
                        id=f"resp_{request_id}",
                        created_at=created_at,
                        status="queued",
                        model=model_id_and_revision,
                        instructions=req.get("instructions"),
                        text={"format": {"type": "text"}},
                        object="response",
                        tools=[],
                        output=[],
                        parallel_tool_calls=req.get("parallel_tool_calls", False),
                        tool_choice="auto",
                        metadata=req.get("metadata"),
                    ),
                )
                sequence_number += 1
                yield self.build_response_event(response_created)

                response_in_progress = ResponseInProgressEvent(
                    type="response.in_progress",
                    sequence_number=sequence_number,
                    response=Response(
                        id=f"resp_{request_id}",
                        created_at=created_at,
                        status="in_progress",
                        model=model_id_and_revision,
                        instructions=req.get("instructions"),
                        text={"format": {"type": "text"}},
                        object="response",
                        tools=[],
                        output=[],
                        parallel_tool_calls=req.get("parallel_tool_calls", False),
                        tool_choice="auto",
                        metadata=req.get("metadata"),
                    ),
                )
                sequence_number += 1
                yield self.build_response_event(response_in_progress)

                # Start the output item. Emit the assistant role to start the stream. Other chunks won't have a role,
                # as it is implicit
                response_output_item_added = ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=sequence_number,
                    output_index=output_index,
                    item=ResponseOutputMessage(
                        id=f"msg_{request_id}", type="message", status="in_progress", role="assistant", content=[]
                    ),
                )
                sequence_number += 1
                yield self.build_response_event(response_output_item_added)

                # Start the content part of the event
                response_content_part_added = ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    item_id=f"msg_{request_id}",
                    sequence_number=sequence_number,
                    output_index=output_index,
                    content_index=content_index,
                    part=ResponseOutputText(type="output_text", text="", annotations=[]),
                )
                sequence_number += 1
                yield self.build_response_event(response_content_part_added)

                # Stream the actual generated text
                results = ""
                for result in streamer:
                    # Temporary hack for GPTOS 3: don't emit the final "<|return|>"
                    if "gptoss" in model.config.architectures[0].lower():
                        if result.endswith("<|return|>"):
                            result = result[: -len("<|return|>")]
                    results += result

                    # (related to temporary hack 2)
                    if filter_cot:
                        if cot_trace_end in results:  # end of reasoning trace observed -> stop filtering
                            filter_cot = False
                            results = ""  # reset the results -> results will now track the final response
                            continue
                        else:
                            continue

                    response_output_text_delta = ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        item_id=f"msg_{request_id}",
                        sequence_number=sequence_number,
                        output_index=output_index,
                        content_index=content_index,
                        delta=result,
                        logprobs=[{"token": "", "logprob": 99.9}],  # TODO: add actual logprobs
                    )
                    sequence_number += 1
                    yield self.build_response_event(response_output_text_delta)

                # Signal the end of the text generation
                response_output_text_done = ResponseTextDoneEvent(
                    type="response.output_text.done",
                    item_id=f"msg_{request_id}",
                    sequence_number=sequence_number,
                    output_index=output_index,
                    content_index=0,
                    text=results,
                    logprobs=[{"token": "", "logprob": 99.9}],  # TODO: add actual logprobs
                )
                sequence_number += 1
                yield self.build_response_event(response_output_text_done)

                # Complete the content part
                response_content_part_done = ResponseContentPartDoneEvent(
                    type="response.content_part.done",
                    item_id=f"msg_{request_id}",
                    sequence_number=sequence_number,
                    output_index=output_index,
                    content_index=content_index,
                    part=ResponseOutputText(type="output_text", text=response_output_text_done.text, annotations=[]),
                )
                sequence_number += 1
                content_index += 1
                yield self.build_response_event(response_content_part_done)

                # Complete the output item
                response_output_item_done = ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=sequence_number,
                    output_index=output_index,
                    item=ResponseOutputMessage(
                        id=f"msg_{request_id}",
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[response_content_part_done.part],
                        annotations=[],
                    ),
                )
                sequence_number += 1
                output_index += 1
                yield self.build_response_event(response_output_item_done)

                # Finally, Complete the event
                response_completed = ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=sequence_number,
                    response=Response(
                        id=f"resp_{request_id}",
                        created_at=created_at,
                        status="completed",
                        model=model_id_and_revision,
                        instructions=req.get("instructions"),
                        text={"format": {"type": "text"}},
                        output=[response_output_item_done.item],
                        object="response",
                        tools=[],
                        parallel_tool_calls=req.get("parallel_tool_calls", False),
                        tool_choice="auto",
                        metadata=req.get("metadata"),
                    ),
                )
                sequence_number += 1
                yield self.build_response_event(response_completed)

                thread.join()
            except Exception as e:
                logger.error(f"Exception in response generation: {str(e)}")
                error_event = ResponseErrorEvent(
                    type="error",
                    sequence_number=sequence_number,
                    message=str(e),
                )
                sequence_number += 1
                yield self.build_response_event(error_event)

                response_failed = ResponseFailedEvent(
                    type="response.failed",
                    sequence_number=sequence_number,
                    response=Response(
                        id=f"resp_{request_id}",
                        created_at=created_at,
                        status="failed",
                        model=model_id_and_revision,
                        instructions=req.get("instructions"),
                        text={"format": {"type": "text"}},
                        output=[],
                        object="response",
                        tools=[],
                        parallel_tool_calls=False,
                        tool_choice="auto",
                        metadata=req.get("metadata"),
                        error=ResponseError(
                            code="server_error",
                            message=str(e),
                        ),
                    ),
                )
                sequence_number += 1
                yield self.build_response_event(response_failed)

            finally:
                thread.join()

        return stream_response(generation_streamer, request_id)

    def generate_transcription(self, req: dict) -> Generator[str, None, None]:
        """
        Generates an OpenAI Transcription using the audio file.

        Args:
            req (`dict`): The request containing the audio file and model information.

        Returns:
            `Generator[str, None, None]`: A generator that yields the transcription result.
        """
        # TODO: implement streaming transcription (currently, it's not streaming)
        if not is_librosa_available():
            raise ImportError(
                "Missing librosa dependency for audio transcription. Please install with `pip install librosa`"
            )
        model_id_and_revision = self.process_model_name(req["model"])
        audio_model, audio_processor = self.load_audio_model_and_processor(model_id_and_revision)

        generation_streamer = TextIteratorStreamer(
            audio_processor.tokenizer, skip_special_tokens=True, skip_prompt=True
        )
        generation_config = create_generation_config_from_req(
            req, model_generation_config=audio_model.generation_config
        )

        # Read the binary audio file using librosa
        model_sampling_rate = audio_processor.feature_extractor.sampling_rate
        audio_bytes = io.BytesIO(req["file"])
        audio_array, _ = librosa.load(audio_bytes, sr=model_sampling_rate, mono=True)
        audio_inputs = audio_processor(audio_array, sampling_rate=model_sampling_rate, return_tensors="pt").to(
            audio_model.device
        )
        audio_inputs["input_features"] = audio_inputs["input_features"].to(audio_model.dtype)

        generation_kwargs = {
            "streamer": generation_streamer,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
        }

        def _generate_transcription():
            generated_ids = audio_model.generate(**audio_inputs, **generation_kwargs)
            transcription_text = audio_processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
            transcription = Transcription(text=transcription_text)
            yield f"{transcription.model_dump_json(exclude_none=True)}"

        return _generate_transcription()

    def is_continuation(self, req: dict) -> bool:
        """
        Determines whether the current request is a continuation of the last request. In other words, if it is the
        same chat session.

        Args:
            req (`dict`): The request to check.

        Returns:
            `True` if the request is a continuation of the last request, `False` otherwise.
        """
        messages = req.get("messages") or req.get("input")  # ChatCompletion and Response have different fields
        req_continues_last_messages = True

        # No cached messages: this is a new request
        if self.last_messages is None:
            req_continues_last_messages = False
        # The new request has no new rounds of conversation: this is a new request
        elif len(self.last_messages) >= len(messages):
            req_continues_last_messages = False
        # Otherwise, check that the last messages are a subset of the new request
        else:
            for i in range(len(self.last_messages)):
                if self.last_messages[i] != messages[i]:
                    req_continues_last_messages = False
                    break

        self.last_messages = messages
        return req_continues_last_messages

    @staticmethod
    def get_quantization_config(args: ServeArguments) -> Optional["BitsAndBytesConfig"]:
        """
        Returns the quantization config for the given CLI arguments.

        Args:
            args (`ServeArguments`): The serve arguments. May contain quantization settings, device, etc.

        Returns:
            `Optional[BitsAndBytesConfig]`: The quantization config.
        """
        if args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # For consistency with model weights, we use the same value as `dtype`
                bnb_4bit_compute_dtype=args.dtype,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=args.use_bnb_nested_quant,
                bnb_4bit_quant_storage=args.dtype,
            )
        elif args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        return quantization_config

    def process_model_name(self, model_id: str) -> str:
        """
        Applies the `force_model` CLI argument and canonicalizes the model name to the format "model_id@revision".
        If the model_id DOESN'T contain an @, it defaults to "model_id@main".

        Args:
            model_id (`str`): The model ID.

        Returns:
            `str`: The canonicalized model name to be used
        """
        if self.args.force_model is not None:
            model_id = self.args.force_model
        if "@" in model_id:
            return model_id
        return f"{model_id}@main"

    def _load_model_and_data_processor(self, model_id_and_revision: str):
        """
        Generic method to load a model and a data processor from a model ID and revision, making use of the serve CLI
        arguments.

        Args:
            model_id_and_revision (`str`):
                The model ID and revision to load.
            model_cls (`type[PreTrainedModel]`):
                The model class to load.

        Returns:
            `tuple[PreTrainedModel, Union[ProcessorMixin, PreTrainedTokenizerFast]]`: The loaded model and
            data processor (tokenizer, audio processor, etc.).
        """
        args = self.args
        logger.info(f"Loading {model_id_and_revision}")

        if "@" in model_id_and_revision:
            model_id, revision = model_id_and_revision.split("@", 1)
        else:
            model_id, revision = model_id_and_revision, "main"

        data_processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=args.trust_remote_code,
        )

        dtype = args.dtype if args.dtype in ["auto", None] else getattr(torch, args.dtype)
        quantization_config = self.get_quantization_config(args)

        model_kwargs = {
            "revision": revision,
            "attn_implementation": args.attn_implementation,
            "dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": args.trust_remote_code,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        architecture = getattr(transformers, config.architectures[0])
        model = architecture.from_pretrained(model_id, **model_kwargs)

        if getattr(model, "hf_device_map", None) is None:
            model = model.to(args.device)

        has_default_max_length = (
            model.generation_config.max_new_tokens is None and model.generation_config.max_length == 20
        )
        has_short_max_new_tokens = (
            model.generation_config.max_new_tokens is not None and model.generation_config.max_new_tokens < 1024
        )
        if has_default_max_length or has_short_max_new_tokens:
            model.generation_config.max_new_tokens = 1024

        logger.info(f"Loaded model {model_id_and_revision}")
        return model, data_processor

    def load_model_and_processor(
        self, model_id_and_revision: str
    ) -> tuple["PreTrainedModel", PreTrainedTokenizerFast]:
        """
        Loads the text model and processor from the given model ID and revision into the ServeCommand instance.

        Args:
            model_id_and_revision (`str`):
                The model ID and revision to load.

        Returns:
            `tuple[PreTrainedModel, PreTrainedTokenizerFast]`: The loaded text model and processor.
        """
        if model_id_and_revision not in self.loaded_models or self.loaded_models[model_id_and_revision].is_deleted():
            model, processor = self._load_model_and_data_processor(model_id_and_revision)
            self.loaded_models[model_id_and_revision] = TimedModel(
                model,
                timeout_seconds=self.args.model_timeout,
                processor=processor,
            )
        else:
            self.loaded_models[model_id_and_revision].reset_timer()
            model = self.loaded_models[model_id_and_revision].model
            processor = self.loaded_models[model_id_and_revision].processor

        return model, processor

    def load_audio_model_and_processor(self, model_id_and_revision: str) -> tuple["PreTrainedModel", ProcessorMixin]:
        """
        Loads the audio model and processor from the given model ID and revision into the ServeCommand instance.

        Args:
            model_id_and_revision (`str`):
                The model ID and revision to load.

        Returns:
            `tuple[PreTrainedModel, ProcessorMixin]`: The loaded audio model and processor.
        """
        if model_id_and_revision not in self.loaded_models or self.loaded_models[model_id_and_revision].is_deleted():
            audio_model, audio_processor = self._load_model_and_data_processor(model_id_and_revision)
            self.loaded_models[model_id_and_revision] = TimedModel(
                audio_model,
                timeout_seconds=self.args.model_timeout,
                processor=audio_processor,
            )
        else:
            self.loaded_models[model_id_and_revision].reset_timer()
            audio_model = self.loaded_models[model_id_and_revision].model
            audio_processor = self.loaded_models[model_id_and_revision].processor

        return audio_model, audio_processor


if __name__ == "__main__":
    serve = ServeCommand()
    serve.run()
