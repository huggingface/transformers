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

import copy
import functools
import json
import re
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from threading import Thread
from typing import Generator, Optional

from huggingface_hub import ModelInfo, model_info

from transformers.utils.import_utils import (
    is_fastapi_available,
    is_openai_available,
    is_pydantic_available,
    is_uvicorn_available,
)

from .. import LogitsProcessorList, PreTrainedTokenizerFast, TextIteratorStreamer
from ..generation.continuous_batching import ContinuousBatchingManager, RequestStatus
from ..utils import is_torch_available, logging
from . import BaseTransformersCLICommand


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
        PreTrainedModel,
    )

if is_pydantic_available() and is_fastapi_available() and is_uvicorn_available() and is_openai_available():
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
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
        ResponseInProgressEvent,
        ResponseOutputItem,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseOutputText,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )
    from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
    from pydantic import BaseModel, TypeAdapter, ValidationError
    from typing_extensions import _TypedDictMeta

    # Expand OpenAI's request input types with an optional `generation_config` field
    class TransformersResponseCreateParamsStreaming(ResponseCreateParamsStreaming, total=False):
        """
        OpenAI's ResponseCreateParamsStreaming with an additional field for the generation config (as a json string).
        """

        generation_config: Optional[str]

    class TransformersCompletionCreateParamsStreaming(CompletionCreateParamsStreaming, total=False):
        """
        OpenAI's CompletionCreateParamsStreaming with an additional field for the generation config (as a json string).
        """

        generation_config: Optional[str]

    # Contrarily to OpenAI's output types, input types are `TypedDict`, which don't have validation
    response_validator = TypeAdapter(TransformersResponseCreateParamsStreaming)
    completion_validator = TypeAdapter(TransformersCompletionCreateParamsStreaming)

    # Define request fields that are not yet used in `transformers serve`. Receiving these fields will raise an
    # HTTPException.
    UNUSED_RESPONSE_FIELDS = {
        "background",
        "include",
        "max_tool_calls",
        "metadata",
        "parallel_tool_calls",
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
    UNUSED_COMPLETION_FIELDS = {
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

    # Completion-specific parameters
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


@dataclass
class ServeArguments:
    r"""
    Arguments for the serve CLI.

    See the metadata arg for each argument's description -- the metadata will be printed with
    `transformers serve --help`
    """

    device: str = field(default="cpu", metadata={"help": "Device to use for inference."})
    torch_dtype: Optional[str] = field(
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

    # Other settings
    log_level: str = field(
        default="info", metadata={"help": "Logging level as a string. Example: 'info' or 'warning'."}
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


class ServeCommand(BaseTransformersCLICommand):
    loaded_model: Optional[str] = None
    running_continuous_batching_manager: Optional[ContinuousBatchingManager] = None

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast

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
        if not is_pydantic_available() or not is_fastapi_available() or not is_uvicorn_available():
            raise ImportError(
                "Missing dependencies for the serving CLI. Please install with `pip install transformers[serving]`"
            )

        self.args = args
        self.use_continuous_batching = self.args.attn_implementation == "sdpa_paged"
        self.enable_cors = self.args.enable_cors

        # State: preserves information about the last call and last KV cache, to determine whether we can reuse the KV
        # cache and avoid re-running prefil
        self.last_messages = None
        self.last_kv_cache = None

        transformers_logger = logging.get_logger("transformers")
        transformers_logger.setLevel(logging.log_levels[self.args.log_level.lower()])

        cb_logger = logging.get_logger("transformers.generation.continuous_batching")
        cb_logger.setLevel(logging.log_levels[self.args.log_level.lower()])

    def validate_request(self, request: dict, schema: _TypedDictMeta, validator: TypeAdapter, unused_fields: set):
        """
        Validates the request against the schema, and checks for unexpected keys.

        Args:
            request (`dict`):
                The request to validate.
            schema (`_TypedDictMeta`):
                The schema of the request to validate. It is a TypedDict definition.
            validator (`TypeAdapter`):
                The validator to use to validate the request. Built from `schema`.
            unused_fields (`set`):
                Fields accepted by `schema`, but not used in `transformers serve`.

        Raises:
            HTTPException: If the request is invalid or contains unexpected or unused fields.
        """
        logger.debug(f"Validating request: {request}")

        # Validate expected keys
        try:
            validator.validate_python(request)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())

        # Validate unexpected keys -- Pydantic doesn't validate extra keys in the request.
        input_keys = set(request.keys())
        possible_keys = schema.__mutable_keys__
        unexpected_keys = input_keys - possible_keys
        if unexpected_keys:
            raise HTTPException(status_code=422, detail=f"Unexpected keys in the request: {unexpected_keys}")

        # Validate unused fields
        unused_fields_in_request = input_keys & unused_fields
        if unused_fields_in_request:
            raise HTTPException(status_code=422, detail=f"Unused fields in the request: {unused_fields_in_request}")

    def build_chat_completions_chunk(
        self,
        request_id: Optional[str] = "",
        content: Optional[str] = None,
        role: Optional[str] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[list[ChoiceDeltaToolCall]] = None,
    ) -> str:
        """
        Builds a chunk of a streaming chat completion response.

        IMPORTANT: The serialized chunk won't contain empty fields (fields with `None`). Some downstream apps,
        like Cursor, assume that when the field exists, it has data.

        Args:
            request_id (`str`):
                The request ID.
            content (`str`, *optional*):
                Content of the response from the model.
            role (`str`, *optional*):
                The role of the next content, until a new role is defined.
            finish_reason (`str`, *optional*):
                The reason the generation by the model has finished.
            tool_calls (`list[ChoiceDeltaToolCall]`, *optional*):
                Data about the tool calls, when they are triggered.

        Returns:
            `str`: The built chunk, a string containing a JSON string with the payload.
        """
        chunk = ChatCompletionChunk(
            id=request_id,
            created=int(time.time()),
            model=self.loaded_model,
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

    def build_responses_chunk(self, response: BaseModel) -> str:
        return f"data: {response.model_dump_json(exclude_none=True)}\n\n"

    def run(self):
        app = FastAPI()

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

        @app.post("/v1/chat/completions")
        def chat_completion(request: dict):
            self.validate_request(
                request=request,
                schema=TransformersCompletionCreateParamsStreaming,
                validator=completion_validator,
                unused_fields=UNUSED_COMPLETION_FIELDS,
            )

            if self.use_continuous_batching:
                output = self.continuous_batching(request)
            else:
                output = self.generate(request)
            return StreamingResponse(output, media_type="text/event-stream")

        @app.post("/v1/responses")
        def responses(request: dict):
            self.validate_request(
                request=request,
                schema=TransformersResponseCreateParamsStreaming,
                validator=response_validator,
                unused_fields=UNUSED_RESPONSE_FIELDS,
            )

            output = self.generate_response(request)
            return StreamingResponse(output, media_type="text/event-stream")

        @app.get("/v1/models")
        def get_all_models():
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model.id,
                            "object": "model",
                            "crated": model.created_at.timestamp(),
                            "owned_by": model.author,
                        }
                        for model in self.get_text_gen_models()
                    ],
                }
            )

        uvicorn.run(app, host=self.args.host, port=self.args.port, log_level=self.args.log_level)

    @functools.lru_cache(maxsize=None)
    def get_text_gen_models(self) -> list[ModelInfo]:
        """
        This is by no means a limit to which models may be instantiated with `transformers serve`: any chat-based
        model working with generate can work.

        This is a limited list of models to ensure we have a discoverable /v1/models endpoint for third-party
        integrations.
        """
        return [
            model_info("Menlo/Jan-nano"),
            model_info("Menlo/Jan-nano-128k"),
            model_info("Qwen/Qwen2.5-0.5B-Instruct"),
            model_info("Qwen/Qwen2.5-3B-Instruct"),
            model_info("Qwen/Qwen2.5-7B-Instruct"),
            model_info("Qwen/Qwen2.5-14B-Instruct"),
            model_info("meta-llama/Llama-3.1-8B-Instruct"),
            model_info("meta-llama/Llama-3.2-1B-Instruct"),
            model_info("meta-llama/Llama-3.3-70B-Instruct"),
        ]

    def continuous_batching(self, req: dict) -> Generator:
        update_model = self.canonicalized_model_name(req["model"]) != self.loaded_model

        if update_model:
            self.model, self.tokenizer = self.load_model_and_tokenizer(req["model"], self.args)

        generation_config = create_generation_config_from_req(
            req,
            model_generation_config=self.model.generation_config,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            num_blocks=1,
            block_size=1024,
            do_sample=False,
            max_batch_tokens=10,
            scheduler="fifo",
        )

        if self.running_continuous_batching_manager is None or update_model:
            self.running_continuous_batching_manager = self.model.init_continuous_batching(
                generation_config=generation_config, streaming=True
            )

            # TODO (Joao, Lysandre): the logits processors should be fixed in continuous batching
            # and correctly applied in non-cb
            self.running_continuous_batching_manager.logit_processor = LogitsProcessorList()
            self.running_continuous_batching_manager.start()

        # TODO (Joao, Lysandre): this should also work with tool support
        inputs = self.tokenizer.apply_chat_template(
            req["messages"], return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        def stream_response(_inputs):
            try:
                max_new_tokens = req.get("max_tokens", generation_config.max_new_tokens) or 1024
                request_id = self.running_continuous_batching_manager.add_request(
                    _inputs, request_id=req.get("request_id"), max_new_tokens=max_new_tokens
                )

                queue_is_flushed = False

                # Emit the assistant role to start the stream. Other chunks won't have a role, as it is implicit
                # they come from the assistant.
                yield self.build_chat_completions_chunk(request_id, role="assistant")

                for result in self.running_continuous_batching_manager:
                    if result.request_id != request_id:
                        continue
                    if req.get("request_id") is not None and not queue_is_flushed:
                        if result.status == RequestStatus.FINISHED:
                            continue
                        else:
                            queue_is_flushed = True

                    finish_reason = "stop" if result.status == RequestStatus.FINISHED else None
                    if result.status == RequestStatus.FINISHED:
                        yield self.build_chat_completions_chunk(request_id, finish_reason=finish_reason)
                        break
                    else:
                        yield self.build_chat_completions_chunk(request_id=request_id, content=result.next_token)

            except Exception as e:
                logger.error(str(e))
                yield f'data: {{"error": "{str(e)}"}}'

        return stream_response(inputs[0])

    def generate(self, req: dict) -> Generator:
        update_model = req["model"] != self.loaded_model
        if update_model:
            self.model, self.tokenizer = self.load_model_and_tokenizer(req["model"], self.args)

        # HACK for tiny-agents: it sends a request after the assistant message (???). Let's assume we can't have a
        # request whose last message is from the assistant.
        if req["messages"][-1]["role"] == "assistant":
            return

        # ====== TOOL PREPROCESSING LOGIC ======
        tool_model_family = None
        for supported_model_families in _MODELS_WITH_TOOL_SUPPORT:
            if supported_model_families in self.model.config.architectures[0].lower():
                tool_model_family = supported_model_families
                break
        # TODO: trigger 2 constrained generations after the tool call start token is emitted:
        # 1. force generation to pick from the tool names
        # 2. force generation to pick from that tool's arguments
        # ====== END OF TOOL PREPROCESSING LOGIC ======

        if tool_model_family is not None:
            text = self.tokenizer.apply_chat_template(
                req["messages"], add_generation_prompt=True, tokenize=False, tools=req.get("tools")
            )
        else:
            text = self.tokenizer.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=False)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)["input_ids"]
        request_id = req.get("request_id", "req_0")

        generation_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)

        generation_config = create_generation_config_from_req(
            req, model_generation_config=self.model.generation_config
        )
        max_new_tokens = req.get("max_tokens", generation_config.max_new_tokens) or 256
        generation_config.max_new_tokens = max_new_tokens

        last_kv_cache = None
        if self.is_continuation(req) and not update_model:
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
            # Thin wrapper to save the KV cache after generation
            def generate_with_cache(**kwargs):
                generate_output = self.model.generate(**kwargs)
                self.last_kv_cache = generate_output.past_key_values

            thread = Thread(target=generate_with_cache, kwargs=generation_kwargs)

            try:
                thread.start()
                tool_state = ToolState()

                # Emit the assistant role to start the stream. Other chunks won't have a role, as it is implicit
                # they come from the assistant.
                yield self.build_chat_completions_chunk(request_id, role="assistant")

                for result in streamer:
                    # ====== TOOL CALL LOGIC ======
                    if tool_model_family is not None:
                        # Start of a tool call: reset state variables, set `inside_tool_call`
                        if result.strip() == _TOOL_CALL_TOKENS[tool_model_family]["start"]:
                            tool_state.inside_tool_call = True
                            continue

                        # End of tool call: reset `inside_tool_call`, emit a `finish_reason`
                        if result.strip() == _TOOL_CALL_TOKENS[tool_model_family]["end"]:
                            tool_state.reset()
                            yield self.build_chat_completions_chunk(
                                request_id=_request_id, role=None, finish_reason="tool_calls"
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

                            yield self.build_chat_completions_chunk(
                                request_id=_request_id, role=None, tool_calls=[tool]
                            )
                            continue
                    # ====== END OF TOOL CALL LOGIC ======

                    # All non-tool related tokens are emitted as assistant messages. Empty text is skipped.
                    if result != "":
                        yield self.build_chat_completions_chunk(_request_id, content=result)
                yield self.build_chat_completions_chunk(_request_id, finish_reason="stop")

                thread.join()
            except Exception as e:
                logger.error(str(e))
                raise
                yield f'data: {{"error": "{str(e)}"}}'

            finally:
                thread.join()

        return stream_response(generation_streamer, request_id)

    def generate_response(self, req: dict) -> Generator:
        # TODO
        # Check generation config parameters
        # Implement KV caching (with previous_request_id?)
        # Implement metadata forwarding (both input and output have a metadata field)
        # Implement non-streaming mode
        # Implement different output_index and content_index than 0

        update_model = req["model"] != self.loaded_model
        if update_model:
            self.model, self.tokenizer = self.load_model_and_tokenizer(req["model"], self.args)

        text = self.tokenizer.apply_chat_template(req["input"], add_generation_prompt=True, tokenize=False)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)["input_ids"]
        request_id = req.get("previous_response_id", "req_0")

        generation_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)

        generation_config = create_generation_config_from_req(
            req, model_generation_config=self.model.generation_config
        )
        max_new_tokens = req.get("max_output_tokens", generation_config.max_new_tokens) or 256
        generation_config.max_new_tokens = max_new_tokens

        generation_kwargs = {
            "inputs": inputs,
            "attention_mask": torch.ones_like(inputs),
            "streamer": generation_streamer,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
        }

        def stream_response(streamer, _request_id):
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

            try:
                thread.start()

                created_at = time.time_ns()

                response_created = ResponseCreatedEvent(
                    type="response.created",
                    response=Response(
                        id=f"resp_{request_id}",
                        created_at=created_at,
                        status="in_progress",
                        model=self.loaded_model,
                        instructions=req.get("instructions"),
                        text={"format": {"type": "text"}},
                    ),
                )
                yield self.build_responses_chunk(response_created)

                response_in_progress = ResponseInProgressEvent(
                    type="response.in_progress",
                    response=Response(
                        id=f"resp_{request_id}",
                        created_at=created_at,
                        status="in_progress",
                        model=self.loaded_model,
                        instructions=req.get("instructions"),
                        text={"format": {"type": "text"}},
                    ),
                )
                yield self.build_responses_chunk(response_in_progress)

                response_output_item_added = ResponseOutputItemAddedEvent(
                    type="response.output_item_added",
                    output_index=0,
                    item=ResponseOutputItem(
                        id=f"msg_{request_id}", type="message", status="in_progress", role="assistant", content=[]
                    ),
                )
                yield self.build_responses_chunk(response_output_item_added)

                response_content_part_added = ResponseContentPartAddedEvent(
                    type="response.content_part.added",
                    item_id=f"msg_{request_id}",
                    output_index=0,
                    content_index=0,
                    part=ResponseOutputText(type="output_text", text="", annotations=[]),
                )
                yield self.build_responses_chunk(response_content_part_added)

                results = ""
                for result in streamer:
                    results += result
                    response_output_text_delta = ResponseTextDeltaEvent(
                        type="response.output_text_delta",
                        item_id=f"msg_{request_id}",
                        output_index=0,
                        content_index=0,
                        delta=result,
                    )
                    yield self.build_responses_chunk(response_output_text_delta)

                response_output_text_done = ResponseTextDoneEvent(
                    type="response.output_text_done",
                    item_id=f"msg_{request_id}",
                    output_index=0,
                    content_index=0,
                    text=results,
                )
                yield self.build_responses_chunk(response_output_text_done)

                response_content_part_done = ResponseContentPartDoneEvent(
                    type="response.content_part_done",
                    item_id=f"msg_{request_id}",
                    output_index=0,
                    content_index=0,
                    part=ResponseOutputText(type="output_text", text=response_output_text_done.text, annotations=[]),
                )
                yield self.build_responses_chunk(response_content_part_done)

                response_output_item_done = ResponseOutputItemDoneEvent(
                    type="response.output_item_done",
                    output_index=0,
                    item=ResponseOutputItem(
                        id=f"msg_{request_id}",
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[response_content_part_done.part],
                    ),
                )
                yield self.build_responses_chunk(response_output_item_done)

                response_completed = ResponseCompletedEvent(
                    type="response.completed",
                    response=Response(
                        id=f"resp_{request_id}",
                        created_at=created_at,
                        status="completed",
                        model=self.loaded_model,
                        instructions=req.get("instructions"),
                        text={"format": {"type": "text"}},
                        output=response_output_item_done.item,
                    ),
                )
                yield self.build_responses_chunk(response_completed)

                thread.join()
            except Exception as e:
                logger.error(str(e))
                yield f'data: {{"error": "{str(e)}"}}'

            finally:
                thread.join()

        return stream_response(generation_streamer, request_id)

    def is_continuation(self, req: dict) -> bool:
        """
        Determines whether the current request is a continuation of the last request. In other words, if it is the
        same chat session.

        Args:
            req (`dict`): The request to check.

        Returns:
            `True` if the request is a continuation of the last request, `False` otherwise.
        """
        req_continues_last_messages = True

        # No cached messages: this is a new request
        if self.last_messages is None:
            req_continues_last_messages = False
        # The new request has no new rounds of conversation: this is a new request
        elif len(self.last_messages) >= len(req["messages"]):
            req_continues_last_messages = False
        # Otherwise, check that the last messages are a subset of the new request
        else:
            for i in range(len(self.last_messages)):
                if self.last_messages[i] != req["messages"][i]:
                    req_continues_last_messages = False
                    break

        self.last_messages = req["messages"]
        return req_continues_last_messages

    @staticmethod
    def get_quantization_config(model_args: ServeArguments) -> Optional["BitsAndBytesConfig"]:
        if model_args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # For consistency with model weights, we use the same value as `torch_dtype`
                bnb_4bit_compute_dtype=model_args.torch_dtype,
                bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
                bnb_4bit_quant_storage=model_args.torch_dtype,
            )
        elif model_args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        return quantization_config

    def canonicalized_model_name(self, model_id: str) -> str:
        if "@" in model_id:
            return model_id
        return f"{model_id}@main"

    def load_model_and_tokenizer(
        self, model_id_and_revision: str, args: ServeArguments
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        logger.warning(f"Loading {model_id_and_revision}")

        if "@" in model_id_and_revision:
            model_id, revision = model_id_and_revision.split("@", 1)
        else:
            model_id, revision = model_id_and_revision, "main"

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=args.trust_remote_code,
        )

        torch_dtype = args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
        quantization_config = self.get_quantization_config(args)

        model_kwargs = {
            "revision": revision,
            "attn_implementation": args.attn_implementation,
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "quantization_config": quantization_config,
            "trust_remote_code": args.trust_remote_code,
        }

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if model.generation_config.max_new_tokens is not None and model.generation_config.max_new_tokens < 1024:
            model.generation_config.max_new_tokens = 1024

        if getattr(model, "hf_device_map", None) is None:
            model = model.to(args.device)

        self.loaded_model = f"{model_id}@{revision}"

        logger.warning(f"Loaded model {self.loaded_model}")
        return model, tokenizer


if __name__ == "__main__":
    serve = ServeCommand()
    serve.run()
