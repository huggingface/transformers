# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import json
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from logging import getLevelNamesMapping
from threading import Thread
from typing import Dict, Optional, List, Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from huggingface_hub import ChatCompletionInputTool
from huggingface_hub.inference._generated.types import BaseInferenceType
from pydantic import BaseModel

from .. import PreTrainedTokenizerFast, TextIteratorStreamer
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


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionInput(BaseModel):
    messages: list[Message]
    stream: Optional[bool] = False
    model: Optional[str] = None
    generation_config: Optional[dict] = None
    request_id: Optional[str] = None
    extra_body: Optional[Dict] = None
    tools: Any = None,

logger = logging.get_logger("transformers/serving")


def serve_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.

    Returns: ServeCommand
    """
    return ServeCommand(args)


@dataclass
class ServeArguments:
    r"""
    Arguments for the serve CLI.

    See the metadata arg for each argument's description -- the medatata will be printed with
    `transformers serve --help`
    """

    # General settings
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt."})

    # Model loading
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use (can be a branch name, tag name or commit id)."},
    )
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
    host: str = field(default="localhost", metadata={"help": "Interface the server will listen to.."})
    port: int = field(default=8000, metadata={"help": "Port the server will listen to."})

    # Other settings
    log_level: str = field(
        default="info", metadata={"help": "Logging level as a string. Example: 'info' or 'warning'."}
    )


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

        group = serve_parser.add_argument_group("Positional arguments")
        group.add_argument("model_name_or_path", type=str, default=None, help="Name of the pre-trained model.")
        serve_parser.set_defaults(func=serve_command_factory)

    def __init__(self, args: ServeArguments):
        self.args = args
        self.model, self.tokenizer = self.load_model_and_tokenizer(args)
        self.use_continuous_batching = self.args.attn_implementation == "sdpa_paged"

        cb_logger = logging.get_logger("transformers.generation.continuous_batching")
        cb_logger.setLevel(getLevelNamesMapping()[self.args.log_level.upper()])

    def build_chunk(self, content: str, tool_call: str, request_id: str) -> str:

        class ChatCompletionStreamOutputFunction(BaseInferenceType):
            arguments: str
            name: Optional[str] = None

        class ChatCompletionStreamOutputDeltaToolCall(BaseInferenceType):
            function: ChatCompletionStreamOutputFunction
            id: str
            index: int
            type: str


        payload = {
            "object": "chat.completion.chunk",
            "id": request_id,
            "created": int(time.time()),
            "model": self.args.model_name_or_path,
            "system_fingerprint": "",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content, 'tool_calls': []},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    def run(self):
        app = FastAPI()

        if self.use_continuous_batching:
            self.continuous_batching(app)
        else:
            self.generate(app)

        uvicorn.run(app, host=self.args.host, port=self.args.port, log_level=self.args.log_level)

    def continuous_batching(self, app):
        generation_config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            num_blocks=1,
            block_size=1024,
            do_sample=False,
            max_batch_tokens=10,
            scheduler="fifo",
        )

        manager: ContinuousBatchingManager = self.model.init_continuous_batching(
            generation_config=generation_config, streaming=True
        )
        manager.start()

        @app.post("/v1/chat/completions")
        def _serve(req: ChatCompletionInput):
            if not req.stream:
                return {"error": "Only streaming mode is supported."}

            chat = req.messages

            inputs = self.tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True, tools=req.tools).to(
                self.model.device
            )

            generation_config = GenerationConfig(**(req.generation_config or {}))

            def stream_response(_inputs):
                max_new_tokens = req.max_tokens or generation_config.max_new_tokens or 256
                request_id = manager.add_request(_inputs, request_id=req.request_id, max_new_tokens=max_new_tokens)
                queue_is_flushed = False

                for result in manager:
                    if req.request_id is not None and not queue_is_flushed:
                        if result.status == RequestStatus.FINISHED:
                            continue
                        else:
                            queue_is_flushed = True

                    yield self.build_chunk(result.next_token, request_id=request_id)

                    if result.status == RequestStatus.FINISHED:
                        break

                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(inputs[0]), media_type="text/event-stream")

    def generate(self, app):
        @app.post("/v1/chat/completions")
        def _serve(req: ChatCompletionInput):
            from pprint import pprint
            pprint(req.tools)
            if not req.stream:
                return {"error": "Only streaming mode is supported."}

            chat = req.messages
            request_id = req.request_id if req.request_id is not None else "req_0"
            text = self.tokenizer.apply_chat_template(chat, tools=req.tools, add_generation_prompt=True, tokenize=False)

#             tools = [
#                 {"function": {"name": "task_complete", "parameters": {"type": "object", "properties": {}}, "description": "Call this tool when the task given by the user is complete"}, "type": "function"},
#                 {"function": {"name": "ask_question", "parameters": {"type": "object", "properties": {}}, "description": "Ask the user for more info required to solve or clarify their problem."}, "type": "function"},
#                 {"function": {"name": "FLUX_1_dev_get_seed", "parameters": {"type": "object", "properties": {"randomize_seed": {"type": "boolean", "description": "If True, a random seed (an integer in [0, MAX_SEED)) is generated using NumPy's default random number generator. If False, the provided seed argument is returned as-is.", "default": True}, "seed": {"type": "number", "description": "The seed value to use if randomize_seed is False."}}}, "description": "Determine and return the random seed to use for model generation."}, "type": "function"},
#                 {"function": {"name": "FLUX_1_dev_infer", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A text prompt in English to guide the image generation. Limited to 77 tokens."}, "seed": {"type": "number", "description": "The seed value used for reproducible image generation."}, "width": {"type": "number", "description": "Width of the output image in pixels. Defaults to 1024.", "default": 1024}, "height": {"type": "number", "description": "Height of the output image in pixels. Defaults to 1024.", "default": 1024}, "guidance_scale": {"type": "number", "description": "Controls how strongly the model follows the prompt.", "default": 3.5}, "num_inference_steps": {"type": "number", "description": "Number of denoising steps during generation. Higher values can improve quality. Defaults to 28.", "default": 28}}}, "description": "Generate an image from a text prompt using the FLUX.1 [dev] model. Note: - Prompts must be written in English. Other languages are not currently supported. - Prompts are limited to 77 tokens due to CLIP tokenizer constraints."}, "type": "function"}
#             ]
#             text ="""<|im_start|>system
# You are an agent - please keep going until the user’s query is completely
# resolved, before ending your turn and yielding back to the user. Only terminate
# your turn when you are sure that the problem is solved, or if you need more
# info from the user to solve the problem.
# If you are not sure about anything pertaining to the user’s request, use your
# tools to read files and gather the relevant information: do NOT guess or make
# up an answer.
# You MUST plan extensively before each function call, and reflect extensively
# on the outcomes of the previous function calls. DO NOT do this entire process
# by making function calls only, as this can impair your ability to solve the
# problem and think insightfully.
#
# # Tools
#
# You may call one or more functions to assist with the user query.
#
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {"function": {"name": "task_complete", "parameters": {"type": "object", "properties": {}}, "description": "Call this tool when the task given by the user is complete"}, "type": "function"}
# {"function": {"name": "ask_question", "parameters": {"type": "object", "properties": {}}, "description": "Ask the user for more info required to solve or clarify their problem."}, "type": "function"}
# {"function": {"name": "FLUX_1_dev_get_seed", "parameters": {"type": "object", "properties": {"randomize_seed": {"type": "boolean", "description": "If True, a random seed (an integer in [0, MAX_SEED)) is generated using NumPy's default random number generator. If False, the provided seed argument is returned as-is.", "default": true}, "seed": {"type": "number", "description": "The seed value to use if randomize_seed is False."}}}, "description": "Determine and return the random seed to use for model generation."}, "type": "function"}
# {"function": {"name": "FLUX_1_dev_infer", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "A text prompt in English to guide the image generation. Limited to 77 tokens."}, "seed": {"type": "number", "description": "The seed value used for reproducible image generation."}, "width": {"type": "number", "description": "Width of the output image in pixels. Defaults to 1024.", "default": 1024}, "height": {"type": "number", "description": "Height of the output image in pixels. Defaults to 1024.", "default": 1024}, "guidance_scale": {"type": "number", "description": "Controls how strongly the model follows the prompt.", "default": 3.5}, "num_inference_steps": {"type": "number", "description": "Number of denoising steps during generation. Higher values can improve quality. Defaults to 28.", "default": 28}}}, "description": "Generate an image from a text prompt using the FLUX.1 [dev] model. Note: - Prompts must be written in English. Other languages are not currently supported. - Prompts are limited to 77 tokens due to CLIP tokenizer constraints."}, "type": "function"}
# </tools>
#
# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call><|im_end|>
# <|im_start|>user
# Generate me an image of a cat on the moon<|im_end|>
# <|im_start|>assistant
#             """

            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)["input_ids"]

            print(self.tokenizer.decode(inputs[0].cpu().numpy()))

            if req.generation_config is None:
                req.generation_config = {'max_new_tokens': 1024, 'max_length': 1024}

            attention_mask = torch.ones_like(inputs)
            generation_streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
            generation_config = GenerationConfig(**(req.generation_config or {}))

            generation_kwargs = {
                "inputs": inputs,
                "attention_mask": attention_mask,
                "streamer": generation_streamer,
                "generation_config": generation_config,
            }

            def stream_response(streamer, _request_id):
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                for result in streamer:
                    print(result)
                    yield self.build_chunk(result, _request_id)
                yield "data: [DONE]\n\n"

                thread.join()

            return StreamingResponse(stream_response(generation_streamer, request_id), media_type="text/event-stream")

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

    def load_model_and_tokenizer(self, args: ServeArguments) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )

        torch_dtype = args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
        quantization_config = self.get_quantization_config(args)
        model_kwargs = {
            "revision": args.model_revision,
            # "attn_implementation": "sdpa_paged",
            "torch_dtype": torch_dtype,
            "device_map": "auto",
            "quantization_config": quantization_config,
        }
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, trust_remote_code=args.trust_remote_code, **model_kwargs
        )

        if model.generation_config.max_length is not None and model.generation_config.max_length < 256:
            model.generation_config.max_length = 256
            model.generation_config.max_new_tokens = 256

        if getattr(model, "hf_device_map", None) is None:
            model = model.to(args.device)

        return model, tokenizer


if __name__ == "__main__":
    serve = ServeCommand()
    serve.run()
