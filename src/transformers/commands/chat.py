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
import copy
import json
import os
import platform
import re
import string
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from threading import Thread
from typing import AsyncIterator, Optional

import yaml
from huggingface_hub import AsyncInferenceClient, ChatCompletionStreamOutput

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
)
from transformers.commands import BaseTransformersCLICommand
from transformers.commands.serving import ServeArguments, ServeCommand
from transformers.utils import is_rich_available, is_torch_available


if platform.system() != "Windows":
    import pwd

if is_rich_available():
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown

if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
    )

ALLOWED_KEY_CHARS = set(string.ascii_letters + string.whitespace)
ALLOWED_VALUE_CHARS = set(
    string.ascii_letters + string.digits + string.whitespace + r".!\"#$%&'()*+,\-/:<=>?@[]^_`{|}~"
)

DEFAULT_EXAMPLES = {
    "llama": {"text": "There is a Llama in my lawn, how can I get rid of it?"},
    "code": {
        "text": (
            "Write a Python function that integrates any Python function f(x) numerically over an arbitrary "
            "interval [x_start, x_end]."
        ),
    },
    "helicopter": {"text": "How many helicopters can a human eat in one sitting?"},
    "numbers": {"text": "Count to 10 but skip every number ending with an 'e'"},
    "birds": {"text": "Why aren't birds real?"},
    "socks": {"text": "Why is it important to eat socks after meditating?"},
    "numbers2": {"text": "Which number is larger, 9.9 or 9.11?"},
}

# Printed at the start of a chat session
HELP_STRING_MINIMAL = """

**TRANSFORMERS CHAT INTERFACE**

Chat interface to try out a model. Besides chatting with the model, here are some basic commands:
- **!help**: shows all available commands (set generation settings, save chat, etc.)
- **!status**: shows the current status of the model and generation settings
- **!clear**: clears the current conversation and starts a new one
- **!exit**: closes the interface
"""


# Printed when the user types `help` in the chat session
HELP_STRING = f"""

**TRANSFORMERS CHAT INTERFACE HELP**

Full command list:
- **!help**: shows this help message
- **!clear**: clears the current conversation and starts a new one
- **!status**: shows the current status of the model and generation settings
- **!example {{NAME}}**: loads example named `{{NAME}}` from the config and uses it as the user input.
Available example names: `{"`, `".join(DEFAULT_EXAMPLES.keys())}`
- **!set {{ARG_1}}={{VALUE_1}} {{ARG_2}}={{VALUE_2}}** ...: changes the system prompt or generation settings (multiple
settings are separated by a space). Accepts the same flags and format as the `generate_flags` CLI argument.
If you're a new user, check this basic flag guide: https://huggingface.co/docs/transformers/llm_tutorial#common-options
- **!save {{SAVE_NAME}} (optional)**: saves the current chat and settings to file by default to
`./chat_history/{{MODEL_NAME}}/chat_{{DATETIME}}.yaml` or `{{SAVE_NAME}}` if provided
- **!exit**: closes the interface
"""


class RichInterface:
    def __init__(self, model_name: Optional[str] = None, user_name: Optional[str] = None):
        self._console = Console()
        if model_name is None:
            self.model_name = "assistant"
        else:
            self.model_name = model_name
        if user_name is None:
            self.user_name = "user"
        else:
            self.user_name = user_name

    async def stream_output(self, stream: AsyncIterator[ChatCompletionStreamOutput]) -> tuple[str, int]:
        self._console.print(f"[bold blue]<{self.model_name}>:")
        with Live(console=self._console, refresh_per_second=4) as live:
            text = ""
            async for token in await stream:
                outputs = token.choices[0].delta.content
                request_id = token.id

                if not outputs:
                    continue

                # Escapes single words encased in <>, e.g. <think> -> \<think\>, for proper rendering in Markdown.
                # It only escapes single words that may have `_`, optionally following a `/` (e.g. </think>)
                outputs = re.sub(r"<(/*)(\w*)>", r"\<\1\2\>", outputs)

                text += outputs
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.

                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")

                markdown = Markdown("".join(lines).strip(), code_theme="github-dark")

                # Update the Live console output
                live.update(markdown, refresh=True)

        self._console.print()

        return text, request_id

    def input(self) -> str:
        """Gets user input from the console."""
        input = self._console.input(f"[bold red]<{self.user_name}>:\n")
        self._console.print()
        return input

    def clear(self):
        """Clears the console."""
        self._console.clear()

    def print_user_message(self, text: str):
        """Prints a user message to the console."""
        self._console.print(f"[bold red]<{self.user_name}>:[/ bold red]\n{text}")
        self._console.print()

    def print_color(self, text: str, color: str):
        """Prints text in a given color to the console."""
        self._console.print(f"[bold {color}]{text}")
        self._console.print()

    def print_help(self, minimal: bool = False):
        """Prints the help message to the console."""
        self._console.print(Markdown(HELP_STRING_MINIMAL if minimal else HELP_STRING))
        self._console.print()

    def print_status(self, model_name: str, generation_config: GenerationConfig, model_kwargs: dict):
        """Prints the status of the model and generation settings to the console."""
        self._console.print(f"[bold blue]Model: {model_name}\n")
        if model_kwargs:
            self._console.print(f"[bold blue]Model kwargs: {model_kwargs}")
        self._console.print(f"[bold blue]{generation_config}")
        self._console.print()


@dataclass
class ChatArguments:
    r"""
    Arguments for the chat CLI.

    See the metadata arg for each argument's description -- the medatata will be printed with
    `transformers chat --help`
    """

    # General settings
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the pre-trained model. The positional argument will take precedence if both are passed."
        },
    )
    user: Optional[str] = field(
        default=None,
        metadata={"help": "Username to display in chat interface. Defaults to the current user's name."},
    )
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt."})
    save_folder: str = field(default="./chat_history/", metadata={"help": "Folder to save chat history."})
    examples_path: Optional[str] = field(default=None, metadata={"help": "Path to a yaml file with examples."})
    verbose: bool = field(default=False, metadata={"help": "Whether to show runtime warnings in the chat interface."})

    # Generation settings
    generation_config: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a local generation config file or to a HuggingFace repo containing a "
                "`generation_config.json` file. Other generation settings passed as CLI arguments will be applied on "
                "top of this generation config."
            ),
        },
    )

    # Model loading
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use (can be a branch name, tag name or commit id)."},
    )
    device: str = field(default="cpu", metadata={"help": "Device to use for inference."})
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "`torch_dtype` is deprecated! Please use `dtype` argument instead.
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
    host: str = field(default="localhost", metadata={"help": "Interface the server will listen to.."})
    port: int = field(default=8000, metadata={"help": "Port the server will listen to."})

    def __post_init__(self):
        """Only used for BC `torch_dtype` argument."""
        # In this case only the BC torch_dtype was given
        if self.torch_dtype is not None and self.dtype == "auto":
            self.dtype = self.torch_dtype


def chat_command_factory(args: Namespace):
    """
    Factory function used to chat with a local model.
    """
    return ChatCommand(args)


class ChatCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        dataclass_types = (ChatArguments,)
        chat_parser = parser.add_parser("chat", dataclass_types=dataclass_types)

        group = chat_parser.add_argument_group("Positional arguments")
        group.add_argument(
            "model_name_or_path_or_address",
            type=str,
            default=None,
            help="Name of the pre-trained model or address to connect to.",
        )
        group.add_argument(
            "generate_flags",
            type=str,
            default=None,
            help=(
                "Flags to pass to `generate`, using a space as a separator between flags. Accepts booleans, numbers, "
                "and lists of integers, more advanced parameterization should be set through --generation-config. "
                "Example: `transformers chat <model_repo> max_new_tokens=100 do_sample=False eos_token_id=[1,2]`. "
                "If you're a new user, check this basic flag guide: "
                "https://huggingface.co/docs/transformers/llm_tutorial#common-options"
            ),
            nargs="*",
        )
        chat_parser.set_defaults(func=chat_command_factory)

    def __init__(self, args):
        if args.model_name_or_path_or_address is not None:
            name = args.model_name_or_path_or_address
            if name.startswith("http") or name.startswith("https") or name.startswith("localhost"):
                self.spawn_backend = False

                if args.host != "localhost" or args.port != 8000:
                    raise ValueError(
                        "Looks like you’ve set both a server address and a custom host/port. "
                        "Please pick just one way to specify the server."
                    )

                args.host, args.port = args.model_name_or_path_or_address.rsplit(":", 1)

                if args.model_name_or_path is None:
                    raise ValueError(
                        "When connecting to a server, please specify a model name with the --model_name_or_path flag."
                    )
            else:
                self.spawn_backend = True
                args.model_name_or_path = args.model_name_or_path_or_address

        if not is_rich_available() and (not is_torch_available() and self.spawn_backend):
            raise ImportError(
                "You need to install rich to use the chat interface. Additionally, you have not specified a remote "
                "endpoint and are therefore spawning a backend. Torch is required for this: (`pip install rich torch`)"
            )
        elif not is_rich_available():
            raise ImportError("You need to install rich to use the chat interface. (`pip install rich`)")
        elif not is_torch_available() and self.spawn_backend:
            raise ImportError(
                "You have not specified a remote endpoint and are therefore spawning a backend. Torch is required "
                "for this: (`pip install rich torch`)"
            )

        self.args = args

    # -----------------------------------------------------------------------------------------------------------------
    # Chat session methods
    @staticmethod
    def get_username() -> str:
        """Returns the username of the current user."""
        if platform.system() == "Windows":
            return os.getlogin()
        else:
            return pwd.getpwuid(os.getuid()).pw_name

    @staticmethod
    def save_chat(chat, args: ChatArguments, filename: Optional[str] = None) -> str:
        """Saves the chat history to a file."""
        output_dict = {}
        output_dict["settings"] = vars(args)
        output_dict["chat_history"] = chat

        folder = args.save_folder

        if filename is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{args.model_name_or_path_or_address}/chat_{time_str}.json"
            filename = os.path.join(folder, filename)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(output_dict, f, indent=4)
        return os.path.abspath(filename)

    @staticmethod
    def clear_chat_history(system_prompt: Optional[str] = None) -> list[dict]:
        """Clears the chat history."""
        if system_prompt is None:
            chat = []
        else:
            chat = [{"role": "system", "content": system_prompt}]
        return chat

    # -----------------------------------------------------------------------------------------------------------------
    # Input parsing methods
    def parse_generate_flags(self, generate_flags: list[str]) -> dict:
        """Parses the generate flags from the user input into a dictionary of `generate` kwargs."""
        if len(generate_flags) == 0:
            return {}

        # Assumption: `generate_flags` is a list of strings, each string being a `flag=value` pair, that can be parsed
        # into a json string if we:
        # 1. Add quotes around each flag name
        generate_flags_as_dict = {'"' + flag.split("=")[0] + '"': flag.split("=")[1] for flag in generate_flags}

        # 2. Handle types:
        # 2. a. booleans should be lowercase, None should be null
        generate_flags_as_dict = {
            k: v.lower() if v.lower() in ["true", "false"] else v for k, v in generate_flags_as_dict.items()
        }
        generate_flags_as_dict = {k: "null" if v == "None" else v for k, v in generate_flags_as_dict.items()}

        # 2. b. strings should be quoted
        def is_number(s: str) -> bool:
            # handle negative numbers
            if s.startswith("-"):
                s = s[1:]
            return s.replace(".", "", 1).isdigit()

        generate_flags_as_dict = {k: f'"{v}"' if not is_number(v) else v for k, v in generate_flags_as_dict.items()}
        # 2. c. [no processing needed] lists are lists of ints because `generate` doesn't take lists of strings :)
        # We also mention in the help message that we only accept lists of ints for now.

        # 3. Join the result into a comma separated string
        generate_flags_string = ", ".join([f"{k}: {v}" for k, v in generate_flags_as_dict.items()])

        # 4. Add the opening/closing brackets
        generate_flags_string = "{" + generate_flags_string + "}"

        # 5. Remove quotes around boolean/null and around lists
        generate_flags_string = generate_flags_string.replace('"null"', "null")
        generate_flags_string = generate_flags_string.replace('"true"', "true")
        generate_flags_string = generate_flags_string.replace('"false"', "false")
        generate_flags_string = generate_flags_string.replace('"[', "[")
        generate_flags_string = generate_flags_string.replace(']"', "]")

        # 6. Replace the `=` with `:`
        generate_flags_string = generate_flags_string.replace("=", ":")

        try:
            processed_generate_flags = json.loads(generate_flags_string)
        except json.JSONDecodeError:
            raise ValueError(
                "Failed to convert `generate_flags` into a valid JSON object."
                "\n`generate_flags` = {generate_flags}"
                "\nConverted JSON string = {generate_flags_string}"
            )
        return processed_generate_flags

    def get_generation_parameterization(
        self, args: ChatArguments, model_generation_config: GenerationConfig
    ) -> tuple[GenerationConfig, dict]:
        """
        Returns a GenerationConfig object holding the generation parameters for the CLI command.
        """
        # No generation config arg provided -> use model's default generation config, then apply CLI defaults
        if args.generation_config is not None:
            if ".json" in args.generation_config:  # is a local file
                dirname = os.path.dirname(args.generation_config)
                filename = os.path.basename(args.generation_config)
                generation_config = GenerationConfig.from_pretrained(dirname, filename)
            else:
                generation_config = GenerationConfig.from_pretrained(args.generation_config)
        else:
            # !!!!!!!!!
            # This is a chat session, so we have a few non-standard defaults
            # !!!!!!!!!
            generation_config = copy.deepcopy(model_generation_config)
            generation_config.update(**{"do_sample": True, "max_new_tokens": 256})

        # Finally: parse and apply `generate_flags`
        parsed_generate_flags = self.parse_generate_flags(args.generate_flags)
        model_kwargs = generation_config.update(**parsed_generate_flags)
        # `model_kwargs` contain non-generation flags in `parsed_generate_flags` that should be passed directly to
        # `generate`
        return generation_config, model_kwargs

    @staticmethod
    def parse_eos_tokens(
        tokenizer: PreTrainedTokenizer,
        generation_config: GenerationConfig,
        eos_tokens: Optional[str],
        eos_token_ids: Optional[str],
    ) -> tuple[int, list[int]]:
        """Retrieves the pad token ID and all possible EOS token IDs."""
        if generation_config.pad_token_id is None:
            pad_token_id = generation_config.eos_token_id
        else:
            pad_token_id = generation_config.pad_token_id

        all_eos_token_ids = []

        if eos_tokens is not None:
            all_eos_token_ids.extend(tokenizer.convert_tokens_to_ids(eos_tokens.split(",")))

        if eos_token_ids is not None:
            all_eos_token_ids.extend([int(token_id) for token_id in eos_token_ids.split(",")])

        if len(all_eos_token_ids) == 0:
            all_eos_token_ids.append(generation_config.eos_token_id)

        return pad_token_id, all_eos_token_ids

    # -----------------------------------------------------------------------------------------------------------------
    # Model loading and performance automation methods
    @staticmethod
    def get_quantization_config(model_args: ChatArguments) -> Optional["BitsAndBytesConfig"]:
        if model_args.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                # For consistency with model weights, we use the same value as `dtype`
                bnb_4bit_compute_dtype=model_args.dtype,
                bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
                bnb_4bit_quant_storage=model_args.dtype,
            )
        elif model_args.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        return quantization_config

    def load_model_and_tokenizer(self, args: ChatArguments) -> tuple["AutoModelForCausalLM", AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path_positional,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )

        dtype = args.dtype if args.dtype in ["auto", None] else getattr(torch, args.dtype)
        quantization_config = self.get_quantization_config(args)
        model_kwargs = {
            "revision": args.model_revision,
            "attn_implementation": args.attn_implementation,
            "dtype": dtype,
            "device_map": "auto",
            "quantization_config": quantization_config,
        }
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path_positional, trust_remote_code=args.trust_remote_code, **model_kwargs
        )

        if getattr(model, "hf_device_map", None) is None:
            model = model.to(args.device)

        return model, tokenizer

    # -----------------------------------------------------------------------------------------------------------------
    # User commands
    def handle_non_exit_user_commands(
        self,
        user_input: str,
        args: ChatArguments,
        interface: RichInterface,
        examples: dict[str, dict[str, str]],
        generation_config: GenerationConfig,
        model_kwargs: dict,
        chat: list[dict],
    ) -> tuple[list[dict], GenerationConfig, dict]:
        """
        Handles all user commands except for `!exit`. May update the chat history (e.g. reset it) or the
        generation config (e.g. set a new flag).
        """
        valid_command = True

        if user_input == "!clear":
            chat = self.clear_chat_history(args.system_prompt)
            interface.clear()

        elif user_input == "!help":
            interface.print_help()

        elif user_input.startswith("!save") and len(user_input.split()) < 2:
            split_input = user_input.split()

            if len(split_input) == 2:
                filename = split_input[1]
            else:
                filename = None
            filename = self.save_chat(chat, args, filename)
            interface.print_color(text=f"Chat saved in {filename}!", color="green")

        elif user_input.startswith("!set"):
            # splits the new args into a list of strings, each string being a `flag=value` pair (same format as
            # `generate_flags`)
            new_generate_flags = user_input[4:].strip()
            new_generate_flags = new_generate_flags.split()
            # sanity check: each member in the list must have an =
            for flag in new_generate_flags:
                if "=" not in flag:
                    interface.print_color(
                        text=(
                            f"Invalid flag format, missing `=` after `{flag}`. Please use the format "
                            "`arg_1=value_1 arg_2=value_2 ...`."
                        ),
                        color="red",
                    )
                    break
            else:
                # parses the new args into a dictionary of `generate` kwargs, and updates the corresponding variables
                parsed_new_generate_flags = self.parse_generate_flags(new_generate_flags)
                new_model_kwargs = generation_config.update(**parsed_new_generate_flags)
                model_kwargs.update(**new_model_kwargs)

        elif user_input.startswith("!example") and len(user_input.split()) == 2:
            example_name = user_input.split()[1]
            if example_name in examples:
                interface.clear()
                chat = []
                interface.print_user_message(examples[example_name]["text"])
                chat.append({"role": "user", "content": examples[example_name]["text"]})
            else:
                example_error = (
                    f"Example {example_name} not found in list of available examples: {list(examples.keys())}."
                )
                interface.print_color(text=example_error, color="red")

        elif user_input == "!status":
            interface.print_status(
                model_name=args.model_name_or_path,
                generation_config=generation_config,
                model_kwargs=model_kwargs,
            )

        else:
            valid_command = False
            interface.print_color(text=f"'{user_input}' is not a valid command. Showing help message.", color="red")
            interface.print_help()

        return chat, valid_command, generation_config, model_kwargs

    # -----------------------------------------------------------------------------------------------------------------
    # Main logic
    def run(self):
        asyncio.run(self._inner_run())

    async def _inner_run(self):
        if self.spawn_backend:
            serve_args = ServeArguments(
                device=self.args.device,
                dtype=self.args.dtype,
                trust_remote_code=self.args.trust_remote_code,
                attn_implementation=self.args.attn_implementation,
                load_in_8bit=self.args.load_in_8bit,
                load_in_4bit=self.args.load_in_4bit,
                bnb_4bit_quant_type=self.args.bnb_4bit_quant_type,
                use_bnb_nested_quant=self.args.use_bnb_nested_quant,
                host=self.args.host,
                port=self.args.port,
                log_level="error",
            )
            serve_command = ServeCommand(serve_args)

            thread = Thread(target=serve_command.run)
            thread.daemon = True
            thread.start()

        model = self.args.model_name_or_path + "@" + self.args.model_revision
        host = "http://localhost" if self.args.host == "localhost" else self.args.host
        client = AsyncInferenceClient(f"{host}:{self.args.port}")

        args = self.args
        if args.examples_path is None:
            examples = DEFAULT_EXAMPLES
        else:
            with open(args.examples_path) as f:
                examples = yaml.safe_load(f)

        if args.user is None:
            user = self.get_username()
        else:
            user = args.user

        model_generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
        generation_config, model_kwargs = self.get_generation_parameterization(args, model_generation_config)

        interface = RichInterface(model_name=args.model_name_or_path, user_name=user)
        interface.clear()
        chat = self.clear_chat_history(args.system_prompt)

        request_id = None

        # Starts the session with a minimal help message at the top, so that a user doesn't get stuck
        interface.print_help(minimal=True)
        while True:
            try:
                user_input = interface.input()

                # User commands
                if user_input.startswith("!"):
                    # `!exit` is special, it breaks the loop
                    if user_input == "!exit":
                        break
                    else:
                        chat, valid_command, generation_config, model_kwargs = self.handle_non_exit_user_commands(
                            user_input=user_input,
                            args=args,
                            interface=interface,
                            examples=examples,
                            generation_config=generation_config,
                            model_kwargs=model_kwargs,
                            chat=chat,
                        )
                    # `!example` sends a user message to the model
                    if not valid_command or not user_input.startswith("!example"):
                        continue
                else:
                    chat.append({"role": "user", "content": user_input})

                stream = client.chat_completion(
                    chat,
                    stream=True,
                    extra_body={
                        "request_id": request_id,
                        "generation_config": generation_config.to_json_string(),
                        "model": model,
                    },
                )

                model_output, request_id = await interface.stream_output(stream)

                chat.append({"role": "assistant", "content": model_output})

            except KeyboardInterrupt:
                break
            finally:
                await client.close()


if __name__ == "__main__":
    args = ChatArguments()
    args.model_name_or_path_or_address = "meta-llama/Llama-3.2-3b-Instruct"
    args.model_name_or_path_or_address = "http://localhost:8000"
    chat = ChatCommand(args)
    chat.run()
