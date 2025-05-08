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
import json
import os
import platform
import string
import time
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from threading import Thread
from typing import Optional

import yaml

from transformers.utils import is_rich_available, is_torch_available

from . import BaseTransformersCLICommand


if platform.system() != "Windows":
    import pwd

if is_rich_available():
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown

if is_torch_available():
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer


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
}

SUPPORTED_GENERATION_KWARGS = [
    "max_new_tokens",
    "do_sample",
    "num_beams",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
]

# Printed at the start of a chat session
HELP_STRING_MINIMAL = """

**TRANSFORMERS CHAT INTERFACE**

Chat interface to try out a model. Besides chatting with the model, here are some basic commands:
- **help**: shows all available commands
- **clear**: clears the current conversation and starts a new one
- **exit**: closes the interface
"""


# Printed when the user types `help` in the chat session
HELP_STRING = f"""

**TRANSFORMERS CHAT INTERFACE HELP**

Full command list:
- **help**: shows this help message
- **clear**: clears the current conversation and starts a new one
- **example {{NAME}}**: loads example named `{{NAME}}` from the config and uses it as the user input. Available example
names: `{"`, `".join(DEFAULT_EXAMPLES.keys())}`
- **set {{SETTING_NAME}}={{SETTING_VALUE}};**: changes the system prompt or generation settings (multiple settings are
separated by a ';'). Available settings: `{"`, `".join(SUPPORTED_GENERATION_KWARGS)}`
- **reset**: same as clear but also resets the generation configs to defaults if they have been changed by **set**
- **save {{SAVE_NAME}} (optional)**: saves the current chat and settings to file by default to
`./chat_history/{{MODEL_NAME}}/chat_{{DATETIME}}.yaml` or `{{SAVE_NAME}}` if provided
- **exit**: closes the interface
"""

# format: (optional CLI arg being deprecated, its current default, corresponding `generate` flag)
_DEPRECATION_MAP = [
    ("max_new_tokens", 256, "max_new_tokens"),
    ("do_sample", True, "do_sample"),
    ("num_beams", 1, "num_beams"),
    ("temperature", 1.0, "temperature"),
    ("top_k", 50, "top_k"),
    ("top_p", 1.0, "top_p"),
    ("repetition_penalty", 1.0, "repetition_penalty"),
    ("eos_tokens", None, "eos_token_id"),
    ("eos_token_ids", None, "eos_token_id"),
]


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

    def stream_output(self, output_stream: TextIteratorStreamer) -> str:
        """Stream output from a role, and return the generated text after it's done steaming."""
        # This method is originally from the FastChat CLI:
        # https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
        # Create a Live context for updating the console output
        text = ""
        self._console.print(f"[bold blue]<{self.model_name}>:")
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for i, outputs in enumerate(output_stream):
                if not outputs or i == 0:
                    continue
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
                live.update(markdown)
        self._console.print()
        return text

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
    max_new_tokens: int = field(default=256, metadata={"help": "Maximum number of tokens to generate."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample outputs during generation."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for beam search."})
    temperature: float = field(default=1.0, metadata={"help": "Temperature parameter for generation."})
    top_k: int = field(default=50, metadata={"help": "Value of k for top-k sampling."})
    top_p: float = field(default=1.0, metadata={"help": "Value of p for nucleus sampling."})
    repetition_penalty: float = field(default=1.0, metadata={"help": "Repetition penalty."})
    eos_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "EOS tokens to stop the generation. If multiple they should be comma separated."},
    )
    eos_token_ids: Optional[str] = field(
        default=None,
        metadata={"help": "EOS token IDs to stop the generation. If multiple they should be comma separated."},
    )

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
            "model_name_or_path_positional", type=str, default=None, help="Name of the pre-trained model."
        )
        group.add_argument(
            "generate_flags",
            type=str,
            default=None,
            help=(
                "Flags to pass to `generate`. See the docs for your selected model, `GenerationConfig` and `generate` "
                "for more details. Usage: `arg_1=value_1 arg_2=value_2 ...`. Example: "
                "`max_new_tokens=100 do_sample=False eos_token_id=[1,2] foo=bar`."
            ),
            nargs="*",
        )
        chat_parser.set_defaults(func=chat_command_factory)

    def __init__(self, args):
        args = self._handle_deprecated_args(args)
        self.args = args

    def _handle_deprecated_args(self, args: ChatArguments) -> ChatArguments:
        """
        Handles deprecated arguments and their deprecation cycle. To be removed after we fully migrated to the new
        args.
        """
        has_warnings = False

        # 1. Model as a positional argument
        args.model_name_or_path_positional = args.model_name_or_path_positional or args.model_name_or_path
        if args.model_name_or_path_positional is None:
            raise ValueError(
                "One of the following must be provided:"
                "\n- The positional argument containing the model repo, e.g. `transformers chat <model_repo>`"
                "\n- the optional --model_name_or_path argument, containing the model repo (deprecated)"
            )
        elif args.model_name_or_path is not None:
            has_warnings = True
            warnings.warn(
                "The --model_name_or_path argument is deprecated will be removed in v4.54.0. Use the positional "
                "argument instead, e.g. `transformers chat <model_repo>`.",
                FutureWarning,
            )
        # 2. Named generate option args
        for deprecated_arg, default_value, new_arg in _DEPRECATION_MAP:
            value = getattr(args, deprecated_arg)
            if value != default_value:
                has_warnings = True
                warnings.warn(
                    f"The --{deprecated_arg} argument is deprecated will be removed in v4.54.0. There are two "
                    "alternative solutions to specify this generation setting: \n"
                    "1. Pass `--generation-config <path_to_file/repo>` to specify a generation config.\n"
                    "2. Pass `generate` flags through positional arguments, e.g. `transformers chat <model_repo> "
                    f"{new_arg}={value}`",
                )

        if has_warnings:
            print("\n(Press any key to continue)")
            input()
        return args

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
            filename = f"{args.model_name_or_path_positional}/chat_{time_str}.json"
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

        # Assumption: `generate_flags` is a list of strings that can be parsed into a json string if we:
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
            return s.replace(".", "", 1).isdigit()

        generate_flags_as_dict = {k: f'"{v}"' if not is_number(v) else v for k, v in generate_flags_as_dict.items()}
        # 2. c. [no processing needed] lists are lists of ints because `generate` doesn't take lists of strings :)

        # 3. Join the the result into a comma separated string
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
            processed_flags = json.loads(generate_flags_string)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid generate flags: {generate_flags_string}")
        return processed_flags

    @staticmethod
    def parse_settings(
        user_input: str, current_args: ChatArguments, interface: RichInterface
    ) -> tuple[ChatArguments, bool]:
        """Parses the settings from the user input into the CLI arguments."""
        settings = user_input[4:].strip().split(";")
        settings = [(setting.split("=")[0], setting[len(setting.split("=")[0]) + 1 :]) for setting in settings]
        settings = dict(settings)
        error = False

        for name in settings:
            if hasattr(current_args, name):
                try:
                    if isinstance(getattr(current_args, name), bool):
                        if settings[name] == "True":
                            settings[name] = True
                        elif settings[name] == "False":
                            settings[name] = False
                        else:
                            raise ValueError
                    else:
                        settings[name] = type(getattr(current_args, name))(settings[name])
                except ValueError:
                    error = True
                    interface.print_color(
                        text=f"Cannot cast setting {name} (={settings[name]}) to {type(getattr(current_args, name))}.",
                        color="red",
                    )
            else:
                interface.print_color(text=f"There is no '{name}' setting.", color="red")

        if error:
            interface.print_color(
                text="There was an issue parsing the settings. No settings have been changed.",
                color="red",
            )
        else:
            for name in settings:
                setattr(current_args, name, settings[name])
                interface.print_color(text=f"Set {name} to {settings[name]}.", color="green")

            time.sleep(1.5)  # so the user has time to read the changes

        return current_args, not error

    @staticmethod
    def parse_eos_tokens(
        tokenizer: AutoTokenizer, eos_tokens: Optional[str], eos_token_ids: Optional[str]
    ) -> tuple[int, list[int]]:
        """Retrieves the pad token ID and all possible EOS token IDs."""
        if tokenizer.pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = tokenizer.pad_token_id

        all_eos_token_ids = []

        if eos_tokens is not None:
            all_eos_token_ids.extend(tokenizer.convert_tokens_to_ids(eos_tokens.split(",")))

        if eos_token_ids is not None:
            all_eos_token_ids.extend([int(token_id) for token_id in eos_token_ids.split(",")])

        if len(all_eos_token_ids) == 0:
            all_eos_token_ids.append(tokenizer.eos_token_id)

        return pad_token_id, all_eos_token_ids

    @staticmethod
    def is_valid_setting_command(s: str) -> bool:
        # First check the basic structure
        if not s.startswith("set ") or "=" not in s:
            return False

        # Split into individual assignments
        assignments = [a.strip() for a in s[4:].split(";") if a.strip()]

        for assignment in assignments:
            # Each assignment should have exactly one '='
            if assignment.count("=") != 1:
                return False

            key, value = assignment.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                return False

            # Keys can only have alphabetic characters, spaces and underscores
            if not set(key).issubset(ALLOWED_KEY_CHARS):
                return False

            # Values can have just about anything that isn't a semicolon
            if not set(value).issubset(ALLOWED_VALUE_CHARS):
                return False

        return True

    # -----------------------------------------------------------------------------------------------------------------
    # Model loading and performance automation methods
    @staticmethod
    def get_quantization_config(model_args: ChatArguments) -> Optional["BitsAndBytesConfig"]:
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

    def load_model_and_tokenizer(self, args: ChatArguments) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path_positional,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )

        torch_dtype = args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
        quantization_config = self.get_quantization_config(args)
        model_kwargs = {
            "revision": args.model_revision,
            "attn_implementation": args.attn_implementation,
            "torch_dtype": torch_dtype,
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
    # Main logic
    def run(self):
        if not is_rich_available():
            raise ImportError("You need to install rich to use the chat interface. (`pip install rich`)")
        if not is_torch_available():
            raise ImportError("You need to install torch to use the chat interface. (`pip install torch`)")

        args = self.args
        args.generate_flags = self.parse_generate_flags(args.generate_flags)
        if args.examples_path is None:
            examples = DEFAULT_EXAMPLES
        else:
            with open(args.examples_path) as f:
                examples = yaml.safe_load(f)

        current_args = copy.deepcopy(args)

        if args.user is None:
            user = self.get_username()
        else:
            user = args.user

        model, tokenizer = self.load_model_and_tokenizer(args)
        generation_streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

        pad_token_id, eos_token_ids = self.parse_eos_tokens(tokenizer, args.eos_tokens, args.eos_token_ids)

        interface = RichInterface(model_name=args.model_name_or_path_positional, user_name=user)
        interface.clear()
        chat = self.clear_chat_history(current_args.system_prompt)

        # Starts the session with a minimal help message at the top, so that a user doesn't get stuck
        interface.print_help(minimal=True)
        while True:
            try:
                user_input = interface.input()

                if user_input == "clear":
                    chat = self.clear_chat_history(current_args.system_prompt)
                    interface.clear()
                    continue

                if user_input == "help":
                    interface.print_help()
                    continue

                if user_input == "exit":
                    break

                if user_input == "reset":
                    interface.clear()
                    current_args = copy.deepcopy(args)
                    chat = self.clear_chat_history(current_args.system_prompt)
                    continue

                if user_input.startswith("save") and len(user_input.split()) < 2:
                    split_input = user_input.split()

                    if len(split_input) == 2:
                        filename = split_input[1]
                    else:
                        filename = None
                    filename = self.save_chat(chat, current_args, filename)
                    interface.print_color(text=f"Chat saved in {filename}!", color="green")
                    continue

                if self.is_valid_setting_command(user_input):
                    current_args, success = self.parse_settings(user_input, current_args, interface)
                    if success:
                        chat = []
                        interface.clear()
                        continue

                if user_input.startswith("example") and len(user_input.split()) == 2:
                    example_name = user_input.split()[1]
                    if example_name in examples:
                        interface.clear()
                        chat = []
                        interface.print_user_message(examples[example_name]["text"])
                        user_input = examples[example_name]["text"]
                    else:
                        example_error = (
                            f"Example {example_name} not found in list of available examples: {list(examples.keys())}."
                        )
                        interface.print_color(text=example_error, color="red")
                        continue

                chat.append({"role": "user", "content": user_input})

                inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(
                    model.device
                )
                attention_mask = torch.ones_like(inputs)
                generation_kwargs = {
                    "inputs": inputs,
                    "attention_mask": attention_mask,
                    "streamer": generation_streamer,
                    "max_new_tokens": current_args.max_new_tokens,
                    "do_sample": current_args.do_sample,
                    "num_beams": current_args.num_beams,
                    "temperature": current_args.temperature,
                    "top_k": current_args.top_k,
                    "top_p": current_args.top_p,
                    "repetition_penalty": current_args.repetition_penalty,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_ids,
                }

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                model_output = interface.stream_output(generation_streamer)
                thread.join()
                chat.append({"role": "assistant", "content": model_output})

            except KeyboardInterrupt:
                break
