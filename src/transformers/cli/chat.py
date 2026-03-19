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
import json
import os
import platform
import re
import string
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any
from urllib.parse import urljoin, urlparse

import httpx
import typer
import yaml
from huggingface_hub import AsyncInferenceClient, ChatCompletionStreamOutput

from transformers import GenerationConfig
from transformers.utils import is_rich_available


try:
    import readline  # noqa importing this enables GNU readline capabilities
except ImportError:
    # some platforms may not support readline: https://docs.python.org/3/library/readline.html
    pass

if platform.system() != "Windows":
    import pwd

if is_rich_available():
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown

DEFAULT_HTTP_ENDPOINT = {"hostname": "localhost", "port": 8000}
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
`./chat_history/{{MODEL_ID}}/chat_{{DATETIME}}.yaml` or `{{SAVE_NAME}}` if provided
- **!exit**: closes the interface
"""


class RichInterface:
    def __init__(self, model_id: str, user_id: str):
        self._console = Console()
        self.model_id = model_id
        self.user_id = user_id

    async def stream_output(self, stream: AsyncIterator[ChatCompletionStreamOutput]) -> tuple[str, str | Any | None]:
        self._console.print(f"[bold blue]<{self.model_id}>:")
        with Live(console=self._console, refresh_per_second=4) as live:
            text = ""
            finish_reason: str | None = None
            async for token in await stream:
                outputs = token.choices[0].delta.content
                finish_reason = getattr(token.choices[0], "finish_reason", finish_reason)

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

        return text, finish_reason

    def input(self) -> str:
        """Gets user input from the console."""
        input = self._console.input(f"[bold red]<{self.user_id}>:\n")
        self._console.print()
        return input

    def clear(self):
        """Clears the console."""
        self._console.clear()

    def print_user_message(self, text: str):
        """Prints a user message to the console."""
        self._console.print(f"[bold red]<{self.user_id}>:[/ bold red]\n{text}")
        self._console.print()

    def print_color(self, text: str, color: str):
        """Prints text in a given color to the console."""
        self._console.print(f"[bold {color}]{text}")
        self._console.print()

    def confirm(self, message: str, default: bool = False) -> bool:
        """Displays a yes/no prompt to the user, returning True for confirmation."""
        default_hint = "Y/n" if default else "y/N"
        response = self._console.input(f"[bold yellow]{message} ({default_hint}): ")
        self._console.print()

        response = response.strip().lower()
        if not response:
            return default

        return response in {"y", "yes"}

    def print_help(self, minimal: bool = False):
        """Prints the help message to the console."""
        self._console.print(Markdown(HELP_STRING_MINIMAL if minimal else HELP_STRING))
        self._console.print()

    def print_status(self, config: GenerationConfig):
        """Prints the status of the model and generation settings to the console."""
        self._console.print(f"[bold blue]Model: {self.model_id}\n")
        self._console.print(f"[bold blue]{config}")
        self._console.print()


class Chat:
    """Chat with a model from the command line."""

    # Defining a class to help with internal state but in practice it's just a method to call
    # TODO: refactor into a proper module with helpers + 1 main method
    def __init__(
        self,
        model_id: Annotated[str, typer.Argument(help="ID of the model to use (e.g. 'HuggingFaceTB/SmolLM3-3B').")],
        base_url: Annotated[
            str | None, typer.Argument(help="Base url to connect to (e.g. http://localhost:8000/v1).")
        ] = f"http://{DEFAULT_HTTP_ENDPOINT['hostname']}:{DEFAULT_HTTP_ENDPOINT['port']}",
        generate_flags: Annotated[
            list[str] | None,
            typer.Argument(
                help=(
                    "Flags to pass to `generate`, using a space as a separator between flags. Accepts booleans, numbers, "
                    "and lists of integers, more advanced parameterization should be set through --generation-config. "
                    "Example: `transformers chat <base_url> <model_id> max_new_tokens=100 do_sample=False eos_token_id=[1,2]`. "
                    "If you're a new user, check this basic flag guide: "
                    "https://huggingface.co/docs/transformers/llm_tutorial#common-options"
                )
            ),
        ] = None,
        # General settings
        user: Annotated[
            str | None,
            typer.Option(help="Username to display in chat interface. Defaults to the current user's name."),
        ] = None,
        system_prompt: Annotated[str | None, typer.Option(help="System prompt.")] = None,
        save_folder: Annotated[str, typer.Option(help="Folder to save chat history.")] = "./chat_history/",
        examples_path: Annotated[str | None, typer.Option(help="Path to a yaml file with examples.")] = None,
        # Generation settings
        generation_config: Annotated[
            str | None,
            typer.Option(
                help="Path to a local generation config file or to a HuggingFace repo containing a `generation_config.json` file. Other generation settings passed as CLI arguments will be applied on top of this generation config."
            ),
        ] = None,
    ) -> None:
        """Chat with a model from the command line."""
        self.base_url = base_url

        parsed = urlparse(self.base_url)
        if parsed.hostname == DEFAULT_HTTP_ENDPOINT["hostname"] and parsed.port == DEFAULT_HTTP_ENDPOINT["port"]:
            self.check_health(self.base_url)

        self.model_id = model_id
        self.system_prompt = system_prompt
        self.save_folder = save_folder

        # Generation settings
        config = load_generation_config(generation_config)
        config.update(do_sample=True, max_new_tokens=256)  # some default values
        config.update(**parse_generate_flags(generate_flags))
        self.config = config

        self.settings = {"base_url": base_url, "model_id": model_id, "config": self.config.to_dict()}

        # User settings
        self.user = user if user is not None else get_username()

        # Load examples
        if examples_path:
            with open(examples_path) as f:
                self.examples = yaml.safe_load(f)
        else:
            self.examples = DEFAULT_EXAMPLES

        # Check requirements
        if not is_rich_available():
            raise ImportError("You need to install rich to use the chat interface. (`pip install rich`)")

        # Run chat session
        asyncio.run(self._inner_run())

    @staticmethod
    def check_health(url):
        health_url = urljoin(url + "/", "health")
        try:
            output = httpx.get(health_url)
            if output.status_code != 200:
                raise ValueError(
                    f"The server running on {url} returned status code {output.status_code} on health check (/health)."
                )
        except httpx.ConnectError:
            raise ValueError(
                f"No server currently running on {url}. To run a local server, please run `transformers serve` in a"
                f"separate shell. Find more information here: https://huggingface.co/docs/transformers/serving"
            )

        return True

    def handle_non_exit_user_commands(
        self,
        user_input: str,
        interface: RichInterface,
        examples: dict[str, dict[str, str]],
        config: GenerationConfig,
        chat: list[dict],
    ) -> tuple[list[dict], GenerationConfig]:
        """
        Handles all user commands except for `!exit`. May update the chat history (e.g. reset it) or the
        generation config (e.g. set a new flag).
        """
        valid_command = True

        if user_input == "!clear":
            chat = new_chat_history(self.system_prompt)
            interface.clear()

        elif user_input == "!help":
            interface.print_help()

        elif user_input.startswith("!save") and len(user_input.split()) < 2:
            split_input = user_input.split()
            filename = (
                split_input[1]
                if len(split_input) == 2
                else os.path.join(self.save_folder, self.model_id, f"chat_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json")
            )
            save_chat(filename=filename, chat=chat, settings=self.settings)
            interface.print_color(text=f"Chat saved to {filename}!", color="green")

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
                # Update config from user flags
                config.update(**parse_generate_flags(new_generate_flags))

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
            interface.print_status(config=config)

        else:
            valid_command = False
            interface.print_color(text=f"'{user_input}' is not a valid command. Showing help message.", color="red")
            interface.print_help()

        return chat, valid_command, config

    async def _inner_run(self):
        interface = RichInterface(model_id=self.model_id, user_id=self.user)
        interface.clear()
        chat = new_chat_history(self.system_prompt)

        # Starts the session with a minimal help message at the top, so that a user doesn't get stuck
        interface.print_help(minimal=True)

        config = self.config

        async with AsyncInferenceClient(base_url=self.base_url) as client:
            pending_user_input: str | None = None
            while True:
                try:
                    if pending_user_input is not None:
                        user_input = pending_user_input
                        pending_user_input = None
                        interface.print_user_message(user_input)
                    else:
                        user_input = interface.input()

                    # User commands
                    if user_input == "!exit":
                        break

                    elif user_input == "!clear":
                        chat = new_chat_history(self.system_prompt)
                        interface.clear()
                        continue

                    elif user_input == "!help":
                        interface.print_help()
                        continue

                    elif user_input.startswith("!save") and len(user_input.split()) < 2:
                        split_input = user_input.split()
                        filename = (
                            split_input[1]
                            if len(split_input) == 2
                            else os.path.join(
                                self.save_folder, self.model_id, f"chat_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
                            )
                        )
                        save_chat(filename=filename, chat=chat, settings=self.settings)
                        interface.print_color(text=f"Chat saved to {filename}!", color="green")
                        continue

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
                            # Update config from user flags
                            config.update(**parse_generate_flags(new_generate_flags))
                        continue

                    elif user_input.startswith("!example") and len(user_input.split()) == 2:
                        example_name = user_input.split()[1]
                        if example_name in self.examples:
                            interface.clear()
                            chat = []
                            interface.print_user_message(self.examples[example_name]["text"])
                            chat.append({"role": "user", "content": self.examples[example_name]["text"]})
                        else:
                            example_error = f"Example {example_name} not found in list of available examples: {list(self.examples.keys())}."
                            interface.print_color(text=example_error, color="red")

                    elif user_input == "!status":
                        interface.print_status(config=config)
                        continue

                    elif user_input.startswith("!"):
                        interface.print_color(
                            text=f"'{user_input}' is not a valid command. Showing help message.", color="red"
                        )
                        interface.print_help()
                        continue

                    else:
                        chat.append({"role": "user", "content": user_input})

                    stream = client.chat_completion(
                        chat,
                        stream=True,
                        model=self.model_id,
                        extra_body={
                            "generation_config": config.to_json_string(),
                            "model": self.model_id,
                        },
                    )

                    model_output, finish_reason = await interface.stream_output(stream)

                    chat.append({"role": "assistant", "content": model_output})

                    if finish_reason == "length":
                        interface.print_color("Generation stopped after reaching the token limit.", "yellow")
                        if interface.confirm("Continue generating?"):
                            pending_user_input = "Please continue. Do not repeat text.”"
                            continue
                except KeyboardInterrupt:
                    break


def load_generation_config(generation_config: str | None) -> GenerationConfig:
    if generation_config is None:
        return GenerationConfig()

    if ".json" in generation_config:  # is a local file
        dirname = os.path.dirname(generation_config)
        filename = os.path.basename(generation_config)
        return GenerationConfig.from_pretrained(dirname, filename)
    else:
        return GenerationConfig.from_pretrained(generation_config)


def parse_generate_flags(generate_flags: list[str] | None) -> dict:
    """Parses the generate flags from the user input into a dictionary of `generate` kwargs."""
    if generate_flags is None or len(generate_flags) == 0:
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
        s = s.removeprefix("-")
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


def new_chat_history(system_prompt: str | None = None) -> list[dict]:
    """Returns a new chat conversation."""
    return [{"role": "system", "content": system_prompt}] if system_prompt else []


def save_chat(filename: str, chat: list[dict], settings: dict) -> str:
    """Saves the chat history to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump({"settings": settings, "chat_history": chat}, f, indent=4)
    return os.path.abspath(filename)


def get_username() -> str:
    """Returns the username of the current user."""
    if platform.system() == "Windows":
        return os.getlogin()
    else:
        return pwd.getpwuid(os.getuid()).pw_name


if __name__ == "__main__":
    Chat(model_id="meta-llama/Llama-3.2-3b-Instruct")
