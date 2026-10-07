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
"""Lightweight entrypoint for the public `transformers` CLI."""

import importlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import click
import typer
from huggingface_hub import check_cli_update, typer_factory
from huggingface_hub.cli._cli_utils import HFCliTyperGroup
from typer.main import get_command as get_typer_command


_CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 120,
}

_SKIP_UPDATE_ARGUMENTS = {
    "-h",
    "--help",
    "--install-completion",
    "--show-completion",
}


@dataclass(frozen=True)
class LazyCommandSpec:
    name: str
    module_name: str
    attr_name: str
    short_help: str
    topic: str = "main"
    aliases: tuple[str, ...] = ()


_COMMAND_SPECS = (
    LazyCommandSpec(
        name="add-new-model-like",
        module_name="transformers.cli.add_new_model_like",
        attr_name="add_new_model_like",
        short_help="Add a new model to the library, based on an existing one.",
    ),
    LazyCommandSpec(
        name="chat",
        module_name="transformers.cli.chat",
        attr_name="Chat",
        short_help="Chat with a model from the command line.",
    ),
    LazyCommandSpec(
        name="download",
        module_name="transformers.cli.download",
        attr_name="download",
        short_help="Download a model and its tokenizer from the Hub.",
    ),
    LazyCommandSpec(
        name="env",
        module_name="transformers.cli.system",
        attr_name="env",
        short_help="Print information about the environment.",
    ),
    LazyCommandSpec(
        name="serve",
        module_name="transformers.cli.serve",
        attr_name="Serve",
        short_help="Run a FastAPI server to serve models on-demand with an OpenAI compatible API.",
    ),
    LazyCommandSpec(
        name="version",
        module_name="transformers.cli.system",
        attr_name="version",
        short_help="Print CLI version.",
    ),
)

_COMMANDS_BY_NAME = {spec.name: spec for spec in _COMMAND_SPECS}
_COMMANDS_BY_ALIAS = {alias: spec for spec in _COMMAND_SPECS for alias in spec.aliases}


def _build_click_command(spec: LazyCommandSpec) -> click.Command:
    command = getattr(importlib.import_module(spec.module_name), spec.attr_name)
    command_app = typer.Typer(
        add_completion=False,
        no_args_is_help=False,
        rich_markup_mode=None,
        pretty_exceptions_enable=False,
        context_settings=_CONTEXT_SETTINGS,
    )
    command_app.command(name=spec.name, short_help=spec.short_help)(command)
    return get_typer_command(command_app)


class TransformersCliGroup(HFCliTyperGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded_commands: dict[str, click.Command] = {}

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        spec = _COMMANDS_BY_NAME.get(cmd_name) or _COMMANDS_BY_ALIAS.get(cmd_name)
        if spec is None:
            return None
        if spec.name not in self._loaded_commands:
            self._loaded_commands[spec.name] = _build_click_command(spec)
        return self._loaded_commands[spec.name]

    def _alias_map(self) -> dict[str, list[str]]:
        return {spec.name: list(spec.aliases) for spec in _COMMAND_SPECS}

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        topics: dict[str, list[tuple[str, str]]] = {}
        alias_map = self._alias_map()

        for spec in _COMMAND_SPECS:
            help_text = spec.short_help
            aliases = alias_map.get(spec.name, [])
            if aliases:
                help_text = f"{help_text} [alias: {', '.join(aliases)}]"
            topics.setdefault(spec.topic, []).append((spec.name, help_text))

        with formatter.section("Main commands"):
            formatter.write_dl(topics.get("main", []))

        for topic in sorted(topics):
            if topic == "main":
                continue
            with formatter.section(f"{topic.capitalize()} commands"):
                formatter.write_dl(topics[topic])

    def list_commands(self, ctx: click.Context) -> list[str]:
        return [spec.name for spec in _COMMAND_SPECS]

    def format_epilog(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if self.epilog:
            formatter.write_paragraph()
            formatter.write_text(self.epilog)


app = typer_factory(help="Transformers CLI", cls=TransformersCliGroup)


@app.callback()
def callback() -> None:
    return None


def _should_check_for_updates(args: Sequence[str]) -> bool:
    if not args:
        return False
    if os.environ.get("_TRANSFORMERS_COMPLETE") is not None:
        return False
    return not any(arg in _SKIP_UPDATE_ARGUMENTS for arg in args)


def main(args: Sequence[str] | None = None):
    cli_args = list(sys.argv[1:] if args is None else args)
    if _should_check_for_updates(cli_args):
        check_cli_update("transformers")
    return app(args=cli_args, prog_name="transformers")


__all__ = ["app", "main"]
