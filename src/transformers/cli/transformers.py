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
"""Transformers CLI."""

from huggingface_hub import check_cli_update, typer_factory

from transformers.agent.output import OutputFormatWithAuto
from transformers.cli.add_new_model_like import add_new_model_like
from transformers.cli.agentic.app import register_agentic_commands
from transformers.cli.chat import Chat
from transformers.cli.download import download
from transformers.cli.serve import Serve
from transformers.cli.system import env, version


try:  # huggingface_hub <= 1.16 ships a ready-made root --format option
    from huggingface_hub.cli._cli_utils import FormatWithAutoOpt
except ImportError:  # huggingface_hub >= 1.17 removed it; rebuild the equivalent.
    # 1.17's typer group consumes --format only *after* the subcommand
    # (`transformers version --format json`); the root option keeps the
    # documented `transformers --format json <command>` form working too.
    from typing import Annotated

    import typer

    from transformers.agent.output import out

    def _apply_format(value: OutputFormatWithAuto) -> OutputFormatWithAuto:
        out.set_mode(value)
        return value

    FormatWithAutoOpt = Annotated[
        OutputFormatWithAuto,
        typer.Option("--format", help="Output format.", callback=_apply_format),
    ]


app = typer_factory(help="Transformers CLI")


@app.callback()
def _root(format: FormatWithAutoOpt = OutputFormatWithAuto.auto):
    """Transformers CLI."""
    # FormatWithAutoOpt's callback already called out.set_mode(format); this
    # callback exists only to expose --format at the top level.


app.command()(add_new_model_like)
app.command(name="chat")(Chat)
app.command()(download)
app.command()(env)
app.command(name="serve")(Serve)
app.command()(version)

register_agentic_commands(app)


def main():
    check_cli_update("transformers")
    app()


if __name__ == "__main__":
    main()
