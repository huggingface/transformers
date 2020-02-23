try:
    import typer
except:
    raise RuntimeError(
        "Using the new CLI requires Typer to be installed"
        'Please install transformers with [new-cli]: pip install "transformers[new-cli]".'
        "Or Typer separately."
    )

from enum import Enum
from transformers.cli.convert import convert
from transformers.cli.download import download
from transformers.cli.env import env
from transformers.cli.run import run
from transformers.cli.serving import serve
from transformers.cli.train import train
from transformers.cli.user import login, logout, s3_app, upload, whoami

app = typer.Typer(no_args_is_help=True)
app.command()(convert)
app.command()(download)
app.command()(env)
app.command()(run)
app.command()(serve)
app.command()(train)

# User commands
app.command()(login)
app.command()(whoami)
app.command()(logout)
app.command()(upload)


app.add_typer(s3_app, name="s3", help="{ls, rm} Run commands directly on s3")


@app.callback()
def main(verbose: bool = False):
    """
    \b
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    """
    if verbose:
        typer.echo("Will write verbose output")
        state["verbose"] = True


if __name__ == "__main__":
    app()
