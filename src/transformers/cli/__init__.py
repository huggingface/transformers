try:
    import typer
except:
    raise RuntimeError(
        "Using the new CLI requires Typer to be installed"
        'Please install transformers with [new-cli]: pip install "transformers[new-cli]".'
        "Or Typer separately."
    )

from transformers.cli.convert import convert
from transformers.cli.download import download
from transformers.cli.env import env
from transformers.cli.run import run
from transformers.cli.serving import serve
from transformers.cli.train import train
from transformers.cli.user import login, logout, s3_app, upload, whoami


app = typer.Typer(no_args_is_help=True)

commands = [convert, download, env, run, serve, train, login, logout, upload, whoami]
for command in commands:
    app.command()(command)

app.add_typer(s3_app, name="s3", help="{ls, rm} Run commands directly on s3")


@app.callback()
def main():
    """
    \b
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
    """
    pass


if __name__ == "__main__":
    app()
