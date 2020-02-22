import os
import sys
from pathlib import Path
from getpass import getpass
from typing import List, Union

from requests.exceptions import HTTPError
import typer

from transformers.commands import BaseTransformersCLICommand
from transformers.hf_api import HfApi, HfFolder


UPLOAD_MAX_FILES = 15
api = HfApi()


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _bold = "\u001b[1m"
    _reset = "\u001b[0m"

    @classmethod
    def bold(cls, s):
        return "{}{}{}".format(cls._bold, s, cls._reset)


def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:
    stackoverflow.com/a/8356620/593036
    stackoverflow.com/questions/9535954/typer.echoing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return "\n".join(lines)


def walk_dir(path: Path):
    return [(f, f.name) for f in path.rglob("*") if f.is_file()]


def login():
    """Log in using the same credentials as on huggingface.co"""
    typer.echo("""
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    """)
    username = typer.prompt("Username")
    password = getpass()
    try:
        token = api.login(username, password)
    except HTTPError as e:
        # probably invalid credentials, display error message.
        typer.echo(e)
        typer.exit(1)
    HfFolder.save_token(token)
    typer.echo("Login successful")
    typer.echo(f"Your token: {token}\n")
    typer.echo(f"Your token has been saved to {HfFolder.path_token}")


def whoami():
    """Find out which huggingface.co account you are logged in as."""
    token = HfFolder.get_token()
    if token is None:
        typer.echo("Not logged in")
        typer.exit()
    try:
        user = api.whoami(token)
        typer.echo(user)
    except HTTPError as e:
        typer.echo(e)


def logout():
    """Log out"""
    token = HfFolder.get_token()
    if token is None:
        typer.echo("Not logged in")
        typer.exit()
    HfFolder.delete_token()
    api.logout(token)
    typer.echo("Successfully logged out.")


def upload(path: Path, target_filename: str):
    """Upload a model saved with `.save_pretrained` for easy public access."""
    token = HfFolder.get_token()
    if token is None:
        typer.echo("Not logged in")
        typer.exit(1)
    if path.is_dir():
        if target_filename is not None:
            raise ValueError("Cannot specify a filename override when uploading a folder.")
        files = walk_dir(path)
    elif path.is_file():
        filename = filename if filename is not None else path.name
        files = [(path, filename)]
    else:
        raise ValueError("Not a valid file or directory: {}".format(path))

    if len(files) > UPLOAD_MAX_FILES:
        typer.echo(
            "About to upload {} files to S3. This is probably wrong. Please filter files before uploading.".format(
                ANSI.bold(len(files))
            )
        )
        typer.exit(1)

    for filepath, filename in files:
        typer.echo("About to upload file {} to S3 under filename {}".format(ANSI.bold(filepath), ANSI.bold(filename)))

    choice = typer.prompt("Proceed? [Y/n]").lower()
    if not (choice == "" or choice == "y" or choice == "yes"):
        typer.echo("Abort")
        typer.exit()
    typer.echo(ANSI.bold("Uploading... This might take a while if files are large"))
    for filepath, filename in files:
        access_url = api.presign_and_upload(token=token, filename=filename, filepath=filepath)
        typer.echo("Your file now lives at:")
        typer.echo(access_url)


s3_app = typer.Typer()


@s3_app.command()
def ls():
    """List your files in s3"""
    token = HfFolder.get_token()
    if token is None:
        typer.echo("Not logged in")
        typer.exit(1)
    try:
        objs = api.list_objs(token)
    except HTTPError as e:
        typer.echo(e)
        typer.exit(1)
    if len(objs) == 0:
        typer.echo("No shared file yet")
        typer.exit()
    rows = [[obj.filename, obj.LastModified, obj.ETag, obj.Size] for obj in objs]
    typer.echo(tabulate(rows, headers=["Filename", "LastModified", "ETag", "Size"]))


@s3_app.command()
def rm(filename: str):
    """Remove FILENAME from s3"""
    token = HfFolder.get_token()
    if token is None:
        typer.echo("Not logged in")
        typer.exit(1)
    try:
        api.delete_obj(token, filename=filename)
    except HTTPError as e:
        typer.echo(e)
        typer.exit(1)
    typer.echo("Done")
