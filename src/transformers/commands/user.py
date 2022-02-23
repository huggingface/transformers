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

import subprocess
from argparse import ArgumentParser
from getpass import getpass
from typing import List, Union

from huggingface_hub.hf_api import HfFolder, create_repo, login, logout, whoami
from requests.exceptions import HTTPError

from . import BaseTransformersCLICommand


UPLOAD_MAX_FILES = 15


class UserCommands(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser("login", help="Log in using the same credentials as on huggingface.co")
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser("whoami", help="Find out which huggingface.co account you are logged in as.")
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        logout_parser = parser.add_parser("logout", help="Log out")
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))

        # new system: git-based repo system
        repo_parser = parser.add_parser(
            "repo",
            help="Deprecated: use `huggingface-cli` instead. " "Commands to interact with your huggingface.co repos.",
        )
        repo_subparsers = repo_parser.add_subparsers(
            help="Deprecated: use `huggingface-cli` instead. huggingface.co repos related commands"
        )
        repo_create_parser = repo_subparsers.add_parser(
            "create", help="Deprecated: use `huggingface-cli` instead. Create a new repo on huggingface.co"
        )
        repo_create_parser.add_argument(
            "name",
            type=str,
            help="Name for your model's repo. Will be namespaced under your username to build the model id.",
        )
        repo_create_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        repo_create_parser.add_argument("-y", "--yes", action="store_true", help="Optional: answer Yes to the prompt")
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _bold = "\u001b[1m"
    _red = "\u001b[31m"
    _gray = "\u001b[90m"
    _reset = "\u001b[0m"

    @classmethod
    def bold(cls, s):
        return f"{cls._bold}{s}{cls._reset}"

    @classmethod
    def red(cls, s):
        return f"{cls._bold}{cls._red}{s}{cls._reset}"

    @classmethod
    def gray(cls, s):
        return f"{cls._gray}{s}{cls._reset}"


def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return "\n".join(lines)


class BaseUserCommand:
    def __init__(self, args):
        self.args = args


class LoginCommand(BaseUserCommand):
    def run(self):
        print(
            ANSI.red(
                "WARNING! `transformers-cli login` is deprecated and will be removed in v5. Please use "
                "`huggingface-cli login` instead."
            )
        )
        print(  # docstyle-ignore
            """
        _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
        _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
        _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
        _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
        _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

        """
        )
        username = input("Username: ")
        password = getpass()
        try:
            token = login(username, password)
        except HTTPError as e:
            # probably invalid credentials, display error message.
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        HfFolder.save_token(token)
        print("Login successful")
        print("Your token:", token, "\n")
        print("Your token has been saved to", HfFolder.path_token)


class WhoamiCommand(BaseUserCommand):
    def run(self):
        print(
            ANSI.red(
                "WARNING! `transformers-cli whoami` is deprecated and will be removed in v5. Please use "
                "`huggingface-cli whoami` instead."
            )
        )
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit()
        try:
            user, orgs = whoami(token)
            print(user)
            if orgs:
                print(ANSI.bold("orgs: "), ",".join(orgs))
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)


class LogoutCommand(BaseUserCommand):
    def run(self):
        print(
            ANSI.red(
                "WARNING! `transformers-cli logout` is deprecated and will be removed in v5. Please use "
                "`huggingface-cli logout` instead."
            )
        )
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit()
        HfFolder.delete_token()
        logout(token)
        print("Successfully logged out.")


class RepoCreateCommand(BaseUserCommand):
    def run(self):
        print(
            ANSI.red(
                "WARNING! Managing repositories through transformers-cli is deprecated. "
                "Please use `huggingface-cli` instead."
            )
        )
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print("Looks like you do not have git installed, please install.")

        try:
            stdout = subprocess.check_output(["git-lfs", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print(
                ANSI.red(
                    "Looks like you do not have git-lfs installed, please install."
                    " You can install from https://git-lfs.github.com/."
                    " Then run `git lfs install` (you only have to do this once)."
                )
            )
        print("")

        user, _ = whoami(token)
        namespace = self.args.organization if self.args.organization is not None else user
        full_name = f"{namespace}/{self.args.name}"
        print(f"You are about to create {ANSI.bold(full_name)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            url = create_repo(token, name=self.args.name, organization=self.args.organization)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour repo now lives at:")
        print(f"  {ANSI.bold(url)}")
        print("\nYou can clone it locally with the command below," " and commit/push as usual.")
        print(f"\n  git clone {url}")
        print("")
