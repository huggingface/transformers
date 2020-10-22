import os
import sys
from argparse import ArgumentParser
from getpass import getpass
from typing import List, Union

from requests.exceptions import HTTPError
from transformers.commands import BaseTransformersCLICommand
from transformers.hf_api import HfApi, HfFolder


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
        # s3
        s3_parser = parser.add_parser("s3", help="{ls, rm} Commands to interact with the files you upload on S3.")
        s3_subparsers = s3_parser.add_subparsers(help="s3 related commands")
        ls_parser = s3_subparsers.add_parser("ls")
        ls_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        ls_parser.set_defaults(func=lambda args: ListObjsCommand(args))
        rm_parser = s3_subparsers.add_parser("rm")
        rm_parser.add_argument("filename", type=str, help="individual object filename to delete from S3.")
        rm_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        rm_parser.set_defaults(func=lambda args: DeleteObjCommand(args))
        # upload
        upload_parser = parser.add_parser("upload", help="Upload a model to S3.")
        upload_parser.add_argument(
            "path", type=str, help="Local path of the model folder or individual file to upload."
        )
        upload_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        upload_parser.add_argument(
            "--filename", type=str, default=None, help="Optional: override individual object filename on S3."
        )
        upload_parser.add_argument("-y", "--yes", action="store_true", help="Optional: answer Yes to the prompt")
        upload_parser.set_defaults(func=lambda args: UploadCommand(args))


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _bold = "\u001b[1m"
    _red = "\u001b[31m"
    _reset = "\u001b[0m"

    @classmethod
    def bold(cls, s):
        return "{}{}{}".format(cls._bold, s, cls._reset)

    @classmethod
    def red(cls, s):
        return "{}{}{}".format(cls._bold + cls._red, s, cls._reset)


class BaseUserCommand:
    def __init__(self, args):
        self.args = args
        self._api = HfApi()


class LoginCommand(BaseUserCommand):
    def run(self):
        print(
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
            token = self._api.login(username, password)
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
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit()
        try:
            user, orgs = self._api.whoami(token)
            print(user)
            if orgs:
                print(ANSI.bold("orgs: "), ",".join(orgs))
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)


class LogoutCommand(BaseUserCommand):
    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit()
        HfFolder.delete_token()
        self._api.logout(token)
        print("Successfully logged out.")


class ListObjsCommand(BaseUserCommand):
    def tabulate(self, rows: List[List[Union[str, int]]], headers: List[str]) -> str:
        """
        Inspired by:
        stackoverflow.com/a/8356620/593036
        stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
        """
        col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
        row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
        lines = []
        lines.append(row_format.format(*headers))
        lines.append(row_format.format(*["-" * w for w in col_widths]))
        for row in rows:
            lines.append(row_format.format(*row))
        return "\n".join(lines)

    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            objs = self._api.list_objs(token, organization=self.args.organization)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        if len(objs) == 0:
            print("No shared file yet")
            exit()
        rows = [[obj.filename, obj.LastModified, obj.ETag, obj.Size] for obj in objs]
        print(self.tabulate(rows, headers=["Filename", "LastModified", "ETag", "Size"]))


class DeleteObjCommand(BaseUserCommand):
    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            self._api.delete_obj(token, filename=self.args.filename, organization=self.args.organization)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("Done")


class UploadCommand(BaseUserCommand):
    def walk_dir(self, rel_path):
        """
        Recursively list all files in a folder.
        """
        entries: List[os.DirEntry] = list(os.scandir(rel_path))
        files = [(os.path.join(os.getcwd(), f.path), f.path) for f in entries if f.is_file()]  # (filepath, filename)
        for f in entries:
            if f.is_dir():
                files += self.walk_dir(f.path)
        return files

    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        local_path = os.path.abspath(self.args.path)
        if os.path.isdir(local_path):
            if self.args.filename is not None:
                raise ValueError("Cannot specify a filename override when uploading a folder.")
            rel_path = os.path.basename(local_path)
            files = self.walk_dir(rel_path)
        elif os.path.isfile(local_path):
            filename = self.args.filename if self.args.filename is not None else os.path.basename(local_path)
            files = [(local_path, filename)]
        else:
            raise ValueError("Not a valid file or directory: {}".format(local_path))

        if sys.platform == "win32":
            files = [(filepath, filename.replace(os.sep, "/")) for filepath, filename in files]

        if len(files) > UPLOAD_MAX_FILES:
            print(
                "About to upload {} files to S3. This is probably wrong. Please filter files before uploading.".format(
                    ANSI.bold(len(files))
                )
            )
            exit(1)

        user, _ = self._api.whoami(token)
        namespace = self.args.organization if self.args.organization is not None else user

        for filepath, filename in files:
            print(
                "About to upload file {} to S3 under filename {} and namespace {}".format(
                    ANSI.bold(filepath), ANSI.bold(filename), ANSI.bold(namespace)
                )
            )

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        print(ANSI.bold("Uploading... This might take a while if files are large"))
        for filepath, filename in files:
            try:
                access_url = self._api.presign_and_upload(
                    token=token, filename=filename, filepath=filepath, organization=self.args.organization
                )
            except HTTPError as e:
                print(e)
                print(ANSI.red(e.response.text))
                exit(1)
            print("Your file now lives at:")
            print(access_url)
