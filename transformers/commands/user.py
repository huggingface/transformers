from argparse import ArgumentParser
from getpass import getpass
import os

from transformers.commands import BaseTransformersCLICommand
from transformers.hf_api import HfApi, HfFolder, HTTPError


class UserCommands(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser('login')
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser('whoami')
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        logout_parser = parser.add_parser('logout')
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))
        list_parser = parser.add_parser('ls')
        list_parser.set_defaults(func=lambda args: ListObjsCommand(args))
        # upload
        upload_parser = parser.add_parser('upload')
        upload_parser.add_argument('path', type=str, help='Local path of the folder or individual file to upload.')
        upload_parser.add_argument('--filename', type=str, default=None, help='Optional: override individual object filename on S3.')
        upload_parser.set_defaults(func=lambda args: UploadCommand(args))



class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """
    _bold = u"\u001b[1m"
    _reset = u"\u001b[0m"
    @classmethod
    def bold(cls, s):
        return "{}{}{}".format(cls._bold, s, cls._reset)


class BaseUserCommand:
    def __init__(self, args):
        self.args = args
        self._api = HfApi()


class LoginCommand(BaseUserCommand):
    def run(self):
        print("""
        _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|  
        _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|        
        _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|    
        _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|        
        _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|  

        """)
        username = input("Username: ")
        password = getpass()
        try:
            token = self._api.login(username, password)
        except HTTPError as e:
            # probably invalid credentials, display error message.
            print(e)
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
            user = self._api.whoami(token)
            print(user)
        except HTTPError as e:
            print(e)


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
    def tabulate(self, rows, headers):
        # type: (List[List[Union[str, int]]], List[str]) -> str
        """
        Inspired by:
        stackoverflow.com/a/8356620/593036
        stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
        """
        col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
        row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
        lines = []
        lines.append(
            row_format.format(*headers)
        )
        lines.append(
            row_format.format(*["-" * w for w in col_widths])
        )
        for row in rows:
            lines.append(
                row_format.format(*row)
            )
        return "\n".join(lines)

    def run(self):
        token = HfFolder.get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            objs = self._api.list_objs(token)
        except HTTPError as e:
            print(e)
            exit(1)
        if len(objs) == 0:
            print("No shared file yet")
            exit()
        rows = [ [
            obj.filename,
            obj.LastModified,
            obj.ETag,
            obj.Size
        ] for obj in objs ]
        print(
            self.tabulate(rows, headers=["Filename", "LastModified", "ETag", "Size"])
        )


class UploadCommand(BaseUserCommand):
    def walk_dir(self, rel_path):
        """
        Recursively list all files in a folder.
        """
        entries: List[os.DirEntry] = list(os.scandir(rel_path))
        files = [
            (
                os.path.join(os.getcwd(), f.path),  # filepath
                f.path  # filename
            )
            for f in entries if f.is_file()
        ]
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

        for filepath, filename in files:
            print(
                "About to upload file {} to S3 under filename {}".format(
                    ANSI.bold(filepath), ANSI.bold(filename)
                )
            )

        choice = input("Proceed? [Y/n] ").lower()
        if not(choice == "" or choice == "y" or choice == "yes"):
            print("Abort")
            exit()
        print(
            ANSI.bold("Uploading... This might take a while if files are large")
        )
        for filepath, filename in files:
            access_url = self._api.presign_and_upload(
                token=token, filename=filename, filepath=filepath
            )
            print("Your file now lives at:")
            print(access_url)
