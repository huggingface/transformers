"""
Implementation of a custom transfer agent for the transfer type "multipart" for git-lfs.

Inspired by: github.com/cbartz/git-lfs-swift-transfer-agent/blob/master/git_lfs_swift_transfer.py

Spec is: github.com/git-lfs/git-lfs/blob/master/docs/custom-transfers.md


To launch debugger while developing:

```
[lfs "customtransfer.multipart"]

path = /path/to/transformers/.env/bin/python

args = -m debugpy --listen 5678 --wait-for-client /path/to/transformers/src/transformers/commands/transformers_cli.py lfs-multipart-upload
```
"""

import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from contextlib import AbstractContextManager
from typing import Dict, Optional

import requests

from transformers.commands import BaseTransformersCLICommand

from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


LFS_MULTIPART_UPLOAD_COMMAND = "lfs-multipart-upload"


class LfsCommands(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        enable_parser = parser.add_parser("lfs-enable-largefiles", help="Configure your repository to enable upload of files > 5GB")
        enable_parser.add_argument("path", type=str, help="Local path to repository you want to configure.")
        enable_parser.set_defaults(func=lambda args: LfsEnableCommand(args))

        upload_parser = parser.add_parser(LFS_MULTIPART_UPLOAD_COMMAND, help="This will get called by git-lfs, do not call it directly.")
        upload_parser.set_defaults(func=lambda args: LfsUploadCommand(args))


class LfsEnableCommand:
    def __init__(self, args):
        self.args = args

    def run(self):
        local_path = os.path.abspath(self.args.path)
        if not os.path.isdir(local_path):
            print("This does not look like a valid git repo")
            exit(1)
        print(local_path)
        subprocess.run("git config lfs.customtransfer.multipart.path transformers-cli".split(), cwd=local_path)
        subprocess.run(f"git config lfs.customtransfer.multipart.args {LFS_MULTIPART_UPLOAD_COMMAND}".split(), cwd=local_path)


def write_msg(msg: Dict):
    """Write out the message in Line delimited JSON."""
    msg = json.dumps(msg) + "\n"
    sys.stdout.write(msg)
    sys.stdout.flush()


def read_msg() -> Optional[Dict]:
    """Read Line delimited JSON from stdin. """
    msg = json.loads(sys.stdin.readline().strip())

    if "terminate" in (msg.get("type"), msg.get("event")):
        # terminate message received
        return None

    if msg.get("event") not in ("download", "upload"):
        logger.critical("Received unexpected message")
        sys.exit(1)

    return msg


class FileSlice(AbstractContextManager):
    def __init__(self, filepath: str, seek_from: int, max_size: int):
        self.filepath = filepath
        self.seek_from = seek_from
        self.max_size = max_size

    def __enter__(self):
        self.f = open(self.filepath, "rb")
        self.f.seek(self.seek_from)

    def read(self, n=-1):
        return self.f.read(n)

    def __exit__(self, exc_type, exc_value, traceback):
        self.f.close()


class LfsUploadCommand:
    def __init__(self, args):
        self.args = args

    def run(self):
        # Immediately after invoking a custom transfer process, git-lfs
        # sends initiation data to the process over stdin.
        # This tells the process useful information about the configuration.
        init_msg = json.loads(sys.stdin.readline().strip())
        if not (init_msg.get("event") == "init" and init_msg.get("operation") == "upload"):
            write_msg({"error": {"code": 32, "message": "Wrong lfs init operation"}})
            sys.exit(1)

        # The transfer process should use the information it needs from the
        # initiation structure, and also perform any one-off setup tasks it
        # needs to do. It should then respond on stdout with a simple empty
        # confirmation structure, as follows:
        write_msg({})

        # After the initiation exchange, git-lfs will send any number of
        # transfer requests to the stdin of the transfer process, in a serial sequence.
        while True:
            msg = read_msg()
            if msg is None:
                # When all transfers have been processed, git-lfs will send
                # a terminate event to the stdin of the transfer process.
                # On receiving this message the transfer process should
                # clean up and terminate. No response is expected.
                sys.exit(0)

            oid = msg["oid"]
            filepath = msg["path"]
            completion_url = msg["action"]["href"]
            header = msg["action"]["header"]
            chunk_size = int(header.pop("chunk_size"))
            presigned_urls: List[str] = list(header.values())

            parts = []
            for i, presigned_url in enumerate(presigned_urls):
                with FileSlice(filepath, seek_from=i * chunk_size, max_size=chunk_size) as data:
                    r = requests.put(presigned_url, data=data)
                    r.raise_for_status()
                    parts.append(
                        {
                            "etag": r.headers.get("etag"),
                            "partNumber": i + 1,
                        }
                    )
                    # In order to support progress reporting while data is uploading / downloading,
                    # the transfer process should post messages to stdout
                    write_msg(
                        {
                            "event": "progress",
                            "oid": oid,
                            "bytesSoFar": i * chunk_size,
                            "bytesSinceLast": chunk_size,
                        }
                    )
                    # Not precise but that's ok.

            r = requests.post(
                completion_url,
                json={
                    "oid": oid,
                    "parts": parts,
                },
            )
            r.raise_for_status()

            write_msg({"event": "complete", "oid": oid})
