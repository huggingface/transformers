# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""
Utility functions for calling processes. Especially useful for calling processes that might fail.
"""

import argparse
import logging
import shlex
import subprocess
import time


logger = logging.getLogger(__name__)

DEFAULT_RETRIES: int = 5
DEFAULT_TIMEOUT: int = 1


def call_proc(cmd) -> tuple[bytes, bytes]:
    """Call a process and return the output and error streams."""
    try:
        p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return (out, err)
    except Exception as e:
        return (b"", str(e).encode())


def retry_call_proc(cmd, retries=DEFAULT_RETRIES, timeout=DEFAULT_TIMEOUT, debug=False) -> bytes | None:
    """Call a process and retry if it fails."""
    for i in range(retries):
        out, err = call_proc(cmd)
        if err:
            if debug:
                logger.error("Failed to run command: %s. Error: %s", cmd, err.decode())
            if timeout > 0:
                time.sleep(timeout)
        else:
            return out
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a command and retry if it fails.")
    parser.add_argument("cmd", help="Command to run.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max retries for command.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout between retries.")
    parser.add_argument("--debug", action="store_true", help="Whether or not this is a patch release.")
    args = parser.parse_args()

    if not args.cmd:
        print("Please provide a command to run.")
        exit(1)

    out = retry_call_proc(cmd=args.cmd, retries=args.retries, timeout=args.timeout, debug=args.debug)
    if out:
        print(out.decode())
    else:
        print("Failed to run the command.")
