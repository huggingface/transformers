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
import sys

from transformers.testing_utils import TestCasePlus, require_torch


class OfflineTests(TestCasePlus):
    @require_torch
    def test_offline_mode(self):

        # this test is a bit tricky since TRANSFORMERS_OFFLINE can only be changed before
        # `transformers` is loaded, and it's too late for inside pytest - so we are changing it
        # while running an external program

        # python one-liner segments

        # this must be loaded before socket.socket is monkey-patched
        load = """
from transformers import BertConfig, BertModel, BertTokenizer
        """

        run = """
mname = "lysandre/tiny-bert-random"
BertConfig.from_pretrained(mname)
BertModel.from_pretrained(mname)
BertTokenizer.from_pretrained(mname)
print("success")
        """

        mock = """
import socket
def offline_socket(*args, **kwargs): raise socket.error("Offline mode is enabled")
socket.socket = offline_socket
        """

        # baseline - just load from_pretrained with normal network
        cmd = [sys.executable, "-c", "\n".join([load, run])]

        # should succeed
        env = self.get_env()
        result = subprocess.run(cmd, env=env, check=False, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("success", result.stdout.decode())

        # next emulate no network
        cmd = [sys.executable, "-c", "\n".join([load, mock, run])]

        # should normally fail as it will fail to lookup the model files w/o the network
        env["TRANSFORMERS_OFFLINE"] = "0"
        result = subprocess.run(cmd, env=env, check=False, capture_output=True)
        self.assertEqual(result.returncode, 1, result.stderr)

        # should succeed as TRANSFORMERS_OFFLINE=1 tells it to use local files
        env["TRANSFORMERS_OFFLINE"] = "1"
        result = subprocess.run(cmd, env=env, check=False, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("success", result.stdout.decode())
