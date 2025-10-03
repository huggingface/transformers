# Copyright 2020 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
import unittest

from transformers import BertConfig, BertModel, BertTokenizer, pipeline, AutoModel
from transformers.testing_utils import TestCasePlus, require_torch

class OfflineTests(TestCasePlus):

    @require_torch
    def test_offline_mode(self):
        # Step 1: Cache Warmup - Download model online (network ON)
        mname = "hf-internal-testing/tiny-random-bert"
        BertConfig.from_pretrained(mname)
        BertModel.from_pretrained(mname)
        BertTokenizer.from_pretrained(mname)
        pipeline(task="fill-mask", model=mname)

        # Step 2: Prepare offline mode test via subprocess
        load = """from transformers import BertConfig, BertModel, BertTokenizer, pipeline"""
        run = """
mname = "hf-internal-testing/tiny-random-bert"
BertConfig.from_pretrained(mname)
BertModel.from_pretrained(mname)
BertTokenizer.from_pretrained(mname)
pipe = pipeline(task="fill-mask", model=mname)
print("success")
"""
        mock = """
import socket
def offline_socket(*args, **kwargs): raise RuntimeError("Offline mode is enabled, we shouldn't access internet")
socket.socket = offline_socket
"""
        stdout, _ = self._execute_with_env(load, run, mock, HF_HUB_OFFLINE="1")
        self.assertIn("success", stdout)

    @require_torch
    def test_offline_mode_no_internet(self):
        mname = "hf-internal-testing/tiny-random-bert"
        BertConfig.from_pretrained(mname)
        BertModel.from_pretrained(mname)
        BertTokenizer.from_pretrained(mname)
        pipeline(task="fill-mask", model=mname)

        load = """from transformers import BertConfig, BertModel, BertTokenizer, pipeline"""
        run = """
mname = "hf-internal-testing/tiny-random-bert"
BertConfig.from_pretrained(mname)
BertModel.from_pretrained(mname)
BertTokenizer.from_pretrained(mname)
pipe = pipeline(task="fill-mask", model=mname)
print("success")
"""
        mock = """
import socket
def offline_socket(*args, **kwargs): raise socket.error("Faking flaky internet")
socket.socket = offline_socket
"""
        stdout, _ = self._execute_with_env(load, run, mock)
        self.assertIn("success", stdout)

    @require_torch
    def test_offline_mode_sharded_checkpoint(self):
        # Warmup cache for sharded checkpoint
        mname = "hf-internal-testing/tiny-random-bert-sharded"
        BertConfig.from_pretrained(mname)
        BertModel.from_pretrained(mname)

        load = """from transformers import BertConfig, BertModel, BertTokenizer"""
        run = """
mname = "hf-internal-testing/tiny-random-bert-sharded"
BertConfig.from_pretrained(mname)
BertModel.from_pretrained(mname)
print("success")
"""
        mock = """
import socket
def offline_socket(*args, **kwargs): raise ValueError("Offline mode is enabled")
socket.socket = offline_socket
"""
        stdout, _ = self._execute_with_env(load, run)
        self.assertIn("success", stdout)

        # Should succeed as HF_HUB_OFFLINE=1 tells it to use local files
        stdout, _ = self._execute_with_env(load, mock, run, HF_HUB_OFFLINE="1")
        self.assertIn("success", stdout)

    @require_torch
    def test_offline_mode_pipeline_exception(self):
        load = """from transformers import pipeline"""
        run = """
mname = "hf-internal-testing/tiny-random-bert"
pipe = pipeline(model=mname)
"""
        mock = """
import socket
def offline_socket(*args, **kwargs): raise socket.error("Offline mode is enabled")
socket.socket = offline_socket
"""
        _, stderr = self._execute_with_env(load, mock, run, should_fail=True, HF_HUB_OFFLINE="1")
        self.assertIn(
            "You cannot infer task automatically within `pipeline` when using offline mode",
            stderr.replace("\n", ""),
        )

    @require_torch
    def test_offline_model_dynamic_model(self):
        mname = "hf-internal-testing/test_dynamic_model"
        from transformers import AutoModel

        # Warmup cache
        AutoModel.from_pretrained(mname, trust_remote_code=True)

        load = """from transformers import AutoModel"""
        run = """
mname = "hf-internal-testing/test_dynamic_model"
AutoModel.from_pretrained(mname, trust_remote_code=True)
print("success")
"""
        # Should succeed normally
        stdout, _ = self._execute_with_env(load, run)
        self.assertIn("success", stdout)

        # Should succeed as HF_HUB_OFFLINE=1 tells it to use local files
        stdout, _ = self._execute_with_env(load, run, HF_HUB_OFFLINE="1")
        self.assertIn("success", stdout)

    def test_is_offline_mode(self):
        load = "from transformers.utils import is_offline_mode"
        run = "print(is_offline_mode())"

        stdout, _ = self._execute_with_env(load, run)
        self.assertIn("False", stdout)

        stdout, _ = self._execute_with_env(load, run, HF_HUB_OFFLINE="1")
        self.assertIn("True", stdout)

        stdout, _ = self._execute_with_env(load, run, HF_HUB_OFFLINE="1")
        self.assertIn("True", stdout)

    def _execute_with_env(self, *commands: tuple[str, ...], should_fail: bool = False, **env) -> tuple[str, str]:
        cmd = [sys.executable, "-c", "\n".join(commands)]
        new_env = self.get_env()
        new_env.update(env)
        result = subprocess.run(cmd, env=new_env, check=False, capture_output=True)
        if should_fail:
            self.assertNotEqual(result.returncode, 0, result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.decode(), result.stderr.decode()
