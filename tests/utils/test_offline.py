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
import tempfile
import unittest

from transformers import BertConfig, BertModel, BertTokenizer, pipeline
from transformers.testing_utils import TestCasePlus, require_torch, slow


class OfflineTests(TestCasePlus):
    @require_torch
    @unittest.skip("This test is failing on main")  # TODO matt/ydshieh, this test needs to be fixed
    def test_offline_mode(self):
        # this test is a bit tricky since TRANSFORMERS_OFFLINE can only be changed before
        # `transformers` is loaded, and it's too late for inside pytest - so we are changing it
        # while running an external program

        # python one-liner segments

        # this must be loaded before socket.socket is monkey-patched
        load = """
from transformers import BertConfig, BertModel, BertTokenizer, pipeline
        """

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

        # Force fetching the files so that we can use the cache
        mname = "hf-internal-testing/tiny-random-bert"
        BertConfig.from_pretrained(mname)
        BertModel.from_pretrained(mname)
        BertTokenizer.from_pretrained(mname)
        pipeline(task="fill-mask", model=mname)

        # baseline - just load from_pretrained with normal network
        # should succeed as TRANSFORMERS_OFFLINE=1 tells it to use local files
        stdout, _ = self._execute_with_env(load, run, mock, TRANSFORMERS_OFFLINE="1")
        self.assertIn("success", stdout)

    @require_torch
    def test_offline_mode_no_internet(self):
        # python one-liner segments
        # this must be loaded before socket.socket is monkey-patched
        load = """
from transformers import BertConfig, BertModel, BertTokenizer, pipeline
        """

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

        # Force fetching the files so that we can use the cache
        mname = "hf-internal-testing/tiny-random-bert"
        BertConfig.from_pretrained(mname)
        BertModel.from_pretrained(mname)
        BertTokenizer.from_pretrained(mname)
        pipeline(task="fill-mask", model=mname)

        # baseline - just load from_pretrained with normal network
        # should succeed
        stdout, _ = self._execute_with_env(load, run, mock)
        self.assertIn("success", stdout)

    @require_torch
    def test_offline_mode_sharded_checkpoint(self):
        # this test is a bit tricky since TRANSFORMERS_OFFLINE can only be changed before
        # `transformers` is loaded, and it's too late for inside pytest - so we are changing it
        # while running an external program

        # python one-liner segments

        # this must be loaded before socket.socket is monkey-patched
        load = """
from transformers import BertConfig, BertModel, BertTokenizer
        """

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

        # baseline - just load from_pretrained with normal network
        # should succeed
        stdout, _ = self._execute_with_env(load, run)
        self.assertIn("success", stdout)

        # next emulate no network
        # Doesn't fail anymore since the model is in the cache due to other tests, so commenting this.
        # self._execute_with_env(load, mock, run, should_fail=True, TRANSFORMERS_OFFLINE="0")

        # should succeed as TRANSFORMERS_OFFLINE=1 tells it to use local files
        stdout, _ = self._execute_with_env(load, mock, run, TRANSFORMERS_OFFLINE="1")
        self.assertIn("success", stdout)

    @require_torch
    def test_offline_mode_pipeline_exception(self):
        load = """
from transformers import pipeline
        """
        run = """
mname = "hf-internal-testing/tiny-random-bert"
pipe = pipeline(model=mname)
        """

        mock = """
import socket
def offline_socket(*args, **kwargs): raise socket.error("Offline mode is enabled")
socket.socket = offline_socket
        """

        _, stderr = self._execute_with_env(load, mock, run, should_fail=True, TRANSFORMERS_OFFLINE="1")
        self.assertIn(
            "You cannot infer task automatically within `pipeline` when using offline mode",
            stderr.replace("\n", ""),
        )

    @require_torch
    def test_offline_model_dynamic_model(self):
        load = """
from transformers import AutoModel
        """
        run = """
mname = "hf-internal-testing/test_dynamic_model"
AutoModel.from_pretrained(mname, trust_remote_code=True)
print("success")
        """

        # baseline - just load from_pretrained with normal network
        # should succeed
        stdout, _ = self._execute_with_env(load, run)
        self.assertIn("success", stdout)

        # should succeed as TRANSFORMERS_OFFLINE=1 tells it to use local files
        stdout, _ = self._execute_with_env(load, run, TRANSFORMERS_OFFLINE="1")
        self.assertIn("success", stdout)

    @require_torch
    def test_offline_model_dynamic_model_save_and_load_basic(self):
        run_save = """
import os
from transformers import AutoModel

mname = "hf-internal-testing/test_dynamic_model"
model = AutoModel.from_pretrained(mname, trust_remote_code=True)
model.save_pretrained('{tmp_dir}')
saved_files = os.listdir('{tmp_dir}')
if ('configuration.py' not in saved_files) or ('modeling.py' not in saved_files):
    raise ValueError(
        f"Expected to find configuration.py and modeling.py in {tmp_dir} but found only {{saved_files}}"
    )
print("success")
        """

        run_load_offline = """
from transformers import AutoModel
AutoModel.from_pretrained('{tmp_dir}', trust_remote_code=True, local_files_only=True)
print("success")
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            # load from_pretrained from remote with normal network - should succeed
            stdout, _ = self._execute_with_env(run_save.format(tmp_dir=tmp_dir))
            self.assertIn("success", stdout)

            # should succeed as TRANSFORMERS_OFFLINE=1 tells it to use local files
            stdout, _ = self._execute_with_env(run_load_offline.format(tmp_dir=tmp_dir), TRANSFORMERS_OFFLINE="1")
            self.assertIn("success", stdout)

    @require_torch
    @slow
    def test_offline_model_dynamic_model_save_and_load_advanced(self):
        run_save = """
import subprocess
subprocess.run(["pip", "install", "einops"]) # dependency of jinaai/jina-embeddings-v3

import os
from transformers import AutoModel, AutoTokenizer

model_name = "jinaai/jina-embeddings-v3"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.save_pretrained('{tmp_dir}')
tokenizer.save_pretrained('{tmp_dir}')

expected_files = [
    'block.py', 'configuration_xlm_roberta.py', 'embedding.py', 'mha.py', 'mlp.py',
    'modeling_lora.py', 'modeling_xlm_roberta.py', 'rotary.py', 'stochastic_depth.py', 'xlm_padding.py'
]
saved_files = os.listdir('{tmp_dir}')
if not all(f in saved_files for f in expected_files):
    raise ValueError(
        f"Expected to find {{expected_files}} in {tmp_dir} but found only {{saved_files}}"
    )
print("success")
        """

        run_load_offline = """
import shutil
import numpy as np
from pathlib import Path
from transformers import AutoModel
from transformers.utils import TRANSFORMERS_CACHE, HF_MODULES_CACHE

# remove cache for jinaai to reproduce the behaviour of a fresh machine without cache
shutil.rmtree(Path(TRANSFORMERS_CACHE) / 'models--jinaai--jina-embeddings-v3', ignore_errors=True)
shutil.rmtree(Path(TRANSFORMERS_CACHE) / 'models--jinaai--xlm-roberta-flash-implementation', ignore_errors=True)
shutil.rmtree(Path(HF_MODULES_CACHE) / 'transformers_modules' / 'jinaai', ignore_errors=True)

model = AutoModel.from_pretrained('{tmp_dir}', trust_remote_code=True, local_files_only=True)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
]
embeddings = model.encode(sentences, task="text-matching")
r = type(embeddings[0] @ embeddings[1])
if r == np.float32:
    print("success")
else:
    raise ValueError(f"Expected embeddings to be of type float, but got {{r}}")
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            # load from_pretrained from remote with normal network - should succeed
            stdout, _ = self._execute_with_env(run_save.format(tmp_dir=tmp_dir))
            self.assertIn("success", stdout)

            # should succeed as TRANSFORMERS_OFFLINE=1 tells it to use local files
            stdout, _ = self._execute_with_env(run_load_offline.format(tmp_dir=tmp_dir), TRANSFORMERS_OFFLINE="1")
            self.assertIn("success", stdout)

    @require_torch
    def test_offline_model_dynamic_model_load_from_cache(self):
        run_online_and_cache = """
from transformers import AutoModel
mname = "hf-internal-testing/test_dynamic_model"
model = AutoModel.from_pretrained(mname, trust_remote_code=True)
        """

        run_offline_and_load_cached = """
from transformers import AutoModel
mname = "hf-internal-testing/test_dynamic_model"
model = AutoModel.from_pretrained(mname, trust_remote_code=True, local_files_only=True)
print("success")
        """

        # load from_pretrained from remote with normal network - should succeed
        self._execute_with_env(run_online_and_cache)

        # should succeed as TRANSFORMERS_OFFLINE=1 tells it to use local files
        stdout, _ = self._execute_with_env(run_offline_and_load_cached, TRANSFORMERS_OFFLINE="1")
        self.assertIn("success", stdout)

    def test_is_offline_mode(self):
        """
        Test `_is_offline_mode` helper (should respect both HF_HUB_OFFLINE and legacy TRANSFORMERS_OFFLINE env vars)
        """
        load = "from transformers.utils import is_offline_mode"
        run = "print(is_offline_mode())"

        stdout, _ = self._execute_with_env(load, run)
        self.assertIn("False", stdout)

        stdout, _ = self._execute_with_env(load, run, TRANSFORMERS_OFFLINE="1")
        self.assertIn("True", stdout)

        stdout, _ = self._execute_with_env(load, run, HF_HUB_OFFLINE="1")
        self.assertIn("True", stdout)

    def _execute_with_env(self, *commands: tuple[str, ...], should_fail: bool = False, **env) -> tuple[str, str]:
        """Execute Python code with a given environment and return the stdout/stderr as strings.

        If `should_fail=True`, the command is expected to fail. Otherwise, it should succeed.
        Environment variables can be passed as keyword arguments.
        """
        # Build command
        cmd = [sys.executable, "-c", "\n".join(commands)]

        # Configure env
        new_env = self.get_env()
        new_env.update(env)

        # Run command
        result = subprocess.run(cmd, env=new_env, check=False, capture_output=True)

        # Check execution
        if should_fail:
            self.assertNotEqual(result.returncode, 0, result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)

        # Return output
        return result.stdout.decode(), result.stderr.decode()
