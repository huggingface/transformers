# Copyright 2024 The HuggingFace Team. All rights reserved.
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
Tests for offline mode functionality.
Regression tests for issue #41311: https://github.com/huggingface/transformers/issues/41311
"""

import os
import subprocess
import sys
import tempfile
import unittest


class TestOfflineMode(unittest.TestCase):
    """
    Test that models can be loaded offline after cache is warmed in a subprocess.
    These are regression tests for issue #41311.
    """

    def test_subprocess_warm_cache_then_offline_load(self):
        """
        Test that warming cache in subprocess allows offline loading in parent process.
        Regression test for: https://github.com/huggingface/transformers/issues/41311
        """
        model_name = "hf-internal-testing/tiny-random-bert"

        with tempfile.TemporaryDirectory() as cache_dir:
            env = os.environ.copy()
            env["HF_HOME"] = cache_dir

            # Step 1: Download model in subprocess
            warm_script = f"""
import os
os.environ["HF_HOME"] = "{cache_dir}"

from transformers import AutoConfig, AutoModel, AutoTokenizer

config = AutoConfig.from_pretrained("{model_name}")
model = AutoModel.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
print("CACHE_WARMED")
"""

            result = subprocess.run(
                [sys.executable, "-c", warm_script],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )

            self.assertEqual(result.returncode, 0, f"Cache warming failed: {result.stderr}")
            self.assertIn("CACHE_WARMED", result.stdout)

            # Step 2: Load offline with socket blocking (after imports)
            offline_script = f"""
import os
os.environ["HF_HOME"] = "{cache_dir}"
os.environ["HF_HUB_OFFLINE"] = "1"

# Import transformers first
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Then block sockets to ensure no network access
import socket
original_socket = socket.socket
def guarded_socket(*args, **kwargs):
    raise RuntimeError("Network access attempted in offline mode!")
socket.socket = guarded_socket

try:
    config = AutoConfig.from_pretrained("{model_name}")
    model = AutoModel.from_pretrained("{model_name}")
    tokenizer = AutoTokenizer.from_pretrained("{model_name}")
    print("OFFLINE_SUCCESS")
except RuntimeError as e:
    if "Network access" in str(e):
        print(f"NETWORK_ATTEMPTED: {{e}}")
        exit(1)
    raise
except Exception as e:
    print(f"FAILED: {{e}}")
    import traceback
    traceback.print_exc()
    exit(1)
"""

            result = subprocess.run(
                [sys.executable, "-c", offline_script],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )

            if "NETWORK_ATTEMPTED" in result.stdout:
                self.fail(f"Network access attempted despite warm cache: {result.stdout}")

            self.assertIn(
                "OFFLINE_SUCCESS",
                result.stdout,
                f"Failed to load offline:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}",
            )
            self.assertEqual(result.returncode, 0)

    def test_pipeline_offline_after_subprocess_warm(self):
        """
        Test pipeline API works offline after subprocess cache warming.
        """
        model_name = "hf-internal-testing/tiny-random-bert"

        with tempfile.TemporaryDirectory() as cache_dir:
            env = os.environ.copy()
            env["HF_HOME"] = cache_dir

            # Warm cache
            warm_script = f"""
import os
os.environ["HF_HOME"] = "{cache_dir}"

from transformers import pipeline

pipe = pipeline("text-classification", model="{model_name}")
print("WARMED")
"""

            result = subprocess.run(
                [sys.executable, "-c", warm_script], capture_output=True, text=True, env=env, timeout=120
            )
            self.assertEqual(result.returncode, 0)

            # Load offline
            offline_script = f"""
import os
os.environ["HF_HOME"] = "{cache_dir}"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import pipeline
import socket

# Block sockets after imports
def no_socket(*args, **kwargs):
    raise RuntimeError("Network blocked!")
socket.socket = no_socket

try:
    pipe = pipeline("text-classification", model="{model_name}")
    print("SUCCESS")
except RuntimeError as e:
    if "Network blocked" in str(e):
        print(f"BLOCKED: {{e}}")
        exit(1)
    raise
except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""

            result = subprocess.run(
                [sys.executable, "-c", offline_script], capture_output=True, text=True, env=env, timeout=120
            )

            self.assertNotIn("BLOCKED", result.stdout, "Network access attempted")
            self.assertIn("SUCCESS", result.stdout)
            self.assertEqual(result.returncode, 0)
