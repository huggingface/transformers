# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import importlib
import importlib.metadata
import importlib.util
import multiprocessing
import sys
import unittest
from unittest.mock import patch


_DIRECT_IMPORTS = (
    "transformers.distributed.pipeline_parallel",
    "transformers.modeling_utils",
    "transformers.models.vit.modeling_vit",
    "transformers.masking_utils",
    "transformers.generation.utils",
)


def _check_config():
    from transformers import BertConfig

    if BertConfig().model_type != "bert":
        raise AssertionError("BertConfig is unavailable")


def _check_import_errors(modules):
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as error:
            if "requires PyTorch >= 2.6" not in str(error):
                raise AssertionError(f"{name}: unexpected ImportError: {error}") from error
        else:
            raise AssertionError(f"{name}: missing ImportError")


def _simulated_old_torch():
    if "transformers" in sys.modules:
        raise AssertionError("transformers imported before version simulation")
    original_version = importlib.metadata.version

    def version(name):
        return "2.5.1" if name == "torch" else original_version(name)

    with patch("importlib.metadata.version", side_effect=version):
        import transformers

        if transformers.is_torch_available():
            raise AssertionError("simulated PyTorch 2.5.1 was considered available")
        _check_config()
        _check_import_errors(_DIRECT_IMPORTS)


def _real_old_torch():
    _check_config()
    _check_import_errors(_DIRECT_IMPORTS)


def _missing_torch():
    _check_config()
    _check_import_errors(("transformers.distributed.pipeline_parallel", "transformers.masking_utils"))


def _supported_torch():
    for name in _DIRECT_IMPORTS:
        importlib.import_module(name)


class UnsupportedTorchImportTest(unittest.TestCase):
    def run_in_fresh_process(self, target):
        process = multiprocessing.get_context("spawn").Process(target=target)
        process.start()
        try:
            process.join(timeout=60)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                self.fail(f"{target.__name__} timed out")
            self.assertEqual(process.exitcode, 0, f"{target.__name__} failed in the child process")
        finally:
            process.close()

    def test_unavailable_torch_direct_imports_report_version(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("version simulation requires an installed torch")
        self.run_in_fresh_process(_simulated_old_torch)

    def test_real_old_torch_direct_imports(self):
        import transformers

        if importlib.util.find_spec("torch") is None or transformers.is_torch_available():
            self.skipTest("requires installed but unsupported PyTorch")
        self.run_in_fresh_process(_real_old_torch)

    def test_real_missing_torch(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("requires PyTorch not installed")
        self.run_in_fresh_process(_missing_torch)

    def test_real_supported_torch_direct_imports(self):
        import transformers

        if not transformers.is_torch_available():
            self.skipTest("requires supported PyTorch")
        self.run_in_fresh_process(_supported_torch)
