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

import os
import subprocess
import sys
import textwrap
import unittest


class UnsupportedTorchImportTest(unittest.TestCase):
    def run_case(self, program):
        env = dict(os.environ, PYTHONNOUSERSITE="1")
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(program)], capture_output=True, text=True, env=env
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unavailable_torch_direct_imports_report_version(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.skipTest("version simulation requires an installed torch")
        self.run_case(
            """
            import importlib
            import importlib.util
            import importlib.metadata
            original_find_spec = importlib.util.find_spec
            original_version = importlib.metadata.version
            # Simulate a discoverable but unsupported torch before transformers is imported.
            def find_spec(name, *args, **kwargs):
                if name == "torch":
                    return original_find_spec("json")
                return original_find_spec(name, *args, **kwargs)
            def version(name):
                if name == "torch":
                    return "2.5.1"
                return original_version(name)
            importlib.util.find_spec = find_spec
            importlib.metadata.version = version
            import transformers
            assert not transformers.is_torch_available()
            from transformers import BertConfig
            assert BertConfig().model_type == "bert"
            for name in ("transformers.distributed.pipeline_parallel", "transformers.modeling_utils",
                         "transformers.models.vit.modeling_vit", "transformers.masking_utils",
                         "transformers.generation.utils"):
                try:
                    importlib.import_module(name)
                except ImportError as error:
                    assert "requires PyTorch >= 2.6" in str(error), (name, repr(error))
                else:
                    raise AssertionError(f"{name}: missing ImportError")
            """
        )

    def test_real_old_torch_direct_imports(self):
        import importlib.util

        import transformers

        if importlib.util.find_spec("torch") is None or transformers.is_torch_available():
            self.skipTest("requires installed but unsupported PyTorch")
        self.run_case(
            """
            import importlib
            import transformers
            from transformers import BertConfig
            assert BertConfig().model_type == "bert"
            for name in ("transformers.distributed.pipeline_parallel", "transformers.modeling_utils",
                         "transformers.models.vit.modeling_vit", "transformers.masking_utils",
                         "transformers.generation.utils"):
                try:
                    importlib.import_module(name)
                except ImportError as error:
                    assert "requires PyTorch >= 2.6" in str(error), (name, repr(error))
                else:
                    raise AssertionError(f"{name}: missing ImportError")
            """
        )

    def test_real_missing_torch(self):
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            self.skipTest("requires PyTorch not installed")
        self.run_case(
            """
            import importlib
            from transformers import BertConfig
            assert BertConfig().model_type == "bert"
            for name in ("transformers.distributed.pipeline_parallel", "transformers.masking_utils"):
                try:
                    importlib.import_module(name)
                except ImportError as error:
                    assert "requires PyTorch >= 2.6" in str(error), (name, repr(error))
                else:
                    raise AssertionError(f"{name}: missing ImportError")
            """
        )

    def test_real_supported_torch_direct_imports(self):
        import transformers

        if not transformers.is_torch_available():
            self.skipTest("requires supported PyTorch")
        self.run_case(
            """
            import importlib
            for name in ("transformers.distributed.pipeline_parallel", "transformers.modeling_utils",
                         "transformers.models.vit.modeling_vit", "transformers.masking_utils",
                         "transformers.generation.utils"):
                importlib.import_module(name)
            """
        )
