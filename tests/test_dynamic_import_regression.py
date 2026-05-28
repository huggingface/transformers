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

import os
import shutil
import sys
import unittest
import importlib
from pathlib import Path

from transformers.dynamic_module_utils import get_class_in_module
from transformers.utils import HF_MODULES_CACHE
from transformers.testing_utils import TestCasePlus


class DynamicImportRegressionTest(TestCasePlus):
    def setUp(self):
        super().setUp()
        self.module_name = "modeling_regression"
        self.class_name = "RegressionConfig"
        self.san_org = "test_regression_org"
        self.repo = "test_repo"
        self.hash = "v1"
        
        # Manually populate the cache with a module that uses relative imports.
        # This mirrors the state of a 'Worker B' in a parallel pytest-xdist run.
        self.base_path = Path(HF_MODULES_CACHE) / "transformers_modules" / self.san_org / self.repo / self.hash
        if self.base_path.exists():
            shutil.rmtree(Path(HF_MODULES_CACHE) / "transformers_modules" / self.san_org)
        
        os.makedirs(self.base_path, exist_ok=True)
        
        # Create __init__.py files for the package structure
        curr = Path(HF_MODULES_CACHE) / "transformers_modules"
        for part in [self.san_org, self.repo, self.hash]:
            curr = curr / part
            os.makedirs(curr, exist_ok=True)
            (curr / "__init__.py").touch()
            
        # Create a utility file for relative import
        with open(self.base_path / "utils.py", "w", encoding="utf-8") as f:
            f.write("def get_val(): return 42\n")

        # Create the main modeling file with a relative import
        with open(self.base_path / f"{self.module_name}.py", "w", encoding="utf-8") as f:
            f.write("from transformers import PretrainedConfig\n")
            f.write("from .utils import get_val\n")
            f.write(f"class {self.class_name}(PretrainedConfig):\n")
            f.write("    model_type = 'regression'\n")

    def tearDown(self):
        super().tearDown()
        if self.base_path.exists():
            shutil.rmtree(Path(HF_MODULES_CACHE) / "transformers_modules" / self.san_org)

    def test_get_class_in_module_with_relative_import(self):
        """
        Verifies that get_class_in_module can successfully load a module with relative imports
        in a fresh process environment. This test passes if the environment is correctly 
        initialized by either the library or the test configuration (conftest.py).
        """
        # Clear existing modules to force a fresh import
        for mod in list(sys.modules.keys()):
            if mod.startswith(f"transformers_modules.{self.san_org}"):
                del sys.modules[mod]
        importlib.invalidate_caches()
        
        # The path relative to HF_MODULES_CACHE
        rel_module_path = f"transformers_modules/{self.san_org}/{self.repo}/{self.hash}/{self.module_name}.py"
        
        # This call will fail with ModuleNotFoundError if the HF_MODULES_CACHE 
        # is not present in sys.path.
        try:
            cls = get_class_in_module(self.class_name, rel_module_path)
            self.assertEqual(cls.__name__, self.class_name)
        except ModuleNotFoundError as e:
            self.fail(f"Dynamic import failed. The environment was not correctly initialized. Error: {e}")
