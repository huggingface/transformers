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

import sys

import numpy

import pkg_resources
from transformers.testing_utils import TestCasePlus
from transformers.utils.versions import require_version, require_version_core, require_version_examples


numpy_ver = numpy.__version__
python_ver = ".".join([str(x) for x in sys.version_info[:3]])


class DependencyVersionCheckTest(TestCasePlus):
    def test_core(self):
        # lt + different version strings
        require_version_core("numpy<1000.4.5")
        require_version_core("numpy<1000.4")
        require_version_core("numpy<1000")

        # le
        require_version_core("numpy<=1000.4.5")
        require_version_core(f"numpy<={numpy_ver}")

        # eq
        require_version_core(f"numpy=={numpy_ver}")

        # ne
        require_version_core("numpy!=1000.4.5")

        # ge
        require_version_core("numpy>=1.0")
        require_version_core("numpy>=1.0.0")
        require_version_core(f"numpy>={numpy_ver}")

        # gt
        require_version_core("numpy>1.0.0")

        # requirement w/o version
        require_version_core("numpy")

        # unmet requirements due to version conflict
        for req in ["numpy==1.0.0", "numpy>=1000.0.0", f"numpy<{numpy_ver}"]:
            try:
                require_version_core(req)
            except pkg_resources.VersionConflict as e:
                self.assertIn(f"{req} is required", str(e))
                self.assertIn("but found", str(e))

        # unmet requirements due to missing module
        for req in ["numpipypie>1", "numpipypie2"]:
            try:
                require_version_core(req)
            except pkg_resources.DistributionNotFound as e:
                self.assertIn(f"The '{req}' distribution was not found and is required by this application", str(e))
                self.assertIn("Try: pip install transformers -U", str(e))

        # bogus requirements formats:
        # 1. whole thing
        for req in ["numpy??1.0.0", "numpy1.0.0"]:
            try:
                require_version_core(req)
            except ValueError as e:
                self.assertIn("requirement needs to be in the pip package format", str(e))
        # 2. only operators
        for req in ["numpy=1.0.0", "numpy == 1.00", "numpy<>1.0.0", "numpy><1.00", "numpy>>1.0.0"]:
            try:
                require_version_core(req)
            except ValueError as e:
                self.assertIn("need one of ", str(e))

    def test_examples(self):
        # the main functionality is tested in `test_core`, this is just the hint check
        try:
            require_version_examples("numpy>1000.4.5")
        except pkg_resources.VersionConflict as e:
            self.assertIn("is required", str(e))
            self.assertIn("pip install -r examples/requirements.txt", str(e))

    def test_python(self):

        # matching requirement
        require_version("python>=3.6.0")

        # not matching requirements
        for req in ["python>9.9.9", "python<3.0.0"]:
            try:
                require_version_core(req)
            except pkg_resources.VersionConflict as e:
                self.assertIn(f"{req} is required", str(e))
                self.assertIn(f"but found python=={python_ver}", str(e))
