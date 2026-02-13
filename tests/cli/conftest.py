# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import pytest
from typer.testing import CliRunner

import transformers.cli.transformers


@pytest.fixture
def cli():
    def _cli_invoke(*args):
        runner = CliRunner()

        old_out_close = sys.stdout.close
        old_err_close = sys.stderr.close

        def _noop(*a, **k):
            return None

        sys.stdout.close = _noop
        sys.stderr.close = _noop
        try:
            return runner.invoke(transformers.cli.transformers.app, list(args), catch_exceptions=False)
        finally:
            sys.stdout.close = old_out_close
            sys.stderr.close = old_err_close

    return _cli_invoke
