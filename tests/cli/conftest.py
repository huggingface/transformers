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
import os
import sys

import pytest
import typer
import typer.main
from click.testing import CliRunner

import transformers.cli.transformers

# Set expandable CUDA memory segments before any CUDA initialization.
# This must be done via os.environ (not torch API) to take effect before the
# CUDA allocator is first used. Without this, loading multiple large models
# sequentially (e.g. gemma-4 in TestMultimodalLM followed by TestToolCallGemma)
# causes fragmentation that prevents new large allocations even when enough
# total GPU memory is free.
_existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _existing:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"{_existing},expandable_segments:True" if _existing else "expandable_segments:True"
    )


@pytest.fixture
def cli():
    app = transformers.cli.transformers.app
    if isinstance(app, typer.Typer):
        app = typer.main.get_command(app)

    def _cli_invoke(*args):
        runner = CliRunner()

        old_out_close = sys.stdout.close
        old_err_close = sys.stderr.close

        def _noop(*a, **k):
            return None

        sys.stdout.close = _noop
        sys.stderr.close = _noop
        try:
            return runner.invoke(app, list(args), catch_exceptions=False)
        finally:
            sys.stdout.close = old_out_close
            sys.stderr.close = old_err_close

    return _cli_invoke
