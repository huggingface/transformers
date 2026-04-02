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
from unittest.mock import patch


def test_top_level_help(cli):
    output = cli("--help")
    assert output.exit_code == 0
    assert "Transformers CLI" in output.output
    assert "Main commands" in output.output
    assert "chat" in output.output
    assert "serve" in output.output


def test_top_level_help_does_not_load_subcommands(cli):
    with patch(
        "transformers_cli.run.importlib.import_module", side_effect=AssertionError("subcommands should stay lazy")
    ):
        output = cli("--help")

    assert output.exit_code == 0
