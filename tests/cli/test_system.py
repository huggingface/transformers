# Copyright 2025-present, the HuggingFace Inc. team.
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

from transformers import __version__


def test_cli_env(cli):
    output = cli("env")
    assert output.exit_code == 0
    assert "Python version" in output.output
    assert "Platform" in output.output
    assert "Using distributed or parallel set-up in script?" in output.output


def test_cli_version(cli):
    output = cli("version")
    assert output.exit_code == 0
    assert output.output.strip() == __version__
