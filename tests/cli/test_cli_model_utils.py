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

def test_cli_env(cli):
    """
    Test the 'env' command which is vital for debugging user issues.
    Checks if it correctly identifies the system environment.
    """
    output = cli("env")
    assert output.exit_code == 0
    out = output.output.lower()
    # Flexible check for key environment indicators
    assert "transformers" in out
    assert "platform" in out
    assert "python" in out
    assert "pytorch" in out


def test_cli_add_new_model_like_help(cli):
    """Checks that the help command for model creation responds correctly."""
    output = cli("add-new-model-like", "--help")
    assert output.exit_code == 0
    assert "Add a new model to the library" in output.output


def test_cli_add_fast_image_processor_help(cli):
    """Checks that the help command for fast image processors responds correctly."""
    output = cli("add-fast-image-processor", "--help")
    assert output.exit_code == 0
    assert "Add a fast image processor" in output.output
